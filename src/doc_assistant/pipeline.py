"""RAG pipeline: retrieval, reranking, and answer generation."""

import hashlib
import sqlite3
from collections import OrderedDict
from collections.abc import Generator, Iterator
from typing import Any, ClassVar

import structlog
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

from doc_assistant import credentials, sparse_index
from doc_assistant.chroma_read import get_all, iter_pages
from doc_assistant.config import (
    BM25_WEIGHT,
    CANDIDATE_K,
    CHROMA_PATH,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_HOST,
    PC_CHROMA_PATH,
    RERANK_CANDIDATE_CAP,
    TOP_K,
    USE_MULTI_QUERY,
    USE_PARENT_CHILD,
)
from doc_assistant.embeddings import (
    get_active_model_name,
    get_collection_name,
    get_embeddings,
)
from doc_assistant.prompts import ANSWER_PROMPT, MULTI_QUERY_PROMPT, REWRITE_PROMPT
from doc_assistant.sparse_index import SparseIndex

log = structlog.get_logger(__name__)

# How many distinct folder scopes keep a prebuilt ensemble warm (ADR-025 F2 / RG-020). The
# scoped BM25 arm is rebuilt over the folder subset (~20 µs/chunk measured), so a single slot
# forced a full rebuild whenever the user alternated between two folders. A small LRU keeps the
# recently-used scopes warm instead. Bounded (not unbounded) because each entry holds a BM25
# index over its subset — memory, not correctness. A named structural constant, not a tunable:
# it trades a little memory for latency and never affects retrieval output (the cached ensemble
# is byte-for-byte the one a rebuild would produce). scope=None (the whole-corpus default path)
# never enters this cache — it uses the prebuilt self.ensemble.
_SCOPED_ENSEMBLE_CACHE_SIZE = 4


def resolve_ensemble_weights(bm25_weight: float | None) -> list[float]:
    """Resolve a BM25-arm weight into ``[bm25, vector]`` ensemble weights.

    ``None`` uses the locked config default (``BM25_WEIGHT``); the vector arm
    always takes the complement so the pair sums to 1.0. Kept as a pure,
    validated function (no I/O, no models) so the sweep driver and the eval CLI
    can probe candidate weights without constructing a ``RAGPipeline`` — building
    one loads the embedder, vector store, and cross-encoder. An out-of-range
    weight raises rather than clamping: a bad ``--bm25-weight`` is a caller error,
    not a value to silently correct.
    """
    weight = BM25_WEIGHT if bm25_weight is None else bm25_weight
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"bm25_weight ({weight}) must be in [0.0, 1.0]")
    return [weight, 1.0 - weight]


def _sigmoid_activation_kwarg() -> dict[str, Any]:
    """Pin the cross-encoder to sigmoid output ([0, 1]) across sentence-transformers
    versions.

    The integrity layer (provenance thresholds, Chunk 2a markers) assumes reranker
    scores are sigmoid-bounded. bge-reranker-base happens to default to sigmoid, but
    we set it explicitly so a library upgrade can't silently switch it to raw logits
    and miscalibrate every confidence marker. The constructor kwarg was renamed
    (``activation_fn`` in ST v4/v5, ``default_activation_function`` in v3), so resolve
    it from the signature instead of hardcoding a name that could raise ``TypeError``.
    """
    import inspect

    from torch import nn

    params = inspect.signature(CrossEncoder.__init__).parameters
    if "activation_fn" in params:
        return {"activation_fn": nn.Sigmoid()}
    if "default_activation_function" in params:
        return {"default_activation_function": nn.Sigmoid()}
    return {}


def build_chat_model(provider: str, model: str) -> Any:
    """Build a streaming LangChain chat model for ``provider``/``model``.

    Parameterized (not config-bound) so a caller can force a specific backend — e.g.
    local Ollama for a free self-eval run — without editing ``.env``. Constructing
    the object makes **no** API call, so this is safe to build off the hot path.

    Intentionally separate from ``llm.LLMClient``: the chat path streams tokens
    through a LangChain model, a different contract from the one-shot ``complete()``
    used by the reviewer and eval judge.

    The API key is resolved **here, per build** (``credentials.resolve_key``) rather than read
    from an import-time constant: ADR-034 lets a user save a key while the app is running, and a
    module-level ``from config import ANTHROPIC_API_KEY`` binding could never see it — the same
    separate-binding trap the module notes in ``src/doc_assistant/CLAUDE.md``. The controller
    rebuilds its chat model after a key change, so the next turn picks the new key up."""
    if provider.lower() == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from pydantic import SecretStr

        return ChatAnthropic(  # type: ignore[call-arg]
            model=model,
            api_key=SecretStr(credentials.resolve_key("anthropic") or ""),
            max_tokens=1024,
            streaming=True,
        )
    from langchain_ollama import OllamaLLM

    return OllamaLLM(model=model, base_url=OLLAMA_HOST)


class SparseRetriever(BaseRetriever):
    """LangChain retriever over the on-disk sparse index (ADR-036).

    A thin adapter, deliberately: `EnsembleRetriever` only needs
    ``invoke(query) -> list[Document]`` and fuses arms by **reciprocal rank**, so this returns its
    ``k`` best in order and never has to reconcile score scales with the vector arm. ``scope`` is
    bound per instance rather than passed per call because the ensemble's interface takes a query
    and nothing else; a scoped turn constructs its own (cheap — no index is rebuilt, unlike the
    in-RAM arm this replaces).
    """

    index: SparseIndex
    k: int
    scope: frozenset[str] | None = None

    # `SparseIndex` is a plain class, not a pydantic model, and BaseRetriever is one.
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return self.index.search(query, self.k, scope=self.scope)


class RAGPipeline:
    def __init__(self, *, bm25_weight: float | None = None) -> None:
        weights = resolve_ensemble_weights(bm25_weight)
        # The effective BM25-arm weight actually wired into the ensemble — recorded
        # so an eval run / sweep can report the arm it measured, not just the flag.
        self.bm25_weight = weights[0]
        active_model = get_active_model_name()
        collection = get_collection_name(active_model)
        log.info("loading_embeddings", model=active_model)
        self.embeddings = get_embeddings(active_model)

        log.info("loading_vector_store")
        chroma_path = PC_CHROMA_PATH if USE_PARENT_CHILD else CHROMA_PATH
        log.info("vector_store", path=chroma_path, collection=collection)
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings,
            collection_name=collection,
        )

        log.info("building_keyword_index")
        # ADR-036 / ADR-038: the sparse arm lives **on disk**, and it is now the only one. Nothing
        # here is corpus-sized in memory. `self._sparse` is None for exactly two reasons — an empty
        # corpus, or a build that failed — and both degrade to vector-only retrieval (see below).
        self._sparse = self._open_sparse_index(chroma_path, collection)
        # LRU memo for scoped ensembles, keyed on the exact hash set. The UI scope is sticky, so
        # consecutive turns share a key; a membership edit changes the key and that entry is never
        # reused (no TTL, no staleness window — the key IS the identity). Since ADR-036 scoping
        # is a WHERE clause rather than a subset index rebuild, this now saves only retriever
        # construction — kept because it is cheap and the key semantics are the useful part.
        # See the F2 spec S5; bounded by _SCOPED_ENSEMBLE_CACHE_SIZE.
        self._scoped: OrderedDict[frozenset[str], EnsembleRetriever] = OrderedDict()
        self._weights = weights
        vector = self.db.as_retriever(
            search_kwargs={
                "k": CANDIDATE_K,
                "filter": {"keep_for_retrieval": {"$ne": False}},
            }
        )
        if self._sparse is not None and self._sparse.chunks:
            log.info("ensemble_weights", bm25=weights[0], vector=weights[1], arm="sparse_index")
            self.ensemble = EnsembleRetriever(
                retrievers=[SparseRetriever(index=self._sparse, k=CANDIDATE_K), vector],
                weights=weights,
            )
        else:
            # Vector-only, and the two ways of getting here mean different things to the user
            # (ADR-038). An empty library is a *supported state*: nothing to index yet, retrieval
            # returns nothing until documents arrive. A failed build over a non-empty corpus is a
            # *degradation*: answers still come back, but keyword matching is gone, so an exact
            # term the vector arm does not embed closely will be missed.
            #
            # Before ADR-038 the second case fell back to an in-RAM BM25 arm and the user was never
            # told. Retiring that arm removes the silent recovery, so the state has to be **said**
            # rather than absorbed — `corpus_stats` reports it as `unavailable` and the Settings
            # panel offers Rebuild as the fix. Never raise: an unwritable data home must not stop
            # the app from answering (inform, don't block).
            if self.keyword_index_unavailable:
                log.warning(
                    "keyword_index_unavailable",
                    hint="vector-only retrieval; rebuild the keyword index to restore terms",
                )
            else:
                log.warning("empty_index", hint="vector-only until documents are ingested")
            self.ensemble = EnsembleRetriever(retrievers=[vector], weights=[1.0])

        # The reranker is loaded **lazily** (see the `reranker` property): its weights are a
        # measured 3.7 s of a 16.1 s cold launch, and nothing needs it until a question has already
        # been retrieved for — by which point the first query absorbs the load invisibly.
        # Baseline: tests/eval/baselines/stage_profile_2026-07-28.md.
        self._reranker: CrossEncoder | None = None

        log.info("loading_llm")
        # The *effective* generation provider/model — starts at the config default; a caller
        # (ChatController, at construction or via a live switch) may move it with set_chat_model
        # (ADR-011, U1c). Read this, not the LLM_PROVIDER/LLM_MODEL constants, to know what's live.
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL
        self.llm = self._build_llm()

    def _open_sparse_index(self, chroma_path: str, collection: str) -> SparseIndex | None:
        """Open the on-disk sparse arm, building it if the corpus moved (ADR-036).

        ``None`` means there is **no keyword arm this process** — since ADR-038 there is no second
        implementation to fall back to. Two distinct causes, separated by ``self._corpus_empty``
        because they mean different things to the user: an empty corpus (nothing to index — a
        supported state) versus a build that failed over a real corpus (a degradation the settings
        surface has to report). Never raised: an unwritable data home must not stop the app from
        answering, only from matching exact terms.

        The build streams pages straight from Chroma into SQLite, so it never materialises the
        corpus (`chroma_read.iter_pages`); a build that accumulated first would pay the peak the
        index exists to remove.
        """
        # Ids only — no documents, metadata or embeddings — so this is a measured ~0.2 s at 33k
        # chunks against the seconds a full read costs. It is the corpus fingerprint (ADR-035/036).
        # Streamed page by page rather than collected: accumulating the id list measured **159 MB
        # of working set at 33k chunks** and was, after this ADR, the only corpus-linear memory
        # left at launch. The digest is order-independent, so paging cannot change it.
        empty = True

        def id_pages() -> Iterator[list[str]]:
            nonlocal empty
            for page in iter_pages(self.db, include=[]):
                ids = [str(i) for i in page["ids"]]
                if ids:
                    empty = False
                yield ids

        path = sparse_index.index_path(chroma_path)
        stamp = sparse_index.fingerprint_from_pages(collection, id_pages())
        # Recorded from the scan that just happened rather than re-derived: the constructor needs
        # to tell "empty corpus" from "index unavailable", and `chunk_count()` would be a second
        # full id scan of the store to learn what this pass already knows.
        self._corpus_empty = empty

        opened = sparse_index.open_index(path, stamp)
        if opened is not None:
            return opened
        if empty:
            # Robustness contract: a 0-document corpus is a supported state. Building an empty
            # index would only be a stale-file hazard for the first real ingest.
            return None
        try:
            return sparse_index.SparseIndex.build(path, stamp, self._iter_retrievable_chunks())
        except (OSError, sqlite3.Error) as e:
            log.warning("sparse_index_build_failed", error=str(e), path=str(path))
            return None

    @property
    def sparse_index_active(self) -> bool:
        """Whether the on-disk keyword arm (ADR-036) is serving this process.

        Since ADR-038 there is no second implementation, so False means **retrieval is
        vector-only** — either the corpus is empty or the index could not be opened or built.
        Public because the answer is a fact about the running pipeline that the settings surface
        reports; `_sparse` itself stays private.
        """
        return self._sparse is not None

    @property
    def keyword_index_unavailable(self) -> bool:
        """True when a non-empty corpus has no keyword arm — the state that needs saying.

        Distinct from `not sparse_index_active`, which is also true for an empty library. That
        difference is the whole point: an empty corpus is nothing to report, while a corpus whose
        index failed to build is answering questions with half its retrieval and no error anywhere
        the user can see.
        """
        return self._sparse is None and not self._corpus_empty

    def rebuild_sparse_index(self) -> int:
        """Rebuild the on-disk keyword index from the store and swap it into the live pipeline.

        The index self-heals at the next launch (a changed corpus moves the fingerprint), so this
        exists only to save a restart after an ingest — which is why the Settings panel offers it
        and why it is **not** destructive: it rewrites derived data that can always be rebuilt.

        Three things have to move together, and forgetting any one of them serves stale results
        from an index the user just rebuilt:

        1. the handle (`self._sparse`) — the old one is closed, so its file can be replaced;
        2. the prebuilt whole-corpus ensemble, whose `SparseRetriever` binds the old handle;
        3. the scoped-ensemble LRU, whose entries bind it too.

        Returns the number of chunks indexed.

        **It is also the recovery path, which is new since ADR-038.** Before the legacy arm was
        retired, a pipeline with no live index was serving keyword results from the in-RAM fallback
        and a rebuild here was meaningless, so this raised. Now that state means keyword matching
        is *off*, and rebuilding is exactly what fixes it — refusing would leave the user staring
        at a button that declines to do the one thing it is for. It therefore runs whether or not
        an index is live, and only an empty corpus is turned away: there is nothing to index, and
        writing an empty index would just be a stale-file hazard for the first real ingest.

        Emptiness is re-derived here from the fingerprint scan rather than read off
        ``self._corpus_empty``, which is a *construction-time* snapshot. The whole point of this
        method is to be called after an ingest, and a pipeline that launched against an empty
        library would otherwise refuse to index the documents the user just added.
        """
        collection = get_collection_name(get_active_model_name())
        chroma_path = PC_CHROMA_PATH if USE_PARENT_CHILD else CHROMA_PATH
        path = sparse_index.index_path(chroma_path)
        empty = True

        def id_pages() -> Iterator[list[str]]:
            nonlocal empty
            for page in iter_pages(self.db, include=[]):
                ids = [str(i) for i in page["ids"]]
                if ids:
                    empty = False
                yield ids

        stamp = sparse_index.fingerprint_from_pages(collection, id_pages())
        self._corpus_empty = empty
        if empty:
            raise RuntimeError("the corpus is empty; there is nothing to index")

        if self._sparse is not None:
            self._sparse.close()
        rebuilt = sparse_index.SparseIndex.build(path, stamp, self._iter_retrievable_chunks())
        self._sparse = rebuilt
        self._scoped.clear()
        vector = self.db.as_retriever(
            search_kwargs={"k": CANDIDATE_K, "filter": {"keep_for_retrieval": {"$ne": False}}}
        )
        self.ensemble = EnsembleRetriever(
            retrievers=[SparseRetriever(index=rebuilt, k=CANDIDATE_K), vector],
            weights=list(self._weights),
        )
        log.info("sparse_index_rebuilt", chunks=rebuilt.chunks)
        return rebuilt.chunks

    def _iter_retrievable_chunks(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Stream ``(text, metadata)`` for every chunk the retriever is allowed to return.

        The ``keep_for_retrieval is False`` exclusion mirrors the vector arm's filter, so both arms
        see the same corpus — it is applied here, once, at build time, rather than per query.
        """
        excluded = 0
        for page in iter_pages(self.db, include=["documents", "metadatas"]):
            for text, meta in zip(page["documents"], page["metadatas"], strict=True):
                if meta and meta.get("keep_for_retrieval") is False:
                    excluded += 1
                    continue
                yield text, dict(meta or {})
        log.info("bm25_excludes", count=excluded)

    def _parent_text_for(self, doc: Document) -> str | None:
        """The parent block a candidate belongs to, or ``None`` if there is none to expand into.

        **Metadata first, then the index, and the order is deliberate.** The vector arm returns
        documents straight from Chroma, which still stores `parent_text` on every child, and a
        document ingested *after* this pipeline was constructed can only be expanded that way — the
        on-disk `parents` table (ADR-036) is a snapshot of the corpus at construction time.
        Preferring metadata therefore keeps the live-ingest case working exactly as it did before
        KI-32, and makes the lookup invisible to any caller handing in a fully-populated document
        (including every existing test). The lookup is what the sparse arm needs, since its chunks
        no longer carry the text.

        ``None`` when there is no index either — a vector-only process (ADR-038) never produces a
        sparse candidate that lacks metadata, so this is a genuinely unreachable-in-practice branch
        kept honest rather than asserted away.
        """
        text = doc.metadata.get("parent_text")
        if text:
            return str(text)
        doc_hash = doc.metadata.get("doc_hash")
        parent_index = doc.metadata.get("parent_index")
        if doc_hash is None or parent_index is None:
            return None
        if self._sparse is None:
            return None
        return self._sparse.parent_text(str(doc_hash), int(parent_index))

    @property
    def reranker(self) -> CrossEncoder:
        """The cross-encoder, loaded on first use and cached for the process lifetime.

        A property rather than an eager attribute so that constructing a pipeline — which the API
        does in its lifespan, before any question exists — does not pay the weight load. Callers
        still read ``self.reranker``, so the seam is invisible to them; a test stubbing the
        reranker can keep assigning to it (the setter below) or set ``_reranker`` directly.
        """
        if self._reranker is None:
            log.info("loading_reranker")
            self._reranker = CrossEncoder("BAAI/bge-reranker-base", **_sigmoid_activation_kwarg())
        return self._reranker

    @reranker.setter
    def reranker(self, value: CrossEncoder) -> None:
        """Kept so existing code (and tests) can inject a fake by plain assignment."""
        self._reranker = value

    def _build_llm(self) -> Any:
        """Build the streaming analysis model from ``LLM_PROVIDER``/``LLM_MODEL``."""
        return build_chat_model(LLM_PROVIDER, LLM_MODEL)

    def set_chat_model(self, provider: str, model: str) -> None:
        """Swap **only** the streaming generation model (ADR-011, U1c desktop provider switch).

        The embedder, vector store, BM25 index, and reranker are untouched — this rebuilds
        nothing expensive, no network I/O (`build_chat_model` only constructs a thin API-client
        wrapper). ``rewrite``/``stream_answer``/``expand_query`` all bind
        ``chain = PROMPT | self.llm`` **fresh on every call**, so an in-flight turn (which already
        captured the old ``self.llm`` in its own local ``chain``) finishes unaffected, and the
        very next call picks up the new model automatically — no extra synchronization needed.
        Idempotent; never assigns any module-global (``config.LLM_PROVIDER`` etc. stay untouched).
        """
        self.llm = build_chat_model(provider, model)
        self.provider = provider
        self.model = model

    def _ensemble_for(self, scope: frozenset[str]) -> EnsembleRetriever:
        """Build (or reuse) an ensemble restricted to the documents in ``scope``.

        Both arms are scoped **before** scoring (ADR-025 fork 3; post-rerank filtering was
        rejected because recall collapses exactly when the scope is small):

        - vector — the same ``keep_for_retrieval`` filter ANDed with ``doc_hash $in [...]``;
        - sparse — a ``WHERE doc_hash IN (...)`` inside the ranked query (ADR-036).

        **The cost this memo was built for is gone.** Scoping used to mean rebuilding BM25 over the
        subset (~20 µs/chunk measured, RG-020), which is why the result is memoised in a small LRU
        keyed on ``scope`` (``_SCOPED_ENSEMBLE_CACHE_SIZE`` entries). With the index on disk the
        scoped arm is one bound parameter list, so the memo now saves only retriever construction
        — kept because it is cheap and because the key semantics (the hash set *is* the identity,
        so a membership edit can never reuse a stale entry) are the part worth having.
        ``self.db``/``self.embeddings``/``self.reranker`` are untouched: ``as_retriever`` is a thin
        wrapper, so nothing expensive is reloaded.

        Note the scoped sparse arm scores against **subset statistics** (avgdl/IDF differ from the
        global index). That is correct — the arm is told to rank within the folder — but it means
        scoped and unscoped scores are not directly comparable (RG-020).
        """
        cached = self._scoped.get(scope)
        if cached is not None:
            self._scoped.move_to_end(scope)  # mark most-recently-used
            return cached

        vector = self.db.as_retriever(
            search_kwargs={
                "k": CANDIDATE_K,
                "filter": {
                    "$and": [
                        {"keep_for_retrieval": {"$ne": False}},
                        {"doc_hash": {"$in": sorted(scope)}},
                    ]
                },
            }
        )
        in_index = self._sparse.doc_hashes() & scope if self._sparse is not None else set()
        if self._sparse is not None and in_index:
            ensemble = EnsembleRetriever(
                retrievers=[
                    SparseRetriever(index=self._sparse, k=CANDIDATE_K, scope=frozenset(in_index)),
                    vector,
                ],
                weights=list(self._weights),
            )
        else:
            # Vector-only, honestly: either there is no keyword index at all (ADR-038), or the
            # scope names documents the index does not hold — every chunk excluded, or the
            # documents removed since it was built. Never widen the scope to compensate.
            log.warning("scoped_sparse_empty", scope_size=len(scope))
            ensemble = EnsembleRetriever(retrievers=[vector], weights=[1.0])
        log.info("scoped_ensemble_built", docs=len(scope), arm="sparse_index")
        self._scoped[scope] = ensemble
        self._scoped.move_to_end(scope)  # newest = most-recently-used
        while len(self._scoped) > _SCOPED_ENSEMBLE_CACHE_SIZE:
            self._scoped.popitem(last=False)  # evict least-recently-used
        return ensemble

    def retrieve(
        self, query: str, top_k: int = TOP_K, *, scope: frozenset[str] | None = None
    ) -> list[Document]:
        """Retrieve top-k documents for `query`. Reranker scores discarded."""
        return [doc for doc, _ in self.retrieve_with_scores(query, top_k, scope=scope)]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = TOP_K,
        *,
        use_multi_query: bool | None = None,
        scope: frozenset[str] | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve top-k as ``(doc, reranker_score)`` pairs.

        Used by the provenance card to record per-chunk attribution and
        by anything that wants to inspect reranker confidence (e.g.,
        Phase 6 dual-interpretation gating).

        ``use_multi_query`` (ADR-010 / feature-rag-sandbox.md) is a request-scoped override
        of the ``USE_MULTI_QUERY`` config default: ``None`` preserves today's behaviour
        (follows the global), ``True``/``False`` forces expansion on/off for this call only —
        no module-global is ever assigned, so concurrent calls can't leak into each other.

        ``scope`` (ADR-025 F2) restricts retrieval to a set of ``doc_hash`` values — a **content
        filter**, not a quality knob: it changes *which documents* are searchable, never *how*
        retrieval works. ``None`` = the whole library (today's path, byte-identical); a non-empty
        set scopes both arms before scoring; an **empty** set retrieves nothing.
        """
        # ADR-025 F2: an EMPTY scope retrieves nothing. It must never widen to the whole corpus —
        # answering over every document when the caller asked for a folder is the `is_archived`
        # lie this feature exists to kill. Returning here also skips the retrievers entirely.
        if scope is not None and not scope:
            log.info("retrieval_scope_empty")
            return []
        # ``scope is None`` keeps the prebuilt ensemble and constructs no filter — the unscoped
        # path stays byte-identical (spec S4, guarded by a test).
        ensemble = self.ensemble if scope is None else self._ensemble_for(scope)

        effective_multi_query = USE_MULTI_QUERY if use_multi_query is None else use_multi_query
        # Multi-Query: generate variations if enabled
        queries = self.expand_query(query) if effective_multi_query else [query]

        # Collect candidates from all queries
        all_candidates: list[Document] = []
        seen_ids: set[str] = set()
        for q in queries:
            candidates = ensemble.invoke(q)
            for doc in candidates:
                # R6: dedup on a full-content hash, not a 50-char prefix — distinct chunks that
                # share a prefix (repeated headers; pre-R1 KI-14 placeholder-prefixed chunks)
                # silently collapsed into one, dropping real candidates before rerank.
                content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
                doc_id = doc.metadata.get("doc_hash", "") + "_" + content_hash
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_candidates.append(doc)

        if not all_candidates:
            return []

        # Bound the cross-encoder input (RERANK_CANDIDATE_CAP). Single-query retrieval unions at
        # most 2*CANDIDATE_K docs, which is < the cap, so the default path is byte-identical; the
        # cap only bites the opt-in multi-query path, where the union grew ~4x. Candidates are
        # accumulated original-query-first with first-seen dedup, so truncating the tail drops the
        # lowest-priority cross-variation hits, never the primary query's. (RG-022.)
        if len(all_candidates) > RERANK_CANDIDATE_CAP:
            log.info(
                "rerank_input_capped",
                candidates=len(all_candidates),
                cap=RERANK_CANDIDATE_CAP,
                queries=len(queries),
            )
            all_candidates = all_candidates[:RERANK_CANDIDATE_CAP]

        # Rerank against the original query.
        # The ignore is the sentence-transformers stub, not this call: its `predict` overloads
        # model a *single* pair or a flat list, never the batch-of-pairs form that is the
        # documented (and shipped) way to score N candidates. Making `reranker` a typed property
        # is what surfaced it — the eager attribute had been inferred as untyped, so this call site
        # was never checked at all. Runtime shape is unchanged.
        pairs = [[query, doc.page_content] for doc in all_candidates]
        scores = self.reranker.predict(pairs)  # type: ignore[arg-type]
        ranked: list[tuple[Document, float]] = sorted(
            zip(all_candidates, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        # Parent-child: dedup by parent BEFORE applying top_k. The
        # reranker_score we return is the *child*'s score that won — the
        # parent is the LLM context, the child is the retrieval evidence.
        if USE_PARENT_CHILD:
            seen_parents: set[tuple[Any, ...]] = set()
            deduped: list[tuple[Document, float]] = []
            for doc, score in ranked:
                # KI-32 step 1: the BM25 arm's copy of a chunk no longer carries `parent_text`, so
                # the text is resolved here (metadata first, then the per-parent map) instead of
                # being read off the candidate. Only the parents that actually win are touched —
                # <= TOP_K dict lookups per turn against ~5x the corpus held in RAM.
                parent_text = self._parent_text_for(doc)
                parent_key = (
                    doc.metadata.get("filename"),
                    doc.metadata.get("parent_index"),
                )

                if parent_text and parent_key not in seen_parents:
                    seen_parents.add(parent_key)
                    new_doc = Document(
                        page_content=parent_text,
                        metadata={k: v for k, v in doc.metadata.items() if k != "parent_text"},
                    )
                    deduped.append((new_doc, float(score)))
                    if len(deduped) >= top_k:
                        break
            return deduped

        # No parent-child: just take top_k
        return [(doc, float(score)) for doc, score in ranked[:top_k]]

    def rewrite(
        self,
        question: str,
        history: list[dict[str, str]],
        counter: Any = None,
    ) -> str:
        if not history:
            return question
        chain = REWRITE_PROMPT | self.llm
        callbacks = [counter] if counter else []
        result = chain.invoke(
            {"history": format_history(history), "question": question},
            config={"callbacks": callbacks},
        )
        return result.content if hasattr(result, "content") else str(result)

    def stream_answer(
        self,
        question: str,
        docs: list[Document],
        counter: Any = None,
        llm: Any | None = None,
    ) -> Generator[str, None, None]:
        """Stream the answer over ``docs``. ``llm`` (ADR-011) pins the turn to a caller's
        snapshot of the generation model: this is a generator, so the chain binds lazily at
        the first token — without the pin, a concurrent ``set_chat_model`` landing between
        the caller's snapshot and that first token would stream on a model the caller never
        recorded. ``None`` = bind ``self.llm`` live (pre-ADR-011 behaviour)."""
        context = format_docs_for_prompt(docs)
        chain = ANSWER_PROMPT | (llm if llm is not None else self.llm)
        callbacks = [counter] if counter else []
        for chunk in chain.stream(
            {"context": context, "question": question},
            config={"callbacks": callbacks},
        ):
            yield chunk.content if hasattr(chunk, "content") else str(chunk)

    def chunk_count(self) -> int:
        return len(get_all(self.db, include=[])["ids"])

    def expand_query(self, query: str) -> list[str]:
        """Generate 3 alternative phrasings of the query.
        Returns a list including the original plus 3 variations.
        """
        chain = MULTI_QUERY_PROMPT | self.llm
        response = chain.invoke({"question": query})
        text = response.content if hasattr(response, "content") else str(response)

        # Parse the JSON array. Be defensive -- LLMs sometimes wrap in markdown.
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            import json

            variations = json.loads(text)
            if not isinstance(variations, list):
                # R6: valid JSON that isn't a list → no variations (the original query is
                # prepended below regardless). Was ``[query]``, which line-262 prepended a
                # SECOND time, so the ensemble ran the same query twice. Match the except branch.
                variations = []
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, fall back to just the original query
            log.warning("multi_query_parse_failed", hint="using original query only")
            variations = []

        # Always include the original query
        return [query] + [v for v in variations if isinstance(v, str) and v.strip()]


# ============================================================
# Formatting helpers
# ============================================================


def format_citation(doc: Document, idx: int) -> str:
    name = doc.metadata.get("filename", "unknown")
    page = doc.metadata.get("page")
    section = doc.metadata.get("section")
    parts = [f"[{idx}] {name}"]
    if page:
        parts.append(f"p.{page}")
    if section:
        parts.append(f'"{section}"')
    return " \xb7 ".join(parts)


def format_docs_for_prompt(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        header = f"[Source {i + 1}: {filename}"
        if page:
            header += f", page {page}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_history(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "(no prior messages)"
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages[-6:])
