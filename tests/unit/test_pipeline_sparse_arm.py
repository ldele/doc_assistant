"""How `RAGPipeline` uses the on-disk sparse arm (ADR-036, KI-32 step 2).

`test_sparse_index.py` pins the index itself; this pins the wiring, and above all the **memory
property the change exists for**: when the on-disk arm is serving, the pipeline holds no corpus in
RAM. Nothing else in the suite would notice if a future edit quietly reintroduced the 195 MB load —
retrieval would still work, tests would still pass, and the ceiling would be back.

Model-free: the pipeline is built with ``__new__`` and a Chroma-shaped fake, so no embedder, no
store and no reranker are loaded.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from doc_assistant import sparse_index
from doc_assistant.pipeline import RAGPipeline, SparseRetriever

_ROWS: list[tuple[str, dict[str, Any]]] = [
    (
        "dense passage retrieval encodes questions",
        {"doc_hash": "aaa", "filename": "dpr.pdf", "parent_index": 0, "parent_text": "PARENT A"},
    ),
    (
        "a cross-encoder reranks candidates",
        {"doc_hash": "aaa", "filename": "dpr.pdf", "parent_index": 0, "parent_text": "PARENT A"},
    ),
    (
        "colbert uses late interaction",
        {"doc_hash": "bbb", "filename": "colbert.pdf", "parent_index": 0, "parent_text": "PAR B"},
    ),
    (
        "an excluded chunk about retrieval",
        {"doc_hash": "ccc", "filename": "x.pdf", "parent_index": 0, "keep_for_retrieval": False},
    ),
]


class _FakeChroma:
    """Chroma-shaped `get()` over the rows above; `as_retriever` records its search_kwargs."""

    def __init__(self, rows: list[tuple[str, dict[str, Any]]]) -> None:
        self.rows = rows
        self.search_kwargs: list[dict[str, Any]] = []

    def get(self, *, limit=None, offset=0, where=None, include=None):
        page = self.rows[offset : offset + limit] if limit else self.rows
        out: dict[str, Any] = {"ids": [f"chunk-{offset + i:04d}" for i in range(len(page))]}
        if include:
            if "documents" in include:
                out["documents"] = [text for text, _ in page]
            if "metadatas" in include:
                out["metadatas"] = [dict(meta) for _, meta in page]
        return out

    def as_retriever(self, *, search_kwargs: dict[str, Any]) -> Any:
        self.search_kwargs.append(search_kwargs)
        return RunnableLambda(lambda _q: [])


def _pipeline(tmp_path, monkeypatch, rows=None) -> RAGPipeline:
    monkeypatch.setattr("doc_assistant.pipeline.USE_PARENT_CHILD", True)
    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", False)
    rag = RAGPipeline.__new__(RAGPipeline)
    rag.db = _FakeChroma(_ROWS if rows is None else rows)
    chroma_path = str(tmp_path / "chroma_pc")
    (tmp_path / "chroma_pc").mkdir(exist_ok=True)
    rag._sparse = rag._open_sparse_index(chroma_path, "langchain")
    rag._scoped = OrderedDict()
    rag._weights = [0.4, 0.6]
    return rag


def _constructed(tmp_path, monkeypatch, rows=_ROWS) -> RAGPipeline:
    """A **really constructed** pipeline: `__init__` runs, with fakes only at its boundaries.

    The rig above assigns the attributes itself, which is fine for behaviour tests and useless for
    the memory property — it would pass even if `__init__` loaded the whole corpus. So the arm
    decision is tested where it is actually made.
    """
    store = tmp_path / "chroma_pc"
    store.mkdir(parents=True, exist_ok=True)
    fake = _FakeChroma(rows)
    monkeypatch.setattr("doc_assistant.pipeline.PC_CHROMA_PATH", str(store))
    monkeypatch.setattr("doc_assistant.pipeline.USE_PARENT_CHILD", True)
    monkeypatch.setattr("doc_assistant.pipeline.get_embeddings", lambda _m: object())
    monkeypatch.setattr("doc_assistant.pipeline.get_active_model_name", lambda: "bge-base")
    monkeypatch.setattr("doc_assistant.pipeline.get_collection_name", lambda _m: "langchain")
    monkeypatch.setattr("doc_assistant.pipeline.Chroma", lambda **_kw: fake)
    monkeypatch.setattr("doc_assistant.pipeline.build_chat_model", lambda _p, _m: object())
    return RAGPipeline()


class TestTheMemoryProperty:
    def test_construction_holds_no_corpus_in_ram(self, tmp_path, monkeypatch):
        """**The reason this module exists**, asserted against a real `__init__`.

        Since ADR-038 there is no in-RAM arm to fall back to, so the property is simply that
        nothing corpus-sized is held: the pipeline exposes no list of `Document`s and no
        parent-text map, and the index reports its chunks from disk. A future edit that
        reintroduced a corpus-sized attribute would be caught by the scan below.
        """
        rag = _constructed(tmp_path, monkeypatch)

        assert rag._sparse is not None
        assert rag._sparse.chunks == 3  # the excluded chunk is not indexed
        corpus_sized = [
            name
            for name, value in vars(rag).items()
            if isinstance(value, (list, dict)) and len(value) >= 3
        ]
        assert corpus_sized == [], f"pipeline holds corpus-sized state: {corpus_sized}"

    def test_the_rig_variant_holds_no_corpus_either(self, tmp_path, monkeypatch):
        rag = _pipeline(tmp_path, monkeypatch)

        assert rag._sparse is not None
        assert rag._sparse.chunks == 3

    def test_excluded_chunks_never_reach_the_index(self, tmp_path, monkeypatch):
        """`keep_for_retrieval=False` is applied once at build time, mirroring the vector arm's
        filter, so both arms search the same corpus."""
        rag = _pipeline(tmp_path, monkeypatch)

        assert rag._sparse is not None
        assert "ccc" not in rag._sparse.doc_hashes()
        # Its own distinctive wording finds nothing of it — other chunks may still match the
        # generic terms, so assert on provenance rather than on an empty result.
        hits = rag._sparse.search("an excluded chunk about retrieval", k=10)
        assert all(d.metadata["filename"] != "x.pdf" for d in hits)


class TestWiring:
    def test_the_ensemble_uses_the_sparse_retriever(self, tmp_path, monkeypatch):
        rag = _pipeline(tmp_path, monkeypatch)
        vector = rag.db.as_retriever(search_kwargs={"k": 20})
        from langchain_classic.retrievers import EnsembleRetriever

        ensemble = EnsembleRetriever(
            retrievers=[SparseRetriever(index=rag._sparse, k=20), vector], weights=[0.4, 0.6]
        )

        hits = ensemble.invoke("dense passage")

        assert [d.metadata["filename"] for d in hits] == ["dpr.pdf"]

    def test_parent_text_falls_back_to_the_index(self, tmp_path, monkeypatch):
        """A sparse hit carries no `parent_text` (it lives once in its own table), so expansion
        has to resolve it through the index or the candidate is silently dropped."""
        rag = _pipeline(tmp_path, monkeypatch)
        hit = rag._sparse.search("dense passage", k=1)[0]

        assert "parent_text" not in hit.metadata
        assert rag._parent_text_for(hit) == "PARENT A"

    def test_metadata_still_wins_over_the_index(self, tmp_path, monkeypatch):
        """The vector arm's documents come straight from Chroma and still carry the text; a
        document ingested after construction can only be expanded that way."""
        rag = _pipeline(tmp_path, monkeypatch)
        doc = Document(
            page_content="child",
            metadata={"doc_hash": "aaa", "parent_index": 0, "parent_text": "ITS OWN"},
        )

        assert rag._parent_text_for(doc) == "ITS OWN"

    def test_a_scoped_turn_scopes_the_sparse_arm(self, tmp_path, monkeypatch):
        rag = _pipeline(tmp_path, monkeypatch)

        ensemble = rag._ensemble_for(frozenset({"bbb"}))

        sparse = ensemble.retrievers[0]
        assert isinstance(sparse, SparseRetriever)
        assert sparse.scope == frozenset({"bbb"})
        hits = sparse.invoke("late interaction")
        assert [d.metadata["filename"] for d in hits] == ["colbert.pdf"]

    def test_a_scope_the_index_does_not_hold_degrades_to_vector_only(self, tmp_path, monkeypatch):
        """Never widen a scope to compensate: answering over the whole corpus when the user asked
        for a folder is the failure ADR-025 F2 exists to prevent."""
        rag = _pipeline(tmp_path, monkeypatch)

        ensemble = rag._ensemble_for(frozenset({"not-in-the-index"}))

        assert len(ensemble.retrievers) == 1  # vector only
        assert not isinstance(ensemble.retrievers[0], SparseRetriever)

    def test_scoped_ensembles_are_memoised_and_bounded(self, tmp_path, monkeypatch):
        rag = _pipeline(tmp_path, monkeypatch)

        first = rag._ensemble_for(frozenset({"aaa"}))
        assert rag._ensemble_for(frozenset({"aaa"})) is first  # reused, not rebuilt

        for i in range(6):
            rag._ensemble_for(frozenset({f"scope-{i}"}))
        assert len(rag._scoped) <= 4  # _SCOPED_ENSEMBLE_CACHE_SIZE


class TestFallback:
    def test_an_empty_corpus_builds_no_index(self, tmp_path, monkeypatch):
        """Robustness contract: 0 documents is a supported state. Writing an empty index would
        only be a stale-file hazard for the first real ingest."""
        rag = _pipeline(tmp_path, monkeypatch, rows=[])

        assert rag._sparse is None
        assert not (tmp_path / sparse_index.INDEX_FILENAME).exists()

    def test_a_build_failure_degrades_instead_of_raising(self, tmp_path, monkeypatch):
        """An unwritable data home must not stop the app from answering (inform, don't block)."""
        rag = RAGPipeline.__new__(RAGPipeline)
        rag.db = _FakeChroma(_ROWS)

        def boom(*a: Any, **k: Any) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr(sparse_index.SparseIndex, "build", boom)

        assert rag._open_sparse_index(str(tmp_path / "chroma_pc"), "langchain") is None

    def test_a_failed_build_over_a_real_corpus_reports_unavailable_not_empty(
        self, tmp_path, monkeypatch
    ):
        """ADR-038's whole point. With no in-RAM arm to absorb it, a failed build means keyword
        matching is **off** — and the two ways of having no index must not look alike: an empty
        library is a supported state, this is a degradation the user has to be told about.

        Non-vacuous by construction: collapsing `keyword_index_unavailable` to
        `not sparse_index_active` makes the empty-corpus assertion below fail.
        """

        def boom(*a: Any, **k: Any) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr(sparse_index.SparseIndex, "build", boom)
        degraded = _constructed(tmp_path, monkeypatch)

        assert degraded.sparse_index_active is False
        assert degraded.keyword_index_unavailable is True
        # Vector-only, but still answering: one arm, no exception.
        assert len(degraded.ensemble.retrievers) == 1

        empty = _constructed(tmp_path / "empty", monkeypatch, rows=[])
        assert empty.sparse_index_active is False
        assert empty.keyword_index_unavailable is False, "an empty library is not a degradation"

    def test_a_stale_index_is_rebuilt_not_served(self, tmp_path, monkeypatch):
        """The corpus changed under a live index: opening must miss, and the rebuild must reflect
        the new content."""
        rag = _pipeline(tmp_path, monkeypatch)
        assert rag._sparse is not None and rag._sparse.chunks == 3
        rag._sparse.close()

        fresh = [("brand new text about frogs", {"doc_hash": "zzz", "parent_index": 0})]
        rag.db = _FakeChroma(fresh)
        rebuilt = rag._open_sparse_index(str(tmp_path / "chroma_pc"), "langchain")

        assert rebuilt is not None
        assert rebuilt.doc_hashes() == {"zzz"}


class TestIndexedDocHashes:
    """`indexed_doc_hashes` — the corpus an eval run measured (RG-021).

    It reuses the same three states the fallback tests above pin, and the reason it must keep
    them apart is that it feeds a *record*: "0 documents" and "I could not tell" are different
    facts about a run, and collapsing them to one empty value would make a degraded run look
    like a clean run over an empty corpus.
    """

    def test_a_live_index_answers_with_its_documents(self, tmp_path, monkeypatch):
        """Excluded chunks are already gone: the arm is built through the same
        `keep_for_retrieval` filter the vector arm applies, so `ccc` is not in the corpus the
        run could retrieve from."""
        rag = _constructed(tmp_path, monkeypatch)

        assert rag.indexed_doc_hashes == {"aaa", "bbb"}

    def test_an_empty_corpus_is_an_empty_set_not_none(self, tmp_path, monkeypatch):
        empty = _constructed(tmp_path / "empty", monkeypatch, rows=[])

        assert empty.indexed_doc_hashes == set()
        assert empty.indexed_doc_hashes is not None

    def test_a_failed_build_over_a_real_corpus_is_unknown(self, tmp_path, monkeypatch):
        """The one honest `None`: retrieval ran vector-only and this process never learned what
        the store held, so the run cannot claim a composition it did not observe."""

        def boom(*a: Any, **k: Any) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr(sparse_index.SparseIndex, "build", boom)
        degraded = _constructed(tmp_path, monkeypatch)

        assert degraded.keyword_index_unavailable is True
        assert degraded.indexed_doc_hashes is None


@pytest.mark.parametrize("scope", [None, frozenset({"aaa"})])
def test_the_sparse_retriever_returns_documents_for_the_ensemble(tmp_path, monkeypatch, scope):
    """`EnsembleRetriever` fuses by reciprocal rank, so the arm only owes it an ordered list."""
    rag = _pipeline(tmp_path, monkeypatch)

    hits = SparseRetriever(index=rag._sparse, k=5, scope=scope).invoke("dense passage retrieval")

    assert hits and all(isinstance(d, Document) for d in hits)


class TestRebuild:
    """`rebuild_sparse_index` (ADR-037) — the Settings button's backend.

    The failure it must not have is silent: rebuilding the file while the ensemble keeps querying
    the old handle would report success and serve stale results, and nothing downstream would
    notice.
    """

    def test_it_rebuilds_from_the_current_store_and_rewires_the_ensemble(
        self, tmp_path, monkeypatch
    ):
        rag = _constructed(tmp_path, monkeypatch)
        assert rag._sparse is not None and rag._sparse.chunks == 3
        before = rag.ensemble.retrievers[0].index

        # The corpus changed underneath the running process.
        fresh = [("a brand new chunk about frogs", {"doc_hash": "zzz", "parent_index": 0})]
        rag.db = _FakeChroma(fresh)

        chunks = rag.rebuild_sparse_index()

        assert chunks == 1
        assert rag._sparse.doc_hashes() == {"zzz"}
        # 2 and 3: the prebuilt ensemble and the scoped memo must not hold the old handle.
        assert rag.ensemble.retrievers[0].index is rag._sparse
        assert rag.ensemble.retrievers[0].index is not before
        assert rag._scoped == {}
        assert [d.page_content for d in rag.ensemble.retrievers[0].invoke("frogs")] == [
            "a brand new chunk about frogs"
        ]

    def test_the_scoped_memo_is_cleared_so_a_folder_turn_cannot_serve_the_old_index(
        self, tmp_path, monkeypatch
    ):
        rag = _constructed(tmp_path, monkeypatch)
        rag._ensemble_for(frozenset({"aaa"}))
        assert rag._scoped

        rag.rebuild_sparse_index()

        assert rag._scoped == {}

    def test_it_recovers_a_pipeline_that_has_no_index(self, tmp_path, monkeypatch):
        """ADR-038 inverted this. While the legacy arm existed, a pipeline with no live index was
        still serving keyword results and a rebuild was meaningless, so this raised. Now that state
        means keyword matching is off, and rebuilding is the *fix* — refusing would leave the user
        with a button that declines to do the one thing it is for."""
        calls = {"n": 0}
        real_build = sparse_index.SparseIndex.build

        def fail_once(*a: Any, **k: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("read-only file system")
            return real_build(*a, **k)

        monkeypatch.setattr(sparse_index.SparseIndex, "build", fail_once)
        rag = _constructed(tmp_path, monkeypatch)
        assert rag.keyword_index_unavailable is True

        chunks = rag.rebuild_sparse_index()

        assert chunks == 3
        assert rag.sparse_index_active is True
        assert rag.keyword_index_unavailable is False
        assert rag.ensemble.retrievers[0].index is rag._sparse

    def test_it_refuses_on_an_empty_corpus(self, tmp_path, monkeypatch):
        """The one refusal left: nothing to index, and an empty index file would just be a
        stale-file hazard for the first real ingest."""
        rag = _constructed(tmp_path, monkeypatch, rows=[])

        with pytest.raises(RuntimeError, match="empty"):
            rag.rebuild_sparse_index()

    def test_it_indexes_documents_ingested_after_an_empty_launch(self, tmp_path, monkeypatch):
        """The regression the construction-time snapshot would have caused. A fresh install
        launches against 0 documents, the user ingests, then presses Rebuild — reading
        `_corpus_empty` from construction would refuse to index the documents just added."""
        rag = _constructed(tmp_path, monkeypatch, rows=[])
        assert rag._corpus_empty is True

        rag.db = _FakeChroma(_ROWS)  # the ingest happened while the process was running
        chunks = rag.rebuild_sparse_index()

        assert chunks == 3
        assert rag.sparse_index_active is True
