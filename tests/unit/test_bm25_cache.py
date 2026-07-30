"""Guard tests for the persisted BM25 snapshot (ADR-035).

The cache is a launch accelerator over a **retrieval input**, so the bar is higher than "it loads":
a stale or mis-paired snapshot would change what the app retrieves, silently. These tests therefore
split into two halves —

1. *equivalence*: the reconstructed index must score identically to a freshly built one;
2. *refusal*: every unhappy path (stale, corrupt, truncated, mis-paired, disabled) must fall back
   to the live build rather than serve a wrong index.

The second half is the one that matters. This session already paid for a silent truncation that
produced well-formed output (KI-31), and a cache is exactly where that shape of bug lives.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from langchain_core.documents import Document

from doc_assistant import bm25_cache
from doc_assistant.knowledge.keywords import tokenize
from doc_assistant.pipeline import _build_bm25

_TEXTS = [
    "Dense passage retrieval encodes questions and passages with a BM25-free dual encoder.",
    "The cross-encoder reranker scores query-passage pairs jointly.",
    "Hybrid retrieval fuses sparse BM25 candidates with dense vector candidates.",
    "Chunking splits a document into parent passages and smaller child chunks.",
    "An ensemble weights the sparse and dense arms before reranking.",
]


def _docs() -> list[Document]:
    return [Document(page_content=t, metadata={"i": i}) for i, t in enumerate(_TEXTS)]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A chroma-shaped directory: the snapshot sits beside it, keyed off `chroma.sqlite3`."""
    monkeypatch.delenv("DOC_BM25_CACHE", raising=False)
    chroma = tmp_path / "chroma_pc"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"pretend-store")
    return str(chroma)


def _ids(n: int) -> list[str]:
    return [f"chunk-{i:04d}" for i in range(n)]


def _save(
    store: str,
    docs: list[Document],
    tokens: list[list[str]] | None = None,
    parents: bm25_cache.ParentTexts | None = None,
) -> bool:
    return bm25_cache.save(
        store,
        "langchain",
        _ids(len(docs)),
        [(d.page_content, d.metadata) for d in docs],
        tokens if tokens is not None else [tokenize(d.page_content) for d in docs],
        parents if parents is not None else {},
    )


# ============================================================
# Equivalence — the cached index must not change retrieval
# ============================================================


class TestEquivalence:
    def test_cached_tokens_produce_the_same_ranking_and_scores(self):
        """The whole justification: identical output, less launch time."""
        docs = _docs()
        fresh = _build_bm25(docs, None)
        cached = _build_bm25(docs, [tokenize(d.page_content) for d in docs])

        for query in ("dense passage retrieval", "cross-encoder rerank", "chunking"):
            fresh_scores = fresh.vectorizer.get_scores(tokenize(query))
            cached_scores = cached.vectorizer.get_scores(tokenize(query))
            assert list(fresh_scores) == list(cached_scores), f"scores differ for {query!r}"

    def test_cached_retriever_returns_the_same_documents(self):
        docs = _docs()
        fresh = _build_bm25(docs, None)
        cached = _build_bm25(docs, [tokenize(d.page_content) for d in docs])
        fresh.k = cached.k = 3

        q = "hybrid sparse and dense retrieval"
        assert [d.page_content for d in fresh.invoke(q)] == [
            d.page_content for d in cached.invoke(q)
        ]

    def test_a_round_trip_through_disk_preserves_documents_and_metadata(self, store):
        docs = _docs()
        assert _save(store, docs)

        loaded = bm25_cache.load(store, "langchain", _ids(len(docs)))
        assert loaded is not None
        payload_docs, tokens, parents = loaded
        assert [t for t, _ in payload_docs] == _TEXTS
        assert [m["i"] for _, m in payload_docs] == list(range(len(_TEXTS)))
        assert tokens == [tokenize(t) for t in _TEXTS]
        assert parents == {}

    def test_the_parent_map_round_trips_with_its_tuple_keys(self, store):
        """Payload v2 (KI-32 step 1): parent text is stored once, keyed on (doc_hash, index)."""
        docs = _docs()
        parents = {("abc123", 0): "the first parent block", ("abc123", 1): "the second one"}
        assert _save(store, docs, parents=parents)

        loaded = bm25_cache.load(store, "langchain", _ids(len(docs)))
        assert loaded is not None
        assert loaded[2] == parents


# ============================================================
# Refusal — every unhappy path falls back to the live build
# ============================================================


class TestRefusesRatherThanServesAWrongIndex:
    def test_absent_cache_returns_none(self, store):
        assert bm25_cache.load(store, "langchain", _ids(5)) is None

    def test_replaced_chunks_invalidate_it_even_at_the_same_count(self, store):
        """The case a bare count misses — exactly what `ingest --rebuild` does (fresh UUIDs)."""
        docs = _docs()
        _save(store, docs)
        assert bm25_cache.load(store, "langchain", _ids(len(docs))) is not None

        rebuilt = [f"rebuilt-{i:04d}" for i in range(len(docs))]
        assert bm25_cache.load(store, "langchain", rebuilt) is None

    def test_a_changed_chunk_count_invalidates_it(self, store):
        _save(store, _docs())
        assert bm25_cache.load(store, "langchain", _ids(999)) is None

    def test_id_order_does_not_matter(self, store):
        """Chroma's page order is not contractual; the fingerprint must not depend on it."""
        docs = _docs()
        _save(store, docs)

        assert bm25_cache.load(store, "langchain", list(reversed(_ids(len(docs))))) is not None

    def test_touching_the_store_file_alone_does_NOT_invalidate(self, store):
        """The regression that made the first implementation never hit once.

        `_fingerprint` originally keyed on `chroma.sqlite3`'s mtime. But *opening* a
        `chromadb.PersistentClient` rewrites that mtime even for a pure read, so every launch
        invalidated the cache it had just written. Only the chunk ids may decide staleness.
        """
        docs = _docs()
        _save(store, docs)
        sqlite_file = Path(store) / "chroma.sqlite3"
        sqlite_file.write_bytes(b"a-read-just-touched-this-file-and-changed-its-size")

        assert bm25_cache.load(store, "langchain", _ids(len(docs))) is not None

    def test_a_different_collection_invalidates_it(self, store):
        """Switching embedding model points retrieval at another collection."""
        _save(store, _docs())
        assert bm25_cache.load(store, "specter2", _ids(len(_TEXTS))) is None

    def test_a_changed_tokenizer_invalidates_it(self, store, monkeypatch):
        """The index *is* tokens — a tokeniser change must not be served from cache."""
        _save(store, _docs())
        assert bm25_cache.load(store, "langchain", _ids(len(_TEXTS))) is not None

        def different_tokenizer(text: str) -> list[str]:
            return text.split()

        monkeypatch.setattr(
            "doc_assistant.knowledge.keywords.tokenize", different_tokenizer, raising=True
        )
        assert bm25_cache.load(store, "langchain", _ids(len(_TEXTS))) is None

    def test_a_corrupt_file_returns_none_instead_of_raising(self, store):
        _save(store, _docs())
        path = Path(store).parent / bm25_cache.CACHE_FILENAME
        path.write_bytes(b"not a pickle at all")

        assert bm25_cache.load(store, "langchain", _ids(len(_TEXTS))) is None

    def test_a_truncated_file_returns_none_instead_of_raising(self, store):
        _save(store, _docs())
        path = Path(store).parent / bm25_cache.CACHE_FILENAME
        blob = path.read_bytes()
        path.write_bytes(blob[: len(blob) // 2])

        assert bm25_cache.load(store, "langchain", _ids(len(_TEXTS))) is None

    def test_a_payload_that_is_not_a_dict_is_refused(self, store):
        path = Path(store).parent / bm25_cache.CACHE_FILENAME
        path.write_bytes(pickle.dumps(["not", "a", "dict"], protocol=5))

        assert bm25_cache.load(store, "langchain", _ids(len(_TEXTS))) is None

    def test_mispaired_docs_and_tokens_are_refused(self, store):
        """KI-31's shape: same file, well-formed, silently pairs doc i with doc j's terms."""
        docs = _docs()
        _save(store, docs)
        path = Path(store).parent / bm25_cache.CACHE_FILENAME
        payload = pickle.loads(path.read_bytes())  # a fixture this test just wrote
        payload["tokens"] = payload["tokens"][:-1]  # one short
        path.write_bytes(pickle.dumps(payload, protocol=5))

        assert bm25_cache.load(store, "langchain", _ids(len(docs))) is None

    def test_a_payload_without_the_parent_map_is_refused(self, store):
        """A v1-shaped payload must not be read as "no parents".

        Serving it would look fine and retrieve worse: with `parent_text` gone from metadata *and*
        no map, every BM25-only hit would fail to expand and be dropped from the answer. Same
        silent-degradation shape as KI-31, so the load seam refuses instead.
        """
        docs = _docs()
        _save(store, docs)
        path = Path(store).parent / bm25_cache.CACHE_FILENAME
        payload = pickle.loads(path.read_bytes())  # a fixture this test just wrote
        del payload["parents"]
        path.write_bytes(pickle.dumps(payload, protocol=5))

        assert bm25_cache.load(store, "langchain", _ids(len(docs))) is None

    def test_a_snapshot_from_the_previous_payload_version_is_stale(self, store, monkeypatch):
        """`_CACHE_VERSION` is in the fingerprint: v1 files are rebuilt, never reinterpreted."""
        monkeypatch.setattr(bm25_cache, "_CACHE_VERSION", 1)
        docs = _docs()
        _save(store, docs)
        assert bm25_cache.load(store, "langchain", _ids(len(docs))) is not None

        monkeypatch.setattr(bm25_cache, "_CACHE_VERSION", 2)
        assert bm25_cache.load(store, "langchain", _ids(len(docs))) is None

    def test_build_ignores_tokens_whose_length_disagrees(self):
        """Belt and braces at the build seam, not just the load seam."""
        docs = _docs()
        built = _build_bm25(docs, [tokenize(d.page_content) for d in docs][:-1])

        assert list(built.vectorizer.get_scores(tokenize("chunking"))) == list(
            _build_bm25(docs, None).vectorizer.get_scores(tokenize("chunking"))
        )


# ============================================================
# Switch + robustness contract
# ============================================================


class TestSwitchAndEmptyCorpus:
    def test_env_switch_disables_both_halves(self, store, monkeypatch):
        monkeypatch.setenv("DOC_BM25_CACHE", "0")
        assert bm25_cache.enabled() is False
        assert _save(store, _docs()) is False
        assert bm25_cache.load(store, "langchain", _ids(len(_TEXTS))) is None

    @pytest.mark.parametrize("value", ["0", "false", "no", "FALSE", " 0 "])
    def test_falsey_values_all_disable(self, monkeypatch, value):
        monkeypatch.setenv("DOC_BM25_CACHE", value)
        assert bm25_cache.enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "anything-else"])
    def test_everything_else_leaves_it_on(self, monkeypatch, value):
        monkeypatch.setenv("DOC_BM25_CACHE", value)
        assert bm25_cache.enabled() is True

    def test_an_empty_corpus_writes_nothing(self, store):
        """Robustness contract: 0 documents is a supported state, and a no-op here."""
        assert bm25_cache.save(store, "langchain", [], [], [], {}) is False
        assert not (Path(store).parent / bm25_cache.CACHE_FILENAME).exists()

    def test_an_unwritable_destination_is_logged_not_raised(self, store, monkeypatch):
        def boom(*a, **k):
            raise OSError("read-only file system")

        monkeypatch.setattr(bm25_cache.tempfile, "mkstemp", boom)
        assert _save(store, _docs()) is False  # no exception escapes

    def test_no_temp_files_are_left_behind(self, store):
        _save(store, _docs())
        assert list(Path(store).parent.glob("*.tmp")) == []
