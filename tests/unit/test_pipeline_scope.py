"""Unit tests for folder-scoped retrieval (ADR-025 F2, feature-corpus-folders-scope.md).

The load-bearing assertions are the two that keep the feature honest:

* **S4** — ``scope=None`` takes the prebuilt ensemble and constructs no filter, so the unscoped
  path is byte-identical to pre-F2.
* **S3** — an *empty* scope retrieves nothing and never widens to the whole corpus. Answering
  over every document when the caller asked for a folder is the failure this feature exists to
  prevent, so it is tested as a behaviour, not left to inspection.

Model-free: a bare pipeline (``__new__``) with fake retrievers/reranker, plus a real
``BM25Retriever`` over four tiny documents where the subset statistics matter.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from doc_assistant.pipeline import RAGPipeline


class _FakeReranker:
    def predict(self, pairs: list[Any]) -> list[float]:
        # Descending, so ranking is deterministic regardless of candidate count.
        return [1.0 - 0.01 * i for i in range(len(pairs))]


class _RecordingDb:
    """Stands in for ``self.db``; records every ``as_retriever`` search_kwargs."""

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.calls: list[dict[str, Any]] = []

    def as_retriever(self, *, search_kwargs: dict[str, Any]) -> Any:
        self.calls.append(search_kwargs)
        hashes = _in_clause(search_kwargs)
        hits = [d for d in self.docs if hashes is None or d.metadata.get("doc_hash") in hashes]
        return RunnableLambda(lambda _q, _hits=hits: list(_hits))


def _in_clause(search_kwargs: dict[str, Any]) -> set[str] | None:
    """Pull the ``doc_hash $in [...]`` set out of a filter, or None if there isn't one."""
    for clause in search_kwargs.get("filter", {}).get("$and", []):
        if "doc_hash" in clause:
            return set(clause["doc_hash"]["$in"])
    return None


def _docs() -> list[Document]:
    return [
        Document(page_content="alpha bm25 retrieval", metadata={"doc_hash": "a", "filename": "a"}),
        Document(page_content="beta bm25 retrieval", metadata={"doc_hash": "b", "filename": "b"}),
        Document(page_content="gamma vectors dense", metadata={"doc_hash": "c", "filename": "c"}),
        Document(page_content="delta vectors dense", metadata={"doc_hash": "d", "filename": "d"}),
    ]


def _rig(monkeypatch: pytest.MonkeyPatch) -> RAGPipeline:
    monkeypatch.setattr("doc_assistant.pipeline.USE_PARENT_CHILD", False)
    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", False)
    rag = RAGPipeline.__new__(RAGPipeline)
    docs = _docs()
    rag._bm25_docs = docs
    rag._scoped = OrderedDict()
    rag._weights = [0.4, 0.6]
    rag.bm25_weight = 0.4
    rag.db = _RecordingDb(docs)
    rag.reranker = _FakeReranker()
    # The prebuilt (unscoped) ensemble: returns everything, and records that it was used.
    rag.unscoped_calls = []  # type: ignore[attr-defined]
    rag.ensemble = RunnableLambda(
        lambda q: (rag.unscoped_calls.append(q), list(docs))[1]  # type: ignore[attr-defined]
    )
    return rag


# --- S4: the unscoped path is untouched -------------------------------------------------------- #


def test_scope_none_uses_the_prebuilt_ensemble_and_builds_no_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag = _rig(monkeypatch)

    out = rag.retrieve_with_scores("bm25", top_k=10)

    assert len(out) == 4
    assert rag.unscoped_calls == ["bm25"]  # type: ignore[attr-defined]
    assert rag.db.calls == []  # no retriever rebuilt, so no filter was ever constructed
    assert not rag._scoped  # the unscoped path never populates the scoped cache


def test_retrieve_passes_scope_through(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _rig(monkeypatch)
    assert [d.metadata["doc_hash"] for d in rag.retrieve("bm25", top_k=10)] == ["a", "b", "c", "d"]
    assert [d.metadata["doc_hash"] for d in rag.retrieve("bm25", top_k=10, scope=frozenset({"a"}))]


# --- S3: an empty scope retrieves nothing, never everything ------------------------------------ #


def test_empty_scope_returns_nothing_and_touches_no_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole feature in one assertion: 'I can't honour your scope' must not become
    'I searched everything'."""
    rag = _rig(monkeypatch)

    assert rag.retrieve_with_scores("bm25", top_k=10, scope=frozenset()) == []
    assert rag.unscoped_calls == []  # type: ignore[attr-defined]
    assert rag.db.calls == []


# --- S6: both arms scope ----------------------------------------------------------------------- #


def test_scope_filters_the_vector_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _rig(monkeypatch)

    out = rag.retrieve_with_scores("vectors", top_k=10, scope=frozenset({"c", "d"}))

    assert {doc.metadata["doc_hash"] for doc, _ in out} <= {"c", "d"}
    assert rag.unscoped_calls == []  # type: ignore[attr-defined]
    assert len(rag.db.calls) == 1
    kwargs = rag.db.calls[0]
    assert _in_clause(kwargs) == {"c", "d"}
    # The pre-existing exclusion filter must survive the AND, not be replaced by the scope.
    assert {"keep_for_retrieval": {"$ne": False}} in kwargs["filter"]["$and"]


def test_scope_filters_the_bm25_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The BM25 index is rebuilt over the subset, so an out-of-scope document cannot be
    keyword-matched even when it is the best lexical hit."""
    rag = _rig(monkeypatch)

    out = rag.retrieve_with_scores("bm25", top_k=10, scope=frozenset({"a"}))

    assert {doc.metadata["doc_hash"] for doc, _ in out} == {"a"}


def test_a_scope_naming_no_indexed_chunk_falls_back_to_vector_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents in the folder but nothing retrievable: vector-only, still scoped — the scope is
    never widened to compensate for an empty BM25 subset."""
    rag = _rig(monkeypatch)

    out = rag.retrieve_with_scores("bm25", top_k=10, scope=frozenset({"ghost"}))

    assert out == []
    assert _in_clause(rag.db.calls[0]) == {"ghost"}


# --- S5: the LRU scoped-ensemble cache ------------------------------------------------- #


def test_repeated_scope_reuses_the_ensemble(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _rig(monkeypatch)

    rag.retrieve_with_scores("bm25", top_k=10, scope=frozenset({"a", "b"}))
    rag.retrieve_with_scores("bm25", top_k=10, scope=frozenset({"a", "b"}))
    assert len(rag.db.calls) == 1  # second turn hit the cache — no rebuild

    # A different scope is a different key, so it builds once.
    rag.retrieve_with_scores("bm25", top_k=10, scope=frozenset({"a"}))
    assert len(rag.db.calls) == 2
    assert _in_clause(rag.db.calls[1]) == {"a"}


def test_lru_keeps_alternating_scopes_warm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of the LRU over the old single slot: alternating between two folders must
    NOT rebuild BM25 every turn. Under the single slot this sequence rebuilt on every switch."""
    rag = _rig(monkeypatch)
    ab, a = frozenset({"a", "b"}), frozenset({"a"})

    rag.retrieve_with_scores("bm25", top_k=10, scope=ab)  # build ab
    rag.retrieve_with_scores("bm25", top_k=10, scope=a)  # build a
    assert len(rag.db.calls) == 2

    # Alternate back and forth: both are still warm, so no further rebuilds.
    rag.retrieve_with_scores("bm25", top_k=10, scope=ab)
    rag.retrieve_with_scores("bm25", top_k=10, scope=a)
    rag.retrieve_with_scores("bm25", top_k=10, scope=ab)
    assert len(rag.db.calls) == 2  # single-slot would have made this 5


def test_lru_evicts_the_least_recently_used_scope_past_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache is bounded: once more than _SCOPED_ENSEMBLE_CACHE_SIZE distinct scopes are seen,
    the least-recently-used one is evicted and must be rebuilt when it returns."""
    monkeypatch.setattr("doc_assistant.pipeline._SCOPED_ENSEMBLE_CACHE_SIZE", 2)
    rag = _rig(monkeypatch)
    a, b, c = frozenset({"a"}), frozenset({"b"}), frozenset({"c"})

    rag.retrieve_with_scores("bm25", top_k=10, scope=a)  # build a  -> cache [a]
    rag.retrieve_with_scores("bm25", top_k=10, scope=b)  # build b  -> cache [a, b]
    rag.retrieve_with_scores("bm25", top_k=10, scope=c)  # build c  -> evicts a, cache [b, c]
    assert len(rag.db.calls) == 3

    rag.retrieve_with_scores("bm25", top_k=10, scope=b)  # b still warm -> no rebuild
    assert len(rag.db.calls) == 3

    rag.retrieve_with_scores("bm25", top_k=10, scope=a)  # a was evicted -> rebuild
    assert len(rag.db.calls) == 4
