"""R6 guard tests — BM25 preprocessing, candidate dedup, multi-query expansion.

Deterministic, no models / Chroma / network: BM25Retriever is pure ``rank_bm25``;
``expand_query`` / ``retrieve_with_scores`` are exercised on a bare pipeline instance
(``__new__``) with fake ``llm`` / ``ensemble`` / ``reranker``.
"""

from __future__ import annotations

import pytest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from doc_assistant.keywords import tokenize
from doc_assistant.pipeline import RAGPipeline, resolve_ensemble_weights

# ---- fix 1: BM25 preprocess_func (casefold + tech-token) --------------------


def test_bm25_preprocess_func_matches_case_where_default_split_misses() -> None:
    # The target says "BM25" (uppercase); the distractor owns the secondary term "ranking".
    # Background docs keep both terms genuinely rare so BM25's IDF stays positive.
    docs = [
        Document(page_content="BM25 BM25 BM25 scoring", metadata={"filename": "target"}),
        Document(page_content="neural ranking model", metadata={"filename": "distractor"}),
        Document(page_content="a note about the weather today", metadata={"filename": "f1"}),
        Document(page_content="cooking recipes and food ideas", metadata={"filename": "f2"}),
        Document(page_content="local sports news roundup", metadata={"filename": "f3"}),
    ]
    query = "bm25 ranking"

    # Default preprocess_func is a bare ``text.split()``: the query's "bm25" never equals
    # the target's "BM25", so only the distractor's "ranking" matches → wrong doc wins.
    default = BM25Retriever.from_documents(docs)
    default.k = 1
    assert default.invoke(query)[0].metadata["filename"] == "distractor"

    # keywords.tokenize casefolds both sides → the target's three "bm25" hits win decisively.
    fixed = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
    fixed.k = 1
    assert fixed.invoke(query)[0].metadata["filename"] == "target"


# ---- fix 2: candidate dedup on a full-content hash -------------------------


def _bare_pipeline() -> RAGPipeline:
    return RAGPipeline.__new__(RAGPipeline)


def test_dedup_keeps_distinct_chunks_sharing_a_50char_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", False)
    monkeypatch.setattr("doc_assistant.pipeline.USE_PARENT_CHILD", False)
    prefix = "H" * 60  # shared > 50 chars → the old key collapsed these two into one
    d1 = Document(page_content=prefix + " ALPHA", metadata={"doc_hash": "h", "filename": "f"})
    d2 = Document(page_content=prefix + " BETA", metadata={"doc_hash": "h", "filename": "f"})

    rag = _bare_pipeline()
    rag.ensemble = RunnableLambda(lambda _q: [d1, d2])

    class _FakeReranker:
        def predict(self, pairs: list) -> list[float]:
            return [0.9, 0.8][: len(pairs)]

    rag.reranker = _FakeReranker()

    out = rag.retrieve_with_scores("q", top_k=5)
    assert len(out) == 2  # both survive; the 50-char-prefix key had collapsed them to 1
    assert {doc.page_content for doc, _ in out} == {prefix + " ALPHA", prefix + " BETA"}


# ---- fix 3: expand_query never double-runs the original query ---------------


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


def _pipeline_returning(content: str) -> RAGPipeline:
    rag = _bare_pipeline()
    rag.llm = RunnableLambda(lambda _prompt_value: _FakeMsg(content))
    return rag


def test_expand_query_non_list_json_does_not_duplicate_the_query() -> None:
    # Valid JSON that isn't a list was the bug: variations=[query] → prepended twice.
    rag = _pipeline_returning('{"queries": "not a list"}')
    assert rag.expand_query("original query") == ["original query"]


def test_expand_query_parse_failure_falls_back_to_original_only() -> None:
    rag = _pipeline_returning("not json at all")
    assert rag.expand_query("q") == ["q"]


def test_expand_query_valid_list_prepends_original() -> None:
    rag = _pipeline_returning('["variation one", "variation two"]')
    assert rag.expand_query("q") == ["q", "variation one", "variation two"]


# ---- --bm25-weight: ensemble-weight resolution -----------------------------


def test_resolve_ensemble_weights_default_is_the_locked_split() -> None:
    # None → the config-locked BM25_WEIGHT (0.4) with the vector complement (0.6).
    weights = resolve_ensemble_weights(None)
    assert weights == pytest.approx([0.4, 0.6])


@pytest.mark.parametrize(
    ("bm25", "expected"),
    [(0.0, [0.0, 1.0]), (0.5, [0.5, 0.5]), (0.7, [0.7, 0.3]), (1.0, [1.0, 0.0])],
)
def test_resolve_ensemble_weights_complements_to_one(bm25: float, expected: list[float]) -> None:
    weights = resolve_ensemble_weights(bm25)
    assert weights == pytest.approx(expected)
    assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
def test_resolve_ensemble_weights_out_of_range_raises(bad: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0\.0, 1\.0\]"):
        resolve_ensemble_weights(bad)
