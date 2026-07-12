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


# ---- ADR-010 / SPRINT-010 (U1): request-scoped use_multi_query override ----------------


def _retrieval_rig(monkeypatch: pytest.MonkeyPatch) -> tuple[RAGPipeline, list[str]]:
    """A bare pipeline with a fake ensemble/reranker and an ``expand_query`` spy, so a test
    can assert whether expansion ran without a real LLM call."""
    monkeypatch.setattr("doc_assistant.pipeline.USE_PARENT_CHILD", False)
    calls: list[str] = []
    rag = _bare_pipeline()
    rag.expand_query = lambda q: (calls.append(q), [q, "a variation"])[1]  # type: ignore[method-assign]
    doc = Document(page_content="hello", metadata={"doc_hash": "h", "filename": "f"})
    rag.ensemble = RunnableLambda(lambda _q: [doc])

    class _FakeReranker:
        def predict(self, pairs: list) -> list[float]:
            return [0.9] * len(pairs)

    rag.reranker = _FakeReranker()
    return rag, calls


def test_use_multi_query_override_ignores_global(monkeypatch: pytest.MonkeyPatch) -> None:
    # override=False skips expansion even though the global default is True.
    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", True)
    rag, calls = _retrieval_rig(monkeypatch)
    rag.retrieve_with_scores("q", top_k=5, use_multi_query=False)
    assert calls == []

    # override=True expands even though the global default is False.
    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", False)
    rag2, calls2 = _retrieval_rig(monkeypatch)
    rag2.retrieve_with_scores("q", top_k=5, use_multi_query=True)
    assert calls2 == ["q"]


def test_use_multi_query_none_follows_the_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", True)
    rag, calls = _retrieval_rig(monkeypatch)
    rag.retrieve_with_scores("q", top_k=5, use_multi_query=None)
    assert calls == ["q"]  # None preserves today's (global-driven) behaviour

    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", False)
    rag2, calls2 = _retrieval_rig(monkeypatch)
    rag2.retrieve_with_scores("q", top_k=5, use_multi_query=None)
    assert calls2 == []


# ---- ADR-011 / SPRINT-012 (U1c): the generation-model swap seam -------------------------


def test_set_chat_model_swaps_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _bare_pipeline()
    rag.llm = "old"
    rag.provider = "anthropic"
    rag.model = "claude-haiku-4-5-20251001"

    calls: list[tuple[str, str]] = []

    def fake_build(provider: str, model: str) -> str:
        calls.append((provider, model))
        return f"{provider}:{model}"

    monkeypatch.setattr("doc_assistant.pipeline.build_chat_model", fake_build)
    rag.set_chat_model("ollama", "llama3.1:8b")

    assert rag.llm == "ollama:llama3.1:8b"
    assert rag.provider == "ollama"
    assert rag.model == "llama3.1:8b"
    assert calls == [("ollama", "llama3.1:8b")]  # exactly one build call — no API call, no reload


def test_in_flight_chain_survives_a_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    # fork F: a chain already bound (`chain = PROMPT | self.llm`, as stream_answer/rewrite/
    # expand_query all do per call) before a swap must keep streaming on the OLD model; the
    # swap only rebinds `self.llm` for the NEXT call to pick up.
    rag = _bare_pipeline()
    rag.provider = "anthropic"
    rag.model = "old-model"
    rag.llm = RunnableLambda(lambda _x: "OLD_RESULT")

    prompt = RunnableLambda(lambda x: x)
    chain = prompt | rag.llm  # simulates a turn already in flight

    monkeypatch.setattr(
        "doc_assistant.pipeline.build_chat_model",
        lambda _p, _m: RunnableLambda(lambda _x: "NEW_RESULT"),
    )
    rag.set_chat_model("ollama", "new-model")

    assert chain.invoke("q") == "OLD_RESULT"  # the pre-built chain is unaffected by the swap
    assert rag.llm.invoke("q") == "NEW_RESULT"  # a fresh chain would use the new model


def test_stream_answer_llm_pin_defeats_the_lazy_bind_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2026-07-11 review follow-up: stream_answer is a generator, so `chain = PROMPT | self.llm`
    # binds at the FIRST TOKEN, not at call time — a swap landing between a caller's snapshot
    # and that first token would stream on the new model while the caller records the old one.
    # Passing llm= pins the turn to the snapshot; the unpinned control shows the race is real.
    rag = _bare_pipeline()
    rag.provider = "anthropic"
    rag.model = "old-model"
    rag.llm = RunnableLambda(lambda _x: _FakeMsg("OLD"))
    snapshot = rag.llm

    pinned = rag.stream_answer("q", [], llm=snapshot)  # generators created pre-swap,
    unpinned = rag.stream_answer("q", [])  # neither started yet

    monkeypatch.setattr(
        "doc_assistant.pipeline.build_chat_model",
        lambda _p, _m: RunnableLambda(lambda _x: _FakeMsg("NEW")),
    )
    rag.set_chat_model("ollama", "new-model")  # the swap lands before the first token

    assert "".join(pinned) == "OLD"  # pinned: the snapshotted instrument generates
    assert "".join(unpinned) == "NEW"  # unpinned control: the lazy bind picks up the swap
