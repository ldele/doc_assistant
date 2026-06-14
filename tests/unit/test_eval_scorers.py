"""Tests for eval scorers (Phase 5 / Feature 2).

LLM judge and embedding scorers are tested with mocked dependencies —
no API calls, no model load.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

from doc_assistant.eval.cases import EvalCase
from doc_assistant.eval.results import EvalOutput
from doc_assistant.eval.scorers import (
    CitationOverlapScorer,
    ContainsAllScorer,
    EmbeddingSimilarityScorer,
    ExactMatchScorer,
    FigureRetrievalScorer,
    LLMJudgeScorer,
    _cosine,
)

# ============================================================
# Helpers
# ============================================================


def _case(**kwargs: Any) -> EvalCase:
    defaults: dict[str, Any] = {"id": "c1", "query": "what?"}
    defaults.update(kwargs)
    return EvalCase(**defaults)


def _out(answer: str = "", citations: list[str] | None = None) -> EvalOutput:
    return EvalOutput(answer=answer, citations=citations or [])


# ============================================================
# ExactMatchScorer
# ============================================================


def test_exact_match_hit():
    r = ExactMatchScorer()(_case(expected_answer="hello"), _out("Hello"))
    assert r.value == 1.0


def test_exact_match_miss():
    r = ExactMatchScorer()(_case(expected_answer="hello"), _out("hellllo"))
    assert r.value == 0.0


def test_exact_match_case_sensitive_mode():
    r = ExactMatchScorer(case_sensitive=True)(_case(expected_answer="hello"), _out("Hello"))
    assert r.value == 0.0


def test_exact_match_no_output():
    r = ExactMatchScorer()(_case(expected_answer="hello"), None)
    assert r.value == 0.0
    assert "no output" in r.details["error"]


def test_exact_match_no_expected():
    r = ExactMatchScorer()(_case(), _out("hello"))
    assert r.value == 0.0
    assert "expected_answer" in r.details["error"]


# ============================================================
# ContainsAllScorer
# ============================================================


def test_contains_all_full_hit():
    r = ContainsAllScorer()(
        _case(expected_substrings=["sodium", "potassium"]),
        _out("Sodium and potassium drive the action potential."),
    )
    assert r.value == 1.0
    assert set(r.details["matched"]) == {"sodium", "potassium"}


def test_contains_all_partial_hit():
    r = ContainsAllScorer()(
        _case(expected_substrings=["sodium", "potassium"]),
        _out("Sodium only."),
    )
    assert math.isclose(r.value, 0.5)
    assert r.details["missing"] == ["potassium"]


def test_contains_all_no_substrings():
    r = ContainsAllScorer()(_case(), _out("anything"))
    assert r.value == 0.0
    assert "expected_substrings" in r.details["error"]


# ============================================================
# CitationOverlapScorer
# ============================================================


def test_citation_overlap_substring_match():
    # Bidirectional substring match: stem matches full filename
    r = CitationOverlapScorer()(
        _case(expected_citations=["hodgkin_huxley_1952", "hebb_1949"]),
        _out(citations=["hodgkin_huxley_1952.pdf", "hebb_1949.pdf", "noise.pdf"]),
    )
    assert r.value == 1.0
    assert r.details["extra"] == ["noise.pdf"]


def test_citation_overlap_partial():
    r = CitationOverlapScorer()(
        _case(expected_citations=["hodgkin_huxley_1952", "hebb_1949"]),
        _out(citations=["hodgkin_huxley_1952.pdf"]),
    )
    assert math.isclose(r.value, 0.5)
    assert r.details["missing"] == ["hebb_1949"]


def test_citation_overlap_no_expected():
    r = CitationOverlapScorer()(_case(), _out(citations=["x.pdf"]))
    assert r.value == 0.0
    assert "expected_citations" in r.details["error"]


# ============================================================
# EmbeddingSimilarityScorer
# ============================================================


def _orthogonal_embedder() -> Any:
    table = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "same": [1.0, 0.0, 0.0],
    }
    return lambda s: table.get(s, [0.0, 0.0, 0.0])


def test_embedding_similarity_identical():
    scorer = EmbeddingSimilarityScorer(_orthogonal_embedder())
    r = scorer(_case(expected_answer="a"), _out("same"))
    assert math.isclose(r.value, 1.0, abs_tol=1e-6)


def test_embedding_similarity_orthogonal():
    scorer = EmbeddingSimilarityScorer(_orthogonal_embedder())
    r = scorer(_case(expected_answer="a"), _out("b"))
    assert r.value == 0.0


def test_embedding_similarity_no_expected():
    scorer = EmbeddingSimilarityScorer(_orthogonal_embedder())
    r = scorer(_case(), _out("a"))
    assert r.value == 0.0


def test_cosine_handles_mismatched_dims():
    assert _cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_cosine_handles_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ============================================================
# LLMJudgeScorer (mocked client)
# ============================================================


def _mock_client(response_text: str) -> Any:
    """An ``LLMClient``-shaped mock: ``.complete(...)`` returns the text."""
    client = MagicMock()
    client.complete.return_value = response_text
    return client


def test_llm_judge_parses_clean_json():
    client = _mock_client('{"faithfulness": 4, "relevance": 5, "completeness": 3}')
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(expected_answer="ref"), _out("answer"))
    assert math.isclose(r.value, 4.0, abs_tol=1e-6)
    assert r.details["faithfulness"] == 4
    assert r.details["relevance"] == 5
    assert r.details["completeness"] == 3


def test_llm_judge_handles_markdown_fenced_json():
    client = _mock_client('```json\n{"faithfulness": 3, "relevance": 3, "completeness": 3}\n```')
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(expected_answer="ref"), _out("answer"))
    assert math.isclose(r.value, 3.0, abs_tol=1e-6)


def test_llm_judge_handles_broken_json():
    client = _mock_client("not json at all")
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(expected_answer="ref"), _out("answer"))
    assert r.value == 0.0
    assert "judge call failed" in r.details["error"]


def test_llm_judge_handles_missing_field():
    client = _mock_client('{"faithfulness": 4, "relevance": 5}')
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(expected_answer="ref"), _out("answer"))
    assert r.value == 0.0
    assert "bad judge response" in r.details["error"]


def test_llm_judge_no_expected():
    client = _mock_client('{"faithfulness": 4, "relevance": 5, "completeness": 3}')
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(), _out("answer"))
    assert r.value == 0.0
    # Should not even have called the client.
    client.complete.assert_not_called()


def test_llm_judge_call_is_isolated():
    """No system prompt, no history, temperature=0 — each call is reproducible."""
    client = _mock_client('{"faithfulness": 5, "relevance": 5, "completeness": 5}')
    scorer = LLMJudgeScorer(client)
    scorer(_case(expected_answer="ref"), _out("answer"))

    call = client.complete.call_args
    messages = call.args[0]
    # Single-turn: exactly one user message, no system message.
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert not any(m["role"] == "system" for m in messages)
    # Deterministic: temperature pinned to 0.
    assert call.kwargs.get("temperature") == 0.0


def test_llm_judge_prompt_instructs_reference_only_grading():
    """Prompt must tell the model to ignore its own knowledge."""
    from doc_assistant.eval.scorers import _JUDGE_PROMPT

    # The prompt's intent: use only the reference, not the model's
    # pretrained knowledge of the topic.
    assert "prior knowledge" in _JUDGE_PROMPT.lower() or "own" in _JUDGE_PROMPT.lower()
    assert "reference" in _JUDGE_PROMPT.lower()
    # Sub-rubric definitions present.
    assert "faithfulness" in _JUDGE_PROMPT.lower()
    assert "relevance" in _JUDGE_PROMPT.lower()
    assert "completeness" in _JUDGE_PROMPT.lower()


# ============================================================
# FigureRetrievalScorer (Feature 4c)
# ============================================================


def _out_retrieved(retrieved: list[dict[str, Any]]) -> EvalOutput:
    return EvalOutput(answer="", raw={"retrieved": retrieved})


def test_figure_retrieval_hit_by_filename_and_page():
    out = _out_retrieved(
        [
            {"filename": "dpr.pdf", "page": 3, "chunk_type": "text"},
            {
                "filename": "rag_lewis_2020.pdf",
                "page": 4,
                "chunk_type": "figure",
                "figure_id": "f1",
            },
        ]
    )
    case = _case(metadata={"expected_figure": {"filename": "rag_lewis_2020.pdf", "page": 4}})
    r = FigureRetrievalScorer()(case, out)
    assert r.value == 1.0
    assert r.details["matched"]["figure_id"] == "f1"


def test_figure_retrieval_miss_wrong_page():
    out = _out_retrieved(
        [{"filename": "rag_lewis_2020.pdf", "page": 9, "chunk_type": "figure", "figure_id": "f1"}]
    )
    case = _case(metadata={"expected_figure": {"filename": "rag_lewis_2020.pdf", "page": 4}})
    r = FigureRetrievalScorer()(case, out)
    assert r.value == 0.0
    assert r.details["n_figure_chunks"] == 1


def test_figure_retrieval_ignores_text_chunks_of_right_doc():
    # The right paper came back, but only as a text chunk — not a figure hit.
    out = _out_retrieved([{"filename": "rag_lewis_2020.pdf", "page": 4, "chunk_type": "text"}])
    case = _case(metadata={"expected_figure": {"filename": "rag_lewis_2020.pdf", "page": 4}})
    r = FigureRetrievalScorer()(case, out)
    assert r.value == 0.0
    assert r.details["n_figure_chunks"] == 0


def test_figure_retrieval_by_figure_id_exact():
    out = _out_retrieved(
        [{"filename": "x.pdf", "page": 2, "chunk_type": "figure", "figure_id": "abc-123"}]
    )
    case = _case(metadata={"expected_figure": {"figure_id": "abc-123"}})
    assert FigureRetrievalScorer()(case, out).value == 1.0


def test_figure_retrieval_no_expected_figure_is_skipped():
    r = FigureRetrievalScorer()(_case(), _out_retrieved([]))
    assert r.value == 0.0
    assert "expected_figure" in r.details["error"]
    assert r.is_skipped


def test_figure_retrieval_no_output():
    case = _case(metadata={"expected_figure": {"filename": "x.pdf"}})
    r = FigureRetrievalScorer()(case, None)
    assert r.value == 0.0
    assert "no output" in r.details["error"]
