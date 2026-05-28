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


def _mock_anthropic(response_text: str) -> Any:
    """Mock that mimics anthropic.Anthropic().messages.create(...).content[0].text."""
    client = MagicMock()
    text_block = MagicMock()
    text_block.text = response_text
    response = MagicMock()
    response.content = [text_block]
    client.messages.create.return_value = response
    return client


def test_llm_judge_parses_clean_json():
    client = _mock_anthropic('{"faithfulness": 4, "relevance": 5, "completeness": 3}')
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(expected_answer="ref"), _out("answer"))
    assert math.isclose(r.value, 4.0, abs_tol=1e-6)
    assert r.details["faithfulness"] == 4
    assert r.details["relevance"] == 5
    assert r.details["completeness"] == 3


def test_llm_judge_handles_markdown_fenced_json():
    client = _mock_anthropic(
        '```json\n{"faithfulness": 3, "relevance": 3, "completeness": 3}\n```'
    )
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(expected_answer="ref"), _out("answer"))
    assert math.isclose(r.value, 3.0, abs_tol=1e-6)


def test_llm_judge_handles_broken_json():
    client = _mock_anthropic("not json at all")
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(expected_answer="ref"), _out("answer"))
    assert r.value == 0.0
    assert "judge call failed" in r.details["error"]


def test_llm_judge_handles_missing_field():
    client = _mock_anthropic('{"faithfulness": 4, "relevance": 5}')
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(expected_answer="ref"), _out("answer"))
    assert r.value == 0.0
    assert "bad judge response" in r.details["error"]


def test_llm_judge_no_expected():
    client = _mock_anthropic('{"faithfulness": 4, "relevance": 5, "completeness": 3}')
    scorer = LLMJudgeScorer(client)
    r = scorer(_case(), _out("answer"))
    assert r.value == 0.0
    # Should not even have called the client.
    client.messages.create.assert_not_called()


def test_llm_judge_call_is_isolated():
    """No system prompt, no history, temperature=0 — each call is reproducible."""
    client = _mock_anthropic('{"faithfulness": 5, "relevance": 5, "completeness": 5}')
    scorer = LLMJudgeScorer(client)
    scorer(_case(expected_answer="ref"), _out("answer"))

    call_kwargs = client.messages.create.call_args.kwargs
    # Single-turn: exactly one user message, no system field.
    assert call_kwargs.get("system") in (None, "")
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0]["role"] == "user"
    # Deterministic: temperature pinned to 0.
    assert call_kwargs.get("temperature") == 0.0


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
