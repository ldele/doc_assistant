"""Tests for the reviewer agent (Phase 6 / Integrity Chunk 2b).

Mocked Anthropic client — no API calls. Persistence covered via a temp
SQLite DB (same monkeypatch pattern as test_provenance.py).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from doc_assistant.provenance import AnswerProvenance, RetrievedChunk
from doc_assistant.reviewer import (
    _REVIEWER_PROMPT,
    ReviewResult,
    _format_evidence,
    get_reviews,
    persist_review,
    review_answer,
)


def _mock_anthropic(response_text: str) -> MagicMock:
    client = MagicMock()
    text_block = MagicMock()
    text_block.text = response_text
    response = MagicMock()
    response.content = [text_block]
    client.messages.create.return_value = response
    return client


def _prov(answer: str = "an answer") -> AnswerProvenance:
    chunks = [
        RetrievedChunk(filename="paper.pdf", page=3, section="Methods", chunk_excerpt="x"),
    ]
    return AnswerProvenance(id="abc", query="what?", answer=answer, retrieved_chunks=chunks)


# ============================================================
# Prompt assembly
# ============================================================


def test_reviewer_prompt_references_evidence_not_reference():
    """Reviewer judges answer against retrieved evidence, NOT a ground-truth reference."""
    assert "EVIDENCE" in _REVIEWER_PROMPT
    assert "retrieved" in _REVIEWER_PROMPT.lower()
    # NOT a ground-truth comparison
    assert "ground truth" not in _REVIEWER_PROMPT.lower()


def test_reviewer_prompt_lists_all_four_dimensions():
    for dim in (
        "faithfulness",
        "citation_density",
        "hedging_adequacy",
        "unsupported_claims_count",
    ):
        assert dim in _REVIEWER_PROMPT


def test_reviewer_prompt_instructs_no_prior_knowledge():
    assert "prior knowledge" in _REVIEWER_PROMPT.lower()


def test_format_evidence_handles_empty():
    assert _format_evidence([]) == "(no chunks retrieved)"


def test_format_evidence_includes_header_per_chunk():
    chunks = [
        RetrievedChunk(filename="a.pdf", page=4, section="Intro", chunk_excerpt="hello"),
        RetrievedChunk(filename="b.pdf", chunk_excerpt="world"),
    ]
    out = _format_evidence(chunks)
    assert "[1]" in out and "a.pdf" in out and "p.4" in out
    assert "[2]" in out and "b.pdf" in out


# ============================================================
# review_answer — parsing
# ============================================================


def test_review_answer_parses_clean_json():
    client = _mock_anthropic(
        '{"faithfulness": 4, "citation_density": 3, "hedging_adequacy": 5, '
        '"unsupported_claims_count": 1, "notes": "minor extrapolation"}'
    )
    result = review_answer(_prov(), client)
    assert result.error is None
    assert result.faithfulness == 4
    assert result.citation_density == 3
    assert result.hedging_adequacy == 5
    assert result.unsupported_claims_count == 1
    assert result.notes == "minor extrapolation"


def test_review_answer_handles_markdown_fence():
    client = _mock_anthropic(
        '```json\n{"faithfulness": 3, "citation_density": 3, '
        '"hedging_adequacy": 3, "unsupported_claims_count": 0, "notes": "ok"}\n```'
    )
    result = review_answer(_prov(), client)
    assert result.error is None
    assert result.faithfulness == 3


def test_review_answer_handles_broken_json():
    client = _mock_anthropic("not json at all")
    result = review_answer(_prov(), client)
    assert result.error is not None
    assert "reviewer call failed" in result.error


def test_review_answer_handles_missing_field():
    client = _mock_anthropic('{"faithfulness": 4}')
    result = review_answer(_prov(), client)
    assert result.error is not None
    assert "bad reviewer response" in result.error


def test_review_answer_is_isolated():
    """Single user message, no system, temperature=0 — same contract as the eval judge."""
    client = _mock_anthropic(
        '{"faithfulness": 4, "citation_density": 4, "hedging_adequacy": 4, '
        '"unsupported_claims_count": 0, "notes": ""}'
    )
    review_answer(_prov(), client)
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs.get("system") in (None, "")
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0]["role"] == "user"
    assert call_kwargs.get("temperature") == 0.0


# ============================================================
# Persistence roundtrip
# ============================================================


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh SQLite DB bound for the test scope. Same pattern as test_provenance."""
    db_file = tmp_path / "test.db"
    db_url = f"sqlite:///{db_file}"

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    test_engine = create_engine(db_url, future=True)
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, future=True
    )

    monkeypatch.setattr(session_mod, "_engine", test_engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", test_session_factory)

    yield db_file

    test_engine.dispose()


def test_persist_and_read_review(temp_db: Path):
    from doc_assistant.provenance import record_answer

    answer_id = record_answer(query="q", answer="a", retrieved_chunks=[])
    result = ReviewResult(
        faithfulness=4,
        citation_density=3,
        hedging_adequacy=5,
        unsupported_claims_count=1,
        notes="ok",
    )
    review_id = persist_review(answer_id, result, reviewer_kind="llm_haiku", model_name="haiku")
    assert review_id  # UUID generated

    reviews = get_reviews(answer_id)
    assert len(reviews) == 1
    r = reviews[0]
    assert r.faithfulness == 4
    assert r.citation_density == 3
    assert r.hedging_adequacy == 5
    assert r.unsupported_claims_count == 1
    assert r.notes == "ok"
    assert r.error is None


def test_get_reviews_orders_most_recent_first(temp_db: Path):
    import time as _t

    from doc_assistant.provenance import record_answer

    answer_id = record_answer(query="q", answer="a", retrieved_chunks=[])
    for i in range(3):
        persist_review(
            answer_id,
            ReviewResult(
                faithfulness=i, citation_density=3, hedging_adequacy=3, unsupported_claims_count=0
            ),
            reviewer_kind="llm_haiku",
        )
        _t.sleep(0.01)
    reviews = get_reviews(answer_id)
    assert len(reviews) == 3
    # Most recent first: last-persisted has faithfulness=2
    assert reviews[0].faithfulness == 2
    assert reviews[-1].faithfulness == 0


def test_persist_error_review(temp_db: Path):
    """Failed reviews (parse errors, API failures) should still persist."""
    from doc_assistant.provenance import record_answer

    answer_id = record_answer(query="q", answer="a", retrieved_chunks=[])
    persist_review(
        answer_id,
        ReviewResult(error="API timeout"),
        reviewer_kind="llm_haiku",
    )
    reviews = get_reviews(answer_id)
    assert len(reviews) == 1
    assert reviews[0].error == "API timeout"
    assert reviews[0].faithfulness is None


def test_get_reviews_empty_for_unknown_answer(temp_db: Path):
    assert get_reviews("does-not-exist") == []
