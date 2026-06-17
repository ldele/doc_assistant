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
    FAILURE_TAGS,
    ReviewResult,
    _coerce_failure_tag,
    _format_evidence,
    build_reviewer_prompt,
    get_reviews,
    persist_review,
    review_answer,
    verdict_from_review,
)


def _mock_client(response_text: str) -> MagicMock:
    """An ``LLMClient``-shaped mock: ``.complete(...)`` returns the text."""
    client = MagicMock()
    client.complete.return_value = response_text
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


def test_reviewer_prompt_offers_the_failure_tag_enum():
    assert "failure_tag" in _REVIEWER_PROMPT
    # every non-neutral tag is offered to the model
    for tag in FAILURE_TAGS:
        assert tag in _REVIEWER_PROMPT


# ============================================================
# failure_tag (Chunk 2c)
# ============================================================


def test_coerce_failure_tag_accepts_known_and_normalises():
    assert _coerce_failure_tag("overclaim") == "overclaim"
    assert _coerce_failure_tag("  No_Hedge ") == "no_hedge"


def test_coerce_failure_tag_falls_back_to_none():
    assert _coerce_failure_tag("made_up_tag") == "none"
    assert _coerce_failure_tag(None) == "none"
    assert _coerce_failure_tag(42) == "none"


def test_contested_evidence_tag_is_available_and_appended():
    # Feature 7d added contested_evidence — present, accepted, and appended last so it
    # never re-buckets historical aggregates.
    assert "contested_evidence" in FAILURE_TAGS
    assert FAILURE_TAGS[-1] == "contested_evidence"
    assert _coerce_failure_tag("Contested_Evidence") == "contested_evidence"


# ============================================================
# verdict_from_review (self-eval roll-up)
# ============================================================


def test_verdict_pass_on_clean_high_faithfulness():
    label, _ = verdict_from_review(
        ReviewResult(faithfulness=5, citation_density=4, hedging_adequacy=4, failure_tag="none")
    )
    assert label == "pass"


def test_verdict_fail_on_reviewer_error():
    label, reason = verdict_from_review(ReviewResult(error="boom"))
    assert label == "fail" and "failed" in reason


def test_verdict_fail_on_low_faithfulness():
    assert verdict_from_review(ReviewResult(faithfulness=2, failure_tag="none"))[0] == "fail"


def test_verdict_fail_on_hard_failure_tag():
    # A hard tag fails even if faithfulness reads okay (the answer contradicts evidence).
    label, _ = verdict_from_review(
        ReviewResult(faithfulness=4, failure_tag="evidence_contradiction")
    )
    assert label == "fail"


def test_verdict_concern_on_soft_tag_and_on_mid_faithfulness():
    assert (
        verdict_from_review(ReviewResult(faithfulness=5, failure_tag="no_hedge"))[0] == "concern"
    )
    assert verdict_from_review(ReviewResult(faithfulness=3, failure_tag="none"))[0] == "concern"


def test_verdict_unsupported_claim_with_high_faithfulness_is_concern_not_fail():
    # 2026-06-17 recalibration: a single unsupported-claim tag must NOT hard-fail an
    # otherwise well-grounded answer (faithfulness is the primary signal).
    label, _ = verdict_from_review(
        ReviewResult(faithfulness=4, citation_density=4, failure_tag="unsupported_claim")
    )
    assert label == "concern"
    # ...but evidence_contradiction (actively wrong) still fails regardless of score.
    assert (
        verdict_from_review(ReviewResult(faithfulness=4, failure_tag="evidence_contradiction"))[0]
        == "fail"
    )
    # ...and a low faithfulness still fails whatever the tag.
    assert (
        verdict_from_review(ReviewResult(faithfulness=2, failure_tag="unsupported_claim"))[0]
        == "fail"
    )


def test_review_answer_parses_failure_tag():
    client = _mock_client(
        '{"faithfulness": 2, "citation_density": 2, "hedging_adequacy": 2, '
        '"unsupported_claims_count": 3, "failure_tag": "overclaim", "notes": "weak"}'
    )
    result = review_answer(_prov(), client)
    assert result.error is None
    assert result.failure_tag == "overclaim"


def test_review_answer_defaults_missing_failure_tag_to_none():
    # failure_tag omitted but all required fields present → parses, tag = "none".
    client = _mock_client(
        '{"faithfulness": 5, "citation_density": 5, "hedging_adequacy": 5, '
        '"unsupported_claims_count": 0, "notes": "clean"}'
    )
    result = review_answer(_prov(), client)
    assert result.error is None
    assert result.failure_tag == "none"


def test_review_answer_coerces_unknown_failure_tag():
    client = _mock_client(
        '{"faithfulness": 3, "citation_density": 3, "hedging_adequacy": 3, '
        '"unsupported_claims_count": 0, "failure_tag": "vibes", "notes": "x"}'
    )
    result = review_answer(_prov(), client)
    assert result.failure_tag == "none"


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


def test_format_evidence_prefers_full_text_over_display_excerpt():
    # The judge sees the wider grounding text, not the 300-char display excerpt
    # (the fix for the evidence-starved-judge finding, 2026-06-17).
    chunks = [RetrievedChunk(filename="a.pdf", chunk_excerpt="short", full_text="WIDE grounding")]
    out = _format_evidence(chunks)
    assert "WIDE grounding" in out and "short" not in out


# ============================================================
# review_answer — parsing
# ============================================================


def test_review_answer_parses_clean_json():
    client = _mock_client(
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
    client = _mock_client(
        '```json\n{"faithfulness": 3, "citation_density": 3, '
        '"hedging_adequacy": 3, "unsupported_claims_count": 0, "notes": "ok"}\n```'
    )
    result = review_answer(_prov(), client)
    assert result.error is None
    assert result.faithfulness == 3


def test_review_answer_handles_broken_json():
    client = _mock_client("not json at all")
    result = review_answer(_prov(), client)
    assert result.error is not None
    assert "reviewer call failed" in result.error
    # The raw output is captured so an opaque local-model failure is debuggable.
    assert result.raw_response == "not json at all"


# ============================================================
# review_answer — transport retry (2026-06-17)
# ============================================================

_GOOD_JSON = (
    '{"faithfulness": 4, "citation_density": 3, "hedging_adequacy": 5, '
    '"unsupported_claims_count": 0, "failure_tag": "none", "notes": "ok"}'
)


def test_review_answer_retries_transient_transport_error_then_recovers():
    client = MagicMock()
    client.complete.side_effect = [ConnectionError("blip"), _GOOD_JSON]
    result = review_answer(_prov(), client, attempts=3)
    assert result.error is None and result.faithfulness == 4
    assert client.complete.call_count == 2  # failed once, succeeded on retry


def test_review_answer_exhausts_retries_then_fails():
    client = MagicMock()
    client.complete.side_effect = ConnectionError("down")
    result = review_answer(_prov(), client, attempts=2)
    assert result.error is not None and "reviewer call failed" in result.error
    assert client.complete.call_count == 2  # tried exactly `attempts` times


def test_review_answer_does_not_retry_parse_failure():
    # A non-JSON completion is deterministic at temp 0 → retrying wastes calls.
    client = _mock_client("not json at all")
    result = review_answer(_prov(), client, attempts=3)
    assert result.error is not None
    assert client.complete.call_count == 1  # parse failure is NOT retried


def test_review_answer_extracts_json_from_surrounding_prose():
    """Local models often wrap the object in prose; extract the brace span."""
    client = _mock_client(
        "Here is my assessment:\n"
        '{"faithfulness": 2, "citation_density": 2, "hedging_adequacy": 2, '
        '"unsupported_claims_count": 3, "notes": "weak"}\n'
        "Hope that helps!"
    )
    result = review_answer(_prov(), client)
    assert result.error is None
    assert result.faithfulness == 2
    assert result.unsupported_claims_count == 3


def test_review_answer_handles_missing_field():
    client = _mock_client('{"faithfulness": 4}')
    result = review_answer(_prov(), client)
    assert result.error is not None
    assert "bad reviewer response" in result.error


def test_review_answer_is_isolated():
    """Single user message, no system, temperature=0 — same contract as the eval judge."""
    client = _mock_client(
        '{"faithfulness": 4, "citation_density": 4, "hedging_adequacy": 4, '
        '"unsupported_claims_count": 0, "notes": ""}'
    )
    review_answer(_prov(), client)
    call = client.complete.call_args
    messages = call.args[0]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert not any(m["role"] == "system" for m in messages)
    assert call.kwargs.get("temperature") == 0.0


def test_build_reviewer_prompt_is_evidence_only():
    """The reviewer prompt carries the evidence + answer, no ground-truth reference."""
    prov = AnswerProvenance(
        id="x",
        query="Q",
        answer="ANSWER_TOKEN",
        retrieved_chunks=[RetrievedChunk(filename="p.pdf", chunk_excerpt="EVIDENCE_TOKEN")],
    )
    prompt = build_reviewer_prompt(prov)
    assert "EVIDENCE_TOKEN" in prompt
    assert "ANSWER_TOKEN" in prompt
    assert "Q" in prompt


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


def test_failure_tag_round_trips(temp_db: Path):
    from doc_assistant.provenance import record_answer

    answer_id = record_answer(query="q", answer="a", retrieved_chunks=[])
    persist_review(
        answer_id,
        ReviewResult(
            faithfulness=2,
            citation_density=2,
            hedging_adequacy=3,
            unsupported_claims_count=2,
            failure_tag="missing_citation",
            notes="n",
        ),
        reviewer_kind="llm_haiku",
    )
    assert get_reviews(answer_id)[0].failure_tag == "missing_citation"
