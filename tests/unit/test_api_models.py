"""Unit tests for the desktop-API pydantic schemas (PR-M2).

The ``TurnResult`` round-trips through ``TurnResultPayload`` with no field loss, and the
request models' ``Literal`` validation rejects bad input. No FastAPI app, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.api.models import AdjudicateRequest, ChatRequest, RagOverrides, TurnResultPayload
from pydantic import ValidationError

from doc_assistant.chat_controller import ClaimView, SourceView, TurnResult, UsageView
from doc_assistant.config import CANDIDATE_K


def _turn_result() -> TurnResult:
    return TurnResult(
        answer="Neurons meet at synapses [1].",
        mode="ai",
        sources=[
            SourceView(
                n=1,
                citation="[1] a.pdf · p.1",
                excerpt="…passage…",
                figure_path=None,
                chunk_key="d1:0",
                markers=["contested"],
            )
        ],
        flagged_claims=[ClaimView(claim_id="c1", n=1, text="a claim", badge="unsupported")],
        usage=UsageView(
            turn_input=10, turn_output=20, session_total=30, cost_usd=0.001, is_local=False
        ),
        standalone_query="how do neurons connect",
        record_id="rec-123",
        provenance_card_md="P",
        claim_review_md="C",
        sources_md="S",
        usage_md="U",
        citation_note_md="",
        download_path=Path("/tmp/session-transcript.md"),
    )


def test_turn_result_payload_round_trips_without_field_loss():
    payload = TurnResultPayload.from_turn_result(_turn_result())

    assert payload.answer == "Neurons meet at synapses [1]."
    assert payload.mode == "ai"
    assert payload.standalone_query == "how do neurons connect"
    assert payload.record_id == "rec-123"
    # nested structured fields survive
    assert payload.sources[0].chunk_key == "d1:0"
    assert payload.sources[0].markers == ["contested"]
    assert payload.flagged_claims[0].claim_id == "c1"
    assert payload.usage.cost_usd == 0.001 and payload.usage.is_local is False
    # the one coercion: Path → str
    assert payload.download_path == str(Path("/tmp/session-transcript.md"))
    # markdown blocks pass through verbatim
    assert (payload.provenance_card_md, payload.sources_md, payload.citation_note_md) == (
        "P",
        "S",
        "",
    )

    # full JSON serialise → parse is identity
    assert TurnResultPayload.model_validate_json(payload.model_dump_json()) == payload


def test_human_mode_payload():
    r = _turn_result()
    r.mode = "human"
    r.flagged_claims = []
    payload = TurnResultPayload.from_turn_result(r)
    assert payload.mode == "human" and payload.flagged_claims == []


def test_adjudicate_request_validates_decision():
    assert AdjudicateRequest(decision="accepted").edited_text is None
    assert AdjudicateRequest(decision="edited", edited_text="fixed").edited_text == "fixed"
    with pytest.raises(ValidationError):
        AdjudicateRequest(decision="bogus")


# ============================================================
# ADR-010 / SPRINT-010 (U1) — RagOverrides wire model
# ============================================================


def test_chat_request_overrides_absent_is_none():
    # Backward compat: no `overrides` in the body → the field defaults to None.
    req = ChatRequest(text="hi", session_id="s1")
    assert req.overrides is None


def test_chat_request_top_k_out_of_range_422():
    with pytest.raises(ValidationError):
        ChatRequest(text="hi", session_id="s1", overrides=RagOverrides(top_k=0))
    with pytest.raises(ValidationError):
        ChatRequest(text="hi", session_id="s1", overrides=RagOverrides(top_k=CANDIDATE_K + 1))
    # In-range is accepted.
    req = ChatRequest(text="hi", session_id="s1", overrides=RagOverrides(top_k=CANDIDATE_K))
    assert req.overrides is not None and req.overrides.top_k == CANDIDATE_K


def test_chat_request_synthesis_mode_rejects_bad_literal():
    with pytest.raises(ValidationError):
        RagOverrides(synthesis_mode="bogus")
    assert RagOverrides(synthesis_mode="human").synthesis_mode == "human"


# ============================================================
# U1b / SPRINT-011 — the two niche knobs
# ============================================================


def test_epistemics_markers_enabled_override_accepts_bool():
    assert RagOverrides(epistemics_markers_enabled=True).epistemics_markers_enabled is True
    assert RagOverrides(epistemics_markers_enabled=False).epistemics_markers_enabled is False
    assert RagOverrides().epistemics_markers_enabled is None


@pytest.mark.parametrize("bad", [199, 6001, 0, -1])
def test_reviewer_evidence_chars_out_of_range_422(bad: int):
    with pytest.raises(ValidationError):
        RagOverrides(reviewer_evidence_chars=bad)


@pytest.mark.parametrize("ok", [200, 1500, 6000])
def test_reviewer_evidence_chars_in_range_accepted(ok: int):
    assert RagOverrides(reviewer_evidence_chars=ok).reviewer_evidence_chars == ok
