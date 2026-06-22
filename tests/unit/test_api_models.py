"""Unit tests for the desktop-API pydantic schemas (PR-M2).

The ``TurnResult`` round-trips through ``TurnResultPayload`` with no field loss, and the
request models' ``Literal`` validation rejects bad input. No FastAPI app, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.api.models import AdjudicateRequest, TurnResultPayload
from pydantic import ValidationError

from doc_assistant.chat_controller import ClaimView, SourceView, TurnResult, UsageView


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
