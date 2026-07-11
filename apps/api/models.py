"""Pydantic request/response schemas for the desktop API (PR-M2).

Mirror the PR-M0/M1 ``ChatController`` value objects so the frontend renders native JSON
(the pre-rendered markdown blocks ride along as strings — a convenience/fallback, not the
only representation). The ``from_*`` constructors convert the dataclasses → payloads with
the one coercion the dataclasses need: ``Path`` → ``str`` for ``download_path``.

The dataclass types are imported under ``TYPE_CHECKING`` only, so importing this module
does not pull the heavy ``chat_controller`` → ``pipeline`` → torch chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from doc_assistant.config import CANDIDATE_K

if TYPE_CHECKING:
    from doc_assistant.chat_controller import ClaimView, SourceView, TurnResult, UsageView


# ============================================================
# Requests
# ============================================================


class RagOverrides(BaseModel):
    """Wire model for a session-scoped, non-persistent RAG-sandbox override (ADR-010).
    ``None`` (a field or the whole object) = use the locked default. ``top_k`` is bounded to
    ``[1, CANDIDATE_K]`` — the candidate pool is fixed at pipeline construction, so a top_k
    above it is meaningless; out-of-range is a 422, never a silent clamp.

    ``epistemics_markers_enabled``/``reviewer_evidence_chars`` (U1b, SPRINT-011, ADR-010's
    2026-07-10 amendment) are the two "must revisit" niche knobs. ``reviewer_evidence_chars``
    is bounded ``[200, 6000]``: the floor sits above the ~300-char display excerpt that was
    empirically shown to starve the reviewer into false "unsupported claim" verdicts
    (`config.py`'s own comment on `REVIEWER_EVIDENCE_CHARS`); the ceiling is a generous 4x the
    1500-char default, bounding judge-token cost without being restrictive for experimentation.
    """

    top_k: int | None = Field(default=None, ge=1, le=CANDIDATE_K)
    synthesis_mode: Literal["ai", "human"] | None = None
    use_multi_query: bool | None = None
    epistemics_markers_enabled: bool | None = None
    reviewer_evidence_chars: int | None = Field(default=None, ge=200, le=6000)


class ChatRequest(BaseModel):
    text: str
    session_id: str
    overrides: RagOverrides | None = None


class AdjudicateRequest(BaseModel):
    decision: Literal["accepted", "rejected", "edited"]
    edited_text: str | None = None


class ExportRequest(BaseModel):
    session_id: str
    dev: bool = False


class SettingsUpdate(BaseModel):
    """User-settable settings: the source documents folder, and (ADR-011, U1c) the LLM
    provider/model to switch to. At least one of ``source_dir`` or the ``llm_provider`` +
    ``llm_model`` pair must be present; the two ``llm_*`` fields travel together or not at all —
    both requests reject an otherwise-empty or half-shaped body with a 422, not a silent no-op."""

    source_dir: str | None = None
    llm_provider: Literal["anthropic", "ollama"] | None = None
    llm_model: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> SettingsUpdate:
        if self.source_dir is None and self.llm_provider is None and self.llm_model is None:
            raise ValueError("at least one of source_dir or llm_provider+llm_model is required")
        if (self.llm_provider is None) != (self.llm_model is None):
            raise ValueError("llm_provider and llm_model must be provided together")
        return self


# ============================================================
# Response payloads (mirror the controller value objects)
# ============================================================


class SourceViewPayload(BaseModel):
    n: int
    citation: str
    excerpt: str
    # The figure *id* (not the server path — no filesystem path crosses the boundary, M2
    # ADR-1); the frontend renders it via GET /api/figures/{figure_id}.
    figure_id: str | None
    chunk_key: str | None
    markers: list[str]

    @classmethod
    def from_view(cls, sv: SourceView) -> SourceViewPayload:
        return cls(
            n=sv.n,
            citation=sv.citation,
            excerpt=sv.excerpt,
            figure_id=sv.figure_id,
            chunk_key=sv.chunk_key,
            markers=list(sv.markers),
        )


class ClaimViewPayload(BaseModel):
    claim_id: str
    n: int
    text: str
    badge: str

    @classmethod
    def from_view(cls, cv: ClaimView) -> ClaimViewPayload:
        return cls(claim_id=cv.claim_id, n=cv.n, text=cv.text, badge=cv.badge)


class UsageViewPayload(BaseModel):
    turn_input: int
    turn_output: int
    session_total: int
    cost_usd: float | None
    is_local: bool

    @classmethod
    def from_view(cls, u: UsageView) -> UsageViewPayload:
        return cls(
            turn_input=u.turn_input,
            turn_output=u.turn_output,
            session_total=u.session_total,
            cost_usd=u.cost_usd,
            is_local=u.is_local,
        )


class TurnResultPayload(BaseModel):
    answer: str
    mode: Literal["ai", "human"]
    sources: list[SourceViewPayload]
    flagged_claims: list[ClaimViewPayload]
    usage: UsageViewPayload
    standalone_query: str
    record_id: str | None
    provenance_card_md: str
    claim_review_md: str
    sources_md: str
    usage_md: str
    citation_note_md: str
    download_path: str | None

    @classmethod
    def from_turn_result(cls, r: TurnResult) -> TurnResultPayload:
        return cls(
            answer=r.answer,
            mode=r.mode,
            sources=[SourceViewPayload.from_view(s) for s in r.sources],
            flagged_claims=[ClaimViewPayload.from_view(c) for c in r.flagged_claims],
            usage=UsageViewPayload.from_view(r.usage),
            standalone_query=r.standalone_query,
            record_id=r.record_id,
            provenance_card_md=r.provenance_card_md,
            claim_review_md=r.claim_review_md,
            sources_md=r.sources_md,
            usage_md=r.usage_md,
            citation_note_md=r.citation_note_md,
            download_path=str(r.download_path) if r.download_path is not None else None,
        )
