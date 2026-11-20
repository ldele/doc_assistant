"""Chat-turn wire models — the answer path (PR-M2).

The request side (``ChatRequest`` + the ADR-010 ``RagOverrides`` governance channel, plus the
adjudicate/export bodies) and the whole ``TurnResultPayload`` tree it answers with: sources,
their always-on epistemic evaluation (ADR-027 D3), flagged claims, usage and the retrieval
scope (ADR-025 F2).

``RagOverrides`` and ``ScopePayload`` are the two types other domains reach for — ``compare``
imports the first, ``conversations`` the second — so this module owns them and imports nothing
from its siblings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from doc_assistant.config import CANDIDATE_K

if TYPE_CHECKING:
    from doc_assistant.chat_controller import (
        ClaimView,
        ScopeView,
        SourceEpistemics,
        SourceEvalSummary,
        SourceView,
        TurnResult,
        UsageView,
    )
    from doc_assistant.turn_details import RecordsView, ReviewView, TurnClaim


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
    # ADR-025 F2 — restrict retrieval to these folders for this turn, searched as one set (A7b).
    # A sibling of `overrides`, not a field inside it: a scope is a *content* filter (which
    # documents), while `RagOverrides` is ADR-010's governance channel for locked *quality* knobs.
    # Only ids cross the wire — the backend resolves membership per turn, so a Library edit can
    # never be out of date by the time the answer is produced. null or [] = the whole library.
    scope_folder_ids: list[str] | None = None
    # A8a: the start screen's mode chip. Read by a chat's first turn only, which stores it with
    # the folders as the chat's setup; every later turn ignores both (`chat_setup`).
    mode_id: str | None = None


class ModeOfferPayload(BaseModel):
    """`/mode <name>`'s offer (A8a, G2): ``continue`` opens *Continue in a new chat*
    prefilled with ``mode_id``; ``set`` sets the start screen's mode chip."""

    action: Literal["continue", "set"]
    mode_id: str


class AdjudicateRequest(BaseModel):
    decision: Literal["accepted", "rejected", "edited"]
    edited_text: str | None = None


class ExportRequest(BaseModel):
    session_id: str
    dev: bool = False


# ============================================================
# Response payloads (mirror the controller value objects)
# ============================================================


class SourceEpistemicsPayload(BaseModel):
    """ADR-027 D3 — one source's always-on epistemic assessment (mirrors `SourceEpistemics`)."""

    coverage: str | None  # corroborated | unique | contested | null (not assessed)
    superseded: bool
    n_claims: int
    year: int | None

    @classmethod
    def from_view(cls, ev: SourceEpistemics) -> SourceEpistemicsPayload:
        return cls(
            coverage=ev.coverage, superseded=ev.superseded, n_claims=ev.n_claims, year=ev.year
        )


class SourceViewPayload(BaseModel):
    n: int
    citation: str
    excerpt: str
    # The figure *id* (not the server path — no filesystem path crosses the boundary, M2
    # ADR-1); the frontend renders it via GET /api/figures/{figure_id}.
    figure_id: str | None
    chunk_key: str | None
    markers: list[str]
    # ADR-027 D3 — always-on per-source evaluation + the rerank score (strip signals).
    reranker_score: float = 0.0
    evaluation: SourceEpistemicsPayload | None = None
    # A2: the kinds of text addressed to an AI the passage carries; empty when none matched.
    addressed_to_ai: list[str] = []

    @classmethod
    def from_view(cls, sv: SourceView) -> SourceViewPayload:
        return cls(
            n=sv.n,
            citation=sv.citation,
            excerpt=sv.excerpt,
            figure_id=sv.figure_id,
            chunk_key=sv.chunk_key,
            markers=list(sv.markers),
            reranker_score=sv.reranker_score,
            addressed_to_ai=list(sv.addressed_to_ai),
            evaluation=(
                SourceEpistemicsPayload.from_view(sv.evaluation)
                if sv.evaluation is not None
                else None
            ),
        )


class SourceEvalSummaryPayload(BaseModel):
    """ADR-027 D3 — strip-level freshness (mirrors `SourceEvalSummary`)."""

    graph_version: str | None
    stale: bool

    @classmethod
    def from_view(cls, s: SourceEvalSummary) -> SourceEvalSummaryPayload:
        return cls(graph_version=s.graph_version, stale=s.stale)


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
    billing: str | None = None

    @classmethod
    def from_view(cls, u: UsageView) -> UsageViewPayload:
        return cls(
            turn_input=u.turn_input,
            turn_output=u.turn_output,
            session_total=u.session_total,
            cost_usd=u.cost_usd,
            is_local=u.is_local,
            billing=u.billing,
        )


class ScopeFolderPayload(BaseModel):
    """One folder a turn asked for. `folder_name` is null when the folder was unknown or deleted;
    `doc_count` is its own count before the union, 0 when it added nothing."""

    folder_id: str
    folder_name: str | None
    doc_count: int


class ScopeConceptPayload(BaseModel):
    """The concept a turn was scoped to (A7b): its documents are where it is present."""

    concept_id: str
    name: str
    graph_stale: bool


class ScopePayload(BaseModel):
    """The retrieval scope a turn ran under (ADR-025 F2, A7b); absent = the whole library.
    Folders or a concept, never both; `doc_count` is the union retrieval searched."""

    folders: list[ScopeFolderPayload] = []
    concept: ScopeConceptPayload | None = None
    doc_count: int

    @classmethod
    def from_view(cls, v: ScopeView) -> ScopePayload:
        return cls(
            folders=[
                ScopeFolderPayload(
                    folder_id=f.folder_id, folder_name=f.folder_name, doc_count=f.doc_count
                )
                for f in v.folders
            ],
            concept=(
                ScopeConceptPayload(
                    concept_id=v.concept.concept_id,
                    name=v.concept.name,
                    graph_stale=v.concept.graph_stale,
                )
                if v.concept is not None
                else None
            ),
            doc_count=v.doc_count,
        )


class TurnClaimPayload(BaseModel):
    """One claim of an answer (A8b): its text, the passages it cites, its retrieval marker
    (`ok` · `weak` · `unsupported`, G4) and the decision recorded on it."""

    claim_id: str
    n: int
    text: str
    marker: str
    sources: list[int] = []
    decision: str = "pending"
    edited_text: str | None = None

    @classmethod
    def from_claim(cls, c: TurnClaim) -> TurnClaimPayload:
        return cls(
            claim_id=c.claim_id,
            n=c.n,
            text=c.text,
            marker=c.marker,
            sources=list(c.sources),
            decision=c.decision,
            edited_text=c.edited_text,
        )


class RecordItemPayload(BaseModel):
    kind: Literal["document", "folder", "concept"]
    id: str
    label: str


class RecordsPayload(BaseModel):
    """A rule-answered turn's lookup (A6, A8b's *What it looked up*)."""

    lookup: str
    args: dict[str, Any] = {}
    rows: int = 0
    items: list[RecordItemPayload] = []
    more: int = 0
    via: str = "rule"
    asked: bool = False

    @classmethod
    def from_view(cls, v: RecordsView) -> RecordsPayload:
        return cls(
            lookup=v.lookup,
            args=dict(v.args),
            rows=v.rows,
            items=[RecordItemPayload(kind=i.kind, id=i.id, label=i.label) for i in v.items],
            more=v.more,
            via=v.via,
            asked=v.asked,
        )


class ReviewPayload(BaseModel):
    """The reviewer's verdict on an answer, as recorded (A8b, a reopened chat)."""

    reviewer_kind: str
    faithfulness: int | None = None
    citation_density: int | None = None
    hedging_adequacy: int | None = None
    unsupported_claims: int | None = None
    failure_tag: str | None = None
    error: str | None = None

    @classmethod
    def from_view(cls, v: ReviewView) -> ReviewPayload:
        return cls(
            reviewer_kind=v.reviewer_kind,
            faithfulness=v.faithfulness,
            citation_density=v.citation_density,
            hedging_adequacy=v.hedging_adequacy,
            unsupported_claims=v.unsupported_claims,
            failure_tag=v.failure_tag,
            error=v.error,
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
    scope: ScopePayload | None = None
    # ADR-027 D3 — strip-level freshness for the always-on source-evaluation strip (per-source
    # evaluation rides on each source). null = no epistemics sidecar / 0-doc → no strip.
    source_eval: SourceEvalSummaryPayload | None = None
    # A2: the web addresses in the passages; an answer's link outside this list renders as text.
    source_links: list[str] = []
    # A8a (G2): set only on a `/mode <name>` reply.
    mode_offer: ModeOfferPayload | None = None
    # A8b: the answer's claims (G4), a records answer's lookup, and what the answer carried
    # (`context_json` as recorded), its mode and prompt version.
    claims: list[TurnClaimPayload] = []
    records: RecordsPayload | None = None
    context: dict[str, Any] | None = None
    chat_mode: str | None = None
    prompt_version: str | None = None

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
            source_links=list(r.source_links),
            download_path=str(r.download_path) if r.download_path is not None else None,
            scope=ScopePayload.from_view(r.scope) if r.scope is not None else None,
            source_eval=(
                SourceEvalSummaryPayload.from_view(r.source_eval)
                if r.source_eval is not None
                else None
            ),
            mode_offer=(
                ModeOfferPayload(action=r.mode_offer[0], mode_id=r.mode_offer[1])
                if r.mode_offer is not None
                else None
            ),
            claims=[TurnClaimPayload.from_claim(c) for c in r.claims],
            records=RecordsPayload.from_view(r.records) if r.records is not None else None,
            context=r.context,
            chat_mode=r.chat_mode,
            prompt_version=r.prompt_version,
        )
