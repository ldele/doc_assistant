"""UI-agnostic turn orchestration (PR-M0 — Tauri desktop-shell migration).

The whole RAG/integrity turn — slash-command dispatch, pending claim-edit handling,
library-query routing, history-aware rewrite, retrieval, figure lookup, source
assembly, ``SYNTHESIS_MODE=human`` branch, answer streaming, provenance capture,
confidence-signal gating + (flagged-only) reviewer call, claim segmentation + eager
persistence, citation audit, usage accounting, and export stashing — used to live
inside the original web-UI ``on_message`` handler, interleaved with UI rendering.

This module lifts that orchestration out of the UI so any frontend renders the same
value object. ``ChatController.handle_message`` yields a stream of :class:`TurnEvent`
(streamed ``Token``s + ``Step`` status updates) terminating in a :class:`Result`
wrapping a :class:`TurnResult` — everything a renderer needs, as data. The desktop
and CLI apps (and, in PR-M2, FastAPI) become thin renderers over this stream.

**No UI-framework import here.** No generation logic moves: ``pipeline.stream_answer``
etc. are called exactly as before. This is a *move*, not a redesign — behaviour is
frozen and guarded by ``tests/integration/test_turn_parity.py``.

See ``docs/archive/pr-m0-chat-controller.md`` and
``docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md``.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import structlog
from langchain_core.documents import Document

from doc_assistant import app_settings, compare, conversations, export
from doc_assistant.commands import execute_command, parse_command
from doc_assistant.config import (
    EPISTEMICS_MARKERS_ENABLED,
    REVIEWER_EVIDENCE_CHARS,
    SYNTHESIS_MODE,
    TOP_K,
    USE_MULTI_QUERY,
    USE_PARENT_CHILD,
)
from doc_assistant.embeddings import get_active_model_name
from doc_assistant.ingest.figures import load_figure_image_paths
from doc_assistant.knowledge.epistemics import (
    MARKER_CONTESTED,
    MARKER_SUPERSEDED,
    ChunkEval,
    current_graph_version,
    derive_markers,
    load_source_evaluations,
)
from doc_assistant.library import document_years, folder_doc_hashes, get_folder
from doc_assistant.pipeline import RAGPipeline, format_citation
from doc_assistant.prompts import ANSWER_PROMPT
from doc_assistant.provenance import (
    AnswerProvenance,
    ConfidenceSignals,
    RetrievedChunk,
    adjudicate_claim,
    compute_confidence_signals,
    prompt_version_hash,
    record_answer,
    record_claims,
    template_hash,
)
from doc_assistant.query_router import answer_library_query, is_library_query
from doc_assistant.reviewer import ReviewResult, persist_review, review_answer
from doc_assistant.synthesis import (
    MARKER_OK,
    Claim,
    audit_citations,
    render_evidence_markdown,
    segment_claims,
)
from doc_assistant.tracking import TokenCounter

log = structlog.get_logger(__name__)

# ============================================================
# Session state (ADR-3) — caller-owned, injected into every call
# ============================================================


@dataclass
class Session:
    """Per-conversation state. Caller-owned; injected into every ``ChatController``
    call. The controller holds no per-turn state in globals — it is stateless across
    turns except for this injected object (multi-session later is non-breaking)."""

    history: list[dict[str, str]] = field(default_factory=list)
    counter: TokenCounter = field(default_factory=TokenCounter)
    export_turns: list[export.ExportTurn] = field(default_factory=list)
    awaiting_edit: dict[str, Any] | None = None
    session_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))


@dataclass(frozen=True)
class RagOverrides:
    """Session-scoped, per-turn RAG knob overrides (ADR-010 / feature-rag-sandbox.md).
    ``None`` (a field or the whole object) = use the locked default. Non-persistent: never
    written to config/.env/app_settings, and never assigned to a module global — threaded
    as an explicit request-scoped parameter so concurrent turns on the shared
    ``ChatController`` singleton cannot leak overrides into each other.

    ``epistemics_markers_enabled``/``reviewer_evidence_chars`` (U1b, SPRINT-011, ADR-010's
    2026-07-10 amendment) are the two "must revisit" niche knobs — same non-persistent,
    request-scoped mechanics as the original three."""

    top_k: int | None = None
    synthesis_mode: str | None = None  # "ai" | "human"
    use_multi_query: bool | None = None
    epistemics_markers_enabled: bool | None = None
    reviewer_evidence_chars: int | None = None


@dataclass(frozen=True)
class _TurnKnobs:
    """The effective per-turn RAG knobs (ADR-010), resolved once from a ``RagOverrides`` plus the
    locked config defaults, together with the provenance ``overrides_note`` derived from them. A
    ``None`` field (or ``overrides=None``) = the locked default. See ``_resolve_turn_knobs``."""

    top_k: int
    synthesis_mode: str
    multi_query: bool
    markers_enabled: bool
    reviewer_evidence_chars: int
    overrides_note: str


# ============================================================
# View models (pure render payload — no UI framework types)
# ============================================================


@dataclass
class ScopeView:
    """The retrieval scope one turn ran under (ADR-025 F2) — render-ready.

    Present only on a scoped turn; ``None`` on ``TurnResult`` means the whole library. This is
    a **content filter** (which documents), not a quality knob, which is why it rides beside
    ``RagOverrides`` rather than inside it (docs/specs/feature-corpus-folders-scope.md, S1).

    ``folder_name is None`` means the folder was deleted between the user picking it and the
    turn running: ``doc_count`` is then 0 and the turn honestly retrieves nothing rather than
    quietly widening to every document (S3).
    """

    folder_id: str
    folder_name: str | None
    doc_count: int


@dataclass(frozen=True)
class SourceEpistemics:
    """One source's epistemic assessment for the always-on D3 strip (ADR-027). ``coverage`` is the
    most-cautionary claim class in the source's chunk (``corroborated``/``unique``/``contested``,
    or ``None`` = not assessed); ``superseded`` flags a superseded-trend claim; ``year`` is the
    doc's publication year. Always attached (D3 is not gated by the D2/E3 influence toggle)."""

    coverage: str | None
    superseded: bool
    n_claims: int
    year: int | None


@dataclass(frozen=True)
class SourceEvalSummary:
    """Strip-level freshness for the D3 source-evaluation surface (ADR-027). ``graph_version`` is
    the build stamp the epistemics sidecar was computed under; ``stale`` is True when the concept
    graph has since been rebuilt without re-running ``compute_epistemics`` (the strip says so, not
    hides it). ``None`` on ``TurnResult`` = no sidecar / 0-doc — the strip degrades to nothing."""

    graph_version: str | None
    stale: bool


@dataclass
class SourceView:
    """One retrieved source, render-ready (side panel / sources block)."""

    n: int
    citation: str  # format_citation(doc, n)
    excerpt: str  # ~800-char side-panel preview (with trailing "..." when truncated)
    figure_path: str | None  # resolved PNG path (local desktop render); never crosses the API
    chunk_key: str | None  # ADR-2; the 7d marker join key
    markers: list[str] = field(default_factory=list)  # PR-M1: contested / superseded_trend (D2)
    figure_id: str | None = None  # PR-M3: the id the web/API renders via GET /api/figures/{id}
    reranker_score: float = 0.0  # per-source rerank score (D3 strip signal, ADR-027)
    evaluation: SourceEpistemics | None = None  # always-on per-source assessment (D3)


@dataclass
class ClaimView:
    """A flagged claim needing adjudication (clean claims are not surfaced)."""

    claim_id: str
    n: int
    text: str
    badge: str  # "unsupported" | "weakly grounded"


@dataclass
class UsageView:
    turn_input: int
    turn_output: int
    session_total: int
    cost_usd: float | None  # None under local provider (no metered cost)
    is_local: bool


@dataclass
class TurnResult:
    """The full render payload for one turn. Renderers map fields to widgets only —
    no business logic. The markdown blocks are pre-rendered by the pure formatters."""

    answer: str  # the raw answer text (no appended blocks)
    mode: Literal["ai", "human"]
    sources: list[SourceView]
    flagged_claims: list[ClaimView]
    usage: UsageView
    standalone_query: str  # post-rewrite query actually searched
    record_id: str | None  # provenance id (for /review, /export-record)
    # Pre-rendered markdown blocks (built by the existing pure formatters):
    provenance_card_md: str
    claim_review_md: str
    sources_md: str
    usage_md: str
    citation_note_md: str  # "" when citations are clean
    # A written export file to offer for download (set by the /export slash command;
    # None otherwise). Lets the renderer attach a download widget without re-deriving
    # dispatch — preserves the original /export behaviour across the UI split.
    download_path: Path | None = None
    # ADR-025 F2 — the retrieval scope this turn ran under; None = the whole library. The
    # renderer MUST surface this whenever it is set: an answer drawn from a subset of the
    # corpus that doesn't say so is the failure this feature was built to prevent.
    scope: ScopeView | None = None
    # ADR-027 D3 — strip-level freshness for the always-on source-evaluation surface (per-source
    # assessment rides on each SourceView.evaluation). None = no sidecar / 0-doc → no strip.
    source_eval: SourceEvalSummary | None = None


# ============================================================
# TurnEvent — a tagged union streamed by handle_message
# ============================================================


@dataclass
class Token:
    """One streamed answer-token delta."""

    text: str


@dataclass
class Step:
    """A progress status update (retrieval / rewrite). Advisory; renderers may show it."""

    name: str
    status: str


@dataclass
class Result:
    """The terminal event: the finished TurnResult."""

    result: TurnResult


TurnEvent = Token | Step | Result


# ============================================================
# Helpers (pure formatters, moved out of the original UI handler)
# ============================================================


def _is_local(provider: str) -> bool:
    """Whether ``provider`` is the local/free Ollama backend.

    ADR-011 (U1c, desktop provider switch): takes the caller's **effective** provider
    (``self.rag.provider``) rather than reading the import-time ``LLM_PROVIDER`` constant, so this
    stays truthful after a live switch. No default — every call site must say which provider it
    means.
    """
    return provider.lower() == "ollama"


# PR-M1 — human labels for the 7d evidence-layer markers (advisory chip, not a gate).
_MARKER_LABELS = {
    MARKER_CONTESTED: "contested in corpus",
    MARKER_SUPERSEDED: "trend superseded",
}


def _marker_chip(markers: list[str]) -> str:
    """A quiet inline chip for a source's 7d markers (PR-M1). Returns "" when clean, so a
    turn with no markers renders **byte-identically** to before (eval-comparability)."""
    if not markers:
        return ""
    labels = [_MARKER_LABELS.get(m, m) for m in markers]
    return " — ⚠ " + " · ".join(labels)


def _sources_block(sources: list[SourceView]) -> str:
    """The visible "Sources:" list — each line is ``citation`` + (PR-M1 marker chip, if
    any). Byte-identical to the citation-only form when no source carries a marker."""
    lines = [sv.citation + _marker_chip(sv.markers) for sv in sources]
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def _overrides_note(
    eff_top_k: int,
    eff_synthesis_mode: str,
    eff_multi_query: bool,
    eff_markers_enabled: bool = EPISTEMICS_MARKERS_ENABLED,
    eff_reviewer_evidence_chars: int = REVIEWER_EVIDENCE_CHARS,
) -> str:
    """ADR-010 Decision 5: provenance shows the *effective* knob values and flags any that
    differ from the locked default. Returns "" (no-op, byte-identical turn) when every
    effective value equals its config default — i.e. ``overrides=None`` or an all-``None``
    ``RagOverrides``. The last two params are U1b's niche knobs (SPRINT-011)."""
    diffs = []
    if eff_top_k != TOP_K:
        diffs.append(f"top_k={eff_top_k} (default {TOP_K})")
    if eff_synthesis_mode != SYNTHESIS_MODE:
        diffs.append(f"synthesis_mode={eff_synthesis_mode} (default {SYNTHESIS_MODE})")
    if eff_multi_query != USE_MULTI_QUERY:
        diffs.append(f"multi_query={eff_multi_query} (default {USE_MULTI_QUERY})")
    if eff_markers_enabled != EPISTEMICS_MARKERS_ENABLED:
        diffs.append(
            f"epistemics_markers_enabled={eff_markers_enabled} "
            f"(default {EPISTEMICS_MARKERS_ENABLED})"
        )
    if eff_reviewer_evidence_chars != REVIEWER_EVIDENCE_CHARS:
        diffs.append(
            f"reviewer_evidence_chars={eff_reviewer_evidence_chars} "
            f"(default {REVIEWER_EVIDENCE_CHARS})"
        )
    if not diffs:
        return ""
    return "\n\n🧪 **Session override (this answer only):** " + " · ".join(diffs)


def _resolve_turn_knobs(overrides: RagOverrides | None) -> _TurnKnobs:
    """Resolve the effective per-turn knobs from ``overrides`` + the locked defaults (ADR-010).

    Request-scoped: reads ``overrides`` and the config constants, never a module global, so
    concurrent turns on the shared controller cannot leak. ``None`` = the locked default.
    ``multi_query`` here is the *effective* value carried into the provenance note only — the
    retrieval call passes the RAW ``overrides.use_multi_query`` (``None`` → the pipeline's own
    default), a deliberately distinct path this resolution does not touch."""
    top_k = overrides.top_k if overrides and overrides.top_k is not None else TOP_K
    synthesis_mode = (
        overrides.synthesis_mode if overrides and overrides.synthesis_mode else SYNTHESIS_MODE
    )
    multi_query = (
        USE_MULTI_QUERY
        if overrides is None or overrides.use_multi_query is None
        else overrides.use_multi_query
    )
    markers_enabled = (
        EPISTEMICS_MARKERS_ENABLED
        if overrides is None or overrides.epistemics_markers_enabled is None
        else overrides.epistemics_markers_enabled
    )
    reviewer_evidence_chars = (
        overrides.reviewer_evidence_chars
        if overrides and overrides.reviewer_evidence_chars is not None
        else REVIEWER_EVIDENCE_CHARS
    )
    return _TurnKnobs(
        top_k=top_k,
        synthesis_mode=synthesis_mode,
        multi_query=multi_query,
        markers_enabled=markers_enabled,
        reviewer_evidence_chars=reviewer_evidence_chars,
        overrides_note=_overrides_note(
            top_k, synthesis_mode, multi_query, markers_enabled, reviewer_evidence_chars
        ),
    )


def _resolve_scope(
    scope_folder_id: str | None,
) -> tuple[frozenset[str] | None, ScopeView | None]:
    """Resolve a folder id into ``(doc_hash scope, ScopeView)`` for one turn (ADR-025 F2).

    ``None`` in, ``(None, None)`` out — the unscoped path, byte-identical to pre-F2.

    A folder that is unknown, deleted, or empty resolves to an **empty** frozenset, never to
    ``None``: the caller must then retrieve nothing. Collapsing "I couldn't honour your scope"
    into "I searched everything" is precisely the silent-lie failure this feature exists to
    prevent, so the two cases are kept structurally distinct all the way down.

    A resolution failure (a broken DB read) is treated the same way — empty, not unscoped.
    """
    if scope_folder_id is None:
        return None, None
    try:
        hashes = folder_doc_hashes(scope_folder_id)
        folder = get_folder(scope_folder_id)
    except Exception as e:  # pragma: no cover - defensive; a scope must never widen on error
        log.warning("scope_resolve_failed", folder_id=scope_folder_id, error=str(e))
        return frozenset(), ScopeView(folder_id=scope_folder_id, folder_name=None, doc_count=0)
    if folder is None:
        log.warning("scope_folder_missing", folder_id=scope_folder_id)
    return (
        frozenset(hashes),
        ScopeView(
            folder_id=scope_folder_id,
            folder_name=folder.name if folder is not None else None,
            doc_count=len(hashes),
        ),
    )


def _scope_dict(scope: ScopeView | None) -> dict[str, Any] | None:
    """``ScopeView`` → the JSON shape persisted in ``answer_records.retrieval_scope_json``.
    ``None`` stays ``None`` so an unscoped turn writes NULL, exactly like every pre-F2 row."""
    if scope is None:
        return None
    return {
        "folder_id": scope.folder_id,
        "folder_name": scope.folder_name,
        "doc_count": scope.doc_count,
    }


def _scope_label(scope: ScopeView | None) -> str | None:
    """One-line scope label for surfaces that show a constraint rather than a full note
    (the A/B compare card). ``None`` on an unscoped run."""
    if scope is None:
        return None
    if scope.folder_name is None:
        return "a folder that no longer exists (0 documents)"
    return f"{scope.folder_name} ({scope.doc_count} document{'' if scope.doc_count == 1 else 's'})"


def _scope_note(scope: ScopeView | None) -> str:
    """Provenance-card line naming the scope. ``""`` on an unscoped turn, so the default turn
    stays byte-identical (the turn-parity test pins this)."""
    if scope is None:
        return ""
    where = f"**{scope.folder_name}**" if scope.folder_name else "a folder that no longer exists"
    return (
        f"\n\n🔎 **Retrieval scope (this answer only):** {where} — "
        f"{scope.doc_count} document{'' if scope.doc_count == 1 else 's'} searched, "
        "not the whole library."
    )


def _format_review_block(review: ReviewResult | None) -> str:
    """Render the reviewer's verdict as a sub-section of the provenance card."""
    if review is None:
        return ""
    if review.error:
        return f"\n\n**Reviewer:** _failed — {review.error}_"
    bits = [
        f"faithfulness `{review.faithfulness}/5`",
        f"citation density `{review.citation_density}/5`",
        f"hedging `{review.hedging_adequacy}/5`",
        f"unsupported claims: `{review.unsupported_claims_count}`",
    ]
    notes = f"  \n_Reviewer notes:_ {review.notes}" if review.notes else ""
    return "\n\n**Reviewer assessment:** " + " · ".join(bits) + notes


def _token_suffix(prov: AnswerProvenance, *, is_local: bool) -> str:
    """Header token tag — provider-aware. Local models report no usage, so a
    `0 tokens` figure would be misleading; show `local` instead."""
    if is_local:
        return " · local"
    total = (prov.token_input or 0) + (prov.token_output or 0)
    return f" · {total:,} tokens"


def _format_provenance_card(
    prov: AnswerProvenance,
    signals: ConfidenceSignals,
    *,
    review: ReviewResult | None = None,
    is_local: bool = False,
) -> str:
    """Render an AnswerProvenance as a plain-markdown card (no raw HTML).

    Clean answers get a compact three-line block; when a confidence signal
    fires the block expands with the signal breakdown, the reviewer verdict,
    and the full per-source reranker scores, led by a ⚠ chip. Filenames are
    not repeated — they live in the always-visible "Sources:" block; the card
    keys scores by source number. Full per-chunk metadata is in the DB /
    `/export-record`.
    """
    id8 = prov.id[:8]
    latency_s = (prov.latency_ms or 0.0) / 1000.0
    meta = (
        f"**Model** `{prov.model_name or '?'}` · "
        f"**Embedding** `{prov.embedding_model or '?'}` · "
        f"**top_k** {prov.top_k} · **parent-child** {prov.use_parent_child}"
    )
    hint = f"_Review:_ `/review {id8}` · _Export:_ `/export-record {id8}`"

    if not signals.any():
        top = (
            f" · **top reranker** `{signals.max_score:.3f}`"
            if signals.max_score is not None
            else ""
        )
        return (
            f"\n\n---\n"
            f"🔍 **Provenance** — `{id8}` · {latency_s:.1f}s"
            f"{_token_suffix(prov, is_local=is_local)}{top}  \n"
            f"{meta}  \n"
            f"{hint}"
        )

    sig_lines = (
        f"- max reranker score: `{signals.max_score:.3f}`"
        f"{' ⚠' if signals.weak_retrieval else ''}  \n"
        f"- top-3 score span: `{signals.top3_span:.3f}`"
        f"{' ⚠' if signals.score_cluster_concern else ''}  \n"
        f"- unique source documents: `{signals.unique_sources}`"
        f"{' ⚠' if signals.single_source_risk else ''}"
    )
    score_lines = "\n".join(
        f"- [{i + 1}] reranker `{c.reranker_score:.3f}`"
        if c.reranker_score is not None
        else f"- [{i + 1}] reranker `-`"
        for i, c in enumerate(prov.retrieved_chunks)
    )
    review_block = _format_review_block(review)
    return (
        f"\n\n---\n"
        f"⚠ **Low confidence: {', '.join(signals.reasons)}** — "
        f"`{id8}` · {latency_s:.1f}s{_token_suffix(prov, is_local=is_local)}  \n"
        f"{meta}  \n"
        f"**Prompt version** `{prov.prompt_version or '?'}`\n\n"
        f"**Confidence signals**  \n{sig_lines}"
        f"{review_block}\n\n"
        f"**Reranker scores** (by source number above)\n{score_lines}\n\n"
        f"{hint}"
    )


def _chunk_key(meta: dict[str, Any]) -> str | None:
    """Epistemics-format join key (ADR-2 / E1.1): ``{document_id}:{chunk_index}`` for a flat
    /baseline chunk, ``{document_id}:p{parent_index}`` for a PC parent (which carries
    ``parent_index``, not ``chunk_index``). ``None`` only when ``document_id`` is missing.

    Both keys are now first-class: ``build_epistemics`` projects the marker sidecar onto **both**
    segmentations (KI-8 re-projection), so ``load_epistemics_index`` resolves either directly —
    no more coarse PC-parent text containment.
    """
    document_id = meta.get("document_id")
    if document_id is None:
        return None
    chunk_index = meta.get("chunk_index")
    if chunk_index is not None:
        return f"{document_id}:{chunk_index}"
    parent_index = meta.get("parent_index")
    if parent_index is not None:
        return f"{document_id}:p{parent_index}"
    return None


def _build_retrieved_chunks(
    scored: list[tuple[Document, float]],
    *,
    reviewer_evidence_chars: int = REVIEWER_EVIDENCE_CHARS,
) -> list[RetrievedChunk]:
    """Build the provenance RetrievedChunk list from (doc, score) pairs.

    ``reviewer_evidence_chars`` (U1b / ADR-010 amendment) defaults to the locked config
    value — callers pass the per-turn effective value to override it, request-scoped."""
    chunks: list[RetrievedChunk] = []
    for doc, score in scored:
        meta = doc.metadata
        chunks.append(
            RetrievedChunk(
                filename=meta.get("filename"),
                doc_id=meta.get("document_id") or meta.get("doc_hash"),
                page=meta.get("page"),
                section=meta.get("section"),
                reranker_score=float(score),
                chunk_excerpt=doc.page_content[:300],
                # Wider grounding for the reviewer (not persisted/displayed).
                full_text=doc.page_content[:reviewer_evidence_chars],
                chunk_key=_chunk_key(meta),
            )
        )
    return chunks


def _build_source_views(
    scored: list[tuple[Document, float]], fig_paths: dict[str, str]
) -> list[SourceView]:
    """Build the render-ready source list (side-panel preview + figure path + key)."""
    views: list[SourceView] = []
    for i, (doc, score) in enumerate(scored):
        meta = doc.metadata
        preview = doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else "")
        figure_id = meta.get("figure_id") or None
        figure_path = fig_paths.get(figure_id) if figure_id else None
        views.append(
            SourceView(
                n=i + 1,
                citation=format_citation(doc, i + 1),
                excerpt=preview,
                figure_path=figure_path,
                chunk_key=_chunk_key(meta),
                figure_id=figure_id,
                reranker_score=round(float(score), 4),  # D3 strip signal (ADR-027)
            )
        )
    return views


def _build_claim_review(claims: list[Claim], claim_ids: list[str]) -> tuple[str, list[ClaimView]]:
    """Render the adjudication section + per-claim view-models for *flagged* claims only.

    Quiet on clean answers (UX: inform, don't clutter): claims marked ``ok`` get
    no view-model; only ``weak``/``unsupported`` claims surface accept/reject/edit.
    All claims are persisted regardless (the eager adjudication log). The pure split
    of the old ``_build_claim_review`` (Decision 5): returns the markdown block + a list
    of :class:`ClaimView`; the renderer builds its own buttons from the view-models.
    """
    flagged = [(c, cid) for c, cid in zip(claims, claim_ids, strict=True) if c.marker != MARKER_OK]
    if not flagged:
        return (
            f"\n\n---\n🔎 **Interpretation** — all {len(claims)} claim(s) grounded "
            "in cited evidence.",
            [],
        )
    lines = [f"\n\n---\n⚠ **{len(flagged)} claim(s) to review** (evidence vs interpretation):"]
    views: list[ClaimView] = []
    for c, cid in flagged:
        n = c.claim_index + 1
        badge = "unsupported" if c.marker != "weak" else "weakly grounded"
        lines.append(f"- **#{n}** {c.text}  _({badge})_")
        views.append(ClaimView(claim_id=cid, n=n, text=c.text, badge=badge))
    return "\n".join(lines), views


def _build_claims_block(
    record_id: str, full_answer: str, retrieved_chunks: list[RetrievedChunk]
) -> tuple[str, list[ClaimView]]:
    """Chunk 2a: segment the answer into claims, eager-persist them, and render the review block
    for the flagged ones (E1.2 — lifted from ``_handle_rag``). Advisory: any failure collapses to
    a "Claim adjudication unavailable" note + no flagged claims, never breaking the turn. Called
    only with a real ``record_id`` (the caller guards on it — no record, no claims)."""
    try:
        claims = segment_claims(full_answer, retrieved_chunks)
        claim_ids = record_claims(record_id, claims)
        return _build_claim_review(claims, claim_ids)
    except Exception as e:
        return f"\n\n_⚠ Claim adjudication unavailable: {e}_", []


def _export_sources(
    scored: list[tuple[Document, float]], fig_paths: dict[str, str]
) -> list[export.ExportSource]:
    """Map (doc, score) pairs to the export's source view (figure paths attached)."""
    sources: list[export.ExportSource] = []
    for i, (doc, score) in enumerate(scored):
        meta = doc.metadata
        fig_id = meta.get("figure_id", "")
        is_figure = meta.get("chunk_type") == "figure"
        sources.append(
            export.ExportSource(
                n=i + 1,
                filename=meta.get("filename"),
                page=meta.get("page"),
                section=meta.get("section"),
                reranker_score=float(score),
                is_figure=is_figure,
                image_path=fig_paths.get(fig_id) if is_figure else None,
                excerpt=doc.page_content[:300],
            )
        )
    return sources


@dataclass(frozen=True)
class _ProvenanceInputs:
    """The inputs to one turn's provenance + reviewer capture (E1.2 — bundled so the extracted
    :meth:`ChatController._capture_provenance_and_review` stays a single-argument seam)."""

    standalone: str
    original_query: str | None
    full_answer: str
    retrieved_chunks: list[RetrievedChunk]
    model_name: str | None
    embedding_model: str
    top_k: int
    token_input: int
    token_output: int
    latency_ms: float
    session_id: str
    scope_view: ScopeView | None
    turn_provider: str
    turn_model: str


@dataclass(frozen=True)
class _ProvenanceOutcome:
    """What one turn's provenance + reviewer capture produced (E1.2)."""

    record_id: str | None
    provenance_block: str
    review: ReviewResult | None


# ============================================================
# Controller
# ============================================================


class ChatController:
    """Owns the turn orchestration. Stateless across turns (the injected ``Session``
    carries per-conversation state). Imports the same library functions the old
    the original UI handler did; no UI-framework import."""

    def __init__(self, rag: RAGPipeline | None = None) -> None:
        if rag is not None:
            self.rag = rag  # test seam (cpc §13) — a fake; never carries a persisted selection
        else:
            self.rag = RAGPipeline()
            # ADR-011 (U1c): apply any persisted provider/model selection so a restart restores
            # it. A fresh RAGPipeline boots on the config default; only swap if the persisted
            # choice actually differs (skip a needless rebuild on the common no-switch boot).
            provider, model = app_settings.get_llm_selection()
            if (
                provider is not None
                and model is not None
                and (provider, model) != (self.rag.provider, self.rag.model)
            ):
                self.rag.set_chat_model(provider, model)
        # Cached once — the prompt template doesn't change between turns.
        self._answer_template_hash = template_hash(str(ANSWER_PROMPT))

    # -- public API -------------------------------------------------------

    def chunk_count(self) -> int:
        return self.rag.chunk_count()

    def reconfigure(self, provider: str, model: str) -> None:
        """Switch the live generation provider/model (ADR-011, U1c desktop provider switch).

        Validates and persists the choice via ``app_settings`` (raises :class:`ValueError` for
        an unknown or keyless provider — the API maps that to 400), then swaps the pipeline's
        generation model with a **direct method call** — never a module-global mutation. An
        in-flight turn already holds its own chain reference (``pipeline.set_chat_model``'s own
        guarantee) and finishes on the old model; the very next turn picks up the new one.
        """
        app_settings.set_llm_selection(provider, model)
        self.rag.set_chat_model(provider, model)

    def compare_retrieval(
        self,
        text: str,
        overrides: RagOverrides,
        scope_folder_id: str | None = None,
    ) -> compare.CompareResult:
        """Retrieval-only A/B compare (U6, ``feature-ab-compare-sandbox.md``).

        Runs ``retrieve_with_scores`` twice on the same raw query — A = locked defaults, B = the
        session ``overrides`` — and returns both ranked source sets + the diff + note. **$0**:
        retrieval only, no generation, no ``self.llm`` touch. Request-scoped: ``overrides`` rides
        the call, no module-global assigned (the ADR-010 isolation invariant). Only
        ``top_k``/``use_multi_query`` affect retrieval; the rest are answer-time (see the note).

        ``scope_folder_id`` (ADR-025 F2) is applied to **both** sides. The comparison exists to
        isolate a knob, so the document set is held constant — and an unscoped diff shown while a
        folder scope is active would describe retrieval the next answer will not perform.
        """
        scope, scope_view = _resolve_scope(scope_folder_id)
        eff_a: dict[str, int | bool] = {"top_k": TOP_K, "use_multi_query": USE_MULTI_QUERY}
        eff_b: dict[str, int | bool] = {
            "top_k": overrides.top_k if overrides.top_k is not None else TOP_K,
            "use_multi_query": (
                overrides.use_multi_query
                if overrides.use_multi_query is not None
                else USE_MULTI_QUERY
            ),
        }
        # A follows the global default (use_multi_query=None); B forces the override's value.
        pairs_a = self.rag.retrieve_with_scores(
            text, top_k=int(eff_a["top_k"]), use_multi_query=None, scope=scope
        )
        pairs_b = self.rag.retrieve_with_scores(
            text,
            top_k=int(eff_b["top_k"]),
            use_multi_query=overrides.use_multi_query,
            scope=scope,
        )
        return compare.build_result(
            text,
            [self._to_compare_source(d, s, i + 1) for i, (d, s) in enumerate(pairs_a)],
            [self._to_compare_source(d, s, i + 1) for i, (d, s) in enumerate(pairs_b)],
            eff_a,
            eff_b,
            _scope_label(scope_view),
        )

    @staticmethod
    def _to_compare_source(doc: Document, score: float, rank: int) -> compare.CompareSource:
        """Map one retrieved ``(Document, score)`` to a :class:`compare.CompareSource`.

        ``identity`` reuses the pipeline's dedup key (``doc_hash + "_" + sha256(page_content)``)
        so a source appearing on both sides matches exactly."""
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        identity = f"{doc.metadata.get('doc_hash', '')}_{content_hash}"
        return compare.CompareSource(
            rank=rank,
            filename=str(doc.metadata.get("filename", "unknown")),
            page=doc.metadata.get("page"),
            section=doc.metadata.get("section"),
            score=float(score),
            excerpt=doc.page_content[:240].strip(),  # a short preview for the compare card
            citation=format_citation(doc, rank),
            identity=identity,
        )

    def adjudicate(self, claim_id: str, decision: str, edited_text: str | None = None) -> None:
        """Record the user's verdict on one flagged claim. Lifts ``_resolve_claim``'s
        core; the renderer owns the success/error messaging (it catches and displays)."""
        adjudicate_claim(claim_id, decision, edited_text=edited_text)

    def export_conversation(self, session: Session, *, dev: bool) -> tuple[str, Path | None]:
        """Render a conversation to markdown, write to ``data/exports/``, and return
        ``(message, path)``. ``path`` is ``None`` when there is nothing to export.

        Sources from the **durable ``AnswerRecord`` transcript** by ``session_id`` — so a
        reopened or resumed past chat exports the same as a live one (the earlier turns live
        only in the store). The in-memory ``export_turns`` are richer (reviewer, figures,
        citation audit), so the dev bundle prefers them when this is a live session."""
        in_memory = session.export_turns
        durable = conversations.conversation_export_turns(session.session_id)
        turns = in_memory if (dev and in_memory) else (durable or in_memory)
        if not turns:
            return ("Nothing to export yet — ask a question first.", None)
        flavour = "debug" if dev else "transcript"
        subtitle = (
            f"Exported {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · session "
            f"{session.session_id}"
        )
        md = export.render_conversation_markdown(
            turns, title="Provenote conversation", subtitle=subtitle, dev=dev
        )
        path = export.write_markdown(f"{session.session_id}-{flavour}.md", md)
        return (f"📄 Exported {len(turns)} turn(s) — {flavour}. Saved to `{path}`.", path)

    def handle_message(
        self,
        session: Session,
        text: str,
        *,
        overrides: RagOverrides | None = None,
        scope_folder_id: str | None = None,
    ) -> Iterator[TurnEvent]:
        """Drive one turn. Ports ``on_message``'s dispatch order verbatim:
        (a) slash command, (b) pending claim-edit, (c) library query, (d) RAG path.

        ``overrides`` (ADR-010) only affects the RAG path — commands/library queries/claim
        edits have no retrieval or synthesis-mode knobs to override. Default ``None`` is
        byte-identical to before this feature existed.

        ``scope_folder_id`` (ADR-025 F2) restricts retrieval to one folder's documents for this
        turn only — request-scoped like ``overrides``, never stored on the session: a scope the
        backend remembered would be a scope the user could forget (spec S9). Same RAG-path-only
        carve."""
        # --- Slash commands ---
        parsed = parse_command(text)
        if parsed is not None:
            cmd, arg = parsed
            try:
                # Export commands need the live session transcript, so they're handled here
                # (stateful) rather than in the stateless commands.execute_command dispatcher.
                if cmd in ("export", "export-conversation", "export_conversation"):
                    msg, path = self.export_conversation(session, dev=False)
                    yield Result(self._command_result(msg, download_path=path))
                elif cmd in ("export-debug", "export_debug"):
                    msg, path = self.export_conversation(session, dev=True)
                    yield Result(self._command_result(msg, download_path=path))
                else:
                    yield Result(self._command_result(execute_command(cmd, arg)))
            except Exception as e:
                # A failing command (empty/missing DB, no API key, …) must not break the
                # turn or the SSE stream — surface it as a normal result.
                yield Result(self._command_result(f"⚠ `/{cmd}` failed: {e}"))
            return

        # --- Chunk 2a: claim edit follow-up (a prior "✎ Edit" set this) ---
        pending_edit = session.awaiting_edit
        if pending_edit is not None:
            session.awaiting_edit = None
            try:
                adjudicate_claim(pending_edit["id"], "edited", edited_text=text)
                yield Result(self._command_result(f"✏️ Claim #{pending_edit['n']} updated."))
            except Exception as e:
                yield Result(self._command_result(f"⚠ Edit failed: {e}"))
            return

        # --- Library metadata questions (answered from SQLite) ---
        if is_library_query(text):
            try:
                yield Result(self._command_result(answer_library_query(text)))
            except Exception as e:
                yield Result(self._command_result(f"⚠ Library query failed: {e}"))
            return

        # --- RAG pipeline ---
        yield from self._handle_rag(session, text, overrides, scope_folder_id)

    # -- internal ---------------------------------------------------------

    def _command_result(self, answer: str, *, download_path: Path | None = None) -> TurnResult:
        """Wrap a command/library/edit string as a minimal TurnResult (no sources,
        claims, or telemetry blocks)."""
        return TurnResult(
            answer=answer,
            mode="ai",
            sources=[],
            flagged_claims=[],
            usage=UsageView(0, 0, 0, None, _is_local(self.rag.provider)),
            standalone_query="",
            record_id=None,
            provenance_card_md="",
            claim_review_md="",
            sources_md="",
            usage_md="",
            citation_note_md="",
            download_path=download_path,
        )

    def _attach_source_evaluation(
        self,
        sources: list[SourceView],
        scored: list[tuple[Document, float]],
        *,
        markers_enabled: bool = EPISTEMICS_MARKERS_ENABLED,
    ) -> SourceEvalSummary | None:
        """ADR-027 **D3**: attach the always-on per-source epistemic evaluation (coverage/direction
        + doc year) to every retrieved source, and return the strip-level freshness. Independent of
        the D2/E3 answer-influence toggle — the strip **always** renders when a concept graph
        exists; ``markers_enabled`` only governs whether the *answer-surface* marker chips
        (``sv.markers``, E1.1) are populated from the same read. Scoped, indexed reads (KI-18), no
        LLM, no provider touched.

        The per-source join is a direct ``chunk_key`` lookup (E1.1 re-projection — flat + PC both
        resolve). Freshness: ``stale`` when the epistemics sidecar's
        ``graph_version`` differs from the current skeleton's — the graph was rebuilt without a
        ``compute_epistemics`` re-run. Returns ``None`` (no strip) when no concept graph is built.

        Advisory: any failure logs a **WARNING** and returns ``None`` rather than breaking the turn
        — a silent failure under an always-on strip is a silently-lying UI."""
        try:
            current = current_graph_version()
            if current is None:
                return None  # no concept graph built → nothing to assess, no strip
            chunk_keys = [sv.chunk_key for sv in sources if sv.chunk_key is not None]
            evals, sidecar_version = load_source_evaluations(chunk_keys)
            document_ids = [
                str(d) for d in (doc.metadata.get("document_id") for doc, _ in scored) if d
            ]
            years = document_years(document_ids)
            for sv, (doc, _score) in zip(sources, scored, strict=True):
                document_id = doc.metadata.get("document_id")
                year = years.get(str(document_id)) if document_id is not None else None
                ev: ChunkEval | None = (
                    evals.get(sv.chunk_key) if sv.chunk_key is not None else None
                )
                sv.evaluation = SourceEpistemics(
                    coverage=ev.coverage if ev is not None else None,
                    superseded=ev.superseded if ev is not None else False,
                    n_claims=ev.n_claims if ev is not None else 0,
                    year=year,
                )
                if ev is not None and markers_enabled:
                    sv.markers = derive_markers(
                        1 if ev.coverage == "contested" else 0, 1 if ev.superseded else 0
                    )
            return SourceEvalSummary(
                graph_version=sidecar_version or current,
                stale=sidecar_version is not None and sidecar_version != current,
            )
        except Exception as exc:
            # Advisory strip must never break a turn — but never silently, either (see above).
            log.warning("attach_source_evaluation_failed", error=str(exc))
            return None

    def _capture_provenance_and_review(self, pin: _ProvenanceInputs) -> _ProvenanceOutcome:
        """Record the answer's provenance + (when a heuristic signal fires and a reviewer is
        available) run the confined LLM reviewer, returning the rendered card block (E1.2 — the
        88-line block lifted verbatim out of ``_handle_rag``). Never blocks the answer: any failure
        collapses to a "Provenance capture failed" card and an empty ``record_id`` (the caller then
        skips claim adjudication). The ``overrides_note``/``scope_note`` suffix is appended by the
        caller, which owns those turn knobs."""
        prov_version = prompt_version_hash(
            template_hash=self._answer_template_hash,
            top_k=pin.top_k,
            use_parent_child=USE_PARENT_CHILD,
            embedding_model=pin.embedding_model,
        )
        record_id: str | None = None
        review: ReviewResult | None = None
        try:
            record_id = record_answer(
                query=pin.standalone,
                original_query=pin.original_query,
                answer=pin.full_answer,
                retrieved_chunks=pin.retrieved_chunks,
                model_name=pin.model_name,
                embedding_model=pin.embedding_model,
                prompt_version=prov_version,
                top_k=pin.top_k,
                use_parent_child=USE_PARENT_CHILD,
                token_input=pin.token_input,
                token_output=pin.token_output,
                latency_ms=pin.latency_ms,
                session_id=pin.session_id,
                retrieval_scope=_scope_dict(pin.scope_view),
            )
            prov = AnswerProvenance(
                id=record_id,
                query=pin.standalone,
                original_query=pin.original_query,
                answer=pin.full_answer,
                retrieved_chunks=pin.retrieved_chunks,
                model_name=pin.model_name,
                embedding_model=pin.embedding_model,
                prompt_version=prov_version,
                top_k=pin.top_k,
                use_parent_child=USE_PARENT_CHILD,
                token_input=pin.token_input,
                token_output=pin.token_output,
                latency_ms=pin.latency_ms,
            )
            signals = compute_confidence_signals(prov)
            # PR 5.1 — quiet UI on clean answers, loud on flagged ones. The card ALWAYS
            # renders (so the provenance id and active model are visible on every answer):
            # a compact neutral line on clean answers, a full ⚠ block when a signal fires.
            if signals.any():
                # PR 6 — when heuristic flags fire AND a reviewer is available, run the LLM
                # reviewer to add depth. ~$0.001 + ~1-2s per flagged answer (free + local
                # under Ollama). Clean answers skip the call. ADR-011 (U1c): the reviewer
                # follows the effective generation provider unless REVIEWER_PROVIDER is
                # explicitly pinned in the environment (resolve_reviewer's own rule).
                from doc_assistant.llm import (
                    get_reviewer_client,
                    resolve_reviewer,
                    reviewer_available,
                )

                reviewer_provider, reviewer_model = resolve_reviewer(
                    pin.turn_provider, pin.turn_model
                )
                if reviewer_available(reviewer_provider):
                    try:
                        review = review_answer(
                            prov, get_reviewer_client(pin.turn_provider, pin.turn_model)
                        )
                        # ADR-011: the recorded kind must match the instrument that actually ran.
                        # A followed switch to Ollama is no longer the Haiku reviewer — labeling it
                        # "llm_haiku" beside an ollama model_name would be a provenance lie.
                        reviewer_kind = (
                            "llm_haiku"
                            if reviewer_provider == "anthropic"
                            else f"llm_{reviewer_provider}"
                        )
                        persist_review(
                            record_id,
                            review,
                            reviewer_kind=reviewer_kind,
                            model_name=reviewer_model,
                        )
                    except Exception as e:
                        review = ReviewResult(error=f"reviewer setup failed: {e}")
            provenance_block = _format_provenance_card(
                prov, signals, review=review, is_local=_is_local(pin.turn_provider)
            )
        except Exception as e:
            # Never let provenance failure break the answer.
            provenance_block = f"\n\n_⚠ Provenance capture failed: {e}_"
        return _ProvenanceOutcome(
            record_id=record_id, provenance_block=provenance_block, review=review
        )

    def _handle_rag(
        self,
        session: Session,
        text: str,
        overrides: RagOverrides | None = None,
        scope_folder_id: str | None = None,
    ) -> Iterator[TurnEvent]:
        rag = self.rag
        history = session.history
        counter = session.counter
        user_question = text

        # --- ADR-011: snapshot the generation instrument for the whole turn. A live provider
        # switch (RAGPipeline.set_chat_model) can land mid-turn; the answer must stream on —
        # and every recorded label (model_name, usage, reviewer resolution) must name — the
        # SAME instrument, so read the trio once here and never through ``rag`` again. ---
        turn_llm = rag.llm
        turn_provider = rag.provider
        turn_model = rag.model

        # --- ADR-010: resolve effective per-turn knobs (None = locked default; never a
        # module-global assignment — request-scoped so concurrent turns can't leak). ---
        knobs = _resolve_turn_knobs(overrides)

        # --- ADR-025 F2: resolve the retrieval scope ONCE for the turn. Membership lives in
        # SQLite and is editable at any moment, so the hash set is read here and then frozen —
        # the answer, the chip, and the provenance record all describe the same set. An unknown
        # or empty folder yields an empty scope and an honest zero-source turn; it never falls
        # back to the whole library (spec S3). ---
        scope, scope_view = _resolve_scope(scope_folder_id)

        pre_in, pre_out = counter.input_tokens, counter.output_tokens
        turn_start = time.monotonic()

        if history:
            standalone = rag.rewrite(user_question, history, counter=counter)
            yield Step("Understanding context", f"Searching for: {standalone}")
        else:
            standalone = user_question

        scored = rag.retrieve_with_scores(
            standalone,
            top_k=knobs.top_k,
            use_multi_query=(overrides.use_multi_query if overrides else None),
            scope=scope,
        )
        yield Step("Searching documents", f"Found {len(scored)} relevant passages")

        docs = [doc for doc, _ in scored]

        # Feature 4c: a retrieved figure chunk (chunk_type='figure') carries the Figure
        # sidecar id — resolve its cropped PNG so a renderer can show it inline. Batch the
        # path lookup (one DB read for the turn).
        fig_ids = [
            fid
            for doc in docs
            if doc.metadata.get("chunk_type") == "figure"
            and (fid := doc.metadata.get("figure_id"))
        ]
        fig_paths = load_figure_image_paths(fig_ids) if fig_ids else {}

        sources = _build_source_views(scored, fig_paths)
        # ADR-027 D3: the always-on source-evaluation strip (per-source coverage/year + freshness);
        # markers_enabled= is U1b's per-turn override over the D2 answer-surface marker chips only.
        source_eval = self._attach_source_evaluation(
            sources, scored, markers_enabled=knobs.markers_enabled
        )
        retrieved_chunks = _build_retrieved_chunks(
            scored, reviewer_evidence_chars=knobs.reviewer_evidence_chars
        )

        # --- synthesis_mode=human (locked default or a per-turn override): evidence only;
        # skip the interpretation call ---
        if knobs.synthesis_mode == "human":
            yield Result(
                self._human_result(
                    session,
                    user_question=user_question,
                    standalone=standalone,
                    scored=scored,
                    fig_paths=fig_paths,
                    sources=sources,
                    retrieved_chunks=retrieved_chunks,
                    turn_start=turn_start,
                    eff_top_k=knobs.top_k,
                    overrides_note=knobs.overrides_note,
                    scope=scope_view,
                    source_eval=source_eval,
                )
            )
            return

        full_answer = ""
        for tok in rag.stream_answer(standalone, docs, counter=counter, llm=turn_llm):
            full_answer += tok
            yield Token(tok)

        turn_in = counter.input_tokens - pre_in
        turn_out = counter.output_tokens - pre_out
        latency_ms = (time.monotonic() - turn_start) * 1000.0

        # --- Provenance capture + reviewer (sidecar; never blocks the answer — E1.2) ---
        embedding_model = get_active_model_name()
        model_name = getattr(turn_llm, "model", None) or getattr(turn_llm, "model_name", None)
        prov_out = self._capture_provenance_and_review(
            _ProvenanceInputs(
                standalone=standalone,
                original_query=user_question if standalone != user_question else None,
                full_answer=full_answer,
                retrieved_chunks=retrieved_chunks,
                model_name=model_name,
                embedding_model=embedding_model,
                top_k=knobs.top_k,
                token_input=turn_in,
                token_output=turn_out,
                latency_ms=latency_ms,
                session_id=session.session_id,
                scope_view=scope_view,
                turn_provider=turn_provider,
                turn_model=turn_model,
            )
        )
        record_id = prov_out.record_id
        review = prov_out.review
        provenance_block = (
            prov_out.provenance_block + knobs.overrides_note + _scope_note(scope_view)
        )

        sources_block = _sources_block(sources)
        usage_block = self._usage_block(
            full_answer, turn_in, turn_out, counter, provider=turn_provider, model=turn_model
        )

        # --- Chunk 2a: segment + eager-persist claims; surface flagged ones ---
        claim_review_block = ""
        flagged_claims: list[ClaimView] = []
        if record_id is not None:
            claim_review_block, flagged_claims = _build_claims_block(
                record_id, full_answer, retrieved_chunks
            )

        # Post-hoc citation audit — quiet unless the model cited badly (out-of-range
        # numbers or malformed forms the [n] parser silently drops). Surface, don't rewrite.
        citation = audit_citations(full_answer, len(docs))
        citation_block = (
            "" if citation.clean else f"\n\n---\n⚠ **Citation check:** {citation.note()}"
        )

        # --- Export: stash this turn + append the per-turn debug log event ---
        reviewer_summary = None
        if review is not None and not review.error:
            reviewer_summary = (
                f"faithfulness {review.faithfulness}/5 · citation {review.citation_density}/5 · "
                f"hedging {review.hedging_adequacy}/5"
            )
        self._append_export_turn(
            session,
            export.ExportTurn(
                question=user_question,
                answer=full_answer,
                standalone_query=standalone,
                sources=_export_sources(scored, fig_paths),
                reviewer_summary=reviewer_summary,
                failure_tag=(review.failure_tag if review is not None else None),
                citation_note=citation.note(),
                token_input=turn_in,
                token_output=turn_out,
                latency_ms=latency_ms,
                model_name=model_name,
                embedding_model=embedding_model,
                record_id=record_id,
            ),
        )

        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": full_answer})

        yield Result(
            TurnResult(
                answer=full_answer,
                mode="ai",
                sources=sources,
                flagged_claims=flagged_claims,
                usage=UsageView(
                    turn_input=turn_in,
                    turn_output=turn_out,
                    session_total=counter.total(),
                    cost_usd=None if _is_local(turn_provider) else counter.cost_usd(),
                    is_local=_is_local(turn_provider),
                ),
                standalone_query=standalone,
                record_id=record_id,
                provenance_card_md=provenance_block,
                claim_review_md=claim_review_block,
                sources_md=sources_block,
                usage_md=usage_block,
                citation_note_md=citation_block,
                scope=scope_view,
                source_eval=source_eval,
            )
        )

    def _human_result(
        self,
        session: Session,
        *,
        user_question: str,
        standalone: str,
        scored: list[tuple[Document, float]],
        fig_paths: dict[str, str],
        sources: list[SourceView],
        retrieved_chunks: list[RetrievedChunk],
        turn_start: float,
        eff_top_k: int = TOP_K,
        overrides_note: str = "",
        scope: ScopeView | None = None,
        source_eval: SourceEvalSummary | None = None,
    ) -> TurnResult:
        """``synthesis_mode=human`` (locked default or a per-turn ADR-010 override) —
        evidence only; no interpretation call. Records provenance silently (no card shown),
        stashes the export turn, updates history."""
        human_answer = "(human synthesis mode — evidence only; no AI interpretation)"
        with contextlib.suppress(Exception):  # provenance is a sidecar, never blocks
            record_answer(
                query=standalone,
                original_query=user_question if standalone != user_question else None,
                answer=human_answer,
                retrieved_chunks=retrieved_chunks,
                embedding_model=get_active_model_name(),
                top_k=eff_top_k,
                use_parent_child=USE_PARENT_CHILD,
                latency_ms=(time.monotonic() - turn_start) * 1000.0,
                session_id=session.session_id,
                retrieval_scope=_scope_dict(scope),
            )
        self._append_export_turn(
            session,
            export.ExportTurn(
                question=user_question,
                answer=human_answer,
                standalone_query=standalone,
                sources=_export_sources(scored, fig_paths),
                embedding_model=get_active_model_name(),
            ),
        )
        session.history.append({"role": "user", "content": user_question})
        session.history.append({"role": "assistant", "content": "(human mode: evidence only)"})
        return TurnResult(
            answer=(
                "🧑 **Human synthesis mode** — evidence only; the interpretation is yours.\n\n"
                + render_evidence_markdown(retrieved_chunks)
                + overrides_note
                # Human mode renders no provenance card, so the scope note rides the answer —
                # otherwise a scoped evidence-only turn would state its scope nowhere.
                + _scope_note(scope)
            ),
            mode="human",
            sources=sources,
            flagged_claims=[],
            usage=UsageView(0, 0, session.counter.total(), None, _is_local(self.rag.provider)),
            standalone_query=standalone,
            record_id=None,
            provenance_card_md="",
            claim_review_md="",
            sources_md="",
            usage_md="",
            citation_note_md="",
            scope=scope,
            source_eval=source_eval,
        )

    def _usage_block(
        self,
        full_answer: str,
        turn_in: int,
        turn_out: int,
        counter: TokenCounter,
        *,
        provider: str,
        model: str,
    ) -> str:
        if _is_local(provider):
            # Local models report no token usage to the LangChain callback, so the real
            # counts are zero — showing "0 tokens / $0.0000" reads as broken. Be honest:
            # no metered cost, with a rough output estimate from text. ``provider``/``model``
            # are the caller's turn snapshot (ADR-011) — never read live off ``self.rag``,
            # which a mid-turn switch may already have moved.
            est_out = max(0, len(full_answer) // 4)
            return (
                f"\n\n---\n"
                f"🖥 **Local model** (`{provider}/{model}`) — no metered token "
                f"cost; provider reports no usage. (~{est_out:,} output tokens, estimated.)"
            )
        turn_total = turn_in + turn_out
        return (
            f"\n\n---\n"
            f"📊 **This turn:** {turn_in:,} in + {turn_out:,} out "
            f"= {turn_total:,} tokens "
            f"(~${(turn_in * 1.0 + turn_out * 5.0) / 1_000_000:.4f})  \n"
            f"**Session total:** {counter.total():,} tokens "
            f"(~${counter.cost_usd():.4f})"
        )

    def _append_export_turn(self, session: Session, turn: export.ExportTurn) -> None:
        """Stash a turn in the session transcript and append its event to the session log."""
        session.export_turns.append(turn)
        with contextlib.suppress(Exception):  # the log is a sidecar — never break a turn
            export.append_log_event(session.session_id, export.log_event(turn))
