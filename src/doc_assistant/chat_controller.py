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
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document

from doc_assistant import app_settings, export
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
from doc_assistant.epistemics import (
    MARKER_CONTESTED,
    MARKER_SUPERSEDED,
    MarkedChunk,
    load_epistemics_index,
    load_marked_chunks,
    markers_for_parent,
)
from doc_assistant.ingest.figures import load_figure_image_paths
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


# ============================================================
# View models (pure render payload — no UI framework types)
# ============================================================


@dataclass
class SourceView:
    """One retrieved source, render-ready (side panel / sources block)."""

    n: int
    citation: str  # format_citation(doc, n)
    excerpt: str  # ~800-char side-panel preview (with trailing "..." when truncated)
    figure_path: str | None  # resolved PNG path (local desktop render); never crosses the API
    chunk_key: str | None  # ADR-2; the 7d marker join key
    markers: list[str] = field(default_factory=list)  # PR-M1: contested / superseded_trend
    figure_id: str | None = None  # PR-M3: the id the web/API renders via GET /api/figures/{id}


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
    """Epistemics-format join key (ADR-2): ``{document_id}:{chunk_index}`` for a flat
    /baseline chunk; ``None`` for a parent-child chunk (which carries ``parent_index``,
    not ``chunk_index``) or a row missing ``document_id``.

    Do **not** invent a ``p{parent_index}`` key — it cannot join against
    ``epistemics.load_epistemics_index`` and would mask the gap. The PC→baseline
    mapping is PR-M1's decision.
    """
    document_id = meta.get("document_id")
    chunk_index = meta.get("chunk_index")
    if document_id is not None and chunk_index is not None:
        return f"{document_id}:{chunk_index}"
    return None  # TODO(PR-M1): PC→baseline chunk-key mapping for parent chunks


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
    for i, (doc, _score) in enumerate(scored):
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

    def adjudicate(self, claim_id: str, decision: str, edited_text: str | None = None) -> None:
        """Record the user's verdict on one flagged claim. Lifts ``_resolve_claim``'s
        core; the renderer owns the success/error messaging (it catches and displays)."""
        adjudicate_claim(claim_id, decision, edited_text=edited_text)

    def export_conversation(self, session: Session, *, dev: bool) -> tuple[str, Path | None]:
        """Render the session's turns to markdown, write to ``data/exports/``, and return
        ``(message, path)``. ``path`` is ``None`` when there is nothing to export (the
        message then explains that). Refined from the spec's ``-> Path`` so the exact
        confirmation text is built here, not in the renderer (no business logic in apps/)."""
        turns = session.export_turns
        if not turns:
            return ("Nothing to export yet — ask a question first.", None)
        flavour = "debug" if dev else "transcript"
        md = export.render_conversation_markdown(
            turns, title=f"doc_assistant session {session.session_id}", dev=dev
        )
        path = export.write_markdown(f"{session.session_id}-{flavour}.md", md)
        return (f"📄 Exported {len(turns)} turn(s) — {flavour}. Saved to `{path}`.", path)

    def handle_message(
        self, session: Session, text: str, *, overrides: RagOverrides | None = None
    ) -> Iterator[TurnEvent]:
        """Drive one turn. Ports ``on_message``'s dispatch order verbatim:
        (a) slash command, (b) pending claim-edit, (c) library query, (d) RAG path.

        ``overrides`` (ADR-010) only affects the RAG path — commands/library queries/claim
        edits have no retrieval or synthesis-mode knobs to override. Default ``None`` is
        byte-identical to before this feature existed."""
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
        yield from self._handle_rag(session, text, overrides)

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

    def _attach_markers(
        self,
        sources: list[SourceView],
        scored: list[tuple[Document, float]],
        *,
        enabled: bool = EPISTEMICS_MARKERS_ENABLED,
    ) -> None:
        """PR-M1: attach 7d epistemics markers (contested / superseded-trend) to each
        source. Flat chunks join directly on ``chunk_key`` against the marker index; PC
        parents map via text containment (ADR-1). Read-only, no LLM, no provider touched
        (honors the credit guard). A clean no-op — every ``markers`` stays empty — when the
        epistemics sidecar is absent/empty, so the turn is byte-identical to before. The
        read sides are loaded at most once per turn (Decision 6).

        Defensive: markers are advisory (inform, never block), so **any** failure to load
        them — e.g. the ``chunk_epistemics`` table absent on an older DB, a Chroma hiccup —
        leaves the sources unmarked rather than breaking the turn.

        ``enabled`` defaults to the locked ``EPISTEMICS_MARKERS_ENABLED`` config default; a
        caller passes the per-turn effective value (U1b / ADR-010 amendment) to override it.
        When disabled this returns before any load, so every ``markers`` stays empty and the
        turn is the byte-identical M0/M1 path."""
        if not enabled:
            return
        try:
            document_ids = [
                str(d) for d in (doc.metadata.get("document_id") for doc, _ in scored) if d
            ]
            index: dict[str, list[str]] | None = None
            marked_by_doc: dict[str, list[MarkedChunk]] | None = None
            for sv, (doc, _score) in zip(sources, scored, strict=True):
                if sv.chunk_key is not None:
                    if index is None:
                        index = load_epistemics_index()
                    markers = index.get(sv.chunk_key)
                    if markers:
                        sv.markers = list(markers)
                    continue
                document_id = doc.metadata.get("document_id")
                if not document_id:
                    continue
                if marked_by_doc is None:
                    marked_by_doc = load_marked_chunks(document_ids)
                markers = markers_for_parent(
                    doc.page_content, marked_by_doc.get(str(document_id), [])
                )
                if markers:
                    sv.markers = markers
        except Exception:
            return  # advisory markers must never break a turn

    def _handle_rag(
        self, session: Session, text: str, overrides: RagOverrides | None = None
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
        eff_top_k = overrides.top_k if overrides and overrides.top_k is not None else TOP_K
        eff_synthesis_mode = (
            overrides.synthesis_mode if overrides and overrides.synthesis_mode else SYNTHESIS_MODE
        )
        eff_multi_query = (
            USE_MULTI_QUERY
            if overrides is None or overrides.use_multi_query is None
            else overrides.use_multi_query
        )
        eff_markers_enabled = (
            EPISTEMICS_MARKERS_ENABLED
            if overrides is None or overrides.epistemics_markers_enabled is None
            else overrides.epistemics_markers_enabled
        )
        eff_reviewer_evidence_chars = (
            overrides.reviewer_evidence_chars
            if overrides and overrides.reviewer_evidence_chars is not None
            else REVIEWER_EVIDENCE_CHARS
        )
        overrides_note = _overrides_note(
            eff_top_k,
            eff_synthesis_mode,
            eff_multi_query,
            eff_markers_enabled,
            eff_reviewer_evidence_chars,
        )

        pre_in, pre_out = counter.input_tokens, counter.output_tokens
        turn_start = time.monotonic()

        if history:
            standalone = rag.rewrite(user_question, history, counter=counter)
            yield Step("Understanding context", f"Searching for: {standalone}")
        else:
            standalone = user_question

        scored = rag.retrieve_with_scores(
            standalone,
            top_k=eff_top_k,
            use_multi_query=(overrides.use_multi_query if overrides else None),
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
        # PR-M1: 7d markers (no-op when sidecar absent); enabled= is U1b's per-turn override.
        self._attach_markers(sources, scored, enabled=eff_markers_enabled)
        retrieved_chunks = _build_retrieved_chunks(
            scored, reviewer_evidence_chars=eff_reviewer_evidence_chars
        )

        # --- synthesis_mode=human (locked default or a per-turn override): evidence only;
        # skip the interpretation call ---
        if eff_synthesis_mode == "human":
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
                    eff_top_k=eff_top_k,
                    overrides_note=overrides_note,
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

        # --- Provenance capture (sidecar; never blocks the answer) ---
        embedding_model = get_active_model_name()
        prov_version = prompt_version_hash(
            template_hash=self._answer_template_hash,
            top_k=eff_top_k,
            use_parent_child=USE_PARENT_CHILD,
            embedding_model=embedding_model,
        )
        model_name = getattr(turn_llm, "model", None) or getattr(turn_llm, "model_name", None)
        record_id: str | None = None
        review: ReviewResult | None = None
        try:
            record_id = record_answer(
                query=standalone,
                original_query=user_question if standalone != user_question else None,
                answer=full_answer,
                retrieved_chunks=retrieved_chunks,
                model_name=model_name,
                embedding_model=embedding_model,
                prompt_version=prov_version,
                top_k=eff_top_k,
                use_parent_child=USE_PARENT_CHILD,
                token_input=turn_in,
                token_output=turn_out,
                latency_ms=latency_ms,
            )
            prov = AnswerProvenance(
                id=record_id,
                query=standalone,
                original_query=user_question if standalone != user_question else None,
                answer=full_answer,
                retrieved_chunks=retrieved_chunks,
                model_name=model_name,
                embedding_model=embedding_model,
                prompt_version=prov_version,
                top_k=eff_top_k,
                use_parent_child=USE_PARENT_CHILD,
                token_input=turn_in,
                token_output=turn_out,
                latency_ms=latency_ms,
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

                reviewer_provider, reviewer_model = resolve_reviewer(turn_provider, turn_model)
                if reviewer_available(reviewer_provider):
                    try:
                        review = review_answer(
                            prov, get_reviewer_client(turn_provider, turn_model)
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
                prov, signals, review=review, is_local=_is_local(turn_provider)
            )
        except Exception as e:
            # Never let provenance failure break the answer.
            provenance_block = f"\n\n_⚠ Provenance capture failed: {e}_"
        provenance_block += overrides_note

        sources_block = _sources_block(sources)
        usage_block = self._usage_block(
            full_answer, turn_in, turn_out, counter, provider=turn_provider, model=turn_model
        )

        # --- Chunk 2a: segment + eager-persist claims; surface flagged ones ---
        claim_review_block = ""
        flagged_claims: list[ClaimView] = []
        if record_id is not None:
            try:
                claims = segment_claims(full_answer, retrieved_chunks)
                claim_ids = record_claims(record_id, claims)
                claim_review_block, flagged_claims = _build_claim_review(claims, claim_ids)
            except Exception as e:
                claim_review_block = f"\n\n_⚠ Claim adjudication unavailable: {e}_"

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
