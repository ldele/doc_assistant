"""Pure formatters and turn-knob resolution, moved out of the original UI handler.

No I/O and no pipeline calls — everything here is a deterministic transform over the view models,
which is what makes the turn parity test (``tests/integration/test_turn_parity.py``) meaningful."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.documents import Document

from doc_assistant import app_settings, export
from doc_assistant.chat_controller.session import RagOverrides, _TurnKnobs
from doc_assistant.chat_controller.views import ClaimView, ScopeView, SourceView
from doc_assistant.config import (
    EPISTEMICS_MARKERS_ENABLED,
    REVIEWER_EVIDENCE_CHARS,
    SYNTHESIS_MODE,
    TOP_K,
    USE_MULTI_QUERY,
)
from doc_assistant.knowledge.epistemics import MARKER_CONTESTED, MARKER_SUPERSEDED
from doc_assistant.library import folder_doc_hashes, get_folder
from doc_assistant.pipeline import format_citation
from doc_assistant.provenance import (
    AnswerProvenance,
    ConfidenceSignals,
    RetrievedChunk,
    record_claims,
)
from doc_assistant.reviewer import ReviewResult
from doc_assistant.synthesis import MARKER_OK, MARKER_WEAK, Claim, segment_claims

log = structlog.get_logger(__name__)


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
    markers_default: bool = EPISTEMICS_MARKERS_ENABLED,
) -> str:
    """ADR-010 Decision 5: provenance shows the *effective* knob values and flags any that
    differ from the locked default. Returns "" (no-op, byte-identical turn) when every
    effective value equals its default — i.e. ``overrides=None`` or an all-``None``
    ``RagOverrides``. The markers baseline is ``markers_default`` — the *persisted-effective*
    default (ADR-027 D2, E3), not the raw config constant: the persisted toggle is the user's
    chosen default, and stamping it "Session override (this answer only)" on every turn would
    be a provenance lie. Only a genuine per-turn U1b diff fires the note."""
    diffs = []
    if eff_top_k != TOP_K:
        diffs.append(f"top_k={eff_top_k} (default {TOP_K})")
    if eff_synthesis_mode != SYNTHESIS_MODE:
        diffs.append(f"synthesis_mode={eff_synthesis_mode} (default {SYNTHESIS_MODE})")
    if eff_multi_query != USE_MULTI_QUERY:
        diffs.append(f"multi_query={eff_multi_query} (default {USE_MULTI_QUERY})")
    if eff_markers_enabled != markers_default:
        diffs.append(
            f"epistemics_markers_enabled={eff_markers_enabled} (default {markers_default})"
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

    Request-scoped: reads ``overrides``, the config constants, and (for the markers knob) the
    persisted answer-layer default — never a module global, so concurrent turns on the shared
    controller cannot leak. ``None`` = the default. The markers baseline is E3's three-layer
    resolution (ADR-027 D2): per-turn U1b override > persisted setting > config default —
    ``app_settings.effective_markers_enabled()`` supplies the lower two layers, re-read each
    turn so a Settings toggle applies from the very next turn without a restart (the ADR-011
    ``effective_llm`` precedent). ``multi_query`` here is the *effective* value carried into
    the provenance note only — the retrieval call passes the RAW ``overrides.use_multi_query``
    (``None`` → the pipeline's own default), a deliberately distinct path this resolution does
    not touch."""
    top_k = overrides.top_k if overrides and overrides.top_k is not None else TOP_K
    synthesis_mode = (
        overrides.synthesis_mode if overrides and overrides.synthesis_mode else SYNTHESIS_MODE
    )
    multi_query = (
        USE_MULTI_QUERY
        if overrides is None or overrides.use_multi_query is None
        else overrides.use_multi_query
    )
    markers_default = app_settings.effective_markers_enabled()
    markers_enabled = (
        markers_default
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
            top_k,
            synthesis_mode,
            multi_query,
            markers_enabled,
            reviewer_evidence_chars,
            markers_default=markers_default,
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
    source_strip_rendered: bool = False,
) -> str:
    """Render an AnswerProvenance as a plain-markdown card (no raw HTML).

    Clean answers get a compact three-line block; when a confidence signal
    fires the block expands with the signal breakdown, the reviewer verdict,
    and the full per-source retrieval-relevance scores, led by a ⚠ chip. Filenames are
    not repeated — they live in the always-visible "Sources:" block; the card
    keys scores by source number. Full per-chunk metadata is in the DB /
    `/export-record`.

    **Every score here is retrieval relevance from the cross-encoder reranker — how well a chunk
    matches the question. None of them is a judgement of the source's quality, and nothing in this
    app produces one.** Said explicitly at each mention because the numbers sit beside the
    epistemic assessment, where a bare decimal invites the other reading (user report 2026-08-05).

    ``source_strip_rendered`` suppresses the per-source list: when the ADR-027 D3 strip is on
    screen it already shows exactly these numbers, keyed the same way, so printing them again put
    the same measure in two places on one answer. The card keeps the *aggregate* signals (max,
    top-3 span) — those are the thresholds the confidence verdict is actually derived from, and
    they appear nowhere else.
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
            f" · **best source relevance** `{signals.max_score:.3f}`"
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
        f"- best source relevance: `{signals.max_score:.3f}`"
        f"{' ⚠' if signals.weak_retrieval else ''}  \n"
        f"- top-3 relevance span: `{signals.top3_span:.3f}`"
        f"{' ⚠' if signals.score_cluster_concern else ''}  \n"
        f"- unique source documents: `{signals.unique_sources}`"
        f"{' ⚠' if signals.single_source_risk else ''}"
    )
    # Omitted when the source-evaluation strip already shows these numbers (see docstring).
    score_block = (
        ""
        if source_strip_rendered
        else "**Source relevance** (reranker score per source number above)\n"
        + "\n".join(
            f"- [{i + 1}] `{c.reranker_score:.3f}`"
            if c.reranker_score is not None
            else f"- [{i + 1}] `-`"
            for i, c in enumerate(prov.retrieved_chunks)
        )
        + "\n\n"
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
        f"{score_block}"
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


#: Presentation labels for a flagged claim's marker (KI-37 — see ``_build_claim_review``).
#: ``uncited`` and ``unresolved citation`` both come from ``MARKER_UNSUPPORTED``; the split is
#: ``Claim.citations``, which is empty only when the sentence carried no citation token at all.
BADGE_WEAK = "weakly grounded"
BADGE_UNCITED = "uncited"
BADGE_UNRESOLVED = "unresolved citation"


def _claim_badge(claim: Claim) -> str:
    """The label a flagged claim shows: what the structural marker actually found."""
    if claim.marker == MARKER_WEAK:
        return BADGE_WEAK
    # MARKER_UNSUPPORTED: either nothing was cited, or every number cited maps to no
    # retrieved source (out-of-range). Different defects, different fixes — say which.
    return BADGE_UNRESOLVED if claim.citations else BADGE_UNCITED


def _build_claim_review(claims: list[Claim], claim_ids: list[str]) -> tuple[str, list[ClaimView]]:
    """Render the adjudication section + per-claim view-models for *flagged* claims only.

    Quiet on clean answers (UX: inform, don't clutter): claims marked ``ok`` get
    no view-model; only ``weak``/``unsupported`` claims surface accept/reject/edit.
    All claims are persisted regardless (the eager adjudication log). The pure split
    of the old ``_build_claim_review`` (Decision 5): returns the markdown block + a list
    of :class:`ClaimView`; the renderer builds its own buttons from the view-models.

    **Badges say what the marker measures, not more** (KI-37, 2026-08-05). The marker layer is
    structural — it asks whether a sentence carries a citation that resolves to a retrieved
    source, nothing about whether the sentence is *true*. It used to render that as
    "unsupported", which (a) collided with the LLM reviewer's ``unsupported claims: N`` shown on
    the same card, where the word means "contradicted by / outrunning the evidence", and (b) read
    as an accusation on a *correct refusal* ("the sources do not mention X" cites nothing, and
    should not). ``MARKER_UNSUPPORTED`` and the persisted ``AnswerClaim.marker`` are unchanged —
    this is presentation only, and it splits the marker the same three ways the RG-012 gate does.
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
        badge = _claim_badge(c)
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
    # ADR-027 D2 (E3): the effective answer-layer epistemics flag for this turn, snapshotted
    # into the AnswerRecord (ADR-011 instrument discipline).
    markers_enabled: bool
    # Whether the always-on source-evaluation strip (ADR-027 D3) will render for this turn — i.e.
    # whether the per-source rerank scores are ALREADY on screen. The provenance card omits its own
    # copy of them when they are; see ``_provenance_card``. False when no concept graph exists, in
    # which case the card is the only per-source surface and keeps the list.
    source_strip_rendered: bool = False


@dataclass(frozen=True)
class _ProvenanceOutcome:
    """What one turn's provenance + reviewer capture produced (E1.2)."""

    record_id: str | None
    provenance_block: str
    review: ReviewResult | None
