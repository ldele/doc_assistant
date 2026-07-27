"""Turn view models — the pure render payload.

No UI-framework types cross this line: every frontend renders the same value objects.
``apps/api/models/chat.py`` mirrors these one-for-one onto the wire."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
