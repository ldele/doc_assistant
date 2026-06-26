"""Page content classification (Phase 6 / Feature 4 foundation).

The shared detection layer under both table extraction (4a) and figure
handling (4b): instead of each feature running its own geometric detector
(which confuses figures, charts and tables — they're all gridded, bounded
regions), classify what a page *contains* once, then route.

The discriminating signals were measured on the eLife corpus and separate
the classes by ~orders of magnitude, so cheap heuristics suffice — no ML,
no Marker:

* **Charts** are drawn as vector paths -> huge ``curve_count`` (1k-78k),
  versus 8-187 on table/text pages.
* **Photos / raster figures** are embedded images -> large
  ``image_area_fraction`` (0.09-0.60), versus 0 on table/text pages.
* **Tables** carry a "Table N" caption, have near-zero curves, and no
  large image; figures carry "Figure N".

**Scope (v1): page-level.** Signals are aggregated per page, so a page that
mixes a table *and* a chart is classified by the dominant signal (chart),
not split into regions. True per-region bbox splitting is the deeper 4b
step; page-level routing already fixes the figure-as-table problem that
caption-gating alone left on caption-less or mixed pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import structlog

log = structlog.get_logger(__name__)

RegionKind = Literal["table", "chart", "photo", "figure", "text"]

# Thresholds measured across the public eLife corpus (~50 pages, 2 docs).
# Tunable; validate on a wider corpus before trusting as universal.
CHART_CURVE_MIN = 1000  # vector curves at/above this => a chart/plot page
IMAGE_AREA_MIN = 0.05  # raster coverage fraction at/above this => a raster figure

# Caption is the semantic anchor that separates a data table from a figure.
TABLE_CAPTION_RE = re.compile(r"(?im)^\s*\**\s*table\s+\d+\s*[.:]")
FIGURE_CAPTION_RE = re.compile(r"(?im)^\s*\**\s*(?:figure|fig\.?)\s+\d+\s*[.:]")


@dataclass
class PageSignals:
    """Cheap per-page signals feeding ``classify_page`` (no I/O — pure data)."""

    page: int  # 1-based
    curve_count: int
    image_area_fraction: float  # sum(raster image area) / page area, in [0, ~1]
    has_table_caption: bool
    has_figure_caption: bool


@dataclass
class PageClassification:
    """The routing verdict for one page."""

    page: int
    kind: RegionKind  # dominant content kind
    is_table_candidate: bool  # safe to run table-structure extraction here
    is_figure: bool  # holds a figure/chart/photo (feeds 4b)
    reason: str


def classify_page(sig: PageSignals) -> PageClassification:
    """Classify a page from its signals. Pure; the heart of the router.

    A page is a *table candidate* only when it carries a table caption and
    is neither chart-dominated nor image-dominated — so a chart page with a
    stray "Table" mention is not mis-routed into table extraction.
    """
    is_chart = sig.curve_count >= CHART_CURVE_MIN
    has_raster_figure = sig.image_area_fraction >= IMAGE_AREA_MIN
    is_table_candidate = sig.has_table_caption and not is_chart and not has_raster_figure
    is_figure = sig.has_figure_caption or is_chart or has_raster_figure

    if is_table_candidate:
        kind: RegionKind = "table"
        reason = "table caption, low curves, no large image"
    elif is_chart:
        kind = "chart"
        reason = f"curve_count {sig.curve_count} >= {CHART_CURVE_MIN}"
    elif has_raster_figure:
        kind = "photo"
        reason = f"image_area_fraction {sig.image_area_fraction:.2f} >= {IMAGE_AREA_MIN}"
    elif sig.has_figure_caption:
        kind = "figure"
        reason = "figure caption, no dominant chart/image signal"
    else:
        kind = "text"
        reason = "no table/figure caption, no chart/image signal"

    return PageClassification(
        page=sig.page,
        kind=kind,
        is_table_candidate=is_table_candidate,
        is_figure=is_figure,
        reason=reason,
    )


# ============================================================
# Signal extraction (impure — opens the PDF via PyMuPDF)
# ============================================================


def page_signals(page: object, page_number: int) -> PageSignals:
    """Compute ``PageSignals`` for one PyMuPDF page.

    ``page`` is typed ``object`` because PyMuPDF is untyped; the attribute
    access is guarded by the library contract, not the type checker.
    """
    raw = page.get_text("dict")  # type: ignore[attr-defined]
    text = page.get_text()  # type: ignore[attr-defined]
    page_rect = page.rect  # type: ignore[attr-defined]
    page_area = abs(page_rect.width * page_rect.height) or 1.0

    image_area = 0.0
    for block in raw.get("blocks", []):
        if block.get("type") == 1:  # raster image block
            x0, y0, x1, y1 = block["bbox"]
            image_area += abs((x1 - x0) * (y1 - y0))

    curve_count = 0
    for drawing in page.get_drawings():  # type: ignore[attr-defined]
        for item in drawing["items"]:
            if item[0] in ("c", "qu"):  # bezier / quad curve ops
                curve_count += 1

    return PageSignals(
        page=page_number,
        curve_count=curve_count,
        image_area_fraction=image_area / page_area,
        has_table_caption=bool(TABLE_CAPTION_RE.search(text)),
        has_figure_caption=bool(FIGURE_CAPTION_RE.search(text)),
    )


def analyze_pages(pdf_path: str) -> list[PageClassification]:
    """Classify every page of a PDF. Lazy-imports PyMuPDF."""
    import pymupdf

    doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    try:
        return [classify_page(page_signals(doc[i], i + 1)) for i in range(len(doc))]
    finally:
        doc.close()  # type: ignore[no-untyped-call]


def table_candidate_pages(pdf_path: str) -> list[int]:
    """1-based page numbers safe to run table extraction on."""
    return [c.page for c in analyze_pages(pdf_path) if c.is_table_candidate]
