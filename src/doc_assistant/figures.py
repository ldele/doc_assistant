"""Figure region detection + caption pairing (Phase 6 / Feature 4b).

A post-ingest enrichment layer (see "Enrichment-Layer Pattern" in
``decisions.md``). The primary extractor drops figures entirely
(``write_images=False``); this module promotes ``regions.py``'s *page-level*
figure verdict to *region-level*: find the figure bbox(es) on each figure
page, pair each with its caption, and hand the caller everything it needs to
crop a PNG and persist a ``Figure`` sidecar row.

Why sidecar, never spliced (ADR-2): figures are binary. Embedding base64 in
the markdown destroys the human-readable cache; placeholder strings without
the image are noise. So the caption text stays in the markdown untouched and
the image persists as a ``Figure`` row + a PNG under ``data/figures/``.

Region geometry (ADR-1): bboxes come from PyMuPDF geometry only — raster
image blocks for photos, the drawing-rect union for vector charts, the
largest non-text block (else caption-only) otherwise. The chart/photo/figure
*discrimination* OpenCV was once slated for is already solved by
``regions.py``'s measured signals, so OpenCV stays out of the v1 hot path; it
is a deferred refinement lever, not a dependency.

Design split (mirrors ``regions.py``): a pure core — ``pair_caption``,
``select_region_bboxes``, ``figure_image_path`` — exhaustively unit-testable
with no I/O, behind a thin impure PyMuPDF boundary (``detect_figure_regions``,
``render_region``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc_assistant.config import FIGURE_DIR, FIGURE_MIN_AREA_FRACTION
from doc_assistant.regions import FIGURE_CAPTION_RE, RegionKind, analyze_pages

log = logging.getLogger(__name__)

# A bbox is (x0, y0, x1, y1) in PDF points; y increases downward.
BBox = tuple[float, float, float, float]


@dataclass
class FigureRegion:
    """One detected figure region. Pure data — no I/O, no DB.

    ``bbox`` / ``caption_bbox`` are ``None`` for a caption-only figure (a
    figure-captioned page with no detectable region). ``extraction_method``
    records which ADR-1 branch produced the region: ``image_block`` |
    ``drawing_union`` | ``largest_block`` | ``caption_only``.
    """

    page: int  # 1-based
    bbox: BBox | None
    kind: RegionKind
    caption: str | None
    caption_bbox: BBox | None
    extraction_method: str


# ============================================================
# Pure core — geometry & caption pairing (no I/O)
# ============================================================


# A region must span at least this many PDF points in BOTH dimensions to be
# croppable. PyMuPDF's PNG writer raises "Invalid bandwriter header dimensions"
# on a clip that rounds to zero pixels, and an inverted/degenerate rect (x1<=x0)
# would sneak past an `abs()`-based area check — so renderability is tested on
# the signed width/height, not the area.
MIN_REGION_DIM = 1.0


def _area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return abs((x1 - x0) * (y1 - y0))


def _is_renderable(bbox: BBox) -> bool:
    """True if ``bbox`` has positive, non-degenerate width and height."""
    return (bbox[2] - bbox[0]) >= MIN_REGION_DIM and (bbox[3] - bbox[1]) >= MIN_REGION_DIM


def _union(rects: list[BBox]) -> BBox:
    """Bounding rectangle of one-or-more bboxes."""
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _clip(bbox: BBox, page_bbox: BBox) -> BBox:
    """Intersect ``bbox`` with the page rectangle."""
    return (
        max(bbox[0], page_bbox[0]),
        max(bbox[1], page_bbox[1]),
        min(bbox[2], page_bbox[2]),
        min(bbox[3], page_bbox[3]),
    )


def _vertical_gap(region_bbox: BBox, caption_bbox: BBox) -> float:
    """Vertical distance between a region and a caption block (0 if overlapping)."""
    _, ry0, _, ry1 = region_bbox
    _, cy0, _, cy1 = caption_bbox
    if cy0 >= ry1:  # caption fully below the region
        return cy0 - ry1
    if cy1 <= ry0:  # caption fully above the region
        return ry0 - cy1
    return 0.0


def select_region_bboxes(
    image_bboxes: list[BBox],
    drawing_rects: list[BBox],
    *,
    kind: RegionKind,
    page_bbox: BBox,
    min_area_fraction: float = FIGURE_MIN_AREA_FRACTION,
) -> list[tuple[BBox | None, str]]:
    """Choose figure-region bboxes from a page's geometry (ADR-1). Pure.

    Returns ``(bbox, extraction_method)`` pairs in reading order. The branches,
    in priority order:

    * **Raster** — any image block at/above ``min_area_fraction`` of the page
      becomes its own region (``image_block``). Sub-threshold blocks (logos,
      icons) and degenerate slivers are dropped here.
    * **Chart** — a ``kind == "chart"`` page with no qualifying image block
      yields one merged region: the clipped union of its drawing rects
      (``drawing_union``), kept only if it clears the area floor and is croppable.
    * **Fallback** — otherwise the largest non-text block (image or drawing)
      above the floor (``largest_block``); if nothing qualifies, a single
      ``(None, "caption_only")`` so a figure-captioned page still records a row.

    Every returned bbox is renderable (``_is_renderable``) — a region too thin
    to crop is never emitted, so the impure render path can't fault on it.
    """
    page_area = _area(page_bbox) or 1.0

    def frac(bb: BBox) -> float:
        return _area(bb) / page_area

    def keep(bb: BBox) -> bool:
        return _is_renderable(bb) and frac(bb) >= min_area_fraction

    big_images = sorted(
        (bb for bb in image_bboxes if keep(bb)),
        key=lambda bb: (bb[1], bb[0]),  # reading order: top-down, then left-right
    )
    if big_images:
        return [(bb, "image_block") for bb in big_images]

    if kind == "chart" and drawing_rects:
        union = _clip(_union(drawing_rects), page_bbox)
        if keep(union):
            return [(union, "drawing_union")]

    candidates = [bb for bb in (*image_bboxes, *drawing_rects) if _is_renderable(bb)]
    if candidates:
        largest = max(candidates, key=_area)
        if frac(largest) >= min_area_fraction:
            return [(largest, "largest_block")]

    return [(None, "caption_only")]


def pair_caption(
    region_bbox: BBox,
    caption_blocks: list[tuple[str, BBox]],
) -> tuple[str, BBox] | None:
    """Pair a region with its nearest caption block. Pure — the heart of 4b.

    ``caption_blocks`` are the ``(text, bbox)`` pairs for blocks matching
    ``FIGURE_CAPTION_RE``. The nearest by vertical gap wins; ties prefer a
    caption *below* the figure (the usual layout). Returns the chosen
    ``(text, bbox)`` (an element of ``caption_blocks``) or ``None`` when the
    list is empty.
    """
    best: tuple[str, BBox] | None = None
    best_key: tuple[float, int] | None = None
    for text, cbbox in caption_blocks:
        gap = _vertical_gap(region_bbox, cbbox)
        is_above = 1 if cbbox[3] <= region_bbox[1] else 0  # below-preferred tiebreak
        key = (gap, is_above)
        if best_key is None or key < best_key:
            best_key = key
            best = (text, cbbox)
    return best


def figure_dir(doc_hash: str) -> Path:
    """The per-document figure directory: ``FIGURE_DIR / doc_hash``."""
    return FIGURE_DIR / doc_hash


def figure_image_path(doc_hash: str, page: int, index: int) -> Path:
    """Stable on-disk path for a region's PNG. Pure → idempotent filenames.

    ``index`` is the 0-based region index in reading order on the page.
    """
    return figure_dir(doc_hash) / f"page{page}_fig{index}.png"


# ============================================================
# Impure boundary — opens the PDF via PyMuPDF
# ============================================================


def _block_text(block: dict[str, Any]) -> str:
    """Reconstruct a text block's string from its spans."""
    lines: list[str] = []
    for line in block.get("lines", []):
        lines.append("".join(span.get("text", "") for span in line.get("spans", [])))
    return "\n".join(lines)


def _caption_blocks(page: object) -> list[tuple[str, BBox]]:
    """Text blocks on a page whose text reads as a figure caption."""
    raw = page.get_text("dict")  # type: ignore[attr-defined]
    out: list[tuple[str, BBox]] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        text = _block_text(block)
        if FIGURE_CAPTION_RE.search(text):
            caption = re.sub(r"\s+", " ", text).strip()
            out.append((caption, tuple(float(v) for v in block["bbox"])))  # type: ignore[arg-type]
    return out


def _page_geometry(page: object) -> tuple[list[BBox], list[BBox], BBox]:
    """Pull ``(image_bboxes, drawing_rects, page_bbox)`` from a PyMuPDF page."""
    raw = page.get_text("dict")  # type: ignore[attr-defined]
    image_bboxes: list[BBox] = [
        tuple(float(v) for v in b["bbox"])  # type: ignore[misc]
        for b in raw.get("blocks", [])
        if b.get("type") == 1  # raster image block
    ]
    drawing_rects: list[BBox] = []
    for drawing in page.get_drawings():  # type: ignore[attr-defined]
        rect = drawing.get("rect")
        if rect is None:
            continue
        bb: BBox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        # Drop degenerate paths (lines, rules, off-page/inverted rects); a
        # signed-dimension check, since `_area` would let an inverted rect pass.
        if _is_renderable(bb):
            drawing_rects.append(bb)
    pr = page.rect  # type: ignore[attr-defined]
    page_bbox: BBox = (float(pr.x0), float(pr.y0), float(pr.x1), float(pr.y1))
    return image_bboxes, drawing_rects, page_bbox


def detect_figure_regions(pdf_path: str) -> list[FigureRegion]:
    """Detect figure regions across a PDF, paired with captions. Impure.

    Gates to ``is_figure`` pages via ``regions.analyze_pages`` (no second
    detector), derives region bboxes per ADR-1, and pairs each region with its
    nearest unused caption on the page. Returns regions in (page, reading)
    order. A document with no figure page yields ``[]``.
    """
    import pymupdf

    classifications = analyze_pages(pdf_path)
    kind_by_page = {c.page: c.kind for c in classifications}
    figure_pages = sorted(c.page for c in classifications if c.is_figure)
    if not figure_pages:
        return []

    out: list[FigureRegion] = []
    doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    try:
        for page_no in figure_pages:
            page = doc[page_no - 1]
            kind = kind_by_page[page_no]
            image_bboxes, drawing_rects, page_bbox = _page_geometry(page)
            candidates = select_region_bboxes(
                image_bboxes, drawing_rects, kind=kind, page_bbox=page_bbox
            )
            remaining = _caption_blocks(page)

            for bbox, method in candidates:
                caption: str | None = None
                caption_bbox: BBox | None = None
                if bbox is None:
                    # caption-only: attach the figure caption that triggered detection
                    if remaining:
                        caption, caption_bbox = remaining.pop(0)
                else:
                    pair = pair_caption(bbox, remaining)
                    if pair is not None:
                        caption, caption_bbox = pair
                        remaining.remove(pair)
                out.append(
                    FigureRegion(
                        page=page_no,
                        bbox=bbox,
                        kind=kind,
                        caption=caption,
                        caption_bbox=caption_bbox,
                        extraction_method=method,
                    )
                )
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return out


def render_region(page: object, bbox: BBox, out_path: Path, *, dpi: int) -> None:
    """Crop ``bbox`` from ``page`` to a PNG at ``out_path``. Impure."""
    import pymupdf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rect = pymupdf.Rect(bbox)  # type: ignore[no-untyped-call]
    pix = page.get_pixmap(clip=rect, dpi=dpi)  # type: ignore[attr-defined]
    pix.save(str(out_path))
