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

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog
from pydantic import BaseModel, Field

from doc_assistant import config
from doc_assistant.config import (
    FIGURE_CAPTION_DESC_MIN_CHARS,
    FIGURE_DIR,
    FIGURE_MIN_AREA_FRACTION,
)

from .regions import FIGURE_CAPTION_RE, RegionKind, analyze_pages

log = structlog.get_logger(__name__)

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


# ============================================================
# Feature 4c — VLM figure description (gated, Anthropic-only)
# ============================================================
# Turns a 4b `Figure` (caption + PNG crop) into a structured description, so
# `caption + description` becomes a retrievable `chunk_type='figure'` chunk.
# Cost-gated: only figures with a rendered PNG and a thin caption, under a
# per-doc budget (see `scripts/extract_figures` vs `scripts/describe_figures`).


class FigureDescription(BaseModel):
    """Schema-first VLM output for one figure (Anthropic tool-use validates it).

    Deliberately carries **no confidence field** — the project surfaces
    retrieval-derived uncertainty markers, never self-reported LLM confidence
    (see the roadmap's "What NOT to do"). ``axes`` / ``trend`` are nullable for
    non-plot figures (micrographs, diagrams, photos).
    """

    figure_type: str = Field(
        description="Kind of figure, e.g. 'bar chart', 'line plot', 'micrograph', "
        "'diagram', 'photo', 'schematic'."
    )
    summary: str = Field(
        description="1-3 sentence description of what the figure shows and conveys."
    )
    key_quantities: list[str] = Field(
        default_factory=list,
        description="Notable values, labels, or units visible in the figure (may be empty).",
    )
    axes: str | None = Field(
        default=None, description="x and y axis meaning if this is a plot, else null."
    )
    trend: str | None = Field(
        default=None, description="Main trend or relationship shown, if any, else null."
    )

    def to_text(self) -> str:
        """Render to a single natural-language string for embedding."""
        parts = [self.summary.strip(), f"Figure type: {self.figure_type.strip()}."]
        if self.axes:
            parts.append(f"Axes: {self.axes.strip()}.")
        if self.trend:
            parts.append(f"Trend: {self.trend.strip()}.")
        if self.key_quantities:
            quantities = "; ".join(q.strip() for q in self.key_quantities if q.strip())
            if quantities:
                parts.append(f"Key quantities: {quantities}.")
        return " ".join(p for p in parts if p).strip()


# Skip-reason vocabulary persisted to `Figure.vlm_call_skipped_reason`.
SKIP_NO_IMAGE = "no_image"
SKIP_CAPTION_SUFFICIENT = "caption_sufficient"
SKIP_BUDGET_EXHAUSTED = "budget_exhausted"
SKIP_IMAGE_MISSING = "image_missing"


def should_describe(
    caption: str | None,
    image_path: str | None,
    *,
    min_caption_chars: int = FIGURE_CAPTION_DESC_MIN_CHARS,
) -> tuple[bool, str | None]:
    """Gate one figure (pure). Returns ``(describe?, skip_reason | None)``.

    Skips a caption-only figure (no PNG to look at) and a figure whose caption is
    already long enough to be self-describing. The per-doc budget is enforced by
    the caller, not here.
    """
    if not image_path:
        return False, SKIP_NO_IMAGE
    if caption and len(caption.strip()) >= min_caption_chars >= 1:
        return False, SKIP_CAPTION_SUFFICIENT
    return True, None


def figure_chunk_text(caption: str | None, vlm_description: str) -> str:
    """Compose the retrievable text for a figure chunk (pure): caption + description."""
    cap = (caption or "").strip()
    desc = vlm_description.strip()
    if cap and desc:
        return f"{cap}\n\n{desc}"
    return desc or cap


# ---- The VLM call (impure boundary) ----------------------------------------

_FIGURE_TOOL_NAME = "record_figure_description"

_VLM_PROMPT = (
    "You are describing a figure cropped from a scientific paper, for a retrieval "
    "index. Describe ONLY what is visibly in the image — do not invent values. "
    "Use the caption for context but describe the figure itself.\n\n"
    "Caption: {caption}\n\n"
    "Call the {tool} tool with a structured description."
)


def figure_tool() -> dict[str, Any]:
    """The Anthropic tool definition whose input schema is ``FigureDescription``."""
    return {
        "name": _FIGURE_TOOL_NAME,
        "description": "Record a structured description of the scientific figure.",
        "input_schema": FigureDescription.model_json_schema(),
    }


def build_vlm_messages(image_b64: str, media_type: str, caption: str) -> list[dict[str, Any]]:
    """Build the Anthropic ``messages`` payload (pure): an image block + a text prompt."""
    prompt = _VLM_PROMPT.format(caption=caption or "(no caption)", tool=_FIGURE_TOOL_NAME)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def extract_tool_use_input(content: Any) -> dict[str, Any]:
    """Pull the ``tool_use`` block's ``input`` from a Messages response (pure).

    Tolerates both SDK block objects and plain dicts, mirroring
    ``llm._extract_anthropic_text``. Raises ``ValueError`` if no tool_use block.
    """
    for block in content or []:
        btype = getattr(block, "type", None)
        if btype is None and isinstance(block, dict):
            btype = block.get("type")
        if btype != "tool_use":
            continue
        data = getattr(block, "input", None)
        if data is None and isinstance(block, dict):
            data = block.get("input")
        if isinstance(data, dict):
            return data
    raise ValueError("no tool_use block in VLM response")


class FigureDescriber(Protocol):
    """A provider-agnostic vision describer (DI seam, mirrors ``llm.LLMClient``)."""

    def describe(
        self, *, image_b64: str, media_type: str, caption: str, model: str, max_tokens: int
    ) -> FigureDescription: ...


class AnthropicVisionDescriber:
    """``FigureDescriber`` over the raw Anthropic SDK (vision + forced tool-use).

    Forces ``tool_choice`` to the figure tool so the model must return a
    schema-shaped ``tool_use`` block; the result is validated by Pydantic. No
    vendor SDK import at module load — ``anthropic`` is imported lazily here.
    """

    def __init__(self, *, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        from doc_assistant.llm import os_trust_http_client

        kwargs: dict[str, Any] = {"api_key": api_key or config.ANTHROPIC_API_KEY}
        http_client = os_trust_http_client()
        if http_client is not None:  # OS-trust TLS for corporate MITM proxies (KI-10)
            kwargs["http_client"] = http_client
        self._client = Anthropic(**kwargs)
        self._tool = figure_tool()

    def describe(
        self, *, image_b64: str, media_type: str, caption: str, model: str, max_tokens: int
    ) -> FigureDescription:
        # Build kwargs as a plain dict and expand — same trick as
        # llm.AnthropicClient.complete — so the loose tool/tool_choice dicts
        # don't have to satisfy the SDK's typed-overload signatures.
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": [self._tool],
            "tool_choice": {"type": "tool", "name": _FIGURE_TOOL_NAME},
            "messages": build_vlm_messages(image_b64, media_type, caption),
        }
        response: Any = self._client.messages.create(**kwargs)
        return FigureDescription.model_validate(extract_tool_use_input(response.content))


def describe_figure(
    image_path: Path,
    caption: str | None,
    describer: FigureDescriber,
    *,
    model: str,
    max_tokens: int = 1024,
) -> FigureDescription:
    """Read a figure PNG, base64-encode it, and describe it via ``describer``. Impure."""
    image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return describer.describe(
        image_b64=image_b64,
        media_type="image/png",
        caption=caption or "",
        model=model,
        max_tokens=max_tokens,
    )


def load_figure_image_paths(figure_ids: list[str]) -> dict[str, str]:
    """Map ``figure_id`` → on-disk PNG path for the given figures (impure, DB read).

    Only figures whose ``image_path`` is set **and** the file still exists are
    returned — a caption-only figure (no rendered region) has no image to show. Used
    by the UI to render a retrieved ``chunk_type='figure'`` chunk's PNG as an image
    element; keeping the lookup here keeps ``apps/`` a thin shell."""
    if not figure_ids:
        return {}

    from sqlalchemy import select

    from doc_assistant.db.models import Figure
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        rows = session.execute(
            select(Figure.id, Figure.image_path).where(Figure.id.in_(figure_ids))
        ).all()
    return {str(fid): str(path) for fid, path in rows if path and Path(path).exists()}
