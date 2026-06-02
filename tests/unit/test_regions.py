"""Tests for the page content classifier (``doc_assistant.regions``).

The pure router (``classify_page``) is covered exhaustively against the
routing matrix. ``analyze_pages`` / ``table_candidate_pages`` are exercised
with a monkeypatched PyMuPDF document so no real PDF is needed.
"""

from __future__ import annotations

from typing import Any

from doc_assistant.regions import (
    CHART_CURVE_MIN,
    IMAGE_AREA_MIN,
    PageSignals,
    analyze_pages,
    classify_page,
    table_candidate_pages,
)


def _sig(
    *,
    page: int = 1,
    curves: int = 0,
    image_frac: float = 0.0,
    table_cap: bool = False,
    figure_cap: bool = False,
) -> PageSignals:
    return PageSignals(
        page=page,
        curve_count=curves,
        image_area_fraction=image_frac,
        has_table_caption=table_cap,
        has_figure_caption=figure_cap,
    )


# ============================================================
# classify_page — the routing matrix (pure)
# ============================================================


def test_table_page():
    c = classify_page(_sig(table_cap=True, curves=43))
    assert c.kind == "table"
    assert c.is_table_candidate is True
    assert c.is_figure is False


def test_chart_page_by_curves():
    c = classify_page(_sig(curves=CHART_CURVE_MIN + 5, figure_cap=True))
    assert c.kind == "chart"
    assert c.is_table_candidate is False
    assert c.is_figure is True


def test_photo_page_by_image_area():
    c = classify_page(_sig(image_frac=IMAGE_AREA_MIN + 0.1, figure_cap=True))
    assert c.kind == "photo"
    assert c.is_table_candidate is False
    assert c.is_figure is True


def test_figure_caption_only_page():
    """A figure with few curves and no big raster image (caption is the signal)."""
    c = classify_page(_sig(figure_cap=True, curves=187))
    assert c.kind == "figure"
    assert c.is_figure is True
    assert c.is_table_candidate is False


def test_plain_text_page():
    c = classify_page(_sig())
    assert c.kind == "text"
    assert c.is_table_candidate is False
    assert c.is_figure is False


def test_table_caption_but_chart_curves_is_not_table():
    """A chart page with a stray 'Table' mention must not be a table candidate."""
    c = classify_page(_sig(table_cap=True, curves=CHART_CURVE_MIN + 1))
    assert c.is_table_candidate is False
    assert c.kind == "chart"


def test_table_caption_but_large_image_is_not_table():
    c = classify_page(_sig(table_cap=True, image_frac=IMAGE_AREA_MIN + 0.2))
    assert c.is_table_candidate is False
    assert c.kind == "photo"


def test_thresholds_are_inclusive_boundaries():
    # Exactly at the chart threshold counts as a chart.
    assert classify_page(_sig(curves=CHART_CURVE_MIN)).kind == "chart"
    # Exactly at the image threshold counts as a photo.
    assert classify_page(_sig(image_frac=IMAGE_AREA_MIN)).kind == "photo"
    # Just below both, with a table caption, is a table.
    assert classify_page(
        _sig(table_cap=True, curves=CHART_CURVE_MIN - 1, image_frac=IMAGE_AREA_MIN - 0.001)
    ).is_table_candidate


# ============================================================
# analyze_pages / table_candidate_pages (monkeypatched pymupdf)
# ============================================================


class _FakeRect:
    width = 100.0
    height = 100.0


class _FakeMuPage:
    """Mimics the PyMuPDF page surface used by ``page_signals``."""

    rect = _FakeRect()

    def __init__(
        self,
        text: str,
        *,
        curves: int = 0,
        image_bboxes: list[tuple[float, float, float, float]] | None = None,
    ) -> None:
        self._text = text
        self._curves = curves
        self._image_bboxes = image_bboxes or []

    def get_text(self, kind: str | None = None) -> Any:
        if kind == "dict":
            blocks: list[dict[str, Any]] = [{"type": 0}]
            for bbox in self._image_bboxes:
                blocks.append({"type": 1, "bbox": bbox})
            return {"blocks": blocks}
        return self._text

    def get_drawings(self) -> list[dict[str, Any]]:
        return [{"items": [("c", None) for _ in range(self._curves)]}]


class _FakeMuDoc:
    def __init__(self, pages: list[_FakeMuPage]) -> None:
        self._pages = pages

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, i: int) -> _FakeMuPage:
        return self._pages[i]

    def close(self) -> None:
        return None


def _patch_pymupdf(monkeypatch: Any, pages: list[_FakeMuPage]) -> None:
    import pymupdf

    monkeypatch.setattr(pymupdf, "open", lambda _path: _FakeMuDoc(pages))


def test_analyze_pages_classifies_each_page(monkeypatch: Any):
    pages = [
        _FakeMuPage("Table 1. real data", curves=40),  # table
        _FakeMuPage("Figure 3. charts", curves=CHART_CURVE_MIN + 100),  # chart
        _FakeMuPage("Figure 2. micrographs", image_bboxes=[(0, 0, 60, 60)]),  # photo (36% area)
        _FakeMuPage("just prose"),  # text
    ]
    _patch_pymupdf(monkeypatch, pages)
    out = analyze_pages("ignored.pdf")
    assert [c.kind for c in out] == ["table", "chart", "photo", "text"]
    assert [c.page for c in out] == [1, 2, 3, 4]


def test_table_candidate_pages_filters_to_tables(monkeypatch: Any):
    pages = [
        _FakeMuPage("Figure 3. charts", curves=CHART_CURVE_MIN + 100),
        _FakeMuPage("Table 1. data", curves=40),
        _FakeMuPage("Table 2. data", curves=10),
    ]
    _patch_pymupdf(monkeypatch, pages)
    assert table_candidate_pages("ignored.pdf") == [2, 3]


def test_image_area_fraction_ignores_tiny_images(monkeypatch: Any):
    # A small image (4% of a 100x100 page) stays below IMAGE_AREA_MIN → not a photo.
    pages = [_FakeMuPage("Table 1. data", curves=10, image_bboxes=[(0, 0, 20, 20)])]
    _patch_pymupdf(monkeypatch, pages)
    out = analyze_pages("ignored.pdf")
    assert out[0].kind == "table"
    assert out[0].is_table_candidate
