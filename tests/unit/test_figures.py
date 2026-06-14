"""Tests for the pure core of ``doc_assistant.figures`` (Feature 4b).

The caption pairer (``pair_caption``), the ADR-1 region chooser
(``select_region_bboxes``) and the path builder (``figure_image_path``) are
exercised directly with synthetic geometry — no PDF, no DB. The impure
boundary (``detect_figure_regions`` / ``render_region`` / the CLI) is covered
by ``tests/integration/test_figures_extract.py``.
"""

from __future__ import annotations

from doc_assistant.figures import (
    figure_image_path,
    pair_caption,
    select_region_bboxes,
)

# A region near the top of a 100x100 page; y increases downward.
REGION = (10.0, 10.0, 90.0, 60.0)


# ============================================================
# pair_caption — the heart of the feature
# ============================================================


def test_caption_below_chosen_when_nearer():
    below_near = ("Figure 1: below", (10.0, 62.0, 90.0, 72.0))  # gap 2
    above_far = ("Figure 2: above", (10.0, 0.0, 90.0, 2.0))  # gap 8
    assert pair_caption(REGION, [below_near, above_far]) == below_near


def test_caption_above_chosen_when_nearer():
    # A region lower on the page, with the nearest caption above it.
    region = (10.0, 40.0, 90.0, 90.0)
    above_near = ("Figure 3: above", (10.0, 30.0, 90.0, 38.0))  # gap 2
    below_far = ("Figure 4: below", (10.0, 95.0, 90.0, 99.0))  # gap 5
    assert pair_caption(region, [below_far, above_near]) == above_near


def test_tie_prefers_caption_below():
    below = ("Figure 1: below", (10.0, 62.0, 90.0, 72.0))  # gap 2
    above = ("Figure 2: above", (10.0, 0.0, 90.0, 8.0))  # gap 2 (tie)
    assert pair_caption(REGION, [above, below]) == below


def test_adjacent_caption_has_zero_gap_and_pairs():
    # Caption top exactly at the region bottom => gap 0, still the nearest.
    adjacent = ("Figure 1: flush", (10.0, 60.0, 90.0, 70.0))
    far = ("Figure 2: far", (10.0, 90.0, 90.0, 99.0))
    assert pair_caption(REGION, [far, adjacent]) == adjacent


def test_no_captions_returns_none():
    assert pair_caption(REGION, []) is None


def test_two_regions_two_captions_no_double_assignment():
    """The caller protocol: pair nearest, drop it, pair the next region."""
    region0 = (10.0, 10.0, 90.0, 50.0)
    region1 = (10.0, 120.0, 90.0, 160.0)
    cap0 = ("Figure 1", (10.0, 52.0, 90.0, 60.0))  # near region0
    cap1 = ("Figure 2", (10.0, 162.0, 90.0, 170.0))  # near region1

    captions = [cap0, cap1]
    first = pair_caption(region0, captions)
    assert first == cap0
    captions.remove(first)
    second = pair_caption(region1, captions)
    assert second == cap1
    assert first != second


# ============================================================
# figure_image_path — stable, idempotent filenames
# ============================================================


def test_figure_image_path_shape():
    p = figure_image_path("abc123", 3, 0)
    assert p.name == "page3_fig0.png"
    assert p.parent.name == "abc123"


def test_figure_image_path_is_stable():
    assert figure_image_path("abc", 3, 0) == figure_image_path("abc", 3, 0)


def test_figure_image_path_distinct_per_index_and_page():
    assert figure_image_path("abc", 3, 0) != figure_image_path("abc", 3, 1)
    assert figure_image_path("abc", 3, 0) != figure_image_path("abc", 4, 0)
    assert figure_image_path("abc", 3, 0) != figure_image_path("xyz", 3, 0)


# ============================================================
# select_region_bboxes — ADR-1 geometry chooser
# ============================================================

PAGE = (0.0, 0.0, 100.0, 100.0)  # area 10000; default floor 0.02 => area 200


def test_image_block_kept_subthreshold_dropped():
    big = (10.0, 10.0, 30.0, 30.0)  # area 400, frac 0.04 -> kept
    small = (50.0, 50.0, 58.0, 58.0)  # area 64, frac 0.0064 -> dropped
    out = select_region_bboxes([big, small], [], kind="photo", page_bbox=PAGE)
    assert out == [(big, "image_block")]


def test_multiple_images_in_reading_order():
    lower = (10.0, 60.0, 40.0, 90.0)
    upper = (10.0, 10.0, 40.0, 40.0)
    out = select_region_bboxes([lower, upper], [], kind="photo", page_bbox=PAGE)
    assert [bb for bb, _ in out] == [upper, lower]  # top-down


def test_chart_uses_drawing_union():
    rects = [(10.0, 10.0, 40.0, 40.0), (50.0, 50.0, 80.0, 80.0)]
    out = select_region_bboxes([], rects, kind="chart", page_bbox=PAGE)
    assert out == [((10.0, 10.0, 80.0, 80.0), "drawing_union")]


def test_figure_page_largest_block_fallback():
    # No image, not a chart, but a sizeable drawing => largest_block.
    rect = (10.0, 10.0, 40.0, 40.0)  # area 900, frac 0.09
    out = select_region_bboxes([], [rect], kind="figure", page_bbox=PAGE)
    assert out == [(rect, "largest_block")]


def test_caption_only_when_nothing_qualifies():
    out = select_region_bboxes([], [], kind="figure", page_bbox=PAGE)
    assert out == [(None, "caption_only")]


def test_lone_subthreshold_image_falls_back_to_caption_only():
    tiny = (0.0, 0.0, 14.0, 14.0)  # area 196, frac 0.0196 < 0.02
    out = select_region_bboxes([tiny], [], kind="figure", page_bbox=PAGE)
    assert out == [(None, "caption_only")]


def test_inverted_rect_is_not_emitted():
    # x1 < x0: an `abs()` area check would pass it, but it can't be cropped.
    inverted = (80.0, 10.0, 10.0, 60.0)  # |area| = 70*50 = 3500
    out = select_region_bboxes([], [inverted], kind="chart", page_bbox=PAGE)
    assert out == [(None, "caption_only")]


def test_zero_height_sliver_is_not_emitted():
    # A horizontal rule masquerading as a drawing: no croppable height.
    sliver = (10.0, 50.0, 90.0, 50.0)
    out = select_region_bboxes([], [sliver], kind="figure", page_bbox=PAGE)
    assert out == [(None, "caption_only")]
