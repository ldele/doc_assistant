"""Guard tests for `is_page_scan` — a scanned page is not a figure.

`select_region_bboxes` had an area **floor** (so a logo never becomes a figure) and no
**ceiling**, so a scanned document — one full-page image per page — produced one "figure" per
page. Measured on the real corpus 2026-08-08: `hebb_1949.pdf` gave **365 figures for 365 pages**
(exactly 1.00/page, zero captions), and **46%** of all 1452 figure rows were page scans.

The cut is structural, not tuned: the area-fraction distribution is bimodal with an effectively
empty band — 783 rows below 0.7, **one** row in [0.7, 0.9), 669 at/above 0.9 — and 0.80/0.85/0.90
all partition it identically.

**The caption exemption was not enough, and these tests now say why (2026-08-09).** The original
rule kept any captioned full-page region, reasoning that a scan has no text layer and therefore no
caption to pair. A scan **with OCR** does: three documents here put a `Fig. N` block on every page,
so every page was stored as a figure — **109 of 962 rows were whole pages** (81 of 81 in one
scanned book), **55 of them already paid for at the VLM**. So a full-page region must now clear two
conditions: a caption *and* a page holding at most `FIGURE_PLATE_MAX_TEXT_LINES` lines of text.
Measured line counts: verified plates 4-13, pages of prose 25-117.
"""

from __future__ import annotations

import pytest

from doc_assistant.config import FIGURE_MAX_AREA_FRACTION, FIGURE_PLATE_MAX_TEXT_LINES
from doc_assistant.ingest.figures import BBox, is_page_scan

PAGE: BBox = (0.0, 0.0, 600.0, 800.0)


def region(fraction: float) -> BBox:
    """A bbox covering `fraction` of PAGE, anchored at the origin."""
    return (0.0, 0.0, 600.0, 800.0 * fraction)


# ---- the ceiling itself ----------------------------------------------------


def test_full_page_image_without_caption_is_a_page_scan() -> None:
    assert is_page_scan(region(1.0), PAGE, caption=None) is True


def test_small_captioned_figure_is_not_a_page_scan() -> None:
    assert is_page_scan(region(0.25), PAGE, caption="Figure 2. A circuit.") is False


def test_small_uncaptioned_figure_is_not_a_page_scan() -> None:
    # The ceiling must not become a general uncaptioned-figure filter — plenty of real
    # figures below the cut have no pairable caption.
    assert is_page_scan(region(0.25), PAGE, caption=None) is False


@pytest.mark.parametrize("fraction", [0.86, 0.9, 0.99, 1.0])
def test_at_or_above_the_ceiling_uncaptioned_is_rejected(fraction: float) -> None:
    assert is_page_scan(region(fraction), PAGE, caption=None) is True


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.7, 0.84])
def test_below_the_ceiling_is_always_kept(fraction: float) -> None:
    assert is_page_scan(region(fraction), PAGE, caption=None) is False


# ---- the caption exemption -------------------------------------------------


def test_full_page_plate_with_a_caption_is_kept() -> None:
    # A genuine plate: the page is the image, its caption and a running head. Rejecting on
    # area alone would trade one systematic error for another.
    assert (
        is_page_scan(region(1.0), PAGE, caption="Figure 7. Whole-page plate.", page_text_lines=5)
        is False
    )


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_a_whitespace_caption_does_not_exempt(blank: str) -> None:
    # An empty string is not a caption; it must not smuggle a page scan through.
    assert is_page_scan(region(1.0), PAGE, caption=blank, page_text_lines=3) is True


# ---- the second condition: the page must not be full of prose ----------------


def test_a_captioned_full_page_over_the_line_budget_is_a_scan() -> None:
    # The 2026-08-09 defect in one assertion: OCR gives a scanned page a "Fig. N" block, so the
    # caption alone waved through 81 pages of prose in a single scanned book.
    assert is_page_scan(region(1.0), PAGE, caption="Fig. 6.19.", page_text_lines=41) is True


@pytest.mark.parametrize("lines", [4, 5, 6, 13, FIGURE_PLATE_MAX_TEXT_LINES])
def test_plate_line_counts_seen_in_the_corpus_are_kept(lines: int) -> None:
    # Every value here is a page confirmed by eye to be a genuine full-page plate.
    assert (
        is_page_scan(region(1.0), PAGE, caption="Figure 5.11: …", page_text_lines=lines) is False
    )


@pytest.mark.parametrize("lines", [25, 41, 45, 84, 117])
def test_prose_line_counts_seen_in_the_corpus_are_rejected(lines: int) -> None:
    # And every value here is a page of prose that was being stored as a figure.
    assert is_page_scan(region(1.0), PAGE, caption="Figure 6.19.", page_text_lines=lines) is True


def test_the_line_budget_does_not_touch_a_sub_page_figure() -> None:
    # The second condition must stay scoped to full-page regions: a normal figure on a normal
    # page of prose is the common case and has nothing to do with page scans.
    assert (
        is_page_scan(region(0.3), PAGE, caption="Figure 2. A circuit.", page_text_lines=45)
        is False
    )


# ---- degenerate inputs -----------------------------------------------------


def test_caption_only_region_is_never_a_page_scan() -> None:
    # bbox is None only because a caption was found — by construction not a scan.
    assert is_page_scan(None, PAGE, caption="Figure 1.") is False
    assert is_page_scan(None, PAGE, caption=None) is False


def test_zero_area_page_does_not_divide_by_zero() -> None:
    degenerate: BBox = (0.0, 0.0, 0.0, 0.0)
    assert is_page_scan(region(1.0), degenerate, caption=None) is False


def test_region_larger_than_the_page_is_still_a_scan() -> None:
    # Image blocks can slightly exceed the page rect (bleed); that is more scan-like,
    # not less.
    oversized: BBox = (-10.0, -10.0, 610.0, 810.0)
    assert is_page_scan(oversized, PAGE, caption=None) is True


# ---- the constant ----------------------------------------------------------


def test_ceiling_sits_inside_the_measured_empty_band() -> None:
    # 783 rows below 0.7, one in [0.7, 0.9), 669 at/above 0.9. A cut outside that band
    # would start splitting a real population instead of separating two.
    assert 0.7 < FIGURE_MAX_AREA_FRACTION < 0.9


def test_ceiling_is_above_the_floor() -> None:
    from doc_assistant.config import FIGURE_MIN_AREA_FRACTION

    assert FIGURE_MIN_AREA_FRACTION < FIGURE_MAX_AREA_FRACTION


def test_line_budget_sits_between_the_two_measured_populations() -> None:
    # Plates ran 4-13 lines and pages of prose 25-117 (measured 2026-08-09 over all 109 full-page
    # rows). A budget outside that band would start splitting one of the two populations.
    assert 13 < FIGURE_PLATE_MAX_TEXT_LINES < 25
