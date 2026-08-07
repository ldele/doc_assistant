"""The text-layer fallback in `extract_pdf_pymupdf` (EX1, 2026-08-07).

A scanned page with an invisible OCR text layer behind a full-page image is rendered by
PyMuPDF4LLM as a picture placeholder; KI-14's stripper then removes even that. Three documents on
the dev corpus lost 97-100% of their text this way and accounted for 6 of the 7 retrieval misses on
the private eval set — while their text layers were perfectly good. These pin the recovery.

`_recover_lost_page` takes the open document rather than a path precisely so it can be tested with
a stub, with no PDF fixture and no PyMuPDF4LLM call.
"""

from __future__ import annotations

import pytest

from doc_assistant.extractors import (
    _TEXT_LAYER_KEPT_MIN,
    _recover_lost_page,
    _substantive_len,
)


class _FakeDoc:
    """Minimal stand-in for a pymupdf Document: indexable, pages expose get_text()."""

    def __init__(self, *texts: str) -> None:
        self._texts = texts

    def __getitem__(self, i: int) -> _FakePage:
        return _FakePage(self._texts[i])


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


REAL_PAGE = (
    "RECEPTIVE FIELDS OF SINGLE NEURONES IN THE CAT'S STRIATE CORTEX "
    "By D. H. HUBEL AND T. N. WIESEL. In the unit of Fig. 1 the strongest inhibitory "
    "responses were obtained with a vertically oriented slit."
)


def test_placeholder_only_markdown_falls_back_to_the_text_layer() -> None:
    """The exact observed failure: the page renders as a picture placeholder."""
    page_md = "**==> picture [331 x 154] intentionally omitted <==**"
    out = _recover_lost_page(_FakeDoc(REAL_PAGE), 0, page_md)
    assert out == REAL_PAGE


def test_heading_scaffolding_only_falls_back() -> None:
    """`hebb_1949` p183 produced literally '## ## ##' from 2,190 characters of text."""
    out = _recover_lost_page(_FakeDoc(REAL_PAGE), 0, "## ## ##")
    assert out == REAL_PAGE


def test_empty_markdown_falls_back() -> None:
    """`hodgkin_huxley_1952` p23 produced nothing at all from 2,047 characters."""
    assert _recover_lost_page(_FakeDoc(REAL_PAGE), 0, "") == REAL_PAGE


def test_a_healthy_page_is_returned_untouched() -> None:
    """Markdown that kept the text must pass through byte-for-byte.

    Healthy pages measured 97.3-108.9% kept; this is the property that keeps the fallback off
    every one of the 93 healthy documents."""
    page_md = f"## Results\n\n{REAL_PAGE}\n"
    assert _recover_lost_page(_FakeDoc(REAL_PAGE), 0, page_md) == page_md


def test_markdown_that_adds_structure_is_not_treated_as_loss() -> None:
    """Ratios above 1.0 are normal — markdown adds headings and table pipes the raw text lacks."""
    page_md = f"# Title\n\n## Section\n\n{REAL_PAGE}\n\n| a | b |\n|---|---|\n"
    assert _recover_lost_page(_FakeDoc(REAL_PAGE), 0, page_md) == page_md


def test_a_page_with_no_text_layer_is_left_alone() -> None:
    """A true scan (`middleton-2001`: 0 characters on every page) is OCR territory, not this.

    Returning the markdown unchanged keeps the honest-absence behaviour: an empty page stays
    empty rather than being 'recovered' into nothing."""
    page_md = "**==> picture [1360 x 2464] intentionally omitted <==**"
    assert _recover_lost_page(_FakeDoc(""), 0, page_md) == page_md


def test_recovery_never_raises_when_the_text_layer_is_unreadable() -> None:
    """A recovery that throws would be worse than the gap it fixes."""

    class _Exploding:
        def __getitem__(self, i: int) -> object:
            raise RuntimeError("corrupt page tree")

    assert _recover_lost_page(_Exploding(), 0, "## ##") == "## ##"


@pytest.mark.parametrize(
    ("kept_fraction", "expect_fallback"),
    [(0.0, True), (0.2, True), (0.49, True), (0.8, False), (1.0, False)],
)
def test_threshold_selects_lost_pages_only(kept_fraction: float, expect_fallback: bool) -> None:
    """Pins the direction of the comparison — an inverted sign would silently discard markdown
    on every healthy page in the corpus."""
    raw = ("word " * 200).strip()  # the recovery returns the STRIPPED text layer
    keep = int(_substantive_len(raw) * kept_fraction)
    page_md = "x" * keep
    out = _recover_lost_page(_FakeDoc(raw), 0, page_md)
    assert (out == raw) is expect_fallback


def test_threshold_sits_between_the_measured_populations() -> None:
    """Healthy pages kept >=97.3%, lost pages <=3.2% (measured 2026-08-07). The constant is only
    defensible while it sits in that gap."""
    assert 0.032 < _TEXT_LAYER_KEPT_MIN < 0.973
