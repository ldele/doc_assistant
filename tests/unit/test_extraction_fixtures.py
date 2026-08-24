"""Extraction checked against **committed document files**, not library round-trips (2026-08-20).

`test_extractors_formats.py` builds each fixture with the same library that reads it back, and says
in its own docstring that this cannot prove anything about files from *other* producers. These
tests close that gap for the two formats where the structure is richest: a real `.epub` on disk
(`tests/fixtures/documents/treatise.epub`) and a hand-authored `.html` article. Both are frozen
artifacts — regenerate the EPUB only via `make_fixtures.py`, and only when you mean to change what
is asserted here.

**These fixtures found three real extraction defects, all now FIXED** (2026-08-20, same day).
They were first pinned as `xfail(strict=True)` while the fix was still a decision; each flipped to
a failing XPASS the moment `_soup_to_markdown` landed, which is precisely what a strict xfail is
for. They are plain regression guards below. Fixing them was near-free because the blast radius
was **zero** — the corpus was 97/97 PDF, and only EPUB and HTML reach this code path, so no
document needed re-ingesting (ADR-042).

The three, kept here because the fixtures exist to keep them fixed:

1. **EPUB emits its own navigation document as prose.** `get_items_of_type(ITEM_DOCUMENT)` returns
   the generated `nav.xhtml` alongside real chapters, so a book's markdown ends with its table of
   contents rendered as body text.
2. **HTML leaks `<head><title>`.** Only script/style/nav/footer are decomposed, so `get_text()`
   emits the page title as a bare first line, indistinguishable from prose.
3. **Inline markup fragments sentences.** `get_text(separator="\\n")` puts a newline at every tag
   boundary, so `Emphasis <em>inside</em> a sentence` becomes three lines. This is the one with
   teeth: scientific prose italicises constantly (gene names, species, emphasis), so this shatters
   sentences across the corpus and degrades both the embedding and the BM25 token stream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.extractors import extract_epub, extract_html, extract_to_markdown

DOCUMENTS = Path(__file__).resolve().parents[1] / "fixtures" / "documents"
EPUB = DOCUMENTS / "treatise.epub"
HTML = DOCUMENTS / "article.html"


@pytest.fixture(scope="module")
def epub_md() -> str:
    return extract_epub(EPUB)


@pytest.fixture(scope="module")
def html_md() -> str:
    return extract_html(HTML)


# ============================================================
# The fixtures themselves
# ============================================================


def test_both_fixture_documents_are_present() -> None:
    """A missing fixture must fail loudly here rather than as a confusing error in every test."""
    assert EPUB.is_file(), f"{EPUB} missing — regenerate with make_fixtures.py"
    assert HTML.is_file(), f"{HTML} missing — it is hand-authored, restore it from git"


def test_the_epub_is_a_real_zip_container() -> None:
    """Guards the commit itself: a text-mode checkout would corrupt the binary silently."""
    assert EPUB.read_bytes()[:2] == b"PK"


# ============================================================
# EPUB — what must survive
# ============================================================


def test_epub_leads_with_the_dublin_core_title(epub_md: str) -> None:
    assert epub_md.startswith("# A Treatise on Cortical Microcircuits")


def test_epub_keeps_every_chapter(epub_md: str) -> None:
    assert "Cortical Microcircuits" in epub_md
    assert "Closing paragraph of the second chapter." in epub_md


def test_epub_maps_heading_levels_rather_than_flattening_them(epub_md: str) -> None:
    assert "# Cortical Microcircuits" in epub_md
    assert "## Méthodes" in epub_md
    assert "### A third-level heading" in epub_md


@pytest.mark.parametrize("text", ["Méthodes", "32°C", "Ramón y Cajal", "Naïve", "Résultats"])
def test_epub_decodes_html_entities_and_keeps_accents(epub_md: str, text: str) -> None:
    """The fixture stores these as `&#233;`-style entities, so arriving decoded is the contract.

    A literal `&#233;` reaching the chunk store would be indexed as a token nobody searches for.
    """
    assert text in epub_md


def test_epub_keeps_block_quotes_and_list_items_as_text(epub_md: str) -> None:
    assert "A quoted passage from Ramón y Cajal." in epub_md
    assert "First listed item" in epub_md
    assert "Second listed item" in epub_md


# ============================================================
# HTML — what must survive
# ============================================================


def test_html_keeps_the_article_headings(html_md: str) -> None:
    assert "# Dendritic Integration in Layer Five" in html_md
    assert "## Abstract" in html_md
    assert "### Statistical treatment" in html_md


@pytest.mark.parametrize(
    "text",
    ["Renée Delacroix", "Björn Åkesson", "32°C", "naïve", "Café", "mean ± s.e.m.", "p < 0.05"],
)
def test_html_decodes_entities_and_keeps_accents(html_md: str, text: str) -> None:
    assert text in html_md


def test_html_keeps_table_cell_text_even_though_it_loses_the_table(html_md: str) -> None:
    """Table *structure* is Marker's job (docs/figures-and-tables.md); the values must still be
    retrievable as text in the meantime, or the numbers vanish from the corpus entirely."""
    assert "Resting potential" in html_md
    # The fixture uses the real `&minus;` a journal would emit (U+2212), but a literal one in this
    # file is exactly what ruff's RUF001 confusables rule rejects — so name it by codepoint.
    minus = chr(0x2212)
    assert f"{minus}65 mV" in html_md
    assert "42 MΩ" in html_md


def test_html_keeps_the_reference_list(html_md: str) -> None:
    assert "Cajal, S. R. (1899)" in html_md


@pytest.mark.parametrize(
    ("marker", "why"),
    [
        ("NAVIGATION_CHROME_MARKER", "<nav> is decomposed"),
        ("FOOTER_CHROME_MARKER", "<footer> is decomposed"),
        ("SCRIPT_BODY_MARKER", "<script> is decomposed"),
        ("Georgia, serif", "<style> is decomposed"),
    ],
)
def test_html_discards_page_chrome(html_md: str, marker: str, why: str) -> None:
    """Unique markers, so a failure names which tag survived rather than just 'something did'."""
    assert marker not in html_md, why


def test_html_keeps_link_text_while_dropping_the_href(html_md: str) -> None:
    assert "reference link" in html_md
    assert "#ref1" not in html_md


# ============================================================
# The three defects these fixtures found — FIXED 2026-08-20, now regression guards
# ============================================================
#
# Each of these arrived as `xfail(strict=True)` while the defect stood, and each flipped to a
# failing XPASS the moment `_soup_to_markdown` landed — which is exactly what the tripwire was
# for. They are plain assertions now. The blast radius was zero: the corpus was 97/97 PDF, and
# only EPUB and HTML reach this code path, so nothing needed re-ingesting (ADR-042).


def test_epub_does_not_emit_its_navigation_document_as_prose(epub_md: str) -> None:
    """The book's own TOC should not be indexed as content.

    The tell is the chapter titles reappearing as bare trailing lines after the last chapter's
    closing paragraph, under a heading repeating the book title.
    """
    body, _, tail = epub_md.rpartition("Closing paragraph of the second chapter.")
    assert body, "fixture changed — the last chapter no longer ends where this test expects"
    assert "A Treatise on Cortical Microcircuits" not in tail


def test_html_does_not_leak_the_page_title_into_the_body(html_md: str) -> None:
    """`<title>` is site furniture: it carries the journal name and an em dash, and it lands
    above the real `<h1>` where nothing downstream can tell it from an opening sentence."""
    assert "Journal of Synthetic Neuroscience" not in html_md


@pytest.mark.parametrize("fixture_name", ["epub_md", "html_md"])
def test_inline_markup_does_not_split_a_sentence(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    """The most damaging of the three, and it affects both formats.

    Scientific prose italicises constantly — gene names, species, emphasis — so every such
    sentence reaches the chunk store broken across lines. That degrades the embedding *and* the
    BM25 token stream, and no health check would ever flag it: the text is all still there.

    Asserted **per line**, deliberately. Normalising the newlines away first rejoins the
    fragments and the test passes against the broken output — which is exactly what the first
    draft of this test did, and what `strict=True` caught.
    """
    md: str = request.getfixturevalue(fixture_name)
    both_fixtures_say = ("must not", "split")
    assert any(all(part in line for part in both_fixtures_say) for line in md.splitlines()), (
        "the sentence is broken across lines at the <strong> boundary"
    )


# ============================================================
# Through the real entry point
# ============================================================


@pytest.mark.parametrize("path", [EPUB, HTML], ids=["epub", "html"])
def test_the_fixtures_extract_through_extract_to_markdown(path: Path) -> None:
    """Everything above calls the format function directly; ingest calls the dispatcher."""
    assert len(extract_to_markdown(path).strip()) > 200
