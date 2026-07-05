"""Tests for document-level metadata extraction (Phase 4)."""

from doc_assistant.metadata_extractor import (
    _arxiv_year_from_filename,
    _extract_doi,
    _extract_title,
    _extract_year,
    _looks_like_author_line,
    extract_metadata,
)

# ============================================================
# Title
# ============================================================


def test_title_picks_first_real_heading():
    md = "## RESEARCH ARTICLE\n\n## **A real paper title**\n\nbody"
    assert _extract_title(md) == "A real paper title"


def test_title_prefers_h1_over_h2():
    """Some PDFs put a journal-citation H2 before the real H1 title."""
    md = (
        "## J. Physiol. (1952) 117, 500-544\n\n"
        "# A QUANTITATIVE DESCRIPTION OF MEMBRANE CURRENT\n\n"
        "body"
    )
    assert _extract_title(md) == "A QUANTITATIVE DESCRIPTION OF MEMBRANE CURRENT"


def test_title_skips_short_headings():
    md = "## OK\n\n## A substantial title\n"
    assert _extract_title(md) == "A substantial title"


# ============================================================
# DOI
# ============================================================


def test_doi_from_url():
    head = "see https://doi.org/10.7554/eLife.04250.001 for details"
    assert _extract_doi(head) == "10.7554/eLife.04250.001"


def test_doi_bare():
    assert _extract_doi("DOI: 10.1038/nrn3901") == "10.1038/nrn3901"


def test_doi_none():
    assert _extract_doi("no doi here") is None


# ============================================================
# Year
# ============================================================


def test_year_published_keyword():
    assert _extract_year("Published: 04 November 2022. Body.") == 2022


def test_year_parens():
    assert _extract_year("Some text (1996) more text") == 1996


def test_year_loose_fallback():
    assert _extract_year("just a year 1973 in prose") == 1973


def test_arxiv_year_from_filename():
    assert _arxiv_year_from_filename("1707.01836v1.pdf") == 2017
    assert _arxiv_year_from_filename("2403.01590v1.md") == 2024
    assert _arxiv_year_from_filename("1909.13868v2.pdf") == 2019


def test_arxiv_year_none_for_non_arxiv():
    assert _arxiv_year_from_filename("example_paper_1952.pdf") is None
    assert _arxiv_year_from_filename(None) is None


# ============================================================
# Author line detection
# ============================================================


def test_authors_multi_bold_with_separators():
    line = "**Laura E Suarez[1,2] *, Yossi Yovel[3] , Olaf Sporns[5] *** "
    ok, cleaned = _looks_like_author_line(line)
    assert ok
    assert "Laura E Suarez" in cleaned


def test_authors_heading_format():
    line = "## Eric Jonas[1] *, Konrad Kording[2][,][3][,][4]"
    ok, _ = _looks_like_author_line(line)
    assert ok


def test_authors_by_prefix():
    line = "#### By A. L. HODGKIN AND A. F. HUXLEY"
    ok, _ = _looks_like_author_line(line)
    assert ok


def test_authors_rejects_abstract():
    line = "Abstract: this paper discusses authors and citations."
    ok, _ = _looks_like_author_line(line)
    assert not ok


def test_authors_rejects_affiliation_line():
    line = "1 Harvard University, Cambridge, MA USA"
    ok, _ = _looks_like_author_line(line)
    assert not ok


def test_authors_rejects_email_line():
    line = "PRANAVSR@CS.STANFORD.EDU AWNI@CS.STANFORD.EDU"
    ok, _ = _looks_like_author_line(line)
    assert not ok


# ============================================================
# End-to-end
# ============================================================


def test_extract_metadata_full_paper():
    md = (
        "## **A paper about something**\n\n"
        "**Alice Author, Bob Builder, and Carol Coder**\n\n"
        "Abstract. This work...\n\n"
        "Published 2024. DOI: 10.1234/foo.bar.5678\n"
    )
    m = extract_metadata(md)
    assert m.title == "A paper about something"
    assert m.authors and "Alice" in m.authors
    assert m.year == 2024
    assert m.doi == "10.1234/foo.bar.5678"
    assert m.confidence >= 0.9


def test_extract_metadata_arxiv_year_hint_used():
    md = "## **Some Paper**\n\n**Foo Bar, Baz Qux**\n\nbody only"
    m = extract_metadata(md, filename="2403.01590v1.pdf")
    assert m.year == 2024


def test_extract_metadata_arxiv_year_overrides_loose_year():
    """When head has a stray in-text year, arxiv filename should win."""
    md = "## **Paper**\n\n**Foo, Bar**\n\nCites (Smith, 1996) earlier work."
    m = extract_metadata(md, filename="1707.01836v1.pdf")
    assert m.year == 2017
