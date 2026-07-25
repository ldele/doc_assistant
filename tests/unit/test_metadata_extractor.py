"""Tests for document-level metadata extraction (Phase 4)."""

from doc_assistant.metadata_extractor import (
    _arxiv_year_from_filename,
    _clean_markdown,
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


def test_authors_rejects_discourse_lead():
    """A discourse/section sentence is never an author list (observed on ACM-style papers)."""
    assert not _looks_like_author_line("However, ideas from these techniques are unexplored.")[0]
    assert not _looks_like_author_line("Additional Key Words and Phrases: Medical Segmentation")[0]
    assert not _looks_like_author_line("We present a hybrid model, and we evaluate it.")[0]


# ============================================================
# Cleaning + boilerplate skips (2026-07 enrichment wiring)
# ============================================================


def test_clean_markdown_strips_backslash_artifacts():
    # markdown hard-break backslashes leaked into author names ("WIESEL\")
    assert _clean_markdown("D. H. HUBEL\\ AND T. N. WIESEL\\") == "D. H. HUBEL AND T. N. WIESEL"


def test_title_skips_publisher_copyright_line():
    """A Springer-style licence line must not be mistaken for the title."""
    md = (
        "# The Author(s), under exclusive licence to Springer Nature\n\n"
        "# The Past, Present, and Future State of the Field\n\nbody"
    )
    assert _extract_title(md) == "The Past, Present, and Future State of the Field"


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


# ============================================================
# KI-26 — banner titles, author-line titles, and years read from the wrong place.
# Each case below is a *real document* from the corpus, reduced to the shape that broke it.
# ============================================================


def test_frontiers_banner_is_not_a_title_and_the_citation_block_supplies_the_real_one():
    """`fnana-*.pdf`: the access banner is an H2 *above* the title, so it won the pick — 7 of 97
    documents were stored as "OPEN ACCESS". The real title is in the self-citation block."""
    md = (
        "TYPE Review PUBLISHED 29 October 2024 DOI 10.3389/fnana.2024.1419108\n\n"
        "## OPEN ACCESS\n\n"
        "EDITED BY Huijiao Liu, China Agricultural University, China\n\n"
        "CITATION\n\n"
        "Pedrao LFAT, Medeiros POS and Falquetto B (2024) Parkinson's disease models and death "
        "signaling: what do we know until now? _Front. Neuroanat._ 18:1419108. doi: 10.3389/x\n\n"
        "## COPYRIGHT\n"
    )
    meta = extract_metadata(md, filename="fnana-18-1419108.pdf")
    assert meta.title == (
        "Parkinson's disease models and death signaling: what do we know until now?"
    )
    assert meta.year == 2024


def test_page_furniture_is_never_a_title():
    for banner in ("OPEN ACCESS", "Disclaimer", "Graphical abstract", "ORIGINAL ARTICLE"):
        md = f"## {banner}\n\n## **The actual paper title goes here**\n\nbody"
        assert _extract_title(md) == "The actual paper title goes here", banner


def test_bold_title_beats_a_later_author_heading():
    """`2606.31856v1.pdf` / `41304_2021_Article_335.pdf`: the title is a bold line and the
    *authors* are the heading. Preferring headings stored the authors as the title on both."""
    md = (
        "**Low-dimensional topology of deep neural networks**\n\n"
        "## **Junyu Ren**[1] **Lek-Heng Lim**[1]\n\n"
        "## **Abstract**\n\nWe study layered models…\n"
    )
    assert _extract_title(md) == "Low-dimensional topology of deep neural networks"


def test_a_real_title_that_looks_like_names_is_still_a_title():
    """The counter-example that rules out any capitalisation heuristic for author lines."""
    md = "## **Attention Is All You Need**\n\n## **Abstract**\n\nThe dominant sequence…\n"
    assert _extract_title(md) == "Attention Is All You Need"


def test_year_ignores_citation_years_in_the_abstract():
    """`dpr_karpukhin_2020.pdf`: no publication keyword in the header, so the unbounded loose scan
    reached into the abstract and returned a *cited* paper's year — it was stored as 2012."""
    md = (
        "## **Dense Passage Retrieval for Open-Domain Question Answering**\n\n"
        "**Vladimir Karpukhin, Barlas Oguz, Sewon Min**\n\nFacebook AI\n\n"
        "## **Abstract**\n\n"
        "Traditional sparse models such as TF-IDF or BM25 (Robertson, 2012) are the de facto…\n"
    )
    assert extract_metadata(md, filename="dpr_karpukhin_2020.pdf").year == 2020
    assert _extract_year(md) is None  # nothing in the front matter claims a year


def test_published_year_beats_accepted_year():
    """`41304_2021_Article_335.pdf`: "Accepted: 14 December 2020 / Published online: 10 June 2021"
    was read as 2020 — the submission date, not the publication date."""
    md = "# **A paper**\n\nAccepted: 14 December 2020 / Published online: 10 June 2021\n"
    assert _extract_year(md) == 2021


def test_published_keyword_may_cross_a_line_break():
    """PMC author manuscripts wrap the line; a `[^\n]` gap missed it and fell through to the PMC
    *availability* year (`nihms-326467.pdf` was stored as 2013 for a 2012 paper)."""
    md = (
        "NIH Public Access **Author Manuscript**\n\n"
        "_Curr Opin Neurobiol_. Author manuscript; available in PMC 2013 February 1.\n\n"
        "## Published in final edited form as:\n\n"
        "Curr Opin Neurobiol. 2012 February ; 22(1): 144\n"
    )
    assert _extract_year(md) == 2012


def test_journal_running_header_supplies_the_year():
    """`chazal_2004-ecg.pdf`: the issue date lives in the IEEE running header, and the only keyword
    in the document is its 2003 submission date."""
    md = (
        "IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 51, NO. 7, JULY 2004 1196\n\n"
        "## Automatic Classification of Heartbeats\n\n"
        "Manuscript received April 25, 2003; revised January 2004.\n"
    )
    assert _extract_year(md) == 2004


def test_filename_year_fills_in_but_never_overrides_the_document():
    """A filename year is the *downloader's* claim: better than a loose scan, weaker than the
    document stating its own date."""
    scan = "SOME OLD SCANNED PAPER\n\nbody text with no metadata at all\n"
    assert extract_metadata(scan, filename="hebb_1949.pdf").year == 1949
    stated = "# **A paper**\n\nPublished 3 March 2018\n"
    assert extract_metadata(stated, filename="paper_2017.pdf").year == 2018


def test_arxiv_id_in_a_filename_is_not_a_year():
    """`1904.01169v3.pdf` is arXiv 2019, not the year 1904 — the arXiv tier owns that filename."""
    from doc_assistant.metadata_extractor import _year_from_filename

    assert _year_from_filename("1904.01169v3.pdf") is None
    assert _arxiv_year_from_filename("1904.01169v3.pdf") == 2019
    assert _year_from_filename("dpr_karpukhin_2020.pdf") == 2020
    assert _year_from_filename("PIIS0002929724003008.pdf") is None  # digits inside a longer token


def test_journal_citation_heading_is_skipped_case_insensitively():
    """The skip rule was dead code: `_is_skippable_heading` lowercases, `_JOURNAL_HEADER` was
    anchored on `^[A-Z]`. Only the (now removed) H1-over-H2 preference was hiding it."""
    from doc_assistant.metadata_extractor import _is_skippable_heading

    assert _is_skippable_heading("J. Physiol. (1952) 117, 500-544")
    assert not _is_skippable_heading("Dense Passage Retrieval for Open-Domain Question Answering")
