"""Tests for tier-1 citation extraction (Phase 4).

Pure-logic tests — no DB. Matching is tested separately as an integration test.
"""

from doc_assistant.citations import (
    _extract_authors_and_title,
    _extract_doi,
    _extract_year,
    _first_author_surname,
    _split_refs,
    _title_similarity,
    detect_references_section,
    extract_from_markdown,
)

# ============================================================
# Section detection
# ============================================================


def test_detect_references_h2_bold():
    md = "# Paper\nfoo\n## **References**\n- author year title.\n"
    section = detect_references_section(md)
    assert section is not None
    start, end = section
    body = md[start:end]
    assert "author year title" in body


def test_detect_references_h4_caps():
    md = "intro\n\n#### REFERENCES\n- ref one\n- ref two\n"
    section = detect_references_section(md)
    assert section is not None


def test_detect_references_bibliography_alias():
    md = "intro\n\n## Bibliography\nfoo\n"
    assert detect_references_section(md) is not None


def test_detect_references_terminates_at_acknowledgements():
    md = (
        "## References\n"
        "- ref one with year 2020\n"
        "- ref two with year 2021\n"
        "## Acknowledgements\n"
        "This is gratitude.\n"
    )
    section = detect_references_section(md)
    assert section is not None
    body = md[section[0] : section[1]]
    assert "ref one" in body
    assert "gratitude" not in body


def test_no_references_section_returns_none():
    md = "# A lecture transcript\n\nThis is just prose.\n"
    assert detect_references_section(md) is None


# ============================================================
# Splitting
# ============================================================


def test_split_bullet_refs():
    block = (
        "- Author A. Title (1999). Journal.\n"
        "- Author B. Title (2000). Journal.\n"
        "- Author C. Title (2001). Journal.\n"
    )
    parts = _split_refs(block)
    assert len(parts) == 3


def test_split_numbered_refs():
    block = "1. Foo (1999).\n2. Bar (2000).\n3. Baz (2001).\n"
    parts = _split_refs(block)
    assert len(parts) == 3


# ============================================================
# Field extraction
# ============================================================


def test_extract_doi_basic():
    assert _extract_doi("see DOI: 10.1038/nrn3901 here") == "10.1038/nrn3901"


def test_extract_doi_trims_trailing_punct():
    assert _extract_doi("(10.1038/nrn3901).") == "10.1038/nrn3901"


def test_extract_doi_none_when_missing():
    assert _extract_doi("no doi here") is None


def test_extract_year_parens():
    assert _extract_year("Author (1999) Title.") == 1999


def test_extract_year_loose():
    assert _extract_year("Author. Title. Journal 1999.") == 1999


def test_extract_year_none():
    assert _extract_year("no year at all") is None


def test_authors_title_format_a_parens_year():
    text = "Hodgkin AL (1952) On the giant axon. J Physiol."
    a, t = _extract_authors_and_title(text, 1952)
    assert a is not None and "Hodgkin" in a
    assert t is not None and "giant axon" in t


def test_authors_title_format_b_year_at_end():
    text = "Hodgkin AL. On the giant axon. _J Physiol_, 117, 1952."
    a, t = _extract_authors_and_title(text, 1952)
    assert a is not None and "Hodgkin" in a
    assert t is not None and "giant axon" in t


# ============================================================
# First-author surname
# ============================================================


def test_surname_caps_with_initials_and():
    assert _first_author_surname("A. L. HODGKIN AND A. F. HUXLEY") == "hodgkin"


def test_surname_comma_format():
    assert _first_author_surname("Hodgkin, A. L.") == "hodgkin"


def test_surname_initials_after_name():
    assert _first_author_surname("Suarez LE, Yovel Y") == "suarez"


def test_surname_with_backslash_artifact():
    """Extraction sometimes leaves stray backslashes — they shouldn't break parsing."""
    assert _first_author_surname("D. H. HUBEL\\ AND T. N. WIESEL\\") == "hubel"


def test_surname_none_for_empty():
    assert _first_author_surname(None) is None
    assert _first_author_surname("") is None


# ============================================================
# Title similarity
# ============================================================


def test_title_similarity_identical():
    assert _title_similarity("foo bar baz", "foo bar baz") == 1.0


def test_title_similarity_different():
    s = _title_similarity("hello world", "completely unrelated text")
    assert s < 0.5


def test_title_similarity_handles_diacritics():
    """Normalization should make these effectively identical."""
    s = _title_similarity("Álvarez-Carretero S.", "Alvarez Carretero S")
    assert s > 0.9


# ============================================================
# End-to-end on synthetic markdown
# ============================================================


def test_extract_from_markdown_minimal():
    md = (
        "# A paper\n\n"
        "## References\n\n"
        "- Smith, J. (2020) On things. Journal.\n"
        "- Doe, J. (2021) On other things. Journal.\n"
        "- Roe, R. (2022) Yet more. Journal.\n"
    )
    r = extract_from_markdown("doc-1", md)
    assert r.references_section_found is True
    assert r.count == 3
    assert all(c.year is not None for c in r.citations)


def test_extract_from_markdown_no_section():
    md = "# A lecture\n\nNo references here.\n"
    r = extract_from_markdown("doc-1", md)
    assert r.references_section_found is False
    assert r.count == 0
