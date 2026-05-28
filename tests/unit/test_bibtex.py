"""Tests for the BibTeX export module (PR 1.5).

Pure-function tests. Document objects are constructed in-memory; no
session, no fixture, no migrations.
"""

from __future__ import annotations

from doc_assistant.bibtex import (
    BibEntry,
    _build_entry,
    _citation_key,
    _dedupe_keys,
    _safe_key_fragment,
    build_entries,
    escape_bibtex,
    render,
)
from doc_assistant.db.models import Document


def _doc(**kwargs: object) -> Document:
    """Construct a Document with sensible defaults for testing."""
    defaults: dict[str, object] = {
        "id": "deadbeef" + "0" * 24,
        "filename": "test.pdf",
        "source_original": "/tmp/test.pdf",
        "source_cache": None,
        "doc_hash": "abc123def4567890",
        "format": "pdf",
        "title": None,
        "authors": None,
        "year": None,
        "doi": None,
    }
    defaults.update(kwargs)
    return Document(**defaults)


# ============================================================
# escape_bibtex
# ============================================================


def test_escape_bibtex_passes_normal_text():
    assert escape_bibtex("A simple title") == "A simple title"


def test_escape_bibtex_escapes_braces():
    assert escape_bibtex("with {curly} braces") == "with \\{curly\\} braces"


def test_escape_bibtex_collapses_newlines():
    assert escape_bibtex("line one\nline two\n\nline three") == "line one line two line three"


def test_escape_bibtex_empty():
    assert escape_bibtex("") == ""


def test_escape_bibtex_preserves_latex_metachars_inside_braces():
    # & % $ # _ are safe inside {...} so we deliberately leave them alone.
    assert escape_bibtex("100% & $5 #1 a_b") == "100% & $5 #1 a_b"


# ============================================================
# _safe_key_fragment
# ============================================================


def test_safe_key_collapses_specials():
    assert _safe_key_fragment("Hodgkin, A. L.") == "hodgkin_a_l"


def test_safe_key_strips_edges():
    assert _safe_key_fragment("  __foo__  ") == "foo"


# ============================================================
# _citation_key
# ============================================================


def test_citation_key_paper_uses_surname_year():
    doc = _doc(authors="A. L. Hodgkin and A. F. Huxley", year=1952)
    assert _citation_key(doc) == "hodgkin_1952"


def test_citation_key_note_uses_filename_stem():
    doc = _doc(format="md", filename="meeting-notes.md")
    assert _citation_key(doc) == "note_meeting_notes"


def test_citation_key_misc_fallback_when_no_author():
    doc = _doc(id="abcd1234" + "0" * 24, format="pdf", authors=None, year=2020)
    assert _citation_key(doc) == "misc_abcd1234"


def test_citation_key_uses_surname_only_when_year_missing():
    doc = _doc(authors="Hodgkin, A. L.", year=None)
    assert _citation_key(doc) == "hodgkin"


# ============================================================
# _dedupe_keys
# ============================================================


def test_dedupe_keys_no_collisions_passes_through():
    es = [
        BibEntry(entry_type="article", key="a_1900", fields={}),
        BibEntry(entry_type="article", key="b_1900", fields={}),
    ]
    out = _dedupe_keys(es)
    assert [e.key for e in out] == ["a_1900", "b_1900"]


def test_dedupe_keys_appends_letter_suffix():
    es = [
        BibEntry(entry_type="article", key="hodgkin_1952", fields={"i": "1"}),
        BibEntry(entry_type="article", key="hodgkin_1952", fields={"i": "2"}),
        BibEntry(entry_type="article", key="hodgkin_1952", fields={"i": "3"}),
    ]
    out = _dedupe_keys(es)
    assert [e.key for e in out] == ["hodgkin_1952a", "hodgkin_1952b", "hodgkin_1952c"]
    assert [e.fields["i"] for e in out] == ["1", "2", "3"]


# ============================================================
# _build_entry — entry type selection
# ============================================================


def test_build_entry_paper_is_article():
    doc = _doc(
        filename="hh.pdf",
        format="pdf",
        title="A quantitative description of membrane current",
        authors="A. L. Hodgkin and A. F. Huxley",
        year=1952,
        doi="10.1113/jphysiol.1952.sp004764",
    )
    e = _build_entry(doc)
    assert e.entry_type == "article"
    assert e.fields["title"].startswith("A quantitative description")
    assert e.fields["author"] == "A. L. Hodgkin and A. F. Huxley"
    assert e.fields["year"] == "1952"
    assert e.fields["doi"] == "10.1113/jphysiol.1952.sp004764"
    assert "filename: hh.pdf" in e.fields["note"]


def test_build_entry_note_is_misc_with_howpublished():
    doc = _doc(filename="meeting.md", format="md", title="Standup notes")
    e = _build_entry(doc)
    assert e.entry_type == "misc"
    assert e.fields["howpublished"] == "Personal note"
    assert e.fields["title"] == "Standup notes"
    assert "filename: meeting.md" in e.fields["note"]


def test_build_entry_pdf_without_metadata_is_misc():
    doc = _doc(filename="anonymous.pdf", format="pdf", authors=None, year=None)
    e = _build_entry(doc)
    assert e.entry_type == "misc"
    # Falls back to filename as title so the entry isn't empty.
    assert e.fields["title"] == "anonymous.pdf"


def test_build_entry_paper_missing_year_falls_back_to_misc():
    doc = _doc(filename="x.pdf", authors="A. L. Hodgkin", year=None)
    e = _build_entry(doc)
    assert e.entry_type == "misc"


def test_build_entry_escapes_braces_in_title():
    doc = _doc(title="Curly {brace} title", authors="X Y", year=2020)
    e = _build_entry(doc)
    assert "\\{brace\\}" in e.fields["title"]


# ============================================================
# build_entries + render — end to end
# ============================================================


def test_build_entries_dedupes_across_corpus():
    docs = [
        _doc(filename="hh1.pdf", authors="A. L. Hodgkin and A. F. Huxley", year=1952),
        _doc(filename="hh2.pdf", authors="Hodgkin, A. L.", year=1952),
    ]
    entries = build_entries(docs)
    keys = [e.key for e in entries]
    assert keys == ["hodgkin_1952a", "hodgkin_1952b"]


def test_render_emits_header_and_sorts_by_key():
    docs = [
        _doc(id="b" * 32, filename="b.md", format="md"),
        _doc(id="a" * 32, filename="a.md", format="md"),
    ]
    out = render(build_entries(docs))
    assert out.startswith("% Generated by doc_assistant")
    # Sorted ascending by key
    a_pos = out.find("@misc{note_a")
    b_pos = out.find("@misc{note_b")
    assert 0 < a_pos < b_pos


def test_render_valid_bibtex_structure():
    docs = [
        _doc(
            filename="hh.pdf",
            title="A quantitative description",
            authors="Hodgkin, A. L. and Huxley, A. F.",
            year=1952,
            doi="10.1113/jphysiol.1952.sp004764",
        )
    ]
    out = render(build_entries(docs))
    assert "@article{hodgkin_1952," in out
    assert "  title = {A quantitative description}," in out
    assert "  year = {1952}," in out
    assert out.rstrip().endswith("}")
