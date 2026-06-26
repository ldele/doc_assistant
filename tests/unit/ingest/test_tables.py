"""Tests for the pdfplumber table-extraction enrichment (Feature 4a).

The pure functions (render, splice, strip, filtering, cell cleaning) are
covered exhaustively. ``extract_tables`` itself opens a PDF via pdfplumber;
it is exercised with a monkeypatched ``pdfplumber.open`` so no real PDF or
the binary dependency's behaviour is needed.
"""

from __future__ import annotations

from typing import Any

from doc_assistant.ingest.tables import (
    _BLOCK_BEGIN,
    _BLOCK_END,
    ExtractedTable,
    extract_tables,
    extract_tables_from_pages,
    has_spliced_tables,
    render_table_markdown,
    splice_tables,
    strip_spliced_tables,
)


def _table(rows: list[list[str]], *, page: int = 1, index: int = 1) -> ExtractedTable:
    return ExtractedTable(page=page, index=index, rows=rows)


# ============================================================
# Rendering
# ============================================================


def test_render_basic_table_is_github_markdown():
    t = _table([["h1", "h2"], ["a", "b"], ["c", "d"]], page=3, index=2)
    md = render_table_markdown(t)
    lines = md.splitlines()
    assert lines[0] == "<!-- table-extracted-by: pdfplumber page=3 table=2 -->"
    assert lines[1] == "| h1 | h2 |"
    assert lines[2] == "| --- | --- |"
    assert lines[3] == "| a | b |"
    assert lines[4] == "| c | d |"


def test_render_pads_ragged_rows():
    t = _table([["h1", "h2", "h3"], ["a"]])
    md = render_table_markdown(t)
    # The short body row is padded to full width.
    assert "| a |  |  |" in md


def test_dims_properties():
    t = _table([["a", "b", "c"], ["d", "e"]])
    assert t.n_rows == 2
    assert t.n_cols == 3


# ============================================================
# Splice / strip — idempotency is the core invariant
# ============================================================


def test_splice_appends_bounded_block():
    md = "# Title\n\nBody text.\n"
    out = splice_tables(md, [_table([["h1", "h2"], ["a", "b"]])])
    assert md.rstrip() in out
    assert _BLOCK_BEGIN in out and _BLOCK_END in out
    assert has_spliced_tables(out)


def test_splice_is_idempotent():
    md = "# Title\n\nBody.\n"
    tables = [_table([["h1", "h2"], ["a", "b"]])]
    once = splice_tables(md, tables)
    twice = splice_tables(once, tables)
    assert once == twice
    assert once.count(_BLOCK_BEGIN) == 1


def test_resplice_replaces_old_block():
    md = "# Title\n\nBody.\n"
    first = splice_tables(md, [_table([["h1", "h2"], ["a", "b"]])])
    second = splice_tables(first, [_table([["x", "y"], ["1", "2"]])])
    assert second.count(_BLOCK_BEGIN) == 1
    assert "| x | y |" in second
    assert "| a | b |" not in second


def test_strip_restores_original_body():
    md = "# Title\n\nBody.\n"
    spliced = splice_tables(md, [_table([["h1", "h2"], ["a", "b"]])])
    stripped = strip_spliced_tables(spliced)
    assert not has_spliced_tables(stripped)
    assert "Body." in stripped
    assert _BLOCK_BEGIN not in stripped


def test_strip_is_noop_without_block():
    md = "# Title\n\nNo tables here.\n"
    assert strip_spliced_tables(md) == md


def test_splice_empty_tables_clears_block():
    md = "# Title\n\nBody.\n"
    spliced = splice_tables(md, [_table([["h1", "h2"], ["a", "b"]])])
    cleared = splice_tables(spliced, [])
    assert not has_spliced_tables(cleared)
    assert "Body." in cleared


# ============================================================
# Page-gated extraction (monkeypatched pdfplumber)
# ============================================================


class _FakePage:
    def __init__(self, tables: list[list[list[Any]]]) -> None:
        self._tables = tables

    def extract_tables(self) -> list[list[list[Any]]]:
        return self._tables


class _FakePdf:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> _FakePdf:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_pdfplumber(monkeypatch: Any, pages: list[_FakePage]) -> None:
    import pdfplumber

    monkeypatch.setattr(pdfplumber, "open", lambda _path: _FakePdf(pages))


def test_extract_from_pages_filters_small_and_empty(monkeypatch: Any):
    pages = [
        _FakePage(
            [
                [["h1", "h2"], ["a", "b"]],  # kept
                [["only one column"]],  # dropped: < MIN_COLS
                [["", ""], ["", ""]],  # dropped: all empty
                [["solo row of two cols"]],  # dropped: < MIN_ROWS
            ]
        )
    ]
    _patch_pdfplumber(monkeypatch, pages)
    tables = extract_tables_from_pages("ignored.pdf", [1])
    assert len(tables) == 1
    assert tables[0].rows == [["h1", "h2"], ["a", "b"]]


def test_extract_from_pages_rejects_prose_misdetected_as_table(monkeypatch: Any):
    """A tall narrow region with one huge run-together cell is prose, not a table."""
    prose = "Theanatomicalmodelgenerates" * 50  # > MAX_CELL_CHARS, no real columns
    pages = [_FakePage([[["", prose], ["", prose], ["", prose]]])]
    _patch_pdfplumber(monkeypatch, pages)
    assert extract_tables_from_pages("ignored.pdf", [1]) == []


def test_extract_from_pages_rejects_single_populated_column(monkeypatch: Any):
    """Data confined to one column (the other entirely empty) isn't tabular."""
    pages = [_FakePage([[["a", ""], ["b", ""], ["c", ""]]])]
    _patch_pdfplumber(monkeypatch, pages)
    assert extract_tables_from_pages("ignored.pdf", [1]) == []


def test_extract_from_pages_only_touches_requested_pages(monkeypatch: Any):
    """Non-candidate pages are never extracted, even if they hold 'tables'."""
    pages = [
        _FakePage([[["p1a", "p1b"], ["x", "y"]]]),  # page 1 — NOT requested
        _FakePage([[["p2a", "p2b"], ["x", "y"]]]),  # page 2 — requested
    ]
    _patch_pdfplumber(monkeypatch, pages)
    tables = extract_tables_from_pages("ignored.pdf", [2])
    assert len(tables) == 1
    assert tables[0].page == 2
    assert tables[0].rows[0] == ["p2a", "p2b"]


def test_extract_from_pages_empty_page_list_is_noop(monkeypatch: Any):
    _patch_pdfplumber(monkeypatch, [_FakePage([[["h1", "h2"], ["a", "b"]]])])
    assert extract_tables_from_pages("ignored.pdf", []) == []


def test_extract_from_pages_cleans_cells_and_numbers_across_pages(monkeypatch: Any):
    pages = [
        _FakePage([[["a\nb", None], ["c|d", "  e  "]]]),
        _FakePage([[["p2h1", "p2h2"], ["x", "y"]]]),
    ]
    _patch_pdfplumber(monkeypatch, pages)
    tables = extract_tables_from_pages("ignored.pdf", [1, 2])
    assert len(tables) == 2
    # Newlines collapsed, None → "", pipe escaped, whitespace trimmed.
    assert tables[0].rows == [["a b", ""], [r"c\|d", "e"]]
    # Indexing is document-wide and page order is preserved.
    assert (tables[0].page, tables[0].index) == (1, 1)
    assert (tables[1].page, tables[1].index) == (2, 2)


# ============================================================
# Orchestration: extract_tables = (regions classifier) ∘ page extraction
# ============================================================


def test_extract_tables_only_extracts_classified_table_pages(monkeypatch: Any):
    """extract_tables delegates page selection to regions.table_candidate_pages."""
    import doc_assistant.ingest.regions as regions

    monkeypatch.setattr(regions, "table_candidate_pages", lambda _p: [2])
    pp_pages = [
        _FakePage([[["fig", "noise"], ["x", "y"]]]),  # page 1 — not a table page
        _FakePage([[["real", "table"], ["1", "2"]]]),  # page 2 — the table page
    ]
    _patch_pdfplumber(monkeypatch, pp_pages)
    tables = extract_tables("ignored.pdf")
    assert len(tables) == 1
    assert tables[0].page == 2
    assert tables[0].rows[0] == ["real", "table"]


def test_extract_tables_empty_when_no_table_pages(monkeypatch: Any):
    import doc_assistant.ingest.regions as regions

    monkeypatch.setattr(regions, "table_candidate_pages", lambda _p: [])
    _patch_pdfplumber(monkeypatch, [_FakePage([[["a", "b"], ["c", "d"]]])])
    assert extract_tables("ignored.pdf") == []
