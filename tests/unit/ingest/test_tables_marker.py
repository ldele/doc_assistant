"""Unit tests for Feature 4a Marker table parse + inline splice (pure; no subprocess)."""

from __future__ import annotations

from doc_assistant.ingest.tables_marker import (
    has_marker_tables,
    parse_marker_tables,
    splice_tables_inline,
    strip_marker_tables,
    strip_pdfplumber_block,
)

# A cached .md as the extractor writes it: page markers + a lossy pymupdf table on p2.
CACHE_MD = """<!-- page:1 -->
# Intro
Background text on page one.

| Year | N |
| --- | --- |
| 2018 | 10 |
| 2019 | 20 |

<!-- page:2 -->
## Results
Table 1 shows accuracy.

| Model | Acc |
| --- | --- |
| BM25 | 42 |
| DPR | 79 |

More prose after the table.

<!-- page:3 -->
## Conclusion
Final remarks, no table here.
"""

# Marker's high-fidelity render of page 2's table (rich: <br>, bold).
MARKER_MD_P2 = """## Results

| Model | Top-20 | Top-100 |
| --- | --- | --- |
| BM25 | 59.1 | 73.7 |
| DPR | 78.4<br>dense | **85.4** |
"""

# Two-page paginated Marker output (delimiter between pages).
MARKER_MD_PAGINATED = """| A | B |
| --- | --- |
| 1 | 2 |
| 3 | 4 |

{0}------------------------------------------------

| X | Y |
| --- | --- |
| 5 | 6 |
| 7 | 8 |
"""


# --- parse_marker_tables -----------------------------------------------------


def test_parse_extracts_table_and_preserves_fidelity() -> None:
    tables = parse_marker_tables(MARKER_MD_P2, [2])
    assert len(tables) == 1
    t = tables[0]
    assert t.page == 2 and t.index == 1
    assert "<br>" in t.markdown and "**85.4**" in t.markdown  # fidelity kept
    assert "Top-100" in t.markdown


def test_parse_attributes_pages_by_requested_order() -> None:
    tables = parse_marker_tables(MARKER_MD_PAGINATED, [4, 7])
    assert [t.page for t in tables] == [4, 7]
    assert [t.index for t in tables] == [1, 2]


def test_parse_no_delimiter_falls_back_to_first_page() -> None:
    tables = parse_marker_tables(MARKER_MD_P2, [9])
    assert len(tables) == 1 and tables[0].page == 9


def test_parse_rejects_non_table_prose() -> None:
    assert parse_marker_tables("Just a paragraph, no pipes here.", [1]) == []


# --- splice_tables_inline ----------------------------------------------------


def test_splice_dedups_lossy_twin_in_page_span() -> None:
    tables = parse_marker_tables(MARKER_MD_P2, [2])
    out = splice_tables_inline(CACHE_MD, tables)
    # The lossy pymupdf table values for p2 are gone; Marker's are in.
    assert "| BM25 | 42 |" not in out
    assert "Top-100" in out and "**85.4**" in out
    # Wrapped in the idempotent marker block.
    assert "<!-- table:marker:page=2:begin -->" in out
    assert "<!-- table:marker:page=2:end -->" in out


def test_splice_leaves_other_pages_untouched() -> None:
    tables = parse_marker_tables(MARKER_MD_P2, [2])
    out = splice_tables_inline(CACHE_MD, tables)
    # Page 1's table (Marker did not process p1) is left alone.
    assert "| 2018 | 10 |" in out
    assert "## Conclusion" in out  # p3 prose intact


def test_splice_is_idempotent() -> None:
    tables = parse_marker_tables(MARKER_MD_P2, [2])
    once = splice_tables_inline(CACHE_MD, tables)
    twice = splice_tables_inline(once, tables)
    assert once == twice


def test_splice_places_table_within_its_page_region() -> None:
    tables = parse_marker_tables(MARKER_MD_P2, [2])
    out = splice_tables_inline(CACHE_MD, tables)
    # The marker block sits between the page-2 and page-3 markers.
    p2 = out.index("<!-- page:2 -->")
    p3 = out.index("<!-- page:3 -->")
    blk = out.index("<!-- table:marker:page=2:begin -->")
    assert p2 < blk < p3


# A page where the caption sits far above the data: caption near the top, then a wall
# of prose, then (in the source) the grid. The splice must anchor the grid to the
# caption, not dump it at the span end — otherwise caption and grid land in different
# parents and the values become unretrievable (the DPR Table-2 failure, #4a).
CACHE_MD_CAPTION_FAR = """<!-- page:5 -->
Table 2: Top-20 & Top-100 retrieval accuracy on test sets.

| Model | Acc |
| --- | --- |
| BM25 | 42 |

Lots of intervening prose about training schemes and efficiency here.

## 5.2 Ablation Study
To understand further how options affect results.

<!-- page:6 -->
## Conclusion
"""

MARKER_MD_WIDE = """| Training | Top-20 | Top-100 |
| --- | --- | --- |
| DPR | 78.4 | 85.4 |
"""


def test_splice_anchors_block_to_caption_not_span_end() -> None:
    tables = parse_marker_tables(MARKER_MD_WIDE, [5])
    out = splice_tables_inline(CACHE_MD_CAPTION_FAR, tables)
    cap = out.index("Table 2: Top-20 & Top-100")
    blk = out.index("<!-- table:marker:page=5:begin -->")
    ablation = out.index("## 5.2 Ablation Study")
    # Grid is anchored right after the caption — before the intervening prose/heading,
    # not appended at the end of the page span.
    assert cap < blk < ablation


def test_splice_no_caption_falls_back_to_span_end() -> None:
    md = "<!-- page:1 -->\nSome prose, no table caption.\n\n<!-- page:2 -->\nEnd.\n"
    out = splice_tables_inline(md, parse_marker_tables(MARKER_MD_WIDE, [1]))
    # No "Table N" caption on page 1 → block appended at the span end (own parent).
    prose = out.index("Some prose")
    blk = out.index("<!-- table:marker:page=1:begin -->")
    p2 = out.index("<!-- page:2 -->")
    assert prose < blk < p2


# --- helpers -----------------------------------------------------------------


def test_strip_and_has_marker_tables_roundtrip() -> None:
    tables = parse_marker_tables(MARKER_MD_P2, [2])
    out = splice_tables_inline(CACHE_MD, tables)
    assert has_marker_tables(out)
    stripped = strip_marker_tables(out)
    assert not has_marker_tables(stripped)
    assert "Top-100" not in stripped  # the Marker table is gone with its block


def test_strip_pdfplumber_block_removes_pdfplumber_splice() -> None:
    from doc_assistant.ingest.tables import ExtractedTable, splice_tables

    md = "<!-- page:1 -->\nText.\n"
    pp = splice_tables(md, [ExtractedTable(page=1, index=1, rows=[["a", "b"], ["1", "2"]])])
    assert "pdfplumber" in pp
    assert "pdfplumber" not in strip_pdfplumber_block(pp)
