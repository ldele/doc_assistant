"""The extraction report's fault flags (0.9.0 plan, session 2), on made-up text.

Each flag is derived from the document's own text, so these tests pin the shape that must be
caught and the near miss that must not: a chapter heading is not a running header, ``well-known``
is not a broken word, a short repeated page number is not a header.
"""

from __future__ import annotations

from doc_assistant.ingest.extraction_report import (
    hyphen_splits,
    page_segments,
    repeated_pages,
    running_lines,
    table_mentions,
    title_is_running_line,
)


def _doc(pages: list[str]) -> str:
    return "".join(f"\n<!-- page:{i} -->\n{text}" for i, text in enumerate(pages, start=1))


def test_page_segments_split_on_markers_and_keep_unmarked_text_whole() -> None:
    segs = page_segments(_doc(["one", "two"]))
    assert [(p, t.strip()) for p, t in segs] == [(1, "one"), (2, "two")]
    assert page_segments("an EPUB has no pages") == [(0, "an EPUB has no pages")]


def test_a_line_on_every_page_top_is_a_running_line_whatever_its_page_number() -> None:
    pages = [
        f"Journal of Things 12 ({i})\nBody text {i}.\nMore.\nEven more.\nEnd {i}" for i in range(6)
    ]
    found = running_lines(page_segments(_doc(pages)))
    assert "journal of things # (#)" in found
    assert found["journal of things # (#)"] == 6


def test_a_heading_on_two_of_ten_pages_and_bare_page_numbers_are_not() -> None:
    pages = [f"Body {i}\nline\nline\nline\n{i}" for i in range(10)]
    pages[3] = "Methods\n" + pages[3]
    pages[4] = "Methods\n" + pages[4]
    found = running_lines(page_segments(_doc(pages)))
    assert "methods" not in found
    assert "#" not in found  # the page number, folded, is too short to be a header


def test_running_lines_need_three_pages() -> None:
    assert running_lines(page_segments(_doc(["Head\nx", "Head\ny"]))) == {}


def test_a_title_that_is_the_running_header_is_flagged_and_a_real_title_is_not() -> None:
    running = {"journal of things # (#)": 6}
    assert title_is_running_line("Journal of Things 12 (3)", running)
    assert not title_is_running_line("Why things fall", running)
    assert not title_is_running_line(None, running)


def test_a_page_identical_to_the_page_before_is_flagged() -> None:
    segs = page_segments(_doc(["alpha", "alpha", "beta", "", ""]))
    assert repeated_pages(segs) == [2]  # empty pages repeat nothing


def test_a_hyphen_split_counts_only_when_the_word_occurs_whole() -> None:
    text = "The exam-\nple shows it. Another example here. A well-\nknown case."
    assert hyphen_splits(text) == ["exam-ple"]


def test_table_mentions_are_found_at_a_line_start_by_page() -> None:
    segs = page_segments(_doc(["Table 2: results\nrow", "as Table 3 shows", "**Table IV** Means"]))
    assert table_mentions(segs) == {1: ["2"], 3: ["IV"]}
