"""Smoke tests for the format extractors."""

from pathlib import Path

import pytest

from doc_assistant.extractors import (
    count_image_placeholders,
    extract_to_markdown,
    is_supported,
    strip_image_placeholders,
)

# The exact frame PyMuPDF4LLM emits (verified across the real corpus cache, 2026-07).
_REAL = "**==> picture [29 x 29] intentionally omitted <==**"


def test_supported_formats():
    assert is_supported(Path("test.txt"))
    assert is_supported(Path("test.rtf"))
    assert is_supported(Path("test.odt"))
    assert is_supported(Path("paper.pdf"))
    assert is_supported(Path("book.epub"))
    assert is_supported(Path("notes.md"))
    assert is_supported(Path("report.docx"))


def test_unsupported_formats():
    assert not is_supported(Path("test.doc"))
    assert not is_supported(Path("test.tex"))
    assert not is_supported(Path("test.mobi"))
    assert not is_supported(Path("malware.exe"))
    assert not is_supported(Path("photo.jpg"))


def test_case_insensitive():
    assert is_supported(Path("test.PDF"))
    assert is_supported(Path("test.EPUB"))


def test_text_extraction(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("hello world", encoding="utf-8")
    result = extract_to_markdown(file)
    assert "hello world" in result


def test_markdown_extraction(tmp_path):
    file = tmp_path / "sample.md"
    file.write_text("# Heading\n\nSome content.", encoding="utf-8")
    result = extract_to_markdown(file)
    assert "# Heading" in result
    assert "Some content." in result


def test_unsupported_format_raises(tmp_path):
    file = tmp_path / "sample.xyz"
    file.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_to_markdown(file)


# --- KI-14: image-placeholder stripping ---


def test_strip_removes_single_placeholder_and_keeps_prose():
    md = f"Intro paragraph.\n\n{_REAL}\n\nMethods paragraph."
    out = strip_image_placeholders(md)
    assert "intentionally omitted" not in out
    assert "==>" not in out and "<==" not in out
    assert "Intro paragraph." in out
    assert "Methods paragraph." in out
    # The blank-line run left behind is collapsed to a single paragraph break.
    assert "\n\n\n" not in out
    assert out == "Intro paragraph.\n\nMethods paragraph."


def test_strip_removes_multiple_placeholders_varying_dimensions():
    md = (
        "A.\n\n"
        "**==> picture [505 x 240] intentionally omitted <==**\n\n"
        "B.\n\n"
        "**==> picture [7 x 155] intentionally omitted <==**\n\n"
        "C."
    )
    out = strip_image_placeholders(md)
    assert count_image_placeholders(out) == 0
    assert out == "A.\n\nB.\n\nC."


def test_strip_tolerant_of_emphasis_and_whitespace():
    # No emphasis, single-star emphasis, and leading/trailing spaces all match.
    variants = [
        "==> picture [10 x 10] intentionally omitted <==",
        "*==> picture [10 x 10] intentionally omitted <==*",
        "   **==> picture [10 x 10] intentionally omitted <==**   ",
    ]
    for line in variants:
        md = f"before\n\n{line}\n\nafter"
        out = strip_image_placeholders(md)
        assert "==>" not in out, line
        assert out == "before\n\nafter", line


def test_strip_matches_non_picture_frame():
    # Anchored on the ==> … <== frame, not the word "picture".
    md = "text\n\n**==> vector graphic omitted <==**\n\nmore"
    assert strip_image_placeholders(md) == "text\n\nmore"


def test_strip_noop_when_no_placeholder_is_byte_identical():
    md = "# Heading\n\nA table:\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\n\nTrailing gap kept."
    # No placeholder → returned byte-for-byte unchanged (hash-stable), 3+ newlines preserved.
    assert strip_image_placeholders(md) is md


def test_strip_preserves_markdown_structures():
    md = f"# Title\n\n| col | val |\n| --- | --- |\n| x | 1 |\n\n{_REAL}\n\n## Section"
    out = strip_image_placeholders(md)
    assert "| col | val |" in out
    assert "| --- | --- |" in out
    assert "# Title" in out
    assert "## Section" in out
    assert count_image_placeholders(out) == 0


def test_strip_is_idempotent():
    md = f"A.\n\n{_REAL}\n\nB.\n\n{_REAL}\n\nC."
    once = strip_image_placeholders(md)
    assert strip_image_placeholders(once) == once


def test_count_image_placeholders():
    assert count_image_placeholders("no placeholders here") == 0
    assert count_image_placeholders(f"x\n{_REAL}\ny") == 1
    assert count_image_placeholders(f"{_REAL}\n{_REAL}\n{_REAL}") == 3


def test_extract_to_markdown_strips_placeholders(tmp_path):
    # A .md source is read verbatim by extract_text then stripped at the single exit.
    file = tmp_path / "doc.md"
    file.write_text(f"Real content.\n\n{_REAL}\n\nMore content.", encoding="utf-8")
    out = extract_to_markdown(file)
    assert "intentionally omitted" not in out
    assert "Real content." in out
    assert "More content." in out


# ---- KI-64: a ligature's second letter in a table cell -------------------------------------
#
# The "fi" of "fine" as PyMuPDF returns it with quad corrections off (measured on the SuperAnimal
# paper, 2026-10-14): the "i" sits at the "n"'s origin with a box of zero width.


def _fine_blocks() -> list[dict]:
    chars = [
        {"c": "f", "bbox": (147.92, 56.35, 151.70, 63.35), "origin": (147.92, 61.74)},
        {"c": "i", "bbox": (151.70, 56.35, 151.70, 63.35), "origin": (151.70, 61.74)},
        {"c": "n", "bbox": (151.70, 56.35, 155.91, 63.35), "origin": (151.70, 61.74)},
        {"c": "e", "bbox": (155.91, 56.35, 159.27, 63.35), "origin": (155.91, 61.74)},
    ]
    span = {"bbox": (147.92, 56.35, 159.27, 63.35), "chars": chars, "char_flags": 0, "flags": 0}
    line = {"bbox": span["bbox"], "dir": (1.0, 0.0), "spans": [span]}
    return [{"type": 0, "bbox": span["bbox"], "lines": [line]}]


CELL = (140.0, 50.0, 200.0, 70.0)


class _FakeTextPage:
    def extractRAWDICT(self) -> dict:  # PyMuPDF's name
        return {"blocks": _fine_blocks()}


def test_a_zero_width_character_gets_a_hair_wide_box_at_its_origin() -> None:
    from doc_assistant.extractors import widen_zero_width_chars

    blocks = _fine_blocks()
    widen_zero_width_chars(blocks)
    f, i, _n, _e = blocks[0]["lines"][0]["spans"][0]["chars"]
    assert i["bbox"][0] == 151.70 and 0 < i["bbox"][2] - i["bbox"][0] < 0.1
    assert f["bbox"] == (147.92, 56.35, 151.70, 63.35)  # a box with width is left alone
    widen_zero_width_chars(blocks)  # a second pass changes nothing
    assert blocks[0]["lines"][0]["spans"][0]["chars"][1]["bbox"] == i["bbox"]


def test_both_table_cell_functions_keep_the_ligatures_second_letter(monkeypatch) -> None:
    import pymupdf.table
    from pymupdf4llm.helpers import utils

    from doc_assistant.extractors import _keep_ligature_letters_in_table_cells

    # As shipped, each drops the "i": the defect itself, so this test can fail. Unwrapped first, in
    # case an earlier test ran the PDF extractor; monkeypatch puts back whatever was there.
    plain_table = getattr(pymupdf.table.extract_cells, "original", pymupdf.table.extract_cells)
    plain_utils = getattr(utils.extract_cells, "original", utils.extract_cells)
    monkeypatch.setattr(pymupdf.table, "extract_cells", plain_table)
    monkeypatch.setattr(utils, "extract_cells", plain_utils)
    assert plain_table(_FakeTextPage(), CELL) == "fne"
    assert plain_utils(_fine_blocks(), CELL) == "fne"

    _keep_ligature_letters_in_table_cells()
    _keep_ligature_letters_in_table_cells()  # idempotent: wrapped once
    assert pymupdf.table.extract_cells(_FakeTextPage(), CELL) == "fine"
    assert utils.extract_cells(_fine_blocks(), CELL) == "fine"
    assert not hasattr(pymupdf.table.extract_cells.original, "original")
    assert not hasattr(utils.extract_cells.original, "original")


def test_the_table_cell_functions_keep_their_shape() -> None:
    """KI-64's wrappers call both libraries' cell functions by these parameters. An upgrade that
    moves them fails here before the ligature letters go missing again."""
    import inspect

    import pymupdf.table
    from pymupdf4llm.helpers import utils

    def params(fn) -> list[str]:
        return list(inspect.signature(getattr(fn, "original", fn)).parameters)

    assert params(pymupdf.table.extract_cells) == ["textpage", "cell", "markdown"]
    assert params(utils.extract_cells) == ["table_blocks", "cell", "markdown", "ocrpage"]
