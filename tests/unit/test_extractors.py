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
