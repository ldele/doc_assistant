"""Smoke tests for the format extractors."""
from pathlib import Path
import pytest

from doc_assistant.extractors import extract_to_markdown, is_supported


def test_supported_formats():
    assert is_supported(Path("paper.pdf"))
    assert is_supported(Path("book.epub"))
    assert is_supported(Path("notes.md"))
    assert is_supported(Path("report.docx"))
    assert not is_supported(Path("malware.exe"))
    assert not is_supported(Path("photo.jpg"))


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