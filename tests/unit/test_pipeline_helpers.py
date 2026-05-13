"""Tests for citation and document formatting helpers."""
from langchain_core.documents import Document

from doc_assistant.pipeline import format_citation, format_docs_for_prompt


def make_doc(content="text", **metadata) -> Document:
    return Document(page_content=content, metadata=metadata)


def test_citation_with_full_metadata():
    doc = make_doc(filename="paper.pdf", page=42, section="Methodology")
    citation = format_citation(doc, 1)
    assert "[1]" in citation
    assert "paper.pdf" in citation
    assert "p.42" in citation
    assert "Methodology" in citation


def test_citation_with_partial_metadata():
    doc = make_doc(filename="paper.pdf")
    citation = format_citation(doc, 1)
    assert "paper.pdf" in citation
    assert "p." not in citation


def test_citation_with_missing_metadata():
    doc = make_doc()
    citation = format_citation(doc, 1)
    assert "[1]" in citation
    assert "unknown" in citation


def test_format_docs_for_prompt_includes_filename():
    docs = [
        make_doc(content="content one", filename="a.pdf"),
        make_doc(content="content two", filename="b.pdf", page=5),
    ]
    formatted = format_docs_for_prompt(docs)
    assert "a.pdf" in formatted
    assert "b.pdf" in formatted
    assert "page 5" in formatted
    assert "content one" in formatted
    assert "content two" in formatted