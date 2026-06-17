"""Tests for citation and document formatting helpers."""

from langchain_core.documents import Document

from doc_assistant.pipeline import build_chat_model, format_citation, format_docs_for_prompt


def test_build_chat_model_local_is_ollama_no_network():
    """A local provider yields an Ollama LLM; construction makes no network call."""
    llm = build_chat_model("ollama", "llama3.1:8b")
    assert type(llm).__name__ == "OllamaLLM"
    assert llm.model == "llama3.1:8b"


def test_build_chat_model_anthropic_is_chatanthropic_no_network():
    llm = build_chat_model("anthropic", "claude-haiku-4-5")
    assert type(llm).__name__ == "ChatAnthropic"


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
