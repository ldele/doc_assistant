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


def test_source_headers_carry_no_square_brackets():
    """The prompt asks for `[3]`; the context must not show the model a bracketed number to copy.

    The header used to read `[Source 3: paper.pdf, page 4]`, and models copied that shape verbatim
    into their prose — `[Source 3: paper.pdf]` resolves to nothing, so a fully attributed answer
    rendered with every claim uncited (RG-012 failure, 2026-08-06). Removing the imitation target
    is only effective if it stays removed."""
    docs = [
        make_doc(content="one", filename="a.pdf"),
        make_doc(content="two", filename="b.pdf", page=5),
    ]
    formatted = format_docs_for_prompt(docs)
    assert "[" not in formatted and "]" not in formatted, formatted
    # The number must still be there, and still be unambiguous.
    assert "Source 1" in formatted
    assert "Source 2" in formatted


def test_a_bracket_in_the_passage_text_is_the_documents_own(monkeypatch):
    """Only the HEADER is guaranteed bracket-free — a technical passage may legitimately contain
    "[2]" as its own citation, and we neither strip nor escape it. Pinned so the assertion above is
    not mistaken for a promise about the whole context block."""
    docs = [make_doc(content="as shown in [2], BM25 is term-based", filename="a.pdf")]
    formatted = format_docs_for_prompt(docs)
    assert "[2]" in formatted  # untouched, by design
    assert formatted.startswith("Source 1 — a.pdf")
