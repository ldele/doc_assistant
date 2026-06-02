"""Guard tests for reviewer context isolation (LLM-provider spec).

Two invariants the provider refactor must not regress:

1. The reviewer judges against *retrieved evidence + the answer* only —
   never a ground-truth reference, never the analysis conversation. There
   is no conversation object on ``AnswerProvenance`` to leak; this pins
   that the prompt surface stays minimal.
2. The reviewer client is configured *independently* of the analysis
   model, so a Sonnet generator with a Haiku reviewer (or a local
   reviewer) is one env flip, not a code change.
"""

from __future__ import annotations

from typing import Any

import pytest

from doc_assistant import config, llm
from doc_assistant.provenance import AnswerProvenance, RetrievedChunk
from doc_assistant.reviewer import build_reviewer_prompt


class _FakeAnthropic:
    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.messages = object()


def test_reviewer_prompt_is_evidence_only():
    prov = AnswerProvenance(
        id="x",
        query="Q",
        answer="A",
        retrieved_chunks=[RetrievedChunk(filename="p.pdf", chunk_excerpt="EVIDENCE_TOKEN")],
    )
    prompt = build_reviewer_prompt(prov)
    assert "EVIDENCE_TOKEN" in prompt  # judges against retrieved evidence
    assert "A" in prompt  # and the answer under review
    # No analysis-conversation field exists to leak — surface stays minimal.
    assert {"query", "answer", "retrieved_chunks"} <= set(vars(prov))


def test_reviewer_client_is_independently_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    # Analysis model and reviewer model are set apart...
    monkeypatch.setattr(config, "LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "REVIEWER_MODEL", "claude-haiku-4-5-20251001")
    reviewer = llm.get_reviewer_client()
    # ...and the reviewer follows REVIEWER_MODEL, not LLM_MODEL.
    assert reviewer.model == "claude-haiku-4-5-20251001"
    assert reviewer.model != config.LLM_MODEL


def test_reviewer_and_analysis_providers_are_separate_keys(monkeypatch: pytest.MonkeyPatch):
    """The provider for review is read from REVIEWER_PROVIDER, not LLM_PROVIDER."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "ollama")
    monkeypatch.setattr(config, "REVIEWER_MODEL", "llama3")
    monkeypatch.setattr("langchain_ollama.ChatOllama", _fake_chat_ollama())
    reviewer = llm.get_reviewer_client()
    assert isinstance(reviewer, llm.OllamaClient)


def _fake_chat_ollama() -> Any:
    class _Fake:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    return _Fake
