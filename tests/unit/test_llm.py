"""Tests for the normalized one-shot LLM protocol (``doc_assistant.llm``).

No network: the Anthropic SDK and ``langchain_ollama.ChatOllama`` are
monkeypatched. Covers the factory, both adapters' ``complete`` contract,
system-message hoisting, and the config-driven reviewer/judge selection.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from doc_assistant import config, llm

# ============================================================
# Fakes for the lazily-imported vendor SDKs
# ============================================================


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        block = type("Block", (), {"text": text})()
        self.content = [block]


class _FakeMessages:
    def __init__(self, sink: dict[str, Any]) -> None:
        self._sink = sink

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self._sink["kwargs"] = kwargs
        return _FakeAnthropicResponse("  hello from anthropic  ")


class _FakeAnthropic:
    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.sink: dict[str, Any] = {}
        self.messages = _FakeMessages(self.sink)


class _FakeChatOllamaResult:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChatOllama:
    last_init: ClassVar[dict[str, Any]] = {}
    last_invoke: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_init = kwargs

    def invoke(self, messages: Any, **kwargs: Any) -> _FakeChatOllamaResult:
        type(self).last_invoke = {"messages": messages, "kwargs": kwargs}
        return _FakeChatOllamaResult("  hello from ollama  ")


@pytest.fixture
def patched_sdks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)
    monkeypatch.setattr("langchain_ollama.ChatOllama", _FakeChatOllama)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")


# ============================================================
# Factory
# ============================================================


def test_make_client_anthropic(patched_sdks: None):
    client = llm.make_client("anthropic", "claude-haiku-4-5-20251001")
    assert isinstance(client, llm.AnthropicClient)
    assert client.model == "claude-haiku-4-5-20251001"


def test_make_client_ollama(patched_sdks: None):
    client = llm.make_client("ollama", "llama3")
    assert isinstance(client, llm.OllamaClient)
    assert client.model == "llama3"


def test_make_client_case_insensitive(patched_sdks: None):
    assert isinstance(llm.make_client("Anthropic", "m"), llm.AnthropicClient)
    assert isinstance(llm.make_client("OLLAMA", "m"), llm.OllamaClient)


def test_make_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm.make_client("openai", "gpt-4")


def test_adapters_satisfy_protocol(patched_sdks: None):
    assert isinstance(llm.make_client("anthropic", "m"), llm.LLMClient)
    assert isinstance(llm.make_client("ollama", "m"), llm.LLMClient)


# ============================================================
# AnthropicClient.complete
# ============================================================


def test_anthropic_complete_returns_stripped_text(patched_sdks: None):
    client = llm.make_client("anthropic", "m")
    out = client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=50)
    assert out == "hello from anthropic"


def test_anthropic_complete_passes_signature(patched_sdks: None):
    client = llm.make_client("anthropic", "m")
    client.complete([{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=77)
    kwargs = client._client.sink["kwargs"]  # type: ignore[attr-defined]
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 77
    assert kwargs["model"] == "m"
    assert "system" not in kwargs  # no system message → no system kwarg
    assert len(kwargs["messages"]) == 1


def test_anthropic_complete_hoists_system_message(patched_sdks: None):
    client = llm.make_client("anthropic", "m")
    client.complete(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U"},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    kwargs = client._client.sink["kwargs"]  # type: ignore[attr-defined]
    assert kwargs["system"] == "SYS"
    # The system turn is hoisted out of the messages array.
    assert [m["role"] for m in kwargs["messages"]] == ["user"]


# ============================================================
# OllamaClient.complete
# ============================================================


def test_ollama_complete_returns_stripped_text(patched_sdks: None):
    client = llm.make_client("ollama", "llama3")
    out = client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=50)
    assert out == "hello from ollama"


def test_ollama_complete_passes_role_content_tuples(patched_sdks: None):
    client = llm.make_client("ollama", "llama3")
    client.complete(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        temperature=0.0,
        max_tokens=10,
    )
    sent = _FakeChatOllama.last_invoke["messages"]
    assert sent == [("system", "S"), ("user", "U")]


def test_ollama_complete_sets_params_on_model_not_invoke(patched_sdks: None):
    """Regression: temperature/num_predict must be set as model attributes,
    not passed to invoke(). Passing them to invoke leaks them to the ollama
    Client.chat() call, which raises
    ``TypeError: Client.chat() got an unexpected keyword argument 'temperature'``.
    """
    client = llm.make_client("ollama", "llama3")
    client.complete([{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=77)
    init = _FakeChatOllama.last_init
    assert init["temperature"] == 0.3
    assert init["num_predict"] == 77
    assert init["model"] == "llama3"
    # invoke() must NOT carry these — that is exactly what broke against a
    # live ollama server.
    assert _FakeChatOllama.last_invoke["kwargs"] == {}


# ============================================================
# Config-driven reviewer / judge selection
# ============================================================


def test_get_reviewer_client_reads_config(patched_sdks: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "REVIEWER_MODEL", "claude-sonnet-4-6")
    client = llm.get_reviewer_client()
    assert isinstance(client, llm.AnthropicClient)
    assert client.model == "claude-sonnet-4-6"


def test_get_judge_client_reads_config(patched_sdks: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "ollama")
    monkeypatch.setattr(config, "JUDGE_MODEL", "llama3")
    client = llm.get_judge_client()
    assert isinstance(client, llm.OllamaClient)
    assert client.model == "llama3"


# ============================================================
# reviewer_available
# ============================================================


def test_reviewer_available_anthropic_needs_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    assert llm.reviewer_available() is False
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    assert llm.reviewer_available() is True


def test_reviewer_available_ollama_needs_no_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "ollama")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    assert llm.reviewer_available() is True
