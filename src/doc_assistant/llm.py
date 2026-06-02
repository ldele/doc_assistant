"""Normalized one-shot LLM protocol (Phase 6 — Feature 1, generation side).

The codebase has **two** LLM call shapes, not one:

* **Streaming chat** (RAG analysis answer) — stays a LangChain model, built
  in ``pipeline._build_llm()`` and driven with ``.stream(...)``.
* **One-shot JSON** (the reviewer agent and the eval LLM-judge) — a single
  request that returns a string to parse. This module owns *that* shape.

A single factory can't serve both: a LangChain chat model has no
``messages.create``, and the reviewer's old ``messages.create`` call was
Anthropic-only, so ``REVIEWER_PROVIDER=ollama`` would have crashed. So the
one-shot path moves behind a small normalized ``LLMClient.complete()``
protocol with Anthropic **and** Ollama adapters — this is what unlocks a
fully-local reviewer/judge once the calibration gate passes
(``tests/eval/TESTING.md``).

Locked design choices
---------------------

* **One method.** ``complete(messages, *, temperature, max_tokens) -> str``.
  Prompt construction, JSON parsing, retries, and cost tracking stay with
  the caller — the adapter only normalizes the transport.
* **Model lives in the client.** ``AnthropicClient(model)`` /
  ``OllamaClient(model)`` bake the model in, so callers (reviewer, judge)
  no longer pass a ``model=`` kwarg around.
* **No vendor SDK at module import.** ``anthropic`` is imported lazily
  inside ``AnthropicClient``; ``langchain_ollama`` inside ``OllamaClient``.
* **Reviewer and judge are pinned instruments.** They default to a fixed,
  version-recorded reference model (``REVIEWER_MODEL`` / ``JUDGE_MODEL``)
  so cross-run numbers stay comparable; moving them to local is a config
  flip, gated on calibration — never silent.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from doc_assistant import config

# A chat message in the normalized shape. ``role`` is one of
# "system" | "user" | "assistant"; ``content`` is plain text.
Message = dict[str, str]


@runtime_checkable
class LLMClient(Protocol):
    """A provider-agnostic one-shot completion client.

    ``complete`` takes a non-empty ``messages`` list, a non-negative
    ``temperature`` and a ``max_tokens >= 1``, and returns the model's
    text. It never returns ``None``; it raises on transport failure so
    the caller can record the error (as ``review_answer`` already does).
    """

    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str: ...


# ============================================================
# Anthropic-response text extraction (vendor-specific, so it lives here)
# ============================================================


def _extract_anthropic_text(response: Any) -> str:
    """Pull text from an Anthropic Messages response. Tolerates SDK shape drift."""
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if hasattr(first, "text"):
            return str(first.text)
        if isinstance(first, dict):
            return str(first.get("text", ""))
    if isinstance(content, str):
        return content
    return str(response)


# ============================================================
# Adapters
# ============================================================


class AnthropicClient:
    """``LLMClient`` over the raw Anthropic SDK (``messages.create``).

    System messages in ``messages`` are hoisted into the API's top-level
    ``system`` kwarg; only user/assistant turns go in the ``messages``
    array, as the SDK requires.
    """

    def __init__(self, model: str, *, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)

    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        response = self._client.messages.create(**kwargs)
        return _extract_anthropic_text(response).strip()


class OllamaClient:
    """``LLMClient`` over a local Ollama server via ``langchain_ollama``.

    No API key. ``host`` defaults to ``config.OLLAMA_HOST``. Messages are
    passed as ``(role, content)`` tuples, which LangChain maps to its
    message types reliably.

    Per-call ``temperature`` / ``max_tokens`` are applied as **model
    attributes** at construction (``temperature`` / ``num_predict``), NOT
    as ``invoke`` kwargs. langchain_ollama only folds its *known* params
    into Ollama's ``options`` dict when they are model attributes; passing
    them to ``invoke`` leaks them as raw kwargs to the ollama
    ``Client.chat()``, which rejects ``temperature`` (``TypeError:
    Client.chat() got an unexpected keyword argument 'temperature'``).
    A fresh client per call is cheap — construction does no network I/O.
    """

    def __init__(self, model: str, *, host: str | None = None) -> None:
        self.model = model
        self._host = host or config.OLLAMA_HOST

    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        from langchain_ollama import ChatOllama

        client = ChatOllama(
            model=self.model,
            base_url=self._host,
            temperature=temperature,
            num_predict=max_tokens,
            # This adapter serves only the one-shot JSON path (reviewer + eval
            # judge — see module docstring). Local models are far less reliable
            # than the API at returning a bare JSON object; Ollama's native JSON
            # mode constrains the output to valid JSON so the caller's
            # json.loads doesn't choke on prose or an empty completion.
            format="json",
        )
        lc_messages = [(m["role"], m["content"]) for m in messages]
        result = client.invoke(lc_messages)
        content = getattr(result, "content", result)
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        return str(content).strip()


# ============================================================
# Factory
# ============================================================


def make_client(provider: str, model: str) -> LLMClient:
    """Construct an ``LLMClient`` for ``provider`` (``anthropic`` | ``ollama``).

    Raises ``ValueError`` on an unknown provider — mirrors
    ``embeddings.get_model_config``.
    """
    key = provider.lower()
    if key == "anthropic":
        return AnthropicClient(model)
    if key == "ollama":
        return OllamaClient(model)
    raise ValueError(f"Unknown LLM provider '{provider}'. Valid options: anthropic, ollama")


def get_reviewer_client() -> LLMClient:
    """The pinned reviewer instrument — reads ``REVIEWER_PROVIDER``/``REVIEWER_MODEL``."""
    return make_client(config.REVIEWER_PROVIDER, config.REVIEWER_MODEL)


def get_judge_client() -> LLMClient:
    """The pinned eval-judge instrument — reads ``JUDGE_PROVIDER``/``JUDGE_MODEL``."""
    return make_client(config.JUDGE_PROVIDER, config.JUDGE_MODEL)


def reviewer_available() -> bool:
    """Whether the reviewer can run given current config.

    Anthropic needs ``ANTHROPIC_API_KEY``; Ollama (local) needs nothing.
    Call sites gate on this instead of hardcoding the API-key check, so a
    fully-local reviewer (``REVIEWER_PROVIDER=ollama``) works without a key.
    """
    if config.REVIEWER_PROVIDER.lower() == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    return True
