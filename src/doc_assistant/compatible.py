"""One client for every OpenAI-compatible chat server (ADR-060, choices 1, 6 and 8).

Our own, over ``httpx``, which is already a dependency and already in the frozen build. It serves
both call shapes the app has (``docs/specs/llm-provider-isolation.md``): a ``langchain_core`` chat
model for the streaming answer path (:class:`CompatibleChatModel`) and an ``LLMClient`` for the
one-shot reviewer (:class:`CompatibleClient`), both over one request function.

**What is sent, and what never is (choice 6).** The body is :func:`request_body`, nothing else:
``model``, ``messages`` and ``stream``. The headers are :func:`request_headers`: ``Content-Type``,
plus ``Authorization: Bearer`` when a key is bound to this address. Never sent: ``temperature``, a
token cap under either name, ``stream_options``, ``response_format``, ``tools``, ``n``, ``seed``,
``logprobs``, ``user``, or a header naming this app. *Compatible is not identical*: Groq
answers 400 to some parameters and Claude to ``temperature``, so the body is the norm's minimum,
pinned by a test, rather than whatever a library sends this year.

**What is read.** :func:`read_stream` takes ``choices[0].delta.content`` when it is a string and
ignores everything else: role-only chunks, usage chunks, reasoning fields, SSE comments. A usage
object is kept when a server sends one unasked, for the prompt-cut check (choice 7).

**TLS** goes through the OS trust store in the frozen build, as KI-10's path does for Claude.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SecretStr

from doc_assistant.providers import Endpoint, ModelList, host_of, now_iso

log = structlog.get_logger(__name__)

CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"

ANSWER_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
"""A local model can think for minutes before its first visible token; the read timeout is the
longest pause between two chunks, not the length of an answer."""

LIST_TIMEOUT = 10.0
"""A model list from a cloud server: one GET, on a person's click."""

LOCAL_CONNECT_SECONDS = 0.5
"""How long a probe of a server on this computer waits to connect. A local server that listens
accepts at once; Windows retries a port nobody listens on for about 2 s per address, so without
this a probe of an LM Studio that is not running spent the whole budget (4 s measured)."""

# Fields under which a model list documents a model's window: OpenRouter's `context_length`,
# Groq's `context_window`, Mistral's `max_context_length` (their docs, checked 2026-11-02).
_WINDOW_FIELDS = ("context_length", "context_window", "max_context_length")


class CompatibleError(RuntimeError):
    """A compatible server answered with an error. ``reason`` is ours, safe to show: it never
    carries the request, the key or a page of HTML."""

    def __init__(self, reason: str, status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


# ============================================================
# The request
# ============================================================


def request_body(model: str, messages: list[dict[str, str]], *, stream: bool) -> dict[str, Any]:
    """The whole request body. Nothing else is ever sent (choice 6)."""
    return {"model": model, "messages": messages, "stream": stream}


def request_headers(key: str | None) -> dict[str, str]:
    """The whole set of headers we add. The key goes only where the caller's endpoint points."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def http_client(timeout: httpx.Timeout | float) -> httpx.Client:
    """An ``httpx`` client that verifies TLS against the OS trust store in the frozen build.

    Behind a TLS-inspecting proxy the proxy's root CA is in the OS store, not in ``certifi``
    (KI-10). A dev run keeps httpx's default, as ``llm.os_trust_http_client`` does for Claude.
    """
    if getattr(sys, "frozen", False):
        try:
            import ssl

            import truststore

            context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            return httpx.Client(timeout=timeout, verify=context)
        except Exception as e:  # truststore not bundled: certifi, as before
            log.info("os_trust_unavailable", error=str(e))
    return httpx.Client(timeout=timeout)


def _join(address: str, path: str) -> str:
    return address.strip().rstrip("/") + path


def _error_reason(status: int, text: str) -> str:
    """One sentence for an error response: the server's own message when it is JSON, else the
    status. An HTML page is never echoed."""
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    message = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
        message = message or payload.get("message") or payload.get("detail")
    if isinstance(message, str) and message.strip():
        return f"the server answered {status}: {message.strip()[:300]}"
    return f"the server answered {status}"


# ============================================================
# The stream reader
# ============================================================


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


StreamEvent = str | Usage


def read_stream(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """The text, and any usage, in a server-sent event stream of chat-completion chunks.

    Yields each ``choices[0].delta.content`` string that is not empty, then a :class:`Usage`
    when a chunk carries one. Ignores SSE comments (``:``), other SSE fields, role-only chunks,
    chunks with no ``choices`` and reasoning fields. Stops at ``[DONE]``. An error object
    mid-stream raises :class:`CompatibleError`, so a half answer never reads as a whole one.
    """
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except ValueError:
            log.warning("compatible_stream_unreadable_chunk", length=len(data))
            continue
        if not isinstance(chunk, dict):
            continue
        if "error" in chunk:
            reason = _error_reason(200, data)
            raise CompatibleError(reason.replace("the server answered 200", "the server stopped"))
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                yield content
        usage = chunk.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
            completion = usage.get("completion_tokens")
            yield Usage(usage["prompt_tokens"], completion if isinstance(completion, int) else 0)


def stream_chat(
    endpoint: Endpoint,
    model: str,
    messages: list[dict[str, str]],
    *,
    client: httpx.Client | None = None,
) -> Iterator[StreamEvent]:
    """POST one chat request to ``endpoint`` and yield what :func:`read_stream` reads.

    A server that cannot be reached raises ``httpx.ConnectError`` as it is (the turn's failure
    sentence matches it by name). A dropped connection mid-answer and an error status raise
    :class:`CompatibleError`.
    """
    owned = client is None
    http = client or http_client(ANSWER_TIMEOUT)
    try:
        with http.stream(
            "POST",
            _join(endpoint.address, CHAT_PATH),
            json=request_body(model, messages, stream=True),
            headers=request_headers(endpoint.key),
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise CompatibleError(
                    _error_reason(response.status_code, response.text), response.status_code
                )
            yield from read_stream(response.iter_lines())
    except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
        raise CompatibleError("the connection dropped before the answer ended") from e
    finally:
        if owned:
            http.close()


# ============================================================
# The model list (choice 8)
# ============================================================


def _parse_models(payload: Any) -> tuple[tuple[str, ...], dict[str, int]] | None:
    """Model ids and documented windows from a list in either shape seen: ``{"data": [...]}`` (the
    norm) or a bare array (Mistral's docs show both)."""
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    ids: list[str] = []
    windows: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            continue
        ids.append(row["id"])
        for name in _WINDOW_FIELDS:
            value = row.get(name)
            if isinstance(value, int) and value > 0:
                windows[row["id"]] = value
                break
    return tuple(sorted(set(ids))), windows


def list_models(
    address: str,
    key: str | None,
    *,
    key_check: str | None = None,
    timeout: float | httpx.Timeout = LIST_TIMEOUT,
    client: httpx.Client | None = None,
) -> ModelList:
    """``GET {address}/models``, as a :class:`ModelList`. Never raises; bills nothing.

    401 or 403, and 400 to a request that carried a key (a GET with no parameters can only be
    refused for its key), read as *refused*; without a key, as *needs a key*. 404 means the address
    is not a compatible server's. ``key_check`` is a path checked first, for a server whose model
    list needs no key (OpenRouter).
    """
    host = host_of(address) or address
    owned = client is None
    http = client or http_client(timeout)
    try:
        if key_check and key:
            checked = http.get(_join(address, key_check), headers=request_headers(key))
            if checked.status_code in (400, 401, 403):
                return ModelList("refused", detail="The server refused this key.")
        response = http.get(_join(address, MODELS_PATH), headers=request_headers(key))
    except Exception as e:
        log.info("compatible_list_unreachable", host=host, error=type(e).__name__)
        return ModelList("unreachable", detail=f"Could not reach {host}.", fetched_at=now_iso())
    finally:
        if owned:
            http.close()
    status = response.status_code
    if status in (401, 403) or (status == 400 and key):
        return ModelList(
            "refused" if key else "needs_key",
            detail="The server refused this key." if key else "This server needs a key.",
            fetched_at=now_iso(),
        )
    if status == 404:
        return ModelList(
            "error",
            detail=f"No model list at this address ({host} answered 404). Check the path, "
            "which usually ends in /v1.",
            fetched_at=now_iso(),
        )
    if status >= 400:
        return ModelList("error", detail=f"{host} answered {status}.", fetched_at=now_iso())
    try:
        parsed = _parse_models(response.json())
    except ValueError:
        parsed = None
    if parsed is None:
        return ModelList(
            "error", detail=f"{host} answered, but not with a model list.", fetched_at=now_iso()
        )
    models, windows = parsed
    return ModelList("ok", models, fetched_at=now_iso(), windows=windows)


# ============================================================
# The two call shapes
# ============================================================

_ROLES = {"human": "user", "ai": "assistant", "system": "system"}


def _as_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = _ROLES.get(m.type, "user")
        content = m.content if isinstance(m.content, str) else str(m.content)
        out.append({"role": role, "content": content})
    return out


class CompatibleChatModel(BaseChatModel):
    """The streaming answer path's chat model for a compatible server.

    ``model`` carries the model name, as ``ChatAnthropic`` and ``OllamaLLM`` do, so the controller
    records it without knowing which adapter ran. The key is held as a ``SecretStr`` and sent only
    to ``address``.
    """

    provider_id: str
    name_shown: str
    address: str
    model: str
    api_key: SecretStr | None = Field(default=None, repr=False)

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def _endpoint(self) -> Endpoint:
        key = self.api_key.get_secret_value() if self.api_key is not None else None
        return Endpoint(self.provider_id, self.name_shown, self.address, key)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for event in stream_chat(self._endpoint(), self.model, _as_messages(messages)):
            if isinstance(event, Usage):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        usage_metadata={
                            "input_tokens": event.prompt_tokens,
                            "output_tokens": event.completion_tokens,
                            "total_tokens": event.prompt_tokens + event.completion_tokens,
                        },
                    )
                )
                continue
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=event))
            if run_manager is not None:
                run_manager.on_llm_new_token(event, chunk=chunk)
            yield chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text: list[str] = []
        usage: Usage | None = None
        for event in stream_chat(self._endpoint(), self.model, _as_messages(messages)):
            if isinstance(event, Usage):
                usage = event
            else:
                text.append(event)
        message = AIMessage(content="".join(text))
        if usage is not None:
            message.usage_metadata = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            }
        return ChatResult(generations=[ChatGeneration(message=message)])


class CompatibleClient:
    """``LLMClient`` for a compatible server: the one-shot reviewer path.

    ``temperature`` and ``max_tokens`` are accepted, as the protocol requires, and never sent
    (choice 6): reviews on these providers sample at the server's default, which the panel's
    *unmeasured* line covers.
    """

    def __init__(self, endpoint: Endpoint, model: str) -> None:
        self.endpoint = endpoint
        self.model = model

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> str:
        text = "".join(
            e for e in stream_chat(self.endpoint, self.model, messages) if isinstance(e, str)
        )
        if not text.strip():
            log.warning("compatible_empty_completion", provider=self.endpoint.id, model=self.model)
        return text.strip()
