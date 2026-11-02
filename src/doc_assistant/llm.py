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
* **Ollama reasoning is off by default.** This adapter's whole job is to
  return a short JSON object, and on a thinking model the reasoning is
  emitted first and eats the same ``num_predict`` budget. Measured against
  ``/api/chat`` with ``qwen3.5:9b`` and the taxonomy pass's own 256-token
  budget: think on → ``done_reason="length"``, ``message.content == ""``;
  think off → ``{"choice": 3, "confidence": 1}`` in 14 tokens. Because
  ``message.thinking`` is a separate field the adapter never read, this
  failed as an *empty string*, so the caller logged "unparseable" and a
  perfectly capable model read as an incapable one. Off is therefore the
  right default for a one-shot JSON call; a caller that wants the trace
  passes ``reasoning=True`` and must raise ``max_tokens`` to match.
  ``think`` is accepted by non-thinking models too (verified on
  ``llama3.1:8b`` and ``qwen2.5:7b``), so this is safe across the board.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from doc_assistant import config, credentials

if TYPE_CHECKING:
    import httpx

log = structlog.get_logger(__name__)

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
# Claude model lines (the API differs across them, so it lives here)
# ============================================================

# The Claude models before Opus 4.7: the ones that take ``temperature``, and that do not think
# unless asked. Listed this way round because a model it does not know is newer, and a request with
# no ``temperature`` is valid on every model, while one that carries it is a 400 on all of these:
# Opus 4.7 and later, Sonnet 5 and later, Haiku 5.5 and Fable.
_OLDER_CLAUDE = re.compile(r"claude-3-.+|claude-(opus|sonnet|haiku)-4(-[0-6])?(-\d{8}|-latest)?")

# Added to ``max_tokens`` on a newer model. From Claude Opus 5 thinking is on by default and spends
# the same budget as the reply, which took a 400-token review to no text at all. A ceiling, not a
# spend: only the tokens used are billed.
THINKING_ROOM = 4096


def is_older_claude(model: str) -> bool:
    """Whether ``model`` is from before Opus 4.7 (see :data:`_OLDER_CLAUDE`)."""
    return _OLDER_CLAUDE.fullmatch(model) is not None


def max_tokens_for(model: str, reply_tokens: int) -> int:
    """The ``max_tokens`` that leaves ``model`` room for a reply of ``reply_tokens``."""
    return reply_tokens if is_older_claude(model) else reply_tokens + THINKING_ROOM


# ============================================================
# Anthropic-response text extraction (vendor-specific, so it lives here)
# ============================================================


def content_text(content: Any) -> str:
    """The text in a model's content: a string, or a list of blocks of any shape.

    A block with no text counts for nothing. A model that thinks puts its thinking block first, so
    the text can be anywhere in the list, and a list is what the chat path gets from such a model
    where it expects a string: a one-element chunk for each thinking event.
    """
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text", "")))
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return "".join(parts)


def _extract_anthropic_text(response: Any) -> str:
    """Pull text from an Anthropic Messages response. Tolerates SDK shape drift."""
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if isinstance(content, (list, str)):
        return content_text(content)
    return str(response)


# ============================================================
# OS-trust HTTP client (KI-10 — corporate TLS-MITM proxy)
# ============================================================


def os_trust_http_client() -> httpx.Client | None:
    """Build an httpx client that verifies outbound TLS against the OS trust store.

    Behind a TLS-inspecting (MITM) corporate proxy the proxy's root CA lives in the
    OS/system trust store, not in the ``certifi`` bundle the Anthropic SDK's httpx
    client pins — so a frozen build SSL-fails the Anthropic call with
    ``CERTIFICATE_VERIFY_FAILED`` (KI-10). Handing the SDK a client whose ``verify``
    context is truststore-backed routes verification through the OS store, and does
    so *inside* ``AnthropicClient`` — it does not depend on
    ``truststore.inject_into_ssl()``'s process-global monkeypatch surviving the
    PyInstaller freeze (the confirmed failure mode; ``docs/desktop-packaging.md`` §KI-10,
    branch B).

    ``DefaultHttpxClient`` (the SDK's own subclass) is used rather than a bare
    ``httpx.Client`` so the SDK's default timeouts / connection limits are preserved
    while the OS-trust ``verify`` context is layered on.

    Returns ``None`` (→ caller uses the SDK's default certifi client) in two cases:
    a **dev / non-frozen** run, where certifi is correct and the entrypoint's
    ``truststore.inject_into_ssl()`` already covers an on-proxy dev turn; or when
    ``truststore`` / the SDK's ``DefaultHttpxClient`` is unavailable. The OS-trust
    client is therefore built **only in the frozen build**, exactly where KI-10
    bites — dev and test behaviour is unchanged. Off-proxy frozen use is unaffected
    too, since the OS store is a superset of certifi's public CAs.
    """
    if not getattr(sys, "frozen", False):
        # dev / tests: SDK default (certifi); on-proxy dev is covered by the entrypoint inject.
        return None
    try:
        import ssl

        import truststore
        from anthropic import DefaultHttpxClient
    except Exception as exc:  # truststore/anthropic not bundled → fall back to certifi
        log.info("os_trust_http_client_unavailable", error=str(exc))
        return None
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return DefaultHttpxClient(verify=ctx)


# ============================================================
# Adapters
# ============================================================


class AnthropicClient:
    """``LLMClient`` over the raw Anthropic SDK (``messages.create``).

    System messages in ``messages`` are hoisted into the API's top-level
    ``system`` kwarg; only user/assistant turns go in the ``messages``
    array, as the SDK requires.

    On a model from Opus 4.7 on (:func:`is_older_claude`) ``temperature`` is not sent, because the
    API refuses it, so that call samples at the default. It asks for low effort and gets
    :data:`THINKING_ROOM` instead: this is a short JSON reply, and the thinking must not spend it.
    The reviewer and judge pinned to Haiku 4.5 keep their temperature 0.
    """

    def __init__(self, model: str, *, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        self.model = model
        if not is_older_claude(model):
            log.info("anthropic_temperature_not_sent", model=model)
        # Resolved at construction, not import: an in-app key (ADR-034) can be saved while the
        # process is running, and every client built after that must see it.
        kwargs: dict[str, Any] = {"api_key": api_key or credentials.resolve_key("anthropic")}
        http_client = os_trust_http_client()
        if http_client is not None:  # OS-trust TLS for corporate MITM proxies (KI-10)
            kwargs["http_client"] = http_client
        self._client = Anthropic(**kwargs)

    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens_for(self.model, max_tokens),
            "messages": convo,
        }
        if is_older_claude(self.model):
            kwargs["temperature"] = temperature
        else:
            kwargs["output_config"] = {"effort": "low"}
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

    ``reasoning`` maps to Ollama's ``think`` and **defaults to False** — see
    the module docstring's locked choice. Pass ``reasoning=None`` to restore
    the model's own default, or ``True`` to keep the trace (langchain then
    files it under ``additional_kwargs["reasoning_content"]`` and leaves the
    content clean, at the cost of generating it).

    ``num_ctx`` is the context window asked of the server, in tokens. ``None``
    leaves Ollama's own default, which depends on the GPU's memory (4,096 on a
    12 GB card), and a prompt longer than the window is cut from the front
    with only a server-log warning: the call succeeds on what is left (KI-65).
    A caller whose prompt grows with its input sets it and keeps the prompt
    plus ``max_tokens`` inside it.
    """

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        reasoning: bool | None = False,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self._host = host or config.OLLAMA_HOST
        self.reasoning = reasoning
        self.num_ctx = num_ctx

    @property
    def host(self) -> str:
        return self._host

    def with_context(self, num_ctx: int) -> OllamaClient:
        """A copy of this client that asks the server for a ``num_ctx``-token window."""
        return OllamaClient(self.model, host=self._host, reasoning=self.reasoning, num_ctx=num_ctx)

    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        from langchain_ollama import ChatOllama

        client = ChatOllama(
            model=self.model,
            base_url=self._host,
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=self.num_ctx,  # None = the server's default (langchain drops it)
            # This adapter serves only the one-shot JSON path (reviewer + eval
            # judge — see module docstring). Local models are far less reliable
            # than the API at returning a bare JSON object; Ollama's native JSON
            # mode constrains the output to valid JSON so the caller's
            # json.loads doesn't choke on prose or an empty completion.
            format="json",
            reasoning=self.reasoning,
        )
        lc_messages = [(m["role"], m["content"]) for m in messages]
        result = client.invoke(lc_messages)
        content = getattr(result, "content", result)
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        text = str(content).strip()
        if not text:
            # An empty completion is the one failure this adapter must never
            # return silently: every caller parses the string, so "" surfaces
            # downstream as "the model gave an unusable answer" when the real
            # cause is upstream (budget exhausted, reasoning re-enabled, a
            # server-side stop). Name it here or it gets misdiagnosed as model
            # incompetence — which is exactly what happened before this guard.
            log.warning(
                "ollama_empty_completion",
                model=self.model,
                max_tokens=max_tokens,
                reasoning=self.reasoning,
            )
        return text


# ============================================================
# Factory
# ============================================================


def make_client(provider: str, model: str) -> LLMClient:
    """Construct an ``LLMClient`` for ``provider``: ``anthropic``, ``ollama``, or an
    OpenAI-compatible preset or server of your own (ADR-060).

    An Ollama client asks for ``config.OLLAMA_CONTEXT_TOKENS``: the server's own default window
    cuts a long reviewer or judge prompt from the front without an error (KI-65's cause, on
    every one-shot caller). A caller that budgets its own prompt narrows it with
    ``with_context`` (the wiki's ``fit_window``). A compatible client is built with the key bound
    to its address, read at construction.

    Raises ``ValueError`` on an unknown provider — mirrors
    ``embeddings.get_model_config``.
    """
    from doc_assistant import providers

    key = provider.strip().lower()
    if key == "anthropic":
        return AnthropicClient(model)
    if key == "ollama":
        return OllamaClient(model, num_ctx=config.OLLAMA_CONTEXT_TOKENS)
    if providers.is_compatible(key):
        from doc_assistant.compatible import CompatibleClient

        return CompatibleClient(providers.endpoint(key), model)
    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        f"Valid options: {', '.join(providers.answer_provider_ids())}"
    )


def resolve_reviewer(
    effective_provider: str | None = None, effective_model: str | None = None
) -> tuple[str, str]:
    """Resolve the reviewer's effective ``(provider, model)`` (ADR-011, U1c).

    With no args: today's behaviour — ``REVIEWER_PROVIDER``/``REVIEWER_MODEL`` (an explicit
    ``.env`` pin, or its default of ``LLM_PROVIDER`` + the pinned reference model).

    A caller passes the *live* generation ``(effective_provider, effective_model)`` to make the
    reviewer **follow** a provider switch (fork C) — but only when ``REVIEWER_PROVIDER`` was never
    explicitly pinned in the environment (:data:`config.REVIEWER_PROVIDER_PINNED`); an explicit pin
    always wins, preserving the cross-run comparability ``REVIEWER_MODEL`` exists for.
    ``REVIEWER_MODEL`` (a Haiku name) would fail on Ollama, so a followed switch uses the effective
    **chat** model, not it. Exposed separately from :func:`get_reviewer_client` so a caller can log
    /persist which instrument actually ran without rebuilding one just to inspect it.
    """
    if config.REVIEWER_PROVIDER_PINNED or effective_provider is None:
        return config.REVIEWER_PROVIDER, config.REVIEWER_MODEL
    return effective_provider, effective_model or config.REVIEWER_MODEL


def get_reviewer_client(
    effective_provider: str | None = None, effective_model: str | None = None
) -> LLMClient:
    """The reviewer instrument — see :func:`resolve_reviewer` for the resolution rule."""
    provider, model = resolve_reviewer(effective_provider, effective_model)
    return make_client(provider, model)


def get_judge_client() -> LLMClient:
    """The pinned eval-judge instrument — reads ``JUDGE_PROVIDER``/``JUDGE_MODEL``."""
    return make_client(config.JUDGE_PROVIDER, config.JUDGE_MODEL)


def provider_available(provider: str) -> bool:
    """Whether ``provider`` is *configured* — i.e. its credential is present.

    Anthropic needs a key — ``.env`` or the in-app store (:func:`credentials.resolve_key`,
    ADR-034); Ollama (local) needs none, so it is always "configured". A compatible preset that
    takes a key needs one bound to its address (ADR-060); a server of your own does not. A
    provider this build does not know is never available. Generalizes the check
    ``reviewer_available`` already did (ADR-011 U1c) — reused by the settings-view provider list
    (fork E) and ``app_settings.set_llm_selection`` (never persist a choice that can't run).

    Deliberately **not** a reachability check: a local Ollama server that is merely not running
    yet must not make the selection invalid (inform-don't-block). Reachability is a separate,
    network-touching question — :func:`ollama_probe`, aggregated by :mod:`doc_assistant.readiness`.
    """
    from doc_assistant import providers

    key = provider.strip().lower()
    if key == "anthropic":
        return bool(credentials.resolve_key("anthropic"))
    if key == "ollama":
        return True
    if providers.is_compatible(key):
        # A preset that takes a key needs one bound to its address; a server's key is optional.
        return not providers.takes_key(key) or providers.endpoint(key).key is not None
    return False


def _same_ollama_model(a: str, b: str) -> bool:
    """Ollama lists ``llama3`` as ``llama3:latest``; either spelling names the same model."""
    return a == b or f"{a}:latest" == b or a == f"{b}:latest"


def model_mismatch(
    provider: str, model: str, installed: tuple[str, ...] | None = None
) -> str | None:
    """Why ``model`` cannot answer on ``provider``, as one sentence; ``None`` when it can, or when
    that cannot be told.

    The two providers name their models apart — every Anthropic model is ``claude-…`` and Ollama
    serves none of them — so a Claude name on Ollama, or any other name on Anthropic, can never
    answer. The 0.7.0 walkthrough's sandbox sat on Ollama with ``claude-haiku-4-5-…`` after a
    provider switch kept the old model. ``installed`` (an Ollama probe's list) adds the one other
    certain case, a server that answers without that model; an empty or absent list proves
    nothing, so it never refuses.
    """
    p = provider.strip().lower()
    name = model.strip()
    is_claude = name.lower().startswith("claude")
    if p == "anthropic" and not is_claude:
        return f"“{name}” is not a Claude model: Anthropic's models are named claude-…."
    if p == "ollama" and is_claude:
        return f"“{name}” is a Claude model, and Ollama does not serve Claude models."
    if p == "ollama" and installed and not any(_same_ollama_model(name, m) for m in installed):
        shown = ", ".join(installed[:6]) + (", …" if len(installed) > 6 else "")
        return f"Ollama has no model “{name}” installed; it has {shown}."
    return None


def reviewer_available(provider: str | None = None) -> bool:
    """Whether the reviewer can run.

    With no args: today's behaviour (checks ``REVIEWER_PROVIDER``). Call sites gate on this
    instead of hardcoding the API-key check, so a fully-local reviewer
    (``REVIEWER_PROVIDER=ollama``) works without a key. ADR-011: pass the *effective* provider to
    check availability for a followed switch instead of the pinned default.
    """
    return provider_available(provider if provider is not None else config.REVIEWER_PROVIDER)


# ============================================================
# Provider probes (first-run setup — ADR-034)
# ============================================================
# `provider_available` answers "is a credential configured?" from local state alone. These two
# answer the *other* first-run question — "will a turn actually work?" — and they touch the
# network, so they are separate functions a caller opts into, never called on the answer path.
# Aggregation + the user-facing next step live in `doc_assistant.readiness`.

PROBE_TIMEOUT_SECONDS = 2.0
"""Wall-clock budget for one setup probe. Structural, not tuned: a first-run panel must answer
within a keystroke's patience, and both probes talk to a local server or a single cloud GET."""


def ollama_probe(
    host: str | None = None, *, timeout: float = PROBE_TIMEOUT_SECONDS
) -> tuple[bool, tuple[str, ...], str | None]:
    """Ask a local Ollama server what it has installed.

    Returns ``(reachable, models, detail)``: ``models`` are the installed model tags (e.g.
    ``("llama3.1:8b", "qwen3.5:9b")``) and ``detail`` is a user-facing reason when
    ``reachable`` is False. Never raises — an unreachable local server is the *normal* state
    before the user installs Ollama, not an error condition.

    It also keeps each model's ``remote_host`` (ADR-060's amendment): a signed-in Ollama serves
    cloud models, which its ``/api/tags`` marks with the *"URL of the upstream Ollama host, if the
    model is remote"*, and the text for such a model goes there. Read back with
    :func:`ollama_remote_host`; in memory only, replaced by every probe of the same host.
    """
    import httpx

    base = (host or config.OLLAMA_HOST).rstrip("/")
    try:
        response = httpx.get(f"{base}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        log.info("ollama_probe_failed", host=base, error=str(e))
        return False, (), f"No Ollama server answering at {base}"
    raw = payload.get("models") if isinstance(payload, dict) else None
    models: list[str] = []
    remote: dict[str, str] = {}
    if isinstance(raw, list):
        for entry in raw:
            name = entry.get("name") or entry.get("model") if isinstance(entry, dict) else None
            if isinstance(name, str) and name:
                models.append(name)
                upstream = entry.get("remote_host")
                if isinstance(upstream, str) and upstream.strip():
                    remote[name] = upstream.strip()
    _OLLAMA_REMOTE[base] = remote
    if not models:
        # Reachable but empty is its own state, and the fix is different (pull a model, not
        # start the server) — so it must not be reported as "unreachable".
        return True, (), f"Ollama is running at {base} but has no models installed"
    return True, tuple(sorted(models)), None


_OLLAMA_REMOTE: dict[str, dict[str, str]] = {}
"""host -> {model: remote_host}, from the last :func:`ollama_probe` of that host."""


def ollama_remote_host(model: str, host: str | None = None) -> str | None:
    """The upstream host the last probe reported for ``model``, or ``None`` (not remote, or not
    probed yet). No network."""
    base = (host or config.OLLAMA_HOST).rstrip("/")
    remote = _OLLAMA_REMOTE.get(base, {})
    return next((r for name, r in remote.items() if _same_ollama_model(model, name)), None)


def ollama_context_length(
    model: str, host: str | None = None, *, timeout: float = PROBE_TIMEOUT_SECONDS
) -> int | None:
    """The context length ``model`` was trained for, as the Ollama server reports it.

    Read from ``/api/show``'s ``model_info`` (the key is ``<architecture>.context_length``,
    e.g. ``qwen2.context_length``). This is the most a ``num_ctx`` request can get: Ollama
    clamps a larger one to it. Returns ``None`` when the server is unreachable, the model is
    not installed, or the field is absent. Never raises.
    """
    import httpx

    base = (host or config.OLLAMA_HOST).rstrip("/")
    try:
        response = httpx.post(f"{base}/api/show", json={"model": model}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        log.info("ollama_show_failed", host=base, model=model, error=str(e))
        return None
    info = payload.get("model_info") if isinstance(payload, dict) else None
    if not isinstance(info, dict):
        return None
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None


def verify_anthropic_key(key: str, *, timeout: float = PROBE_TIMEOUT_SECONDS) -> tuple[str, str]:
    """Check an Anthropic key with a **free** call, returning ``(status, detail)``.

    ``status`` is one of:

    * ``"ok"`` — the key authenticated.
    * ``"invalid"`` — the API rejected it (401/403). A caller storing keys must refuse this one.
    * ``"unreachable"`` — no verdict: no network, a proxy, a timeout, an SDK that is absent. The
      key may well be fine, so a caller stores it and says it could not be checked
      (inform-don't-block) rather than discarding what the user typed.

    Uses ``models.list()`` — a metadata GET that consumes **no tokens and bills nothing** — so
    first-run verification can never surprise a user with a charge (KI-4's discipline applied to
    the setup path).
    """
    cleaned = key.strip()
    if not cleaned:
        return "invalid", "The key is empty."
    try:
        from anthropic import Anthropic
    except Exception as e:  # pragma: no cover - the SDK is a declared dependency
        return "unreachable", f"The Anthropic SDK is unavailable ({e})."
    kwargs: dict[str, Any] = {"api_key": cleaned, "timeout": timeout, "max_retries": 0}
    http_client = os_trust_http_client()
    if http_client is not None:  # OS-trust TLS for corporate MITM proxies (KI-10)
        kwargs["http_client"] = http_client
    try:
        Anthropic(**kwargs).models.list(limit=1)
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        if status_code in (401, 403):
            log.info("anthropic_key_rejected", status_code=status_code)
            return "invalid", "The API rejected this key. Check it in the Anthropic Console."
        log.info("anthropic_key_unverified", error=str(e))
        return "unreachable", f"Could not reach the Anthropic API to check the key ({e})."
    return "ok", "Key verified."


def list_claude_models(key: str, *, timeout: float = 10.0) -> Any:
    """The Claude models this key can use, from the Models API, as a ``providers.ModelList``.

    ``GET /v1/models`` through the SDK: free, no tokens (KI-4's discipline, as
    :func:`verify_anthropic_key`). A list proves the key and the model, never the balance: on
    2026-11-01 this machine's key listed 14 models and a message on it was refused for lack of
    credit. Never raises.
    """
    from doc_assistant.providers import ModelList, now_iso

    try:
        from anthropic import Anthropic
    except Exception as e:  # pragma: no cover - the SDK is a declared dependency
        return ModelList("unreachable", detail=f"The Anthropic SDK is unavailable ({e}).")
    kwargs: dict[str, Any] = {"api_key": key.strip(), "timeout": timeout, "max_retries": 0}
    http_client = os_trust_http_client()
    if http_client is not None:  # OS-trust TLS for corporate MITM proxies (KI-10)
        kwargs["http_client"] = http_client
    try:
        models = tuple(sorted(m.id for m in Anthropic(**kwargs).models.list(limit=100)))
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        if status_code in (401, 403):
            return ModelList("refused", detail="Anthropic refused this key.", fetched_at=now_iso())
        log.info("claude_models_unlisted", error=type(e).__name__)
        return ModelList(
            "unreachable", detail="Could not reach the Anthropic API.", fetched_at=now_iso()
        )
    return ModelList("ok", models, fetched_at=now_iso())


# ============================================================
# Enrichment-CLI cost guard (the 2026-06-15 credit-burn footgun)
# ============================================================
# Provider *behaviour* lives here next to make_client/reviewer_available; the *policy* is one test,
# ADR-060's amendment: a run is guarded when its provider bills, or when who bills is not known
# (`providers.Billing.guard_trips`). Every enrichment CLI that does an ``--apply`` run routes
# through this one helper so a run the user believes is "local" can never silently spend.


class ProviderCostError(RuntimeError):
    """A paid-provider ``--apply`` run cannot proceed (e.g. missing API key).

    Carries a user-facing, already-translated message; CLIs print it and exit
    non-zero rather than letting the failure surface deep in the vendor SDK.
    """


def _assume_yes() -> bool:
    """Whether to skip the interactive abort window (automation / CI)."""
    return os.getenv("DOC_ASSUME_YES", "").strip().lower() in {"1", "true", "yes", "on"}


def assert_provider_intent(
    provider: str,
    *,
    operation: str,
    apply: bool = True,
    model: str | None = None,
    scope: str | None = None,
    abort_seconds: float = 3.0,
) -> None:
    """Guard a CLI enrichment run from *silently* spending API credits.

    No-op when ``apply`` is False (dry runs never bill) or when nobody bills ``provider``'s
    ``model``: an Ollama on this computer serving the model itself, or a provider the user ticked
    as not billing. Anything billed **or not known** trips it (ADR-060's amendment):

    * raises :class:`ProviderCostError` if the provider's credential is missing
      (currently: ``anthropic`` without ``ANTHROPIC_API_KEY``), so ``--apply``
      fails loudly up front instead of erroring mid-batch in the SDK;
    * otherwise prints a bordered banner to **stderr** — naming the operation, provider, model,
      the host the text goes to and who bills — then sleeps ``abort_seconds`` so a ``Ctrl-C``
      cleanly aborts *before* any call. Set ``DOC_ASSUME_YES=1`` (or pass ``abort_seconds=0``) to
      skip only the pause in automation; the banner still prints.

    With ``model`` on Ollama, the server is asked once which models are remote: a signed-in Ollama
    on this computer serves cloud models that ollama.com bills. An Ollama that does not answer
    cannot forward anything, so it reads by its address.

    Generalises the inline guard first shipped in ``build_concept_graph``
    (Feature 7) so every enrichment CLI shares one code path. ASCII-only output
    (Windows stderr may be cp1252; an emoji would raise UnicodeEncodeError).
    """
    if not apply:
        return
    from doc_assistant import providers

    key = provider.strip().lower()
    if key == "ollama" and model:
        ollama_probe()  # records each model's remote_host
    facts = providers.describe(key, model)
    if not facts.billing.guard_trips:
        return  # nobody bills this run

    if key == "anthropic" and not credentials.resolve_key("anthropic"):
        raise ProviderCostError(
            f"{operation}: --apply with --provider anthropic needs ANTHROPIC_API_KEY "
            "in your .env — or a key saved in the desktop app's Setup panel "
            "(or run a local provider, e.g. --provider ollama)."
        )

    billed = facts.billing.kind == "vendor"
    border = "=" * 72
    lines = [
        "",
        border,
        f"  WARNING: {'PAID API RUN' if billed else 'BILLING NOT KNOWN'} -- {operation}",
        f"  Provider : {key}" + (f"   Model: {model}" if model else ""),
        f"  Sends to : {facts.route.host or 'an address that could not be read'}",
        f"  Billed by: {facts.billing.vendor if billed else 'not known'}",
    ]
    if scope:
        lines.append(f"  Scope    : {scope}")
    if billed:
        lines.append(f"  This spends real {facts.billing.vendor} credits.")
    else:
        lines.append("  Provenote cannot see whether this server bills. It counts as billed.")
    lines.append("  An Ollama model on this computer is free.")
    if abort_seconds > 0 and not _assume_yes():
        lines.append(f"  Ctrl-C now to abort -- continuing in {abort_seconds:.0f}s...")
    lines.append(border)
    # A deliberate, formatted stderr block the user reads to decide whether to Ctrl-C
    # before a paid run — an interactive CLI safety prompt, not an observability event,
    # so it stays a direct stderr write (ADR-003 ADR-B: preserve stderr semantics) rather
    # than collapsing into a structlog line.
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()

    if abort_seconds > 0 and not _assume_yes():
        time.sleep(abort_seconds)
