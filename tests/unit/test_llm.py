"""Tests for the normalized one-shot LLM protocol (``doc_assistant.llm``).

No network: the Anthropic SDK and ``langchain_ollama.ChatOllama`` are
monkeypatched. Covers the factory, both adapters' ``complete`` contract,
system-message hoisting, and the config-driven reviewer/judge selection.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import httpx
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
    def __init__(self, *, api_key: str | None = None, http_client: Any = None) -> None:
        self.api_key = api_key
        self.http_client = http_client
        self.sink: dict[str, Any] = {}
        self.messages = _FakeMessages(self.sink)


class _FakeChatOllamaResult:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChatOllama:
    last_init: ClassVar[dict[str, Any]] = {}
    last_invoke: ClassVar[dict[str, Any]] = {}
    #: what the fake server returns — overridden to reproduce the empty completion a
    #: thinking model produces when its reasoning exhausts ``num_predict``.
    content: ClassVar[str] = "  hello from ollama  "

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_init = kwargs

    def invoke(self, messages: Any, **kwargs: Any) -> _FakeChatOllamaResult:
        type(self).last_invoke = {"messages": messages, "kwargs": kwargs}
        return _FakeChatOllamaResult(type(self).content)


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
# OS-trust HTTP client (KI-10 — corporate TLS-MITM proxy)
# ============================================================


def test_anthropic_client_uses_os_trust_context_when_truststore_present(
    patched_sdks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In the frozen build, when ``truststore`` is importable, the SDK is handed an
    http client whose ``verify`` context is the OS-trust one (KI-10 branch B)."""
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)  # simulate the frozen build
    sentinel_ctx = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr("truststore.SSLContext", lambda _proto: sentinel_ctx)

    def fake_default_httpx(*, verify: Any, **_kw: Any) -> Any:
        captured["verify"] = verify
        return object()

    monkeypatch.setattr("anthropic.DefaultHttpxClient", fake_default_httpx)

    client = llm.make_client("anthropic", "m")
    # The SDK received an explicit OS-trust http client (not the certifi default)...
    assert client._client.http_client is not None  # type: ignore[attr-defined]
    # ...built from the truststore-backed verify context.
    assert captured["verify"] is sentinel_ctx


def test_anthropic_client_falls_back_cleanly_when_truststore_absent(
    patched_sdks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even in the frozen build, if ``truststore`` cannot be imported, construction
    still succeeds and no custom http client is handed to the SDK — it uses its own
    default (certifi). No live paid call (cpc §13); construction only."""
    import builtins
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)  # frozen, but truststore missing
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "truststore":
            raise ImportError("truststore unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = llm.make_client("anthropic", "m")
    assert client._client.http_client is None  # type: ignore[attr-defined]


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
# OllamaClient reasoning (thinking models)
# ============================================================


class _FakeLog:
    """Records ``.warning`` events; every other structlog method is a chainable no-op — so the
    assertion doesn't depend on structlog being bridged to stdlib logging."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **kw: Any) -> None:
        self.warnings.append((event, kw))

    def __getattr__(self, _name: str):  # info/debug/error/bind/... → chainable no-op
        return lambda *a, **k: self


def test_ollama_disables_reasoning_by_default(patched_sdks: None):
    """Regression: a thinking model emits its reasoning into ``message.thinking`` and burns the
    same ``num_predict`` budget, so with reasoning left at the model's default the adapter got
    ``content == ""`` and every caller logged "unparseable" — a working model reading as a broken
    one. Verified against a live server with qwen3.5:9b at the taxonomy pass's 256-token budget.
    """
    client = llm.make_client("ollama", "qwen3.5:9b")
    client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=256)
    assert _FakeChatOllama.last_init["reasoning"] is False


@pytest.mark.parametrize("reasoning", [True, None])
def test_ollama_reasoning_is_overridable(patched_sdks: None, reasoning: bool | None):
    """A caller that wants the trace (or the model's own default) can still ask for it."""
    client = llm.OllamaClient("qwen3.5:9b", reasoning=reasoning)
    client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=2048)
    assert _FakeChatOllama.last_init["reasoning"] is reasoning


def test_ollama_empty_completion_warns(
    patched_sdks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty completion must be named, not returned silently.

    Every caller parses the returned string, so "" is indistinguishable downstream from "the
    model answered nonsense" — which is precisely how the reasoning bug hid. The adapter is the
    only layer that knows the budget and the reasoning flag, so it is the layer that must say so.
    """
    fake_log = _FakeLog()
    monkeypatch.setattr(llm, "log", fake_log)
    monkeypatch.setattr(_FakeChatOllama, "content", "   ")

    client = llm.make_client("ollama", "qwen3.5:9b")
    out = client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=256)

    assert out == ""  # still returns a string — the caller's contract is unchanged
    events = [e for e, _ in fake_log.warnings]
    assert "ollama_empty_completion" in events
    payload = dict(fake_log.warnings[0][1])
    assert payload["model"] == "qwen3.5:9b"
    assert payload["max_tokens"] == 256


def test_ollama_non_empty_completion_does_not_warn(
    patched_sdks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_log = _FakeLog()
    monkeypatch.setattr(llm, "log", fake_log)
    client = llm.make_client("ollama", "llama3")
    client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=256)
    assert fake_log.warnings == []


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


# ============================================================
# ADR-011 / SPRINT-012 (U1c) — provider_available + reviewer-follows-the-switch
# ============================================================


def test_provider_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    assert llm.provider_available("anthropic") is False
    assert llm.provider_available("ollama") is True
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    assert llm.provider_available("anthropic") is True


def test_reviewer_available_effective_provider_overrides_config(monkeypatch: pytest.MonkeyPatch):
    # The effective-provider param checks THAT provider's key, not REVIEWER_PROVIDER's.
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    assert llm.reviewer_available() is False  # unpinned default check: anthropic, no key
    assert llm.reviewer_available("ollama") is True  # a followed switch to ollama needs no key


def test_resolve_reviewer_with_no_args_matches_todays_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "REVIEWER_PROVIDER_PINNED", False)
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "REVIEWER_MODEL", "claude-haiku-4-5-20251001")
    assert llm.resolve_reviewer() == ("anthropic", "claude-haiku-4-5-20251001")


def test_get_reviewer_client_follows_unpinned_switch(
    patched_sdks: None, monkeypatch: pytest.MonkeyPatch
):
    # REVIEWER_PROVIDER was never explicitly set in the environment → follow the live switch.
    monkeypatch.setattr(config, "REVIEWER_PROVIDER_PINNED", False)
    client = llm.get_reviewer_client("ollama", "llama3.1:8b")
    assert isinstance(client, llm.OllamaClient)
    assert client.model == "llama3.1:8b"


def test_get_reviewer_client_respects_explicit_pin(
    patched_sdks: None, monkeypatch: pytest.MonkeyPatch
):
    # An explicit .env REVIEWER_PROVIDER pin wins even when a switch is passed in.
    monkeypatch.setattr(config, "REVIEWER_PROVIDER_PINNED", True)
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "REVIEWER_MODEL", "claude-haiku-4-5-20251001")
    client = llm.get_reviewer_client("ollama", "llama3.1:8b")  # the switch is ignored
    assert isinstance(client, llm.AnthropicClient)
    assert client.model == "claude-haiku-4-5-20251001"


def test_get_reviewer_client_no_args_is_byte_identical_to_today(
    patched_sdks: None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "REVIEWER_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "REVIEWER_MODEL", "claude-sonnet-4-6")
    client = llm.get_reviewer_client()
    assert isinstance(client, llm.AnthropicClient)
    assert client.model == "claude-sonnet-4-6"


# ============================================================
# Enrichment-CLI cost guard (assert_provider_intent)
# ============================================================


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture abort-window sleeps instead of actually sleeping."""
    calls: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: calls.append(s))
    monkeypatch.delenv("DOC_ASSUME_YES", raising=False)
    return calls


def test_paid_providers_policy() -> None:
    # Declarative policy: anthropic bills, ollama is local/free.
    assert "anthropic" in config.PAID_PROVIDERS
    assert "ollama" not in config.PAID_PROVIDERS


@pytest.mark.skipif(
    os.getenv("WIKI_LLM_PROVIDER") is not None or os.getenv("WIKI_LLM_MODEL") is not None,
    reason="env override set — default not observable",
)
def test_wiki_defaults_to_local_not_inherited() -> None:
    # The footgun fix: the wiki generator defaults to local Ollama explicitly,
    # NOT to LLM_PROVIDER/LLM_MODEL (which are anthropic/haiku in api mode).
    assert config.WIKI_LLM_PROVIDER == "ollama"
    assert config.WIKI_LLM_MODEL == "llama3"


def test_guard_dry_run_is_noop(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    # apply=False never bills — even anthropic with no key is a silent no-op.
    llm.assert_provider_intent("anthropic", operation="x", apply=False)
    assert capsys.readouterr().err == ""
    assert no_sleep == []


def test_guard_local_provider_is_noop(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    llm.assert_provider_intent("ollama", operation="wiki", apply=True)
    assert capsys.readouterr().err == ""
    assert no_sleep == []


def test_guard_paid_missing_key_raises(monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    with pytest.raises(llm.ProviderCostError, match="ANTHROPIC_API_KEY"):
        llm.assert_provider_intent("anthropic", operation="my op", apply=True)
    assert no_sleep == []  # never reaches the warn/sleep path


def test_guard_paid_with_key_warns_and_aborts(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    llm.assert_provider_intent(
        "anthropic", operation="wiki topic summarisation", model="claude-haiku-4-5", scope="all"
    )
    err = capsys.readouterr().err
    assert "PAID API RUN" in err
    assert "wiki topic summarisation" in err
    assert "claude-haiku-4-5" in err
    assert "Ctrl-C" in err
    assert no_sleep == [3.0]  # default abort window slept


def test_guard_case_insensitive_provider(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    llm.assert_provider_intent("Anthropic", operation="op")
    assert "PAID API RUN" in capsys.readouterr().err
    assert no_sleep == [3.0]


def test_guard_abort_seconds_zero_warns_without_sleeping(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    llm.assert_provider_intent("anthropic", operation="op", abort_seconds=0)
    err = capsys.readouterr().err
    assert "PAID API RUN" in err
    assert "Ctrl-C" not in err  # no abort-window line when there is no window
    assert no_sleep == []


def test_guard_assume_yes_skips_abort_window(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("DOC_ASSUME_YES", "1")
    llm.assert_provider_intent("anthropic", operation="op")
    err = capsys.readouterr().err
    assert "PAID API RUN" in err  # banner still prints (never silent)
    assert "Ctrl-C" not in err
    assert no_sleep == []  # automation: no interactive pause


# ============================================================
# ADR-034 — the in-app key store reaches every Anthropic call site
# ============================================================


def test_stored_key_is_used_when_the_env_has_none(
    patched_sdks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of ADR-034: a packaged install has no .env, so the key saved in the app must
    # reach the SDK. Regression guard for the separate-binding trap — a call site that read an
    # import-time constant would silently keep sending an empty key.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    from doc_assistant import credentials

    credentials.set_stored_key("anthropic", "sk-ant-from-app")
    client = llm.AnthropicClient("claude-haiku-4-5-20251001")
    assert client._client.api_key == "sk-ant-from-app"  # pragma: allowlist secret
    assert llm.provider_available("anthropic") is True


def test_cost_guard_accepts_a_key_saved_in_the_app(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    # ...and still prints the banner. An in-app key must not become a quiet way to spend.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    from doc_assistant import credentials

    credentials.set_stored_key("anthropic", "sk-ant-from-app")
    llm.assert_provider_intent("anthropic", operation="op", abort_seconds=0)
    assert "PAID API RUN" in capsys.readouterr().err


# ============================================================
# ADR-034 — provider probes (setup path only; never on the answer path)
# ============================================================


class _FakeResponse:
    def __init__(self, payload: Any, *, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


def test_ollama_probe_lists_installed_models(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def _get(url: str, **kwargs: Any) -> _FakeResponse:
        calls["url"] = url
        calls["timeout"] = kwargs.get("timeout")
        return _FakeResponse({"models": [{"name": "qwen3.5:9b"}, {"name": "llama3.1:8b"}]})

    monkeypatch.setattr(httpx, "get", _get)
    reachable, models, detail = llm.ollama_probe("http://localhost:11434/")
    assert (reachable, detail) == (True, None)
    assert models == ("llama3.1:8b", "qwen3.5:9b")  # sorted, so the UI order is stable
    assert calls["url"] == "http://localhost:11434/api/tags"  # trailing slash not doubled
    assert calls["timeout"] == llm.PROBE_TIMEOUT_SECONDS


def test_ollama_probe_never_raises_when_the_server_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An absent local server is the NORMAL first-run state, not an error condition.
    def _boom(url: str, **kwargs: Any) -> _FakeResponse:
        raise OSError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    reachable, models, detail = llm.ollama_probe("http://localhost:11434")
    assert (reachable, models) == (False, ())
    assert "11434" in (detail or "")  # the message names the host the user must fix


def test_ollama_probe_distinguishes_running_but_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda url, **k: _FakeResponse({"models": []}))
    reachable, models, detail = llm.ollama_probe()
    assert (reachable, models) == (True, ())
    assert "no models" in (detail or "")


def test_ollama_probe_tolerates_a_shape_it_does_not_know(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ollama has used both `name` and `model` keys; an unrecognized payload must degrade to
    # "reachable, models unknown", never crash the setup panel.
    monkeypatch.setattr(
        httpx, "get", lambda url, **k: _FakeResponse({"models": [{"model": "m:1"}]})
    )
    assert llm.ollama_probe()[1] == ("m:1",)
    monkeypatch.setattr(httpx, "get", lambda url, **k: _FakeResponse("not a dict"))
    assert llm.ollama_probe()[0] is True


def test_verify_anthropic_key_ok_uses_a_free_metadata_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class _Models:
        def list(self, **kwargs: Any) -> object:
            seen["list"] = kwargs
            return object()

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            seen["init"] = kwargs
            self.models = _Models()
            self.messages = None  # a completion call would be a paid one — there must be none

    monkeypatch.setattr("anthropic.Anthropic", _Client)
    status, detail = llm.verify_anthropic_key(" sk-ant-padded ")
    assert status == "ok"
    assert detail
    assert seen["init"]["api_key"] == "sk-ant-padded"  # pragma: allowlist secret
    assert seen["init"]["max_retries"] == 0  # a first-run probe must not hang on retries
    assert "list" in seen


def test_verify_anthropic_key_rejects_a_bad_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self

        def list(self, **kwargs: Any) -> object:
            err = RuntimeError("unauthorized")
            err.status_code = 401  # type: ignore[attr-defined]
            raise err

    monkeypatch.setattr("anthropic.Anthropic", _Client)
    status, detail = llm.verify_anthropic_key("sk-ant-wrong")
    assert status == "invalid"
    assert "rejected" in detail.lower()


def test_verify_anthropic_key_reports_no_verdict_when_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No network is NOT evidence the key is bad — the caller stores it and says so.
    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self

        def list(self, **kwargs: Any) -> object:
            raise OSError("getaddrinfo failed")

    monkeypatch.setattr("anthropic.Anthropic", _Client)
    status, _ = llm.verify_anthropic_key("sk-ant-maybe-fine")
    assert status == "unreachable"


def test_verify_anthropic_key_rejects_an_empty_key() -> None:
    assert llm.verify_anthropic_key("   ")[0] == "invalid"
