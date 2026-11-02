"""Tests for the presets and the two facts the screens state (ADR-060 amended; its tests 3, 4, 5).

No network: everything here is local state, and the Ollama probe's upstream hosts are put in place
the way the probe records them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog

from doc_assistant import app_settings, config, credentials, llm, providers
from doc_assistant.providers import PRESETS, describe, host_of, reach_of, route_of


@pytest.fixture(autouse=True)
def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:11434")


# ============================================================
# Test 3: the address table
# ============================================================

_SAYS = "Your questions and the passages found for them"


@pytest.mark.parametrize(
    ("address", "reach", "sentence"),
    [
        ("http://localhost:1234/v1", "this_computer", f"{_SAYS} stay on this computer."),
        # Resolved by the OS, and Windows does not map it to loopback: not promised as local.
        ("http://lab.localhost:1234/v1", "elsewhere", f"{_SAYS} go to lab.localhost."),
        ("http://127.0.0.1:1234/v1", "this_computer", f"{_SAYS} stay on this computer."),
        ("http://[::1]:1234/v1", "this_computer", f"{_SAYS} stay on this computer."),
        ("http://0.0.0.0:1234/v1", "this_computer", f"{_SAYS} stay on this computer."),
        (
            "http://10.0.0.5:1234/v1",
            "local_network",
            f"{_SAYS} leave this computer: they go to 10.0.0.5, on your local network.",
        ),
        (
            "http://172.20.1.1:1234/v1",
            "local_network",
            f"{_SAYS} leave this computer: they go to 172.20.1.1, on your local network.",
        ),
        (
            "http://192.168.1.40:1234/v1",
            "local_network",
            f"{_SAYS} leave this computer: they go to 192.168.1.40, on your local network.",
        ),
        (
            "http://169.254.1.1:1234/v1",
            "local_network",
            f"{_SAYS} leave this computer: they go to 169.254.1.1, on your local network.",
        ),
        (
            "http://[fe80::1]:1234/v1",
            "local_network",
            f"{_SAYS} leave this computer: they go to fe80::1, on your local network.",
        ),
        ("http://8.8.8.8/v1", "elsewhere", f"{_SAYS} go to 8.8.8.8."),
        (
            "http://localhost.example.com/v1",
            "elsewhere",
            f"{_SAYS} go to localhost.example.com.",
        ),
        # A name is never looked up: the sentence is a promise, and a name can point anywhere.
        ("http://gpu-box:1234/v1", "elsewhere", f"{_SAYS} go to gpu-box."),
        ("https://api.groq.com/openai/v1", "elsewhere", f"{_SAYS} go to api.groq.com."),
        ("", "elsewhere", f"{_SAYS} leave this computer: the address could not be read."),
        (
            "http://[::1",
            "elsewhere",
            f"{_SAYS} leave this computer: the address could not be read.",
        ),
        (
            "http://not a host/v1",
            "elsewhere",
            f"{_SAYS} leave this computer: the address could not be read.",
        ),
        (
            "http://host:port/v1",
            "elsewhere",
            f"{_SAYS} leave this computer: the address could not be read.",
        ),
    ],
)
def test_the_address_table(address: str, reach: str, sentence: str) -> None:
    route = route_of(address)
    assert (route.reach, route.sentence) == (reach, sentence)


@pytest.mark.parametrize(
    ("host", "reach"),
    [
        ("localhost", "this_computer"),
        ("127.0.0.1", "this_computer"),
        ("::1", "this_computer"),
        ("0.0.0.0", "this_computer"),
        ("10.0.0.5", "local_network"),
        ("fe80::1", "local_network"),
        ("gpu-box", "elsewhere"),
        ("", "elsewhere"),
    ],
)
def test_bare_hosts(host: str, reach: str) -> None:
    assert reach_of(host) == reach


def test_a_bare_host_and_port_reads_like_ollama_host() -> None:
    assert host_of("localhost:11434") == "localhost"


# ============================================================
# The billing table (the amendment's test 3 addition)
# ============================================================


def _server(name: str, address: str) -> str:
    return providers.save_server(name, address).id


def test_a_cloud_preset_is_billed_by_its_vendor() -> None:
    facts = describe("groq")
    assert (facts.billing.line, facts.billing.guard_trips) == ("Billed by Groq", True)
    assert facts.route.sentence.endswith("go to api.groq.com.")


def test_lm_studio_bills_nobody() -> None:
    facts = describe("lmstudio")
    assert (facts.billing.kind, facts.billing.guard_trips) == ("nobody", False)
    assert facts.route.reach == "this_computer"


def test_an_ollama_model_with_a_remote_host_is_billed_by_ollama() -> None:
    # The shape documented in Ollama's openapi.yaml; not yet seen on this machine's Ollama.
    llm._OLLAMA_REMOTE["http://localhost:11434"] = {"gemma4:cloud": "https://ollama.com:443"}
    facts = describe("ollama", "gemma4:cloud")
    assert (facts.billing.line, facts.billing.guard_trips) == ("Billed by Ollama", True)
    assert facts.route.sentence.endswith("go to ollama.com.")
    # The same Ollama, a model on this computer: nobody bills, and the text stays here.
    local = describe("ollama", "llama3.1:8b")
    assert (local.billing.kind, local.route.reach) == ("nobody", "this_computer")


@pytest.mark.parametrize(
    "address", ["http://localhost:1234/v1", "http://192.168.1.40:1234/v1", "https://lab.org/v1"]
)
def test_a_server_of_your_own_bills_until_you_say_it_does_not(address: str) -> None:
    sid = _server("Lab server", address)
    unticked = describe(sid)
    assert (unticked.billing.line, unticked.billing.guard_trips) == ("Billing not known", True)
    app_settings.set_not_billed(sid, True)
    ticked = describe(sid)
    assert (ticked.billing.line, ticked.billing.guard_trips) == ("Not billed (your word)", False)
    assert ticked.route == unticked.route  # the tick changes who bills, never where text goes


def test_an_ollama_on_another_machine_is_not_known_until_ticked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "OLLAMA_HOST", "http://192.168.1.40:11434")
    assert providers.can_tick("ollama")
    assert describe("ollama").billing.kind == "unknown"
    app_settings.set_not_billed("ollama", True)
    assert describe("ollama").billing.kind == "your_word"


def test_claude_is_billed_by_anthropic_and_cannot_be_ticked() -> None:
    assert describe("anthropic").billing.line == "Billed by Anthropic"
    assert not providers.can_tick("anthropic")
    assert not providers.can_tick("groq")


def test_an_unknown_provider_counts_as_billed() -> None:
    assert describe("nonesuch").billing.guard_trips


# ============================================================
# Test 4: every preset row
# ============================================================


@pytest.mark.parametrize("row", PRESETS, ids=lambda p: p.id)
def test_every_preset_row_is_well_formed(row: providers.Preset) -> None:
    assert host_of(row.address), "the address parses"
    local = route_of(row.address).reach == "this_computer"
    assert row.address.startswith("http://" if local else "https://")
    if row.billed_by is not None:
        assert row.key_page and row.key_page.startswith("https://"), "a key page when it bills"
    assert row.checked >= "2026-11-01"
    assert not row.id.startswith(providers.SERVER_PREFIX)


def test_the_six_presets_of_the_wireframe() -> None:
    assert [p.name for p in PRESETS] == [
        "LM Studio",
        "OpenAI",
        "Google Gemini",
        "Mistral",
        "Groq",
        "OpenRouter",
    ]


# ============================================================
# Servers of your own and test 5: keys
# ============================================================


def test_a_server_is_saved_and_named_apart_from_the_presets() -> None:
    first = _server("OpenAI", "http://localhost:8080/v1")
    second = _server("OpenAI", "http://localhost:8081/v1")
    assert first == "server-openai" and second == "server-openai-2"
    assert providers.kind_of(first) == "server"
    assert first in providers.answer_provider_ids()


def test_an_address_without_a_scheme_is_refused_with_how_to_write_it() -> None:
    with pytest.raises(ValueError, match="http://"):
        providers.save_server("Lab", "localhost:1234/v1")


def test_changing_a_servers_address_drops_its_key() -> None:
    sid = _server("Lab", "http://localhost:1234/v1")
    credentials.set_bound_key(sid, "k-123", "http://localhost:1234/v1")
    providers.save_server("Lab", "http://localhost:1234/v1/", sid)  # the same address
    assert providers.endpoint(sid).key == "k-123"
    providers.save_server("Lab", "http://192.168.1.40:1234/v1", sid)
    assert providers.endpoint(sid).key is None
    assert credentials.bound_address(sid) is None


def test_a_key_goes_only_to_the_address_it_was_saved_with() -> None:
    credentials.set_bound_key("groq", "gsk-1", "https://evil.example/v1")
    assert providers.endpoint("groq").key is None  # the preset's address is not that one


def test_an_exported_vendor_key_never_reaches_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-exported")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-exported")
    assert providers.endpoint("groq").key is None
    assert providers.endpoint("openai").key is None
    assert not llm.provider_available("groq")


def test_the_old_anthropic_field_still_reads_beside_bound_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    credentials.set_stored_key("anthropic", "sk-ant-old")
    credentials.set_bound_key("groq", "gsk-new", "https://api.groq.com/openai/v1")
    assert credentials.resolve_key("anthropic") == "sk-ant-old"
    assert providers.endpoint("groq").key == "gsk-new"


def test_removing_a_server_forgets_its_key_and_tick() -> None:
    sid = _server("Lab", "http://localhost:1234/v1")
    credentials.set_bound_key(sid, "k", "http://localhost:1234/v1")
    app_settings.set_not_billed(sid, True)
    assert providers.remove_server(sid)
    assert credentials.bound_address(sid) is None
    assert sid not in app_settings.get_not_billed()
    assert not providers.remove_server(sid)


def test_no_log_line_carries_a_key() -> None:
    with structlog.testing.capture_logs() as logs:
        credentials.set_bound_key("groq", "gsk-very-secret", "https://api.groq.com/openai/v1")
        providers.endpoint("groq")
        credentials.clear_bound_key("groq")
    assert "gsk-very-secret" not in repr(logs)
    assert "gsk-very-secret" not in repr(providers.endpoint("openai"))


def test_an_endpoint_never_prints_its_key() -> None:
    point = providers.Endpoint("groq", "Groq", "https://x", "gsk-secret")
    assert "gsk-secret" not in repr(point)


# ============================================================
# Choice 7: the prompt-cut check
# ============================================================


def test_a_server_that_read_under_half_the_prompt_is_warned_about() -> None:
    # KI-66's case: about 5,000 tokens sent, 2,050 read.
    note = providers.prompt_cut_note(2050, 20_000)
    assert note is not None and "2,050" in note and "5,000" in note


@pytest.mark.parametrize(("reported", "chars"), [(0, 20_000), (4000, 20_000), (2500, 20_000)])
def test_no_warning_without_usage_or_above_half(reported: int, chars: int) -> None:
    assert providers.prompt_cut_note(reported, chars) is None


def test_the_window_note_names_the_ollama_window() -> None:
    assert f"{config.OLLAMA_CONTEXT_TOKENS:,}" in providers.window_note()


def test_the_review_is_listed_as_sent_only_when_the_reviewer_follows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "REVIEWER_PROVIDER_PINNED", False)
    assert any("review" in s for s in providers.sends())
    monkeypatch.setattr(config, "REVIEWER_PROVIDER_PINNED", True)
    assert not any("review" in s for s in providers.sends())
