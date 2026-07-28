"""Tests for first-run readiness (`doc_assistant.readiness`, ADR-034).

No network: `llm.ollama_probe` is monkeypatched in every test that probes, and the Anthropic side
is pure local state. The probes themselves are tested in `test_llm.py` against a fake transport.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant import app_settings, config, credentials, llm, readiness


@pytest.fixture
def clean_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A first-run box: no key anywhere, no persisted provider selection."""
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", tmp_path / "credentials.json")
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)


def _probe(reachable: bool, models: tuple[str, ...], detail: str | None = None):
    return lambda *a, **k: (reachable, models, detail)


def test_anthropic_unready_without_a_key(clean_install: None) -> None:
    p = readiness.provider_readiness("anthropic")
    assert (p.configured, p.ready, p.key_source) == (False, False, None)
    assert p.action  # a first-run user is told what to do, not just that it is broken


def test_anthropic_ready_names_the_key_source(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials.set_stored_key("anthropic", "sk-ant-1234")
    p = readiness.provider_readiness("anthropic")
    assert (p.ready, p.key_source, p.key_hint) == (True, "app", "...1234")
    assert "this app" in p.detail
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-envkey")
    assert ".env" in readiness.provider_readiness("anthropic").detail


def test_ollama_unreachable_is_reported_not_hidden(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "ollama_probe", _probe(False, (), "No Ollama server answering"))
    p = readiness.provider_readiness("ollama")
    assert (p.configured, p.reachable, p.ready) == (True, False, False)
    assert "ollama serve" in (p.action or "")


def test_ollama_reachable_but_empty_is_its_own_state(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fix differs from "unreachable" (pull a model, not start a server), so the two states
    # must not collapse into one.
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, (), "no models installed"))
    p = readiness.provider_readiness("ollama")
    assert (p.reachable, p.ready) == (True, False)
    assert "ollama pull" in (p.action or "")


def test_ollama_ready_lists_installed_models(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b", "qwen3.5:9b")))
    p = readiness.provider_readiness("ollama")
    assert (p.ready, p.models) == (True, ("llama3.1:8b", "qwen3.5:9b"))
    assert p.action is None


def test_probe_false_touches_no_network(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("probe=False must not call out")

    monkeypatch.setattr(llm, "ollama_probe", _boom)
    p = readiness.provider_readiness("ollama", probe=False)
    assert p.reachable is None  # unknown, and says so


def test_unknown_provider_raises(clean_install: None) -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        readiness.provider_readiness("openai")


def test_zero_documents_reports_honestly(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The 0-document half of the robustness contract: an empty corpus is a legitimate state that
    # reports what to do, never an error.
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b",)))
    state = readiness.setup_state(chunk_count=0, document_count=0)
    documents = next(s for s in state.steps if s.id == "documents")
    assert (documents.done, state.ready) == (False, False)
    assert documents.action


def test_ready_when_provider_and_corpus_are_both_there(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b",)))
    app_settings.set_llm_selection("ollama", "llama3.1:8b")
    state = readiness.setup_state(chunk_count=1234, document_count=7)
    assert state.ready is True
    assert state.active_ready is True
    assert all(s.done for s in state.steps)
    assert "7 documents" in next(s for s in state.steps if s.id == "documents").detail


def test_active_provider_drives_the_step_and_names_the_alternative(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Anthropic is the config default and has no key here; Ollama is ready. The step must track the
    # *active* provider (that is what the next turn will use) while pointing at the faster fix.
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b",)))
    state = readiness.setup_state(chunk_count=10, document_count=1)
    provider_step = next(s for s in state.steps if s.id == "provider")
    assert provider_step.done is False
    assert "Ollama is ready" in provider_step.detail
    assert state.ready is False


def test_an_unknown_active_provider_degrades_instead_of_raising(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hand-edited settings.json (or a build that dropped a provider) must not 500 the setup view
    # the user needs in order to fix it.
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("openai", "gpt-4"))
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b",)))
    state = readiness.setup_state(chunk_count=10, document_count=1)
    assert state.active_ready is False
    assert state.ready is False
