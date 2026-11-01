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


def test_a_claude_model_on_ollama_is_not_ready_and_says_why(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The 0.7.0 walkthrough's sandbox: the provider switched to Ollama and kept the Claude model,
    # so every question failed while the step read done.
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("ollama", "claude-haiku-4-5"))
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b",)))
    state = readiness.setup_state(chunk_count=10, document_count=1)
    step = next(s for s in state.steps if s.id == "provider")
    assert step.done is False and state.active_ready is False
    assert "Claude model" in step.detail
    assert step.action is not None and "Use for answers" in step.action


def test_a_model_the_answering_ollama_lacks_is_named_with_what_it_has(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("ollama", "mistral:7b"))
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b", "qwen2.5:7b")))
    step = next(
        s
        for s in readiness.setup_state(chunk_count=10, document_count=1).steps
        if s.id == "provider"
    )
    assert step.done is False
    assert "“mistral:7b”" in step.detail and "llama3.1:8b, qwen2.5:7b" in step.detail


def test_an_installed_model_or_an_unprobed_server_is_left_alone(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("ollama", "llama3"))
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3:latest",)))
    assert readiness.setup_state(chunk_count=10, document_count=1).active_ready is True
    # Without a probe nothing is known about what is installed, so nothing is refused.
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("ollama", "mistral:7b"))
    assert readiness.setup_state(chunk_count=10, document_count=1, probe=False).active_ready


def _rows(state: readiness.SetupState) -> tuple[readiness.SetupStep, readiness.SetupStep]:
    model, documents = state.steps
    assert (model.id, documents.id) == ("provider", "documents")
    return model, documents


def test_a_fresh_install_reads_as_two_rows_not_set(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "ollama_probe", _probe(False, (), "No Ollama server answering"))
    model, documents = _rows(readiness.setup_state(chunk_count=0, document_count=0))
    assert (model.title, model.detail, model.status) == (
        "Model",
        "No model chosen yet.",
        "Not set",
    )
    assert (documents.title, documents.detail, documents.status) == (
        "Documents",
        "Nothing indexed yet.",
        "Not set",
    )
    # Nothing was chosen, so nothing has gone offline: only a chosen engine turns the row red.
    assert (model.offline, documents.offline) == (False, False)


def test_ready_names_the_model_and_the_documents(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b",)))
    app_settings.set_llm_selection("ollama", "llama3.1:8b")
    model, documents = _rows(readiness.setup_state(chunk_count=99_999, document_count=1_204))
    assert (model.detail, model.status, model.done) == ("llama3.1:8b", "Engine online", True)
    assert (documents.detail, documents.status) == ("1,204 documents indexed", "Indexed")


def test_one_document_is_singular(clean_install: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b",)))
    _, documents = _rows(readiness.setup_state(chunk_count=12, document_count=1))
    assert documents.detail == "1 document indexed"


def test_chunks_with_no_document_count_never_read_as_zero_indexed(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failed count arrives as 0 (`services.setup_state_dict`), and the chunks are real.
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, ("llama3.1:8b",)))
    _, documents = _rows(readiness.setup_state(chunk_count=1_500, document_count=0))
    assert (documents.done, documents.detail) == (True, "1,500 chunks indexed")


def test_an_offline_engine_turns_the_row_red_and_keeps_its_fix(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "ollama_probe", _probe(False, (), "No Ollama server answering at x"))
    app_settings.set_llm_selection("ollama", "llama3.1:8b")
    state = readiness.setup_state(chunk_count=10, document_count=1)
    model, _ = _rows(state)
    assert (model.done, model.offline, model.status) == (False, True, "Engine offline")
    assert "No Ollama server answering" in model.detail
    assert "ollama serve" in (model.action or "")
    assert state.ready is False


def test_an_ollama_with_no_models_is_online_but_not_set(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "ollama_probe", _probe(True, (), "no models installed"))
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("ollama", "llama3.1:8b"))
    model, _ = _rows(readiness.setup_state(chunk_count=10, document_count=1))
    assert (model.status, model.offline) == ("Not set", False)


def test_an_alternative_reads_as_its_own_sentence(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The probe's sentence has no full stop, and the alternative used to run straight on from it.
    credentials.set_stored_key("anthropic", "sk-ant-1234")
    monkeypatch.setattr(llm, "ollama_probe", _probe(False, (), "No Ollama server answering at x"))
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("ollama", "llama3.1:8b"))
    model, _ = _rows(readiness.setup_state(chunk_count=10, document_count=1))
    assert model.detail.endswith("answering at x. Claude is ready, if you'd rather use that.")


def test_claude_reads_online_on_its_key_with_no_request(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADR-034: Anthropic is never probed, so the words stand on the key alone.
    credentials.set_stored_key("anthropic", "sk-ant-1234")
    monkeypatch.setattr(llm, "ollama_probe", _probe(False, ()))
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("anthropic", "claude-haiku-4-5"))
    model, _ = _rows(readiness.setup_state(chunk_count=10, document_count=1))
    assert (model.detail, model.status) == ("claude-haiku-4-5", "Engine online")


def test_an_ollama_nobody_has_asked_is_not_called_online(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("ollama", "llama3.1:8b"))
    model, _ = _rows(readiness.setup_state(chunk_count=10, document_count=1, probe=False))
    assert (model.done, model.status) == (True, "Not checked")


@pytest.mark.parametrize(
    ("address", "group", "where"),
    [
        ("http://localhost:11434", "local", "stay on this computer"),
        ("http://127.0.0.1:11434", "local", "stay on this computer"),
        ("http://[::1]:11434", "local", "stay on this computer"),
        ("0.0.0.0:11434", "local", "stay on this computer"),
        ("http://192.168.1.20:11434", "server", "go to 192.168.1.20"),
        ("https://lab.example.org", "server", "go to lab.example.org"),
        ("lab-box:11434", "server", "go to lab-box"),
        ("http://[::1", "server", "go to http://[::1"),
    ],
)
def test_where_the_text_goes_comes_from_the_address(
    clean_install: None, monkeypatch: pytest.MonkeyPatch, address: str, group: str, where: str
) -> None:
    monkeypatch.setattr(config, "OLLAMA_HOST", address)
    ollama = readiness.provider_readiness("ollama", probe=False)
    assert ollama.group == group
    assert where in ollama.destination


def test_claude_names_where_its_text_goes(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    claude = readiness.provider_readiness("anthropic")
    assert (claude.name, claude.group) == ("Claude", "cloud")
    assert claude.destination.endswith("go to api.anthropic.com.")
    # The SDK honours this variable, so a sentence that ignored it would be false.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.org/anthropic")
    assert "gateway.example.org" in readiness.provider_readiness("anthropic").destination
