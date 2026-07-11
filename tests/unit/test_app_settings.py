"""Tests for the user-settable runtime settings (`doc_assistant.app_settings`).

ADR-011 (U1c, desktop provider switch): `get_llm_selection`/`set_llm_selection`/`effective_llm`
mirror the already-tested `source_dir` persistence pattern (`settings.json` in a temp dir — never
the real data home). No network, no vendor SDK (`provider_available` only checks a config
attribute).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant import app_settings, config


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    return tmp_path


def test_llm_selection_absent_by_default(settings_file: Path) -> None:
    assert app_settings.get_llm_selection() == (None, None)


def test_llm_selection_round_trips(settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    app_settings.set_llm_selection("anthropic", "claude-haiku-4-5-20251001")
    assert app_settings.get_llm_selection() == ("anthropic", "claude-haiku-4-5-20251001")


def test_llm_selection_persists_across_a_fresh_read(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)  # ollama needs no key
    app_settings.set_llm_selection("ollama", "llama3.1:8b")
    # A second, independent read (not just the same in-memory call) sees the persisted value.
    assert app_settings.load_user_settings()["llm_provider"] == "ollama"
    assert app_settings.load_user_settings()["llm_model"] == "llama3.1:8b"


def test_set_llm_selection_rejects_unknown_provider(settings_file: Path) -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        app_settings.set_llm_selection("openai", "gpt-4")
    assert app_settings.get_llm_selection() == (None, None)  # never persisted


def test_set_llm_selection_rejects_keyless_provider(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    with pytest.raises(ValueError, match="no credential"):
        app_settings.set_llm_selection("anthropic", "claude-haiku-4-5-20251001")
    assert app_settings.get_llm_selection() == (None, None)  # never persisted


def test_effective_llm_falls_back_to_config_default(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "LLM_MODEL", "claude-haiku-4-5-20251001")
    assert app_settings.effective_llm() == ("anthropic", "claude-haiku-4-5-20251001")


def test_effective_llm_prefers_the_persisted_selection(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)  # ollama needs no key
    app_settings.set_llm_selection("ollama", "llama3.1:8b")
    assert app_settings.effective_llm() == ("ollama", "llama3.1:8b")
