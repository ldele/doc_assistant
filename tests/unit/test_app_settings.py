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


def test_set_llm_selection_rejects_empty_model(settings_file: Path) -> None:
    # A blank/whitespace model would build a nameless chat model and then be silently dropped by
    # get_llm_selection's truthiness gate on the next boot — reject it instead of corrupting state.
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="model must not be empty"):
            app_settings.set_llm_selection("ollama", blank)
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


# ============================================================
# ADR-027 D2 (E3) — the persisted answer-layer epistemics toggle
# ============================================================


def test_markers_enabled_absent_by_default(settings_file: Path) -> None:
    assert app_settings.get_markers_enabled() is None


def test_markers_enabled_round_trips_both_values(settings_file: Path) -> None:
    app_settings.set_markers_enabled(False)
    assert app_settings.get_markers_enabled() is False
    assert app_settings.load_user_settings()["epistemics_markers_enabled"] is False
    app_settings.set_markers_enabled(True)
    assert app_settings.get_markers_enabled() is True


def test_effective_markers_falls_back_to_config_default(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "EPISTEMICS_MARKERS_ENABLED", True)
    assert app_settings.effective_markers_enabled() is True
    monkeypatch.setattr(config, "EPISTEMICS_MARKERS_ENABLED", False)
    assert app_settings.effective_markers_enabled() is False


def test_effective_markers_prefers_the_persisted_choice(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The persisted choice wins over the config default IN BOTH DIRECTIONS — a persisted
    # False must not be shadowed by a True default, and vice versa (ADR-027 D2's layering).
    monkeypatch.setattr(config, "EPISTEMICS_MARKERS_ENABLED", True)
    app_settings.set_markers_enabled(False)
    assert app_settings.effective_markers_enabled() is False
    monkeypatch.setattr(config, "EPISTEMICS_MARKERS_ENABLED", False)
    app_settings.set_markers_enabled(True)
    assert app_settings.effective_markers_enabled() is True


def test_markers_enabled_ignores_a_non_bool_stored_value(settings_file: Path) -> None:
    # A hand-edited settings.json ("true" as a string, 1, null) must degrade to "never set",
    # not crash or truthiness-coerce — the same fail-safe posture as load_user_settings itself.
    app_settings.save_user_settings({"epistemics_markers_enabled": "true"})
    assert app_settings.get_markers_enabled() is None
