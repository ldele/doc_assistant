"""Tests for the in-app API-key store (`doc_assistant.credentials`, ADR-034).

Mirrors the `app_settings` pattern: a JSON file in a temp dir, never the real data home (the
autouse `_isolate_credentials` fixture in `tests/conftest.py` already redirects it — these tests
pin it explicitly too, so a reader of this file can see where the bytes go). No network: this
module only reads/writes local state, and `env_key` reads a `config` attribute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_assistant import config, credentials


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)  # no .env key unless a test sets one
    return path


def test_absent_store_reads_as_no_key(store: Path) -> None:
    assert credentials.get_stored_key("anthropic") is None
    assert credentials.resolve_key("anthropic") is None
    assert credentials.key_source("anthropic") is None
    assert credentials.key_hint("anthropic") is None


def test_stored_key_round_trips(store: Path) -> None:
    credentials.set_stored_key("anthropic", "sk-ant-secret-AB12")
    assert credentials.get_stored_key("anthropic") == "sk-ant-secret-AB12"
    assert json.loads(store.read_text(encoding="utf-8"))["anthropic_api_key"] == (
        "sk-ant-secret-AB12"
    )


def test_stored_key_is_trimmed(store: Path) -> None:
    # Pasting from a console/console-copy commonly drags whitespace along; a key with a trailing
    # newline authenticates nowhere and would look like "the key is wrong".
    credentials.set_stored_key("anthropic", "  sk-ant-padded  \n")
    assert credentials.get_stored_key("anthropic") == "sk-ant-padded"


def test_blank_key_is_refused(store: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        credentials.set_stored_key("anthropic", "   ")
    assert not store.exists()  # nothing written


def test_env_wins_over_the_stored_key(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The locked precedence (ADR-034 D2). The CLI runners read the import-time config constant and
    # cannot see the store, so if the store won, the app and the CLI would use different keys.
    credentials.set_stored_key("anthropic", "sk-ant-from-app")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert credentials.resolve_key("anthropic") == "sk-ant-from-env"
    assert credentials.key_source("anthropic") == "env"


def test_stored_key_is_used_when_env_is_empty(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `cp .env.example .env` leaves ANTHROPIC_API_KEY= — an empty string, not an absent one. That
    # must not shadow the key the user saved in the app.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    credentials.set_stored_key("anthropic", "sk-ant-from-app")
    assert credentials.resolve_key("anthropic") == "sk-ant-from-app"
    assert credentials.key_source("anthropic") == "app"


def test_clear_removes_only_the_app_key(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    credentials.set_stored_key("anthropic", "sk-ant-from-app")
    assert credentials.clear_stored_key("anthropic") is True
    assert credentials.clear_stored_key("anthropic") is False  # idempotent, reports honestly
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert credentials.resolve_key("anthropic") == "sk-ant-from-env"  # .env untouched


def test_hint_shows_only_the_last_four_characters(store: Path) -> None:
    credentials.set_stored_key("anthropic", "sk-ant-api03-verylongsecret-WXYZ")
    hint = credentials.key_hint("anthropic")
    assert hint == "...WXYZ"
    assert "verylongsecret" not in (hint or "")


def test_unreadable_store_reads_as_no_key(store: Path) -> None:
    store.write_text("{not json", encoding="utf-8")
    assert credentials.get_stored_key("anthropic") is None  # fail-safe, no raise


def test_a_keyless_provider_is_a_programming_error(store: Path) -> None:
    # Ollama needs no credential; accepting one would write a field nothing ever reads.
    with pytest.raises(ValueError, match="takes no API key"):
        credentials.set_stored_key("ollama", "whatever")


def test_keyed_providers_is_the_single_list(store: Path) -> None:
    assert credentials.keyed_providers() == ("anthropic",)
