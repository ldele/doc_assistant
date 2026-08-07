"""Guards `config._load_env`'s precedence rule (KI-38).

The rule: a **non-empty** process environment variable beats `.env`; `.env` beats an absent
or empty one. Both halves are load-bearing and they pull in opposite directions, so each has
its own test:

* the *empty* half is why the override existed at all (a host exporting an empty
  ``ANTHROPIC_API_KEY`` must not shadow the real key in ``.env``);
* the *non-empty* half is KI-38 — under the previous ``load_dotenv(override=True)`` every
  env-var override in the repo was silently ineffective, from ``LLM_PROVIDER=ollama`` (which
  billed Anthropic) to the chunking sweep's own grid (which compared one config with itself).

None of that failed loudly, so these tests are the only thing standing between the rule and a
future "simplification" back to ``override=True``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from doc_assistant.config import _load_env

# The `ANTHROPIC_API_KEY` entries below are fixture literals, not credentials — the real key's
# name is what the test is about (it is the variable the override existed to protect), so the
# `allowlist secret` pragmas keep detect-secrets quiet without weakening the scan elsewhere.
ENV_BODY = "\n".join(
    [
        "ANTHROPIC_API_KEY=key-from-dotenv",  # pragma: allowlist secret
        "LLM_PROVIDER=anthropic",
        "CHILD_CHUNK_SIZE=400",
    ]
)


@pytest.fixture
def env_file(tmp_path: Path) -> str:
    path = tmp_path / ".env"
    path.write_text(ENV_BODY, encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from "none of these keys are set"."""
    for key in ("ANTHROPIC_API_KEY", "LLM_PROVIDER", "CHILD_CHUNK_SIZE"):
        monkeypatch.delenv(key, raising=False)


def test_absent_env_var_takes_the_dotenv_value(env_file: str) -> None:
    _load_env(env_file)
    assert os.environ["LLM_PROVIDER"] == "anthropic"
    assert os.environ["CHILD_CHUNK_SIZE"] == "400"


def test_empty_env_var_does_not_shadow_dotenv(
    env_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original reason for the override: an empty host key must not win."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    _load_env(env_file)
    assert os.environ["ANTHROPIC_API_KEY"] == "key-from-dotenv"  # pragma: allowlist secret


def test_whitespace_only_env_var_does_not_shadow_dotenv(
    env_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    _load_env(env_file)
    assert os.environ["ANTHROPIC_API_KEY"] == "key-from-dotenv"  # pragma: allowlist secret


def test_real_env_var_beats_dotenv(env_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """KI-38 itself: `LLM_PROVIDER=ollama <cmd>` must not silently run on Anthropic."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _load_env(env_file)
    assert os.environ["LLM_PROVIDER"] == "ollama"


def test_sweep_style_overrides_survive(env_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """`scripts/sweep_chunking.py` passes its grid as env vars that `.env` also defines."""
    monkeypatch.setenv("CHILD_CHUNK_SIZE", "256")
    _load_env(env_file)
    assert os.environ["CHILD_CHUNK_SIZE"] == "256"


def test_keys_absent_from_dotenv_are_left_alone(
    env_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOC_DATA_DIR", "/somewhere/else")
    _load_env(env_file)
    assert os.environ["DOC_DATA_DIR"] == "/somewhere/else"


def test_missing_dotenv_file_is_not_an_error(tmp_path: Path) -> None:
    """The frozen build ships no `.env` — discovery returning nothing must be a no-op."""
    _load_env(str(tmp_path / "does-not-exist.env"))
    _load_env("")
    assert "LLM_PROVIDER" not in os.environ
