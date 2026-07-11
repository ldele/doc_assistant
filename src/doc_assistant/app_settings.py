"""User-configurable runtime settings (the desktop "point at a folder" flow).

The *locked* RAG knobs live in :mod:`doc_assistant.config` (changed only via an eval
experiment). This module owns the *user*-facing settings a non-dev sets at runtime through the
desktop app — currently just the **source documents folder** to ingest from — persisted as JSON
in the data home so the choice survives a sidecar restart.

Kept out of ``config`` (which is import-time + effectively immutable) because these are mutable,
user-owned, and per-install. The data *home* (where the index/DB live) stays managed by
``config._resolve_data_path`` (per-user when frozen, ASCII-safe Chroma via KI-11); the user only
points at where *their documents* are.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

from doc_assistant import config

log = structlog.get_logger(__name__)

SETTINGS_PATH = config.DATA_PATH / "settings.json"


def load_user_settings() -> dict[str, Any]:
    """Read the persisted user settings; ``{}`` if absent or unreadable (fail-safe)."""
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as e:
        log.warning("user_settings_unreadable", path=str(SETTINGS_PATH), error=str(e))
        return {}
    return data if isinstance(data, dict) else {}


def save_user_settings(settings: dict[str, Any]) -> None:
    """Persist the user settings as JSON in the data home (creating the dir if needed)."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def get_source_dir() -> Path:
    """The folder ingest reads documents from.

    Precedence: ``DOC_SOURCE_DIR`` env override > the persisted ``source_dir`` > the default
    ``config.DOCS_PATH`` (``<data home>/sources``).
    """
    override = os.getenv("DOC_SOURCE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    stored = load_user_settings().get("source_dir")
    if isinstance(stored, str) and stored:
        return Path(stored).expanduser().resolve()
    return config.DOCS_PATH


def set_source_dir(path: str) -> Path:
    """Validate ``path`` is an existing directory, persist it, and return the resolved path.

    Raises :class:`ValueError` if the path doesn't exist or isn't a directory (the API maps that
    to 400) — inform-don't-corrupt: never persist a folder we can't read.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"not a directory: {resolved}")
    settings = load_user_settings()
    settings["source_dir"] = str(resolved)
    save_user_settings(settings)
    log.info("source_dir_set", path=str(resolved))
    return resolved


# ============================================================
# LLM provider/model selection (ADR-011, U1c — desktop provider switch)
# ============================================================
# A non-secret, user-owned, per-install choice — same shape as source_dir. The API key stays in
# .env (v1 handles no secret); this module only remembers *which already-configured provider* the
# user picked, so it survives a sidecar restart. `chat_controller.ChatController` applies it at
# construction (RAGPipeline.set_chat_model) and on a live POST /api/settings switch.


def get_llm_selection() -> tuple[str | None, str | None]:
    """The persisted ``(provider, model)`` selection, or ``(None, None)`` if never set."""
    stored = load_user_settings()
    provider = stored.get("llm_provider")
    model = stored.get("llm_model")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return provider, model
    return None, None


def set_llm_selection(provider: str, model: str) -> None:
    """Validate and persist a provider/model choice.

    Raises :class:`ValueError` (the API maps that to 400) for an unknown provider or one whose
    credential is absent — inform-don't-corrupt: never persist a selection that can't run.
    """
    from doc_assistant.llm import provider_available

    key = provider.strip().lower()
    if key not in ("anthropic", "ollama"):
        raise ValueError(f"unknown provider '{provider}' — valid options: anthropic, ollama")
    if not provider_available(key):
        raise ValueError(f"provider '{key}' has no credential configured (add it to .env)")
    settings = load_user_settings()
    settings["llm_provider"] = key
    settings["llm_model"] = model
    save_user_settings(settings)
    log.info("llm_selection_set", provider=key, model=model)


def effective_llm() -> tuple[str, str]:
    """The live ``(provider, model)``: the persisted selection if present, else the config
    default (``config.LLM_PROVIDER``/``LLM_MODEL``). The single source of "what's actually live" —
    ``RAGPipeline``/``ChatController`` and the settings view both read through this, never the
    import-time config constants directly, so a switch and a fresh boot agree."""
    provider, model = get_llm_selection()
    if provider is not None and model is not None:
        return provider, model
    return config.LLM_PROVIDER, config.LLM_MODEL
