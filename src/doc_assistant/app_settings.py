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
