"""Per-install API credentials — the in-app "bring your own key" store.

Why this is a module of its own, and not a few more keys in :mod:`doc_assistant.app_settings`:

* ``app_settings`` holds **non-secret**, user-owned preferences (source folder, provider choice,
  epistemics default) and its JSON is safe to paste into a bug report. An API key is not. Keeping
  the secret in a separate file (``credentials.json``) means ``settings.json`` stays shareable and
  there is exactly **one** path in the codebase that reads key material off disk.
* The store is a *fallback*, not a replacement for ``.env``: the environment still wins (see
  :func:`resolve_key`), so the CLI/enrichment runners — which read the import-time
  ``config.ANTHROPIC_API_KEY`` constant — never disagree with the app about which key is live.

Locked choices (ADR-034):

* **Environment wins.** ``ANTHROPIC_API_KEY`` from the process env / ``.env`` takes precedence over
  a stored key; the stored key is what a packaged install (no ``.env``) and a first-run tester use.
  :func:`key_source` names which one is live so the UI can never claim a key it is not using.
* **Never echoed back.** Nothing here returns key material to a caller that only wants to *display*
  state — that is what :func:`key_hint` is for (last 4 characters).
* **Never logged.** Log lines carry the provider and the hint, never the key.
* The file lives in the data home (outside the repo, so it cannot be committed) and is written with
  owner-only permissions where the OS honours them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import structlog

from doc_assistant import config

log = structlog.get_logger(__name__)

CREDENTIALS_PATH = config.DATA_PATH / "credentials.json"

KeySource = Literal["env", "app"]

# provider -> (field name in credentials.json, config constant holding the env value).
# Claude only: Ollama needs no secret, and every other provider's key is bound to its address in
# `keys` (below), never read from the environment (ADR-060).
_KEYED_PROVIDERS: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
}


def keyed_providers() -> tuple[str, ...]:
    """Providers that need an API key (so callers don't hardcode the list)."""
    return tuple(_KEYED_PROVIDERS)


def _fields(provider: str) -> tuple[str, str]:
    """The (json field, config constant) pair for ``provider``.

    Raises :class:`ValueError` for a provider that takes no key — a caller asking to store a
    credential for Ollama has a bug, and silently accepting it would create a file field nothing
    ever reads.
    """
    try:
        return _KEYED_PROVIDERS[provider.strip().lower()]
    except KeyError as e:
        raise ValueError(
            f"provider '{provider}' takes no API key — keyed providers: "
            f"{', '.join(_KEYED_PROVIDERS)}"
        ) from e


def _load() -> dict[str, Any]:
    """Read the credential file; ``{}`` if absent or unreadable (fail-safe, like app_settings)."""
    try:
        data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as e:
        log.warning("credentials_unreadable", path=str(CREDENTIALS_PATH), error=str(e))
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict[str, Any]) -> None:
    """Write the credential file, owner-only where the OS honours it.

    ``chmod`` is best-effort by design: on Windows it only clears the read-only bit and does not
    tighten the ACL, so this narrows exposure on POSIX without pretending to on Windows (the data
    home is already per-user there). A failure to chmod must never lose the user's key, so it is
    logged, not raised.
    """
    path = Path(CREDENTIALS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError as e:  # pragma: no cover - platform dependent
        log.debug("credentials_chmod_failed", path=str(path), error=str(e))


def get_stored_key(provider: str) -> str | None:
    """The key this install has stored for ``provider``, or ``None``."""
    field, _ = _fields(provider)
    value = _load().get(field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def set_stored_key(provider: str, key: str) -> None:
    """Persist ``key`` for ``provider``.

    Raises :class:`ValueError` on a blank key: storing one would read as "configured" everywhere
    while resolving to nothing (inform-don't-corrupt — the same rule
    ``app_settings.set_llm_selection`` applies to a blank model). Verifying that the key actually
    *works* is the caller's job (``llm.verify_anthropic_key``) — this module owns storage only.
    """
    field, _ = _fields(provider)
    cleaned = key.strip()
    if not cleaned:
        raise ValueError("API key must not be empty")
    data = _load()
    data[field] = cleaned
    _save(data)
    log.info("api_key_stored", provider=provider.strip().lower(), hint=_mask(cleaned))


def clear_stored_key(provider: str) -> bool:
    """Remove the stored key for ``provider``. Returns whether one was actually removed."""
    field, _ = _fields(provider)
    data = _load()
    if field not in data:
        return False
    data.pop(field)
    _save(data)
    log.info("api_key_cleared", provider=provider.strip().lower())
    return True


def env_key(provider: str) -> str | None:
    """The key ``provider`` gets from the environment/``.env``, or ``None``.

    Read as an **attribute** of ``config`` (never a ``from config import`` binding) so a test's
    ``monkeypatch.setattr(config, "ANTHROPIC_API_KEY", ...)`` — and any future runtime reload —
    is actually seen here.
    """
    _, constant = _fields(provider)
    value = getattr(config, constant, None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve_key(provider: str) -> str | None:
    """The key actually used for ``provider``: the environment first, then the stored key."""
    return env_key(provider) or get_stored_key(provider)


def key_source(provider: str) -> KeySource | None:
    """Where the live key comes from — ``"env"``, ``"app"``, or ``None`` if there is none."""
    if env_key(provider):
        return "env"
    if get_stored_key(provider):
        return "app"
    return None


def _mask(key: str) -> str:
    """A displayable remnant of a key: the last 4 characters, never more."""
    tail = key.strip()[-4:]
    return f"...{tail}" if tail else ""


def key_hint(provider: str) -> str | None:
    """Last-4 hint for the live key, for display. ``None`` when no key is configured."""
    key = resolve_key(provider)
    return _mask(key) if key else None


# ============================================================
# Keys bound to an address (ADR-060 — every provider but Claude)
# ============================================================
# `keys: {id: {key, address}}` in the same file. A key is sent to the address it was saved with and
# nowhere else, so a lookup names the address it is for and gets nothing when that differs. These
# providers read this store only: no environment variable is consulted, so a key exported for
# another tool is never used without the user typing it here.


def same_address(a: str, b: str) -> bool:
    """Whether two addresses are the same, ignoring case and a trailing slash."""
    return a.strip().rstrip("/").lower() == b.strip().rstrip("/").lower()


def _bound(provider: str) -> dict[str, str] | None:
    keys = _load().get("keys")
    entry = keys.get(provider) if isinstance(keys, dict) else None
    if not isinstance(entry, dict):
        return None
    key, address = entry.get("key"), entry.get("address")
    if isinstance(key, str) and key.strip() and isinstance(address, str):
        return {"key": key.strip(), "address": address}
    return None


def get_bound_key(provider: str, address: str) -> str | None:
    """The key saved for ``provider`` if it was saved for ``address``; else ``None``."""
    entry = _bound(provider)
    if entry is None or not same_address(entry["address"], address):
        return None
    return entry["key"]


def bound_address(provider: str) -> str | None:
    """The address ``provider``'s key was saved with, or ``None`` when it has no key."""
    entry = _bound(provider)
    return entry["address"] if entry else None


def bound_key_hint(provider: str, address: str) -> str | None:
    """Last-4 hint for the key bound to ``address``. ``None`` when there is none."""
    key = get_bound_key(provider, address)
    return _mask(key) if key else None


def set_bound_key(provider: str, key: str, address: str) -> None:
    """Store ``key`` for ``provider``, bound to ``address``. A blank key raises ``ValueError``."""
    cleaned = key.strip()
    if not cleaned:
        raise ValueError("API key must not be empty")
    data = _load()
    stored = data.get("keys")
    keys: dict[str, Any] = stored if isinstance(stored, dict) else {}
    keys[provider] = {"key": cleaned, "address": address.strip()}
    data["keys"] = keys
    _save(data)
    log.info("api_key_stored", provider=provider, hint=_mask(cleaned))


def clear_bound_key(provider: str) -> bool:
    """Remove ``provider``'s bound key. Returns whether one was removed."""
    data = _load()
    keys = data.get("keys")
    if not isinstance(keys, dict) or provider not in keys:
        return False
    keys.pop(provider)
    _save(data)
    log.info("api_key_cleared", provider=provider)
    return True
