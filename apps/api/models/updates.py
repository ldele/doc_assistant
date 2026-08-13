"""Update-check wire models (ADR-044).

Mirrors ``doc_assistant.update_check.UpdateStatus`` plus the one user-settable field. The
TypeScript contract lives at ``apps/desktop/src/lib/core/types/updates.ts`` and changes in the
same commit as this file (``apps/api/CLAUDE.md``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class UpdateStatusOut(BaseModel):
    """What the app knows about whether a newer release exists.

    ``state`` has three values, not two. ``unknown`` means "never checked" or "the check failed" —
    a distinction the user does not need, since the action is the same. What it must never mean is
    ``current``: reporting "up to date" because the network was down is the failure ADR-044 exists
    to prevent, so ``reason`` carries the plain-words explanation instead.
    """

    state: Literal["current", "update_available", "unknown"]
    current_version: str
    latest_version: str | None = None
    release_url: str
    checked_at: str | None = None
    reason: str | None = None
    # Whether the *automatic* daily check is on. A manual check runs regardless (ADR-044).
    auto_check_enabled: bool


class UpdateSettingsUpdate(BaseModel):
    """Turn the automatic daily check on or off. The only writable field in this domain —
    the repository checked against is deliberately not configurable (ADR-044: pointing an update
    banner at an arbitrary host is a way to get someone to install something they did not choose).
    """

    auto_check_enabled: bool
