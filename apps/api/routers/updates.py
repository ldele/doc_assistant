"""Updates router — "is there a newer release?", and nothing that installs one (ADR-044).

Three routes, and the split between the first two is the whole design:

* ``GET /api/updates`` reads the **cached** answer. It performs a network call only when the user
  has opted into automatic checks *and* one is due (24 h). A page load must never cost a request.
* ``POST /api/updates/check`` always performs one. An explicit press is its own consent, so it
  ignores the toggle — otherwise a user who declined background traffic could never find out
  whether they are current, which is the state the feature exists to resolve.
* ``POST /api/updates/settings`` toggles the automatic check.

No route here downloads, writes, executes or elevates anything. The only terminal action is the
``release_url`` the client opens in a browser.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from apps.api.models.updates import UpdateSettingsUpdate
from doc_assistant import __version__, app_settings, update_check

router = APIRouter()


def _cached_status() -> update_check.UpdateStatus:
    """The stored observation, with the verdict **recomputed** against the running version.

    What is persisted is the newest version the last successful check saw, never the verdict it
    reached. Recomputing has two consequences that a stored verdict would get wrong: after the
    user installs the update, the same stored version now reads ``current`` without another
    network call; and a stored ``current`` cannot outlive the release that supersedes it, because
    there is no stored ``current`` to outlive anything.

    ``checked_at`` always travels with it, so the UI can say *how old* this answer is rather than
    implying it is live.
    """
    last = app_settings.get_update_last_checked()
    if last is None:
        return update_check.UpdateStatus("unknown", __version__, reason="not checked yet")
    seen = app_settings.get_update_last_seen_version()
    if seen is None:
        # A stamp with no version means the last check failed. Still `unknown` — never `current`.
        return update_check.UpdateStatus(
            "unknown", __version__, checked_at=last, reason="the last check did not complete"
        )
    state = "update_available" if update_check.is_newer(seen, __version__) else "current"
    return update_check.UpdateStatus(state, __version__, seen, checked_at=last)


def _run_check() -> update_check.UpdateStatus:
    """Check, then stamp the clock — including on failure, so a down server is not retried
    every launch (the rate limit protects the endpoint, not just the user's network)."""
    status = update_check.check_now()
    if status.checked_at is not None:
        app_settings.set_update_last_checked(status.checked_at)
        # Cleared on failure: an answer with a fresh timestamp that no successful check backs
        # would be exactly the false confidence ADR-044's three-state rule forbids.
        app_settings.set_update_last_seen_version(status.latest_version)
    return status


def _payload(status: update_check.UpdateStatus) -> dict[str, Any]:
    return {**status.as_dict(), "auto_check_enabled": app_settings.get_update_check_enabled()}


@router.get("/api/updates")
def get_updates() -> dict[str, Any]:
    """The current answer, checking the network only if the user opted in and one is due."""
    if app_settings.get_update_check_enabled() and update_check.due_for_check(
        app_settings.get_update_last_checked()
    ):
        return _payload(_run_check())
    return _payload(_cached_status())


@router.post("/api/updates/check")
def post_check() -> dict[str, Any]:
    """Check now, regardless of the automatic-check toggle (ADR-044)."""
    return _payload(_run_check())


@router.post("/api/updates/settings")
def post_update_settings(body: UpdateSettingsUpdate) -> dict[str, Any]:
    """Turn the automatic daily check on or off."""
    app_settings.set_update_check_enabled(body.auto_check_enabled)
    return _payload(_cached_status())
