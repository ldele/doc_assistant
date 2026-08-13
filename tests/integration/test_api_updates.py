"""Integration tests for the update-check endpoints (ADR-044).

No socket is opened: `update_check.check_now` is patched **on the router module**, which is the
binding the handlers actually call (`src/doc_assistant/CLAUDE.md` — patch the module that owns the
name, never a package that re-exports it). The persisted settings file is redirected to a temp
path so no test touches the real data home.

What these tests pin is the split that *is* the design: a GET must not cost a network request
unless the user opted in and one is due, while an explicit POST /check always runs one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from apps.api.main import create_app
from apps.api.routers import updates as updates_router
from fastapi.testclient import TestClient

from doc_assistant import __version__, app_settings, update_check


class FakeController:
    def chunk_count(self) -> int:
        return 0


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    return TestClient(create_app(controller=FakeController()))


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count network checks, and return a canned 'update available' each time."""
    seen: list[int] = []

    def fake_check_now() -> update_check.UpdateStatus:
        seen.append(1)
        return update_check.UpdateStatus(
            "update_available",
            __version__,
            "99.0.0",
            update_check.RELEASES_PAGE_URL,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    monkeypatch.setattr(updates_router.update_check, "check_now", fake_check_now)
    return seen


def test_get_does_not_hit_the_network_when_auto_check_is_off(
    client: TestClient, calls: list[int]
) -> None:
    """The default. A page load must never cost a request the user did not opt into."""
    body = client.get("/api/updates").json()
    assert calls == []
    assert body["state"] == "unknown"
    assert body["auto_check_enabled"] is False
    assert body["current_version"] == __version__


def test_get_checks_when_enabled_and_due(client: TestClient, calls: list[int]) -> None:
    client.post("/api/updates/settings", json={"auto_check_enabled": True})
    body = client.get("/api/updates").json()
    assert len(calls) == 1
    assert body["state"] == "update_available"
    assert body["latest_version"] == "99.0.0"


def test_a_second_get_inside_the_interval_does_not_recheck(
    client: TestClient, calls: list[int]
) -> None:
    client.post("/api/updates/settings", json={"auto_check_enabled": True})
    client.get("/api/updates")
    client.get("/api/updates")
    assert len(calls) == 1


def test_the_check_is_due_again_after_the_interval(
    client: TestClient, calls: list[int], tmp_path: Path
) -> None:
    client.post("/api/updates/settings", json={"auto_check_enabled": True})
    client.get("/api/updates")
    stale = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    app_settings.set_update_last_checked(stale)
    client.get("/api/updates")
    assert len(calls) == 2


def test_manual_check_runs_even_with_auto_check_off(client: TestClient, calls: list[int]) -> None:
    """An explicit press is its own consent (ADR-044) — otherwise a user who declined background
    traffic could never find out whether they are current."""
    body = client.post("/api/updates/check").json()
    assert len(calls) == 1
    assert body["state"] == "update_available"
    assert body["auto_check_enabled"] is False


def test_a_failed_check_reports_unknown_and_still_stamps_the_clock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two things at once: the offline answer is never "current", and the failure still rate-limits
    so a down server is not retried on every launch."""
    monkeypatch.setattr(
        updates_router.update_check,
        "check_now",
        lambda: update_check.UpdateStatus(
            "unknown",
            __version__,
            checked_at=datetime.now(timezone.utc).isoformat(),
            reason="could not reach the update server",
        ),
    )
    body = client.post("/api/updates/check").json()
    assert body["state"] == "unknown"
    assert body["latest_version"] is None
    assert body["reason"] == "could not reach the update server"
    assert app_settings.get_update_last_checked() is not None


def test_an_update_found_earlier_survives_the_next_page_load(
    client: TestClient, calls: list[int]
) -> None:
    """The regression this design exists to avoid: a GET inside the 24 h window must not forget
    the update the last check found and silently drop the banner."""
    client.post("/api/updates/check")
    body = client.get("/api/updates").json()
    assert len(calls) == 1, "the cached read must not re-check"
    assert body["state"] == "update_available"
    assert body["latest_version"] == "99.0.0"
    assert body["checked_at"] is not None, "a cached verdict must say how old it is"


def test_the_cached_verdict_is_recomputed_not_replayed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored *version* rather than a stored verdict means that once the running version
    catches up, the same cached observation reads `current` with no further network call."""
    monkeypatch.setattr(
        updates_router.update_check,
        "check_now",
        lambda: update_check.UpdateStatus(
            "update_available",
            "0.0.1",
            __version__,  # the release that was found IS what we are now running
            update_check.RELEASES_PAGE_URL,
            checked_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    client.post("/api/updates/check")
    assert client.get("/api/updates").json()["state"] == "current"


def test_a_failed_check_clears_a_previous_verdict(
    client: TestClient, calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale "update available" must not keep showing under a fresh timestamp that no
    successful check backs."""
    client.post("/api/updates/check")
    monkeypatch.setattr(
        updates_router.update_check,
        "check_now",
        lambda: update_check.UpdateStatus(
            "unknown",
            __version__,
            checked_at=datetime.now(timezone.utc).isoformat(),
            reason="could not reach the update server",
        ),
    )
    client.post("/api/updates/check")
    body = client.get("/api/updates").json()
    assert body["state"] == "unknown"
    assert body["latest_version"] is None


def test_the_toggle_round_trips(client: TestClient) -> None:
    assert client.post("/api/updates/settings", json={"auto_check_enabled": True}).json()[
        "auto_check_enabled"
    ]
    assert client.get("/api/updates").json()["auto_check_enabled"] is True
    assert (
        client.post("/api/updates/settings", json={"auto_check_enabled": False}).json()[
            "auto_check_enabled"
        ]
        is False
    )


def test_the_toggle_rejects_a_missing_field(client: TestClient) -> None:
    assert client.post("/api/updates/settings", json={}).status_code == 422
