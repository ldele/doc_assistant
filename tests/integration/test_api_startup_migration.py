"""E0.5a — a failed startup migration fails the boot, it is not swallowed.

KI-23 moved ``init_db()`` into the API lifespan so a stale schema is migrated before serving. But
the answer path now carries additive columns (ADR-025 F2), so a **failed** migration would break
every chat turn at runtime — a worse, later failure than refusing to start. These guards pin that
the lifespan re-raises on a migration error (boot fails with a clear message) and still boots
cleanly when the migration succeeds.

Non-vacuous: with the earlier ``except Exception: log.error(...)`` swallow, entering the app's
lifespan succeeds even when ``init_db`` raises, so ``test_boot_fails_when_migration_fails`` fails.

Deterministic + offline: ``init_db`` and the controller are both stubbed — no real DB, no models.
"""

from __future__ import annotations

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient


class _FakeController:
    def chunk_count(self) -> int:
        return 0


def test_boot_fails_when_migration_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> list[str]:
        raise RuntimeError("column add failed: disk I/O error")

    monkeypatch.setattr("apps.api.main.init_db", _boom)
    app = create_app(controller=_FakeController())  # type: ignore[arg-type]
    # Entering the TestClient runs the lifespan startup, where the failed migration must re-raise.
    with pytest.raises(RuntimeError, match="refusing to serve a stale schema"), TestClient(app):
        pass


def test_boot_succeeds_when_migration_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.api.main.init_db", lambda: [])  # no columns added, no error
    app = create_app(controller=_FakeController())  # type: ignore[arg-type]
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
