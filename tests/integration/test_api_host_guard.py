"""Security S-5 (docs/security.md S1a): the API serves only the Host headers it is told to.

The API listens on loopback, but a web page can still reach it by DNS rebinding — a hostile name
that resolves to 127.0.0.1, sent by the browser as the Host. So a foreign Host gets 400, loopback
names are served, and the list is an explicit setting, never "everything" by accident. The
suite-wide fixture allows ``testserver``; each test here sets the list it is about.
"""

from __future__ import annotations

import pytest
from apps.api.main import allowed_hosts, create_app
from fastapi.testclient import TestClient


class _FakeController:
    def chunk_count(self) -> int:
        return 0


def _status(base_url: str) -> int:
    app = create_app(controller=_FakeController())  # type: ignore[arg-type]
    return TestClient(app, base_url=base_url).get("/api/health").status_code


def test_a_foreign_host_is_refused_and_loopback_is_served(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOC_API_ALLOWED_HOSTS", raising=False)
    assert allowed_hosts() == ["127.0.0.1", "localhost"]
    assert _status("http://attacker.example") == 400  # a rebinding name that points at loopback
    assert _status("http://127.0.0.1:8001") == 200  # the desktop app and the sidecar's own URL
    assert _status("http://localhost:8001") == 200  # the Docker healthcheck's URL


def test_the_list_can_name_another_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_API_ALLOWED_HOSTS", " api.internal , 127.0.0.1 ")
    assert allowed_hosts() == ["api.internal", "127.0.0.1"]
    assert _status("http://api.internal") == 200
    assert _status("http://localhost") == 400  # the setting replaces the default, not adds to it


def test_a_blank_setting_means_loopback_not_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_API_ALLOWED_HOSTS", " , ")
    assert allowed_hosts() == ["127.0.0.1", "localhost"]
    assert _status("http://attacker.example") == 400
