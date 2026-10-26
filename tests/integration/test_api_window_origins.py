"""KI-67: the installed window can read what the API answers, on every platform it ships to.

The window calls `http://127.0.0.1:8001` from its own origin, so it reads a response only when the
API names that origin in `Access-Control-Allow-Origin`. Tauri 2 serves the window from
`http://tauri.localhost` on Windows and `tauri://localhost` on macOS and Linux. 0.6.0 and the
first 0.7.0 builds listed only the second: on Windows every answer was discarded and the app sat
at "still starting" with its engine running. RG-012 called the API from PowerShell, which ignores
CORS, so it passed; it now asks as the window does.
"""

from __future__ import annotations

import pytest
from apps.api.launch_token import TOKEN_HEADER
from apps.api.main import create_app
from fastapi.testclient import TestClient

WINDOWS_WINDOW = "http://tauri.localhost"
UNIX_WINDOW = "tauri://localhost"
TOKEN = "launch-token-for-the-test"


class _FakeController:
    def chunk_count(self) -> int:
        return 0


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DOC_API_TOKEN", TOKEN)
    app = create_app(controller=_FakeController())  # type: ignore[arg-type]
    return TestClient(app, base_url="http://127.0.0.1:8001")


@pytest.mark.parametrize("origin", [WINDOWS_WINDOW, UNIX_WINDOW])
def test_the_installed_window_can_read_a_read(
    monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    """The first call the window makes; on Windows it was refused on every try."""
    r = _client(monkeypatch).get("/api/health", headers={"Origin": origin})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize("origin", [WINDOWS_WINDOW, UNIX_WINDOW])
def test_the_installed_window_can_send_a_write_with_its_token(
    monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    """A write carries `X-Doc-Api-Token`, so the browser asks first; both answers must name the
    window, or the token header never leaves it."""
    client = _client(monkeypatch)
    preflight = client.options(
        "/api/documents/add",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{TOKEN_HEADER.lower()}",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == origin
    refused = client.post("/api/documents/add", json={}, headers={"Origin": origin})
    assert refused.status_code == 401  # no token: refused, and readable by the window
    assert refused.headers.get("access-control-allow-origin") == origin


def test_any_other_origin_reads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The list stays explicit: a page elsewhere gets its answer without the header, so its
    browser discards it."""
    r = _client(monkeypatch).get("/api/health", headers={"Origin": "http://attacker.example"})
    assert "access-control-allow-origin" not in r.headers
