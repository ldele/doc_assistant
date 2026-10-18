"""Security S-6 (docs/security.md S1b): a request that changes something carries the launch token.

The desktop shell mints a token per launch and gives it to the sidecar in ``DOC_API_TOKEN`` and to
its window; the window sends it in ``X-Doc-Api-Token``. With a token set, adding, indexing and
deleting without it answer 401, reads stay open, and the 401 carries the CORS headers so the window
can read it. Without one, nothing is checked (``just api``, the suite, the headless image).
"""

from __future__ import annotations

import pytest
from apps.api import launch_token
from apps.api.launch_token import TOKEN_HEADER, configured_token
from apps.api.main import create_app
from fastapi.testclient import TestClient

TOKEN = "launch-token-for-the-test"
TAURI = "tauri://localhost"


class _FakeController:
    def chunk_count(self) -> int:
        return 0


def _client(monkeypatch: pytest.MonkeyPatch, token: str | None) -> TestClient:
    if token is None:
        monkeypatch.delenv("DOC_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("DOC_API_TOKEN", token)
    app = create_app(controller=_FakeController())  # type: ignore[arg-type]

    def probe() -> dict[str, bool]:
        return {"reached": True}

    app.add_api_route("/probe", probe, methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    return TestClient(app)


def test_add_ingest_and_delete_answer_401_without_the_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(monkeypatch, TOKEN)
    for r in (
        client.post("/api/documents/add", json={"paths": ["C:/Windows/win.ini"]}),
        client.post("/api/ingest", json={}),
        client.delete("/api/library/documents/any-id"),
    ):
        assert r.status_code == 401
        assert r.json() == {"detail": "This request needs the token the app was started with."}


def test_a_wrong_or_empty_token_is_refused_too(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, TOKEN)
    for sent in ("not-the-token", "", TOKEN + "x", TOKEN.upper()):
        r = client.post("/api/documents/add", json={}, headers={TOKEN_HEADER: sent})
        assert r.status_code == 401


def test_the_token_lets_the_request_reach_the_route(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, TOKEN)
    # An empty body fails the route's own validation: the request got past the token check.
    r = client.post("/api/documents/add", json={}, headers={TOKEN_HEADER: TOKEN})
    assert r.status_code == 422
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        r = client.request(method, "/probe", headers={TOKEN_HEADER: TOKEN})
        assert r.status_code == 200, method


def test_every_method_that_can_change_something_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(monkeypatch, TOKEN)
    assert client.get("/probe").json() == {"reached": True}
    assert client.get("/api/health").status_code == 200
    assert client.head("/probe").status_code != 401
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert client.request(method, "/probe").status_code == 401, method


def test_a_refusal_carries_cors_headers_and_a_preflight_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(monkeypatch, TOKEN)
    preflight = client.options(
        "/api/documents/add",
        headers={
            "Origin": TAURI,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{TOKEN_HEADER.lower()}",
        },
    )
    assert preflight.status_code == 200
    refused = client.post("/api/documents/add", json={}, headers={"Origin": TAURI})
    assert refused.status_code == 401
    assert refused.headers["access-control-allow-origin"] == TAURI


def test_without_a_token_nothing_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    for unset in (None, "", "   "):
        client = _client(monkeypatch, unset)
        assert client.post("/api/documents/add", json={}).status_code == 422
        assert client.delete("/probe").status_code == 200


def test_the_setting_is_read_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_API_TOKEN", f"  {TOKEN}\n")
    assert configured_token() == TOKEN
    monkeypatch.setenv("DOC_API_TOKEN", " ")
    assert configured_token() is None
    monkeypatch.delenv("DOC_API_TOKEN")
    assert configured_token() is None


class _Warnings:
    """Records ``.warning`` calls, independent of the global structlog configuration."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, **kw: object) -> None:
        self.events.append((event, kw))


def test_a_refusal_is_logged_without_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, TOKEN)
    recorder = _Warnings()
    monkeypatch.setattr(launch_token, "log", recorder)
    client.delete("/probe", headers={TOKEN_HEADER: "a-guess"})
    client.post("/probe")
    assert recorder.events == [
        ("security_token_refused", {"method": "DELETE", "path": "/probe", "header_present": True}),
        ("security_token_refused", {"method": "POST", "path": "/probe", "header_present": False}),
    ]
    assert TOKEN not in str(recorder.events) and "a-guess" not in str(recorder.events)
