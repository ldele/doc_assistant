"""Integration tests for the settings + ingest endpoints — the "point at a folder" flow.

All fakes (cpc §13): a fake controller supplies ``chunk_count``; an injected ``ingest_fn``
records the scope + returns canned stats; an injected ``controller_factory`` returns a fresh
fake whose ``chunk_count`` reflects the post-ingest corpus — proving the live reload. The
persisted settings file is redirected to a temp path so no test touches the real data home.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient


class FakeController:
    def __init__(self, count: int = 0) -> None:
        self._count = count
        # ADR-011 (U1c): every reconfigure() call, in order.
        self.reconfigure_calls: list[tuple[str, str]] = []

    def chunk_count(self) -> int:
        return self._count

    def reconfigure(self, provider: str, model: str) -> None:
        # Exercises the SAME validation/persistence a real ChatController.reconfigure runs
        # (app_settings.set_llm_selection) — proves the API maps its ValueError to 400 — without
        # a real RAGPipeline/pipeline swap.
        from doc_assistant import app_settings

        app_settings.set_llm_selection(provider, model)
        self.reconfigure_calls.append((provider, model))


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the persisted user settings to a temp file (never the real data home)."""
    from doc_assistant import app_settings

    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    return tmp_path


def _poll_until(client: TestClient, *, state: str, tries: int = 60) -> dict:
    for _ in range(tries):
        st = client.get("/api/ingest/status").json()
        if st["state"] == state:
            return st
        time.sleep(0.05)
    return client.get("/api/ingest/status").json()


def test_get_settings_reports_home_source_and_count(settings_file: Path) -> None:
    client = TestClient(create_app(controller=FakeController(count=42)))
    body = client.get("/api/settings").json()
    assert body["chunk_count"] == 42
    assert "data_home" in body and "source_dir" in body and "source_dir_exists" in body
    assert "top_k" in body and "provider" in body  # the read view is a superset of the knobs


def test_post_settings_sets_and_persists_source_dir(settings_file: Path, tmp_path: Path) -> None:
    docs = tmp_path / "my_papers"
    docs.mkdir()
    client = TestClient(create_app(controller=FakeController()))
    r = client.post("/api/settings", json={"source_dir": str(docs)})
    assert r.status_code == 200
    assert Path(r.json()["source_dir"]) == docs.resolve()
    assert r.json()["source_dir_exists"] is True
    # persisted: a fresh GET reflects the chosen folder
    assert Path(client.get("/api/settings").json()["source_dir"]) == docs.resolve()


def test_post_settings_rejects_nonexistent_dir(settings_file: Path, tmp_path: Path) -> None:
    client = TestClient(create_app(controller=FakeController()))
    r = client.post("/api/settings", json={"source_dir": str(tmp_path / "nope")})
    assert r.status_code == 400


# ============================================================
# ADR-011 / SPRINT-012 (U1c) — desktop provider switch
# ============================================================


def test_post_settings_provider_switch_reconfigures_controller(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("doc_assistant.config.ANTHROPIC_API_KEY", None)  # ollama needs no key
    fake = FakeController()
    client = TestClient(create_app(controller=fake))
    r = client.post("/api/settings", json={"llm_provider": "ollama", "llm_model": "llama3.1:8b"})
    assert r.status_code == 200
    assert fake.reconfigure_calls == [("ollama", "llama3.1:8b")]
    body = r.json()
    assert body["provider"] == "ollama" and body["model"] == "llama3.1:8b"
    # persisted: a fresh GET reflects the switch
    fresh = client.get("/api/settings").json()
    assert fresh["provider"] == "ollama" and fresh["model"] == "llama3.1:8b"


def test_post_settings_keyless_provider_is_400(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("doc_assistant.config.ANTHROPIC_API_KEY", None)
    fake = FakeController()
    client = TestClient(create_app(controller=fake))
    r = client.post(
        "/api/settings",
        json={"llm_provider": "anthropic", "llm_model": "claude-haiku-4-5-20251001"},
    )
    assert r.status_code == 400
    assert fake.reconfigure_calls == []  # rejected before the controller was ever touched


def test_post_settings_llm_fields_must_travel_together(settings_file: Path) -> None:
    client = TestClient(create_app(controller=FakeController()))
    assert client.post("/api/settings", json={"llm_provider": "ollama"}).status_code == 422
    assert client.post("/api/settings", json={"llm_model": "llama3"}).status_code == 422


def test_post_settings_source_dir_only_backward_compat(
    settings_file: Path, tmp_path: Path
) -> None:
    # No llm_* sent → byte-identical to pre-U1c behaviour.
    docs = tmp_path / "papers_backcompat"
    docs.mkdir()
    fake = FakeController()
    client = TestClient(create_app(controller=fake))
    r = client.post("/api/settings", json={"source_dir": str(docs)})
    assert r.status_code == 200
    assert fake.reconfigure_calls == []


# ============================================================
# ADR-027 D2 (E3) — the persisted answer-layer epistemics toggle
# ============================================================


def test_post_settings_persists_epistemics_toggle(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("doc_assistant.config.EPISTEMICS_MARKERS_ENABLED", True)
    client = TestClient(create_app(controller=FakeController()))
    r = client.post("/api/settings", json={"epistemics_markers_enabled": False})
    assert r.status_code == 200
    assert r.json()["epistemics_markers_enabled"] is False
    # persisted: a fresh GET reflects the choice (the effective value, not the config default)
    assert client.get("/api/settings").json()["epistemics_markers_enabled"] is False
    # and it survives to the module layer the turn resolution reads
    from doc_assistant import app_settings

    assert app_settings.get_markers_enabled() is False


def test_settings_view_serves_the_effective_epistemics_default(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Never set → the config default; set → the persisted choice wins (ADR-011's effective-value
    # rule, same as provider/model — the raw constant would go stale on the first toggle).
    monkeypatch.setattr("doc_assistant.config.EPISTEMICS_MARKERS_ENABLED", False)
    client = TestClient(create_app(controller=FakeController()))
    assert client.get("/api/settings").json()["epistemics_markers_enabled"] is False
    client.post("/api/settings", json={"epistemics_markers_enabled": True})
    assert client.get("/api/settings").json()["epistemics_markers_enabled"] is True


def test_post_settings_toggle_only_body_is_valid(settings_file: Path) -> None:
    # The toggle alone satisfies the at-least-one-field validator; an empty body still 422s.
    client = TestClient(create_app(controller=FakeController()))
    assert (
        client.post("/api/settings", json={"epistemics_markers_enabled": True}).status_code == 200
    )
    assert client.post("/api/settings", json={}).status_code == 422


def test_settings_view_reports_providers_list(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("doc_assistant.config.ANTHROPIC_API_KEY", None)
    client = TestClient(create_app(controller=FakeController()))
    providers = {p["id"]: p for p in client.get("/api/settings").json()["providers"]}
    assert providers["anthropic"] == {"id": "anthropic", "available": False, "paid": True}
    assert providers["ollama"] == {"id": "ollama", "available": True, "paid": False}


def test_ingest_runs_on_the_folder_and_reloads_controller(
    settings_file: Path, tmp_path: Path
) -> None:
    docs = tmp_path / "papers"
    docs.mkdir()
    seen: dict[str, str] = {}

    def fake_ingest(*, scope: str) -> dict[str, int]:
        seen["scope"] = scope
        return {"added": 3, "skipped": 1, "error": 0}

    app = create_app(
        controller=FakeController(count=0),
        ingest_fn=fake_ingest,
        controller_factory=lambda: FakeController(count=99),  # the post-ingest corpus
    )
    client = TestClient(app)
    client.post("/api/settings", json={"source_dir": str(docs)})

    r = client.post("/api/ingest")
    assert r.status_code == 202
    assert r.json()["state"] in ("running", "done")

    st = _poll_until(client, state="done")
    assert st["state"] == "done"
    assert (st["added"], st["skipped"], st["errors"]) == (3, 1, 0)
    assert seen["scope"] == str(docs.resolve())  # ingested the chosen folder
    # the controller was reloaded → the live chunk_count reflects the new corpus
    assert client.get("/api/settings").json()["chunk_count"] == 99


def test_ingest_surfaces_failure_in_status(settings_file: Path) -> None:
    def boom(*, scope: str) -> dict[str, int]:
        raise RuntimeError("disk full")

    app = create_app(
        controller=FakeController(),
        ingest_fn=boom,
        controller_factory=lambda: FakeController(),
    )
    client = TestClient(app)
    client.post("/api/ingest")
    st = _poll_until(client, state="error")
    assert st["state"] == "error"
    assert "disk full" in (st["message"] or "")


def test_ingest_rejects_concurrent_run(settings_file: Path) -> None:
    release = threading.Event()

    def slow_ingest(*, scope: str) -> dict[str, int]:
        release.wait(timeout=5)
        return {"added": 0, "skipped": 0, "error": 0}

    app = create_app(
        controller=FakeController(),
        ingest_fn=slow_ingest,
        controller_factory=lambda: FakeController(),
    )
    client = TestClient(app)
    assert client.post("/api/ingest").status_code == 202  # first starts
    assert client.post("/api/ingest").status_code == 409  # second rejected while running
    release.set()
    _poll_until(client, state="done")
