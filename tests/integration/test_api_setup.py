"""Integration tests for the first-run setup endpoints (ADR-034).

All fakes (cpc §13): a fake controller supplies ``chunk_count`` and records
``refresh_chat_model``; ``llm.ollama_probe`` and the key verifier are monkeypatched, so **no
network and no paid call** happens here. The credential store and the settings file are both
redirected to a temp path, so no test touches the real data home.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant import app_settings, config, credentials, llm


class FakeController:
    def __init__(self, count: int = 0) -> None:
        self._count = count
        self.refreshes = 0

    def chunk_count(self) -> int:
        return self._count

    def refresh_chat_model(self) -> tuple[str, str]:
        self.refreshes += 1
        return app_settings.effective_llm()


@pytest.fixture
def clean_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", tmp_path / "credentials.json")
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    # Default: no local server. Each test that needs one says so.
    monkeypatch.setattr(llm, "ollama_probe", lambda *a, **k: (False, (), "nothing listening"))
    # The library may or may not exist on the box running the suite; the count is not what these
    # tests are about, and services.setup_state_dict already degrades to 0 on failure.
    monkeypatch.setattr("doc_assistant.library.count_documents", lambda: 0)


def test_setup_reports_both_steps_outstanding_on_a_fresh_install(clean_install: None) -> None:
    client = TestClient(create_app(controller=FakeController(count=0)))
    body = client.get("/api/setup").json()
    assert body["ready"] is False
    steps = {s["id"]: s for s in body["steps"]}
    assert steps["provider"]["done"] is False
    assert steps["documents"]["done"] is False
    # Every unfinished step carries the action that finishes it — that is the point of the payload.
    assert steps["provider"]["action"] and steps["documents"]["action"]


def test_setup_never_leaks_key_material(clean_install: None) -> None:
    credentials.set_stored_key("anthropic", "sk-ant-supersecret-ZZZZ")
    client = TestClient(create_app(controller=FakeController(count=5)))
    raw = client.get("/api/setup").text
    assert "supersecret" not in raw
    assert "...ZZZZ" in raw  # only the last-4 hint crosses the wire
    anthropic = next(
        p for p in client.get("/api/setup").json()["providers"] if p["id"] == "anthropic"
    )
    assert (anthropic["configured"], anthropic["key_source"]) == (True, "app")


def test_setup_probe_false_skips_the_network(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("probe=false must not call out")

    monkeypatch.setattr(llm, "ollama_probe", _boom)
    client = TestClient(create_app(controller=FakeController()))
    ollama = next(
        p for p in client.get("/api/setup?probe=false").json()["providers"] if p["id"] == "ollama"
    )
    assert ollama["reachable"] is None


def test_setup_lists_installed_ollama_models(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "ollama_probe", lambda *a, **k: (True, ("llama3.1:8b",), None))
    client = TestClient(create_app(controller=FakeController(count=1)))
    ollama = next(p for p in client.get("/api/setup").json()["providers"] if p["id"] == "ollama")
    assert (ollama["ready"], ollama["models"]) == (True, ["llama3.1:8b"])


def test_post_key_verifies_stores_and_refreshes_the_model(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "verify_anthropic_key", lambda k, **kw: ("ok", "Key verified."))
    controller = FakeController(count=3)
    client = TestClient(create_app(controller=controller))
    r = client.post("/api/setup/anthropic-key", json={"key": "sk-ant-good"})
    assert r.status_code == 200
    body = r.json()
    assert (body["stored"], body["verification"]) == (True, "ok")
    assert credentials.get_stored_key("anthropic") == "sk-ant-good"
    # The saved key must reach the *next turn*, not the next restart.
    assert controller.refreshes == 1
    assert body["setup"]["providers"][0]["ready"] is True


def test_post_key_rejected_by_the_api_is_a_400_and_stores_nothing(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        llm, "verify_anthropic_key", lambda k, **kw: ("invalid", "The API rejected")
    )
    client = TestClient(create_app(controller=FakeController()))
    r = client.post("/api/setup/anthropic-key", json={"key": "sk-ant-bad"})
    assert r.status_code == 400
    assert "rejected" in r.json()["detail"]
    assert (
        credentials.get_stored_key("anthropic") is None
    )  # a broken install never looks configured


def test_post_key_unverifiable_is_stored_with_the_reason(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Offline / behind a proxy: discarding what the user typed would be the worse failure.
    monkeypatch.setattr(
        llm, "verify_anthropic_key", lambda k, **kw: ("unreachable", "No network.")
    )
    client = TestClient(create_app(controller=FakeController()))
    r = client.post("/api/setup/anthropic-key", json={"key": "sk-ant-maybe"})
    assert r.status_code == 200
    assert r.json()["verification"] == "unreachable"
    assert "No network." in r.json()["detail"]
    assert credentials.get_stored_key("anthropic") == "sk-ant-maybe"


def test_post_empty_key_is_a_422(clean_install: None) -> None:
    client = TestClient(create_app(controller=FakeController()))
    assert client.post("/api/setup/anthropic-key", json={"key": ""}).status_code == 422


def test_delete_key_removes_the_app_key_only(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials.set_stored_key("anthropic", "sk-ant-app")
    controller = FakeController()
    client = TestClient(create_app(controller=controller))
    body = client.delete("/api/setup/anthropic-key").json()
    assert body["removed"] is True
    assert credentials.get_stored_key("anthropic") is None
    assert controller.refreshes == 1
    # A second delete is honest about having removed nothing.
    assert client.delete("/api/setup/anthropic-key").json()["removed"] is False


def test_delete_key_survives_a_model_rebuild_failure(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Clearing the key while Anthropic is active cannot rebuild a usable model — the user must
    # still be able to remove it, and the returned setup state reports the consequence.
    class Failing(FakeController):
        def refresh_chat_model(self) -> tuple[str, str]:
            raise RuntimeError("no credential")

    credentials.set_stored_key("anthropic", "sk-ant-app")
    client = TestClient(create_app(controller=Failing()))
    r = client.delete("/api/setup/anthropic-key")
    assert r.status_code == 200
    assert r.json()["setup"]["ready"] is False
