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
from apps.api.models.setup import SetupStateModel
from fastapi.testclient import TestClient

from doc_assistant import app_settings, compatible, config, credentials, llm
from doc_assistant.providers import ModelList


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
    # LM Studio sits on this computer, so a probe asks it too.
    monkeypatch.setattr(
        compatible, "list_models", lambda *a, **k: ModelList("unreachable", detail="nothing")
    )
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


def test_setup_matches_its_wire_model(clean_install: None) -> None:
    # The route returns a plain dict, so nothing else holds the model to what is served.
    client = TestClient(create_app(controller=FakeController(count=3)))
    state = SetupStateModel.model_validate(client.get("/api/setup").json())
    assert [s.id for s in state.steps] == ["provider", "documents"]
    assert all(s.status for s in state.steps)
    assert all(p.name and p.destination for p in state.providers)


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
    r = client.post("/api/setup/providers/anthropic/key", json={"key": "sk-ant-good"})
    assert r.status_code == 200
    body = r.json()
    assert (body["stored"], body["verification"]) == (True, "ok")
    assert credentials.get_stored_key("anthropic") == "sk-ant-good"
    # The saved key must reach the *next turn*, not the next restart.
    assert controller.refreshes == 1
    claude = next(p for p in body["setup"]["providers"] if p["id"] == "anthropic")
    assert claude["ready"] is True


def test_post_key_rejected_by_the_api_is_a_400_and_stores_nothing(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        llm, "verify_anthropic_key", lambda k, **kw: ("invalid", "The API rejected")
    )
    client = TestClient(create_app(controller=FakeController()))
    r = client.post("/api/setup/providers/anthropic/key", json={"key": "sk-ant-bad"})
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
    r = client.post("/api/setup/providers/anthropic/key", json={"key": "sk-ant-maybe"})
    assert r.status_code == 200
    assert r.json()["verification"] == "unreachable"
    assert "No network." in r.json()["detail"]
    assert credentials.get_stored_key("anthropic") == "sk-ant-maybe"


def test_post_empty_key_is_a_422(clean_install: None) -> None:
    client = TestClient(create_app(controller=FakeController()))
    assert client.post("/api/setup/providers/anthropic/key", json={"key": ""}).status_code == 422


def test_delete_key_removes_the_app_key_only(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials.set_stored_key("anthropic", "sk-ant-app")
    controller = FakeController()
    client = TestClient(create_app(controller=controller))
    body = client.delete("/api/setup/providers/anthropic/key").json()
    assert body["removed"] is True
    assert credentials.get_stored_key("anthropic") is None
    assert controller.refreshes == 1
    # A second delete is honest about having removed nothing.
    assert client.delete("/api/setup/providers/anthropic/key").json()["removed"] is False


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
    r = client.delete("/api/setup/providers/anthropic/key")
    assert r.status_code == 200
    assert r.json()["setup"]["ready"] is False


# ============================================================
# ADR-060 — keys by provider id, servers of your own, the billing tick
# ============================================================

_REAL_LIST = compatible.list_models  # the fixture stubs it; these tests talk to a real socket


def _provider(body: dict, pid: str) -> dict:
    return next(p for p in body["providers"] if p["id"] == pid)


def test_a_preset_key_is_checked_on_its_free_list_and_bound_to_its_address(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant import providers
    from tests.fixtures.compatible_server import FakeCompatibleServer

    with FakeCompatibleServer(key="gsk-good") as server:
        row = providers.Preset("groq", "Groq", server.url("cloud"), "Groq", "https://x", "2026")
        monkeypatch.setitem(providers._PRESETS_BY_ID, "groq", row)
        monkeypatch.setattr(compatible, "list_models", _REAL_LIST)
        client = TestClient(create_app(controller=FakeController(count=3)))
        refused = client.post("/api/setup/providers/groq/key", json={"key": "gsk-bad"})
        assert refused.status_code == 400
        assert credentials.bound_address("groq") is None  # nothing stored
        ok = client.post("/api/setup/providers/groq/key", json={"key": "gsk-good"})
    assert ok.status_code == 200
    groq = _provider(ok.json()["setup"], "groq")
    assert (groq["ready"], groq["key_status"], groq["key_hint"]) == (True, "accepted", "...good")
    assert groq["models"] == ["fake-model", "fake-model-large"]
    assert credentials.get_bound_key("groq", server.url("cloud")) == "gsk-good"
    # The check was a model list, never a completion: nothing was billed.
    assert not server.chat_requests()


def test_an_unreachable_preset_key_is_stored_with_the_reason(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(create_app(controller=FakeController()))
    r = client.post("/api/setup/providers/openai/key", json={"key": "sk-maybe"})
    assert (r.status_code, r.json()["verification"]) == (200, "unreachable")
    assert "Saved" in r.json()["detail"]
    assert credentials.get_bound_key("openai", "https://api.openai.com/v1") == "sk-maybe"


def test_ollama_takes_no_key_and_an_unknown_provider_is_a_404(clean_install: None) -> None:
    client = TestClient(create_app(controller=FakeController()))
    assert client.post("/api/setup/providers/ollama/key", json={"key": "x"}).status_code == 400
    assert client.post("/api/setup/providers/nonesuch/key", json={"key": "x"}).status_code == 404


def test_test_connection_saves_a_server_and_lists_its_models(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fixtures.compatible_server import FakeCompatibleServer

    monkeypatch.setattr(compatible, "list_models", _REAL_LIST)
    with FakeCompatibleServer() as server:
        client = TestClient(create_app(controller=FakeController()))
        r = client.post("/api/setup/servers", json={"name": "Lab server", "address": server.url()})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "server-lab-server"
    assert body["listing"]["models"] == ["fake-model", "fake-model-large"]
    lab = _provider(body["setup"], "server-lab-server")
    assert lab["destination"].endswith("stay on this computer.")
    assert (lab["billing"], lab["can_tick"]) == ("Billing not known", True)


def test_test_connection_with_a_refused_key_saves_nothing(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fixtures.compatible_server import FakeCompatibleServer

    monkeypatch.setattr(compatible, "list_models", _REAL_LIST)
    with FakeCompatibleServer(key="right") as server:
        client = TestClient(create_app(controller=FakeController()))
        r = client.post(
            "/api/setup/servers",
            json={"name": "Lab", "address": server.url("cloud"), "key": "wrong"},
        )
    assert r.status_code == 400
    assert app_settings.get_servers() == []


def test_a_server_address_that_cannot_be_used_is_a_400(clean_install: None) -> None:
    client = TestClient(create_app(controller=FakeController()))
    r = client.post("/api/setup/servers", json={"name": "Lab", "address": "gpu-box:1234"})
    assert r.status_code == 400 and "http://" in r.json()["detail"]


def test_the_server_answering_now_cannot_be_removed(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant import providers

    sid = providers.save_server("Lab", "http://localhost:1234/v1").id
    client = TestClient(create_app(controller=FakeController()))
    monkeypatch.setattr(app_settings, "effective_llm", lambda: (sid, "m"))
    assert client.delete(f"/api/setup/servers/{sid}").status_code == 409
    monkeypatch.setattr(app_settings, "effective_llm", lambda: ("ollama", "m"))
    assert client.delete(f"/api/setup/servers/{sid}").status_code == 200
    assert client.delete(f"/api/setup/servers/{sid}").status_code == 404


def test_the_billing_tick_is_only_where_provenote_cannot_know(clean_install: None) -> None:
    from doc_assistant import providers

    sid = providers.save_server("Lab", "http://localhost:1234/v1").id
    client = TestClient(create_app(controller=FakeController()))
    r = client.post(f"/api/setup/providers/{sid}/billing", json={"not_billed": True})
    assert _provider(r.json()["setup"], sid)["billing"] == "Not billed (your word)"
    groq = client.post("/api/setup/providers/groq/billing", json={"not_billed": True})
    assert groq.status_code == 400


def test_the_models_route_asks_the_server_once(
    clean_install: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant import providers
    from tests.fixtures.compatible_server import FakeCompatibleServer

    monkeypatch.setattr(compatible, "list_models", _REAL_LIST)
    with FakeCompatibleServer() as server:
        sid = providers.save_server("Lab", server.url()).id
        client = TestClient(create_app(controller=FakeController()))
        assert client.get(f"/api/setup/providers/{sid}/models").status_code == 405
        listing = client.post(f"/api/setup/providers/{sid}/models").json()
        state = client.get("/api/setup?probe=false").json()
    assert listing["status"] == "ok" and listing["fetched_at"]
    # The setup state reads it back from memory, without asking again.
    assert _provider(state, sid)["models"] == ["fake-model", "fake-model-large"]
    assert len([r for r in server.requests if r["path"].endswith("/models")]) == 1


def test_settings_takes_a_preset_once_it_has_a_key(clean_install: None) -> None:
    class Switching(FakeController):
        def reconfigure(self, provider: str, model: str) -> None:
            app_settings.set_llm_selection(provider, model)

        def corpus_stats(self):  # read by the settings view the route returns
            from doc_assistant import corpus_stats as stats

            return stats.corpus_stats(documents=0, chunks=0, keyword_index_on_disk=False)

    client = TestClient(create_app(controller=Switching()))
    body = {"llm_provider": "groq", "llm_model": "llama-3.3"}
    r = client.post("/api/settings", json=body)
    assert r.status_code == 400 and "needs an API key" in r.json()["detail"]
    assert app_settings.get_llm_selection() == (None, None)
    credentials.set_bound_key("groq", "gsk-1", "https://api.groq.com/openai/v1")
    assert client.post("/api/settings", json=body).status_code == 200
    assert app_settings.get_llm_selection() == ("groq", "llama-3.3")
