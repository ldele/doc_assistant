"""Integration tests for the definitions router (ADR-053, ROADMAP 93 slice a).

The rules live in ``knowledge.definitions`` and are unit-tested in
``tests/unit/test_definitions.py``; here the HTTP side: every mutation answers with the refreshed
candidates, a candidate from another concept is a 404 rather than a silent write, and the
vocabulary search reaches concepts the graph does not show. Temp DB, fake controller — no model
load, no vector store, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.db.models import Concept
from doc_assistant.db.session import session_scope


class FakeController:
    def chunk_count(self) -> int:
        return 0


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield
    engine.dispose()


@pytest.fixture
def client(temp_db: None) -> TestClient:
    return TestClient(create_app(controller=FakeController()))


def _seed() -> None:
    with session_scope() as s:
        s.add(Concept(id="kd", label="knowledge distillation", kind="concept", graph_include=True))
        s.add(Concept(id="viral", label="viral", kind="concept", graph_include=False))
        s.add(Concept(id="virus", label="virus", kind="concept", graph_include=True))
        s.add(Concept(id="field", label="Virology", kind="domain"))


def test_a_concept_with_nothing_yet_says_so(client: TestClient) -> None:
    _seed()
    body = client.get("/api/concepts/kd/definitions").json()
    assert body["candidates"] == [] and body["chosen_id"] is None
    assert body["thin"] is True and body["extracted"] is False and body["can_undo"] is False
    assert client.get("/api/concepts/nope/definitions").status_code == 404
    assert client.get("/api/concepts/field/definitions").status_code == 404  # a field


def test_write_choose_dismiss_undo_round_trip(client: TestClient) -> None:
    _seed()
    first = client.post(
        "/api/concepts/kd/definitions", json={"text": "A student mimics a teacher."}
    )
    assert first.status_code == 201
    body = first.json()
    mine = body["chosen_id"]
    assert body["candidates"][0]["source"] == "user" and body["candidates"][0]["grade"] == "user"

    second = client.post(
        "/api/concepts/kd/definitions", json={"text": "Model compression.", "choose": False}
    ).json()
    other = next(c["id"] for c in second["candidates"] if c["id"] != mine)
    assert second["chosen_id"] == mine  # choose=False replaced nothing

    chosen = client.post(f"/api/concepts/kd/definitions/{other}/choose").json()
    assert chosen["chosen_id"] == other
    assert {c["id"]: c["status"] for c in chosen["candidates"]}[mine] == "suggested"

    dismissed = client.post(f"/api/concepts/kd/definitions/{other}/dismiss").json()
    assert dismissed["chosen_id"] is None

    undone = client.post("/api/concepts/kd/definitions/undo").json()
    assert undone["chosen_id"] == other
    client.post("/api/concepts/kd/definitions/undo")
    back = client.post("/api/concepts/kd/definitions/undo").json()
    assert back["chosen_id"] is None  # the first write's choice undone too
    assert len(back["candidates"]) == 2  # and nothing was ever deleted
    assert client.post("/api/concepts/kd/definitions/undo").status_code == 409


def test_a_candidate_of_another_concept_is_refused(client: TestClient) -> None:
    _seed()
    theirs = client.post("/api/concepts/virus/definitions", json={"text": "An infectious agent."})
    theirs_id = theirs.json()["chosen_id"]
    assert client.post(f"/api/concepts/kd/definitions/{theirs_id}/choose").status_code == 404
    assert client.post(f"/api/concepts/kd/definitions/{theirs_id}/dismiss").status_code == 404
    assert client.post(f"/api/concepts/virus/definitions/{theirs_id}/restore").status_code == 409


def test_an_empty_definition_is_refused(client: TestClient) -> None:
    _seed()
    assert client.post("/api/concepts/kd/definitions", json={"text": ""}).status_code == 422
    assert client.post("/api/concepts/kd/definitions", json={"text": "   "}).status_code == 400


def test_extraction_from_the_app_stores_suggestions_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel's "look in my library" runs the same extractor as the CLI, on this concept only.
    The vector store is replaced by two chunks; nothing gets chosen."""
    _seed()
    chunks = [
        ("s:p0", "s", "Knowledge distillation refers to training a small model on a large one."),
    ]
    monkeypatch.setattr(
        "doc_assistant.knowledge.definitions.chunks_mentioning", lambda _labels: chunks
    )
    body = client.post("/api/concepts/kd/definitions/extract").json()
    assert body["extracted"] is True and body["chosen_id"] is None
    (candidate,) = body["candidates"]
    assert candidate["source"] == "passage" and candidate["grade"] == "strong"
    assert candidate["chunk_key"] == "s:p0" and candidate["reasons"]
    again = client.post("/api/concepts/kd/definitions/extract").json()
    assert len(again["candidates"]) == 1  # idempotent


def test_vocabulary_search_reaches_concepts_off_the_graph(client: TestClient) -> None:
    _seed()
    matches = client.get("/api/concepts/search", params={"q": "vir"}).json()
    assert [m["label"] for m in matches] == ["virus", "viral"]  # graph first among prefix matches
    assert matches[1]["on_graph"] is False
    assert [m["label"] for m in client.get("/api/concepts/search?q=viral").json()] == ["viral"]
    assert client.get("/api/concepts/search?q=").json() == []
    assert client.get("/api/concepts/search?q=%25").json() == []  # a literal %, not a wildcard
