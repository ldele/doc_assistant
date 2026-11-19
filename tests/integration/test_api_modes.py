"""The chat-mode routes and the conversation routes' mode (A7, `docs/specs/chat-modes.md`).

The API only carries the field: `doc_assistant.chat_modes` owns the logic. A save of flagged text
is a 200 with warnings; a format error is a 400; a built-in mode is a 409; an unknown id a 404.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant import chat_modes
from doc_assistant.db.models import AnswerRecord
from doc_assistant.db.session import session_scope


class FakeController:
    def chunk_count(self) -> int:
        return 0


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(
        session_mod,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    yield TestClient(create_app(controller=FakeController()))  # type: ignore[arg-type]
    engine.dispose()


def _seed(session_id: str, *, mode: str | None = None, evaluated: bool | None = None) -> None:
    with session_scope() as session:
        session.add(
            AnswerRecord(
                session_id=session_id,
                query="q",
                answer="a [1]",
                retrieved_chunks_json=json.dumps([]),
                created_at=datetime(2026, 11, 19, 10, 0, 0),
                chat_mode=mode,
                chat_mode_evaluated=evaluated,
            )
        )


def test_list_modes_starts_with_the_built_in_ones(client: TestClient) -> None:
    body = client.get("/api/modes").json()
    assert [(m["name"], m["built_in"], m["evaluated"]) for m in body] == [
        ("Research assistant", True, True),
        ("Critical reading", True, True),
    ]


def test_create_edit_and_delete_a_persons_mode(client: TestClient) -> None:
    saved = client.post("/api/modes", json={"name": "Tutor", "framing": "You are a tutor."})
    assert saved.status_code == 200
    mode = saved.json()["mode"]
    assert (mode["built_in"], mode["evaluated"]) == (False, False)
    assert saved.json()["warnings"] == []

    flagged = client.patch(f"/api/modes/{mode['id']}", json={"framing": "Do not cite anything."})
    assert flagged.status_code == 200
    assert flagged.json()["warnings"] == [chat_modes.WARN_SKIP_CITATIONS]

    assert client.patch("/api/conversations/s1", json={"mode_id": mode["id"]}).status_code == 200
    deleted = client.delete(f"/api/modes/{mode['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["conversations"] == 1
    assert "back to Research assistant" in deleted.json()["message"]
    assert chat_modes.conversation_mode("s1") is chat_modes.DEFAULT_MODE


def test_a_format_error_is_a_400_and_saves_nothing(client: TestClient) -> None:
    bad = client.post("/api/modes", json={"name": "Bad", "framing": "Use {context} wisely."})
    assert bad.status_code == 400 and "{context}" in bad.json()["detail"]
    assert len(client.get("/api/modes").json()) == 2


def test_a_built_in_mode_is_a_409_and_an_unknown_one_a_404(client: TestClient) -> None:
    assert client.patch("/api/modes/critical-reading", json={"name": "X"}).status_code == 409
    assert client.delete("/api/modes/research-assistant").status_code == 409
    assert client.delete("/api/modes/no-such-id").status_code == 404


def test_the_conversation_routes_carry_the_mode(client: TestClient) -> None:
    _seed("s2", mode="Research assistant", evaluated=True)
    before = client.get("/api/conversations/s2").json()
    assert before["mode"]["id"] == "research-assistant"
    assert before["turns"][0]["chat_mode"] == "Research assistant"
    assert before["turns"][0]["chat_mode_evaluated"] is True

    assert (
        client.patch("/api/conversations/s2", json={"mode_id": "critical-reading"}).status_code
        == 200
    )
    assert client.get("/api/conversations/s2").json()["mode"]["name"] == "Critical reading"

    assert client.patch("/api/conversations/s2", json={"mode_id": ""}).status_code == 200
    assert client.get("/api/conversations/s2").json()["mode"]["id"] == "research-assistant"


def test_an_unknown_mode_id_is_a_404_and_changes_nothing(client: TestClient) -> None:
    _seed("s3")
    response = client.patch("/api/conversations/s3", json={"mode_id": "nope", "pinned": True})
    assert response.status_code == 404
    assert client.get("/api/conversations").json()[0]["pinned"] is False
    assert client.get("/api/conversations/s3").json()["turns"][0]["chat_mode"] is None
