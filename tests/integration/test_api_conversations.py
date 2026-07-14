"""Integration tests for the conversation-history endpoints (feature-conversation-history.md).

Seed ``AnswerRecord`` rows into a temp DB, then hit the endpoints via ``TestClient``. The
endpoints never touch the controller (they read the store directly), so a minimal fake suffices.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.db.models import AnswerRecord
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


def _seed(
    session_id: str | None, query: str, when: datetime, chunks: list[dict] | None = None
) -> None:
    with session_scope() as session:
        session.add(
            AnswerRecord(
                session_id=session_id,
                query=query,
                answer="an answer",
                retrieved_chunks_json=json.dumps(chunks or []),
                created_at=when,
            )
        )


def _client() -> TestClient:
    return TestClient(create_app(controller=FakeController()))  # type: ignore[arg-type]


def test_list_conversations_endpoint(temp_db: None) -> None:
    _seed("s1", "hello world", datetime(2026, 7, 13, 10, 0, 0))
    _seed("s1", "again", datetime(2026, 7, 13, 10, 5, 0))
    _seed(None, "orphan pre-fix row", datetime(2026, 7, 1, 9, 0, 0))

    body = _client().get("/api/conversations").json()
    assert len(body) == 1
    assert body[0]["session_id"] == "s1"
    assert body[0]["title"] == "hello world"
    assert body[0]["turn_count"] == 2
    # UTC-tagged so a browser doesn't read it as local time.
    assert body[0]["started_at"].endswith("Z") or "+00:00" in body[0]["started_at"]


def test_get_conversation_endpoint(temp_db: None) -> None:
    _seed(
        "s1",
        "q one",
        datetime(2026, 7, 13, 10, 0, 0),
        chunks=[{"filename": "p.pdf", "page": 3, "chunk_excerpt": "ex"}],
    )
    body = _client().get("/api/conversations/s1").json()
    assert body["title"] == "q one"
    assert len(body["turns"]) == 1
    assert body["turns"][0]["sources"][0]["citation"] == "[1] p.pdf \xb7 p.3"
    assert body["turns"][0]["sources"][0]["excerpt"] == "ex"


def test_get_conversation_unknown_returns_404(temp_db: None) -> None:
    assert _client().get("/api/conversations/nope").status_code == 404


def test_patch_conversation_pin_archive_soft_delete(temp_db: None) -> None:
    _seed("s1", "keep", datetime(2026, 7, 13, 10, 0, 0))
    _seed("s2", "pin then delete", datetime(2026, 7, 13, 11, 0, 0))
    client = _client()

    # pin s1 (the older one) → it sorts first and carries the flag
    assert client.patch("/api/conversations/s1", json={"pinned": True}).json() == {"ok": True}
    body = client.get("/api/conversations").json()
    assert body[0]["session_id"] == "s1" and body[0]["pinned"] is True

    # archive surfaces as a flag but stays listed
    client.patch("/api/conversations/s1", json={"archived": True})
    assert client.get("/api/conversations").json()[0]["archived"] is True

    # soft-delete s2 → drops from the list, then restore brings it back
    client.patch("/api/conversations/s2", json={"deleted": True})
    assert [c["session_id"] for c in client.get("/api/conversations").json()] == ["s1"]
    client.patch("/api/conversations/s2", json={"deleted": False})
    assert {c["session_id"] for c in client.get("/api/conversations").json()} == {"s1", "s2"}
