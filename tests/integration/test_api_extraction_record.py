"""The extraction-record route (0.9.0), over the wire.

`GET /api/library/documents/{id}/extraction-record`. What the record says is
`tests/unit/test_extraction_record.py`'s; this file checks what the HTTP layer adds: the 404, the
null record of a document never recorded, and that a written record validates against the model.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from doc_assistant.db.models import Base, Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest import record


class FakeController:
    """The minimum surface `create_app` needs; the record route touches none of it."""

    chunk_count = 0

    def corpus_stats(self) -> dict[str, int]:
        return {"chunk_count": 0}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    Base.metadata.create_all(engine)
    yield TestClient(create_app(controller=FakeController()))
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(db_path)


def _doc(tmp_path: Path) -> str:
    cached = tmp_path / "a.md"
    cached.write_text("\n<!-- page:1 -->\nTable 1: x\n", encoding="utf-8")
    with session_scope() as session:
        doc = Document(
            filename="a.pdf",
            source_original=str(tmp_path / "a.pdf"),
            source_cache=str(cached),
            doc_hash="h1",
            format="pdf",
            title="A",
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def test_an_unknown_document_is_a_404(client: TestClient) -> None:
    assert client.get("/api/library/documents/nope/extraction-record").status_code == 404


def test_a_document_never_recorded_has_a_null_record(client: TestClient, tmp_path: Path) -> None:
    doc = _doc(tmp_path)
    r = client.get(f"/api/library/documents/{doc}/extraction-record")
    assert r.status_code == 200
    assert r.json() == {"document_id": doc, "record": None}


def test_a_written_record_comes_back_with_its_states_and_warnings(
    client: TestClient, tmp_path: Path
) -> None:
    doc = _doc(tmp_path)
    record.write_record(
        doc,
        "index",
        {"references": ("ok", {"heading": True, "entries": 0, "notes": [], "seconds": 0.0})},
    )
    body = client.get(f"/api/library/documents/{doc}/extraction-record").json()["record"]
    assert body["states"] == {
        "references": "none_found",
        "figures": "not_run",
        "tables": "not_run",
    }
    ids = {w["id"] for w in body["warnings"]}
    assert "references_heading_only" in ids and "metadata_missing" in ids
    assert body["stale"] is False
