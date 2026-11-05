"""The library summary routes (ROADMAP 112), over the wire.

`GET /api/library/summary` and the document list's `issue` filter that *Show the N* opens. The
counts themselves are `tests/unit/test_library_summary.py`'s; this file checks what the HTTP layer
adds: status codes, the session's last run, and that the wire carries UTC timestamps.
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

from doc_assistant.db.models import Base, Document, Folder
from doc_assistant.db.session import session_scope


class FakeController:
    """The minimum surface `create_app` needs; the summary routes touch none of it."""

    chunk_count = 0

    def corpus_stats(self) -> dict[str, int]:
        return {"chunk_count": 0}


class FakeRun:
    """An `IngestRun` as the route reads it: only `counts()`."""

    def __init__(self, failed: int) -> None:
        self.failed = failed

    def counts(self) -> dict[str, int]:
        return {"failed": self.failed, "added": 3, "total": 3 + self.failed}


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


def _doc(name: str, **kw: object) -> str:
    fields: dict[str, object] = {
        "filename": name,
        "source_original": f"C:/library/{name}",
        "doc_hash": name,
        "format": "pdf",
        "title": name,
        "authors": "A. Author",
        "year": 2021,
        "extraction_health": "healthy",
        "chunk_count": 4,
        "page_count": 2,
    }
    fields.update(kw)
    with session_scope() as session:
        doc = Document(**fields)
        session.add(doc)
        session.flush()
        return str(doc.id)


def test_summary_at_zero_documents(client: TestClient) -> None:
    r = client.get("/api/library/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["scope"] == "library"
    assert body["documents"] == 0
    assert body["need_attention"] == 0
    assert body["types"] == []
    assert body["years"]["by_decade"] == []
    assert body["indexing"]["last_run_failed"] is None
    assert body["counted_at"].endswith(("+00:00", "Z"))


def test_unknown_folder_is_a_404(client: TestClient) -> None:
    assert client.get("/api/library/summary", params={"folder_id": "nope"}).status_code == 404


def test_folder_scope_over_the_wire(client: TestClient) -> None:
    a = _doc("a.pdf", year=None)
    _doc("b.pdf")
    with session_scope() as session:
        folder = Folder(name="Thesis")
        session.add(folder)
        session.flush()
        doc = session.get(Document, a)
        assert doc is not None
        doc.folders.append(folder)
        folder_id = str(folder.id)
    body = client.get("/api/library/summary", params={"folder_id": folder_id}).json()
    assert body["scope"] == "folder"
    assert body["folder_name"] == "Thesis"
    assert body["documents"] == 1
    assert body["metadata"]["missing_year"] == 1
    assert body["indexing"]["files_not_in_library"] is None


def test_last_run_failures_come_from_the_session(client: TestClient) -> None:
    client.app.state.ingest_run = FakeRun(failed=2)  # type: ignore[attr-defined]
    body = client.get("/api/library/summary").json()
    assert body["indexing"]["last_run_failed"] == 2


def test_documents_issue_filter(client: TestClient) -> None:
    _doc("dated.pdf")
    undated = _doc("undated.pdf", year=None)
    everything = client.get("/api/library/documents").json()
    assert len(everything) == 2  # no parameter: today's list, unchanged
    r = client.get("/api/library/documents", params={"issue": "missing_year"})
    assert r.status_code == 200
    assert [d["id"] for d in r.json()] == [undated]
    summary = client.get("/api/library/summary").json()
    assert summary["metadata"]["missing_year"] == len(r.json())


def test_unknown_issue_is_a_400(client: TestClient) -> None:
    r = client.get("/api/library/documents", params={"issue": "nope"})
    assert r.status_code == 400
    assert "missing_year" in r.json()["detail"]
