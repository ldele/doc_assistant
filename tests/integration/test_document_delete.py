"""Integration tests for safe-delete (ADR-014, library.delete_document + DELETE route).

Seeds ``Document`` rows into a temp DB with a real temp file as the source. A fake Chroma records
chunk deletion; ``send2trash`` is monkeypatched so no real file is ever recycled. Covers: unknown →
None, success (file trashed + row gone + chunks removed), file-already-gone, trash-failure aborts
the delete, and the DELETE route (200 / 404 / 409).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope


class FakeChroma:
    """Chunk ids per doc_hash; `.get(where=...)` returns them, `.delete(ids=...)` records them."""

    def __init__(self, ids_by_hash: dict[str, list[str]]) -> None:
        self._ids = ids_by_hash
        self.deleted: list[str] = []

    def get(self, *, where: dict[str, Any], include: list[str]) -> dict[str, Any]:
        return {"ids": list(self._ids.get(where["doc_hash"], []))}

    def delete(self, *, ids: list[str]) -> None:
        self.deleted.extend(ids)


class _FakeRag:
    def __init__(self, db: FakeChroma) -> None:
        self.db = db


class _FakeController:
    def __init__(self, db: FakeChroma) -> None:
        self.rag = _FakeRag(db)

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


def _seed_doc(filename: str, *, source_original: str) -> str:
    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=source_original,
            doc_hash=f"hash-{filename}",
            format="pdf",
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _exists(doc_id: str) -> bool:
    with session_scope() as session:
        return session.get(Document, doc_id) is not None


def test_delete_unknown_returns_none(temp_db: None) -> None:
    from doc_assistant.library import delete_document

    assert delete_document("nope", FakeChroma({})) is None


def test_delete_trashes_file_and_removes_row_and_chunks(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant.library import delete_document

    src = tmp_path / "paper.pdf"
    src.write_text("x")
    doc_id = _seed_doc("paper.pdf", source_original=str(src))
    chroma = FakeChroma({"hash-paper.pdf": ["c1", "c2", "c3"]})
    trashed: list[str] = []
    monkeypatch.setattr("send2trash.send2trash", lambda p: trashed.append(p))

    result = delete_document(doc_id, chroma)

    assert result is not None
    assert result.trashed_file is True and result.chunks_removed == 3
    assert trashed == [str(src)]  # the resolved source path was recycled
    assert chroma.deleted == ["c1", "c2", "c3"]  # its chunks left the index
    assert not _exists(doc_id)  # DB row gone


def test_delete_when_file_already_gone(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant.library import delete_document

    doc_id = _seed_doc("gone.pdf", source_original=str(tmp_path / "missing.pdf"))
    monkeypatch.setattr(
        "send2trash.send2trash", lambda p: pytest.fail("should not trash a missing file")
    )

    result = delete_document(doc_id, FakeChroma({}))

    assert result is not None and result.trashed_file is False
    assert not _exists(doc_id)  # still removed from the library


def test_delete_aborts_when_trash_fails(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant.library import delete_document

    src = tmp_path / "locked.pdf"
    src.write_text("x")
    doc_id = _seed_doc("locked.pdf", source_original=str(src))

    def boom(_p: str) -> None:
        raise OSError("locked")

    monkeypatch.setattr("send2trash.send2trash", boom)

    with pytest.raises(RuntimeError):
        delete_document(doc_id, FakeChroma({"hash-locked.pdf": ["c1"]}))
    assert _exists(doc_id)  # the row survives a failed trash — no orphaned indexed file


def _client(chroma: FakeChroma) -> TestClient:
    return TestClient(create_app(controller=_FakeController(chroma)))  # type: ignore[arg-type]


def test_delete_route(temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "paper.pdf"
    src.write_text("x")
    doc_id = _seed_doc("paper.pdf", source_original=str(src))
    monkeypatch.setattr("send2trash.send2trash", lambda p: None)
    client = _client(FakeChroma({"hash-paper.pdf": ["c1", "c2"]}))

    r = client.delete(f"/api/library/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json() == {"filename": "paper.pdf", "trashed_file": True, "chunks_removed": 2}
    assert not _exists(doc_id)

    assert client.delete("/api/library/documents/nope").status_code == 404


def test_delete_route_409_on_trash_failure(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "locked.pdf"
    src.write_text("x")
    doc_id = _seed_doc("locked.pdf", source_original=str(src))

    def boom(_p: str) -> None:
        raise OSError("locked")

    monkeypatch.setattr("send2trash.send2trash", boom)
    client = _client(FakeChroma({}))

    assert client.delete(f"/api/library/documents/{doc_id}").status_code == 409
    assert _exists(doc_id)  # not deleted
