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
from sqlalchemy import select

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
    """ADR-014's behaviour, now reached by opting in (`delete_file=True`) — spec case 7."""
    from doc_assistant.library import delete_document

    src = tmp_path / "paper.pdf"
    src.write_text("x")
    doc_id = _seed_doc("paper.pdf", source_original=str(src))
    chroma = FakeChroma({"hash-paper.pdf": ["c1", "c2", "c3"]})
    trashed: list[str] = []
    monkeypatch.setattr("send2trash.send2trash", lambda p: trashed.append(p))

    result = delete_document(doc_id, chroma, delete_file=True)

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

    # Opts in deliberately: with `delete_file=False` this would pass without exercising anything,
    # since nothing is trashed either way.
    result = delete_document(doc_id, FakeChroma({}), delete_file=True)

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
        delete_document(doc_id, FakeChroma({"hash-locked.pdf": ["c1"]}), delete_file=True)
    assert _exists(doc_id)  # the row survives a failed trash — no orphaned indexed file


def _client(chroma: FakeChroma) -> TestClient:
    return TestClient(create_app(controller=_FakeController(chroma)))  # type: ignore[arg-type]


def test_delete_route(temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "paper.pdf"
    src.write_text("x")
    doc_id = _seed_doc("paper.pdf", source_original=str(src))
    monkeypatch.setattr("send2trash.send2trash", lambda p: None)
    client = _client(FakeChroma({"hash-paper.pdf": ["c1", "c2"]}))

    r = client.delete(f"/api/library/documents/{doc_id}?delete_file=true")
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

    assert client.delete(f"/api/library/documents/{doc_id}?delete_file=true").status_code == 409
    assert _exists(doc_id)  # not deleted


# ============================================================
# ADR-046 §2 — delete asks, and defaults to library-only (spec cases 6 and 7)
# ============================================================


def test_delete_defaults_to_library_only_and_leaves_the_file(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec case 6, and the half of ADR-046 that changes what `delete` *means* by default.

    ADR-014 binned the source unconditionally, which is right for a copy the app made and wrong for
    a file the user keeps in their own folder. The default is now the branch that cannot destroy
    anything.
    """
    from doc_assistant.library import delete_document

    src = tmp_path / "mine.pdf"
    src.write_text("the user's own file")
    doc_id = _seed_doc("mine.pdf", source_original=str(src))
    chroma = FakeChroma({"hash-mine.pdf": ["c1", "c2"]})
    monkeypatch.setattr(
        "send2trash.send2trash", lambda p: pytest.fail("the default must never bin a file")
    )

    result = delete_document(doc_id, chroma)

    assert result is not None
    assert result.trashed_file is False
    assert src.exists(), "the file is the user's; library-only delete must not touch it"
    assert src.read_text() == "the user's own file", "and must not rewrite it either"
    assert result.chunks_removed == 2
    assert chroma.deleted == ["c1", "c2"], "it does still leave the search index"
    assert not _exists(doc_id), "and the library row still goes"


def test_a_library_only_delete_keeps_the_registry_row(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is still on disk and no longer indexed — which is precisely `new`, not `missing`.

    Keeping the row is what lets the next scan offer the file again, truthfully.
    """
    from doc_assistant.library import delete_document

    src = tmp_path / "kept.pdf"
    src.write_text("x")
    doc_id = _seed_doc("kept.pdf", source_original=str(src))
    _seed_source_row("kept.pdf", root_path=tmp_path)

    delete_document(doc_id, FakeChroma({}))

    assert _source_rows() == ["kept.pdf"], "the file is still there, so the row is still true"


def test_deleting_the_file_forgets_the_registry_row(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KI-52. The row used to survive, so a file the user deleted *through the app* came back as
    `missing` in Sources — the app misreporting its own action, with no way to clear it."""
    from doc_assistant.library import delete_document

    src = tmp_path / "binned.pdf"
    src.write_text("x")
    doc_id = _seed_doc("binned.pdf", source_original=str(src))
    _seed_source_row("binned.pdf", root_path=tmp_path)
    monkeypatch.setattr("send2trash.send2trash", lambda p: None)

    delete_document(doc_id, FakeChroma({}), delete_file=True)

    assert _source_rows() == [], "the file is gone, so the row can only ever say `missing`"


def test_forgetting_a_row_resolves_through_its_own_root(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-named file under another root must survive — the mistake that once deleted an
    unrelated document out of the library (see `library/add.py::_delete_copied_file`)."""
    from doc_assistant.library import delete_document

    library = tmp_path / "library"
    zotero = tmp_path / "zotero"
    library.mkdir()
    zotero.mkdir()
    (library / "notes.pdf").write_text("library copy")
    target = zotero / "notes.pdf"
    target.write_text("the one being deleted")

    doc_id = _seed_doc("notes.pdf", source_original=str(target))
    _seed_source_row("notes.pdf", root_path=library, root_id="library")
    _seed_source_row("notes.pdf", root_path=zotero, root_id="zot", kind="referenced")
    monkeypatch.setattr("send2trash.send2trash", lambda p: None)

    delete_document(doc_id, FakeChroma({}), delete_file=True)

    with session_scope() as session:
        from doc_assistant.db.models import SourceFile

        remaining = [
            (r.root_id, r.rel_path) for r in session.execute(select(SourceFile)).scalars()
        ]
    assert remaining == [("library", "notes.pdf")], "only the row under the named root goes"
    assert (library / "notes.pdf").exists(), "the same-named library file is untouched"


def _seed_source_row(
    rel_path: str, *, root_path: Path, root_id: str = "library", kind: str = "library"
) -> None:
    from doc_assistant.db.models import SourceFile, SourceRoot

    with session_scope() as session:
        if session.get(SourceRoot, root_id) is None:
            session.add(SourceRoot(id=root_id, path=str(root_path), kind=kind))
            session.flush()
        session.add(
            SourceFile(root_id=root_id, rel_path=rel_path, format="pdf", size=1, mtime=0.0)
        )


def _source_rows() -> list[str]:
    from doc_assistant.db.models import SourceFile

    with session_scope() as session:
        return sorted(r.rel_path for r in session.execute(select(SourceFile)).scalars())
