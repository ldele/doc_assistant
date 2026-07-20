"""Integration tests for folders (ADR-025 F1, docs/specs/feature-corpus-folders.md).

Covers the ``library.py`` CRUD surface (create/rename/delete + bulk membership), the
invariants the spec names — idempotent case-insensitive create (D4), id-based filtering
(D2), archived documents excluded from counts (D5), folder delete never touching documents
(D6) — and the six API routes (200/400/404) plus the narrowed read-path write trap (D7).
Temp file-backed SQLite, no LLM, no model load.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base, Document, Folder, document_folders
from doc_assistant.db.session import session_scope


class _FakeController:
    def chunk_count(self) -> int:
        return 0


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed_doc(filename: str, *, archived: bool = False) -> str:
    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=f"/tmp/{filename}",
            doc_hash=f"hash-{filename}",
            format="pdf",
            is_archived=archived,
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _client() -> TestClient:
    return TestClient(create_app(controller=_FakeController()))  # type: ignore[arg-type]


def _row_count(table: object) -> int:
    with session_scope() as session:
        return int(session.execute(select(func.count()).select_from(table)).scalar() or 0)


# --- library.py CRUD -------------------------------------------------------------------------- #


def test_create_is_idempotent_case_insensitively(env: Path) -> None:
    """The DB can't enforce this — uq_folder_name_parent never fires for NULL parents (D4)."""
    from doc_assistant.library import create_folder, list_folders

    first = create_folder("Demo corpus")
    again = create_folder("  demo CORPUS  ")

    assert again.id == first.id
    assert again.name == "Demo corpus"  # the original name wins, not the re-spelling
    assert len(list_folders()) == 1


def test_create_rejects_a_blank_name(env: Path) -> None:
    from doc_assistant.library import create_folder

    with pytest.raises(ValueError, match="must not be blank"):
        create_folder("   ")


def test_folders_are_flat_in_v1(env: Path) -> None:
    """D1 — nothing sets a parent; the hierarchical column stays unused until F2 decides."""
    from doc_assistant.library import create_folder

    assert create_folder("Papers").parent_id is None


def test_rename_and_collision(env: Path) -> None:
    from doc_assistant.library import create_folder, rename_folder

    papers = create_folder("Papers")
    create_folder("Demo corpus")

    renamed = rename_folder(papers.id, "Reading list")
    assert renamed is not None
    assert renamed.name == "Reading list"

    with pytest.raises(ValueError, match="already exists"):
        rename_folder(papers.id, "demo corpus")

    # Renaming a folder to its own current name is not a collision with itself.
    same = rename_folder(papers.id, "Reading list")
    assert same is not None and same.name == "Reading list"


def test_unknown_folder_returns_none_or_false(env: Path) -> None:
    from doc_assistant.library import (
        add_documents_to_folder,
        delete_folder,
        folder_document_ids,
        get_folder,
        remove_documents_from_folder,
        rename_folder,
    )

    assert get_folder("nope") is None
    assert rename_folder("nope", "x") is None
    assert add_documents_to_folder("nope", ["d"]) is None
    assert remove_documents_from_folder("nope", ["d"]) is None
    assert folder_document_ids("nope") == []
    assert delete_folder("nope") is False


def test_membership_is_idempotent_and_skips_unknown_docs(env: Path) -> None:
    from doc_assistant.library import add_documents_to_folder, remove_documents_from_folder

    folder = _make_folder("Papers")
    doc_id = _seed_doc("a.pdf")

    first = add_documents_to_folder(folder, [doc_id, "does-not-exist"])
    assert first is not None and first.doc_count == 1

    again = add_documents_to_folder(folder, [doc_id])
    assert again is not None and again.doc_count == 1

    removed = remove_documents_from_folder(folder, [doc_id, "does-not-exist"])
    assert removed is not None and removed.doc_count == 0
    # Removing a non-member is a no-op, not an error.
    assert remove_documents_from_folder(folder, [doc_id]) is not None


def test_a_document_can_live_in_two_folders(env: Path) -> None:
    """ADR-025 fork 1 — membership is many-to-many, overlap allowed."""
    from doc_assistant.library import add_documents_to_folder, list_documents

    a, b = _make_folder("Demo corpus"), _make_folder("Papers")
    doc_id = _seed_doc("shared.pdf")
    add_documents_to_folder(a, [doc_id])
    add_documents_to_folder(b, [doc_id])

    summary = next(d for d in list_documents() if d.id == doc_id)
    assert sorted(summary.folders) == ["Demo corpus", "Papers"]
    assert sorted(summary.folder_ids) == sorted([a, b])


def test_doc_count_and_membership_exclude_archived(env: Path) -> None:
    """D5 — the count must agree with what the grid shows, or it lies."""
    from doc_assistant.library import add_documents_to_folder, folder_document_ids, get_folder

    folder = _make_folder("Papers")
    live = _seed_doc("live.pdf")
    hidden = _seed_doc("archived.pdf", archived=True)
    add_documents_to_folder(folder, [live, hidden])

    summary = get_folder(folder)
    assert summary is not None and summary.doc_count == 1
    assert folder_document_ids(folder) == [live]
    # The membership row survives, so un-archiving restores it.
    assert _row_count(document_folders) == 2


def test_list_documents_filters_by_folder_id(env: Path) -> None:
    """D2 — filtering is by id, because a root folder name is not a key."""
    from doc_assistant.library import add_documents_to_folder, list_documents

    folder = _make_folder("Papers")
    inside = _seed_doc("inside.pdf")
    _seed_doc("outside.pdf")
    add_documents_to_folder(folder, [inside])

    assert [d.id for d in list_documents(folder_id=folder)] == [inside]
    assert list_documents(folder_id="does-not-exist") == []
    assert len(list_documents()) == 2


def test_deleting_a_folder_never_deletes_documents(env: Path) -> None:
    """D6 — a folder is not a container of files; only its membership rows go."""
    from doc_assistant.library import add_documents_to_folder, delete_folder

    papers, other = _make_folder("Papers"), _make_folder("Demo corpus")
    doc_id = _seed_doc("a.pdf")
    add_documents_to_folder(papers, [doc_id])
    add_documents_to_folder(other, [doc_id])

    assert delete_folder(papers) is True

    assert _row_count(Document) == 1
    assert _row_count(Folder) == 1
    # Only the deleted folder's membership row cascaded; the other folder still has the doc.
    with session_scope() as session:
        rows = session.execute(select(document_folders.c.folder_id)).scalars().all()
    assert list(rows) == [other]


def _make_folder(name: str) -> str:
    from doc_assistant.library import create_folder

    return create_folder(name).id


# --- API routes ------------------------------------------------------------------------------- #


def test_routes_create_list_rename_delete(env: Path) -> None:
    client = _client()

    created = client.post("/api/library/folders", json={"name": "Demo corpus"})
    assert created.status_code == 200
    folder_id = created.json()["id"]
    assert created.json() == {
        "id": folder_id,
        "name": "Demo corpus",
        "description": None,
        "parent_id": None,
        "doc_count": 0,
    }

    listed = client.get("/api/library/folders")
    assert listed.status_code == 200
    assert [f["name"] for f in listed.json()] == ["Demo corpus"]

    renamed = client.patch(f"/api/library/folders/{folder_id}", json={"name": "Demo"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Demo"

    assert client.delete(f"/api/library/folders/{folder_id}").json() == {"ok": True}
    assert client.get("/api/library/folders").json() == []


def test_routes_membership(env: Path) -> None:
    client = _client()
    folder_id = client.post("/api/library/folders", json={"name": "Papers"}).json()["id"]
    doc_id = _seed_doc("a.pdf")

    added = client.post(
        f"/api/library/folders/{folder_id}/documents", json={"document_ids": [doc_id]}
    )
    assert added.status_code == 200
    assert added.json()["doc_count"] == 1

    doc = client.get("/api/library/documents").json()[0]
    assert doc["folders"] == ["Papers"] and doc["folder_ids"] == [folder_id]

    removed = client.delete(f"/api/library/folders/{folder_id}/documents/{doc_id}")
    assert removed.status_code == 200
    assert removed.json()["doc_count"] == 0


def test_routes_404_on_unknown_folder(env: Path) -> None:
    client = _client()
    assert client.patch("/api/library/folders/nope", json={"name": "x"}).status_code == 404
    assert client.delete("/api/library/folders/nope").status_code == 404
    assert (
        client.post(
            "/api/library/folders/nope/documents", json={"document_ids": ["d"]}
        ).status_code
        == 404
    )
    assert client.delete("/api/library/folders/nope/documents/d").status_code == 404


def test_routes_400_on_blank_and_collision(env: Path) -> None:
    client = _client()
    folder_id = client.post("/api/library/folders", json={"name": "Papers"}).json()["id"]
    client.post("/api/library/folders", json={"name": "Demo"})

    # Pydantic min_length rejects "" at 422; a whitespace-only name reaches library.py -> 400.
    assert client.post("/api/library/folders", json={"name": ""}).status_code == 422
    assert client.post("/api/library/folders", json={"name": "   "}).status_code == 400
    collision = client.patch(f"/api/library/folders/{folder_id}", json={"name": "demo"})
    assert collision.status_code == 400
    assert "already exists" in collision.json()["detail"]


def test_read_routes_write_nothing(env: Path) -> None:
    """D7 — the L4 write trap, narrowed: F1 *is* a write path, but the read routes are not."""
    client = _client()
    folder_id = client.post("/api/library/folders", json={"name": "Papers"}).json()["id"]
    doc_id = _seed_doc("a.pdf")
    client.post(f"/api/library/folders/{folder_id}/documents", json={"document_ids": [doc_id]})

    before = (_row_count(Folder), _row_count(document_folders), _row_count(Document))
    client.get("/api/library/folders")
    client.get("/api/library/documents")
    assert (_row_count(Folder), _row_count(document_folders), _row_count(Document)) == before
