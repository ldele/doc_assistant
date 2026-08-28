"""`POST /api/documents/inspect` — the review sheet's data source over the wire (AD2).

The endpoint's defining property is what it does *not* do: spec constraint 2 says nothing is
copied, registered or indexed before the review sheet has been shown and confirmed, and inspect
being a separate, read-only call from apply (AD3) is how that stays structural. The last test here
asserts it against the registry rather than trusting the docstring.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker


class FakeChroma:
    """The two calls `purge_document_record` makes, and a record of what was deleted.

    Same shape as `tests/integration/test_document_delete.py`'s: `undo-add` reaches the index now
    (KI-51 part 1 — undo removes the document the add produced, not just the registry row), so the
    controller fake has to carry a `.rag.db` the way the real one does.
    """

    def __init__(self, ids_by_hash: dict[str, list[str]] | None = None) -> None:
        self._ids = ids_by_hash or {}
        self.deleted: list[str] = []

    def get(self, *, where: dict[str, Any], include: list[str]) -> dict[str, Any]:
        return {"ids": list(self._ids.get(where["doc_hash"], []))}

    def delete(self, *, ids: list[str]) -> None:
        self.deleted.extend(ids)


class _FakeRag:
    def __init__(self, db: FakeChroma) -> None:
        self.db = db


class FakeController:
    """The minimum surface `create_app` needs; inspect touches none of it."""

    def __init__(self) -> None:
        self.chunk_count = 0
        self.rag = _FakeRag(FakeChroma())

    def corpus_stats(self) -> dict[str, int]:
        return {"chunk_count": 0}


@pytest.fixture
def temp_database(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    from doc_assistant.db.models import Base

    Base.metadata.create_all(engine)
    # AD3b: `SourceFile.root_id` carries a literal DEFAULT pointing at the library root, and the
    # FK is enforced — so the row has to exist before anything inserts. `init_db` guarantees this
    # in production (`_seed_library_root`); this fixture stands in for it.
    from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceRoot

    with sessionmaker(bind=engine, future=True)() as s:
        s.add(SourceRoot(id=LIBRARY_ROOT_ID, path=str(Path(path).parent), kind="library"))
        s.commit()
    yield path
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


@pytest.fixture
def client(temp_database, tmp_path, monkeypatch):
    """A client whose source dir is a temp folder, so no test reads the real library."""
    monkeypatch.setenv("DOC_SOURCE_DIR", str(tmp_path))
    return TestClient(create_app(controller=FakeController()))


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_a_supported_file_comes_back_ready_to_add(client, tmp_path):
    f = _write(tmp_path / "paper.pdf", b"%PDF-1.4 body")
    r = client.post("/api/documents/inspect", json={"paths": [str(f)]})
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"total": 1, "add": 1}
    (row,) = body["files"]
    assert row["name"] == "paper.pdf"
    assert row["verdict"] == "add"
    assert row["selected_by_default"] is True


def test_an_unsupported_file_carries_its_advisory_over_the_wire(client, tmp_path):
    f = _write(tmp_path / "manuscript.doc", b"junk")
    (row,) = client.post("/api/documents/inspect", json={"paths": [str(f)]}).json()["files"]
    assert row["verdict"] == "unsupported"
    assert row["advisory"] == "DOC format is not supported. Convert to DOCX or PDF first."
    assert row["selected_by_default"] is False


def test_exceptions_sort_above_clean_files(client, tmp_path):
    """The sheet paginates, so page one must carry everything worth seeing (grill branch 7)."""
    paths = [
        str(_write(tmp_path / "ok-a.pdf", b"a")),
        str(_write(tmp_path / "bad.doc", b"b")),
        str(_write(tmp_path / "ok-b.pdf", b"c")),
    ]
    files = client.post("/api/documents/inspect", json={"paths": paths}).json()["files"]
    assert [f["verdict"] for f in files] == ["unsupported", "add", "add"]


def test_a_dropped_folder_is_expanded_server_side(client, tmp_path):
    """The client never walks a folder — the recursion rule lives with `scan_sources`."""
    drop = tmp_path / "drop"
    _write(drop / "one.pdf", b"1")
    _write(drop / "nested" / "two.pdf", b"2")

    body = client.post("/api/documents/inspect", json={"paths": [str(drop)]}).json()
    assert body["counts"]["total"] == 2
    assert {f["name"] for f in body["files"]} == {"one.pdf", "two.pdf"}


def test_a_vanished_path_is_reported_rather_than_400ing_the_batch(client, tmp_path):
    """One bad path must not cost the user the other eleven — inform, don't block."""
    good = str(_write(tmp_path / "good.pdf", b"g"))
    gone = str(tmp_path / "not-here.pdf")
    body = client.post("/api/documents/inspect", json={"paths": [good, gone]}).json()
    assert body["counts"]["total"] == 2
    verdicts = {f["name"]: f["verdict"] for f in body["files"]}
    assert verdicts == {"good.pdf": "add", "not-here.pdf": "unreadable"}


def test_an_empty_drop_is_an_empty_answer_not_an_error(client):
    """A drag that carried nothing addable is not a failure (0-document robustness)."""
    r = client.post("/api/documents/inspect", json={"paths": []})
    assert r.status_code == 200
    assert r.json() == {"files": [], "counts": {"total": 0}}


def test_a_duplicate_is_named_against_the_file_it_matches(client, tmp_path):
    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import sha256_file

    body_bytes = b"%PDF-1.4 " + b"q" * 400
    registered = _write(tmp_path / "cajal-1899.pdf", body_bytes)
    with session_scope() as session:
        session.add(
            SourceFile(
                rel_path="cajal-1899.pdf",
                format="pdf",
                size=len(body_bytes),
                mtime=0.0,
                source_sha256=sha256_file(registered),
            )
        )

    copy = _write(tmp_path / "downloads" / "same-paper.pdf", body_bytes)
    (row,) = client.post("/api/documents/inspect", json={"paths": [str(copy)]}).json()["files"]
    assert row["verdict"] == "duplicate"
    assert row["duplicate_of"] == "library:cajal-1899.pdf", "a key, not a bare rel_path (AD3b)"
    assert row["selected_by_default"] is False


def test_inspecting_registers_nothing(client, tmp_path):
    """Spec constraint 2, asserted at the boundary that actually matters to a user."""
    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope

    def rows() -> int:
        with session_scope() as session:
            return int(session.execute(select(func.count()).select_from(SourceFile)).scalar_one())

    before = rows()
    paths = [
        str(_write(tmp_path / "a.pdf", b"a")),
        str(_write(tmp_path / "b.epub", b"b")),
        str(_write(tmp_path / "c.doc", b"c")),
    ]
    assert client.post("/api/documents/inspect", json={"paths": paths}).status_code == 200
    assert rows() == before


# ============================================================
# AD3 — apply / undo over the wire
# ============================================================


def test_add_copies_registers_and_leaves_the_original(client, tmp_path):
    src = _write(tmp_path / "inbox" / "paper.pdf", b"%PDF-1.4 body")
    r = client.post("/api/documents/add", json={"paths": [str(src)], "mode": "copy"})
    assert r.status_code == 200
    body = r.json()
    assert body["stopped_early"] is False
    assert [o["rel_path"] for o in body["added"]] == ["paper.pdf"]
    assert (tmp_path / "paper.pdf").exists()
    assert src.exists(), "the user's original is never moved"


def test_add_reports_the_failure_and_what_it_never_tried(client, tmp_path):
    good = str(_write(tmp_path / "in" / "a.pdf", b"a"))
    gone = str(tmp_path / "in" / "missing.pdf")
    later = str(_write(tmp_path / "in" / "c.pdf", b"c"))

    body = client.post("/api/documents/add", json={"paths": [good, gone, later]}).json()
    assert body["stopped_early"] is True
    assert [o["rel_path"] for o in body["added"]] == ["a.pdf"]
    assert body["failed"]["name"] == "missing.pdf"
    assert body["not_attempted"] == [later]


def test_undo_removes_what_add_created(client, tmp_path):
    srcs = [str(_write(tmp_path / "in" / f"{i}.pdf", bytes([i]) * 30)) for i in range(3)]
    added = client.post("/api/documents/add", json={"paths": srcs}).json()["added"]

    r = client.post("/api/documents/undo-add", json={"rel_paths": [o["rel_path"] for o in added]})
    assert r.status_code == 200
    assert r.json() == {"undone": 3}
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == []


def test_undo_reaches_the_index_over_the_wire(temp_database, tmp_path, monkeypatch):
    """KI-51 part 1, at the boundary: the route must hand its Chroma handle to `undo_add`.

    The unit tests cover the removal and its guards; this covers the *wiring*, which is exactly
    what a route reaching for `controller.rag.db` can get wrong — and did, the first time it was
    written against a fake controller that had no `.rag`.
    """
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    monkeypatch.setenv("DOC_SOURCE_DIR", str(tmp_path))
    chroma = FakeChroma({"h-wire": ["c1", "c2"]})
    controller = FakeController()
    controller.rag = _FakeRag(chroma)
    c = TestClient(create_app(controller=controller))

    src = _write(tmp_path / "in" / "paper.pdf", b"%PDF-1.4 wire")
    key = c.post("/api/documents/add", json={"paths": [str(src)]}).json()["added"][0]["key"]

    # Stand in for the ingest the user would have run: a Document row pointing at the copy.
    with session_scope() as session:
        session.add(
            Document(
                id="doc-wire",
                filename="paper.pdf",
                source_original=str(tmp_path / "paper.pdf"),
                source_cache=None,
                doc_hash="h-wire",
                format="pdf",
                extractor_used="pymupdf",
                extraction_health="ok",
                chunk_count=2,
                page_count=1,
            )
        )

    assert c.post("/api/documents/undo-add", json={"rel_paths": [key]}).json() == {"undone": 1}
    with session_scope() as session:
        assert session.get(Document, "doc-wire") is None, "the document must go with the row"
    assert chroma.deleted == ["c1", "c2"], "and its chunks, or it stays retrievable"


def test_reference_mode_registers_over_the_wire_without_copying(client, tmp_path):
    """AD3b: ADR-046's second placement mode, end to end. Nothing is copied anywhere."""
    # Deliberately OUTSIDE the library root (which this fixture sets to tmp_path): a file that
    # already lives inside the library is a different case, registered under the library root —
    # see test_a_file_already_inside_the_library_references_the_library_root.
    with tempfile.TemporaryDirectory() as elsewhere:
        src = _write(Path(elsewhere) / "a.pdf", b"a")
        r = client.post("/api/documents/add", json={"paths": [str(src)], "mode": "reference"})
        assert r.status_code == 200
        body = r.json()
        assert body["failed"] is None and len(body["added"]) == 1
        assert body["added"][0]["key"], "the client needs a key to undo with"
        assert src.exists(), "the original stays where the user keeps it"
        # No copy may have landed in the library root.
        assert list(tmp_path.glob("*.pdf")) == []

        listed = client.get("/api/sources").json()
        referenced = [f for f in listed if f["root_kind"] == "referenced"]
        assert len(referenced) == 1, "the referenced file shows up in the registry listing"
        assert referenced[0]["rel_path"] == "a.pdf"
        assert referenced[0]["key"] == f"{referenced[0]['root_id']}:a.pdf"


def test_an_unknown_mode_is_a_422(client, tmp_path):
    src = str(_write(tmp_path / "a.pdf", b"a"))
    r = client.post("/api/documents/add", json={"paths": [src], "mode": "teleport"})
    assert r.status_code == 422


def test_add_is_refused_while_an_ingest_is_running(client, tmp_path):
    """Mirrors /api/ingest's own 409: adding mid-scan would race the registry."""
    src = str(_write(tmp_path / "a.pdf", b"a"))
    app = client.app
    with app.state.ingest_lock:
        app.state.ingest_status.state = "running"
    try:
        assert client.post("/api/documents/add", json={"paths": [src]}).status_code == 409
        assert client.post("/api/documents/undo-add", json={"rel_paths": []}).status_code == 409
    finally:
        with app.state.ingest_lock:
            app.state.ingest_status.state = "idle"
