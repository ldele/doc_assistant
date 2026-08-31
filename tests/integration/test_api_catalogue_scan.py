"""`POST /api/catalogue/zotero/scan` — the route that turns a reference manager into paths.

The defining property, and the one asserted here, is where it **stops**. It reads the catalogue,
records what it says, and hands back absolute paths. It does not add, does not copy, does not
register a root and does not index — because the review sheet, the duplicate rule and the
copy-or-reference choice already exist for dropped files and an import must reach the user through
the same ones (ADR-049).

The other thing worth pinning: a missing library is a **404 carrying a sentence**, not a 500. Not
having Zotero installed is an ordinary state of the world.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker

from tests.unit.adapters.test_zotero import _Library


class _FakeChroma:
    def get(self, *, where: dict[str, Any], include: list[str]) -> dict[str, Any]:
        return {"ids": []}

    def delete(self, *, ids: list[str]) -> None:
        return None


class _FakeRag:
    def __init__(self) -> None:
        self.db = _FakeChroma()


class FakeController:
    def __init__(self) -> None:
        self.chunk_count = 0
        self.rag = _FakeRag()

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
    yield path
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


@pytest.fixture
def client(temp_database, tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_SOURCE_DIR", str(tmp_path / "library"))
    return TestClient(create_app(controller=FakeController()))


@pytest.fixture
def zotero_dir(tmp_path: Path) -> Path:
    lib = _Library(tmp_path / "Zotero")
    paper = lib.item("PAPER001", title="A Curated Title", date="2020", DOI="10.1000/x")
    lib.creator(paper, "Ada", "Lovelace")
    lib.attachment("ATTACH01", parent=paper, filename="paper.pdf")
    untitled = lib.item("PAPER002")
    lib.attachment("ATTACH02", parent=untitled, filename="untitled.pdf")
    binned = lib.item("PAPER003", title="Binned")
    lib.attachment("ATTACH03", parent=binned, filename="binned.pdf")
    lib.trash(binned)
    lib.close()
    return lib.root


def test_a_scan_returns_paths_and_says_what_it_left_out(client, zotero_dir: Path) -> None:
    r = client.post("/api/catalogue/zotero/scan", json={"data_dir": str(zotero_dir)})
    assert r.status_code == 200
    body = r.json()

    assert body["label"] == "Zotero"
    assert body["found"] == 2
    assert {Path(p).name for p in body["paths"]} == {"paper.pdf", "untitled.pdf"}
    assert body["skipped"] == {"in the Zotero trash": 1}
    # Only one of the two has a title, and saying so is the reason to import rather than browse.
    assert body["with_metadata"] == 1
    assert Path(body["root"]) == zotero_dir / "storage"


def test_a_scan_records_the_metadata_it_read(client, zotero_dir: Path) -> None:
    """Recorded now because this is when the catalogue is open; applied at the next ingest."""
    from doc_assistant.db.models import ExternalMetadata
    from doc_assistant.db.session import session_scope

    client.post("/api/catalogue/zotero/scan", json={"data_dir": str(zotero_dir)})

    with session_scope() as session:
        rows = session.execute(select(ExternalMetadata)).scalars().all()
        titles = {r.title for r in rows}
        assert len(rows) == 2
        assert "A Curated Title" in titles
        assert all(r.source == "zotero" for r in rows)


def test_a_scan_stages_nothing_and_registers_nothing(client, zotero_dir: Path) -> None:
    """Spec constraint 2, restated for imports: nothing exists until the sheet is confirmed."""
    from doc_assistant.db.models import Document, SourceFile, SourceRoot
    from doc_assistant.db.session import session_scope

    client.post("/api/catalogue/zotero/scan", json={"data_dir": str(zotero_dir)})

    with session_scope() as session:
        for table in (SourceFile, SourceRoot, Document):
            count = session.execute(select(func.count()).select_from(table)).scalar_one()
            assert count == 0, f"{table.__name__} rows appeared during a scan"


def test_scanning_twice_does_not_duplicate_the_records(client, zotero_dir: Path) -> None:
    from doc_assistant.db.models import ExternalMetadata
    from doc_assistant.db.session import session_scope

    client.post("/api/catalogue/zotero/scan", json={"data_dir": str(zotero_dir)})
    client.post("/api/catalogue/zotero/scan", json={"data_dir": str(zotero_dir)})

    with session_scope() as session:
        count = session.execute(select(func.count()).select_from(ExternalMetadata)).scalar_one()
    assert count == 2


def test_no_library_there_is_a_404_with_a_sentence(client, tmp_path: Path) -> None:
    r = client.post("/api/catalogue/zotero/scan", json={"data_dir": str(tmp_path / "nope")})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "zotero.sqlite" in detail and " " in detail, "the detail must read as a sentence"


def test_a_folder_holding_something_else_is_a_404_not_a_500(client, tmp_path: Path) -> None:
    root = tmp_path / "Zotero"
    root.mkdir()
    connection = sqlite3.connect(root / "zotero.sqlite")
    connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    r = client.post("/api/catalogue/zotero/scan", json={"data_dir": str(root)})
    assert r.status_code == 404


def test_an_empty_body_looks_where_zotero_puts_it(client, tmp_path, monkeypatch) -> None:
    """The default is the whole point: the common case should not need a folder picker."""
    from doc_assistant.adapters import zotero

    home = tmp_path / "home"
    lib = _Library(home / "Zotero")
    item = lib.item("PAPER001", title="Found by default")
    lib.attachment("ATTACH01", parent=item, filename="paper.pdf")
    lib.close()
    monkeypatch.setattr(zotero.Path, "home", staticmethod(lambda: home))

    r = client.post("/api/catalogue/zotero/scan", json={})
    assert r.status_code == 200
    assert r.json()["found"] == 1
