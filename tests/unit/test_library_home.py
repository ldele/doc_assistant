"""`library_home.inspect_library`: the read-only check before the app opens a library (ADR-061).

Libraries are built in temp folders: a `library.db` from the real schema and a vector-store
SQLite with the three tables the count reads. Where the app keeps a folder's stores depends on the
machine (`config.chroma_base_for`: relocated on Windows when the path is not ASCII, as the temp
folder is on an accented account), so each test pins it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from doc_assistant import config, library_home
from doc_assistant.db.models import Base, Document
from doc_assistant.embeddings import get_active_model_name, get_collection_name


@pytest.fixture(autouse=True)
def stores_inside(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stores live inside each library folder unless a test says otherwise."""
    monkeypatch.setattr(config, "chroma_base_for", lambda data_path: data_path)


def _make_library(folder: Path, docs: list[tuple[str, str, str]], chunks: int) -> Path:
    """A library at ``folder``: rows ``(filename, source_original, source_cache)`` and ``chunks``
    embeddings in the searched store, under this app's collection name."""
    folder.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{folder / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for i, (filename, original, cache) in enumerate(docs):
            session.add(
                Document(
                    filename=filename,
                    source_original=original,
                    source_cache=cache,
                    doc_hash=f"h{i}",
                    format="md",
                )
            )
        session.commit()
    engine.dispose()
    store = folder / library_home.SEARCHED_STORE
    store.mkdir(exist_ok=True)
    with sqlite3.connect(store / "chroma.sqlite3") as db:
        db.execute("CREATE TABLE collections (id TEXT, name TEXT)")
        db.execute("CREATE TABLE segments (id TEXT, collection TEXT)")
        db.execute("CREATE TABLE embeddings (id INTEGER, segment_id TEXT)")
        db.execute(
            "INSERT INTO collections VALUES ('c1', ?)",
            (get_collection_name(get_active_model_name()),),
        )
        db.execute("INSERT INTO segments VALUES ('s1', 'c1')")
        db.executemany("INSERT INTO embeddings VALUES (?, 's1')", [(i,) for i in range(chunks)])
    return folder


def _made_here(tmp_path: Path, chunks: int = 7) -> Path:
    folder = tmp_path / "lib"
    docs = []
    for name in ("a.md", "b.md"):
        source = folder / "sources" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("text", encoding="utf-8")
        docs.append((name, str(source), str(folder / "cache" / name)))
    return _make_library(folder, docs, chunks)


def test_a_library_made_in_place_may_open_with_nothing_to_say(tmp_path: Path) -> None:
    found = library_home.inspect_library(_made_here(tmp_path))
    assert found.is_library and found.can_open
    assert (found.documents, found.chunks, found.sources_found) == (2, 7, 2)
    assert found.cache_named_folder is None
    assert found.problems == [] and found.notes == []


def test_a_copy_says_its_cache_names_another_folder(tmp_path: Path) -> None:
    original = tmp_path / "original"
    copy = tmp_path / "copy"
    docs = [
        (n, str(original / "sources" / n), str(original / "cache" / n)) for n in ("a.md", "b.md")
    ]
    _make_library(copy, docs, chunks=7)
    (copy / "sources").mkdir()
    (copy / "sources" / "a.md").write_text("its own copy", encoding="utf-8")

    found = library_home.inspect_library(copy)

    assert found.can_open  # safe: nothing outside it will be written (ADR-061 choice 4)
    assert found.sources_found == 1
    assert found.cache_named_folder == str(original)
    assert any("copied or moved" in n for n in found.notes)
    assert any(n.startswith("1 of 2 source files") for n in found.notes)


def test_a_folder_that_is_not_a_library_cannot_open(tmp_path: Path) -> None:
    assert not library_home.inspect_library(tmp_path).can_open
    assert "holds no library.db" in library_home.inspect_library(tmp_path).problems[0]
    missing = library_home.inspect_library(tmp_path / "nowhere")
    assert not missing.can_open and "does not exist" in missing.problems[0]


def test_the_library_already_open_is_not_opened_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _made_here(tmp_path)
    monkeypatch.setattr(config, "DATA_PATH", folder)
    found = library_home.inspect_library(folder)
    assert found.is_current and not found.can_open


def test_no_chunks_for_this_model_is_said_not_blocked(tmp_path: Path) -> None:
    found = library_home.inspect_library(_made_here(tmp_path, chunks=0))
    assert found.can_open
    assert any("no chunks for this app's embedding model" in n for n in found.notes)


def test_stores_stranded_in_an_accented_folder_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-061 choice 5: the app would read the relocated place, which is empty."""
    folder = _made_here(tmp_path)
    monkeypatch.setattr(config, "chroma_base_for", lambda data_path: tmp_path / "relocated")
    found = library_home.inspect_library(folder)
    assert not found.can_open
    assert "outside plain ASCII" in found.problems[0]


def test_stores_already_relocated_count_where_the_app_reads_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A library the installed app made at an accented path keeps its stores under %PROGRAMDATA%,
    and opens fine."""
    folder = _made_here(tmp_path, chunks=0)
    relocated = _make_library(tmp_path / "relocated", [], chunks=5)
    monkeypatch.setattr(config, "chroma_base_for", lambda data_path: relocated)
    found = library_home.inspect_library(folder)
    assert found.can_open and found.chunks == 5


def test_the_check_writes_nothing(tmp_path: Path) -> None:
    folder = _made_here(tmp_path)

    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in folder.rglob("*")
            if p.is_file()
        }

    before = snapshot()
    library_home.inspect_library(folder)
    assert snapshot() == before  # no file changed, none added (no journal, no lock file)


class _Controller:
    def chunk_count(self) -> int:
        return 0


def test_the_route_reports_the_check(tmp_path: Path) -> None:
    client = TestClient(create_app(controller=_Controller()))  # type: ignore[arg-type]
    body = client.post("/api/libraries/inspect", json={"path": str(_made_here(tmp_path))}).json()
    assert body["can_open"] is True and body["documents"] == 2 and body["chunks"] == 7
    refused = client.post("/api/libraries/inspect", json={"path": str(tmp_path / "x")}).json()
    assert refused["can_open"] is False and refused["problems"]
