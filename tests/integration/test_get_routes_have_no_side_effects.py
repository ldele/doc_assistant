"""No GET route writes or calls out (S-11's floor test, finding S15).

The launch token (S-6) guards every POST, PUT, PATCH and DELETE and leaves reads open, so a web
page can still make the window's engine run a GET (an image tag is enough), without reading the
answer. That is harmless only while a GET changes nothing and reaches nothing. Each GET route is
called once over a library where a write or a call-out would show:

- a database write: any INSERT, UPDATE, DELETE or REPLACE on the library's connection;
- a file written or removed under the data home (the library's database files aside);
- a network call: any outbound connection, refused here before it leaves.

The routes are read off the app, so a route added later is checked without editing this file.
Four routes have side effects today: the three S15 named, and one this test found. They are
listed below with what each does, and the list can only shrink: a listed route that no longer
writes or calls out fails too, so the line is removed when it is fixed.
"""

from __future__ import annotations

import re
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import event

#: The GET routes known to write or call out, and what they do. Fixing one removes its line.
KNOWN: dict[str, str] = {
    "/api/sources": "commits the registry rows its walk of the source folder finds (S15)",
    "/api/updates": (
        "checks GitHub and records the answer in settings when the person opted in and a "
        "check is due (S15)"
    ),
    "/api/setup": "probes the configured Ollama host to say whether it answers (S15)",
    "/api/ingest/plan": (
        "commits the registry rows its walk of the source folder finds, as /api/sources does "
        "(found by this test, not in S15)"
    ),
}

#: Routes that stream rather than answer, walked by their own tests.
_STREAMS = {"/api/chat/stream"}


class _EmptyStore:
    def get(self, **_: Any) -> dict[str, Any]:
        return {"ids": [], "documents": [], "metadatas": []}

    def count(self) -> int:
        return 0


class _Rag:
    def __init__(self) -> None:
        self.db = _EmptyStore()
        self.pc_db = _EmptyStore()


class _Engine:
    def __init__(self) -> None:
        self.rag = _Rag()

    def chunk_count(self) -> int:
        return 0

    def corpus_stats(self) -> Any:
        from doc_assistant.corpus_stats import corpus_stats

        return corpus_stats(documents=0, chunks=0, keyword_index_on_disk=False)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


@dataclass
class _Seen:
    writes: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    recording: bool = False


_WRITE = re.compile(r"^\s*(INSERT|UPDATE|DELETE|REPLACE)\b", re.IGNORECASE)
_DB_FILES = re.compile(r"\.db(-journal|-wal|-shm)?$")


def _files(home: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(home)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in home.rglob("*")
        if p.is_file() and not _DB_FILES.search(p.name)
    }


@pytest.fixture
def engine_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant import app_settings, credentials
    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    home = tmp_path / "home"
    sources = home / "sources"
    sources.mkdir(parents=True)
    (sources / "paper.md").write_text("# A paper\n\nText.\n", encoding="utf-8")
    monkeypatch.setenv("DOC_SOURCE_DIR", str(sources))
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", home / "settings.json")
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", home / "credentials.json")
    app_settings.set_update_check_enabled(True)  # opted in, never checked: a check is due

    engine = create_engine(f"sqlite:///{home / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(
        session_mod,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    seen = _Seen()

    @event.listens_for(engine, "before_cursor_execute")
    def _on_sql(_conn: Any, _cur: Any, statement: str, *_a: Any) -> None:
        if seen.recording and _WRITE.match(statement):
            seen.writes.append(statement.split("(")[0].strip()[:60])

    def _no_call_out(address: Any, *_a: Any, **_k: Any) -> Any:
        if seen.recording:
            seen.calls.append(f"{address[0]}:{address[1]}")
        raise OSError("network calls are refused in this test")

    monkeypatch.setattr(socket, "create_connection", _no_call_out)
    app = create_app(controller=_Engine())  # type: ignore[arg-type]
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, home, seen
    engine.dispose()


def _get_paths() -> list[str]:
    app = create_app(controller=_Engine())  # type: ignore[arg-type]
    return sorted(
        {
            r.path
            for r in app.routes
            if isinstance(r, APIRoute) and "GET" in r.methods and r.path not in _STREAMS
        }
    )


def _side_effects(engine_home: Any, path: str) -> list[str]:
    client, home, seen = engine_home
    before = _files(home)
    seen.recording = True
    client.get(re.sub(r"\{[^}]+\}", "no-such-id", path))
    seen.recording = False
    after = _files(home)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    return (
        [f"database: {w}" for w in seen.writes]
        + [f"file: {c}" for c in changed]
        + [f"network: {c}" for c in seen.calls]
    )


@pytest.mark.parametrize("path", _get_paths())
def test_a_get_route_writes_nothing_and_calls_nothing(engine_home: Any, path: str) -> None:
    effects = _side_effects(engine_home, path)
    if path in KNOWN:
        assert effects, f"{path} no longer writes or calls out: remove it from KNOWN"
    else:
        assert effects == [], f"GET {path} has side effects: {effects}"


def test_every_known_route_still_exists() -> None:
    assert set(KNOWN) <= set(_get_paths()), "a listed route is gone: remove its line"
