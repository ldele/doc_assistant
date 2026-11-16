"""Every GET route on an empty library: no 500, and no count divided by zero (the robustness
contract's 0-document half; the plan's A3 entry).

The routes are read off the app itself, so a route added later is walked without editing this
file. A path parameter gets an id nothing has; a 404 is an honest answer to it. What is refused
is a server error, and a ratio in a body (``NaN``, ``Infinity``) that only an empty library would
produce. The suite's own data home is empty (`tests/conftest.py`); the database here is too.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient


class _EmptyStore:
    """A vector store holding nothing."""

    def get(self, **_: Any) -> dict[str, Any]:
        return {"ids": [], "documents": [], "metadatas": []}

    def count(self) -> int:
        return 0


class _Rag:
    def __init__(self) -> None:
        self.db = _EmptyStore()
        self.pc_db = _EmptyStore()


class _EmptyController:
    """The engine as the GET routes read it, over a library with nothing in it."""

    def __init__(self) -> None:
        self.rag = _Rag()

    def chunk_count(self) -> int:
        return 0

    def corpus_stats(self) -> Any:  # the real assembly, over nothing
        from doc_assistant.corpus_stats import corpus_stats

        return corpus_stats(documents=0, chunks=0, keyword_index_on_disk=False)

    def __getattr__(self, name: str) -> Any:  # anything else a route reads: nothing there
        raise AttributeError(name)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}", future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(
        session_mod,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    app = create_app(controller=_EmptyController())  # type: ignore[arg-type]
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    engine.dispose()


#: Routes that stream or wait rather than answer, walked by their own tests.
_NOT_A_READ = {"/api/chat/stream", "/api/updates/check"}


def _get_paths() -> list[str]:
    app = create_app(controller=_EmptyController())  # type: ignore[arg-type]
    paths: list[str] = []
    for route in app.routes:
        if (
            isinstance(route, APIRoute)
            and "GET" in route.methods
            and route.path not in _NOT_A_READ
        ):
            paths.append(route.path)
    return sorted(set(paths))


def _no_bad_number(value: Any, where: str) -> None:
    if isinstance(value, float):
        assert math.isfinite(value), f"{where}: {value}"
    elif isinstance(value, dict):
        for k, v in value.items():
            _no_bad_number(v, f"{where}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _no_bad_number(v, f"{where}[{i}]")


@pytest.mark.parametrize("path", _get_paths())
def test_a_get_route_answers_an_empty_library(client: TestClient, path: str) -> None:
    url = re.sub(r"\{[^}]+\}", "no-such-id", path)
    r = client.get(url)
    assert r.status_code < 500, f"{url}: {r.status_code} {r.text[:300]}"
    if r.headers.get("content-type", "").startswith("application/json") and r.content:
        assert "NaN" not in r.text and "Infinity" not in r.text, url
        _no_bad_number(r.json(), url)
