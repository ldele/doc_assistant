"""The chunk-embedding loader skips vectors whose document is gone (2026-09-18).

A chunk left in the vector store by a removed document still carries its ``document_id``. The
loader grouped it like any other, so ``compute_doc_vectors --apply`` wrote an edge for a document
the library no longer has, the ``doc_similarities`` foreign key failed, and the whole edge set
rolled back — one stale chunk from a deleted test document blocked similarity for all 103
documents. The loader now drops such chunks and says so.

Fake vector store, temp SQLite; no model, no network.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
import doc_assistant.doc_vectors as dv
from doc_assistant.db.models import Base, Document
from doc_assistant.db.session import session_scope


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


class _FakeClient:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_collection(self, name: str) -> object:
        return object()


def test_a_chunk_whose_document_is_gone_is_skipped(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with session_scope() as s:
        s.add(
            Document(
                id="live", filename="a.pdf", source_original="a.pdf", doc_hash="h1", format="pdf"
            )
        )
    store: dict[str, Any] = {
        "embeddings": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        "metadatas": [
            {"document_id": "live"},
            {"document_id": "removed-test-document"},  # left behind by a deleted document
            {"doc_hash": "h1"},  # an old chunk that only carries its hash
        ],
    }
    monkeypatch.setattr(chromadb, "PersistentClient", _FakeClient)
    monkeypatch.setattr(dv, "get_all", lambda coll, include: store)

    grouped = dv.load_chunk_embeddings_by_document()

    assert set(grouped) == {"live"}
    assert len(grouped["live"]) == 2
    assert all(isinstance(v, np.ndarray) for v in grouped["live"])
