"""The orphan sweep must not delete a document that merely re-extracted (ADR-047).

This is the regression test for a real data loss. `_existing_document_id`'s source-path fallback
was in place and verified in isolation, but `main()` runs the orphan cleanup *first*, and that
pass was hash-keyed too: it classified every re-extracted document as `stale` and deleted its row,
FK-cascading the sidecars, before the fallback was ever consulted.

Measured on a 97-document library the day it happened: **767 of 881 figure rows destroyed**, along
with 1,170 keywords and 381 epistemics rows, while the run reported `exit=0`.

The distinction that fixes it is the one these tests pin: a source file that is **gone** ends its
document; a source file that is **still there** and merely extracts differently does not.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def temp_database(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _record):
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


class _FakeChroma:
    """Stands in for the metadata read; the sweep only ever asks for `metadatas`."""

    def __init__(self, metadatas):
        self._metadatas = metadatas
        self.deleted: list[dict] = []

    def get(self, *_a, **_k):
        return {"metadatas": self._metadatas, "ids": [str(i) for i in range(len(self._metadatas))]}

    def delete(self, where=None, **_k):
        self.deleted.append(where or {})


def _seed(doc_id: str, doc_hash: str, source: str) -> None:
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        s.add(
            Document(
                id=doc_id,
                filename=os.path.basename(source),
                source_original=source,
                source_cache=None,
                doc_hash=doc_hash,
                format="pdf",
            )
        )


def test_a_re_extracted_document_keeps_its_row(temp_database, tmp_path, monkeypatch):
    """The exact failure. Source present, text changed -> `stale` -> must NOT be deleted."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import cleanup

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"%PDF-1.4")
    _seed("doc-1", "hash-OLD", str(src))

    monkeypatch.setattr(cleanup, "load_or_extract", lambda _p: "the NEW extracted text")
    monkeypatch.setattr(cleanup, "doc_hash", lambda _t: "hash-NEW")

    db = _FakeChroma([{"doc_hash": "hash-OLD", "source_original": str(src)}])
    result = cleanup.cleanup_orphans_sqlite(db)

    assert result.stale == ["hash-OLD"], "a changed extraction is stale"
    assert result.gone == [], "the file is still on disk"
    with session_scope() as s:
        assert s.execute(select(Document)).scalars().all(), "the row must survive"


def test_a_vanished_document_is_still_removed(temp_database, tmp_path, monkeypatch):
    """The fix must not neuter the sweep — a deleted file still ends its document."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import cleanup

    missing = tmp_path / "deleted.pdf"  # never created
    _seed("doc-2", "hash-GONE", str(missing))

    db = _FakeChroma([{"doc_hash": "hash-GONE", "source_original": str(missing)}])
    result = cleanup.cleanup_orphans_sqlite(db)

    assert result.gone == ["hash-GONE"]
    assert result.stale == []
    with session_scope() as s:
        assert not s.execute(select(Document)).scalars().all(), "the row must go"


def test_both_kinds_of_dead_hash_lose_their_vectors(temp_database, tmp_path, monkeypatch):
    """Chunks are keyed by hash, so neither kind still describes the file."""
    from doc_assistant.ingest import cleanup

    src = tmp_path / "here.pdf"
    src.write_bytes(b"%PDF-1.4")
    gone = tmp_path / "vanished.pdf"
    _seed("doc-3", "h-stale", str(src))
    _seed("doc-4", "h-gone", str(gone))

    monkeypatch.setattr(cleanup, "load_or_extract", lambda _p: "new text")
    monkeypatch.setattr(cleanup, "doc_hash", lambda _t: "h-new")

    db = _FakeChroma(
        [
            {"doc_hash": "h-stale", "source_original": str(src)},
            {"doc_hash": "h-gone", "source_original": str(gone)},
        ]
    )
    result = cleanup.cleanup_orphans_sqlite(db)
    assert set(result.dead_chunk_hashes) == {"h-stale", "h-gone"}


def test_figures_are_swept_for_gone_only(temp_database, tmp_path, monkeypatch):
    """A figure is a crop of the PDF's page — changing the TEXT extractor cannot invalidate it.

    Deleting stale figure dirs is what turned a re-extraction into the loss of 767 figure rows.
    """
    from doc_assistant.ingest import cleanup, figures

    monkeypatch.setattr(figures, "figure_dir", lambda h: tmp_path / "figs" / h)
    monkeypatch.setattr(cleanup, "figure_dir", lambda h: tmp_path / "figs" / h)
    for h in ("h-stale", "h-gone"):
        d = tmp_path / "figs" / h
        d.mkdir(parents=True)
        (d / "page1_fig0.png").write_bytes(b"png")

    cleanup.cleanup_orphan_figures(["h-gone"])  # gone only, as `main` now passes

    assert not (tmp_path / "figs" / "h-gone").exists(), "a vanished document's crops go"
    assert (tmp_path / "figs" / "h-stale").exists(), "a re-extracted document's crops stay"


def test_a_transient_extract_failure_deletes_nothing(temp_database, tmp_path, monkeypatch):
    """Pre-existing guarantee, re-asserted: a read error must never remove live data."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import cleanup

    src = tmp_path / "locked.pdf"
    src.write_bytes(b"%PDF-1.4")
    _seed("doc-5", "h-1", str(src))

    def boom(_p):
        raise OSError("file is locked")

    monkeypatch.setattr(cleanup, "load_or_extract", boom)
    db = _FakeChroma([{"doc_hash": "h-1", "source_original": str(src)}])
    result = cleanup.cleanup_orphans_sqlite(db)

    assert result.gone == [] and result.stale == []
    with session_scope() as s:
        assert s.execute(select(Document)).scalars().all()
