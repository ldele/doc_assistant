"""A document's identity survives its own re-extraction (ADR-047).

Identity used to be `doc_hash(extracted_text)` alone, which made it hostage to the extractor:
every extraction improvement changed the text, changed the hash, minted a fresh id, and cut loose
everything keyed to the old one. Measured on the live 97-document library before this existed, a
single extractor change would have orphaned **4,123 rows** — 881 figure descriptions, 1,455
keywords, 445 epistemics rows, and 19 nobody can regenerate (18 folder assignments and a metadata
override the user typed by hand).

So the resolver falls back to `source_original`: the same file is the same document. These tests
pin both halves of that — what must survive, and what must NOT be inherited.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

SRC = r"C:\library\paper.pdf"


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


def _ingest(doc_id: str, *, doc_hash: str, source: str = SRC, chunks: int = 10) -> str:
    from doc_assistant.ingest.store import upsert_document_in_sqlite

    return upsert_document_in_sqlite(
        document_id=doc_id,
        filename="paper.pdf",
        source_original=source,
        source_cache="c.md",
        doc_hash=doc_hash,
        format="pdf",
        extractor_used="pymupdf",
        extraction_health="ok",
        chunk_count=chunks,
        page_count=3,
    )


def _first_ingest() -> str:
    from doc_assistant.ingest.store import _existing_document_id

    doc_id = _existing_document_id("hash-v1", SRC) or "doc-0001"
    _ingest(doc_id, doc_hash="hash-v1")
    return doc_id


def test_the_hash_is_still_the_first_key(temp_database):
    """Unchanged extraction must stay a single indexed lookup — the fallback is a fallback."""
    from doc_assistant.ingest.store import _existing_document_id

    first = _first_ingest()
    assert _existing_document_id("hash-v1") == first, "no source path needed when the hash matches"


def test_a_re_extraction_keeps_the_same_id(temp_database):
    """The whole point. A changed extractor must not mint a new document."""
    from doc_assistant.ingest.store import _existing_document_id

    first = _first_ingest()
    assert _existing_document_id("hash-v2-CHANGED", SRC) == first


def test_sidecars_stay_linked_across_a_re_extraction(temp_database):
    """What the id actually protects: 881 figure rows on the real library."""
    from doc_assistant.db.models import Document, Figure
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.store import _existing_document_id

    first = _first_ingest()
    with session_scope() as s:
        s.add(
            Figure(
                document_id=first,
                doc_hash="hash-v1",
                page=1,
                kind="figure",
                image_path="f.png",
                caption="Figure 1",
            )
        )

    second = _existing_document_id("hash-v2-CHANGED", SRC) or "doc-NEW"
    _ingest(second, doc_hash="hash-v2-CHANGED", chunks=12)

    with session_scope() as s:
        docs = s.execute(select(Document)).scalars().all()
        figs = s.execute(select(Figure)).scalars().all()
        assert len(docs) == 1, "a second row would mean the id was not reused"
        assert all(f.document_id == docs[0].id for f in figs), "the figure must still resolve"


def test_the_row_records_the_new_hash(temp_database):
    """The identity is stable; the content is not. The row must say what it actually holds."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.store import _existing_document_id

    _first_ingest()
    second = _existing_document_id("hash-v2-CHANGED", SRC) or "doc-NEW"
    _ingest(second, doc_hash="hash-v2-CHANGED")

    with session_scope() as s:
        (doc,) = s.execute(select(Document)).scalars().all()
        assert doc.doc_hash == "hash-v2-CHANGED"


def test_the_upsert_is_keyed_on_the_id_not_the_hash(temp_database):
    """The primary-key collision this would otherwise cause.

    `upsert_document_in_sqlite` used to look the row up by `doc_hash`. With the ADR-047 fallback
    the caller reuses the id while the hash moves, so a hash lookup misses, falls through to the
    insert branch, and collides on the primary key. Keyed on the id, it updates in place.
    """
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    first = _first_ingest()
    _ingest(first, doc_hash="a-totally-different-hash", chunks=99)

    with session_scope() as s:
        rows = s.execute(select(Document)).scalars().all()
        assert len(rows) == 1
        assert rows[0].chunk_count == 99


def test_a_different_file_does_not_inherit_an_identity(temp_database):
    """The fallback is per-path. A genuinely new document must be genuinely new."""
    from doc_assistant.ingest.store import _existing_document_id

    _first_ingest()
    assert _existing_document_id("hash-other", r"C:\library\other.pdf") is None


def test_no_source_path_means_no_fallback(temp_database):
    """A caller that cannot say where the file came from gets the old, strict behaviour."""
    from doc_assistant.ingest.store import _existing_document_id

    _first_ingest()
    assert _existing_document_id("hash-v2-CHANGED") is None


def test_the_path_comparison_is_normalised(temp_database):
    """The same file reaches this code resolved and unresolved depending on the caller."""
    from doc_assistant.ingest.store import _existing_document_id

    first = _first_ingest()
    assert _existing_document_id("hash-v2", r"c:\LIBRARY\paper.pdf") == first
    assert _existing_document_id("hash-v2", "C:/library/paper.pdf") == first


def test_replacing_the_file_at_a_path_inherits_the_identity(temp_database):
    """**The documented trade-off, asserted so it cannot change silently** (ADR-047).

    A different document written to the same path takes over the previous one's id and its
    sidecars. For a library where the path is the document's address this is the intended
    reading; it is recorded here because it is the cost of the fallback, not an accident.
    """
    from doc_assistant.ingest.store import _existing_document_id

    first = _first_ingest()
    assert _existing_document_id("an-entirely-unrelated-document", SRC) == first


# ============================================================
# The run-scoped path index — same answers, one query instead of one per document.
# ============================================================


def test_the_path_index_answers_exactly_as_the_table_scan_did(temp_database):
    """Equivalence is the whole requirement: this is an optimisation, not a behaviour change.

    The fallback re-read every Document row for each document whose normalised path it had to
    resolve — the common case during a corpus-wide re-extraction, since every hash moves, so the
    identity fallback was O(documents²) against the ~10,000-document contract.
    """
    from doc_assistant.ingest.store import _existing_document_id, build_path_index

    first = _ingest("doc-0001", doc_hash="hash-v1")
    _ingest("doc-0002", doc_hash="other-hash", source=r"C:\library\second.pdf")
    index = build_path_index()

    for probe in (SRC, r"c:\LIBRARY\paper.pdf", "C:/library/paper.pdf"):
        assert _existing_document_id("hash-v2-CHANGED", probe, path_index=index) == first
        assert _existing_document_id("hash-v2-CHANGED", probe) == first, "the scan agrees"

    # And it must not invent a match the scan would not have made.
    assert _existing_document_id("hash-x", r"C:\library\never.pdf", path_index=index) is None
    assert _existing_document_id("hash-x", r"C:\library\never.pdf") is None


def test_the_path_index_never_overrides_an_exact_hash_match(temp_database):
    """Rule 1 still wins: the index is consulted only after the hash and exact-path lookups."""
    from doc_assistant.ingest.store import _existing_document_id, build_path_index

    first = _ingest("doc-0001", doc_hash="hash-v1")
    index = build_path_index()
    assert _existing_document_id("hash-v1", path_index=index) == first
    assert _existing_document_id("hash-v1", r"C:\somewhere\else.pdf", path_index=index) == first


# ============================================================
# Carrying the figures across (ADR-047) — all three updates, or none.
# ============================================================


def _figure(document_id: str, doc_hash: str, image_path: str) -> None:
    from doc_assistant.db.models import Figure
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(
            Figure(
                document_id=document_id,
                doc_hash=doc_hash,
                page=1,
                kind="figure",
                image_path=image_path,
            )
        )


def test_figures_move_with_the_document_when_the_directory_can_be_renamed(
    temp_database, tmp_path, monkeypatch
):
    from doc_assistant.db.models import Figure
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import figures as figures_mod
    from doc_assistant.ingest.store import repoint_figures

    monkeypatch.setattr(figures_mod, "FIGURE_DIR", tmp_path / "figures")
    _ingest("doc-0001", doc_hash="hash-v1")
    old = figures_mod.figure_dir("hash-v1")
    old.mkdir(parents=True)
    (old / "page1_fig0.png").write_bytes(b"\x89PNG")
    _figure("doc-0001", "hash-v1", str(old / "page1_fig0.png"))

    assert repoint_figures("doc-0001", "hash-v2") == 1

    with session_scope() as session:
        row = session.execute(select(Figure)).scalar_one()
        assert row.doc_hash == "hash-v2"
        assert Path(row.image_path).exists(), "the stored path must still resolve"


def test_figures_do_not_move_when_the_destination_is_already_occupied(
    temp_database, tmp_path, monkeypatch
):
    """All three updates or none — the rows must not outrun the directory.

    The rename is skipped when the destination exists, but the row updates used to run anyway:
    every `image_path` was rewritten into a directory that does not hold those crops, and the
    real ones were left under a hash `hashes_with_no_figure_rows` would then read as dead and
    delete. Bailing keeps every stored path resolving.
    """
    from doc_assistant.db.models import Figure
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import figures as figures_mod
    from doc_assistant.ingest.store import repoint_figures

    monkeypatch.setattr(figures_mod, "FIGURE_DIR", tmp_path / "figures")
    _ingest("doc-0001", doc_hash="hash-v1")
    old = figures_mod.figure_dir("hash-v1")
    old.mkdir(parents=True)
    (old / "page1_fig0.png").write_bytes(b"\x89PNG")
    _figure("doc-0001", "hash-v1", str(old / "page1_fig0.png"))
    figures_mod.figure_dir("hash-v2").mkdir(parents=True)  # already occupied

    assert repoint_figures("doc-0001", "hash-v2") == 0, "nothing moved, so nothing is repointed"

    with session_scope() as session:
        row = session.execute(select(Figure)).scalar_one()
        assert row.doc_hash == "hash-v1", (
            "the row must not claim a hash its crops do not sit under"
        )
        assert Path(row.image_path).exists()
