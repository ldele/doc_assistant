"""Tests for the library data access layer."""

import contextlib
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def temp_database(monkeypatch):
    """Replace the global engine with one pointing to a temp database.

    This is the correct way to isolate database tests — patch the engine,
    not just the config string.
    """
    # Create temp DB file
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    # Build a fresh engine pointing at the temp file
    test_engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(test_engine, "connect")
    def _enable_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Patch both the engine and the session factory in db.session
    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", test_engine)
    test_session_factory = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, future=True
    )
    monkeypatch.setattr(session_module, "_SessionLocal", test_session_factory)

    # Create tables in the temp DB
    from doc_assistant.db.models import Base

    Base.metadata.create_all(test_engine)

    yield path

    # Teardown
    test_engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


# Now all the test functions take temp_database as a parameter
# to opt into the isolated DB:


def test_empty_library_summary(temp_database):
    from doc_assistant.library import library_summary

    s = library_summary()
    assert s.total_documents == 0
    assert s.total_chunks == 0


def test_add_and_list_document(temp_database):
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library import list_documents

    with session_scope() as session:
        doc = Document(
            filename="test.pdf",
            source_original="/tmp/test.pdf",
            doc_hash="abcd1234",
            format="pdf",
            extraction_health="healthy",
            chunk_count=10,
        )
        session.add(doc)

    docs = list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "test.pdf"


def test_filter_by_health(temp_database):
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library import list_documents

    with session_scope() as session:
        for i, h in enumerate(["healthy", "broken", "healthy"]):
            session.add(
                Document(
                    filename=f"doc{i}.pdf",
                    source_original=f"/tmp/doc{i}.pdf",
                    doc_hash=f"hash{i}",
                    format="pdf",
                    extraction_health=h,
                    chunk_count=10,
                )
            )

    healthy = list_documents(health="healthy")
    broken = list_documents(health="broken")
    assert len(healthy) == 2
    assert len(broken) == 1


def test_short_id_lookup(temp_database):
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library import find_document_by_short_id

    with session_scope() as session:
        doc = Document(
            filename="test.pdf",
            source_original="/tmp/test.pdf",
            doc_hash="hashxyz",
            format="pdf",
        )
        session.add(doc)
        session.flush()
        full_id = doc.id

    found = find_document_by_short_id(full_id[:8])
    assert found == full_id


# ============================================================
# Chunk browser (Library L1 — feature-library-browser.md)
# ============================================================
# `group_children` is pure — flat Chroma child chunks -> ordered parent blocks. No DB, no Chroma.
# The impure `get_document_chunks` (SQLite + live handle) is covered by the API integration test.


def _chunk(p, c, text, parent_text="P", keep=None):
    return {
        "parent_index": p,
        "child_index": c,
        "parent_text": parent_text,
        "text": text,
        "keep_for_retrieval": keep,
    }


def test_group_children_orders_and_groups():
    from doc_assistant.library import ChunkChild, ParentBlock, group_children

    # Deliberately out of order across two parents; child 0 of parent 0 is not retrievable.
    blocks = group_children(
        [
            _chunk(1, 1, "p1c1", parent_text="P1"),
            _chunk(0, 1, "p0c1", parent_text="P0"),
            _chunk(0, 0, "p0c0", parent_text="P0", keep=False),
            _chunk(1, 0, "p1c0", parent_text="P1"),
        ]
    )

    assert [b.parent_index for b in blocks] == [0, 1]  # parents ordered by parent_index
    assert isinstance(blocks[0], ParentBlock)
    assert blocks[0].parent_text == "P0"  # from the first-seen child of the parent
    assert [(c.child_index, c.text, c.retrievable) for c in blocks[0].children] == [
        (0, "p0c0", False),  # keep_for_retrieval=False -> retrievable=False
        (1, "p0c1", True),  # None -> retrievable (kept)
    ]
    assert isinstance(blocks[0].children[0], ChunkChild)
    assert [c.text for c in blocks[1].children] == ["p1c0", "p1c1"]


def test_group_children_drops_rows_missing_index():
    from doc_assistant.library import group_children

    blocks = group_children(
        [_chunk(0, 0, "keep"), _chunk(None, 0, "drop-no-parent"), _chunk(0, None, "drop-no-child")]
    )
    assert len(blocks) == 1
    assert [c.text for c in blocks[0].children] == ["keep"]


def test_group_children_parent_index_zero_is_not_dropped():
    from doc_assistant.library import group_children

    # Falsy-but-valid indices (0) must survive — the guard is `is None`, not truthiness.
    blocks = group_children([_chunk(0, 0, "zero")])
    assert [b.parent_index for b in blocks] == [0]
    assert blocks[0].children[0].child_index == 0


def test_group_children_empty():
    from doc_assistant.library import group_children

    assert group_children([]) == []


# ============================================================
# ADR-027 D1 (E4) — document_connections (the exploration bundle)
# ============================================================


def _seed_document(filename: str) -> str:
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=f"/tmp/{filename}",
            doc_hash=f"hash-{filename}",
            format="pdf",
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _seed_similarity(
    source_id: str, target_id: str, score: float, model: str = "bge-base"
) -> None:
    from doc_assistant.db.models import DocSimilarity
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(
            DocSimilarity(
                source_document_id=source_id,
                target_document_id=target_id,
                embedding_model=model,
                score=score,
            )
        )


def _seed_citation(
    source_id: str,
    *,
    target_id: str | None = None,
    title: str | None = None,
    year: int | None = None,
) -> None:
    from doc_assistant.db.models import Citation
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(
            Citation(
                source_document_id=source_id,
                target_document_id=target_id,
                target_title=title,
                target_year=year,
                extraction_method="regex",
            )
        )


def test_document_connections_unknown_doc_is_none(temp_database):
    from doc_assistant.library import document_connections

    assert document_connections("no-such-id") is None


def test_document_connections_empty_sidecars_degrade_to_empty_lists(temp_database):
    # The 0-doc/0-sidecar contract: a known doc with nothing computed returns an all-empty
    # bundle, never an error — the panel renders an honest empty state from this.
    from doc_assistant.library import document_connections

    doc = _seed_document("a.pdf")
    bundle = document_connections(doc)
    assert bundle is not None
    assert bundle.related == []
    assert bundle.cites == []
    assert bundle.cited_by == []
    assert bundle.external_refs == []
    assert bundle.external_total == 0


def test_document_connections_related_scoped_to_embedding_model(temp_database):
    # The similarity read is scoped to the embedder in use — edges computed under another
    # model must not leak into the panel (they describe a different geometry).
    from doc_assistant.library import document_connections

    a, b = _seed_document("a.pdf"), _seed_document("b.pdf")
    _seed_similarity(a, b, 0.91, model="bge-base")
    scoped = document_connections(a, embedding_model="bge-base")
    other = document_connections(a, embedding_model="specter2")
    assert scoped is not None and [r.target_document_id for r in scoped.related] == [b]
    assert other is not None and other.related == []


def test_document_connections_splits_internal_and_external_citations(temp_database):
    from doc_assistant.library import document_connections

    a, b = _seed_document("a.pdf"), _seed_document("b.pdf")
    _seed_citation(a, target_id=b, title="Resolved in-corpus paper")
    _seed_citation(a, title="External titled ref", year=2019)
    _seed_citation(a)  # unresolved AND untitled — not showable, not counted

    bundle = document_connections(a)
    assert bundle is not None
    assert [c.target_document_id for c in bundle.cites] == [b]
    assert [e.target_title for e in bundle.external_refs] == ["External titled ref"]
    assert bundle.external_total == 1  # the untitled row is excluded from the count too


def test_document_connections_dedupes_cited_by_with_count(temp_database):
    # A doc citing the subject 3 times is ONE row with n_citations=3 — the panel lists
    # documents, not raw citation rows.
    from doc_assistant.library import document_connections

    a, c = _seed_document("subject.pdf"), _seed_document("citer.pdf")
    for _ in range(3):
        _seed_citation(c, target_id=a, title="Subject paper")

    bundle = document_connections(a)
    assert bundle is not None
    assert len(bundle.cited_by) == 1
    assert bundle.cited_by[0].document_id == c
    assert bundle.cited_by[0].n_citations == 3


def test_document_connections_caps_external_refs_but_reports_total(temp_database):
    # No silent truncation: the list is capped, the total is the full titled count.
    from doc_assistant.library import document_connections

    a = _seed_document("a.pdf")
    for i in range(5):
        _seed_citation(a, title=f"External ref {i}")

    bundle = document_connections(a, external_cap=2)
    assert bundle is not None
    assert len(bundle.external_refs) == 2
    assert bundle.external_total == 5
