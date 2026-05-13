"""Tests for the library data access layer."""
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
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
    try:
        os.unlink(path)
    except OSError:
        pass


# Now all the test functions take temp_database as a parameter
# to opt into the isolated DB:

def test_empty_library_summary(temp_database):
    from doc_assistant.library import library_summary
    s = library_summary()
    assert s.total_documents == 0
    assert s.total_chunks == 0


def test_add_and_list_document(temp_database):
    from doc_assistant.db.session import session_scope
    from doc_assistant.db.models import Document
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
    from doc_assistant.db.session import session_scope
    from doc_assistant.db.models import Document
    from doc_assistant.library import list_documents

    with session_scope() as session:
        for i, h in enumerate(["healthy", "broken", "healthy"]):
            session.add(Document(
                filename=f"doc{i}.pdf",
                source_original=f"/tmp/doc{i}.pdf",
                doc_hash=f"hash{i}",
                format="pdf",
                extraction_health=h,
                chunk_count=10,
            ))

    healthy = list_documents(health="healthy")
    broken = list_documents(health="broken")
    assert len(healthy) == 2
    assert len(broken) == 1


def test_short_id_lookup(temp_database):
    from doc_assistant.db.session import session_scope
    from doc_assistant.db.models import Document
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