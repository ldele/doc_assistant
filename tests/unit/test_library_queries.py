"""Tests for library metadata query detection and response."""

import contextlib
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from doc_assistant.query_router import is_library_query


@pytest.fixture
def temp_database(monkeypatch):
    """Isolated temp DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    test_engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(test_engine, "connect")
    def _enable_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", test_engine)
    test_session_factory = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, future=True
    )
    monkeypatch.setattr(session_module, "_SessionLocal", test_session_factory)

    from doc_assistant.db.models import Base

    Base.metadata.create_all(test_engine)

    yield path

    test_engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


# ============================================================
# Detection tests — pure logic, no DB
# ============================================================


class TestIsLibraryQuery:
    @pytest.mark.parametrize(
        "query",
        [
            "What is the latest document I added?",
            "show me the newest paper",
            "How many documents do I have?",
            "how many PDFs are in my library",
            "list all my documents",
            "show all papers",
            "what's in my library?",
            "what is in the database",
            "library stats",
            "library summary",
            "any broken documents?",
            "show marginal files",
            "document count",
            "paper total",
        ],
    )
    def test_detects_library_queries(self, query):
        assert is_library_query(query), f"Should detect: {query!r}"

    @pytest.mark.parametrize(
        "query",
        [
            "What does Hodgkin say about action potentials?",
            "Explain the methodology in the Hubel paper",
            "Compare the results across the three studies",
            "What is an embedding model?",
            "How does BM25 work?",
            "Tell me about neural networks",
        ],
    )
    def test_ignores_content_queries(self, query):
        assert not is_library_query(query), f"Should NOT detect: {query!r}"

    @pytest.mark.parametrize(
        "query",
        [
            # Topic-bearing "list/show" queries are CONTENT questions, not a
            # request to dump the library — they must fall through to RAG.
            "show my papers about deep learning",
            "list my pdfs on retrieval augmented generation",
            "display all my documents regarding action potentials",
            "what's in my library about embeddings",
            "show me papers discussing BM25",
        ],
    )
    def test_topical_list_queries_route_to_content(self, query):
        assert not is_library_query(query), f"Should NOT detect (topical): {query!r}"


# ============================================================
# Response tests — need DB
# ============================================================


class TestLibraryResponses:
    def test_empty_library_count(self, temp_database):
        from doc_assistant.library import library_summary

        s = library_summary()
        assert s.total_documents == 0

    def test_latest_returns_newest(self, temp_database):
        from doc_assistant.db.models import Document
        from doc_assistant.db.session import session_scope
        from doc_assistant.library import list_documents

        with session_scope() as session:
            session.add(
                Document(
                    filename="old.pdf",
                    source_original="/tmp/old.pdf",
                    doc_hash="hash_old",
                    format="pdf",
                    extraction_health="healthy",
                    chunk_count=10,
                )
            )
        with session_scope() as session:
            session.add(
                Document(
                    filename="new.pdf",
                    source_original="/tmp/new.pdf",
                    doc_hash="hash_new",
                    format="pdf",
                    extraction_health="healthy",
                    chunk_count=5,
                )
            )

        docs = list_documents()
        with_dates = [d for d in docs if d.added_at]
        assert len(with_dates) == 2
        latest = max(with_dates, key=lambda d: d.added_at)
        assert latest.filename == "new.pdf"

    def test_filter_broken(self, temp_database):
        from doc_assistant.db.models import Document
        from doc_assistant.db.session import session_scope
        from doc_assistant.library import list_documents

        with session_scope() as session:
            session.add(
                Document(
                    filename="good.pdf",
                    source_original="/tmp/good.pdf",
                    doc_hash="hash_good",
                    format="pdf",
                    extraction_health="healthy",
                    chunk_count=10,
                )
            )
            session.add(
                Document(
                    filename="bad.pdf",
                    source_original="/tmp/bad.pdf",
                    doc_hash="hash_bad",
                    format="pdf",
                    extraction_health="broken",
                    chunk_count=2,
                )
            )

        broken = list_documents(health="broken")
        assert len(broken) == 1
        assert broken[0].filename == "bad.pdf"

    def test_document_count(self, temp_database):
        from doc_assistant.db.models import Document
        from doc_assistant.db.session import session_scope
        from doc_assistant.library import library_summary

        with session_scope() as session:
            for i in range(3):
                session.add(
                    Document(
                        filename=f"doc{i}.pdf",
                        source_original=f"/tmp/doc{i}.pdf",
                        doc_hash=f"hash{i}",
                        format="pdf",
                        chunk_count=10,
                    )
                )

        s = library_summary()
        assert s.total_documents == 3
        assert s.total_chunks == 30
