"""Guard tests for the shared ``--doc`` resolver (KI-30).

Before this existed, ``--doc`` meant three different things across four sidecar runners, and two of
them (`extract_citations`, `extract_doc_metadata`) filtered on ``doc_hash`` only — so passing the
document id that every other surface in the app hands out exited 1 with "No documents matched."

The first test in `TestResolveDocumentPrefix` is that exact regression. The runner tests at the
bottom pin the wiring, because a correct resolver that no runner calls fixes nothing.
"""

import contextlib
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def temp_database(monkeypatch):
    """Isolated temp DB (same shape as tests/unit/test_library_queries.py)."""
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
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True),
    )

    from doc_assistant.db.models import Base

    Base.metadata.create_all(test_engine)

    yield path

    test_engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


# A realistic-looking id/hash pair, because the bug was specifically about which column a prefix
# was matched against. Both are synthetic fixtures, not credentials.
_DOC_ID = "fe739b78-eb66-465e-a0e1-cb140831995a"
_DOC_HASH = "8fc5f0c8b3696c1c"  # pragma: allowlist secret


def _add_doc(doc_id: str, doc_hash: str, filename: str) -> None:
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(
            Document(
                id=doc_id,
                filename=filename,
                source_original=f"/tmp/{filename}",
                doc_hash=doc_hash,
                format="pdf",
                extraction_health="healthy",
                chunk_count=10,
            )
        )


class TestResolveDocumentPrefix:
    def test_a_real_document_id_resolves(self, temp_database):
        """KI-30's headline symptom: the id the app hands out must resolve."""
        from doc_assistant.library import resolve_document_prefix

        _add_doc(_DOC_ID, _DOC_HASH, "paper.pdf")

        ref = resolve_document_prefix(_DOC_ID)
        assert ref.id == _DOC_ID
        assert ref.doc_hash == _DOC_HASH
        assert ref.filename == "paper.pdf"

    def test_an_id_prefix_resolves(self, temp_database):
        from doc_assistant.library import resolve_document_prefix

        _add_doc(_DOC_ID, _DOC_HASH, "paper.pdf")

        assert resolve_document_prefix("fe739b78").id == _DOC_ID

    def test_a_doc_hash_prefix_still_resolves(self, temp_database):
        """The old behaviour must keep working — hashes are what the older runners printed."""
        from doc_assistant.library import resolve_document_prefix

        _add_doc(_DOC_ID, _DOC_HASH, "paper.pdf")

        assert resolve_document_prefix("8fc5f0c8").id == _DOC_ID

    def test_id_wins_over_hash_when_a_prefix_could_match_both(self, temp_database):
        """Both columns are hex, so collisions are possible; id is the documented winner."""
        from doc_assistant.library import resolve_document_prefix

        _add_doc("abc11111-0000-0000-0000-000000000001", "ffffffffffffffff", "by-id.pdf")
        _add_doc("def22222-0000-0000-0000-000000000002", "abc1abc1abc1abc1", "by-hash.pdf")

        assert resolve_document_prefix("abc1").filename == "by-id.pdf"

    def test_no_match_raises(self, temp_database):
        from doc_assistant.library import DocumentPrefixError, resolve_document_prefix

        _add_doc(_DOC_ID, _DOC_HASH, "paper.pdf")

        with pytest.raises(DocumentPrefixError, match="matched no document"):
            resolve_document_prefix("zzzzzzzz")

    def test_ambiguous_prefix_raises_and_names_the_candidates(self, temp_database):
        from doc_assistant.library import DocumentPrefixError, resolve_document_prefix

        _add_doc("aaaa1111-0000-0000-0000-000000000001", "1111111111111111", "one.pdf")
        _add_doc("aaaa2222-0000-0000-0000-000000000002", "2222222222222222", "two.pdf")

        with pytest.raises(DocumentPrefixError, match="ambiguous") as exc:
            resolve_document_prefix("aaaa")
        assert "one.pdf" in str(exc.value)
        assert "two.pdf" in str(exc.value)

    def test_blank_prefix_raises_rather_than_matching_everything(self, temp_database):
        from doc_assistant.library import DocumentPrefixError, resolve_document_prefix

        _add_doc(_DOC_ID, _DOC_HASH, "paper.pdf")

        with pytest.raises(DocumentPrefixError, match="non-empty"):
            resolve_document_prefix("   ")

    def test_empty_corpus_raises_instead_of_crashing(self, temp_database):
        """Robustness contract: 0 documents is a supported state."""
        from doc_assistant.library import DocumentPrefixError, resolve_document_prefix

        with pytest.raises(DocumentPrefixError, match="matched no document"):
            resolve_document_prefix("anything")

    def test_underscore_is_not_a_like_wildcard(self, temp_database):
        """``_`` matches one char in LIKE; unescaped it would resolve the wrong document."""
        from doc_assistant.library import DocumentPrefixError, resolve_document_prefix

        _add_doc("aXb11111-0000-0000-0000-000000000001", "1111111111111111", "real.pdf")

        with pytest.raises(DocumentPrefixError, match="matched no document"):
            resolve_document_prefix("a_b")


class TestRunnersUseTheSharedResolver:
    """A correct resolver nobody calls fixes nothing — pin the wiring per runner (KI-30)."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "scripts.extract_citations",
            "scripts.extract_doc_metadata",
            "scripts.compute_doc_vectors",
            "scripts.extract_keywords",
        ],
    )
    def test_runner_imports_the_shared_resolver(self, module_name):
        import importlib

        module = importlib.import_module(module_name)
        from doc_assistant.library.documents import resolve_document_prefix

        assert module.resolve_document_prefix is resolve_document_prefix

    @pytest.mark.parametrize(
        "module_name",
        ["scripts.extract_citations", "scripts.extract_doc_metadata"],
    )
    def test_the_two_hash_only_runners_no_longer_filter_on_doc_hash(self, module_name):
        """Regression pin: both used ``Document.doc_hash.startswith(args.doc)``."""
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module_name).main)
        assert "doc_hash.startswith" not in source
