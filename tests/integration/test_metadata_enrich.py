"""Integration tests for the metadata-enrichment runner (metadata_enrich.py).

Seeds ``Document`` rows into a temp DB and fakes ``load_document_texts`` to return canned
markdown, then exercises the persist path: dry-run writes nothing, apply fills NULL columns
only, and ``force`` overwrites. The extractor itself is unit-tested in
``test_metadata_extractor.py`` — here we prove the DB wiring and idempotency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope

# Clean header markdown → title / authors / year / doi all extractable.
_MD = (
    "## **A Test Paper Title**\n\n"
    "**Alice Author, Bob Builder**\n\n"
    "Abstract. Body.\n\nPublished 2024. DOI: 10.1234/foo.bar\n"
)


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield
    engine.dispose()


def _seed_doc(filename: str, *, title: str | None = None) -> str:
    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=f"/tmp/{filename}",
            doc_hash=f"hash-{filename}",
            format="pdf",
            title=title,
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _fake_corpus(doc_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "doc_assistant.metadata_enrich.load_document_texts",
        lambda ids=None: [(doc_id, "paper.pdf", _MD)],
    )


def _row_meta(doc_id: str) -> tuple[str | None, str | None, int | None, str | None]:
    with session_scope() as session:
        doc = session.get(Document, doc_id)
        assert doc is not None
        return doc.title, doc.authors, doc.year, doc.doi


def test_apply_populates_null_fields(temp_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from doc_assistant.metadata_enrich import enrich_metadata

    doc_id = _seed_doc("paper.pdf")
    _fake_corpus(doc_id, monkeypatch)

    result = enrich_metadata(apply=True)

    title, authors, year, doi = _row_meta(doc_id)
    assert title == "A Test Paper Title"
    assert authors and "Alice" in authors
    assert year == 2024
    assert doi == "10.1234/foo.bar"
    assert result.total_fields_written == 4
    assert set(result.docs[0].written_fields) == {"title", "authors", "year", "doi"}


def test_dry_run_writes_nothing(temp_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from doc_assistant.metadata_enrich import enrich_metadata

    doc_id = _seed_doc("paper.pdf")
    _fake_corpus(doc_id, monkeypatch)

    result = enrich_metadata(apply=False)

    assert _row_meta(doc_id) == (None, None, None, None)  # untouched
    assert result.total_fields_written == 0
    assert result.docs[0].written_fields == ()
    assert result.n_title == 1  # extraction still reported


def test_idempotent_keeps_existing_title(temp_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from doc_assistant.metadata_enrich import enrich_metadata

    doc_id = _seed_doc("paper.pdf", title="Human-Curated Title")
    _fake_corpus(doc_id, monkeypatch)

    result = enrich_metadata(apply=True)  # force defaults False

    title, authors, _year, _doi = _row_meta(doc_id)
    assert title == "Human-Curated Title"  # NOT overwritten
    assert authors and "Alice" in authors  # the NULL field was still filled
    assert "title" not in result.docs[0].written_fields
    assert "authors" in result.docs[0].written_fields


def test_force_overwrites_existing_title(temp_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from doc_assistant.metadata_enrich import enrich_metadata

    doc_id = _seed_doc("paper.pdf", title="Stale Title")
    _fake_corpus(doc_id, monkeypatch)

    enrich_metadata(apply=True, force=True)

    title, _authors, _year, _doi = _row_meta(doc_id)
    assert title == "A Test Paper Title"  # overwritten under --force
