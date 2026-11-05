"""The app's own index run fills title, authors, year and DOI (`ingest._fill_missing_metadata`).

Before this step only a catalogue import gave a document metadata at index time, so a library
indexed in the app alone had none on any document (KI-87's library). The step is the lowest of
ADR-049's three slots: it runs after the catalogue's pass and writes a column only while it is
blank, so neither an earlier extraction nor a catalogue's answer is ever replaced.

Offline and deterministic: `.md` sources, a fake embedder, temp data dirs and a temp SQLite.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, ingest, metadata_enrich
from doc_assistant.adapters.catalogue import ExternalDocument, record_external
from doc_assistant.db.models import Base
from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.session import session_scope

_PAPER = """## **A Test Paper Title**

**Alice Author, Bob Builder**

Abstract. Some prose with enough content to chunk and embed for this test.

Published 2024.
"""

#: The same file after an edit: a different title on the first page, and a DOI it lacked before.
_PAPER_REVISED = """## **A Revised Paper Title**

**Alice Author, Bob Builder**

Abstract. Some prose with enough content to chunk and embed for this test, now revised.

Published 2024. DOI: 10.1234/foo.bar
"""

#: No heading, no names, no year, no DOI: the extractor finds nothing in it.
_NOTES = """plain notes without any structure, written in lower case and saying nothing about who
wrote them or when.

more lines of ordinary prose follow here so there is something to chunk and embed.
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    docs = tmp_path / "sources"
    for d in (docs, tmp_path / "cache", tmp_path / "chroma", tmp_path / "chroma_pc"):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DOCS_PATH", docs)
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "PC_CHROMA_PATH", str(tmp_path / "chroma_pc"))
    monkeypatch.setattr(
        ingest, "get_embeddings", lambda name=None: DeterministicFakeEmbedding(size=16)
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    try:
        yield docs
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _metadata(filename: str) -> dict[str, object]:
    """The four columns as plain values (a row read after its session closes is detached)."""
    with session_scope() as session:
        row = session.execute(
            select(
                DBDocument.id,
                DBDocument.title,
                DBDocument.authors,
                DBDocument.year,
                DBDocument.doi,
            ).where(DBDocument.filename == filename)
        ).one()
    return {"id": row[0], "title": row[1], "authors": row[2], "year": row[3], "doi": row[4]}


def test_an_empty_library_indexes_and_reads_no_metadata(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0 documents: the run completes and the step returns before it reads anything."""

    def _trap(document_ids: object) -> metadata_enrich.MetadataFill:
        raise AssertionError("the metadata step ran on a run that indexed nothing")

    monkeypatch.setattr(metadata_enrich, "fill_missing_metadata", _trap)

    assert ingest.main() == {"added": 0, "skipped": 0, "error": 0}


def test_fill_with_no_documents_reports_zeroes(env: Path) -> None:
    result = metadata_enrich.fill_missing_metadata([])
    assert (result.considered, result.filled, result.fields) == (0, 0, {})


def test_an_index_run_fills_the_metadata_the_text_holds(env: Path) -> None:
    (env / "paper.md").write_text(_PAPER, encoding="utf-8")

    assert ingest.main()["added"] == 1

    doc = _metadata("paper.md")
    assert doc["title"] == "A Test Paper Title"
    assert doc["authors"] == "Alice Author, Bob Builder"
    assert doc["year"] == 2024
    assert doc["doi"] is None, "the text holds no DOI, so none is invented"


def test_a_re_index_never_replaces_an_extracted_value(env: Path) -> None:
    """The file changes under the same path, so the re-index re-reads it (ADR-047 keeps its id).
    The first extraction's title stays; only the column that was still blank is filled."""
    source = env / "paper.md"
    source.write_text(_PAPER, encoding="utf-8")
    ingest.main()
    first = _metadata("paper.md")

    source.write_text(_PAPER_REVISED, encoding="utf-8")
    assert ingest.main()["added"] == 1

    again = _metadata("paper.md")
    assert again["id"] == first["id"], "the re-read must be the same document"
    assert again["title"] == "A Test Paper Title"
    assert again["authors"] == "Alice Author, Bob Builder"
    assert again["doi"] == "10.1234/foo.bar", "a blank column is filled on a re-read"


def test_a_re_index_never_replaces_a_catalogue_value(env: Path) -> None:
    """A catalogue's answer outranks the extractor's (ADR-049 D3). Fields it lacks come from the
    text; the ones it has stay its own, on the first index and on a re-index."""
    source = env / "paper.md"
    source.write_text(_PAPER, encoding="utf-8")
    with session_scope() as session:
        record_external(
            session,
            [
                ExternalDocument(
                    path=source, title="What The Catalogue Says", authors="Ada Lovelace"
                )
            ],
            source="zotero",
        )

    ingest.main()
    doc = _metadata("paper.md")
    assert doc["title"] == "What The Catalogue Says"
    assert doc["authors"] == "Ada Lovelace"
    assert doc["year"] == 2024, "a field the catalogue lacks comes from the text"

    source.write_text(_PAPER_REVISED, encoding="utf-8")
    assert ingest.main()["added"] == 1

    again = _metadata("paper.md")
    assert again["title"] == "What The Catalogue Says"
    assert again["authors"] == "Ada Lovelace"
    assert again["doi"] == "10.1234/foo.bar"


def test_a_document_whose_text_yields_nothing_keeps_its_blanks(env: Path) -> None:
    (env / "notes.md").write_text(_NOTES, encoding="utf-8")

    assert ingest.main()["added"] == 1

    doc = _metadata("notes.md")
    assert (doc["title"], doc["authors"], doc["year"], doc["doi"]) == (None, None, None, None)


def test_a_failing_metadata_step_never_fails_the_ingest(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken(document_ids: object) -> metadata_enrich.MetadataFill:
        raise RuntimeError("cache unreadable")

    monkeypatch.setattr(metadata_enrich, "fill_missing_metadata", _broken)
    (env / "paper.md").write_text(_PAPER, encoding="utf-8")

    assert ingest.main()["added"] == 1
    assert _metadata("paper.md")["title"] is None
