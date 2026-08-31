"""Metadata imported from an outside catalogue (ADR-049, ROADMAP 17).

Three slots hold a document's title and they are not interchangeable, which is the whole reason
this layer exists:

* `Document.title` — the best answer *the machine* has. Every metadata pass overwrites it.
* `DocumentMeta.title_override` — what the **user typed here** (ADR-013). Nothing but the user
  writes it, and it wins at read time.
* `ExternalMetadata.title` — what the user's **catalogue** says. Curated by a person, but not by
  a person using this app.

The catalogue's answer beats the extractor's guess and loses to an in-app edit, and the tests
below pin both halves of that. The second half matters more: importing a Zotero library must not
silently replace corrections the user has already made here.

Offline and deterministic — `.md` sources and a fake embedder, so no PDF extractor is involved.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, ingest, reingest
from doc_assistant.adapters.catalogue import (
    ExternalDocument,
    apply_external_metadata,
    external_for_path,
    record_external,
)
from doc_assistant.db.models import Base, DocumentMeta
from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.session import session_scope

_BODY = """# A misleading first page

Journal of Things, Volume 4

Some prose with enough content to chunk and embed for this test.
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


def _record(path: Path, **over: object) -> None:
    fields: dict[str, object] = {
        "title": "What The Catalogue Says",
        "authors": "Ada Lovelace, Alan Turing",
        "year": 1949,
        "doi": "10.1000/curated",
        "item_type": "journalArticle",
        "collections": ("Reading / RAG",),
        "external_key": "PAPER001",
    }
    fields.update(over)
    with session_scope() as session:
        record_external(session, [ExternalDocument(path=path, **fields)], source="zotero")  # type: ignore[arg-type]


def _document(filename: str = "paper.md") -> dict[str, object]:
    """The document's columns as plain values — an ORM row read after its session closes is
    detached, and touching an attribute then raises rather than returning what was stored."""
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


# --- recording ------------------------------------------------------------------------------ #


def test_re_importing_the_same_library_corrects_rather_than_duplicates(env: Path) -> None:
    """Re-import is how a user picks up edits they made in the catalogue since last time."""
    source = env / "paper.md"
    source.write_text(_BODY, encoding="utf-8")
    _record(source, title="First answer")
    _record(source, title="Corrected in Zotero")

    with session_scope() as session:
        row = external_for_path(session, source)
        assert row is not None and row.title == "Corrected in Zotero"
        from doc_assistant.db.models import ExternalMetadata

        assert len(session.execute(select(ExternalMetadata)).scalars().all()) == 1


def test_a_record_can_be_written_before_the_document_exists(env: Path) -> None:
    """The point of a separate table: import happens before extraction, and may never get there."""
    source = env / "never-indexed.md"
    source.write_text(_BODY, encoding="utf-8")
    _record(source)
    with session_scope() as session:
        assert external_for_path(session, source) is not None
        assert apply_external_metadata(session).filled == 0


# --- applying ------------------------------------------------------------------------------- #


def test_an_ingest_gives_the_document_the_catalogues_metadata(env: Path) -> None:
    """End to end: the import records it, and the next ingest is where it lands."""
    source = env / "paper.md"
    source.write_text(_BODY, encoding="utf-8")
    _record(source)

    assert ingest.main()["added"] == 1

    doc = _document()
    assert doc["title"] == "What The Catalogue Says"
    assert doc["authors"] == "Ada Lovelace, Alan Turing"
    assert doc["year"] == 1949
    assert doc["doi"] == "10.1000/curated"


def test_a_document_no_catalogue_describes_is_left_alone(env: Path) -> None:
    (env / "paper.md").write_text(_BODY, encoding="utf-8")
    (env / "other.md").write_text(_BODY.replace("misleading", "ordinary"), encoding="utf-8")
    _record(env / "paper.md")
    ingest.main()

    assert _document("paper.md")["title"] == "What The Catalogue Says"
    assert _document("other.md")["title"] != "What The Catalogue Says"


def test_applying_is_idempotent(env: Path) -> None:
    source = env / "paper.md"
    source.write_text(_BODY, encoding="utf-8")
    _record(source)
    ingest.main()

    with session_scope() as session:
        second = apply_external_metadata(session)
    # Nothing left to change — the count is documents *changed*, not documents matched.
    assert second.considered == 1 and second.filled == 0


def test_fill_the_blanks_mode_leaves_an_existing_value_alone(env: Path) -> None:
    source = env / "paper.md"
    source.write_text(_BODY, encoding="utf-8")
    _record(source)
    ingest.main()
    with session_scope() as session:
        doc = session.execute(select(DBDocument)).scalar_one()
        doc.title = "Something the extractor found later"
        doc.year = None

    with session_scope() as session:
        applied = apply_external_metadata(session, overwrite=False)

    assert applied.filled == 1
    doc = _document()
    assert doc["title"] == "Something the extractor found later"
    assert doc["year"] == 1949, "a blank field should still be filled"


def test_an_in_app_edit_still_wins(env: Path) -> None:
    """ADR-013's slot is the user's own. An import must not silently undo a correction they made
    here — this is the one direction that would lose work rather than improve it."""
    from doc_assistant.library.documents import effective_metadata

    source = env / "paper.md"
    source.write_text(_BODY, encoding="utf-8")
    _record(source)
    ingest.main()
    doc_id = str(_document()["id"])
    with session_scope() as session:
        session.add(DocumentMeta(document_id=doc_id, title_override="What I Called It"))

    with session_scope() as session:
        apply_external_metadata(session)
        doc = session.get(DBDocument, doc_id)
        assert doc is not None
        title = effective_metadata(doc, session.get(DocumentMeta, doc_id))[0]
    assert title == "What I Called It"


def test_matching_survives_a_differently_written_path(env: Path) -> None:
    """The registry's own normalisation, so a separator or case difference is not a new file."""
    source = env / "paper.md"
    source.write_text(_BODY, encoding="utf-8")
    _record(Path(str(source).replace("/", "\\")) if "/" in str(source) else source)
    ingest.main()
    assert _document()["title"] == "What The Catalogue Says"


# --- the re-run must not undo it -------------------------------------------------------------- #


def test_re_running_metadata_restores_the_catalogues_answer(env: Path) -> None:
    """The trap this closes: "Metadata" is the cheapest, safest-looking box in the re-run dialog,
    and without this it would replace a curated title with the extractor's guess at it."""
    source = env / "paper.md"
    source.write_text(_BODY, encoding="utf-8")
    _record(source)
    ingest.main()
    doc_id = str(_document()["id"])
    with session_scope() as session:
        doc = session.get(DBDocument, doc_id)
        assert doc is not None
        doc.title = "A Misleading First Page"

    result = reingest.rerun([doc_id], ["metadata"])

    assert result.outcomes[0].status == "ok"
    assert "zotero" in result.outcomes[0].detail
    assert _document()["title"] == "What The Catalogue Says"


def test_re_running_metadata_on_an_unimported_document_still_extracts(env: Path) -> None:
    """The guard must not turn the ordinary path off for everyone else."""
    (env / "paper.md").write_text(_BODY, encoding="utf-8")
    ingest.main()
    result = reingest.rerun([str(_document()["id"])], ["metadata"])
    assert result.outcomes[0].status in {"ok", "skipped"}
    assert "zotero" not in result.outcomes[0].detail
