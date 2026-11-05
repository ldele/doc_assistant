"""Guard tests for the library summary (`library.summary`, ROADMAP 112).

The summary's promise is that every number is a count of something named, at 0 documents as at
10,000, and that *Show the N* opens exactly the N documents the count names. So the central test
checks each count against the document list's `issue` filter, which reads the same predicate.

A temp SQLite bound to the global session machinery, invented documents, no Chroma, no model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import (
    LIBRARY_ROOT_ID,
    Base,
    Citation,
    Document,
    DocumentMeta,
    ExternalMetadata,
    Figure,
    Folder,
    Keyword,
    SourceFile,
    SourceRoot,
)
from doc_assistant.db.session import session_scope
from doc_assistant.ingest.registry import pathkey
from doc_assistant.library.documents import list_documents
from doc_assistant.library.summary import (
    ISSUE_IDS,
    UnknownIssueError,
    issue_document_ids,
    summarise_collection,
)


@pytest.fixture
def lib(tmp_path: Path) -> Iterator[Path]:
    """An empty temp library; yields its folder."""
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "cache").mkdir()
    try:
        yield tmp_path
    finally:
        session_mod._engine, session_mod._SessionLocal = orig_engine, orig_factory
        engine.dispose()


def _doc(home: Path, name: str, **kw: Any) -> str:
    """One document, healthy, with text, a full set of metadata unless told otherwise."""
    fields: dict[str, Any] = {
        "filename": name,
        "source_original": str(home / "sources" / name),
        "source_cache": str(home / "cache" / (Path(name).stem + ".md")),
        "doc_hash": hashlib.sha256(name.encode("utf-8")).hexdigest(),
        "format": Path(name).suffix.lstrip(".") or "pdf",
        "title": f"Title of {name}",
        "authors": "A. Author",
        "year": 2020,
        "doi": f"10.1/{name}",
        "extraction_health": "healthy",
        "chunk_count": 10,
        "page_count": 5,
    }
    fields.update(kw)
    with session_scope() as session:
        doc = Document(**fields)
        session.add(doc)
        session.flush()
        return str(doc.id)


def _references(doc_id: str, n: int) -> None:
    with session_scope() as session:
        for i in range(n):
            session.add(Citation(source_document_id=doc_id, raw_citation_text=f"ref {i}"))


def _figures(doc_id: str, n: int, described: int = 0) -> None:
    with session_scope() as session:
        for i in range(n):
            session.add(
                Figure(
                    document_id=doc_id,
                    doc_hash="h",
                    page=1,
                    kind="figure",
                    vlm_description="A chart." if i < described else None,
                )
            )


def _keyword(doc_id: str, name: str = "retrieval") -> None:
    with session_scope() as session:
        doc = session.get(Document, doc_id)
        assert doc is not None
        kw = session.query(Keyword).filter_by(name=name).one_or_none() or Keyword(name=name)
        doc.keywords.append(kw)


def _folder(name: str, doc_ids: list[str]) -> str:
    with session_scope() as session:
        folder = Folder(name=name)
        session.add(folder)
        session.flush()
        for d in doc_ids:
            doc = session.get(Document, d)
            assert doc is not None
            doc.folders.append(folder)
        return str(folder.id)


def _spacing_record(home: Path, name: str, runs: list[dict[str, Any]]) -> None:
    cached = home / "cache" / (Path(name).stem + ".md")
    cached.with_name(cached.name + ".spacing.json").write_text(
        json.dumps({"version": 1, "runs": runs}), encoding="utf-8"
    )


def _registry(home: Path, files: list[tuple[str, bool, datetime]]) -> None:
    """Library-root registry rows: (rel_path, excluded, last_seen)."""
    with session_scope() as session:
        session.add(SourceRoot(id=LIBRARY_ROOT_ID, path=str(home / "sources"), kind="library"))
        # The files' foreign key is enforced, and no relationship orders the insert.
        session.flush()
        for rel, excluded, seen in files:
            session.add(
                SourceFile(
                    root_id=LIBRARY_ROOT_ID,
                    rel_path=rel,
                    format=Path(rel).suffix.lstrip("."),
                    size=1,
                    mtime=0.0,
                    excluded=excluded,
                    last_seen=seen,
                )
            )


# ---- 0 documents -------------------------------------------------------------------------------


def test_empty_library_reads_zero(lib: Path) -> None:
    s = summarise_collection()
    assert s is not None
    assert (s.documents, s.passages, s.pages, s.paged_documents) == (0, 0, 0, 0)
    assert s.indexing.no_text == s.indexing.flagged_marginal == s.indexing.flagged_broken == 0
    assert s.indexing.files_not_in_library == 0
    assert s.indexing.files_seen_at is None
    assert s.metadata.missing_title == 0
    assert s.found.references.on_record == s.found.references.none_on_record == 0
    assert s.types == []
    assert s.years.by_decade == []
    assert s.years.oldest is None and s.years.newest is None
    assert s.added.first is None and s.added.last is None
    assert s.need_attention == 0


def test_empty_library_still_counts_files_not_in_it(lib: Path) -> None:
    now = datetime(2026, 11, 5, 12, 0)
    _registry(lib, [("a.pdf", False, now), ("b.pdf", False, now), ("c.pdf", True, now)])
    s = summarise_collection()
    assert s is not None
    assert s.documents == 0
    assert s.indexing.files_not_in_library == 2
    assert s.indexing.files_excluded == 1
    assert s.need_attention == 2


# ---- one definition per count ------------------------------------------------------------------


@pytest.fixture
def mixed(lib: Path) -> Path:
    """At least one document per issue, and one document with none."""
    clean = _doc(lib, "clean.pdf")
    _references(clean, 3)
    _figures(clean, 2, described=1)
    _keyword(clean)
    _doc(lib, "scan.pdf", chunk_count=0, extraction_health="broken", page_count=12)
    _doc(lib, "thin.pdf", extraction_health="marginal")
    _doc(lib, "wreck.pdf", extraction_health="broken")
    _doc(lib, "untitled.pdf", title=None, authors="", year=None, doi=None)
    _doc(lib, "article.html", format="html", page_count=None)
    _doc(lib, "spaced.pdf")
    _spacing_record(lib, "spaced.pdf", [{"spaced": "A B S", "in_text": True}])
    _doc(lib, "spaced-off.pdf")
    _spacing_record(lib, "spaced-off.pdf", [{"spaced": "X Y", "in_text": False}])
    return lib


def test_each_count_matches_its_filter(mixed: Path) -> None:
    s = summarise_collection()
    assert s is not None
    expected = {
        "no_text": s.indexing.no_text,
        "flagged": s.indexing.flagged_marginal + s.indexing.flagged_broken,
        "missing_title": s.metadata.missing_title,
        "missing_authors": s.metadata.missing_authors,
        "missing_year": s.metadata.missing_year,
        "missing_doi": s.metadata.missing_doi,
        "no_references": s.found.references.none_on_record,
        "no_figures": s.found.figures.none_on_record,
        "no_keywords": s.found.keywords.none_on_record,
        "letter_spaced": s.found.letter_spacing.recorded,
    }
    assert set(expected) == set(ISSUE_IDS)
    for issue, count in expected.items():
        assert len(list_documents(issue=issue)) == count, issue
        assert len(issue_document_ids(issue)) == count, issue
    # The fixture is built so that no count is vacuous.
    assert all(count > 0 for count in expected.values()), expected


def test_the_mixed_library_counts(mixed: Path) -> None:
    s = summarise_collection()
    assert s is not None
    assert s.documents == 8
    assert s.indexing.no_text == 1
    assert (s.indexing.flagged_marginal, s.indexing.flagged_broken) == (1, 1)
    assert s.need_attention == 3  # no files registered, so the three document issues
    assert s.found.references.on_record == 1 and s.found.references.rows == 3
    assert s.found.figures.rows == 2 and s.found.figures.described == 1
    assert s.found.figures.pdfs == 7
    assert s.found.figures.none_on_record == 6  # the HTML page is not a PDF: never counted
    assert s.found.letter_spacing.recorded == 1
    assert s.paged_documents == 7
    assert [(t.format, t.documents) for t in s.types] == [("pdf", 7), ("html", 1)]


def test_unknown_issue_is_refused(lib: Path) -> None:
    with pytest.raises(UnknownIssueError):
        list_documents(issue="no_such_issue")


# ---- what each count means ---------------------------------------------------------------------


def test_override_counts_as_present(lib: Path) -> None:
    doc = _doc(lib, "a.pdf", title=None)
    _doc(lib, "b.pdf", title="   ")
    with session_scope() as session:
        session.add(DocumentMeta(document_id=doc, title_override="Given by the user"))
    s = summarise_collection()
    assert s is not None
    assert s.metadata.missing_title == 1  # b, whose title is blank
    assert s.metadata.set_by_you == 1


def test_with_any_field_tells_not_read_from_missing(lib: Path) -> None:
    _doc(lib, "bare.pdf", title=None, authors=None, year=None, doi=None)
    s = summarise_collection()
    assert s is not None
    assert s.metadata.with_any_field == 0  # nothing has read metadata for any document
    _doc(lib, "dated.pdf", title=None, authors=None, year=2001, doi=None)
    s = summarise_collection()
    assert s is not None
    assert s.metadata.with_any_field == 1


def test_catalogue_metadata_is_counted(lib: Path) -> None:
    _doc(lib, "a.pdf")
    _doc(lib, "b.pdf")
    with session_scope() as session:
        key = pathkey(lib / "sources" / "a.pdf")
        session.add(ExternalMetadata(source="catalogue", path_key=key))
    s = summarise_collection()
    assert s is not None
    assert s.metadata.from_a_catalogue == 1


def test_no_text_is_not_also_flagged(lib: Path) -> None:
    _doc(lib, "scan.pdf", chunk_count=0, extraction_health="broken")
    s = summarise_collection()
    assert s is not None
    assert s.indexing.no_text == 1
    assert s.indexing.flagged_broken == 0
    assert s.need_attention == 1


def test_unclassified_health_is_apart(lib: Path) -> None:
    _doc(lib, "old.pdf", extraction_health=None)
    s = summarise_collection()
    assert s is not None
    assert s.indexing.not_classified == 1
    assert s.indexing.flagged_marginal == s.indexing.flagged_broken == 0


def test_archived_counts_nowhere(lib: Path) -> None:
    _doc(lib, "gone.pdf", is_archived=True, chunk_count=0, title=None)
    s = summarise_collection()
    assert s is not None
    assert s.documents == 0
    assert s.indexing.no_text == 0
    assert s.metadata.missing_title == 0


def test_implausible_year_is_apart(lib: Path) -> None:
    _doc(lib, "a.pdf", year=0)
    _doc(lib, "b.pdf", year=3000)
    _doc(lib, "c.pdf", year=1987)
    _doc(lib, "d.pdf", year=1981)
    _doc(lib, "e.pdf", year=None)
    s = summarise_collection()
    assert s is not None
    assert s.metadata.implausible_year == 2
    assert s.metadata.missing_year == 1
    assert [(d.decade, d.documents) for d in s.years.by_decade] == [(1980, 2)]
    assert s.years.no_year == 1
    assert (s.years.oldest, s.years.newest) == (1981, 1987)


def test_letter_spacing_reads_the_record(lib: Path) -> None:
    _doc(lib, "in.pdf")
    _spacing_record(lib, "in.pdf", [{"spaced": "A B", "in_text": True}])
    _doc(lib, "out.pdf")
    _spacing_record(lib, "out.pdf", [{"spaced": "A B", "in_text": False}])
    _doc(lib, "none.pdf")
    _doc(lib, "page.html", format="html")
    _spacing_record(lib, "page.html", [{"spaced": "A B", "in_text": True}])  # never read
    s = summarise_collection()
    assert s is not None
    assert s.found.letter_spacing.recorded == 1
    assert s.found.letter_spacing.pdfs == 3


# ---- scope -------------------------------------------------------------------------------------


def test_folder_scope(lib: Path) -> None:
    a = _doc(lib, "a.pdf", year=None)
    b = _doc(lib, "b.pdf")
    _doc(lib, "c.pdf", year=None)
    now = datetime(2026, 11, 5, 12, 0)
    _registry(lib, [("x.pdf", False, now)])
    first = _folder("Thesis", [a, b])
    _folder("Elsewhere", [b])
    s = summarise_collection(first)
    assert s is not None
    assert s.scope == "folder" and s.folder_name == "Thesis"
    assert s.documents == 2
    assert s.metadata.missing_year == 1
    assert s.indexing.files_not_in_library is None
    assert s.indexing.files_excluded is None
    assert s.folders.also_elsewhere == 1
    assert s.folders.in_a_folder is None
    assert len(list_documents(issue="missing_year", folder_id=first)) == 1


def test_library_scope_counts_folder_membership(lib: Path) -> None:
    a = _doc(lib, "a.pdf")
    _doc(lib, "b.pdf")
    _folder("Thesis", [a])
    s = summarise_collection()
    assert s is not None
    assert (s.folders.in_a_folder, s.folders.in_none) == (1, 1)
    assert s.folders.also_elsewhere is None


def test_unknown_folder_is_none(lib: Path) -> None:
    assert summarise_collection("no-such-folder") is None


def test_files_not_in_library_from_the_last_look(lib: Path) -> None:
    _doc(lib, "indexed.pdf")
    last = datetime(2026, 11, 5, 12, 0)
    earlier = last - timedelta(days=3)
    _registry(
        lib,
        [
            ("indexed.pdf", False, last),  # has a document
            ("new.pdf", False, last),  # present, not in the library
            ("left-out.pdf", True, last),  # the user excluded it
            ("gone.pdf", False, earlier),  # not seen at the last look: gone from the folder
        ],
    )
    s = summarise_collection()
    assert s is not None
    assert s.indexing.files_not_in_library == 1
    assert s.indexing.files_excluded == 1
    assert s.indexing.files_seen_at == last


# ---- scale and side effects --------------------------------------------------------------------


def _summary_size(start: int, n: int, home: Path) -> int:
    with session_scope() as session:  # one session: a thousand rows in well under a second
        for i in range(start, start + n):
            name = f"doc{i:05d}.pdf"
            session.add(
                Document(
                    filename=name,
                    source_original=str(home / "sources" / name),
                    source_cache=str(home / "cache" / f"doc{i:05d}.md"),
                    doc_hash=f"h{i}",
                    format="pdf",
                    year=1900 + (i % 120),
                    extraction_health="healthy",
                    chunk_count=10,
                    page_count=5,
                )
            )
    s = summarise_collection()
    assert s is not None
    return len(json.dumps(asdict(s), default=str))


def test_wire_size_does_not_grow(lib: Path) -> None:
    small = _summary_size(0, 10, lib)
    # A second library in the same test would need a second engine; grow this one instead.
    large = _summary_size(10, 990, lib)
    # What may grow: the decades (bounded by the plausible range) and the digits of each count.
    assert large - small < 600, (small, large)


def test_summary_writes_nothing(mixed: Path) -> None:
    db = mixed / "library.db"
    before = db.read_bytes()
    cache_before = sorted(p.name for p in (mixed / "cache").iterdir())
    summarise_collection()
    list_documents(issue="letter_spaced")
    assert db.read_bytes() == before
    assert sorted(p.name for p in (mixed / "cache").iterdir()) == cache_before
