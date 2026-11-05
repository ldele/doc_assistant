"""The library summary: what went wrong at indexing, and what is known about the documents.

One read over a collection, the whole library or one folder, for the Library pane's *Summary*
layout (ROADMAP 112, `docs/specs/feature-library-summary.md`). Read-only: nothing here writes to
the database, the cache or the registry. In particular the registry is read as it stood at the
folder's last look (`SourceFile.last_seen`) rather than scanned, because a scan writes rows.

**One definition per count.** Each *issue* is one predicate over one document's facts
(:data:`ISSUES`). The counts sum it and the document list's ``issue`` filter
(`library.documents.list_documents`) selects by it, so the number on the screen and the documents
*Show the N* opens cannot disagree.

**"On record", not "absent".** The app's own indexing reads no metadata, references, figures or
keywords (only a catalogue import brings metadata, ADR-049), and nothing records that a re-run part
ran. So a document with no reference rows may have none, or may never have been looked at; the
counts say *on record* and the screen says why. The trust of each signal is the spec's table.

**Scale.** One pass over the collection's documents plus one grouped query per related table, never
a query per document; `citations.source_document_id` has no index, so a per-document ``EXISTS``
would read that table once per document. The one per-document cost is the letter-spacing record, a
file beside each PDF's cached text: one open attempt per PDF. Nothing on the wire grows with the
library: types are bounded by formats, decades by the plausible-year range.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from doc_assistant.db.models import (
    LIBRARY_ROOT_ID,
    Citation,
    Document,
    DocumentMeta,
    ExternalMetadata,
    Figure,
    Folder,
    SourceFile,
    SourceRoot,
    document_folders,
    document_keywords,
)
from doc_assistant.db.session import session_scope
from doc_assistant.library.citations import plausible_year
from doc_assistant.library.documents import effective_metadata, is_customized

log = structlog.get_logger(__name__)

__all__ = [
    "ISSUES",
    "ISSUE_IDS",
    "AddedRange",
    "CollectionSummary",
    "DecadeCount",
    "FigureCounts",
    "FolderCounts",
    "FoundCounts",
    "IndexingCounts",
    "MetadataCounts",
    "OnRecord",
    "SpacingCounts",
    "TypeCount",
    "UnknownIssueError",
    "YearCounts",
    "issue_document_ids",
    "summarise_collection",
]

#: Extraction health values that flag a document that still gave text (`health.py`).
_FLAGGED = frozenset({"marginal", "broken"})


class UnknownIssueError(ValueError):
    """An ``issue`` the summary does not define. The API maps it to a 400 naming the known ones."""

    def __init__(self, issue: str) -> None:
        self.issue = issue
        super().__init__(f"unknown issue {issue!r}; known: {', '.join(ISSUE_IDS)}")


@dataclass(frozen=True)
class _Facts:
    """What the summary knows about one document. Built once per read, never stored."""

    id: str
    format: str
    health: str | None
    chunk_count: int
    page_count: int | None
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    added_at: datetime | None
    customized: bool
    from_catalogue: bool
    references: int
    figures: int
    figures_described: int
    keywords: int
    folders: int
    #: ``None`` when the record was not read for this request (only the spacing issue needs it).
    letter_spaced: bool | None


def _blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _no_text(d: _Facts) -> bool:
    return d.chunk_count == 0


def _flagged(d: _Facts) -> bool:
    return not _no_text(d) and d.health in _FLAGGED


#: Every issue *Show the N* can open, in the order the screen lists them. One predicate each.
ISSUES: dict[str, Callable[[_Facts], bool]] = {
    "no_text": _no_text,
    "flagged": _flagged,
    "missing_title": lambda d: _blank(d.title),
    "missing_authors": lambda d: _blank(d.authors),
    "missing_year": lambda d: d.year is None,
    "missing_doi": lambda d: _blank(d.doi),
    "no_references": lambda d: d.references == 0,
    # Figure detection reads PDFs only, so another format is never "without figures".
    "no_figures": lambda d: d.format == "pdf" and d.figures == 0,
    "no_keywords": lambda d: d.keywords == 0,
    "letter_spaced": lambda d: bool(d.letter_spaced),
}
ISSUE_IDS: tuple[str, ...] = tuple(ISSUES)


# ---- what crosses the boundary -----------------------------------------------------------------


@dataclass(frozen=True)
class IndexingCounts:
    """Recorded when each file was indexed, so known for every document."""

    no_text: int
    flagged_marginal: int
    flagged_broken: int
    #: Health never recorded (indexed before it existed).
    not_classified: int
    #: Library scope only (``None`` in a folder): files in the documents folder at its last look
    #: that are in no document and that the user did not exclude.
    files_not_in_library: int | None
    files_excluded: int | None
    #: The folder's last look (a scan by Settings > Documents or an index run); ``None``: never.
    files_seen_at: datetime | None


@dataclass(frozen=True)
class MetadataCounts:
    missing_title: int
    missing_authors: int
    missing_year: int
    missing_doi: int
    #: A year is set but cannot be a publication year (`plausible_year`); in no decade.
    implausible_year: int
    set_by_you: int
    from_a_catalogue: int
    #: Documents with at least one of title, authors, year, DOI. At 0 with documents present no
    #: source has filled any field (indexing reads none), which the screen says once rather than
    #: as four rows of defects.
    with_any_field: int


@dataclass(frozen=True)
class OnRecord:
    on_record: int
    none_on_record: int
    rows: int = 0


@dataclass(frozen=True)
class FigureCounts:
    on_record: int
    none_on_record: int
    rows: int
    described: int
    pdfs: int


@dataclass(frozen=True)
class SpacingCounts:
    recorded: int
    pdfs: int


@dataclass(frozen=True)
class FoundCounts:
    references: OnRecord
    figures: FigureCounts
    letter_spacing: SpacingCounts
    keywords: OnRecord


@dataclass(frozen=True)
class TypeCount:
    format: str
    documents: int


@dataclass(frozen=True)
class DecadeCount:
    decade: int
    documents: int


@dataclass(frozen=True)
class YearCounts:
    by_decade: list[DecadeCount]
    no_year: int
    oldest: int | None
    newest: int | None


@dataclass(frozen=True)
class FolderCounts:
    """Library scope fills the first two, a folder's scope the third; the others are ``None``."""

    in_a_folder: int | None = None
    in_none: int | None = None
    also_elsewhere: int | None = None


@dataclass(frozen=True)
class AddedRange:
    first: datetime | None
    last: datetime | None


@dataclass(frozen=True)
class CollectionSummary:
    """The *Summary* layout's numbers for one collection. Counts only: never a ratio."""

    scope: Literal["library", "folder"]
    folder_id: str | None
    folder_name: str | None
    counted_at: datetime
    documents: int
    passages: int
    pages: int
    paged_documents: int
    #: no text + flagged + files not in the library: the screen's *Need attention* tile.
    need_attention: int
    indexing: IndexingCounts
    metadata: MetadataCounts
    found: FoundCounts
    types: list[TypeCount] = field(default_factory=list)
    years: YearCounts = field(default_factory=lambda: YearCounts([], 0, None, None))
    folders: FolderCounts = field(default_factory=FolderCounts)
    added: AddedRange = field(default_factory=lambda: AddedRange(None, None))


# ---- reading the facts -------------------------------------------------------------------------


def _per_document(session: Session, column: Any) -> dict[str, int]:
    """``document id -> row count`` for one related table, in one grouped pass."""
    rows = session.execute(select(column, func.count()).group_by(column)).all()
    return {str(doc_id): int(n) for doc_id, n in rows}


def _letter_spaced(source_original: str, source_cache: str | None) -> bool:
    """Whether the document's last extraction recorded a letter-spaced run that reached its text.

    No record means none was found, or the text was read before the record existed (ADR-056); both
    read as ``False`` here, which is why the screen says *recorded* rather than *none*."""
    from doc_assistant.ingest.cache import get_cache_path, read_spacing_record

    cached = Path(source_cache) if source_cache else get_cache_path(Path(source_original))
    return any(run.get("in_text") for run in read_spacing_record(cached))


def _facts(session: Session, document_ids: set[str] | None, *, spacing: bool) -> list[_Facts]:
    """One :class:`_Facts` per live document, restricted to ``document_ids`` when given."""
    from doc_assistant.ingest.registry import pathkey

    query = select(Document).where(Document.is_archived.is_(False))
    docs = [
        d
        for d in session.execute(query).scalars()
        if document_ids is None or str(d.id) in document_ids
    ]
    if not docs:
        return []
    overrides = {m.document_id: m for m in session.execute(select(DocumentMeta)).scalars()}
    catalogued = set(session.execute(select(ExternalMetadata.path_key)).scalars())
    references = _per_document(session, Citation.source_document_id)
    figures = _per_document(session, Figure.document_id)
    described = {
        str(doc_id): int(n)
        for doc_id, n in session.execute(
            select(Figure.document_id, func.count())
            .where(Figure.vlm_description.is_not(None))
            .group_by(Figure.document_id)
        ).all()
    }
    keywords = _per_document(session, document_keywords.c.document_id)
    folders = _per_document(session, document_folders.c.document_id)

    out: list[_Facts] = []
    for d in docs:
        doc_id = str(d.id)
        meta = overrides.get(doc_id)
        title, authors, year = effective_metadata(d, meta)
        fmt = (d.format or "").lower()
        out.append(
            _Facts(
                id=doc_id,
                format=fmt,
                health=d.extraction_health,
                chunk_count=int(d.chunk_count or 0),
                page_count=d.page_count,
                title=title,
                authors=authors,
                year=year,
                doi=d.doi,
                added_at=d.added_at,
                customized=is_customized(meta),
                from_catalogue=pathkey(d.source_original) in catalogued,
                references=references.get(doc_id, 0),
                figures=figures.get(doc_id, 0),
                figures_described=described.get(doc_id, 0),
                keywords=keywords.get(doc_id, 0),
                folders=folders.get(doc_id, 0),
                letter_spaced=(
                    _letter_spaced(d.source_original, d.source_cache)
                    if spacing and fmt == "pdf"
                    else (False if spacing else None)
                ),
            )
        )
    return out


def _folder_scope(session: Session, folder_id: str | None) -> tuple[set[str] | None, str | None]:
    """The document ids a folder holds and its name; ``(None, None)`` for the whole library.

    Raises ``LookupError`` for a folder that does not exist."""
    if folder_id is None:
        return None, None
    folder = session.get(Folder, folder_id)
    if folder is None:
        raise LookupError(folder_id)
    ids = set(
        session.execute(
            select(document_folders.c.document_id).where(document_folders.c.folder_id == folder_id)
        ).scalars()
    )
    return {str(i) for i in ids}, str(folder.name)


def _files(session: Session) -> tuple[int, int, datetime | None]:
    """Files in the documents folder at its last look: not in the library, and excluded.

    The documents folder is the ``library`` root. A referenced root is left out on purpose: it is
    someone else's folder (a reference manager's storage, say), registered because one file in it
    was added, and its other files were never meant for the library. A row counts when its
    ``last_seen`` is the root's newest, i.e. the file was there at the last look; an older row is a
    file that has since gone."""
    from doc_assistant.ingest.registry import pathkey

    root = session.get(SourceRoot, LIBRARY_ROOT_ID)
    rows = session.execute(
        select(SourceFile.rel_path, SourceFile.excluded, SourceFile.last_seen).where(
            SourceFile.root_id == LIBRARY_ROOT_ID
        )
    ).all()
    if root is None or not rows:
        return 0, 0, None
    seen = max(row.last_seen for row in rows)
    # Every document, archived ones included: an archived document's file is in the library.
    indexed = {pathkey(s) for s in session.execute(select(Document.source_original)).scalars()}
    base = Path(root.path)
    not_in = excluded = 0
    for row in rows:
        if row.last_seen != seen or pathkey(base / row.rel_path) in indexed:
            continue
        if row.excluded:
            excluded += 1
        else:
            not_in += 1
    return not_in, excluded, seen


def _count(facts: Iterable[_Facts], issue: str) -> int:
    test = ISSUES[issue]
    return sum(1 for d in facts if test(d))


def summarise_collection(folder_id: str | None = None) -> CollectionSummary | None:
    """The summary of the whole library, or of one folder. ``None`` for an unknown folder.

    Correct at 0 documents (every count 0, no ratio anywhere) and read-only."""
    with session_scope() as session:
        try:
            scope_ids, folder_name = _folder_scope(session, folder_id)
        except LookupError:
            return None
        facts = _facts(session, scope_ids, spacing=True)
        files = _files(session) if folder_id is None else None

    no_text = _count(facts, "no_text")
    marginal = sum(1 for d in facts if _flagged(d) and d.health == "marginal")
    broken = sum(1 for d in facts if _flagged(d) and d.health == "broken")
    not_in_library = files[0] if files is not None else None

    pdfs = sum(1 for d in facts if d.format == "pdf")
    years = [d.year for d in facts if d.year is not None]
    plausible = [y for y in years if plausible_year(y) is not None]
    decades = Counter(y // 10 * 10 for y in plausible)
    added = [d.added_at for d in facts if d.added_at is not None]
    types = Counter(d.format for d in facts)

    with_refs = sum(1 for d in facts if d.references > 0)
    with_keywords = sum(1 for d in facts if d.keywords > 0)
    in_a_folder = sum(1 for d in facts if d.folders > 0)

    return CollectionSummary(
        scope="library" if folder_id is None else "folder",
        folder_id=folder_id,
        folder_name=folder_name,
        counted_at=datetime.now(timezone.utc),
        documents=len(facts),
        passages=sum(d.chunk_count for d in facts),
        pages=sum(d.page_count or 0 for d in facts),
        paged_documents=sum(1 for d in facts if d.page_count is not None),
        need_attention=no_text + marginal + broken + (not_in_library or 0),
        indexing=IndexingCounts(
            no_text=no_text,
            flagged_marginal=marginal,
            flagged_broken=broken,
            not_classified=sum(1 for d in facts if d.health is None and not _no_text(d)),
            files_not_in_library=not_in_library,
            files_excluded=files[1] if files is not None else None,
            files_seen_at=files[2] if files is not None else None,
        ),
        metadata=MetadataCounts(
            missing_title=_count(facts, "missing_title"),
            missing_authors=_count(facts, "missing_authors"),
            missing_year=_count(facts, "missing_year"),
            missing_doi=_count(facts, "missing_doi"),
            implausible_year=len(years) - len(plausible),
            set_by_you=sum(1 for d in facts if d.customized),
            from_a_catalogue=sum(1 for d in facts if d.from_catalogue),
            with_any_field=sum(
                1
                for d in facts
                if not (_blank(d.title) and _blank(d.authors) and d.year is None and _blank(d.doi))
            ),
        ),
        found=FoundCounts(
            references=OnRecord(
                on_record=with_refs,
                none_on_record=_count(facts, "no_references"),
                rows=sum(d.references for d in facts),
            ),
            figures=FigureCounts(
                on_record=sum(1 for d in facts if d.figures > 0),
                none_on_record=_count(facts, "no_figures"),
                rows=sum(d.figures for d in facts),
                described=sum(d.figures_described for d in facts),
                pdfs=pdfs,
            ),
            letter_spacing=SpacingCounts(recorded=_count(facts, "letter_spaced"), pdfs=pdfs),
            keywords=OnRecord(
                on_record=with_keywords, none_on_record=_count(facts, "no_keywords")
            ),
        ),
        types=[
            TypeCount(fmt, n) for fmt, n in sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        years=YearCounts(
            by_decade=[DecadeCount(dec, n) for dec, n in sorted(decades.items())],
            no_year=len(facts) - len(years),
            oldest=min(plausible) if plausible else None,
            newest=max(plausible) if plausible else None,
        ),
        folders=(
            FolderCounts(in_a_folder=in_a_folder, in_none=len(facts) - in_a_folder)
            if folder_id is None
            else FolderCounts(also_elsewhere=sum(1 for d in facts if d.folders > 1))
        ),
        added=AddedRange(first=min(added) if added else None, last=max(added) if added else None),
    )


def issue_document_ids(issue: str, folder_id: str | None = None) -> set[str]:
    """The ids of the documents an issue names, in the library or one folder.

    Raises :class:`UnknownIssueError` for an issue :data:`ISSUES` does not define, and returns an
    empty set for an unknown folder (the list it filters is empty there too)."""
    if issue not in ISSUES:
        raise UnknownIssueError(issue)
    test = ISSUES[issue]
    with session_scope() as session:
        try:
            scope_ids, _ = _folder_scope(session, folder_id)
        except LookupError:
            return set()
        facts = _facts(session, scope_ids, spacing=issue == "letter_spaced")
    return {d.id for d in facts if test(d)}
