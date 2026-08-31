"""The vendor-neutral half of an ingestion adapter (ADR-049).

One shape — `ExternalDocument` — that every adapter returns, and the three things the rest of the
app does with it: record what the catalogue said, look it up by path, and apply it to the documents
that now exist.

**Why the metadata is kept at all.** A reference manager's title, authors and year are curated by a
person; `metadata_extractor` guesses them from a PDF's first page and sometimes picks the journal
name (KI-54). When the user has already done that work, throwing it away and re-deriving a worse
answer would be perverse. So an import records what the catalogue said, and it becomes the
document's metadata the moment the document exists.

**Why applying it is a separate step.** Import happens before extraction — there is no `Document`
row to write onto yet, and there may never be one if the user does not go on to index the file.
`apply_external_metadata` is therefore idempotent and safe to call whenever: after an ingest, after
a re-run, or over the whole library.

Nothing in this module knows a vendor schema. That lives in `adapters/zotero.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.models import ExternalMetadata
from doc_assistant.ingest.registry import pathkey

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ExternalDocument:
    """One file an outside catalogue knows about, in our words rather than the catalogue's.

    `path` is where the file is **now**, absolute and resolved — an adapter's job includes turning
    whatever storage scheme the vendor uses into a path this app can open. Everything else is
    optional, because a catalogue entry with nothing but a file is still worth importing.
    """

    path: Path
    title: str | None = None
    #: Comma-joined, matching `Document.authors` — one convention across the whole app.
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    #: The catalogue's own type name (`journalArticle`, `book`, …). Recorded, not yet acted on.
    item_type: str | None = None
    #: Collection / shelf / folder names the item sits in, outermost first.
    collections: tuple[str, ...] = ()
    #: The catalogue's opaque id for the item, kept so a later re-sync can find it again.
    external_key: str | None = None


@dataclass(frozen=True)
class ExternalScan:
    """What reading a catalogue found, including what it deliberately left out.

    `skipped` is a reason → count map and it is not decoration: a scan that silently dropped four
    hundred web snapshots and reported "37 documents" would look like a broken catalogue rather
    than a working filter. Every entry the adapter declined is counted under a reason a person can
    read.
    """

    #: The folder to register as a source root — the catalogue's storage directory.
    root: Path
    documents: tuple[ExternalDocument, ...] = ()
    skipped: dict[str, int] = field(default_factory=dict)
    #: Human-readable name of the catalogue, for the UI. A label, never an import.
    label: str = ""

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())


class CatalogueUnavailable(RuntimeError):
    """The catalogue could not be read, with a reason meant for a person.

    A missing or unreadable catalogue is an ordinary outcome — the user may not have that program
    installed, may have it open, may have pointed at the wrong folder — so it is an exception with
    a sentence in it, never a stack trace shown to the user.
    """


# --- recording what a catalogue said --------------------------------------------------------- #


def record_external(session: Session, records: Iterable[ExternalDocument], *, source: str) -> int:
    """Upsert one row per record, keyed by `(source, path_key)`. Returns rows written.

    Upsert rather than insert because re-importing the same library is the normal way to pick up
    edits made in the catalogue since last time — the second import should correct the first, not
    collide with it or double it.
    """
    written = 0
    for record in records:
        key = pathkey(record.path)
        row = session.execute(
            select(ExternalMetadata).where(
                ExternalMetadata.source == source, ExternalMetadata.path_key == key
            )
        ).scalar_one_or_none()
        if row is None:
            row = ExternalMetadata(source=source, path_key=key)
            session.add(row)
        row.external_key = record.external_key
        row.title = record.title
        row.authors = record.authors
        row.year = record.year
        row.doi = record.doi
        row.item_type = record.item_type
        row.collections_json = json.dumps(list(record.collections)) if record.collections else None
        written += 1
    session.flush()
    return written


def external_for_path(session: Session, path: Path | str) -> ExternalMetadata | None:
    """The catalogue's record for a file, if any catalogue described it."""
    key = pathkey(path)
    return (
        session.execute(select(ExternalMetadata).where(ExternalMetadata.path_key == key))
        .scalars()
        .first()
    )


# --- applying it to the documents that exist ------------------------------------------------- #

#: The fields a catalogue can supply, in the order the report lists them.
_FIELDS = ("title", "authors", "year", "doi")


@dataclass(frozen=True)
class MetadataApplication:
    """What `apply_external_metadata` did. `filled` counts documents, not fields."""

    considered: int
    filled: int
    fields: dict[str, int] = field(default_factory=dict)


def apply_external_metadata(
    session: Session,
    *,
    document_ids: Sequence[str] | None = None,
    overwrite: bool = True,
) -> MetadataApplication:
    """Copy the catalogue's metadata onto the documents whose source file it describes.

    `overwrite=True` by default, and deliberately: the slot being written is
    `Document.title`/`authors`/`year`/`doi`, which holds *the machine's best answer* (ADR-013 keeps
    the user's own edits in `DocumentMeta`, which this never touches). A curated catalogue entry is
    a better machine answer than an extracted guess, so it should win — that is the entire reason
    to import it. Pass `overwrite=False` for a fill-the-blanks pass.

    Matching is by `pathkey(Document.source_original)`, the same normalisation the registry uses,
    so a drive-letter or separator difference does not read as a different file.

    Idempotent: running it twice writes the same values. Safe on an empty library — it reads
    nothing and reports zeroes.
    """
    rows = session.execute(select(ExternalMetadata)).scalars().all()
    if not rows:
        return MetadataApplication(considered=0, filled=0)
    by_key = {r.path_key: r for r in rows}

    stmt = select(DBDocument)
    if document_ids is not None:
        if not document_ids:
            return MetadataApplication(considered=0, filled=0)
        stmt = stmt.where(DBDocument.id.in_(list(document_ids)))
    documents = session.execute(stmt).scalars().all()

    counts: dict[str, int] = {}
    filled = 0
    considered = 0
    for doc in documents:
        record = by_key.get(pathkey(str(doc.source_original or "")))
        if record is None:
            continue
        considered += 1
        touched = False
        for name in _FIELDS:
            value = getattr(record, name)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            current = getattr(doc, name)
            if not overwrite and current:
                continue
            if current == value:
                continue
            setattr(doc, name, value)
            counts[name] = counts.get(name, 0) + 1
            touched = True
        if touched:
            filled += 1
    session.flush()
    if filled:
        log.info("external_metadata_applied", documents=filled, fields=counts)
    return MetadataApplication(considered=considered, filled=filled, fields=counts)
