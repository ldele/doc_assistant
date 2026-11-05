"""Populate ``Document.title``/``authors``/``year``/``doi`` from cached markdown.

Wires the (previously unwired) deterministic ``metadata_extractor.extract_metadata``
onto the ``Document`` registry. The extractor pulls the four bibliographic fields from
each document's header region with academic-paper heuristics; this module is the runner
that persists them, so the library grid shows real titles instead of filenames and
search/navigation gains author text.

Enrichment-Layer discipline (root ``CLAUDE.md``): reads each document's cached markdown
and writes **only** the four metadata columns on ``Document`` — never the chunk store.
Deterministic and free (no LLM). Idempotent — a column is written only when it is
currently ``NULL``, unless ``force`` overwrites; a later manual edit is therefore never
clobbered by a re-run. ``apply=False`` writes nothing (dry run), so extraction quality
can be reviewed before touching the DB.

``fill_missing_metadata`` is the same write for the app's own index run (``ingest.main``): only the
documents a run indexed, only their blank columns, and only after a catalogue's answer has been
applied (ADR-049 D3: the extractor is the lowest of the three slots).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from doc_assistant.knowledge.keywords import load_document_texts
from doc_assistant.metadata_extractor import DocMetadata, extract_metadata

if TYPE_CHECKING:
    from doc_assistant.db.models import Document

log = structlog.get_logger(__name__)

# The four columns this runner may fill, in report order.
_FIELDS: tuple[str, ...] = ("title", "authors", "year", "doi")

#: How many documents' cached texts `fill_missing_metadata` holds at once. Structural, not tuned:
#: a text is read whole, and a re-read of the whole library hands the step every document in it
#: (~10,000 under the robustness contract), which must not all sit in memory together.
_FILL_BATCH = 100


@dataclass(frozen=True)
class DocMetadataResult:
    """Per-document extraction + what was persisted (for reporting)."""

    document_id: str
    filename: str
    metadata: DocMetadata
    written_fields: tuple[str, ...]


@dataclass(frozen=True)
class MetadataEnrichmentResult:
    """Whole-run result across the corpus."""

    docs: list[DocMetadataResult]
    n_documents: int
    n_title: int
    n_authors: int
    n_year: int
    n_doi: int
    total_fields_written: int


def _is_blank(value: object) -> bool:
    """``NULL`` or an all-whitespace string: the catalogue's fill-the-blanks rule
    (`adapters.catalogue.apply_external_metadata`) and the library summary's *missing*."""
    return value is None or (isinstance(value, str) and not value.strip())


def _fill_fields(doc: Document, meta: DocMetadata, *, force: bool) -> tuple[str, ...]:
    """Copy the extracted fields onto a loaded ``Document`` row. Returns the columns written.

    A field is written only when the extractor found a value **and** the column is blank
    (or ``force``).
    """
    written: list[str] = []
    for name in _FIELDS:
        value = getattr(meta, name)
        if _is_blank(value):
            continue
        if not force and not _is_blank(getattr(doc, name)):
            continue
        setattr(doc, name, value)
        written.append(name)
    return tuple(written)


def _persist_metadata(document_id: str, meta: DocMetadata, *, force: bool) -> tuple[str, ...]:
    """Write the extracted fields onto one ``Document``. Idempotent per column (`_fill_fields`)."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return ()
        return _fill_fields(doc, meta, force=force)


def enrich_metadata(
    *,
    apply: bool,
    force: bool = False,
    document_id: str | None = None,
) -> MetadataEnrichmentResult:
    """Extract bibliographic metadata from cached markdown and optionally persist it.

    Runs ``extract_metadata`` over every non-archived document's cached markdown (or the
    single ``document_id``). Deterministic and free. With ``apply`` each document's found
    fields are written per :func:`_persist_metadata` (only-if-``NULL`` unless ``force``);
    ``apply=False`` is a dry run that writes nothing.
    """
    ids = [document_id] if document_id is not None else None
    corpus = load_document_texts(ids)

    docs: list[DocMetadataResult] = []
    n_title = n_authors = n_year = n_doi = 0
    total_written = 0
    for doc_id, filename, markdown in corpus:
        meta = extract_metadata(markdown, filename=filename)
        if meta.title:
            n_title += 1
        if meta.authors:
            n_authors += 1
        if meta.year is not None:
            n_year += 1
        if meta.doi is not None:
            n_doi += 1

        written: tuple[str, ...] = ()
        if apply:
            written = _persist_metadata(doc_id, meta, force=force)
            total_written += len(written)
        docs.append(
            DocMetadataResult(
                document_id=doc_id,
                filename=filename,
                metadata=meta,
                written_fields=written,
            )
        )

    if apply:
        log.info(
            "metadata_enriched",
            documents=len(docs),
            title=n_title,
            authors=n_authors,
            year=n_year,
            doi=n_doi,
            fields_written=total_written,
        )
    return MetadataEnrichmentResult(
        docs=docs,
        n_documents=len(docs),
        n_title=n_title,
        n_authors=n_authors,
        n_year=n_year,
        n_doi=n_doi,
        total_fields_written=total_written,
    )


@dataclass(frozen=True)
class MetadataFill:
    """What `fill_missing_metadata` did. ``filled`` counts documents, not fields, as the
    catalogue's `MetadataApplication` does; ``fields`` counts per column."""

    considered: int
    filled: int
    fields: dict[str, int] = field(default_factory=dict)


def fill_missing_metadata(document_ids: Iterable[str]) -> MetadataFill:
    """Fill these documents' blank title, authors, year and DOI from their own cached text.

    The index run's step (`ingest._fill_missing_metadata`). A column that holds anything (an
    earlier extraction, a catalogue's answer) is never replaced, and the user's own edits live in
    ``DocumentMeta``, which this neither reads nor writes and which wins at read time (ADR-013).
    A document with no blank column is not read at all; one whose text yields nothing keeps its
    blanks; an archived document or one with no cached text is skipped (`load_document_texts`).

    Works in batches of `_FILL_BATCH` documents: one read of the four columns, one read of the
    texts that still have a blank, one write. Deterministic, no LLM, and nothing but these four
    columns on ``Document`` (Enrichment-Layer Pattern).
    """
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    ids = sorted(set(document_ids))
    considered = filled = 0
    counts: dict[str, int] = {}
    for start in range(0, len(ids), _FILL_BATCH):
        batch = ids[start : start + _FILL_BATCH]
        with session_scope() as session:
            rows = session.execute(
                select(
                    Document.id, Document.title, Document.authors, Document.year, Document.doi
                ).where(Document.id.in_(batch))
            ).all()
        blank = [str(row[0]) for row in rows if any(_is_blank(v) for v in row[1:])]
        if not blank:
            continue
        extracted = [
            (doc_id, extract_metadata(text, filename=filename))
            for doc_id, filename, text in load_document_texts(blank)
        ]
        with session_scope() as session:
            for doc_id, meta in extracted:
                doc = session.get(Document, doc_id)
                if doc is None:  # deleted between the two reads
                    continue
                considered += 1
                written = _fill_fields(doc, meta, force=False)
                if written:
                    filled += 1
                    for name in written:
                        counts[name] = counts.get(name, 0) + 1
    return MetadataFill(considered=considered, filled=filled, fields=counts)
