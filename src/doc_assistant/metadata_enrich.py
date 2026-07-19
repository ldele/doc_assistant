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
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from doc_assistant.knowledge.keywords import load_document_texts
from doc_assistant.metadata_extractor import DocMetadata, extract_metadata

log = structlog.get_logger(__name__)

# The four columns this runner may fill, in report order.
_FIELDS: tuple[str, ...] = ("title", "authors", "year", "doi")


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


def _persist_metadata(document_id: str, meta: DocMetadata, *, force: bool) -> tuple[str, ...]:
    """Write the extracted fields onto one ``Document``. Idempotent per column.

    A field is written only when the extractor found a value **and** the column is
    currently ``NULL`` (or ``force``). Returns the columns actually written.
    """
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    written: list[str] = []
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return ()
        for field in _FIELDS:
            value = getattr(meta, field)
            if value is None:
                continue
            if getattr(doc, field) is not None and not force:
                continue
            setattr(doc, field, value)
            written.append(field)
    return tuple(written)


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
