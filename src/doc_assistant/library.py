"""Data access layer for the document library.

This module provides a stable Python API over the SQLite store. The UI
calls into this module rather than touching SQLAlchemy directly, so
swapping the UI or the storage backend doesn't require coordinated
changes.

All functions return plain dataclasses, not SQLAlchemy models. This
keeps the UI layer free of session lifecycle concerns.
"""
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select, func

from doc_assistant.db.models import (
    Document, IngestionEvent, Folder, Tag, Keyword
)
from doc_assistant.db.session import session_scope


# ============================================================
# Data classes (returned to UI)
# ============================================================

@dataclass
class DocumentSummary:
    """One row in the library list."""
    id: str
    filename: str
    title: str | None
    format: str
    health: str | None
    chunk_count: int | None
    page_count: int | None
    folders: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    added_at: datetime | None = None


@dataclass
class DocumentDetails:
    """Full details for one document."""
    id: str
    filename: str
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    notes: str | None
    format: str
    doc_hash: str
    source_original: str
    source_cache: str | None
    extractor_used: str | None
    extraction_health: str | None
    chunk_count: int | None
    page_count: int | None
    extracted_at: datetime | None
    added_at: datetime | None
    updated_at: datetime | None
    folders: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    ingestion_history: list[dict] = field(default_factory=list)


@dataclass
class LibrarySummary:
    """High-level counts for the whole library."""
    total_documents: int
    total_chunks: int
    by_health: dict[str, int]
    by_format: dict[str, int]


# ============================================================
# Query functions
# ============================================================

def list_documents(
    health: str | None = None,
    format: str | None = None,
    tag: str | None = None,
    folder: str | None = None,
) -> list[DocumentSummary]:
    """Return documents matching the filters.

    All filters are optional. None means no filter on that dimension.
    Filters are combined with AND.
    """
    with session_scope() as session:
        query = select(Document).where(Document.is_archived == False)

        if health:
            query = query.where(Document.extraction_health == health)
        if format:
            query = query.where(Document.format == format)
        if tag:
            query = query.join(Document.tags).where(Tag.name == tag)
        if folder:
            query = query.join(Document.folders).where(Folder.name == folder)

        query = query.order_by(Document.filename)
        docs = session.execute(query).scalars().unique().all()

        return [
            DocumentSummary(
                id=d.id,
                filename=d.filename,
                title=d.title,
                format=d.format,
                health=d.extraction_health,
                chunk_count=d.chunk_count,
                page_count=d.page_count,
                folders=[f.name for f in d.folders],
                tags=[t.name for t in d.tags],
                keywords=[k.name for k in d.keywords],
                added_at=d.added_at,
            )
            for d in docs
        ]


def get_document_details(doc_id: str) -> DocumentDetails | None:
    """Return everything we know about a single document."""
    with session_scope() as session:
        doc = session.execute(
            select(Document).where(Document.id == doc_id)
        ).scalar_one_or_none()
        if not doc:
            return None

        history = [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "extractor": e.extractor,
                "chunks_produced": e.chunks_produced,
                "health_status": e.health_status,
                "notes": e.notes,
            }
            for e in doc.ingestion_events
        ]

        return DocumentDetails(
            id=doc.id,
            filename=doc.filename,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            notes=doc.notes,
            format=doc.format,
            doc_hash=doc.doc_hash,
            source_original=doc.source_original,
            source_cache=doc.source_cache,
            extractor_used=doc.extractor_used,
            extraction_health=doc.extraction_health,
            chunk_count=doc.chunk_count,
            page_count=doc.page_count,
            extracted_at=doc.extracted_at,
            added_at=doc.added_at,
            updated_at=doc.updated_at,
            folders=[f.name for f in doc.folders],
            tags=[t.name for t in doc.tags],
            keywords=[k.name for k in doc.keywords],
            ingestion_history=history,
        )


def library_summary() -> LibrarySummary:
    """Return high-level counts for the library."""
    with session_scope() as session:
        total_docs = session.execute(
            select(func.count(Document.id)).where(Document.is_archived == False)
        ).scalar() or 0

        total_chunks_query = select(func.coalesce(func.sum(Document.chunk_count), 0))
        total_chunks_query = total_chunks_query.where(Document.is_archived == False)
        total_chunks = session.execute(total_chunks_query).scalar() or 0

        by_health = Counter()
        by_format = Counter()
        for doc in session.execute(
            select(Document).where(Document.is_archived == False)
        ).scalars():
            by_health[doc.extraction_health or "unknown"] += 1
            by_format[doc.format] += 1

        return LibrarySummary(
            total_documents=total_docs,
            total_chunks=int(total_chunks),
            by_health=dict(by_health),
            by_format=dict(by_format),
        )


def find_document_by_short_id(short_id: str) -> str | None:
    """Find a document by a UUID prefix (first 8+ chars).

    Returns the full UUID if exactly one match, else None.
    """
    with session_scope() as session:
        matches = session.execute(
            select(Document.id).where(Document.id.like(f"{short_id}%"))
        ).scalars().all()
        if len(matches) == 1:
            return matches[0]
        return None