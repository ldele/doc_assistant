"""Plain dataclasses returned across the library boundary.

No SQLAlchemy models cross this line: callers get inert data, free of
session-lifecycle concerns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ============================================================
# Data classes (returned to UI)
# ============================================================


@dataclass
class DocumentSummary:
    """One row in the library list.

    ``title``/``authors``/``year`` are the **effective** values (user override ?? auto-extracted);
    ``customized`` is True when a ``DocumentMeta`` override is in force for any of them (ADR-013).
    """

    id: str
    filename: str
    title: str | None
    format: str
    health: str | None
    chunk_count: int | None
    page_count: int | None
    authors: str | None = None
    year: int | None = None
    customized: bool = False
    folders: list[str] = field(default_factory=list)
    folder_ids: list[str] = field(default_factory=list)
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
    ingestion_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LibrarySummary:
    """High-level counts for the whole library."""

    total_documents: int
    total_chunks: int
    by_health: dict[str, int]
    by_format: dict[str, int]
