"""Document queries plus the browse-time write paths.

Listing/detail reads, the ADR-013 metadata overrides (+ reveal-in-file-manager), and the
ADR-014 safe delete (source file to the Recycle Bin, then row/meta/chunks)."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select

from doc_assistant.db.models import Document, DocumentMeta, Folder, Tag
from doc_assistant.db.session import session_scope
from doc_assistant.library.models import DocumentDetails, DocumentSummary

log = structlog.get_logger(__name__)


# ============================================================
# Query functions
# ============================================================


def list_documents(
    health: str | None = None,
    format: str | None = None,
    tag: str | None = None,
    folder_id: str | None = None,
) -> list[DocumentSummary]:
    """Return documents matching the filters.

    All filters are optional. None means no filter on that dimension.
    Filters are combined with AND.

    ``folder_id`` filters by folder **id**, not name: ``uq_folder_name_parent`` does not bite at
    the root level (SQLite treats NULL parents as distinct), so a name is not a key (ADR-025 F1).
    """
    with session_scope() as session:
        query = select(Document).where(Document.is_archived.is_(False))

        if health:
            query = query.where(Document.extraction_health == health)
        if format:
            query = query.where(Document.format == format)
        if tag:
            query = query.join(Document.tags).where(Tag.name == tag)
        if folder_id:
            query = query.join(Document.folders).where(Folder.id == folder_id)

        query = query.order_by(Document.filename)
        docs = session.execute(query).scalars().unique().all()

        # Batch-load user overrides once, then merge effective = override ?? auto (ADR-013).
        overrides = {m.document_id: m for m in session.execute(select(DocumentMeta)).scalars()}

        summaries: list[DocumentSummary] = []
        for d in docs:
            m = overrides.get(d.id)
            title = (m.title_override if m and m.title_override is not None else None) or d.title
            authors = (
                m.authors_override if m and m.authors_override is not None else None
            ) or d.authors
            year = (m.year_override if m and m.year_override is not None else None) or d.year
            customized = m is not None and any(
                v is not None for v in (m.title_override, m.authors_override, m.year_override)
            )
            summaries.append(
                DocumentSummary(
                    id=d.id,
                    filename=d.filename,
                    title=title,
                    format=d.format,
                    health=d.extraction_health,
                    chunk_count=d.chunk_count,
                    page_count=d.page_count,
                    authors=authors,
                    year=year,
                    customized=customized,
                    folders=[f.name for f in d.folders],
                    folder_ids=[f.id for f in d.folders],
                    tags=[t.name for t in d.tags],
                    keywords=[k.name for k in d.keywords],
                    added_at=d.added_at,
                )
            )
        return summaries


def count_documents() -> int:
    """How many live (non-archived) documents the library holds.

    A ``COUNT`` rather than ``len(list_documents())``: the caller (the first-run setup view) wants
    one number, and building a ``DocumentSummary`` per document — plus loading every override row —
    to discard all of it would scale the cost with the corpus for nothing (KI-18 discipline).
    """
    with session_scope() as session:
        query = select(func.count()).select_from(Document).where(Document.is_archived.is_(False))
        return int(session.execute(query).scalar_one())


def document_years(document_ids: list[str]) -> dict[str, int]:
    """Publication year per document (a **scoped** ``SELECT id, year``) — for the ADR-027 D3 source
    strip. Docs with no year are omitted. Scoped to the retrieved sources, not the whole-corpus
    ``concept_skeleton.load_doc_years`` (KI-18 discipline: a per-turn read must not scale with the
    corpus). Returns ``{}`` for an empty request."""
    if not document_ids:
        return {}
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    years: dict[str, int] = {}
    with session_scope() as session:
        stmt = select(Document.id, Document.year).where(Document.id.in_(document_ids))
        for doc_id, year in session.execute(stmt):
            if doc_id is not None and year is not None:
                years[str(doc_id)] = int(year)
    return years


def get_document_details(doc_id: str) -> DocumentDetails | None:
    """Return everything we know about a single document."""
    with session_scope() as session:
        doc = session.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
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


# ============================================================
# Metadata overrides + reveal (ADR-013 — first browse-time write path)
# ============================================================


def _dedup_override(value: str | None, auto: str | None) -> str | None:
    """The override to store for a text field: None if blank or equal to the auto default."""
    stripped = (value or "").strip()
    if not stripped or stripped == (auto or "").strip():
        return None
    return stripped


def set_document_meta(
    document_id: str,
    *,
    title: str | None = None,
    authors: str | None = None,
    year: int | None = None,
) -> None:
    """Replace a document's user metadata overrides with the given *effective* values (ADR-013).

    The editor sends the whole small metadata form, so this is a replace, not a partial patch:
    each field's override is stored only when it is non-blank **and** differs from the
    auto-extracted default (so re-saving an untouched field creates no override). When nothing
    differs from the defaults the sidecar row is dropped (the document is no longer "customized").
    """
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return
        t_over = _dedup_override(title, doc.title)
        a_over = _dedup_override(authors, doc.authors)
        y_over = year if (year is not None and year != doc.year) else None

        meta = session.get(DocumentMeta, document_id)
        if t_over is None and a_over is None and y_over is None:
            if meta is not None:
                session.delete(meta)
            return
        if meta is None:
            meta = DocumentMeta(document_id=document_id)
            session.add(meta)
        meta.title_override = t_over
        meta.authors_override = a_over
        meta.year_override = y_over


def clear_document_meta(document_id: str) -> None:
    """Reset a document to its auto-extracted defaults by deleting its override row."""
    with session_scope() as session:
        meta = session.get(DocumentMeta, document_id)
        if meta is not None:
            session.delete(meta)


def resolve_source_path(source_original: str, filename: str) -> Path | None:
    """The on-disk source file, or None if it can't be located.

    ``source_original`` may be stored resolved or not; fall back to ``DOCS_PATH / filename``
    (mirrors the extract-* scripts' resolver). Returns None when the file has moved/been deleted.
    """
    p = Path(source_original)
    if p.exists():
        return p
    from doc_assistant.config import DOCS_PATH

    alt = Path(DOCS_PATH) / filename
    return alt if alt.exists() else None


def reveal_document_source(document_id: str) -> bool:
    """Open the OS file manager with the document's source file selected (local desktop action).

    Returns False if the document is unknown or its source file can't be located. The reveal runs
    on whatever host the API runs on — always the user's machine (local-first). ADR-013.
    """
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return False
        path = resolve_source_path(doc.source_original, doc.filename)
    if path is None:
        log.warning("reveal_source_not_found", document_id=document_id)
        return False
    _reveal_in_file_manager(path)
    return True


def _reveal_in_file_manager(path: Path) -> None:
    """Reveal ``path`` in the OS file manager, file selected. List-form args, never a shell."""
    if sys.platform == "win32":
        # explorer selects the file inside its folder; it exits non-zero even on success.
        subprocess.run(["explorer", f"/select,{path}"], check=False)  # nosec B603 B607
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)  # nosec B603 B607
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)  # nosec B603 B607


@dataclass
class DeleteResult:
    """Outcome of a document delete (ADR-014)."""

    filename: str
    trashed_file: bool  # source file moved to the Recycle Bin (False = it was already gone)
    chunks_removed: int  # chunks dropped from the live search index


def delete_document(document_id: str, chroma_db: Any) -> DeleteResult | None:
    """Safe-delete a document: source file → Recycle Bin, then drop its DB row + index chunks.

    Returns None if the document is unknown. The source file is moved to the OS Recycle Bin FIRST
    (recoverable); only on success (or when the file is already gone) does the removal proceed, so
    a locked/undeletable file leaves the library entry intact rather than orphaning a still-indexed
    file on disk. Removal then: deletes the ``Document`` row (FK-cascades citations / parts /
    similarities, and since ADR-026 the ``DocumentMeta`` override too), the doc's chunks from the
    live Chroma store, its figure dir, and its cached ``.md``. ADR-014.
    """
    from send2trash import send2trash

    from doc_assistant.ingest.cleanup import cleanup_orphan_figures

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        filename = doc.filename
        doc_hash_val = doc.doc_hash
        source_original = doc.source_original
        source_cache = doc.source_cache

    # 1. Recycle the source file first (recoverable). A trash failure aborts the whole delete.
    path = resolve_source_path(source_original, filename)
    trashed = False
    if path is not None:
        try:
            send2trash(str(path))
            trashed = True
        except Exception as e:
            log.warning("delete_trash_failed", document_id=document_id, error=str(e))
            raise RuntimeError(f"could not move {filename} to the Recycle Bin") from e

    # 2. Drop the DB row (+ cascades). The override delete is redundant since ADR-026 gave
    # document_meta a real FK, and is kept only so this path reads as the complete story.
    with session_scope() as session:
        meta = session.get(DocumentMeta, document_id)
        if meta is not None:
            session.delete(meta)
        doc = session.get(Document, document_id)
        if doc is not None:
            session.delete(doc)

    # 3. Remove the doc's chunks from the live search index (count for the caller).
    chunks_removed = 0
    try:
        found = chroma_db.get(where={"doc_hash": doc_hash_val}, include=[])
        ids = list(found.get("ids", []))
        chunks_removed = len(ids)
        if ids:
            chroma_db.delete(ids=ids)
    except Exception as e:
        log.warning("delete_chunks_failed", document_id=document_id, error=str(e))

    # 4. On-disk sidecars: figure dir (by hash) + the cached markdown.
    cleanup_orphan_figures([doc_hash_val])
    if source_cache:
        cache_path = Path(source_cache)
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError as e:
                log.warning("delete_cache_failed", file=cache_path.name, error=str(e))

    log.info(
        "document_deleted",
        document_id=document_id,
        trashed_file=trashed,
        chunks_removed=chunks_removed,
    )
    return DeleteResult(filename=filename, trashed_file=trashed, chunks_removed=chunks_removed)
