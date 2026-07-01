"""SQLite + Chroma store helpers for ingest — the data-access layer.

Which hashes are indexed (Chroma) or rowed (SQLite), resolving/committing the
Document row, and materialising described figures into retrieval chunks. No chunking
or orchestration logic lives here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_chroma import Chroma
from sqlalchemy import select

from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.models import Figure, IngestionEvent
from doc_assistant.db.session import session_scope

from .figures import figure_chunk_text


def get_indexed_hashes(db: Chroma) -> set[str]:
    data = db.get(include=["metadatas"])
    return {m.get("doc_hash") for m in data["metadatas"] if m and m.get("doc_hash")}


def get_document_row_hashes() -> set[str]:
    """The doc_hashes that currently have a committed Document row in SQLite.

    The SQLite-side counterpart to ``get_indexed_hashes`` (the Chroma side). The
    dedup gate in ``main`` subtracts the Chroma intersection from this set to find
    *inverse orphans* — chunks present in both stores with no Document row — the
    one partial-write shape the intersection gate alone cannot self-heal (F1).
    """
    with session_scope() as session:
        rows = session.execute(select(DBDocument.doc_hash)).scalars().all()
    return {str(h) for h in rows if h}


def _existing_document_id(doc_hash: str) -> str | None:
    """The id of the Document row already recorded for ``doc_hash``, if any.

    Read-only. ``process_one_document`` calls this to resolve the id a re-ingest
    must reuse (so the document's figures and other id-keyed sidecars stay linked)
    *before* the Chroma writes — without committing a row. The row is written last,
    only if both vector writes land (F1, see ``process_one_document``).
    """
    with session_scope() as session:
        existing = session.execute(
            select(DBDocument.id).where(DBDocument.doc_hash == doc_hash)
        ).scalar_one_or_none()
        return str(existing) if existing is not None else None


def upsert_document_in_sqlite(
    document_id: str,
    filename: str,
    source_original: str,
    source_cache: str | None,
    doc_hash: str,
    format: str,
    extractor_used: str,
    chunk_count: int,
    page_count: int | None = None,
    extraction_health: str | None = None,
) -> str:
    """Create or update the Document row for ``doc_hash``. Returns its id.

    ``document_id`` is resolved by the caller (``_existing_document_id`` for a
    re-ingest, a fresh UUID for a new document) and is the same id already stamped
    into the chunk metadata, so the row and its chunks share one identity. Called
    **after** the Chroma writes succeed, so this commit is the last step of a
    document's ingest — a vector-write failure aborts before any row is written.

    If a row with ``doc_hash`` exists, update it and log a re-ingestion event;
    otherwise create a new row with ``document_id`` as its primary key.
    """
    with session_scope() as session:
        existing = session.execute(
            select(DBDocument).where(DBDocument.doc_hash == doc_hash)
        ).scalar_one_or_none()

        if existing:
            # Re-ingestion of an existing document
            existing.chunk_count = chunk_count
            existing.extractor_used = extractor_used
            existing.extracted_at = datetime.now(timezone.utc)
            if page_count is not None:
                existing.page_count = page_count

            event = IngestionEvent(
                document_id=existing.id,
                event_type="reextract",
                extractor=extractor_used,
                chunks_produced=chunk_count,
                health_status=extraction_health,
            )
            session.add(event)
            return str(existing.id)
        else:
            # New document — keyed by the pre-resolved id (matches the chunk metadata).
            document = DBDocument(
                id=document_id,
                filename=filename,
                source_original=source_original,
                source_cache=source_cache,
                doc_hash=doc_hash,
                format=format,
                extractor_used=extractor_used,
                extraction_health=extraction_health,
                chunk_count=chunk_count,
                page_count=page_count,
                extracted_at=datetime.now(timezone.utc),
            )
            session.add(document)

            event = IngestionEvent(
                document_id=document_id,
                event_type="extract",
                extractor=extractor_used,
                chunks_produced=chunk_count,
                health_status=extraction_health,
            )
            session.add(event)
            return document_id


def figure_units(document_id: str) -> list[tuple[str, int, str]]:
    """Return ``(chunk_text, page, figure_id)`` for a doc's *described* figures.

    Feature 4c: a figure becomes a retrievable chunk only once a VLM description
    exists (the caption alone is already in the markdown text chunks). The
    ``Figure`` sidecar is written by ``scripts/describe_figures``; ingest — the
    one component allowed to write the chunk store — materialises it here, the
    same separation tables use (4a writes the markdown, ingest reads it).
    """
    with session_scope() as session:
        rows = session.execute(
            select(Figure.id, Figure.page, Figure.caption, Figure.vlm_description)
            .where(Figure.document_id == document_id, Figure.vlm_description.is_not(None))
            .order_by(Figure.page, Figure.id)
        ).all()
    units: list[tuple[str, int, str]] = []
    for fig_id, page, caption, vlm in rows:
        text = figure_chunk_text(caption, vlm or "")
        if text.strip():
            units.append((text, int(page), str(fig_id)))
    return units
