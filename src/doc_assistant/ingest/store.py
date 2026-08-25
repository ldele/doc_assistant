"""SQLite + Chroma store helpers for ingest — the data-access layer.

Which hashes are indexed (Chroma) or rowed (SQLite), resolving/committing the
Document row, and materialising described figures into retrieval chunks. No chunking
or orchestration logic lives here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import structlog
from langchain_chroma import Chroma
from sqlalchemy import select

from doc_assistant.chroma_read import get_all
from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.models import Figure, IngestionEvent
from doc_assistant.db.session import session_scope

from .figures import figure_chunk_text

log = structlog.get_logger(__name__)


def get_indexed_hashes(db: Chroma) -> set[str]:
    data = get_all(db, include=["metadatas"])
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


def _existing_document_id(doc_hash: str, source_original: str | Path | None = None) -> str | None:
    """The id of the Document row this content already has, if any (ADR-047).

    Read-only. ``process_one_document`` calls this to resolve the id a re-ingest must reuse — so
    the document's figures and other id-keyed sidecars stay linked — *before* the Chroma writes,
    without committing a row. The row is written last, only if both vector writes land (F1).

    Two keys, tried in order:

    1. **``doc_hash``** — the extracted text. Exact, and the answer whenever extraction is stable.
    2. **``source_original``** — the file it came from. This is the ADR-047 fallback, and it is
       what makes a document survive its own re-extraction: every extractor improvement changes
       the text, which changes the hash, which under (1) alone minted a fresh id and cut loose
       everything keyed to the old one. Measured on the live library before this existed: an
       extractor change would have orphaned **4,123 rows**, including 881 figure descriptions and
       19 rows nobody can regenerate (18 folder assignments and a metadata override the user
       typed).

    The path comparison is normalised (case and separators, via the registry's ``pathkey``) rather
    than a raw string equality, because the same file reaches this code both resolved and
    unresolved depending on the caller. The exact match is tried first so the common case stays a
    single indexed lookup.

    **The consequence, stated because it is a real trade:** a *different* document written to the
    same path inherits the previous one's id and its sidecars. For a library where the path is the
    document's address that is the intended reading — see ADR-047, which records why.
    """
    with session_scope() as session:
        by_hash = session.execute(
            select(DBDocument.id).where(DBDocument.doc_hash == doc_hash)
        ).scalar_one_or_none()
        if by_hash is not None:
            return str(by_hash)
        if source_original is None:
            return None

        exact = session.execute(
            select(DBDocument.id).where(DBDocument.source_original == str(source_original))
        ).scalar_one_or_none()
        if exact is not None:
            log.info("document_id_reused_by_path", source=str(source_original), match="exact")
            return str(exact)

        from doc_assistant.ingest.registry import pathkey

        wanted = pathkey(source_original)
        rows = session.execute(select(DBDocument.id, DBDocument.source_original)).all()
        for row_id, row_source in rows:
            if row_source and pathkey(row_source) == wanted:
                log.info(
                    "document_id_reused_by_path", source=str(source_original), match="normalised"
                )
                return str(row_id)
        return None


def repoint_figures(document_id: str, new_hash: str) -> int:
    """Move a document's figures onto its new ``doc_hash`` after a re-extraction (ADR-047).

    Figures are the one sidecar keyed *both* ways — ``document_id`` **and** ``doc_hash``, plus
    on-disk PNGs under ``FIGURE_DIR/{doc_hash}/`` whose absolute paths are stored in
    ``figures.image_path``. The id half survives a re-extraction on its own; the hash half does
    not, so without this the rows would point at a hash no document has and a directory the next
    figure pass would not write to.

    They are worth carrying rather than regenerating: a figure is a crop of **the PDF's own
    page**, so changing the *text* extractor cannot invalidate it — and regenerating means paying
    for the VLM descriptions again (881 of them on the current library).

    Three updates, and all three or none: rename the directory, repoint ``image_path``, set
    ``doc_hash``. Returns how many rows moved. A missing directory is not an error — figure
    detection may simply never have run for this document.
    """
    from doc_assistant.ingest.figures import figure_dir

    with session_scope() as session:
        rows = (
            session.execute(select(Figure).where(Figure.document_id == document_id))
            .scalars()
            .all()
        )
        old_hashes = {str(r.doc_hash) for r in rows if r.doc_hash and r.doc_hash != new_hash}
        if not rows or not old_hashes:
            return 0

        for old_hash in old_hashes:
            src_dir, dest_dir = figure_dir(old_hash), figure_dir(new_hash)
            if src_dir.exists() and not dest_dir.exists():
                try:
                    dest_dir.parent.mkdir(parents=True, exist_ok=True)
                    src_dir.rename(dest_dir)
                except OSError as e:
                    # Leave the rows pointing at the old location rather than half-moving them:
                    # a stored `image_path` that resolves is worth more than a tidy hash.
                    log.warning("figure_dir_move_failed", document_id=document_id, error=str(e))
                    return 0

        moved = 0
        for row in rows:
            old_hash = str(row.doc_hash) if row.doc_hash else ""
            if old_hash and old_hash != new_hash:
                if row.image_path:
                    row.image_path = str(row.image_path).replace(old_hash, new_hash)
                row.doc_hash = new_hash
                moved += 1
        log.info("figures_repointed", document_id=document_id, count=moved)
        return moved


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

    Looked up by ``document_id`` — the id the caller already resolved — and **not** by
    ``doc_hash`` (ADR-047). Those were the same question until the identity fallback existed;
    they are not any more. A re-extraction reuses the id while the hash moves, so a hash lookup
    would miss the row, fall through to the insert, and collide on the primary key. Resolving
    identity is `_existing_document_id`'s job; this function's job is to write.

    If the row exists, update it and log a re-ingestion event; otherwise create it with
    ``document_id`` as its primary key.
    """
    with session_scope() as session:
        existing = session.execute(
            select(DBDocument).where(DBDocument.id == document_id)
        ).scalar_one_or_none()

        if existing:
            # Re-ingestion of an existing document
            existing.chunk_count = chunk_count
            existing.extractor_used = extractor_used
            existing.extracted_at = datetime.now(timezone.utc)
            # The text may have changed under the same identity — that is the ADR-047 case, and
            # the row has to record what it actually holds now.
            existing.doc_hash = doc_hash
            existing.source_original = source_original
            existing.source_cache = source_cache
            existing.extraction_health = extraction_health
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


def figure_captions(document_id: str) -> dict[str, str]:
    """``{figure_id: caption}`` for a document's captioned figures.

    Deliberately separate from :func:`figure_units` rather than widening its tuple: that
    signature is monkeypatched by ingest's write-ordering guard tests, and a caller that
    patches only ``figure_units`` should still work — it simply finds no caption here,
    which degrades a figure to a self-contained chunk (the pre-2026-08-09 behaviour)
    instead of raising.

    The caption is what carries the figure's printed label, and the label is the only
    thing that can find the passage citing it.
    """
    with session_scope() as session:
        rows = session.execute(
            select(Figure.id, Figure.caption).where(
                Figure.document_id == document_id, Figure.caption.is_not(None)
            )
        ).all()
    return {str(fig_id): caption for fig_id, caption in rows if caption and caption.strip()}
