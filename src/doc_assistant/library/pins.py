"""Pinned-source matching and removal (demo-corpus cleanup; rides ADR-014).

Content-hash matching so a pinned demo document is identified by what it is, not where it sits."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.library.documents import delete_document
from doc_assistant.library.models import LibrarySummary

log = structlog.get_logger(__name__)


# ============================================================
# Pinned-source removal (demo-corpus cleanup; rides ADR-014)
# ============================================================


@dataclass(frozen=True)
class SourcePin:
    """A manifest-pinned source file: display name + exact content identity."""

    filename: str
    sha256: str
    size_bytes: int


@dataclass
class SourceMatch:
    """A file on disk whose exact bytes match a pin, plus its library row (if any)."""

    path: Path
    pin: SourcePin
    document_id: str | None  # the ingested row; None = file never ingested (or ambiguous)
    ambiguous: bool = False  # >1 library row shares the filename — never auto-delete


@dataclass
class SourceRemoval:
    """Outcome for one matched file."""

    filename: str  # on-disk name
    deleted_document: bool  # a library row (+ chunks + sidecars) was removed
    trashed_file: bool
    chunks_removed: int
    skipped_ambiguous: bool = False
    failed: bool = False  # trash refused (e.g. file locked) — everything left intact


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def match_pinned_sources(pins: Sequence[SourcePin], sources_dir: Path) -> list[SourceMatch]:
    """Files under ``sources_dir`` whose exact bytes match a pin, with their library rows.

    File matching is by **content** (size fast-path, then SHA-256), never by name, so a
    renamed pinned file is still found; only size-candidate files are ever hashed, so a
    large unrelated corpus costs stat calls, not reads. The library row is then looked up
    by the on-disk name against ``Document.filename`` — content can't bridge that hop
    (``doc_hash`` hashes extracted text, not file bytes) — so a file renamed *after*
    ingest matches as file-only and its stale row is left for the ingest orphan cleanup.
    Several rows sharing one filename is marked ambiguous and never auto-deleted.
    Missing/empty dir → [].
    """
    by_size: dict[int, list[SourcePin]] = {}
    for pin in pins:
        by_size.setdefault(pin.size_bytes, []).append(pin)

    matches: list[SourceMatch] = []
    if not sources_dir.is_dir():
        return matches
    for path in sorted(p for p in sources_dir.rglob("*") if p.is_file()):
        candidates = by_size.get(path.stat().st_size)
        if not candidates:
            continue
        digest = _file_sha256(path)
        matched_pin = next((p for p in candidates if p.sha256 == digest), None)
        if matched_pin is None:
            continue
        with session_scope() as session:
            row_ids = (
                session.execute(select(Document.id).where(Document.filename == path.name))
                .scalars()
                .all()
            )
        matches.append(
            SourceMatch(
                path=path,
                pin=matched_pin,
                document_id=str(row_ids[0]) if len(row_ids) == 1 else None,
                ambiguous=len(row_ids) > 1,
            )
        )
    return matches


def remove_pinned_sources(
    matches: Sequence[SourceMatch], chunk_stores: Sequence[Any]
) -> list[SourceRemoval]:
    """Safe-remove matched files: everything recoverable, nothing hard-deleted.

    An ingested match goes through :func:`delete_document` (ADR-014 semantics — Recycle
    Bin first, then row/chunks/sidecars) against ``chunk_stores[0]`` (the live index);
    the same document's chunks are then swept from any additional stores. A never-ingested
    match is simply moved to the OS trash, as is a matched file that survives its row
    delete (``source_original`` pointing elsewhere). Ambiguous matches are skipped. A
    refused trash (locked file) fails that one match and leaves it intact; the batch
    continues. Recovery: restore from the Recycle Bin, or re-download + re-ingest.
    """
    from send2trash import send2trash

    if not chunk_stores:
        raise ValueError("chunk_stores must contain at least the live index")
    live, *rest = chunk_stores

    results: list[SourceRemoval] = []
    for match in matches:
        name = match.path.name
        if match.ambiguous:
            log.warning("pinned_removal_ambiguous", file=name)
            results.append(SourceRemoval(name, False, False, 0, skipped_ambiguous=True))
            continue

        deleted_doc = False
        trashed = False
        chunks_removed = 0
        try:
            if match.document_id is not None:
                with session_scope() as session:
                    doc = session.get(Document, match.document_id)
                    doc_hash_val = doc.doc_hash if doc is not None else None
                deleted = delete_document(match.document_id, live)
                if deleted is not None:
                    deleted_doc = True
                    trashed = deleted.trashed_file
                    chunks_removed = deleted.chunks_removed
                if doc_hash_val is not None:
                    for store in rest:
                        try:
                            found = store.get(where={"doc_hash": doc_hash_val}, include=[])
                            ids = list(found.get("ids", []))
                            if ids:
                                store.delete(ids=ids)
                                chunks_removed += len(ids)
                        except Exception as e:
                            log.warning("pinned_removal_chunks_failed", file=name, error=str(e))
            if match.path.exists():
                send2trash(str(match.path))
                trashed = True
        except (RuntimeError, OSError) as e:
            log.warning("pinned_removal_failed", file=name, error=str(e))
            results.append(SourceRemoval(name, deleted_doc, trashed, chunks_removed, failed=True))
            continue
        results.append(SourceRemoval(name, deleted_doc, trashed, chunks_removed))
    return results


def library_summary() -> LibrarySummary:
    """Return high-level counts for the library."""
    with session_scope() as session:
        total_docs = (
            session.execute(
                select(func.count(Document.id)).where(Document.is_archived.is_(False))
            ).scalar()
            or 0
        )

        total_chunks_query = select(func.coalesce(func.sum(Document.chunk_count), 0))
        total_chunks_query = total_chunks_query.where(Document.is_archived.is_(False))
        total_chunks = session.execute(total_chunks_query).scalar() or 0

        by_health: Counter[str] = Counter()
        by_format: Counter[str] = Counter()
        for doc in session.execute(
            select(Document).where(Document.is_archived.is_(False))
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
        matches = (
            session.execute(select(Document.id).where(Document.id.like(f"{short_id}%")))
            .scalars()
            .all()
        )
        if len(matches) == 1:
            return str(matches[0])
        return None
