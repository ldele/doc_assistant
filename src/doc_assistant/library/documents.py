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


def effective_metadata(
    doc: Document, meta: DocumentMeta | None
) -> tuple[str | None, str | None, int | None]:
    """The title/authors/year a reader should see: the user's override, else what extraction found.

    ADR-013's merge rule, in one place because it was in two. `list_documents` applied it inline
    while `get_document_details` returned the raw columns, so the same document showed a corrected
    title in the grid and the original on its own page — found 2026-08-19 on an OCR-derived title,
    where the difference between the two was the whole point of editing it.

    `or` rather than a None-check on the override, matching the original: an override stored as an
    empty string falls back to the extracted value. ADR-013 clears a blank field rather than
    storing one, so this is a belt-and-braces path, not the normal one.
    """
    title = (
        meta.title_override if meta and meta.title_override is not None else None
    ) or doc.title
    authors = (
        meta.authors_override if meta and meta.authors_override is not None else None
    ) or doc.authors
    year = (meta.year_override if meta and meta.year_override is not None else None) or doc.year
    return title, authors, year


def is_customized(meta: DocumentMeta | None) -> bool:
    """Whether the user has overridden any field — what a Reset affordance keys off."""
    return meta is not None and any(
        v is not None for v in (meta.title_override, meta.authors_override, meta.year_override)
    )


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
            title, authors, year = effective_metadata(d, m)
            customized = is_customized(m)
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
                    source_path=d.source_original,
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

    # Deliberately the EXTRACTED year, not the ADR-013 override. This feeds the year-aware
    # epistemics rule (G3), which is an analysis over what the corpus says, not a display of what
    # the user prefers to see — and silently re-deriving `superseded_trend` from a metadata edit
    # would make a knowledge-layer verdict move for a reason no baseline records. If that is ever
    # wanted it is a decision with an eval, not a consistency fix.
    years: dict[str, int] = {}
    with session_scope() as session:
        stmt = select(Document.id, Document.year).where(Document.id.in_(document_ids))
        for doc_id, year in session.execute(stmt):
            if doc_id is not None and year is not None:
                years[str(doc_id)] = int(year)
    return years


class DocumentPrefixError(ValueError):
    """A ``--doc`` prefix matched no document, or matched more than one.

    Carries a CLI-ready message so every runner reports the same thing (KI-30).
    """


@dataclass(frozen=True)
class DocumentRef:
    """The identifiers a sidecar runner needs for one document.

    Both are carried because the enrichment layer is split on which one it keys by: the
    citation/metadata tables filter on ``doc_hash``, while every user-facing surface (the API,
    the graph, the library grid, ``list_documents``) hands out ``id``.
    """

    id: str
    doc_hash: str
    filename: str


def resolve_document_prefix(prefix: str) -> DocumentRef:
    """Resolve a ``--doc`` argument (an ``id`` or ``doc_hash`` prefix) to exactly one document.

    The single entry point behind every runner's ``--doc`` flag (KI-30). Before this existed the
    flag meant three different things across four runners, and two of them rejected the very id
    the rest of the app hands out.

    ``id`` is tried first and wins outright: that is the identifier a caller actually has. Only
    when the prefix matches no id at all does it fall back to ``doc_hash``, so the hashes printed
    by the older runners keep working. Archived documents are included — a runner that wants them
    excluded filters its own work set; refusing to *resolve* them would make the flag lie.

    Wildcards are escaped, so a literal ``_`` in a prefix cannot silently act as a single-character
    ``LIKE`` wildcard.

    Raises:
        DocumentPrefixError: when the prefix is blank, matches nothing, or is ambiguous.
    """
    if not prefix.strip():
        raise DocumentPrefixError("--doc needs a non-empty id or doc_hash prefix.")

    with session_scope() as session:
        for column, label in ((Document.id, "id"), (Document.doc_hash, "doc_hash")):
            rows = session.execute(
                select(Document.id, Document.doc_hash, Document.filename)
                .where(column.startswith(prefix, autoescape=True))
                .order_by(Document.filename)
            ).all()
            if len(rows) == 1:
                doc_id, doc_hash, filename = rows[0]
                return DocumentRef(id=str(doc_id), doc_hash=str(doc_hash), filename=str(filename))
            if len(rows) > 1:
                names = ", ".join(f"{r[0][:8]} ({r[2]})" for r in rows[:5])
                more = f", +{len(rows) - 5} more" if len(rows) > 5 else ""
                raise DocumentPrefixError(
                    f"--doc {prefix!r} is ambiguous: {len(rows)} documents share that "
                    f"{label} prefix — {names}{more}. Pass more characters."
                )

    raise DocumentPrefixError(
        f"--doc {prefix!r} matched no document (tried it as an id prefix, then as a "
        f"doc_hash prefix)."
    )


def get_document_details(doc_id: str) -> DocumentDetails | None:
    """Return everything we know about a single document."""
    with session_scope() as session:
        doc = session.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
        if not doc:
            return None

        # The same override merge the grid applies (ADR-013). Without it this view answered
        # with the extracted values, so editing a title fixed the list but not the page it
        # was edited on.
        meta = session.execute(
            select(DocumentMeta).where(DocumentMeta.document_id == doc.id)
        ).scalar_one_or_none()
        title, authors, year = effective_metadata(doc, meta)

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
            title=title,
            authors=authors,
            year=year,
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


def purge_document_record(
    document_id: str,
    chroma_db: Any,
    *,
    doc_hash: str,
    source_cache: str | None,
) -> int:
    """Remove a document *everywhere except the file on disk*. Returns the chunk count removed.

    Split out of `delete_document` so `library.add.undo_add` can reuse it. The two callers differ
    entirely on the file and not at all on the record: `delete_document` sends the source to the
    Recycle Bin first, while undo either deletes the copy the app itself made or — for a referenced
    file — must not touch it at all (the ADR-014 amendment). Keeping the record half in one place
    is what stops undo from growing a second, drifting copy of "what removing a document means".

    Removes: the ``Document`` row (FK-cascades citations / parts / similarities, and the
    ``DocumentMeta`` override since ADR-026), the doc's chunks from the live Chroma store, its
    figure directory and its cached markdown. Each on-disk step is guarded — a locked file leaves a
    warning, not a half-removed document.
    """
    from doc_assistant.ingest.cleanup import cleanup_orphan_figures

    with session_scope() as session:
        meta = session.get(DocumentMeta, document_id)
        if meta is not None:
            session.delete(meta)
        doc = session.get(Document, document_id)
        if doc is not None:
            session.delete(doc)

    chunks_removed = 0
    try:
        found = chroma_db.get(where={"doc_hash": doc_hash}, include=[])
        ids = list(found.get("ids", []))
        chunks_removed = len(ids)
        if ids:
            chroma_db.delete(ids=ids)
    except Exception as e:
        log.warning("delete_chunks_failed", document_id=document_id, error=str(e))

    cleanup_orphan_figures([doc_hash])
    if source_cache:
        cache_path = Path(source_cache)
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError as e:
                log.warning("delete_cache_failed", file=cache_path.name, error=str(e))
    return chunks_removed


def _forget_source_row(source_original: str | None) -> bool:
    """Drop the ``SourceFile`` row naming ``source_original``. Returns whether one went (KI-52).

    Matched through ``registry.pathkey`` rather than raw string equality, for the same reason
    ``_existing_document_id`` does: the same file reaches the two tables spelled differently
    depending on which caller wrote it. Resolving each row through **its own root** is what keeps a
    referenced file in the user's folder from matching a same-named file in the library.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile, SourceRoot
    from doc_assistant.ingest.registry import pathkey

    if not source_original:
        return False
    wanted = pathkey(source_original)
    with session_scope() as session:
        roots = {r.id: r.path for r in session.execute(select(SourceRoot)).scalars()}
        for row in session.execute(select(SourceFile)).scalars():
            base = roots.get(row.root_id)
            if base is None:
                continue
            if pathkey(Path(base) / row.rel_path) == wanted:
                session.delete(row)
                log.info("source_row_forgotten", rel_path=row.rel_path, root_id=row.root_id)
                return True
    return False


def delete_document(
    document_id: str, chroma_db: Any, *, delete_file: bool = False
) -> DeleteResult | None:
    """Remove a document from the library, and — only if asked — its source file too.

    Returns None if the document is unknown.

    **ADR-046 §2 amends ADR-014: this no longer bins the source unconditionally.** ADR-014 made
    "delete" mean "delete the file", which is right for a copy the app made and wrong for a file
    the user keeps in their own folder and merely pointed the library at. The choice is now the
    caller's, per deletion, and the safe branch is the default:

    * ``delete_file=False`` (**default**) — *remove from library*. The row, its chunks, its figure
      directory and its cached markdown go; **the source file is untouched.** The registry row is
      also kept: the file is still on disk and not indexed, which is exactly what
      ``derive_status`` reports as ``new`` — the file is a candidate again, which is true.
    * ``delete_file=True`` — ADR-014's behaviour, now opt-in. **The ordering ADR-014 chose is
      preserved**: the file goes to the Recycle Bin *first* (recoverable) and only on success does
      the removal proceed, so a locked file leaves the library entry intact rather than orphaning a
      still-indexed file on disk. The registry row goes too — see below.

    **For a referenced document the caller must have shown the real path** before passing
    ``delete_file=True``. That is a UI contract (ADR-046 §2) and cannot be enforced here; the
    library exposes the path on ``DocumentSummary.source_path`` so the caller has no excuse.

    **KI-52, fixed here:** the ``SourceFile`` registry row used to survive every delete, so a file
    the user deleted *through the app* came back as ``missing`` in Sources — the app misreporting
    its own action, with no way to clear it. The row now goes with the file, and only with the
    file: when the source stays on disk, the row is still true.
    """
    from send2trash import send2trash

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        filename = doc.filename
        doc_hash_val = doc.doc_hash
        source_original = doc.source_original
        source_cache = doc.source_cache

    # 1. Recycle the source file first (recoverable) — only when the caller asked for it.
    #    A trash failure aborts the whole delete, so the library never lists a document whose file
    #    we failed to remove *and* whose row we removed anyway.
    path = resolve_source_path(source_original, filename)
    trashed = False
    if delete_file and path is not None:
        try:
            send2trash(str(path))
            trashed = True
        except Exception as e:
            log.warning("delete_trash_failed", document_id=document_id, error=str(e))
            raise RuntimeError(f"could not move {filename} to the Recycle Bin") from e

    # 2-4. Everything except the file: the row (+ cascades), the chunks, the on-disk sidecars.
    chunks_removed = purge_document_record(
        document_id, chroma_db, doc_hash=doc_hash_val, source_cache=source_cache
    )

    # The registry row is only false once the file is gone; while the file is on disk the row
    # correctly says "here, and not indexed" (KI-52).
    if delete_file:
        _forget_source_row(source_original)

    log.info(
        "document_deleted",
        document_id=document_id,
        delete_file=delete_file,
        trashed_file=trashed,
        chunks_removed=chunks_removed,
    )
    return DeleteResult(filename=filename, trashed_file=trashed, chunks_removed=chunks_removed)
