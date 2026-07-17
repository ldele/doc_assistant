"""Data access layer for the document library.

This module provides a stable Python API over the SQLite store. The UI
calls into this module rather than touching SQLAlchemy directly, so
swapping the UI or the storage backend doesn't require coordinated
changes.

All functions return plain dataclasses, not SQLAlchemy models. This
keeps the UI layer free of session lifecycle concerns.
"""

# subprocess is used only by _reveal_in_file_manager to open a local file in the OS file manager.
import subprocess  # nosec B404
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select

from doc_assistant.db.models import Document, DocumentMeta, Folder, Tag
from doc_assistant.db.session import session_scope

log = structlog.get_logger(__name__)

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
        query = select(Document).where(Document.is_archived.is_(False))

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
                    tags=[t.name for t in d.tags],
                    keywords=[k.name for k in d.keywords],
                    added_at=d.added_at,
                )
            )
        return summaries


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
    similarities), the ``DocumentMeta`` override (no FK — explicit), the doc's chunks from the live
    Chroma store, its figure dir, and its cached ``.md``. ADR-014.
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

    # 2. Drop the DB row (+ cascades) and the override sidecar (no FK).
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


# ============================================================
# Keyword families (feature-tag-families.md — PR-1)
# ============================================================
# A family = a curated Concept whose ConceptAlias rows carry member Keyword names
# (ADR-015). Reuses the existing concept vocabulary — no new schema. A keyword
# belongs to at most one family; assigning it to a second family moves it.


@dataclass
class KeywordFamily:
    """A canonical tag + its member keyword names, with a union doc_count.

    ``aliases`` excludes the canonical label itself (mirrors ``GlossaryEntry``).
    ``doc_count`` is the number of documents carrying *any* member keyword (canonical or
    alias), matched case-insensitively against ``Keyword.name``.
    """

    id: str
    canonical: str
    aliases: list[str] = field(default_factory=list)
    doc_count: int = 0


def _family_doc_count(session: Any, names: list[str]) -> int:
    """Union of documents carrying any of ``names`` as a Keyword, case-insensitive."""
    from doc_assistant.db.models import Keyword, document_keywords

    if not names:
        return 0
    lowered = {n.casefold() for n in names}
    stmt = (
        select(func.count(func.distinct(document_keywords.c.document_id)))
        .select_from(document_keywords)
        .join(Keyword, Keyword.id == document_keywords.c.keyword_id)
        .where(func.lower(Keyword.name).in_(lowered))
    )
    return int(session.execute(stmt).scalar() or 0)


def _build_family(session: Any, concept: Any) -> KeywordFamily:
    aliases = sorted(a.alias for a in concept.aliases if a.alias != concept.label)
    doc_count = _family_doc_count(session, [concept.label, *aliases])
    return KeywordFamily(
        id=str(concept.id), canonical=concept.label, aliases=aliases, doc_count=doc_count
    )


def list_keyword_families() -> list[KeywordFamily]:
    """All curated concepts as keyword families, each with its union doc_count."""
    from doc_assistant.db.models import Concept

    with session_scope() as session:
        concepts = list(session.execute(select(Concept)).scalars())
        families = [_build_family(session, c) for c in concepts]
    families.sort(key=lambda f: f.canonical.casefold())
    return families


def get_keyword_family(concept_id: str) -> KeywordFamily | None:
    """One keyword family by id, or None if unknown."""
    from doc_assistant.db.models import Concept

    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None:
            return None
        return _build_family(session, concept)


def create_keyword_family(canonical: str, members: list[str] | None = None) -> KeywordFamily:
    """Create a keyword family (a curated Concept) with initial member keywords.

    Idempotent by canonical label (matches ``add_concept``'s get-or-create). Any member
    keyword already belonging to another family is moved (a keyword belongs to at most one
    family — ADR-015).
    """
    from doc_assistant.concept_skeleton import add_concept

    canonical = canonical.strip()
    if not canonical:
        raise ValueError("canonical must not be blank")
    concept_id = add_concept(label=canonical)
    for member in members or []:
        add_family_member(concept_id, member)
    family = get_keyword_family(concept_id)
    if family is None:  # pragma: no cover - add_concept above guarantees the row exists
        raise RuntimeError(f"keyword family {concept_id!r} vanished immediately after creation")
    return family


def rename_keyword_family(concept_id: str, new_canonical: str) -> KeywordFamily | None:
    """Rename a family's canonical label. Returns None if the family is unknown."""
    from doc_assistant.concept_skeleton import rename_concept

    new_canonical = new_canonical.strip()
    if not new_canonical:
        raise ValueError("new_canonical must not be blank")
    if not rename_concept(concept_id, new_canonical):
        return None
    return get_keyword_family(concept_id)


def add_family_member(concept_id: str, keyword_name: str) -> KeywordFamily | None:
    """Assign a keyword to a family. Returns None if the family is unknown.

    A keyword belongs to at most one family: if it's already an alias of another family,
    it's removed from there first (moved, not duplicated). Idempotent — assigning an
    already-member keyword is a no-op. (Does not check whether ``keyword_name`` collides
    with *another* family's canonical label — an edge case left to the Manage view.)
    """
    from doc_assistant.db.models import Concept, ConceptAlias

    keyword_name = keyword_name.strip()
    if not keyword_name:
        raise ValueError("keyword_name must not be blank")
    lowered = keyword_name.casefold()
    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None:
            return None
        others = (
            session.execute(
                select(ConceptAlias).where(
                    func.lower(ConceptAlias.alias) == lowered,
                    ConceptAlias.concept_id != concept_id,
                )
            )
            .scalars()
            .all()
        )
        for other in others:
            session.delete(other)
        already_member = concept.label.casefold() == lowered or any(
            a.alias.casefold() == lowered for a in concept.aliases
        )
        if not already_member:
            concept.aliases.append(ConceptAlias(alias=keyword_name))
        session.flush()
        return _build_family(session, concept)


def remove_family_member(concept_id: str, keyword_name: str) -> KeywordFamily | None:
    """Remove a keyword from a family's alias set. Returns None if the family is unknown.

    A no-op if ``keyword_name`` isn't a member alias (idempotent) — the canonical label
    itself can't be "removed" this way; rename or delete the family instead.
    """
    from doc_assistant.db.models import Concept

    lowered = keyword_name.strip().casefold()
    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None:
            return None
        row = next((a for a in concept.aliases if a.alias.casefold() == lowered), None)
        if row is not None:
            concept.aliases.remove(row)
        session.flush()
        return _build_family(session, concept)


def delete_keyword_family(concept_id: str) -> bool:
    """Delete a family. Returns True if it existed."""
    from doc_assistant.concept_skeleton import delete_concept

    return delete_concept(concept_id)


# ============================================================
# Chunk browser (Library space L1 — docs/specs/feature-library-browser.md)
# ============================================================
# A read-only view of the chunks the two-tier retriever stores for a document:
# parent blocks (the parent_text the LLM reads / a citation shows), each carrying
# its embedded child chunks. Reads the live Chroma handle via a metadata filter —
# no embeddings, no BM25, no generation, no writes. Markers + figures are L1b.


@dataclass
class ChunkChild:
    """One embedded child chunk within a parent block."""

    child_index: int
    text: str
    retrievable: bool  # False only when keep_for_retrieval metadata is explicitly False


@dataclass
class ParentBlock:
    """One parent block (the unit the LLM reads) + its ordered child chunks."""

    parent_index: int
    parent_text: str
    children: list[ChunkChild]


@dataclass
class DocumentChunkView:
    """A document's header + its chunks grouped into parent blocks (L1 browser detail).

    NULL metadata (``title``/``authors``/``year`` are often absent on real corpora) stays None;
    the renderer omits it rather than showing a blank label.
    """

    id: str
    filename: str
    format: str
    title: str | None
    authors: str | None
    year: int | None
    chunk_count: int | None
    health: str | None
    parents: list[ParentBlock]
    child_count: int


def group_children(chunks: list[dict[str, Any]]) -> list[ParentBlock]:
    """Group flat child chunks into ordered parent blocks — **pure**, the browser's core.

    Each input dict is one child chunk: ``parent_index`` (int), ``child_index`` (int),
    ``parent_text`` (str), ``text`` (the child's own text), ``keep_for_retrieval`` (bool | None).
    A chunk missing ``parent_index`` or ``child_index`` is dropped (logged count) — it cannot be
    placed. Parents are ordered by ``parent_index``; children within a parent by ``child_index``;
    ``parent_text`` is taken from each parent's first-seen child.
    """
    by_parent: dict[int, list[dict[str, Any]]] = {}
    parent_text: dict[int, str] = {}
    dropped = 0
    for chunk in chunks:
        p_idx = chunk.get("parent_index")
        c_idx = chunk.get("child_index")
        if p_idx is None or c_idx is None:
            dropped += 1
            continue
        by_parent.setdefault(int(p_idx), []).append(chunk)
        parent_text.setdefault(int(p_idx), str(chunk.get("parent_text") or ""))
    if dropped:
        log.info("library_chunks_dropped", count=dropped, reason="missing parent/child index")

    blocks: list[ParentBlock] = []
    for p_idx in sorted(by_parent):
        children = [
            ChunkChild(
                child_index=int(c["child_index"]),
                text=str(c.get("text") or ""),
                retrievable=c.get("keep_for_retrieval") is not False,
            )
            for c in sorted(by_parent[p_idx], key=lambda c: int(c["child_index"]))
        ]
        blocks.append(
            ParentBlock(parent_index=p_idx, parent_text=parent_text[p_idx], children=children)
        )
    return blocks


def get_document_chunks(doc_id: str, chroma: Any) -> DocumentChunkView | None:
    """One document's header + its chunks grouped into parent blocks, or ``None`` if unknown.

    ``chroma`` is the live handle (``ChatController.rag.db``) — a metadata-filtered ``get``, no
    embeddings, no generation. A document that exists but has zero stored chunks returns a view
    with ``parents=[]`` (honest empty-state), not ``None`` (which means "unknown document").
    """
    with session_scope() as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            return None
        # Read the scalar fields inside the session (avoids a detached lazy-load after close).
        d_id = str(doc.id)
        d_filename = doc.filename
        d_format = doc.format
        d_title = doc.title
        d_authors = doc.authors
        d_year = doc.year
        d_chunk_count = doc.chunk_count
        d_health = doc.extraction_health

    result = chroma.get(where={"document_id": doc_id}, include=["documents", "metadatas"])
    documents: list[str] = result.get("documents") or []
    metadatas: list[dict[str, Any]] = result.get("metadatas") or []
    chunks: list[dict[str, Any]] = [
        {
            "parent_index": (meta or {}).get("parent_index"),
            "child_index": (meta or {}).get("child_index"),
            "parent_text": (meta or {}).get("parent_text"),
            "text": text,
            "keep_for_retrieval": (meta or {}).get("keep_for_retrieval"),
        }
        for text, meta in zip(documents, metadatas, strict=True)
    ]
    parents = group_children(chunks)
    return DocumentChunkView(
        id=d_id,
        filename=d_filename,
        format=d_format,
        title=d_title,
        authors=d_authors,
        year=d_year,
        chunk_count=d_chunk_count,
        health=d_health,
        parents=parents,
        child_count=len(chunks),
    )


# ============================================================
# Citation queries (Phase 4)
# ============================================================


@dataclass
class GraphNode:
    """One node in a citation subgraph."""

    id: str
    filename: str
    title: str | None
    is_center: bool


@dataclass
class GraphEdge:
    """One directed edge in a citation subgraph."""

    source: str
    target: str


@dataclass
class CitationGraph:
    """Result of `graph_subgraph` — typed alternative to dict[str, Any]."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass
class CitationEdge:
    """A single citation edge — source -> target, internal or external."""

    raw_text: str | None
    target_title: str | None
    target_authors: str | None
    target_year: int | None
    target_doi: str | None
    target_document_id: str | None  # None = external (not in library)
    target_filename: str | None  # convenience for UI
    extraction_method: str | None
    confidence: float | None


def _row_to_edge(row: Any) -> CitationEdge:
    return CitationEdge(
        raw_text=row.raw_citation_text,
        target_title=row.target_title,
        target_authors=row.target_authors,
        target_year=row.target_year,
        target_doi=row.target_doi,
        target_document_id=row.target_document_id,
        target_filename=row.target_filename,
        extraction_method=row.extraction_method,
        confidence=row.confidence,
    )


def cites_out(doc_id: str) -> list[CitationEdge]:
    """Return all citations this doc makes (papers it cites)."""
    from doc_assistant.db.models import Citation

    with session_scope() as session:
        stmt = (
            select(
                Citation.raw_citation_text,
                Citation.target_title,
                Citation.target_authors,
                Citation.target_year,
                Citation.target_doi,
                Citation.target_document_id,
                Document.filename.label("target_filename"),
                Citation.extraction_method,
                Citation.confidence,
            )
            .outerjoin(Document, Document.id == Citation.target_document_id)
            .where(Citation.source_document_id == doc_id)
            .order_by(Citation.target_year.desc().nulls_last(), Citation.target_authors)
        )
        return [_row_to_edge(r) for r in session.execute(stmt).all()]


def cited_by(doc_id: str) -> list[tuple[str, str, str | None]]:
    """Return (source_doc_id, source_filename, raw_citation_text) for incoming citations."""
    from doc_assistant.db.models import Citation

    with session_scope() as session:
        stmt = (
            select(
                Document.id,
                Document.filename,
                Citation.raw_citation_text,
            )
            .join(Citation, Citation.source_document_id == Document.id)
            .where(Citation.target_document_id == doc_id)
            .order_by(Document.filename)
        )
        return [(str(r[0]), str(r[1]), r[2]) for r in session.execute(stmt).all()]


def graph_subgraph(doc_id: str, depth: int = 1) -> CitationGraph:
    """Return a CitationGraph centered on doc_id (internal edges only)."""
    from doc_assistant.db.models import Citation

    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    frontier = {doc_id}
    visited: set[str] = set()

    with session_scope() as session:
        center = session.execute(
            select(Document.id, Document.filename, Document.title).where(Document.id == doc_id)
        ).first()
        if center is None:
            return CitationGraph()
        nodes[doc_id] = GraphNode(
            id=doc_id, filename=center.filename, title=center.title, is_center=True
        )

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                outs = session.execute(
                    select(
                        Citation.target_document_id,
                        Document.filename,
                        Document.title,
                    )
                    .join(Document, Document.id == Citation.target_document_id)
                    .where(Citation.source_document_id == nid)
                    .where(Citation.target_document_id.is_not(None))
                ).all()
                for tgt_id, tgt_fn, tgt_title in outs:
                    if tgt_id not in nodes:
                        nodes[tgt_id] = GraphNode(
                            id=tgt_id, filename=tgt_fn, title=tgt_title, is_center=False
                        )
                        next_frontier.add(tgt_id)
                    edges.append(GraphEdge(source=nid, target=tgt_id))
                ins = session.execute(
                    select(
                        Citation.source_document_id,
                        Document.filename,
                        Document.title,
                    )
                    .join(Document, Document.id == Citation.source_document_id)
                    .where(Citation.target_document_id == nid)
                ).all()
                for src_id, src_fn, src_title in ins:
                    if src_id not in nodes:
                        nodes[src_id] = GraphNode(
                            id=src_id, filename=src_fn, title=src_title, is_center=False
                        )
                        next_frontier.add(src_id)
                    edges.append(GraphEdge(source=src_id, target=nid))
            frontier = next_frontier
            if not frontier:
                break

    edge_keys: set[tuple[str, str]] = set()
    deduped: list[GraphEdge] = []
    for e in edges:
        key = (e.source, e.target)
        if key not in edge_keys:
            edge_keys.add(key)
            deduped.append(e)
    return CitationGraph(nodes=list(nodes.values()), edges=deduped)


# ============================================================
# Similarity queries (Phase 4 close-out)
# ============================================================


@dataclass
class SimilarDoc:
    """One neighbour returned by `similar_docs`."""

    target_document_id: str
    target_filename: str
    target_title: str | None
    score: float


def similar_docs(
    doc_id: str,
    *,
    limit: int = 10,
    embedding_model: str | None = None,
) -> list[SimilarDoc]:
    """Return the top-N most similar documents to ``doc_id``.

    Reads pre-computed edges from the ``doc_similarities`` sidecar
    table. Empty list if no edges exist (run
    ``scripts/compute_doc_vectors.py --apply``).
    """
    from doc_assistant.db.models import DocSimilarity

    with session_scope() as session:
        stmt = (
            select(
                DocSimilarity.target_document_id,
                Document.filename,
                Document.title,
                DocSimilarity.score,
            )
            .join(Document, Document.id == DocSimilarity.target_document_id)
            .where(DocSimilarity.source_document_id == doc_id)
        )
        if embedding_model is not None:
            stmt = stmt.where(DocSimilarity.embedding_model == embedding_model)
        stmt = stmt.order_by(DocSimilarity.score.desc()).limit(limit)
        return [
            SimilarDoc(
                target_document_id=str(target_id),
                target_filename=str(filename),
                target_title=title,
                score=float(score),
            )
            for target_id, filename, title, score in session.execute(stmt).all()
        ]
