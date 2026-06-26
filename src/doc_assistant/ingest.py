import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, select
from tqdm import tqdm

from doc_assistant import config
from doc_assistant.config import (
    CACHE_PATH,
    CHROMA_PATH,
    DOCS_PATH,
    PC_CHROMA_PATH,
    PDF_EXTRACTOR,
)
from doc_assistant.db.migrations import init_db
from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.models import Figure, IngestionEvent
from doc_assistant.db.session import session_scope
from doc_assistant.embeddings import (
    get_active_model_name,
    get_collection_name,
    get_embeddings,
)
from doc_assistant.extractors import extract_to_markdown, is_supported
from doc_assistant.figures import figure_chunk_text, figure_dir
from doc_assistant.fsutil import atomic_write_text
from doc_assistant.tables_marker import TABLE_BLOCK_RE

log = structlog.get_logger(__name__)

PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")
HEADING_MARKER = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)

# A table caption is short; never pull a large block of prose into a table's parent
# when absorbing the caption attached immediately before a spliced table block.
_MAX_ABSORBED_CAPTION_CHARS = 1000
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")

# Splitter sizes are config-driven (see config.PARENT_CHUNK_SIZE etc.) so a
# chunking sweep can vary them via env without editing source. The factories
# read ``config`` attributes at call time, which keeps them monkeypatch-able
# in tests; the module-level singletons below preserve the original import-time
# construction for the hot path.


def _make_parent_splitter() -> RecursiveCharacterTextSplitter:
    """Large-passage splitter for parent chunks (sent to the LLM)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )


def _make_child_splitter() -> RecursiveCharacterTextSplitter:
    """Small-passage splitter for child chunks (embedded for retrieval)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )


def _make_baseline_splitter() -> RecursiveCharacterTextSplitter:
    """Single-chunk splitter for the baseline (non parent-child) store."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.BASELINE_CHUNK_SIZE,
        chunk_overlap=config.BASELINE_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
    )


_pc_parent_splitter = _make_parent_splitter()
_pc_child_splitter = _make_child_splitter()


def get_cache_path(original: Path) -> Path:
    relative = original.relative_to(DOCS_PATH)
    return CACHE_PATH / relative.with_suffix(".md")


def is_cache_fresh(original: Path, cached: Path) -> bool:
    if not cached.exists():
        return False
    return cached.stat().st_mtime >= original.stat().st_mtime


def load_or_extract(original: Path) -> str:
    cached = get_cache_path(original)
    if is_cache_fresh(original, cached):
        return cached.read_text(encoding="utf-8")

    log.info("extracting", file=original.name)
    text = extract_to_markdown(original, pdf_extractor=PDF_EXTRACTOR)
    # Atomic write: this cached .md is the source-of-truth the next ingest re-hashes;
    # a crash mid-write must not leave a truncated cache that is_cache_fresh trusts
    # (the same hazard the table-splice writers share — see fsutil.atomic_write_text).
    atomic_write_text(cached, text)
    return text


def doc_hash(text: str) -> str:
    """Content-only hash. Path-independent so documents survive moves/renames."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


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


def _find_orphan_hashes(
    hash_to_meta: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Classify stored doc-hashes that no current source still produces.

    Returns ``(gone, stale)``:

    * ``gone``  — the hash's source file no longer exists on disk.
    * ``stale`` — the source file is still there, but its current (cache-backed)
      content now hashes to something else, so this pre-change hash is a leftover
      duplicate. This is exactly what a Marker table-splice creates: splicing
      tables into the cached ``.md`` changes ``doc_hash(text)``, so an incremental
      ingest adds the new-hash document *beside* the old one. Detecting only
      ``gone`` (the original behaviour) left these stale copies behind, so the
      store ended up with two hashes per changed file and only ``--rebuild``
      (a full wipe + re-embed) could clean it.

    Each surviving source is re-hashed once via ``load_or_extract`` (cache-backed,
    so cheap when the cache is fresh — the splice case). A source that can't be
    read, or extracts empty, is left untouched: a transient extract failure must
    never delete live chunks.
    """
    source_to_hashes: dict[str, set[str]] = {}
    for h, meta in hash_to_meta.items():
        source_to_hashes.setdefault(str(meta.get("source_original", "")), set()).add(h)

    gone: list[str] = []
    stale: list[str] = []
    for source, hashes in source_to_hashes.items():
        path = Path(source)
        if not source or not path.exists():
            gone.extend(hashes)
            continue
        try:
            text = load_or_extract(path)
        except Exception as e:
            log.warning("rehash_failed", file=path.name, error=str(e), hint="keeping it")
            continue
        if not text.strip():
            continue
        current = doc_hash(text)
        stale.extend(h for h in hashes if h != current)
    return gone, stale


def cleanup_orphans_sqlite(db_for_metadata: Chroma) -> list[str]:
    """Remove SQLite rows for documents no current source still produces.

    Two kinds of orphan are removed (see ``_find_orphan_hashes``): documents whose
    source file is gone, and the pre-change hash of a document whose *content*
    changed (e.g. tables spliced into its cached ``.md``). Returns the orphan
    hashes for downstream Chroma cleanup.
    """
    data = db_for_metadata.get(include=["metadatas"])
    hash_to_meta: dict[str, dict[str, Any]] = {}
    for meta in data["metadatas"]:
        if meta and meta.get("doc_hash"):
            hash_to_meta[meta["doc_hash"]] = meta

    gone, stale = _find_orphan_hashes(hash_to_meta)
    orphan_hashes = gone + stale
    if not orphan_hashes:
        return []

    if stale:
        # A content change mints a NEW document_id for the new hash, so any sidecar
        # enrichment keyed to the OLD id is now stale. Deleting the old Document row
        # FK-cascades its outbound citations + doc_similarities (ondelete=CASCADE)
        # and NULLs inbound citation targets (ondelete=SET NULL); the new content
        # starts with none. Re-run the citation / doc-vector enrichment afterwards.
        log.info(
            "enrichment_dropped",
            count=len(stale),
            hint="old enrichment (citations, doc_similarities) dropped; re-run to rebuild",
        )

    log.info("removing_orphans", count=len(orphan_hashes))
    with session_scope() as session:
        for h in orphan_hashes:
            doc = session.execute(
                select(DBDocument).where(DBDocument.doc_hash == h)
            ).scalar_one_or_none()
            if doc:
                session.delete(doc)

    return orphan_hashes


def cleanup_orphans_chroma(
    db: Chroma, orphan_hashes: list[str], also_clean_cache: bool = False
) -> None:
    """Delete chunks for orphan documents from a Chroma store.

    When ``also_clean_cache`` is set the cached ``.md`` sidecar is removed too — but
    ONLY for orphans whose source file is gone. A content-changed document is also
    an orphan (its pre-change hash no longer matches), yet its cache holds the *new*
    content the fresh hash re-ingests from; deleting it would destroy the live
    extraction. Gate cache removal on source existence, not on orphan-ness.
    """
    if not orphan_hashes:
        return

    orphan_set = set(orphan_hashes)
    orphan_caches: list[Path] = []
    if also_clean_cache:
        data = db.get(include=["metadatas"])
        for meta in data["metadatas"]:
            if not meta or meta.get("doc_hash") not in orphan_set:
                continue
            if Path(str(meta.get("source_original", ""))).exists():
                continue  # source still here — its cache is the live copy, keep it
            cache_path = Path(str(meta.get("source_cache", "")))
            if cache_path.exists():
                orphan_caches.append(cache_path)

    for h in orphan_hashes:
        db.delete(where={"doc_hash": h})

    if also_clean_cache and orphan_caches:
        for cache_path in set(orphan_caches):
            try:
                cache_path.unlink()
            except Exception as e:
                log.warning("cache_delete_failed", file=cache_path.name, error=str(e))
        log.info("removed_orphan_caches", count=len(set(orphan_caches)))


def cleanup_orphan_figures(orphan_hashes: list[str]) -> None:
    """Remove the on-disk figure PNG dirs for orphan documents.

    Figure *rows* FK-cascade-delete with their Document (``cleanup_orphans_sqlite``),
    but the cropped PNGs under ``FIGURE_DIR/{doc_hash}/`` (``figures.figure_dir``) are
    on-disk sidecars with no DB cascade — without this sweep they accumulate forever
    as documents are removed or their content changes. Keyed by ``doc_hash``, so it is
    correct for BOTH orphan kinds (``_find_orphan_hashes``): a gone source, and a
    content change (a new ``doc_hash`` leaves the old hash's figure dir dead — its PNGs
    never match the current content). Re-extraction writes the new hash's dir afresh.

    Gated by the same ``scope is None`` guard as the whole cleanup block (in ``main``);
    a ``--path`` run must not delete out-of-scope figures. (Unlike ``also_clean_cache``,
    an orthogonal *source-existence* gate, this sweep deliberately removes BOTH gone-
    and stale-orphan figure dirs.)

    Coupling: ingest cleanup <-> the figures on-disk layout (``config.FIGURE_DIR /
    {doc_hash}/``, via ``figures.figure_dir``). If that layout changes, this follows it.
    """
    if not orphan_hashes:
        return
    removed = 0
    for h in orphan_hashes:
        fig_dir = figure_dir(h)
        if not fig_dir.exists():
            continue
        # Per-hash try/except (not rmtree(ignore_errors=True)) so a locked PNG on Windows
        # surfaces a warning instead of a silently-incomplete sweep, and ``removed``
        # counts actual deletions — mirrors cleanup_orphans_chroma's cache-delete posture.
        try:
            shutil.rmtree(fig_dir)
            removed += 1
        except OSError as e:
            log.warning("figure_dir_delete_failed", dir=str(fig_dir), error=str(e))
    if removed:
        log.info("removed_orphan_figures", count=removed)


def load_documents() -> list[Document]:
    documents: list[Document] = []
    files = [p for p in DOCS_PATH.rglob("*") if p.is_file() and is_supported(p)]
    log.info("found_files", count=len(files))

    for path in files:
        try:
            text = load_or_extract(path)
            if not text.strip():
                log.info("skipping_empty", file=path.name)
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_original": str(path),
                        "source_cache": str(get_cache_path(path)),
                        "filename": path.name,
                        "format": path.suffix.lower().lstrip("."),
                        "doc_hash": doc_hash(text),
                    },
                )
            )
        except Exception as e:
            log.warning("document_error", file=path.name, error=str(e))

    return documents


def extract_chunk_metadata(
    chunk_text: str, full_text: str, chunk_start: int
) -> dict[str, int | str | None]:
    """Find the nearest preceding heading and current page number."""
    # Find page number -- last page marker at or before this chunk's start
    text_before = full_text[: chunk_start + len(chunk_text)]
    page_matches = list(PAGE_MARKER.finditer(text_before))
    page: int | None = int(page_matches[-1].group(1)) if page_matches else None

    heading_matches = list(HEADING_MARKER.finditer(text_before))
    section: str | None
    if heading_matches:
        raw_section = heading_matches[-1].group(2).strip()
        section = re.sub(r"[*_`]+", "", raw_section).strip()
        # Empty after stripping = not a real heading
        section = section if section else None
    else:
        section = None

    return {"page": page, "section": section}


def compute_health_signals(documents: list[Document], full_text: str) -> dict[str, int | float]:
    """Compute signals for health classification from a list of chunks."""
    if not documents:
        return {
            "chunk_count": 0,
            "avg_chunk_length": 0.0,
            "section_detection_rate": 0.0,
            "reference_flagged_ratio": 0.0,
        }

    chunk_lengths = [len(d.page_content) for d in documents]
    sections_detected = sum(1 for d in documents if d.metadata.get("section"))

    return {
        "chunk_count": len(documents),
        "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
        "section_detection_rate": sections_detected / len(documents),
        "reference_flagged_ratio": 0.0,
    }


def clean_chunk_text(text: str) -> str:
    """Remove page markers from displayed text (keep them only for metadata)."""
    return PAGE_MARKER.sub("", text).strip()


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


def _split_trailing_paragraph(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(head, trailing_paragraph)`` at the last blank line.

    The trailing paragraph is everything after the final blank line — the caption the
    splice attaches (single newline) immediately before a table block. ``head`` is the
    rest. With no blank line the whole input is the trailing paragraph.
    """
    matches = list(_BLANK_LINE_RE.finditer(text))
    if not matches:
        return "", text
    boundary = matches[-1]
    return text[: boundary.end()], text[boundary.end() :]


def _table_aware_parents(text: str) -> list[str]:
    """Split ``text`` into parent passages, keeping spliced tables retrievable.

    Each spliced table block (``<!-- table:<engine>:page=N:begin -->`` … ``:end -->``)
    is kept **whole** as a single parent and is **co-located with its caption** (the
    caption paragraph the splice attached right before it). A wide table otherwise both
    (a) splits mid-grid across parents and (b) is orphaned from its caption: the
    caption (e.g. "Table 2: Top-20 & Top-100 retrieval accuracy …") is the natural
    query magnet, so retrieval surfaces the caption parent while the grid parent — the
    one holding the numbers — ranks below the candidate pool and never reaches the LLM.
    Binding caption + grid into one atomic parent makes the caption child map straight
    back to the values. Non-table prose is chunked normally. See docs/DEVLOG.md
    2026-06-06.
    """
    parents: list[str] = []
    cursor = 0
    for m in TABLE_BLOCK_RE.finditer(text):
        head, caption = _split_trailing_paragraph(text[cursor : m.start()])
        if len(caption.strip()) > _MAX_ABSORBED_CAPTION_CHARS:
            head, caption = head + caption, ""  # too long to be a caption — leave it
        if head.strip():
            parents.extend(_pc_parent_splitter.split_text(head))
        block = (caption + m.group(0)).strip()
        if block:
            parents.append(block)
        cursor = m.end()
    tail = text[cursor:]
    if tail.strip():
        parents.extend(_pc_parent_splitter.split_text(tail))
    return parents


def build_parent_child_chunks(text: str, base_metadata: dict[str, Any]) -> list[Document]:
    """Produce child chunks each carrying its parent text in metadata.

    Table-aware (see ``_table_aware_parents``): spliced table blocks stay whole and
    travel with their caption, so a wide table's values stay retrievable. Documents
    without spliced tables chunk exactly as before.
    """
    parents = _table_aware_parents(text)
    children: list[Document] = []
    for parent_idx, parent_text in enumerate(parents):
        for child_idx, child_text in enumerate(_pc_child_splitter.split_text(parent_text)):
            meta = {
                **base_metadata,
                "parent_text": parent_text,
                "parent_index": parent_idx,
                "child_index": child_idx,
            }
            children.append(Document(page_content=child_text, metadata=meta))
    return children


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


def process_one_document(
    path: Path,
    db: Chroma,
    pc_db: Chroma,
    splitter: RecursiveCharacterTextSplitter,
    indexed: set[str],
) -> str:
    try:
        text = load_or_extract(path)
        if not text.strip():
            return "skipped"

        h = doc_hash(text)
        if h in indexed:
            return "skipped"

        # Split with positions tracked
        raw_chunks: list[str] = splitter.split_text(text)
        if not raw_chunks:
            return "skipped"

        documents: list[Document] = []
        cursor = 0
        for i, chunk_text in enumerate(raw_chunks):
            chunk_start = text.find(chunk_text, cursor)
            if chunk_start == -1:
                chunk_start = cursor
            cursor = chunk_start + len(chunk_text)

            extra = extract_chunk_metadata(chunk_text, text, chunk_start)

            documents.append(
                Document(
                    page_content=clean_chunk_text(chunk_text),
                    metadata={
                        "source_original": str(path),
                        "source_cache": str(get_cache_path(path)),
                        "filename": path.name,
                        "format": path.suffix.lower().lstrip("."),
                        "doc_hash": h,
                        "chunk_index": i,
                        "page": extra["page"],
                        "section": extra["section"],
                    },
                )
            )

        # The recorded chunk_count is the baseline-chunk count, *excluding* the figure
        # chunks appended below — snapshot it before they are added so committing the
        # SQLite row last (below) records the same value the original pre-figure order did.
        baseline_chunk_count = len(documents)

        pages = [int(m.group(1)) for m in PAGE_MARKER.finditer(text)]
        page_count = max(pages) if pages else None

        # Compute health classification
        from doc_assistant.health import classify_document_health

        signals = compute_health_signals(documents, text)
        health = classify_document_health(
            chunk_count=int(signals["chunk_count"]),
            avg_chunk_length=float(signals["avg_chunk_length"]),
            page_count=page_count,
            section_detection_rate=float(signals["section_detection_rate"]),
            format=path.suffix.lower().lstrip("."),
            reference_flagged_ratio=float(signals["reference_flagged_ratio"]),
        )

        # Resolve the document id WITHOUT writing the row (F1). A re-ingest reuses the
        # existing row's id so its figures + other id-keyed sidecars stay linked; a new
        # document mints a fresh UUID (figure_units() then finds none — a new doc has no
        # figures yet). The row is committed *last*, only after both Chroma writes land,
        # so a vector-write failure can never leave a committed Document row with no
        # chunks. The id is needed up front because it is stamped into every chunk's
        # metadata and is the key figure_units() queries on.
        # Coupling: ingest.py <-> db Document (this id) <-> both Chroma collections.
        document_id = _existing_document_id(h) or str(uuid4())

        # Stamp health and document_id onto chunks
        for doc in documents:
            doc.metadata["document_id"] = document_id
            doc.metadata["health"] = health.status

        # Feature 4c: append described figures as `chunk_type='figure'` chunks
        # (caption + VLM description). No-op until describe_figures has run.
        fig_units = figure_units(document_id)
        pc_base_metadata = {
            "source_original": str(path),
            "source_cache": str(get_cache_path(path)),
            "filename": path.name,
            "format": path.suffix.lower().lstrip("."),
            "doc_hash": h,
            "document_id": document_id,
            "health": health.status,
        }
        for j, (fig_text, fig_page, fig_id) in enumerate(fig_units):
            documents.append(
                Document(
                    page_content=fig_text,
                    metadata={
                        **pc_base_metadata,
                        "chunk_index": len(raw_chunks) + j,
                        "page": fig_page,
                        "section": None,
                        "chunk_type": "figure",
                        "figure_id": fig_id,
                    },
                )
            )

        # --- Vector-store writes. BOTH must land before the SQLite row is committed
        # (F1): the row write below is the last step, so an exception in either Chroma
        # write aborts the document with no orphaned Document row left behind.
        existing_baseline = db.get(where={"doc_hash": h}, include=[])
        if existing_baseline["ids"]:
            log.info("removing_existing_baseline", count=len(existing_baseline["ids"]), hash=h)
            db.delete(ids=existing_baseline["ids"])
        db.add_documents(documents)

        pc_chunks = build_parent_child_chunks(text, pc_base_metadata)

        # A figure is an atomic retrieval unit (like a kept-whole table): one
        # self-contained parent==child chunk each, appended after the prose parents.
        next_parent = max((c.metadata["parent_index"] for c in pc_chunks), default=-1) + 1
        for j, (fig_text, fig_page, fig_id) in enumerate(fig_units):
            pc_chunks.append(
                Document(
                    page_content=fig_text,
                    metadata={
                        **pc_base_metadata,
                        "parent_text": fig_text,
                        "parent_index": next_parent + j,
                        "child_index": 0,
                        "page": fig_page,
                        "chunk_type": "figure",
                        "figure_id": fig_id,
                    },
                )
            )

        existing_pc = pc_db.get(where={"doc_hash": h}, include=[])
        if existing_pc["ids"]:
            log.info("removing_existing_pc", count=len(existing_pc["ids"]), hash=h)
            pc_db.delete(ids=existing_pc["ids"])
        pc_db.add_documents(pc_chunks)

        # --- Both vector stores updated; commit the Document row + its ingestion event
        # last, keyed by the pre-resolved document_id already stamped into the chunks.
        upsert_document_in_sqlite(
            document_id=document_id,
            filename=path.name,
            source_original=str(path),
            source_cache=str(get_cache_path(path)),
            doc_hash=h,
            format=path.suffix.lower().lstrip("."),
            extractor_used=PDF_EXTRACTOR,
            chunk_count=baseline_chunk_count,
            page_count=page_count,
            extraction_health=health.status,
        )

        # Print a warning if anything's amiss
        if health.status != "healthy":
            log.warning(
                "extraction_health", status=health.status, file=path.name, reasons=health.reasons
            )

        indexed.add(h)
        return "added"
    except Exception as e:
        log.warning("document_error", file=path.name, error=str(e))
        return "error"


def _resolve_walk_root(scope: str | None) -> Path:
    """Map a --path argument to a directory or file to walk.

    Accepts an absolute path, a path relative to the CWD, or a path
    relative to DOCS_PATH. Returns the resolved path. Raises FileNotFoundError
    if nothing matches.
    """
    if scope is None:
        return DOCS_PATH
    candidates = [Path(scope), Path.cwd() / scope, DOCS_PATH / scope]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"--path '{scope}' not found (tried absolute, cwd-relative, and DOCS_PATH-relative)"
    )


def main(
    force_rebuild: bool = False,
    skip_cleanup: bool = False,
    scope: str | None = None,
) -> dict[str, int]:
    # Ensure the SQLite schema exists. Idempotent (create_all no-ops when the
    # tables are already present), so this is safe on every run and removes the
    # fresh-clone footgun of having to run migrations manually before ingest.
    init_db()

    # parents=True: the Chroma base may be a relocated ASCII path with new intermediate
    # dirs (KI-11, config._chroma_base), not just DATA_PATH/chroma.
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    Path(PC_CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    active_model = get_active_model_name()
    collection = get_collection_name(active_model)
    log.info("embedding_model", model=active_model, collection=collection)
    embeddings = get_embeddings(active_model)

    if force_rebuild:
        if scope is not None:
            raise ValueError("--rebuild and --path are mutually exclusive (rebuild is global)")
        log.warning("force_rebuild", hint="clearing vector stores and SQLite document records")
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        shutil.rmtree(PC_CHROMA_PATH, ignore_errors=True)
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        Path(PC_CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        with session_scope() as session:
            session.execute(delete(DBDocument))

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection,
    )
    pc_db = Chroma(
        persist_directory=PC_CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection,
    )

    # Orphan cleanup is global by design — skip when scoping to a subset,
    # otherwise a partial walk would falsely flag everything outside the
    # scope as missing-on-disk.
    if not skip_cleanup and not force_rebuild and scope is None:
        orphan_hashes = cleanup_orphans_sqlite(db)
        cleanup_orphans_chroma(db, orphan_hashes, also_clean_cache=True)
        cleanup_orphans_chroma(pc_db, orphan_hashes, also_clean_cache=False)
        cleanup_orphan_figures(orphan_hashes)

    # The dedup gate is the INTERSECTION of the two stores on purpose: a hash counts as
    # "already indexed" only when it is present in BOTH the baseline and the parent-child
    # collection. That self-heals a partial *Chroma* write — if a document landed in one
    # store but the other add_documents failed, the hash is missing from the intersection
    # and is reprocessed next run, completing the write. A future refactor must keep this an
    # intersection (not a union / single store) or a half-written document is treated as
    # done and never repaired.
    indexed = get_indexed_hashes(db) & get_indexed_hashes(pc_db)

    # Inverse-orphan reconciliation (the SQLite-side twin of the self-heal above). The
    # intersection gate repairs a partial *Chroma* write, but not its inverse: both vector
    # writes landing while the final upsert_document_in_sqlite commit fails leaves the hash in
    # BOTH stores (so in the intersection) with no Document row. Subtract those no-row hashes
    # from the dedup set so the document is reprocessed and its row committed on THIS run —
    # nothing is deleted (process_one_document removes+re-adds chunks idempotently). A
    # gone/content-changed no-row hash is already swept by cleanup_orphans_* above, so only the
    # source-present + unchanged shape reaches here; that one used to need `--rebuild`. The
    # warning makes the drift measurable. Runs unconditionally — the gate must stay correct even
    # under --path / --skip-cleanup. See docs/DEVLOG.md (F1).
    inverse_orphans = indexed - get_document_row_hashes()
    if inverse_orphans:
        log.warning(
            "chroma_chunks_without_document_row",
            count=len(inverse_orphans),
            hashes=sorted(inverse_orphans),
            hint="reprocessing to recommit the missing Document row(s)",
        )
        indexed -= inverse_orphans

    log.info("already_indexed", count=len(indexed))

    splitter = _make_baseline_splitter()

    walk_root = _resolve_walk_root(scope)
    if walk_root.is_file():
        files = [walk_root] if is_supported(walk_root) else []
    else:
        files = [p for p in walk_root.rglob("*") if p.is_file() and is_supported(p)]
    log.info("found_files", count=len(files), scope=str(walk_root) if scope is not None else None)

    stats: dict[str, int] = {"added": 0, "skipped": 0, "error": 0}
    for path in tqdm(files, desc="Processing"):
        result = process_one_document(path, db, pc_db, splitter, indexed)
        stats[result] += 1

    log.info(
        "ingest_complete",
        added=stats["added"],
        skipped=stats["skipped"],
        errors=stats["error"],
    )
    return stats


if __name__ == "__main__":
    import argparse

    # `python -m doc_assistant.ingest` is a program entrypoint (not library import),
    # so it configures logging here — the only place src/ does, and only when run as a
    # program. Without it the converted progress events would be silenced (ADR-003).
    from doc_assistant.logging_config import configure_logging

    configure_logging(json=config.LOG_JSON, level=config.LOG_LEVEL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the vector store and re-embed everything",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip the orphan cleanup pass",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Limit ingest to one file or subdirectory. Accepts an absolute path, "
            "a path relative to CWD, or a path relative to DOCS_PATH. "
            "Orphan cleanup is skipped when --path is set."
        ),
    )
    args = parser.parse_args()
    main(force_rebuild=args.rebuild, skip_cleanup=args.skip_cleanup, scope=args.path)
