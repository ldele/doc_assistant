import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from doc_assistant.figures import figure_chunk_text
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
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def doc_hash(text: str) -> str:
    """Content-only hash. Path-independent so documents survive moves/renames."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def get_indexed_hashes(db: Chroma) -> set[str]:
    data = db.get(include=["metadatas"])
    return {m.get("doc_hash") for m in data["metadatas"] if m and m.get("doc_hash")}


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


def upsert_document_in_sqlite(
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
    """Create or update a Document row in SQLite. Returns the document's UUID.

    If a document with the same doc_hash exists, return its ID and log a
    re-ingestion event. Otherwise create a new Document row.
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
            # New document
            document = DBDocument(
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
            session.flush()  # generate the UUID

            event = IngestionEvent(
                document_id=document.id,
                event_type="extract",
                extractor=extractor_used,
                chunks_produced=chunk_count,
                health_status=extraction_health,
            )
            session.add(event)
            return str(document.id)


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

        document_id = upsert_document_in_sqlite(
            filename=path.name,
            source_original=str(path),
            source_cache=str(get_cache_path(path)),
            doc_hash=h,
            format=path.suffix.lower().lstrip("."),
            extractor_used=PDF_EXTRACTOR,
            chunk_count=len(documents),
            page_count=page_count,
            extraction_health=health.status,
        )

        # Print a warning if anything's amiss
        if health.status != "healthy":
            log.warning(
                "extraction_health", status=health.status, file=path.name, reasons=health.reasons
            )

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

    indexed = get_indexed_hashes(db) & get_indexed_hashes(pc_db)
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
