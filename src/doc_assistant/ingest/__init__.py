"""Ingestion pipeline package.

Turns source documents (PDF/EPUB/HTML/DOCX/MD) into retrievable chunks across the
two Chroma collections + the SQLite Document row, with orphan cleanup and
partial-write self-heal. The former monolithic ``ingest.py`` is split into cohesive
layers:

* ``cache``    — extraction cache + content hashing (bottom layer)
* ``chunking`` — text → parent/child chunks, metadata, health signals (pure)
* ``store``    — SQLite + Chroma read/write helpers (data access)
* ``cleanup``  — orphan detection + cross-store cleanup
* (this module) — ``process_one_document`` / ``main`` orchestration; the CLI lives in ``__main__``

The names in ``__all__`` are re-exported so ``from doc_assistant.ingest import …``
keeps working unchanged after the split. Path/model config is read dynamically via
``config.X`` so tests monkeypatch one seam (``config``) for all layers.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import structlog
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, func, select
from tqdm import tqdm

from doc_assistant import config
from doc_assistant.db.migrations import init_db
from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.models import document_folders
from doc_assistant.db.session import session_scope
from doc_assistant.embeddings import (
    get_active_model_name,
    get_collection_name,
    get_embeddings,
)
from doc_assistant.extractors import is_supported

from .cache import doc_hash, get_cache_path, is_cache_fresh, load_or_extract
from .chunking import (
    PAGE_MARKER,
    _make_baseline_splitter,
    _make_child_splitter,
    _make_parent_splitter,
    build_parent_child_chunks,
    clean_chunk_text,
    compute_health_signals,
    extract_chunk_metadata,
)
from .cleanup import (
    _find_orphan_hashes,
    cleanup_orphan_figures,
    cleanup_orphans_chroma,
    cleanup_orphans_sqlite,
)
from .store import (
    _existing_document_id,
    figure_units,
    get_document_row_hashes,
    get_indexed_hashes,
    upsert_document_in_sqlite,
)

log = structlog.get_logger(__name__)

__all__ = [
    # markers / config-driven splitter factories (chunking)
    "PAGE_MARKER",
    "_existing_document_id",
    # cleanup
    "_find_orphan_hashes",
    "_make_baseline_splitter",
    "_make_child_splitter",
    "_make_parent_splitter",
    "build_parent_child_chunks",
    "clean_chunk_text",
    "cleanup_orphan_figures",
    "cleanup_orphans_chroma",
    "cleanup_orphans_sqlite",
    "compute_health_signals",
    # cache + hashing
    "doc_hash",
    "extract_chunk_metadata",
    "figure_units",
    # embeddings passthrough (used by callers/tests via the ingest namespace)
    "get_active_model_name",
    "get_cache_path",
    "get_collection_name",
    "get_document_row_hashes",
    "get_embeddings",
    # store helpers
    "get_indexed_hashes",
    "is_cache_fresh",
    # orchestration (defined here)
    "load_documents",
    "load_or_extract",
    "main",
    "process_one_document",
    "upsert_document_in_sqlite",
]


def load_documents() -> list[Document]:
    documents: list[Document] = []
    files = [p for p in config.DOCS_PATH.rglob("*") if p.is_file() and is_supported(p)]
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
        # Coupling: ingest <-> db Document (this id) <-> both Chroma collections.
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
            extractor_used=config.PDF_EXTRACTOR,
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
        return config.DOCS_PATH
    candidates = [Path(scope), Path.cwd() / scope, config.DOCS_PATH / scope]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"--path '{scope}' not found (tried absolute, cwd-relative, and DOCS_PATH-relative)"
    )


def _drop_excluded(walked: list[Path]) -> tuple[list[Path], int]:
    """Drop files flagged ``excluded`` in the registry from an implicit walk (Decision 5).

    Exclusions are keyed by rel_path under the configured source dir; a walked file outside that
    dir (a ``--path`` elsewhere) can't match an exclusion and is kept. Returns ``(kept, skipped)``.
    Registry / app_settings are imported lazily to keep the locked core's top-level imports intact.
    """
    from doc_assistant import app_settings
    from doc_assistant.ingest import registry

    with session_scope() as session:
        excluded = registry.excluded_rel_paths(session)
    if not excluded:
        return walked, 0
    source_dir = app_settings.get_source_dir().resolve()
    kept: list[Path] = []
    skipped = 0
    for f in walked:
        try:
            rel = f.resolve().relative_to(source_dir).as_posix()
        except ValueError:
            rel = ""
        if rel and rel in excluded:
            skipped += 1
        else:
            kept.append(f)
    return kept, skipped


def _resolve_ingest_files(scope: str | None, files: list[Path] | None) -> tuple[list[Path], int]:
    """Resolve the files to ingest + the excluded-skipped count (feature-selective-ingestion.md).

    ``files`` given → used as-is (an explicit selection; exclusions already applied or overridden
    upstream). Otherwise walk ``_resolve_walk_root(scope)`` as before, then subtract standing
    exclusions.
    """
    if files is not None:
        return list(files), 0
    walk_root = _resolve_walk_root(scope)
    if walk_root.is_file():
        walked = [walk_root] if is_supported(walk_root) else []
    else:
        walked = [p for p in walk_root.rglob("*") if p.is_file() and is_supported(p)]
    return _drop_excluded(walked)


def _dry_run_plan(scope: str | None, files: list[Path] | None) -> dict[str, int]:
    """The dry-run plan (Decision 6): resolve + classify stat-only, report — never loads
    embeddings or opens Chroma, so a monkeypatched `get_embeddings` trap stays untouched."""
    from doc_assistant.ingest import registry

    to_process, excluded_skipped = _resolve_ingest_files(scope, files)
    with session_scope() as session:
        plan = registry.plan_files(session, to_process)
    plan["excluded"] = excluded_skipped
    log.info("dry_run_plan", **plan)
    return plan


def main(
    force_rebuild: bool = False,
    skip_cleanup: bool = False,
    scope: str | None = None,
    files: list[Path] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    # Selective ingestion (feature-selective-ingestion.md, S1). `files` is an explicit,
    # already-validated absolute path list (from `registry.resolve_selection` for the API, or CLI
    # `--files`) — mutually exclusive with the walk-scoping flags. `dry_run` reports the plan
    # without loading embeddings or opening Chroma (Decision 6).
    if files is not None and (scope is not None or force_rebuild):
        raise ValueError("files= is mutually exclusive with --path and --rebuild")

    # Ensure the SQLite schema exists. Idempotent (create_all no-ops when the
    # tables are already present), so this is safe on every run and removes the
    # fresh-clone footgun of having to run migrations manually before ingest.
    init_db()

    if dry_run:
        return _dry_run_plan(scope, files)

    # parents=True: the Chroma base may be a relocated ASCII path with new intermediate
    # dirs (KI-11, config._chroma_base), not just DATA_PATH/chroma.
    config.CACHE_PATH.mkdir(parents=True, exist_ok=True)
    Path(config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.PC_CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    active_model = get_active_model_name()
    collection = get_collection_name(active_model)
    log.info("embedding_model", model=active_model, collection=collection)
    embeddings = get_embeddings(active_model)

    if force_rebuild:
        if scope is not None:
            raise ValueError("--rebuild and --path are mutually exclusive (rebuild is global)")
        log.warning("force_rebuild", hint="clearing vector stores and SQLite document records")
        shutil.rmtree(config.CHROMA_PATH, ignore_errors=True)
        shutil.rmtree(config.PC_CHROMA_PATH, ignore_errors=True)
        Path(config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        Path(config.PC_CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        with session_scope() as session:
            # Deleting the Document rows cascades through `document_folders` (ON DELETE CASCADE),
            # so EVERY folder loses its members and is left as an empty shell — a data loss the
            # user cannot otherwise see. Naming the count is the honest minimum until the proper
            # fix (snapshot + restore) lands; the demo folder is the only one that repopulates
            # itself, because a rebuild makes every document look newly ingested (ADR-025 F3,
            # spec M3/M9). See `.claude/KNOWN_ISSUES.md`.
            membership_q = select(func.count()).select_from(document_folders)
            memberships = session.execute(membership_q).scalar()
            if memberships:
                log.warning(
                    "rebuild_clears_folder_membership",
                    memberships=int(memberships),
                    hint="folders survive but are emptied; re-add by hand "
                    "(the demo folder refills itself)",
                )
            session.execute(delete(DBDocument))

    db = Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection,
    )
    pc_db = Chroma(
        persist_directory=config.PC_CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection,
    )

    # Orphan cleanup is global by design — skip when scoping to a subset (--path or an explicit
    # `files=` selection), otherwise a partial walk would falsely flag everything outside the
    # scope as missing-on-disk.
    if not skip_cleanup and not force_rebuild and scope is None and files is None:
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

    to_process, excluded_skipped = _resolve_ingest_files(scope, files)
    log.info(
        "found_files",
        count=len(to_process),
        scope=str(_resolve_walk_root(scope)) if scope is not None else None,
        excluded_skipped=excluded_skipped,
    )

    # ADR-025 F3 — the demo auto-assign trigger is "a Document row that did not exist before this
    # run", captured as a set-difference around the loop. Deliberately NOT keyed on
    # process_one_document's "added": that is also returned for *re*-ingests (the inverse-orphan
    # repair above, a --path rerun), and re-assigning an existing document would re-fight a
    # membership the user edited by hand. Snapshotting *here* — after the --rebuild wipe — makes a
    # rebuild repopulate the demo folder it just emptied, which is the one honest exception
    # (spec M1/M3, docs/specs/feature-corpus-folders-demo.md).
    rows_before = get_document_row_hashes()

    stats: dict[str, int] = {"added": 0, "skipped": 0, "error": 0}
    for path in tqdm(to_process, desc="Processing"):
        result = process_one_document(path, db, pc_db, splitter, indexed)
        stats[result] += 1

    _assign_demo_folder(get_document_row_hashes() - rows_before)

    log.info(
        "ingest_complete",
        added=stats["added"],
        skipped=stats["skipped"],
        errors=stats["error"],
    )
    return stats


def _assign_demo_folder(new_hashes: set[str]) -> None:
    """Put newly-ingested demo-manifest files into the demo folder (ADR-025 F3).

    Never fails an ingest that otherwise succeeded: the documents are indexed and answerable
    either way, so a folder-assignment problem is a warning, not a failure. Imported lazily to
    keep the locked ingest core's top-level imports intact (the ``_drop_excluded`` precedent).
    """
    if not new_hashes:
        return
    try:
        from doc_assistant import demo_corpus

        demo_corpus.assign_new_documents(new_hashes)
    except Exception as e:  # inform, never block an ingest that otherwise succeeded
        log.warning("demo_folder_assign_failed", error=str(e))
