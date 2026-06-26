"""Orphan detection + cross-store cleanup for ingest.

Removes what a no-longer-current document leaves behind across the three stores:
stale/gone SQLite rows, their Chroma chunks (+ optionally the cached ``.md``), and
the on-disk figure PNG dirs. Depends only on the ``cache`` layer (re-hashes a source
to detect a content change).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import structlog
from langchain_chroma import Chroma
from sqlalchemy import select

from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.session import session_scope

from .cache import doc_hash, load_or_extract
from .figures import figure_dir

log = structlog.get_logger(__name__)


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
