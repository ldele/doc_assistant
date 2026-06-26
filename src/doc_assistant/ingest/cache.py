"""Extraction cache + content hashing — the bottom layer of the ingest package.

Turns a source file into its cached markdown (extracting + caching on a miss) and
hashes that content. The cached ``.md`` is the source-of-truth the rest of the
pipeline re-reads, so writes go through the atomic helper. Path/extractor config is
read dynamically (``config.X``) so a single seam is monkeypatch-able in tests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

from doc_assistant import config
from doc_assistant.extractors import extract_to_markdown
from doc_assistant.fsutil import atomic_write_text

log = structlog.get_logger(__name__)


def get_cache_path(original: Path) -> Path:
    relative = original.relative_to(config.DOCS_PATH)
    return config.CACHE_PATH / relative.with_suffix(".md")


def is_cache_fresh(original: Path, cached: Path) -> bool:
    if not cached.exists():
        return False
    return cached.stat().st_mtime >= original.stat().st_mtime


def load_or_extract(original: Path) -> str:
    cached = get_cache_path(original)
    if is_cache_fresh(original, cached):
        return cached.read_text(encoding="utf-8")

    log.info("extracting", file=original.name)
    text = extract_to_markdown(original, pdf_extractor=config.PDF_EXTRACTOR)
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
