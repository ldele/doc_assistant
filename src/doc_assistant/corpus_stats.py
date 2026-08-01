"""What this corpus costs on this machine — the facts behind the Settings "Corpus" panel (ADR-037).

**Why this exists.** `docs/performance.md` answers "what does it cost at 10x" for a developer
reading the repo. A user with a growing library needs the same answer about *their* install, and
the question they actually ask is "will this still work when I have thousands of documents?". The
measured answer since ADR-036 is *yes for memory, watch your disk and your first ingest* — so this
module reports documents, chunks, disk by artifact, and which keyword-index implementation is
serving. It reports; it decides nothing.

**Facts, not copy.** The panel's sentence about memory is frontend text; what crosses the wire is
`keyword_index.mode`, because the honest sentence differs per state — the on-disk index means
memory does not grow with the library, while `unavailable` means retrieval is running on one arm
and the user has something to fix. Phrasing lives where phrasing is.

**Never blocks.** Every size is best-effort: an unreadable or absent path contributes 0 rather
than raising: a settings panel that fails to open teaches the user nothing (inform, don't block).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from doc_assistant import sparse_index
from doc_assistant.config import CACHE_PATH, CHROMA_PATH, PC_CHROMA_PATH, SQLITE_PATH

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DiskUsage:
    """Bytes per on-disk artifact, and the total. ``None`` where an artifact does not exist."""

    vector_store_bytes: int
    baseline_store_bytes: int
    keyword_index_bytes: int
    document_store_bytes: int
    extraction_cache_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class KeywordIndexState:
    """Which sparse implementation is live, and what it costs.

    ``mode`` is the load-bearing field:

    * ``on_disk`` — the ADR-036 SQLite/FTS5 index. Memory does not grow with the corpus.
    * ``unavailable`` — documents are indexed for vector search but the keyword index could not be
      opened or built, so **retrieval is running on one arm**. Answers still come back; an exact
      term the embedder does not place nearby will be missed. Rebuilding is the fix.
    * ``disabled`` — no keyword arm because there is nothing to index (an empty corpus); retrieval
      is vector-only and that is the correct, supported state.

    ``unavailable`` replaced ``in_memory`` when ADR-038 retired the legacy arm. It is not a rename:
    the old value meant "working, but the expensive way" and needed no user action, while this one
    means a capability is *missing*. The reason the state has to be reported at all is that
    retiring the fallback removed the silent recovery that used to absorb it.
    """

    mode: str
    bytes: int | None
    built_at: str | None


@dataclass(frozen=True)
class CorpusStats:
    """Everything the Corpus panel shows. Serialised straight to the settings payload."""

    documents: int
    chunks: int
    disk: DiskUsage
    keyword_index: KeywordIndexState

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tree_bytes(path: str | Path) -> int:
    """Total size under ``path`` (a file or a directory), or 0 if it cannot be read.

    Walks with ``os.scandir`` and reads sizes from the directory entries, so a Chroma store of a
    few thousand files costs one stat per file and no reads. Errors are swallowed per entry: a
    locked or vanishing file during a walk should cost that file's bytes from the total, not the
    whole panel.
    """
    target = Path(path)
    try:
        if target.is_file():
            return target.stat().st_size
        if not target.is_dir():
            return 0
    except OSError:
        return 0

    total = 0
    stack = [target]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat().st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _keyword_index_state(*, on_disk: bool, chunks: int) -> KeywordIndexState:
    path = sparse_index.index_path(PC_CHROMA_PATH)
    if not on_disk:
        # No keyword arm. The two causes need different words: chunks with no index is a
        # degradation the user should act on, while no chunks is simply an empty library.
        # Size and build time are deliberately withheld — a stale file may well still be on disk,
        # and reporting its bytes would describe an index that is not serving anything.
        mode = "unavailable" if chunks else "disabled"
        return KeywordIndexState(mode=mode, bytes=None, built_at=None)
    try:
        stat = path.stat()
    except OSError:
        return KeywordIndexState(mode="on_disk", bytes=None, built_at=None)
    built = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return KeywordIndexState(mode="on_disk", bytes=stat.st_size, built_at=built)


def corpus_stats(*, documents: int, chunks: int, keyword_index_on_disk: bool) -> CorpusStats:
    """Assemble the panel's facts.

    ``keyword_index_on_disk`` is passed in rather than discovered: the answer is a property of the
    **live pipeline** (whether it actually wired a keyword arm at construction), not of whether a
    file happens to exist on disk. That distinction outlived the legacy arm it was written for — a
    stale index file left behind by a failed build would otherwise report the reassuring answer
    while the process retrieves on the vector arm alone.
    """
    vector = _tree_bytes(PC_CHROMA_PATH)
    baseline = _tree_bytes(CHROMA_PATH)
    index = _tree_bytes(sparse_index.index_path(PC_CHROMA_PATH))
    documents_db = _tree_bytes(SQLITE_PATH)
    cache = _tree_bytes(CACHE_PATH)
    return CorpusStats(
        documents=documents,
        chunks=chunks,
        disk=DiskUsage(
            vector_store_bytes=vector,
            baseline_store_bytes=baseline,
            keyword_index_bytes=index,
            document_store_bytes=documents_db,
            extraction_cache_bytes=cache,
            total_bytes=vector + baseline + index + documents_db + cache,
        ),
        keyword_index=_keyword_index_state(on_disk=keyword_index_on_disk, chunks=chunks),
    )
