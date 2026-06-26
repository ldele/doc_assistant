"""Small filesystem helpers shared across the cache writers.

Centralises the one contract that several writers need but each open-coded:
an **atomic** overwrite of a text file. The extraction markdown cache
(``ingest.load_or_extract``) and the two table-splice passes
(``scripts/extract_tables_marker``, ``scripts/extract_tables``) all rewrite the
cached ``.md`` *in place*, and that cache is the source-of-truth the next ingest
re-hashes. A crash mid-write would leave a half-written cache that
``ingest.is_cache_fresh`` then trusts → a corrupt re-ingest. Routing them all
through ``atomic_write_text`` is the named coupling: these three are the writers
of the ingest source-truth cache, and they share this write contract.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (write-temp-then-rename).

    Writes to a temp file in the *same directory* (so the final ``os.replace`` is a
    same-filesystem rename, atomic on POSIX and Windows), fsyncs it, then replaces
    the target. A reader therefore sees either the old file or the complete new one,
    never a partial write. If anything fails before the replace, the temp file is
    removed and the original is left untouched.

    Scope: this guarantees *atomicity* (no truncated/partial file is ever visible) — the
    truncated-cache hazard it exists to remove. It does not fsync the parent directory, so
    it is not a full power-loss *durability* guarantee for the rename itself; that failure
    mode is benign here (a lost rename leaves the previous *complete* cache, so the next
    ingest simply re-extracts), and a parent-dir fsync is unsupported on the Windows target.

    Newline handling matches ``Path.write_text`` (default text mode), so the on-disk
    bytes are unchanged from the bare ``write_text`` calls this replaces.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
