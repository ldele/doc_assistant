"""Selective-ingestion source registry (feature-selective-ingestion.md, S1).

Owns *which files enter* the locked primary ingest path — it never extracts, hashes,
chunks, or touches Chroma / the markdown cache. Pre-ingest bookkeeping in the library
SQLite (`SourceFile`), not an enrichment sidecar (it derives nothing from content).

House split:
- **Pure core (this half):** status derivation + selection validation — no I/O, exhaustively
  unit-tested.
- **Impure boundary (below):** the stat-only scan, the PATCH seam, selection resolution — the
  only functions that touch the session or the filesystem.

`doc_type` is dormant in v1 (grill lock 2026-07-15): the `SourceFile.doc_type` column exists but
nothing seeds/reads/writes it, so there is deliberately **no `default_doc_type` seeding function
here yet** — it lands with the column's activation, not before (no dead code).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.models import SourceFile, _utcnow
from doc_assistant.extractors import SUPPORTED_EXTENSIONS, is_supported
from doc_assistant.ingest.cache import get_cache_path, is_cache_fresh

log = structlog.get_logger(__name__)

# Derived ingestion status (Decision 3). Persisted nowhere — computed at read time.
STATUS_NEW = "new"
STATUS_CHANGED = "changed"
STATUS_INGESTED = "ingested"
STATUS_MISSING = "missing"

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def derive_status(file_exists: bool, cache_fresh: bool, has_document: bool) -> str:
    """The pure status truth table (Decision 3) — 8 input combos → 4 statuses.

    - not ``file_exists``                      → ``missing``  (a row with no file on disk)
    - file, no ``Document`` row                → ``new``      (never successfully ingested)
    - file, ``Document`` row, cache fresh      → ``ingested``
    - file, ``Document`` row, cache stale      → ``changed``  (source newer than cache → re-embed)

    ``cache_fresh`` collapses "no cache entry" and "stale cache entry" into one ``False``: both
    mean the markdown cache does not reflect the current source bytes. A file with no ``Document``
    row is ``new`` regardless of an incidental cache (it has no retrievable chunks yet).
    """
    if not file_exists:
        return STATUS_MISSING
    if not has_document:
        return STATUS_NEW
    return STATUS_INGESTED if cache_fresh else STATUS_CHANGED


class InvalidSelection(ValueError):
    """A requested selection held unusable rel_paths, grouped by reason.

    Carries ``offenders`` (reason → the raw paths at fault) so the API layer can turn it into a
    400 that names every offender in one response.
    """

    def __init__(self, offenders: dict[str, list[str]]) -> None:
        self.offenders = {reason: paths for reason, paths in offenders.items() if paths}
        detail = "; ".join(
            f"{reason}: {', '.join(paths)}" for reason, paths in self.offenders.items()
        )
        super().__init__(f"invalid selection — {detail}")


def _normalize_rel(raw: str) -> str:
    """POSIX-normalize a requested rel_path. Does NOT resolve against the filesystem.

    Backslashes → forward slashes (so a Windows-style path validates the same), trims a single
    leading ``./``. A leading ``..`` is preserved on purpose — traversal is caught, not silently
    stripped.
    """
    s = raw.strip().replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s


def validate_selection(requested: list[str], known: set[str]) -> list[str]:
    """Normalize + validate requested rel_paths against the known registry set (pure).

    Returns the normalized rel_paths (first-seen order, de-duplicated) when all are valid;
    otherwise raises `InvalidSelection` listing every offender. Each path is categorized by its
    first failing check in priority order: **absolute** path → ``..`` **traversal** →
    **unsupported** suffix → **unknown** rel_path (not in the scanned registry).
    """
    absolute: list[str] = []
    traversal: list[str] = []
    unsupported: list[str] = []
    unknown: list[str] = []
    valid: list[str] = []
    seen: set[str] = set()

    for raw in requested:
        norm = _normalize_rel(raw)
        pp = PurePosixPath(norm)
        if norm.startswith("/") or pp.is_absolute() or _DRIVE_RE.match(norm):
            absolute.append(raw)
        elif ".." in pp.parts:
            traversal.append(raw)
        elif pp.suffix.lower() not in SUPPORTED_EXTENSIONS:
            unsupported.append(raw)
        elif norm not in known:
            unknown.append(raw)
        elif norm not in seen:
            seen.add(norm)
            valid.append(norm)

    offenders = {
        "absolute": absolute,
        "traversal": traversal,
        "unsupported": unsupported,
        "unknown": unknown,
    }
    if any(offenders.values()):
        raise InvalidSelection(offenders)
    return valid


# ============================================================
# Impure boundary — the only functions that touch the session or the filesystem.
# ============================================================


@dataclass(frozen=True)
class SourceView:
    """One registry row as seen by the API/CLI: identity + derived status + user intent.

    ``doc_type`` is always ``None`` in v1 (the dormant column) — carried so the wire shape is
    forward-stable when it graduates.
    """

    rel_path: str
    format: str
    size: int
    mtime: float
    status: str
    excluded: bool
    doc_type: str | None


def _pathkey(p: Path | str) -> str:
    """A comparison key for an absolute source path — case-normalized, separator-normalized.

    Reconciles the few ways `Document.source_original` may have been stored (resolved vs not) with
    the scanned absolute path, without a symlink-resolving filesystem read.
    """
    return os.path.normcase(os.path.abspath(str(p)))


def _document_source_keys(session: Session) -> set[str]:
    """The normalized absolute paths of every ingested `Document`, for the `has_document` join."""
    return {_pathkey(s) for s in session.execute(select(DBDocument.source_original)).scalars()}


def _cache_is_fresh(file: Path) -> bool:
    """`is_cache_fresh` guarded for a source file outside the cache root (custom source dir).

    `get_cache_path` is `config.DOCS_PATH`-relative; a file elsewhere has no resolvable cache, so
    treat it as not-fresh (it derives `new`/`changed`) rather than crash the whole scan.
    """
    try:
        cached = get_cache_path(file)
    except ValueError:
        return False
    return is_cache_fresh(file, cached)


def scan_sources(session: Session, source_dir: Path) -> list[SourceView]:
    """Stat-only walk of ``source_dir``: upsert rows, refresh ``last_seen``, derive each status.

    No extraction, hashing, or content reads — listing a large corpus is instant. A file that has
    vanished keeps its row (it derives ``missing``); a re-appeared file refreshes in place. Returns
    every row (present + missing), rel_path-sorted, with a freshly derived status.
    """
    now = _utcnow()
    root = source_dir.resolve()
    on_disk: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and is_supported(p):
            on_disk[p.relative_to(root).as_posix()] = p

    rows: dict[str, SourceFile] = {
        r.rel_path: r for r in session.execute(select(SourceFile)).scalars()
    }
    doc_keys = _document_source_keys(session)

    for rel, path in on_disk.items():
        stat = path.stat()
        fmt = path.suffix.lower().lstrip(".")
        row = rows.get(rel)
        if row is None:
            row = SourceFile(
                rel_path=rel,
                format=fmt,
                size=stat.st_size,
                mtime=stat.st_mtime,
                first_seen=now,
                last_seen=now,
            )
            session.add(row)
            rows[rel] = row
        else:
            row.format = fmt
            row.size = stat.st_size
            row.mtime = stat.st_mtime
            row.last_seen = now
    session.flush()

    views: list[SourceView] = []
    for rel, row in sorted(rows.items()):
        disk_path = on_disk.get(rel)
        if disk_path is None:
            status = derive_status(False, False, False)
        else:
            status = derive_status(
                True, _cache_is_fresh(disk_path), _pathkey(disk_path) in doc_keys
            )
        views.append(
            SourceView(
                rel_path=rel,
                format=row.format,
                size=row.size,
                mtime=row.mtime,
                status=status,
                excluded=row.excluded,
                doc_type=row.doc_type,
            )
        )
    return views


def set_source_meta(
    session: Session, rel_path: str, *, excluded: bool | None = None
) -> SourceFile:
    """PATCH seam: update user intent on one registry row. v1 sets ``excluded`` only.

    Raises ``KeyError(rel_path)`` for an unknown row (the API maps that to 404). ``doc_type`` is
    intentionally not a parameter yet (dormant column) — it lands with the facet's activation.
    """
    row = session.execute(
        select(SourceFile).where(SourceFile.rel_path == rel_path)
    ).scalar_one_or_none()
    if row is None:
        raise KeyError(rel_path)
    if excluded is not None:
        row.excluded = excluded
    session.flush()
    return row


def view_for(session: Session, source_dir: Path, rel_path: str) -> SourceView | None:
    """The current `SourceView` for one rel_path (freshly derived status), or ``None`` if no row.

    Used by ``PATCH /api/sources`` to echo the updated row without a full re-scan.
    """
    row = session.execute(
        select(SourceFile).where(SourceFile.rel_path == rel_path)
    ).scalar_one_or_none()
    if row is None:
        return None
    disk_path = source_dir.resolve() / rel_path
    if disk_path.is_file():
        status = derive_status(
            True, _cache_is_fresh(disk_path), _pathkey(disk_path) in _document_source_keys(session)
        )
    else:
        status = STATUS_MISSING
    return SourceView(
        rel_path=row.rel_path,
        format=row.format,
        size=row.size,
        mtime=row.mtime,
        status=status,
        excluded=row.excluded,
        doc_type=row.doc_type,
    )


def excluded_rel_paths(session: Session) -> set[str]:
    """The rel_paths currently flagged ``excluded`` — what an implicit ingest walk must skip."""
    return {
        r.rel_path
        for r in session.execute(select(SourceFile).where(SourceFile.excluded.is_(True))).scalars()
    }


def plan_files(session: Session, files: list[Path]) -> dict[str, int]:
    """Stat-only ingest plan for a list of on-disk files (Decision 6, dry-run).

    Returns ``{would_add, would_reembed, skip_unchanged}`` using the same status signals as
    `scan_sources` (Document rows + cache freshness) — never loads embeddings or opens Chroma.
    """
    doc_keys = _document_source_keys(session)
    plan = {"would_add": 0, "would_reembed": 0, "skip_unchanged": 0}
    for f in files:
        status = derive_status(True, _cache_is_fresh(f), _pathkey(f) in doc_keys)
        if status == STATUS_NEW:
            plan["would_add"] += 1
        elif status == STATUS_CHANGED:
            plan["would_reembed"] += 1
        else:  # ingested
            plan["skip_unchanged"] += 1
    return plan


def resolve_selection(
    session: Session, source_dir: Path, requested: list[str] | None
) -> list[Path]:
    """Turn a selection predicate into explicit absolute paths for `ingest.main(files=…)`.

    - ``requested is None`` → every supported file on disk *minus* the ``excluded`` ones (an
      implicit walk honors the standing exclusions; the skipped count is logged).
    - a list → `validate_selection` against what is actually on disk (not the possibly-stale
      registry), then absolute paths. An **explicit** pick **overrides** ``excluded`` (Decision 5),
      logged. Raises `InvalidSelection` (→ API 400) if any path is unusable.
    """
    root = source_dir.resolve()
    on_disk: dict[str, Path] = {
        p.relative_to(root).as_posix(): p
        for p in root.rglob("*")
        if p.is_file() and is_supported(p)
    }
    excluded = excluded_rel_paths(session)

    if requested is None:
        kept: list[Path] = []
        skipped = 0
        for rel, path in sorted(on_disk.items()):
            if rel in excluded:
                skipped += 1
                continue
            kept.append(path)
        if skipped:
            log.info("excluded_skipped", count=skipped)
        return kept

    valid = validate_selection(requested, set(on_disk))
    overridden = [rel for rel in valid if rel in excluded]
    if overridden:
        log.info("excluded_overridden_by_explicit_selection", paths=overridden)
    return [on_disk[rel] for rel in valid]
