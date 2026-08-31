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

from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceFile, SourceRoot, _utcnow
from doc_assistant.db.models import Document as DBDocument
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
    #: ADR-046 (AD3b) — which root this row belongs to. `rel_path` is relative to *this* root, so
    #: the pair is the identity; `rel_path` alone stopped being unique when a second root existed.
    root_id: str = LIBRARY_ROOT_ID
    #: ``library`` or ``referenced`` — what delete branches on (ADR-014 as amended by ADR-046).
    root_kind: str = "library"
    #: False when the row's root is unreachable right now (unplugged drive, offline share). Its
    #: files still derive ``missing``, but this says *why* — see `RootView.available`.
    root_available: bool = True


@dataclass(frozen=True)
class RootView:
    """One registered root plus whether it can be reached **right now**.

    ``available`` is derived, never stored (see `db.models.SourceRoot`): a drive that is
    unplugged this second may be back the next, so persisting it would create a second truth that
    goes stale silently. It is the difference between *"your 400 documents were deleted"* and
    *"the drive holding them is not connected"* — the honest-degradation contract makes that
    distinction the app's job, not the user's to infer from 400 identical `missing` badges.
    """

    id: str
    path: str
    kind: str
    available: bool


def pathkey(p: Path | str) -> str:
    """Public alias of `_pathkey` — the comparison key callers need to test `excluded_paths`."""
    return _pathkey(p)


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
    """`is_cache_fresh`, guarded so one unresolvable path cannot crash a whole scan.

    Since AD3b `get_cache_path` resolves for a file anywhere on disk — a referenced file gets a
    digest-keyed entry — so this no longer papers over "outside the library folder", which used to
    make every referenced file read as `new` forever. The guard stays because the scan runs over
    whatever is on disk and one pathological path should degrade to `new`/`changed`, not take the
    listing down with it.
    """
    try:
        cached = get_cache_path(file)
    except ValueError:
        return False
    return is_cache_fresh(file, cached)


def source_key(root_id: str, rel_path: str) -> str:
    """The wire/selection identifier for one registered file: ``"<root_id>:<rel_path>"``.

    A single opaque string rather than a pair because it travels through `validate_selection`,
    the ingest `paths` list and the desktop client, all of which handle one identifier per file.
    A bare `rel_path` remains a legal shorthand for the library root, so every caller written
    before AD3b keeps working unchanged — see `split_key`.
    """
    return f"{root_id}:{rel_path}"


def split_key(key: str, known_root_ids: set[str]) -> tuple[str, str]:
    """Inverse of `source_key`, tolerant of the bare-`rel_path` shorthand.

    The prefix before the first ``:`` is treated as a root id **only when it is actually one**.
    That check is what makes the shorthand unambiguous: a POSIX filename may legally contain a
    colon, so splitting blindly would turn `my:notes.pdf` into root ``my``. Unknown prefix →
    the whole string is a library-root rel_path.
    """
    head, sep, tail = key.partition(":")
    if sep and head in known_root_ids:
        return head, tail
    return LIBRARY_ROOT_ID, key


def ensure_library_root(session: Session, source_dir: Path) -> SourceRoot:
    """Get the one ``library`` root, creating it or refreshing its path.

    The path is **updated in place** rather than re-seeded when the user moves their library
    (`app_settings.get_source_dir()` changes): the row's id is what every `SourceFile.root_id`
    points at, so replacing the row would orphan all of them. This is the only writer of the
    library root's path, and it runs on every scan, so the row cannot drift from the setting.
    """
    root = session.get(SourceRoot, LIBRARY_ROOT_ID)
    path = str(source_dir.resolve())
    if root is None:
        root = SourceRoot(id=LIBRARY_ROOT_ID, path=path, kind="library", added_at=_utcnow())
        session.add(root)
        session.flush()
    elif root.path != path:
        log.info("library_root_moved", was=root.path, now=path)
        root.path = path
        session.flush()
    return root


def register_root(session: Session, path: Path) -> SourceRoot:
    """Get (or create) the ``referenced`` root for a folder the user keeps their own files in.

    Idempotent on the resolved path, so referencing a second file from a folder already
    registered reuses the root instead of minting a duplicate. Never returns the library root:
    a path *inside* the library folder is a copy-in, not a reference, and `library/add.py`
    rejects that before it gets here.
    """
    resolved = str(path.resolve())
    existing = session.execute(select(SourceRoot).where(SourceRoot.kind == "referenced")).scalars()
    for root in existing:
        if _pathkey(root.path) == _pathkey(resolved):
            return root
    root = SourceRoot(path=resolved, kind="referenced", added_at=_utcnow())
    session.add(root)
    session.flush()
    return root


def root_containing(session: Session, path: Path) -> SourceRoot | None:
    """The registered ``referenced`` root this file already sits under, if there is one.

    Reference-adding a file registers a root for its parent directory, which is right when the
    user hands over a folder of papers and wrong when they hand over a *catalogue*: Zotero keeps
    every attachment in its own `storage/<key>/` directory, so per-parent would mint one root per
    document — five hundred rows, each stat-ed on every scan, for one library. Preferring a root
    that already contains the file fixes that without guessing how far up the user meant, because
    the ancestor is one they (or an import on their behalf) established explicitly.

    The deepest match wins, so a nested root registered later still owns its own files. The
    library root is not a candidate: a file inside the library folder is handled before this is
    reached, and a *referenced* file must never be adopted by the root the app owns.
    """
    key = _pathkey(path)
    best: SourceRoot | None = None
    for root in session.execute(
        select(SourceRoot).where(SourceRoot.kind == "referenced")
    ).scalars():
        prefix = _pathkey(root.path)
        # `os.path.join(prefix, "")` appends the platform separator, so `C:\Papers` does not
        # swallow `C:\PapersArchive` — a plain `startswith` on the bare prefix would.
        contained = key == prefix or key.startswith(os.path.join(prefix, ""))
        if contained and (best is None or len(_pathkey(best.path)) < len(prefix)):
            best = root
    return best


def _root_available(root: SourceRoot) -> bool:
    """Can this root be reached right now? A stat, not a walk — it runs before every scan.

    `Path.is_dir()` on a disconnected network share can block, but it is the same call the scan
    itself would make one line later; doing it here means the answer is *"the root is gone"*
    rather than *"every file under it vanished at once"*.
    """
    try:
        return Path(root.path).is_dir()
    except OSError:  # unreachable UNC path, permission denied on the mount point
        return False


def list_roots(session: Session, source_dir: Path) -> list[RootView]:
    """Every registered root, library first, each with its live availability."""
    ensure_library_root(session, source_dir)
    roots = session.execute(select(SourceRoot)).scalars().all()
    ordered = sorted(roots, key=lambda r: (r.kind != "library", r.path))
    return [
        RootView(id=r.id, path=r.path, kind=r.kind, available=_root_available(r)) for r in ordered
    ]


def scan_root(
    session: Session, root: SourceRoot, *, available: bool | None = None
) -> list[SourceView]:
    """Stat-only walk of one root: upsert its rows, refresh ``last_seen``, derive each status.

    No extraction, hashing, or content reads — listing a large corpus is instant. A file that has
    vanished keeps its row (it derives ``missing``); a re-appeared file refreshes in place.

    **An unavailable root is not walked at all.** Its rows are returned untouched, still deriving
    ``missing`` but carrying ``root_available=False``, and critically their ``last_seen`` is *not*
    refreshed and no row is deleted — an unplugged drive must not look like a deletion, and the
    rows have to survive it being plugged back in.
    """
    now = _utcnow()
    rows: dict[str, SourceFile] = {
        r.rel_path: r
        for r in session.execute(select(SourceFile).where(SourceFile.root_id == root.id)).scalars()
    }
    reachable = _root_available(root) if available is None else available

    on_disk: dict[str, Path] = {}
    if reachable:
        base = Path(root.path).resolve()
        for p in base.rglob("*"):
            if p.is_file() and is_supported(p):
                on_disk[p.relative_to(base).as_posix()] = p

        # ⚠ `origin` decides whether delete may bin the file (ADR-014 as amended by ADR-046), so
        # a scan must state it rather than fall through to the column DEFAULT — which is
        # ``'copied'``, and would mark **every file the walk discovers under a referenced root**
        # as one the app owns. Measured before this line existed: referencing one paper out of a
        # Zotero folder registered that folder as a root, and the next scan claimed ownership of
        # every other document in it.
        scanned_origin = "copied" if root.kind == "library" else "referenced"

        for rel, path in on_disk.items():
            stat = path.stat()
            fmt = path.suffix.lower().lstrip(".")
            row = rows.get(rel)
            if row is None:
                row = SourceFile(
                    root_id=root.id,
                    rel_path=rel,
                    format=fmt,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    origin=scanned_origin,
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
                # Repair the impossible combination, and only that one. ``copied`` under a
                # referenced root cannot be true — `library/add.py` writes ``copied`` only with
                # the library root — so a row saying it was written by the DEFAULT, before the
                # line above existed. The reverse is legal and never touched: a file already
                # sitting inside the library folder registers under the *library* root with
                # ``origin='referenced'`` (`_reference_target`), because the app did not put it
                # there. Self-healing on the next scan rather than a migration, since the scan is
                # what owns these rows.
                if root.kind != "library" and row.origin == "copied":
                    log.info("source_origin_repaired", root=root.path, rel_path=rel)
                    row.origin = "referenced"
        session.flush()

    doc_keys = _document_source_keys(session)
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
                root_id=root.id,
                root_kind=root.kind,
                root_available=reachable,
            )
        )
    return views


def scan_sources(session: Session, source_dir: Path) -> list[SourceView]:
    """Scan **every** registered root and return all rows, library root first (ADR-046, AD3b).

    Was a single-root walk of ``source_dir`` until AD3b; ``source_dir`` now names the *library*
    root specifically, which this keeps current via `ensure_library_root`. Ordering is
    (library-first, root path, rel_path) so the list is stable across calls rather than
    dict-ordered.
    """
    views: list[SourceView] = []
    for rv in list_roots(session, source_dir):
        root = session.get(SourceRoot, rv.id)
        if root is not None:
            views.extend(scan_root(session, root, available=rv.available))
    return views


def set_source_meta(
    session: Session,
    rel_path: str,
    *,
    excluded: bool | None = None,
    root_id: str = LIBRARY_ROOT_ID,
) -> SourceFile:
    """PATCH seam: update user intent on one registry row. v1 sets ``excluded`` only.

    Raises ``KeyError`` for an unknown row (the API maps that to 404). ``doc_type`` is
    intentionally not a parameter yet (dormant column) — it lands with the facet's activation.
    ``root_id`` defaults to the library root so pre-AD3b callers are unchanged.
    """
    row = session.execute(
        select(SourceFile).where(SourceFile.rel_path == rel_path, SourceFile.root_id == root_id)
    ).scalar_one_or_none()
    if row is None:
        raise KeyError(source_key(root_id, rel_path))
    if excluded is not None:
        row.excluded = excluded
    session.flush()
    return row


def view_for(
    session: Session, source_dir: Path, rel_path: str, *, root_id: str = LIBRARY_ROOT_ID
) -> SourceView | None:
    """The current `SourceView` for one row (freshly derived status), or ``None`` if no row.

    Used by ``PATCH /api/sources`` to echo the updated row without a full re-scan. ``source_dir``
    is still taken so the library root's path stays current on this path too.
    """
    row = session.execute(
        select(SourceFile).where(SourceFile.rel_path == rel_path, SourceFile.root_id == root_id)
    ).scalar_one_or_none()
    if row is None:
        return None
    root = session.get(SourceRoot, root_id)
    if root is None:  # unreachable while the FK holds — narrows for mypy
        return None
    if root.kind == "library":
        ensure_library_root(session, source_dir)
    available = _root_available(root)
    disk_path = Path(root.path).resolve() / rel_path
    if available and disk_path.is_file():
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
        root_id=root.id,
        root_kind=root.kind,
        root_available=available,
    )


def excluded_paths(session: Session) -> set[str]:
    """The **absolute** path keys of every row flagged ``excluded``, across all roots.

    Absolute rather than root-relative since AD3b: an exclusion now has to be recognised on a
    file walked from anywhere, and the old rel_path form silently could not match a file outside
    the one source dir. Joining to the root here removes that whole class of near-miss.
    """
    rows = session.execute(
        select(SourceFile.rel_path, SourceRoot.path)
        .join(SourceRoot, SourceFile.root_id == SourceRoot.id)
        .where(SourceFile.excluded.is_(True))
    ).all()
    return {_pathkey(Path(root_path) / rel) for rel, root_path in rows}


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

    - ``requested is None`` → every supported file on disk under **every reachable root**, minus
      the ``excluded`` ones (an implicit walk honors the standing exclusions; the skipped count is
      logged).
    - a list → `validate_selection` against what is actually on disk (not the possibly-stale
      registry), then absolute paths. An **explicit** pick **overrides** ``excluded``
      (Decision 5), logged. Raises `InvalidSelection` (→ API 400) if any path is unusable.

    Entries may be `source_key`s (``"<root_id>:<rel_path>"``) or bare rel_paths, which mean the
    library root — so a caller written before AD3b behaves exactly as it did.

    **An unreachable root contributes nothing rather than raising.** Its files are genuinely not
    ingestable right now, and failing the whole selection because one referenced drive is
    unplugged would block ingesting the library that *is* present. They surface as ``missing`` in
    the registry view, which is where that fact belongs.
    """
    roots = list_roots(session, source_dir)
    known_ids = {r.id for r in roots}

    # (root_id, rel_path) -> absolute path, across every reachable root. A tuple key rather than
    # a joined string so nothing downstream has to re-parse one.
    on_disk: dict[tuple[str, str], Path] = {}
    for rv in roots:
        if not rv.available:
            log.info("root_unavailable_skipped", root=rv.path, kind=rv.kind)
            continue
        base = Path(rv.path).resolve()
        for p in base.rglob("*"):
            if p.is_file() and is_supported(p):
                on_disk[(rv.id, p.relative_to(base).as_posix())] = p

    excluded = excluded_paths(session)

    if requested is None:
        kept: list[Path] = []
        skipped = 0
        for _pair, path in sorted(on_disk.items()):
            if _pathkey(path) in excluded:
                skipped += 1
                continue
            kept.append(path)
        if skipped:
            log.info("excluded_skipped", count=skipped)
        return kept

    # Validate the **rel_path**, never the composite key. Prefixing a key onto a request before
    # validating would defeat the traversal check outright: `PurePosixPath("library:../evil.pdf")`
    # has parts `("library:..", "evil.pdf")`, so `".." in parts` is False and `../evil.pdf` walks
    # straight through the guard. So requests are split by root first and each root's rel_paths
    # are validated against that root's own on-disk set — which is also what keeps the offender
    # lists reporting what the caller actually sent.
    by_root: dict[str, list[str]] = {}
    for raw in requested:
        root_id, rel = split_key(raw, known_ids)
        by_root.setdefault(root_id, []).append(rel)

    # An unreachable root contributes nothing to `on_disk`, so validating against it would report
    # every one of its files as `unknown` and fail the **whole** selection — blocking the library
    # that is present because a reference drive is unplugged. Dropped with a count instead, which
    # is what the implicit branch above already does; those files read `missing` in the registry
    # view, which is where "not here right now" belongs.
    unreachable = {rv.id for rv in roots if not rv.available}
    for root_id in list(by_root):
        if root_id in unreachable:
            dropped = by_root.pop(root_id)
            log.info("root_unavailable_selection_skipped", root_id=root_id, count=len(dropped))

    merged: dict[str, list[str]] = {}
    keys: list[str] = []
    for root_id, rels in by_root.items():
        known_rels = {rel for rid, rel in on_disk if rid == root_id}
        try:
            keys.extend(source_key(root_id, rel) for rel in validate_selection(rels, known_rels))
        except InvalidSelection as e:
            for reason, paths in e.offenders.items():
                merged.setdefault(reason, []).extend(paths)
    if merged:
        raise InvalidSelection(merged)

    overridden = [k for k in keys if _pathkey(on_disk[split_key(k, known_ids)]) in excluded]
    if overridden:
        log.info("excluded_overridden_by_explicit_selection", paths=overridden)
    return [on_disk[split_key(k, known_ids)] for k in keys]
