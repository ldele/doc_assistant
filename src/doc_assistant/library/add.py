"""Add-documents: inspect a set of candidate paths and say what would happen to each (AD2).

`inspect` is the review sheet's whole data source, and it is **read-only by construction** — it
stats, hashes and classifies, and writes nothing except an opportunistic hash cache on rows it
already had to read. That is spec constraint 2 ("nothing is copied, registered or indexed before
the review sheet is shown and confirmed") made structural rather than remembered: there is no code
path from here to the chunk store.

**Identity is the sha256 of the source bytes** (ADR-046). That deliberately leads ADR-042 — the
project's current `doc_hash` is over the *extracted* text and answers a different question
("already indexed?"). The two coexist until RG-027 collapses them, and they can disagree; the ADR
says so on purpose.

**Why hashing the library does not cost anything.** A naive duplicate check would hash every
registered file on every inspect — hundreds of megabytes for a 97-document corpus. Instead the
registry's already-scanned `size` is the discriminator: a candidate can only duplicate a file of
exactly the same byte length, so only same-size rows are ever hashed, and in the common case
(nothing matches) the library is not read at all. Size is cheap and already there; sha256 is the
confirmation, never the search.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog

from doc_assistant.extractors import get_format_status

log = structlog.get_logger(__name__)

#: Bytes per read when hashing. Large enough that a 10 MB PDF is a handful of syscalls, small
#: enough that a pathological file cannot balloon resident memory.
_HASH_CHUNK = 1 << 20

Verdict = Literal["add", "unsupported", "duplicate", "unreadable"]


@dataclass(frozen=True)
class FileVerdict:
    """What `inspect` decided about one candidate path, and why.

    `advisory` is the sentence shown to the user. For `unsupported` it is
    `get_format_status`'s own text, verbatim — that string already names the conversion target
    ("Convert to DOCX or PDF first"), and rewording it here would fork the message.
    """

    path: str
    name: str
    verdict: Verdict
    size: int | None = None
    sha256: str | None = None
    advisory: str | None = None
    #: Set on `duplicate`: the `rel_path` of the registered file this one matches.
    duplicate_of: str | None = None

    @property
    def selected_by_default(self) -> bool:
        """Only a clean `add` starts ticked. Everything else is shown and left for the user."""
        return self.verdict == "add"


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes. Binary — no encoding concerns apply."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def expand_paths(paths: Iterable[Path]) -> list[Path]:
    """Resolve a drop or a pick into the concrete files it names.

    A dropped **folder recurses fully**, matching `registry.scan_sources`'s `root.rglob("*")` —
    the grill settled this (branch 3): the depth is reported as a file count before anything
    happens, rather than being a setting nobody would find. Order is stable: the order given,
    with each directory's contents sorted so two identical drops inspect identically.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            if p.is_dir():
                found = sorted(q for q in p.rglob("*") if q.is_file())
            elif p.exists():
                found = [p]
            else:
                found = [p]  # keep it: `inspect` reports it as unreadable rather than dropping it
        except OSError:  # pragma: no cover - permission-denied on a directory walk
            log.warning("expand_failed", path=str(p))
            found = [p]
        for q in found:
            if q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _size_index(session: object) -> dict[int, list[tuple[str, str | None]]]:
    """`{size: [(rel_path, cached_sha256), ...]}` for every registered source file.

    Built from the registry's stat-only columns, so it costs one query and no file reads.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile

    index: dict[int, list[tuple[str, str | None]]] = {}
    rows = session.execute(  # type: ignore[attr-defined]
        select(SourceFile.rel_path, SourceFile.size, SourceFile.source_sha256)
    ).all()
    for rel_path, size, cached in rows:
        index.setdefault(int(size), []).append((str(rel_path), cached))
    return index


def inspect(paths: Sequence[Path], *, source_dir: Path | None = None) -> list[FileVerdict]:
    """Classify each candidate path. Reads the filesystem and the registry; mutates neither.

    Directories are expanded first, so the caller can hand over whatever the drop produced.

    The only write is an opportunistic `source_sha256` fill on registry rows that had to be hashed
    to answer a duplicate question — a cache of work already done, never new work. It is committed
    separately from any decision the user then makes.
    """
    from doc_assistant.app_settings import get_source_dir
    from doc_assistant.db.session import session_scope

    root = (source_dir or get_source_dir()).resolve()
    candidates = expand_paths(paths)
    verdicts: list[FileVerdict] = []

    with session_scope() as session:
        index = _size_index(session)
        newly_hashed: dict[str, str] = {}

        for path in candidates:
            name = path.name
            try:
                size = path.stat().st_size
            except OSError:
                verdicts.append(
                    FileVerdict(
                        path=str(path),
                        name=name,
                        verdict="unreadable",
                        advisory="This file could not be read. It may have been moved or renamed.",
                    )
                )
                continue

            supported, advisory = get_format_status(path)
            if not supported:
                verdicts.append(
                    FileVerdict(
                        path=str(path),
                        name=name,
                        verdict="unsupported",
                        size=size,
                        advisory=advisory,
                    )
                )
                continue

            # Size first: only a file of exactly this byte length can be the same bytes, so the
            # library is hashed only when a collision is actually possible.
            digest: str | None = None
            duplicate_of: str | None = None
            for rel_path, cached in index.get(size, []):
                if cached is None:
                    registered = root / rel_path
                    try:
                        cached = sha256_file(registered)
                    except OSError:
                        continue  # a registered file that has since vanished cannot be matched
                    newly_hashed[rel_path] = cached
                if digest is None:
                    try:
                        digest = sha256_file(path)
                    except OSError:
                        break
                if digest == cached:
                    duplicate_of = rel_path
                    break

            if duplicate_of is not None:
                verdicts.append(
                    FileVerdict(
                        path=str(path),
                        name=name,
                        verdict="duplicate",
                        size=size,
                        sha256=digest,
                        duplicate_of=duplicate_of,
                        advisory="Already in your library.",
                    )
                )
                continue

            verdicts.append(
                FileVerdict(path=str(path), name=name, verdict="add", size=size, sha256=digest)
            )

        if newly_hashed:
            _cache_hashes(session, newly_hashed)

    return verdicts


def _cache_hashes(session: object, hashes: dict[str, str]) -> None:
    """Persist hashes computed while answering a duplicate question. Never computes new ones."""
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile

    rows = (
        session.execute(  # type: ignore[attr-defined]
            select(SourceFile).where(SourceFile.rel_path.in_(list(hashes)))
        )
        .scalars()
        .all()
    )
    for row in rows:
        row.source_sha256 = hashes[row.rel_path]
    log.info("source_hashes_cached", count=len(rows))


def sort_for_review(verdicts: Sequence[FileVerdict]) -> list[FileVerdict]:
    """Non-`add` verdicts first, stable within each group (grill branch 7).

    The batch is uncapped and the sheet paginates, so the first page must carry everything the
    user needs to see. Sorting the exceptions up is what makes "and N more" only ever mean clean
    files — a page that hid a warning would read as approval of something never shown.
    """
    order = {"unsupported": 0, "unreadable": 1, "duplicate": 2, "add": 3}
    return sorted(verdicts, key=lambda v: order.get(v.verdict, 99))


def summarise(verdicts: Sequence[FileVerdict]) -> dict[str, int]:
    """Counts per verdict, plus `total` — the numbers the sheet's header states."""
    counts: dict[str, int] = {"total": len(verdicts)}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


# ============================================================
# AD3 — apply. Copy the file in, register it, and stop honestly on failure.
# ============================================================

#: Modes from ADR-046. `reference` is decided and specced but NOT built (AD3b): the schema is
#: cheap, but `registry.scan_sources` / `derive_status` / `resolve_selection` / `list_sources` are
#: all keyed on a bare root-relative `rel_path` and each needs the root dimension first. Accepting
#: the value here and refusing it loudly beats pretending it works.
AddMode = Literal["copy", "reference"]


@dataclass(frozen=True)
class AddOutcome:
    """What happened to one file. Per-file, never a batch total (the Track B lesson)."""

    path: str
    name: str
    ok: bool
    rel_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class AddResult:
    """The result of an apply run, including what it did not get to.

    `not_attempted` is the honest half: the run **stops at the first failure** (grill branch 6),
    so the files after it were never touched and must not read as skipped-by-choice. The caller
    asks the user to keep what landed or undo it, which is why `added` carries `rel_path`s — undo
    needs to name exactly what to remove, not re-derive it.
    """

    added: list[AddOutcome]
    failed: AddOutcome | None
    not_attempted: list[str]

    @property
    def stopped_early(self) -> bool:
        return self.failed is not None


def _free_destination(root: Path, name: str) -> Path:
    """A destination under `root` that does not overwrite an existing file.

    Two different papers can share a filename; ADR-043 says received content is preserved
    verbatim, and silently overwriting one with the other would destroy a document the user still
    has in their library. Byte-identical files never reach here — those are `duplicate` verdicts.
    """
    candidate = root / name
    if not candidate.exists():
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    for n in range(2, 1000):
        candidate = root / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
    raise OSError(f"could not find a free name for {name!r} in {root}")


def apply_add(
    paths: Sequence[Path], *, mode: AddMode = "copy", source_dir: Path | None = None
) -> AddResult:
    """Copy each file into the library folder and register it. **Indexing is not done here.**

    Separation on purpose: indexing goes through the existing `POST /api/ingest` with an explicit
    `paths` list (spec constraint 4), so there is one ingest path in the system rather than two.

    Stops at the first failure and reports what was added, what failed and what was never tried.
    """
    import shutil

    from doc_assistant.app_settings import get_source_dir
    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope

    if mode == "reference":
        raise NotImplementedError(
            "reference-in-place is decided (ADR-046) but not built (AD3b): the source registry "
            "is keyed on a root-relative path and needs a root dimension first."
        )

    root = (source_dir or get_source_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)

    added: list[AddOutcome] = []
    failed: AddOutcome | None = None
    remaining: list[str] = []

    for i, src in enumerate(paths):
        if failed is not None:
            remaining.append(str(src))
            continue
        try:
            dest = _free_destination(root, src.name)
            shutil.copy2(src, dest)
            rel = dest.relative_to(root).as_posix()
            stat = dest.stat()
            with session_scope() as session:
                session.add(
                    SourceFile(
                        rel_path=rel,
                        format=dest.suffix.lstrip(".").lower(),
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        source_sha256=sha256_file(dest),
                        origin="copied",
                    )
                )
            added.append(AddOutcome(path=str(src), name=src.name, ok=True, rel_path=rel))
        except (OSError, ValueError) as e:
            log.warning("add_failed", path=str(src), error=str(e))
            failed = AddOutcome(path=str(src), name=src.name, ok=False, error=str(e))
            remaining.extend(str(p) for p in paths[i + 1 :])
            break

    log.info("add_applied", added=len(added), failed=failed is not None, left=len(remaining))
    return AddResult(added=added, failed=failed, not_attempted=remaining)


def undo_add(rel_paths: Sequence[str], *, source_dir: Path | None = None) -> int:
    """Reverse an apply: delete the copies and their registry rows. Returns how many were undone.

    Deletes outright rather than sending to the Recycle Bin (ADR-014): these are copies the app
    made seconds ago and the user is explicitly rejecting, so the original is untouched and there
    is nothing to recover. **Only ever called with `rel_path`s this module just created**, which
    is why it cannot reach a referenced original.
    """
    from sqlalchemy import select

    from doc_assistant.app_settings import get_source_dir
    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope

    root = (source_dir or get_source_dir()).resolve()
    undone = 0
    with session_scope() as session:
        rows = (
            session.execute(select(SourceFile).where(SourceFile.rel_path.in_(list(rel_paths))))
            .scalars()
            .all()
        )
        for row in rows:
            if row.origin != "copied":
                continue  # never remove a file the app does not own
            with contextlib.suppress(OSError):
                (root / row.rel_path).unlink()
            session.delete(row)
            undone += 1
    log.info("add_undone", count=undone)
    return undone
