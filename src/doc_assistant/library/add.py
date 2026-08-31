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
from typing import TYPE_CHECKING, Any, Literal

import structlog

from doc_assistant.extractors import get_format_status

if TYPE_CHECKING:  # import cost only — the ORM session is a type here, never constructed
    from datetime import datetime

    from sqlalchemy.orm import Session

    from doc_assistant.db.models import SourceFile

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
    #: Set on `duplicate`: the `registry.source_key` of the registered file this one matches —
    #: a key, not a bare rel_path, since AD3b made the library span roots.
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


@dataclass(frozen=True)
class _Registered:
    """One registry row as the duplicate check needs it: its key, where it really is, its hash."""

    key: str
    path: Path
    sha256: str | None


def _size_index(session: Session) -> dict[int, list[_Registered]]:
    """`{size: [(key, absolute_path, cached_sha256), ...]}` for every registered source file.

    Built from the registry's stat-only columns joined to each row's root, so it costs one query
    and no file reads. **The join is what makes duplicate detection span roots** (AD3b): before
    it, every rel_path was resolved against the library folder, so a file registered under a
    *referenced* root resolved to a path that does not exist, the read failed, and the candidate
    came back `add` — silently re-adding something the library already had.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile, SourceRoot
    from doc_assistant.ingest.registry import source_key

    index: dict[int, list[_Registered]] = {}
    rows = session.execute(
        select(
            SourceFile.root_id,
            SourceFile.rel_path,
            SourceFile.size,
            SourceFile.source_sha256,
            SourceRoot.path,
        ).join(SourceRoot, SourceFile.root_id == SourceRoot.id)
    ).all()
    for root_id, rel_path, size, cached, root_path in rows:
        index.setdefault(int(size), []).append(
            _Registered(
                key=source_key(str(root_id), str(rel_path)),
                path=Path(str(root_path)) / str(rel_path),
                sha256=cached,
            )
        )
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

    library = (source_dir or get_source_dir()).resolve()
    candidates = expand_paths(paths)
    verdicts: list[FileVerdict] = []

    with session_scope() as session:
        # Make the stored library-root path match the configured source dir before anything
        # resolves against it: `_size_index` joins each row to its root's path, so a stale entry
        # here would send every duplicate check looking in the wrong folder. Idempotent, and it
        # adds no rows — the "inspect registers nothing" guarantee is about the registry, not
        # about refusing to keep a path honest.
        from doc_assistant.ingest.registry import ensure_library_root

        ensure_library_root(session, library)
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
            for candidate in index.get(size, []):
                cached = candidate.sha256
                if cached is None:
                    try:
                        cached = sha256_file(candidate.path)
                    except OSError:
                        continue  # a registered file that has since vanished cannot be matched
                    newly_hashed[candidate.key] = cached
                if digest is None:
                    try:
                        digest = sha256_file(path)
                    except OSError:
                        break
                if digest == cached:
                    duplicate_of = candidate.key
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


def _cache_hashes(session: Session, hashes: dict[str, str]) -> None:
    """Persist hashes computed while answering a duplicate question. Never computes new ones.

    Keyed by `registry.source_key`, not by `rel_path`: since AD3b the same relative path may exist
    under two roots, and matching on `rel_path` alone would write one root's hash onto the other's
    row — a wrong cached identity, which is worse than no cache at all.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile, SourceRoot
    from doc_assistant.ingest.registry import split_key

    known = {r.id for r in session.execute(select(SourceRoot)).scalars()}
    wanted = {split_key(key, known): digest for key, digest in hashes.items()}
    rows = (
        session.execute(
            select(SourceFile).where(SourceFile.rel_path.in_({rel for _root, rel in wanted}))
        )
        .scalars()
        .all()
    )
    for row in rows:
        digest = wanted.get((row.root_id, row.rel_path))
        if digest is not None and row.source_sha256 is None:
            row.source_sha256 = digest


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

#: Placement modes from ADR-046. **Both are built since AD3b** — `copy` puts the file in the
#: library folder and the app owns it; `reference` registers it where it already lives and the app
#: must never bin it.
AddMode = Literal["copy", "reference"]


@dataclass(frozen=True)
class AddOutcome:
    """What happened to one file. Per-file, never a batch total (the Track B lesson)."""

    path: str
    name: str
    ok: bool
    #: Where the file landed, **relative to its own root** — `library/rag.pdf`, not a key. Kept
    #: honest and separate from `key` rather than overloaded: this is what a human reads.
    rel_path: str | None = None
    #: `registry.source_key(root_id, rel_path)` — the identifier undo and selection take. Since
    #: AD3b a bare `rel_path` no longer identifies a row on its own.
    key: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class AddResult:
    """The result of an apply run, including what it did not get to.

    `not_attempted` is the honest half: the run **stops at the first failure** (grill branch 6),
    so the files after it were never touched and must not read as skipped-by-choice. The caller
    asks the user to keep what landed or undo it, which is why `added` carries a `key` — undo
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
    paths: Sequence[Path],
    *,
    mode: AddMode = "copy",
    source_dir: Path | None = None,
    reference_root: Path | None = None,
) -> AddResult:
    """Bring each file into the library and register it. **Indexing is not done here.**

    Two placement modes, both shipping in v1 (ADR-046, over a recommendation of copy-in-first):

    * ``copy`` — the file is copied into the library folder and registered under the **library**
      root. The app owns the copy, so delete may bin it.
    * ``reference`` — nothing is copied. The file is registered where it already lives, under a
      **referenced** root for its parent folder, and `origin='referenced'` is what stops delete
      from ever binning a file the app does not own (ADR-014 as amended).

    ``reference_root`` names the folder the whole batch belongs under, for a caller that *knows* —
    an ingestion adapter reading a catalogue that told it where its files live (ADR-049). Without
    it the per-parent rule applies, which is right for a dropped folder and catastrophic for a
    Zotero library, where every attachment sits in its own `storage/<key>/` directory: one root
    per document, each stat-ed on every scan. Given, it registers **one** root and every file in
    the batch lands under it. Ignored for ``copy``, which has a root already.

    Separation on purpose: indexing goes through the existing `POST /api/ingest` with an explicit
    `paths` list (spec constraint 4), so there is one ingest path in the system rather than two.

    Stops at the first failure and reports what was added, what failed and what was never tried.
    **Every failure comes back as that report**, including a database one: the registry's
    ``(root_id, rel_path)`` uniqueness is enforced, and re-registering an already-registered file
    — a referenced path added twice, or one whose size changed since the last scan so `inspect`'s
    size index missed it — raises `IntegrityError`, which is neither `OSError` nor `ValueError`.
    Letting it escape took the whole `AddResult` with it, so the files that *had* landed were
    reported to nobody and could not be undone.
    """
    import shutil

    from sqlalchemy.exc import IntegrityError, SQLAlchemyError

    from doc_assistant.app_settings import get_source_dir
    from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import registry

    library = (source_dir or get_source_dir()).resolve()
    library.mkdir(parents=True, exist_ok=True)

    # Registered once, outside the loop: it is one root for the batch, and registering it per file
    # would be the linear scan `register_root` does, repeated for every file in the import.
    if mode == "reference" and reference_root is not None:
        with session_scope() as session:
            registry.ensure_library_root(session, library)
            registry.register_root(session, reference_root)

    added: list[AddOutcome] = []
    failed: AddOutcome | None = None
    remaining: list[str] = []

    for i, src in enumerate(paths):
        if failed is not None:
            remaining.append(str(src))
            continue
        # The one file this iteration created, if any. Claimed *before* the copy so a copy that
        # dies partway leaves a truncated file this knows to remove; `_free_destination`
        # guarantees the name was free, so it is ours to delete and never the user's.
        copied_to: Path | None = None
        try:
            with session_scope() as session:
                registry.ensure_library_root(session, library)
                if mode == "copy":
                    dest = _free_destination(library, src.name)
                    copied_to = dest
                    shutil.copy2(src, dest)
                    root_id, rel = LIBRARY_ROOT_ID, dest.relative_to(library).as_posix()
                else:
                    dest = src.resolve()
                    root_id, rel = _reference_target(session, dest, library)
                stat = dest.stat()
                session.add(
                    SourceFile(
                        root_id=root_id,
                        rel_path=rel,
                        format=dest.suffix.lstrip(".").lower(),
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        source_sha256=sha256_file(dest),
                        origin="copied" if mode == "copy" else "referenced",
                    )
                )
            copied_to = None  # registered — it belongs to the library now, not to this attempt
            added.append(
                AddOutcome(
                    path=str(src),
                    name=src.name,
                    ok=True,
                    rel_path=rel,
                    key=registry.source_key(root_id, rel),
                )
            )
        except (OSError, ValueError, SQLAlchemyError) as e:
            # `session_scope` rolled the database back; the filesystem has no such thing. A copy
            # this run made and failed to register would be invisible to `undo_add` (no row, no
            # key) and perfectly visible to the next `scan_root`, which would adopt it as a
            # document the user was just told had failed to add.
            if copied_to is not None:
                with contextlib.suppress(OSError):
                    copied_to.unlink()
            # Only claim "already registered" when the driver actually said UNIQUE. An
            # `IntegrityError` could in principle be a foreign-key failure instead, and a
            # confident wrong cause is worse than an ugly right one (KI-37).
            message = str(e)
            if isinstance(e, IntegrityError) and "unique" in message.lower():
                message = f"{src.name} is already registered in the library"
            log.warning("add_failed", path=str(src), mode=mode, error=message)
            failed = AddOutcome(path=str(src), name=src.name, ok=False, error=message)
            remaining.extend(str(p) for p in paths[i + 1 :])
            break

    log.info(
        "add_applied", mode=mode, added=len(added), failed=failed is not None, left=len(remaining)
    )
    return AddResult(added=added, failed=failed, not_attempted=remaining)


def _reference_target(session: Session, dest: Path, library: Path) -> tuple[str, str]:
    """Pick the root a referenced file registers under, and its path relative to that root.

    A file that already sits **inside the library folder** is registered under the library root
    rather than getting a second root pointing at the same place — otherwise the same bytes would
    hold two rows with two origins, and delete would branch on whichever one it happened to read.
    Its origin still records ``referenced``: the app did not put it there, so it must not bin it.

    Otherwise, a **root already registered above this file wins** — the deepest one. That is not
    the guess the next rule refuses to make: an ancestor root exists only because the user, or an
    import acting for them, established it. It is also what keeps Zotero survivable, since Zotero
    gives every attachment its own `storage/<key>/` directory and the per-parent rule would mint a
    root per document.

    Failing that, a root for the file's **parent directory**. Per-directory rather than per-file so
    that referencing twenty papers out of one folder yields one root, and per-parent rather than
    per-ancestor because guessing how far up the user meant would be guessing — a folder the user
    drops is handled by the caller expanding it, and each contained file lands under its own
    directory's root.
    """
    from doc_assistant.db.models import LIBRARY_ROOT_ID
    from doc_assistant.ingest import registry

    if not dest.is_file():
        raise OSError(f"not a file: {dest}")
    try:
        return LIBRARY_ROOT_ID, dest.relative_to(library).as_posix()
    except ValueError:
        pass
    existing = registry.root_containing(session, dest)
    if existing is not None:
        return existing.id, dest.relative_to(Path(existing.path)).as_posix()
    root = registry.register_root(session, dest.parent)
    return root.id, dest.name


#: How long after a row was created ``undo_add`` will still delete the file behind it. Structural,
#: not tuned: it bounds *undo* to the action it undoes. The affordance is offered by a sheet that
#: is still open from the add, so minutes are generous — while a key replayed later (a retried
#: request, a stale client, a script) meets a registry row that is simply removed, leaving the
#: bytes for the next scan to re-register. That asymmetry is the whole point: a file left behind
#: is recoverable, a file deleted is not (KI-49).
UNDO_DELETE_WINDOW_SECONDS = 30 * 60


def undo_add(
    rel_paths: Sequence[str],
    *,
    source_dir: Path | None = None,
    chroma_db: Any | None = None,
) -> int:
    """Reverse an apply: drop the registry rows, the documents they produced, and only the files
    the app itself made.

    The two halves are deliberately separate since AD3b, because the two placement modes undo
    differently:

    * ``copied`` — the row goes **and** the file goes. Deleted outright rather than sent to the
      Recycle Bin (ADR-014): it is a copy the app made seconds ago and the user is explicitly
      rejecting, so the original is untouched and there is nothing to recover.
    * ``referenced`` — the row goes and **the file is never touched**. It is the user's own file
      sitting in their own folder; un-adding it from the library must not reach outside. This is
      the ADR-014 amendment enforced where it cannot be forgotten.

    **Three conditions gate the delete, and each closes a measured hole.** The row's root must be
    the *library* root — the path is resolved through that root rather than assumed to be under
    the library folder, which is how a key naming a file in the user's Zotero folder came to
    delete an unrelated same-named document out of the library. ``origin`` must be ``copied``.
    And the row must be younger than :data:`UNDO_DELETE_WINDOW_SECONDS`, so this is an undo of a
    just-completed add rather than a delete-by-key for anything the app ever copied in. A
    declined delete is logged, never silent; the row still goes either way.

    **Three things go, not one (KI-51).** The registry row was once all undo removed, which left an
    add that had already been indexed only half undone: the `Document` row and its chunks survived,
    so the library still listed — and could still cite — a document whose file undo had just
    deleted. Undo now also:

    * **removes the document the add produced**, when `chroma_db` is supplied and the document is
      provably this add's (same resolved path, and `added_at` inside the undo window). Without a
      `chroma_db` the row is left alone rather than orphaning its chunks — a caller that cannot
      reach the index cannot finish the job, and half-removing is worse than not starting.
    * **un-references a root it has just emptied**, so a `reference`-mode add that is undone stops
      the folder being scanned. Previously the root survived, the next scan re-found the file as
      `new`, and the following "index all" re-ingested exactly what the user had undone.

    The library root is never dropped, however empty it gets: it is the app's own folder, not a
    reference the user made.

    Accepts `registry.source_key`s or bare rel_paths (library root). Returns how many rows went.
    """
    from datetime import timedelta

    from sqlalchemy import select

    from doc_assistant.app_settings import get_source_dir
    from doc_assistant.db.models import SourceFile, SourceRoot, _utcnow
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import registry

    library = (source_dir or get_source_dir()).resolve()
    cutoff = _utcnow() - timedelta(seconds=UNDO_DELETE_WINDOW_SECONDS)
    undone = 0
    with session_scope() as session:
        # The delete resolves each row against **its own root's stored path**, so that path has to
        # be current before anything is unlinked — otherwise a library the user moved since the
        # add would send the unlink at the old folder.
        registry.ensure_library_root(session, library)
        known = {r.id for r in session.execute(select(SourceRoot)).scalars()}
        wanted = {registry.split_key(k, known) for k in rel_paths}
        if not wanted:
            return 0
        rows = (
            session.execute(
                select(SourceFile).where(SourceFile.rel_path.in_({rel for _root, rel in wanted}))
            )
            .scalars()
            .all()
        )
        touched_roots: set[str] = set()
        for row in rows:
            if (row.root_id, row.rel_path) not in wanted:
                continue  # same rel_path under a root the caller did not name
            # Resolve before the row goes: afterwards the root lookup it needs may be gone too.
            absolute = _row_path(session, row, library=library)
            if chroma_db is not None:
                _purge_document_for(absolute, chroma_db=chroma_db, cutoff=cutoff)
            if row.origin == "copied":
                _delete_copied_file(session, row, library=library, cutoff=cutoff)
            touched_roots.add(row.root_id)
            session.delete(row)
            undone += 1
        # Flush so the emptiness check below sees the deletions above rather than the pre-undo
        # state; the surrounding `session_scope` still owns the commit.
        session.flush()
        for root_id in touched_roots:
            _drop_root_if_emptied(session, root_id)
    log.info("add_undone", count=undone)
    return undone


def _row_path(session: Session, row: SourceFile, *, library: Path) -> Path:
    """The absolute path a registry row names, resolved through **its own** root.

    Never assume the library folder: a row under a referenced root is relative to the user's own
    directory, and resolving it against the library is exactly the mistake that once deleted an
    unrelated same-named document (see `_delete_copied_file`).
    """
    from doc_assistant.db.models import SourceRoot

    root = session.get(SourceRoot, row.root_id)
    base = Path(root.path) if root is not None else library
    return base / row.rel_path


def _purge_document_for(absolute: Path, *, chroma_db: Any, cutoff: datetime) -> bool:
    """Remove the `Document` this add produced for `absolute`, if it is provably this add's.

    **The guard is the whole function.** A path can carry a document the user already had — they
    may be re-adding a file that was ingested months ago, or replacing one at a path that already
    held something (the ADR-047 trade-off, where a new file inherits the previous document's id).
    Removing that would destroy a document, its folder membership and its metadata overrides on the
    strength of an undo the user meant to apply to *their own* add. So the document must have been
    added inside the same window that lets undo delete a file, and a decline is logged, never
    silent.

    The file itself is not touched here — the caller owns that, because the two placement modes
    disagree about it and only the caller knows which one this was.
    """
    from datetime import timezone

    from sqlalchemy import select

    from doc_assistant.db.models import Document as DBDocument
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import pathkey
    from doc_assistant.library.documents import purge_document_record

    wanted = pathkey(absolute)
    with session_scope() as session:
        rows = session.execute(
            select(
                DBDocument.id,
                DBDocument.source_original,
                DBDocument.added_at,
                DBDocument.doc_hash,
                DBDocument.source_cache,
            )
        ).all()
    match = next(
        (r for r in rows if r.source_original and pathkey(r.source_original) == wanted), None
    )
    if match is None:
        return False

    added_at = match.added_at
    if added_at is not None and added_at.tzinfo is None:
        # SQLite `DateTime` reads back naive; re-stamp it as the UTC it provably is, exactly as
        # `_delete_copied_file` does for `first_seen`.
        added_at = added_at.replace(tzinfo=timezone.utc)
    if added_at is None or added_at < cutoff:
        log.warning(
            "undo_document_purge_declined",
            reason="outside_undo_window",
            document_id=str(match.id),
            path=str(absolute),
        )
        return False

    chunks = purge_document_record(
        str(match.id), chroma_db, doc_hash=str(match.doc_hash), source_cache=match.source_cache
    )
    log.info(
        "undo_document_purged",
        document_id=str(match.id),
        path=str(absolute),
        chunks_removed=chunks,
    )
    return True


def _drop_root_if_emptied(session: Session, root_id: str) -> bool:
    """Drop a **referenced** root that no longer holds a single file. Returns whether it went.

    Undoing the last file of a referenced root has to withdraw the reference too, or the folder
    stays registered: the next `scan_sources` re-discovers the file as `new`, and the next
    "index all" re-ingests precisely what the user undid — without asking. The library root is
    exempt: it is the app's own folder and an empty library is a normal state, not a stale
    reference.
    """
    from sqlalchemy import func, select

    from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceFile, SourceRoot

    if root_id == LIBRARY_ROOT_ID:
        return False
    root = session.get(SourceRoot, root_id)
    if root is None or root.kind != "referenced":
        return False
    remaining = session.execute(
        select(func.count()).select_from(SourceFile).where(SourceFile.root_id == root_id)
    ).scalar_one()
    if remaining:
        return False
    session.delete(root)
    log.info("undo_root_unreferenced", root_id=root_id, path=root.path)
    return True


def _delete_copied_file(
    session: Session, row: SourceFile, *, library: Path, cutoff: datetime
) -> bool:
    """Bin the file behind one ``copied`` row, or explain in the log why it was left alone.

    Split out so the three conditions read as three conditions. Returns whether the file went.
    """
    from datetime import timezone

    from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceRoot

    if row.root_id != LIBRARY_ROOT_ID:
        # `origin='copied'` under a non-library root is not a fact the app can act on: copy mode
        # only ever writes the library root, so this row was written by something else and its
        # `rel_path` is relative to a folder that is not the library.
        log.warning("undo_delete_declined", reason="not_library_root", rel_path=row.rel_path)
        return False
    # `DateTime` carries no timezone in SQLite, so a value written from `_utcnow()` (aware, UTC)
    # reads back naive. Comparing the two raises rather than mis-ordering, which is the good
    # failure mode — but this guard must not be the thing that breaks undo, so the naive read is
    # re-stamped as what it provably is.
    first_seen = row.first_seen
    if first_seen is not None and first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)
    if first_seen is None or first_seen < cutoff:
        log.warning("undo_delete_declined", reason="outside_undo_window", rel_path=row.rel_path)
        return False
    root = session.get(SourceRoot, row.root_id)
    base = Path(root.path) if root is not None else library
    with contextlib.suppress(OSError):
        (base / row.rel_path).unlink()
        return True
    return False
