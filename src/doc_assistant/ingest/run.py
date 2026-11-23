"""One ingest run as a person watches it: each file's outcome, and pause and stop (ROADMAP 31).

`ingest.main` owns the loop. This object is what the loop reports each file to and what it asks,
between two documents, whether to wait or to end. The API keeps one per run and reads it for the
indexing screen; the CLI passes none and runs exactly as before.

**Pause and stop take effect between documents, never inside one.** Documents run one at a time,
and a document is written to both vector stores and then to SQLite: stopping half-way through one
would leave the partial write the dedup gate exists to repair. So a request is recorded at once
and honoured at the next :meth:`IngestRun.checkpoint`, and :meth:`IngestRun.control` says which
of the two it is (``pausing`` until the document in flight finishes, then ``paused``).

**Stop ends the loop; it does not raise.** The steps after the loop (folder assignment, imported
metadata) and the API's controller reload still run, so every finished document goes live and
the next run skips it through the ordinary dedup gate. Nothing is lost by stopping.

**Scale.** At 10,000 documents the status poll cannot carry every file, so the status carries
:meth:`IngestRun.counts` and the files come a page at a time from :meth:`IngestRun.page`.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

FileState = Literal[
    "waiting", "reading", "added", "unchanged", "done", "failed", "not_started", "removed"
]
"""Where one item of a run stands. ``not_started`` is an item a stopped run never reached;
``done`` is a re-run that finished (A8″: a re-run is an item of the same list); ``removed`` is a
waiting item the person took out, kept so ids never move, and never counted or listed."""

ItemKind = Literal["file", "folder", "rerun"]
"""What a row is (A8″, spec `ingestion-queue.md` decision 2): a file to index; the library folder
to index, a row that becomes its files when it starts; or one document's parts to re-run."""

Control = Literal["none", "pausing", "paused", "stopping", "finishing"]
"""What has been asked of a running run: ``pausing`` and ``stopping`` wait for the document in
flight, ``paused`` means the loop is waiting between two documents. ``finishing`` is the time after
the loop and before the run reports its end, while what finished is made searchable: nothing more
can be paused or stopped then. A stopped run stays ``stopping`` through that time."""

FileGroup = Literal["all", "in_progress", "waiting", "not_started", "finished", "failed"]
"""The screen's filters. ``finished`` is a file that is now in the library, added or unchanged."""

RerunScope = Literal["failed", "remaining"]
"""What a follow-up run takes from this one: the files that failed, or the ones never reached."""

_GROUPS: dict[FileGroup, frozenset[FileState]] = {
    "all": frozenset(
        ("waiting", "reading", "added", "unchanged", "done", "failed", "not_started")
    ),
    "in_progress": frozenset(("reading",)),
    "waiting": frozenset(("waiting",)),
    "not_started": frozenset(("not_started",)),
    "finished": frozenset(("added", "unchanged", "done")),
    "failed": frozenset(("failed",)),
}

#: The order the screen lists files in: what is moving, then what needs the reader's attention,
#: then what is still to come, then what is done. Within a state, the run's own order.
_ORDER: dict[FileState, int] = {
    "reading": 0,
    "failed": 1,
    "waiting": 2,
    "not_started": 3,
    "added": 4,
    "unchanged": 5,
    "done": 6,
}

#: A failure's reason is shown on one row; an exception message can be a page of traceback text.
#: Structural, not tuned: a row holds about this much before it wraps twice.
REASON_MAX_CHARS = 200

NO_TEXT = "No text found. If it is a scan, it needs OCR before it can be searched."
UNCHANGED = "Already in your library"


@dataclass(frozen=True)
class DocumentResult:
    """What :func:`doc_assistant.ingest.process_document` did with one file.

    ``result`` is the run's own tally, unchanged since before this module existed (``main``
    returns the counts of it). ``passages`` is the baseline passage count recorded on the
    document; ``reason`` is a sentence for a file that was not indexed, ``None`` otherwise.
    """

    result: Literal["added", "skipped", "error"]
    passages: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class FileOutcome:
    """One row of the indexing screen: a file, the library folder, or a document's re-run.

    ``id`` is stable for the life of the run, so a row can be removed or expanded while others
    are added (A8″). ``document_id`` and ``parts`` say what a re-run row re-runs."""

    path: str
    name: str
    state: FileState
    detail: str | None = None
    id: str = ""
    kind: ItemKind = "file"
    document_id: str | None = None
    parts: tuple[str, ...] = ()


def outcome_of(path: str, name: str, done: DocumentResult) -> FileOutcome:
    """The row a finished document shows, in the words a person reads.

    A document recorded with no passages counts as *added* in the run's tally (its row exists)
    but shows as failed here: it is in the library and no question will ever find it, which is
    the thing the person needs to see. Retrying it is harmless, since a document with no passages
    is not in the vector stores and so is not skipped as already indexed.
    """
    if done.result == "error":
        return FileOutcome(path, name, "failed", done.reason or "Could not read this file.")
    if done.result == "skipped":
        if done.reason:
            return FileOutcome(path, name, "failed", done.reason)
        return FileOutcome(path, name, "unchanged", UNCHANGED)
    if done.passages == 0:
        return FileOutcome(path, name, "failed", NO_TEXT)
    unit = "passage" if done.passages == 1 else "passages"
    return FileOutcome(path, name, "added", f"{done.passages} {unit}")


def short_reason(error: BaseException) -> str:
    """One line of an exception, short enough for a row, never empty."""
    text = str(error).strip().splitlines()[0].strip() if str(error).strip() else ""
    text = text or type(error).__name__
    if len(text) > REASON_MAX_CHARS:
        text = text[: REASON_MAX_CHARS - 1].rstrip() + "…"
    return text


class RunSink(Protocol):
    """What `ingest.main`'s loop reports to and asks: a whole run, or one batch of a longer list
    (`IngestRun.view`, A8″). Indexes are the loop's own, in the order of ``begin``'s paths."""

    def begin(self, paths: list[Path]) -> None: ...
    def checkpoint(self) -> bool: ...
    def reading(self, index: int) -> None: ...
    def record(self, index: int, done: DocumentResult) -> None: ...
    def end(self) -> None: ...
    def stop_requested(self) -> bool: ...


@dataclass(frozen=True)
class Batch:
    """The next stretch of the list to run: consecutive waiting files, or one folder or re-run."""

    kind: ItemKind
    ids: tuple[str, ...]


class IngestRun:
    """The shared state of one run: written by the ingest thread, read and steered by the API.

    Every method is safe from any thread. The controls (:meth:`pause`, :meth:`resume`,
    :meth:`stop`) return at once; the loop's side (:meth:`checkpoint`) is the only call that
    blocks, and only while paused.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resume = threading.Event()
        self._resume.set()
        self._stop = threading.Event()
        self._at_checkpoint = False
        self._ended = False
        self._closed = False
        self._files: list[FileOutcome] = []
        self._ids = itertools.count(1)

    def _new_id(self) -> str:
        return f"i{next(self._ids)}"

    def _index_of(self, item_id: str) -> int:
        for i, f in enumerate(self._files):
            if f.id == item_id:
                return i
        raise KeyError(item_id)

    # --- the list, from the queue (A8″) ------------------------------------------------------

    def append_files(self, paths: list[Path], detail: str | None = None) -> list[str]:
        """Put files at the end of the list, waiting. A file already waiting or reading is not
        added twice. Returns the ids of the rows added; ``[]`` once the run is closed."""
        from doc_assistant.ingest.registry import pathkey

        with self._lock:
            if self._closed:
                return []
            live = {
                pathkey(f.path)
                for f in self._files
                if f.kind == "file" and f.state in ("waiting", "reading")
            }
            added: list[str] = []
            for p in paths:
                key = pathkey(p)
                if key in live:
                    continue
                live.add(key)
                row = FileOutcome(str(p), p.name, "waiting", detail, id=self._new_id())
                self._files.append(row)
                added.append(row.id)
            return added

    def append_folder(self, source: Path, name: str) -> str | None:
        """Put the library folder at the end: one row, which becomes its files when it starts.
        Not added twice while one waits. ``None`` once the run is closed."""
        with self._lock:
            if self._closed:
                return None
            if any(f.kind == "folder" and f.state == "waiting" for f in self._files):
                return next(
                    f.id for f in self._files if f.kind == "folder" and f.state == "waiting"
                )
            row = FileOutcome(str(source), name, "waiting", None, id=self._new_id(), kind="folder")
            self._files.append(row)
            return row.id

    def append_rerun(
        self, document_id: str, name: str, parts: tuple[str, ...], detail: str | None = None
    ) -> str | None:
        """Put one document's re-run at the end. A re-run of the same document already waiting
        takes the new parts instead of a second row. ``None`` once the run is closed."""
        with self._lock:
            if self._closed:
                return None
            for i, f in enumerate(self._files):
                if f.kind == "rerun" and f.document_id == document_id and f.state == "waiting":
                    merged = tuple(dict.fromkeys(f.parts + parts))
                    self._files[i] = replace(f, parts=merged)
                    return f.id
            row = FileOutcome(
                document_id,
                name,
                "waiting",
                detail,
                id=self._new_id(),
                kind="rerun",
                document_id=document_id,
                parts=parts,
            )
            self._files.append(row)
            return row.id

    def remove(self, item_id: str) -> bool:
        """Take a waiting row out. ``False`` for a row that is not waiting, or not there."""
        with self._lock:
            try:
                i = self._index_of(item_id)
            except KeyError:
                return False
            if self._files[i].state != "waiting":
                return False
            self._files[i] = replace(self._files[i], state="removed")
            return True

    def next_batch(self) -> Batch | None:
        """The first waiting row and, for files, the waiting files right after it."""
        with self._lock:
            waiting = [f for f in self._files if f.state == "waiting"]
            if not waiting:
                return None
            first = waiting[0]
            if first.kind != "file":
                return Batch(first.kind, (first.id,))
            start = self._index_of(first.id)
            ids: list[str] = []
            for f in self._files[start:]:
                if f.state == "removed":
                    continue
                if f.kind != "file" or f.state != "waiting":
                    break
                ids.append(f.id)
            return Batch("file", tuple(ids))

    def close_if_drained(self) -> bool:
        """Close the run if nothing waits, atomically with every append: after ``True`` an add
        starts a new run instead of joining one whose loop has ended."""
        with self._lock:
            if self._stop.is_set() or not any(f.state == "waiting" for f in self._files):
                self._closed = True
                return True
            return False

    def current(self) -> str | None:
        """The name of the row being read, or ``None`` between two."""
        with self._lock:
            return next((f.name for f in self._files if f.state == "reading"), None)

    @property
    def closed(self) -> bool:
        """The loop has decided to end: an add now starts a new run."""
        with self._lock:
            return self._closed

    def ahead_of(self, item_id: str) -> int:
        """How many rows wait or run before ``item_id``, in list order."""
        with self._lock:
            ahead = 0
            for f in self._files:
                if f.id == item_id:
                    return ahead
                if f.state in ("waiting", "reading"):
                    ahead += 1
            return 0

    def item(self, item_id: str) -> FileOutcome:
        with self._lock:
            return self._files[self._index_of(item_id)]

    def mark(self, item_id: str, state: FileState, detail: str | None = None) -> None:
        """Set one row's state and what happened (a re-run row's outcome, a folder's end)."""
        with self._lock:
            i = self._index_of(item_id)
            self._files[i] = replace(self._files[i], state=state, detail=detail)

    def expand(self, item_id: str, paths: list[Path]) -> list[str]:
        """A folder row starts: it becomes one waiting row per file, in its place."""
        with self._lock:
            i = self._index_of(item_id)
            rows = [FileOutcome(str(p), p.name, "waiting", None, id=self._new_id()) for p in paths]
            self._files[i : i + 1] = rows
            return [r.id for r in rows]

    def view(self, batch: Batch) -> BatchView:
        """What `ingest.main` reports one batch to (`RunSink`)."""
        return BatchView(self, batch)

    # --- the controls, from the API -------------------------------------------------------------

    def pause(self) -> None:
        """Ask the loop to wait after the document in flight. A no-op once stop was asked."""
        if not self._stop.is_set():
            self._resume.clear()

    def resume(self) -> None:
        """Let a paused loop go on."""
        self._resume.set()

    def stop(self) -> None:
        """Ask the loop to end after the document in flight. Wakes a paused loop to do it."""
        self._stop.set()
        self._resume.set()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def control(self) -> Control:
        """What has been asked and whether the loop has acted on it yet."""
        with self._lock:
            if self._stop.is_set():
                return "stopping"
            if self._ended:
                return "finishing"
            if not self._resume.is_set():
                return "paused" if self._at_checkpoint else "pausing"
            return "none"

    # --- the loop's side, from `ingest.main` ----------------------------------------------------

    def begin(self, paths: list[Path]) -> None:
        """Every file of the run, waiting, in the order the loop will take them."""
        with self._lock:
            self._files = [
                FileOutcome(str(p), p.name, "waiting", id=self._new_id()) for p in paths
            ]

    def checkpoint(self) -> bool:
        """Between two documents: wait while paused, then say whether to stop (``True``)."""
        with self._lock:
            self._at_checkpoint = True
        try:
            self._resume.wait()
        finally:
            with self._lock:
                self._at_checkpoint = False
        return self._stop.is_set()

    def reading(self, index: int) -> None:
        with self._lock:
            row = self._files[index]
            self._files[index] = replace(row, state="reading", detail=None)

    def record(self, index: int, done: DocumentResult) -> None:
        with self._lock:
            row = self._files[index]
            out = outcome_of(row.path, row.name, done)
            self._files[index] = replace(row, state=out.state, detail=out.detail)

    def end(self) -> None:
        """The loop is over. Files a stop left behind become *not started*."""
        with self._lock:
            self._ended = True
            self._closed = True
            self._files = [
                replace(f, state="not_started") if f.state == "waiting" else f for f in self._files
            ]

    # --- reads, for the status and the screen ---------------------------------------------------

    @property
    def stopped(self) -> bool:
        """The loop ended because it was asked to."""
        with self._lock:
            return self._ended and self._stop.is_set()

    def counts(self) -> dict[str, int]:
        """How many files are in each state, plus ``total``. Cheap enough for every poll."""
        with self._lock:
            counts: dict[str, int] = dict.fromkeys(_ORDER, 0)
            shown = [f for f in self._files if f.state != "removed"]
            for f in shown:
                counts[f.state] += 1
            counts["total"] = len(shown)
            return counts

    def page(
        self, offset: int = 0, limit: int = 100, group: FileGroup = "all"
    ) -> tuple[list[FileOutcome], int]:
        """One page of the files in ``group``, in screen order, and how many the group holds."""
        wanted = _GROUPS[group]
        with self._lock:
            rows = [(i, f) for i, f in enumerate(self._files) if f.state in wanted]
        rows.sort(key=lambda pair: (_ORDER[pair[1].state], pair[0]))
        start = max(0, offset)
        return [f for _, f in rows[start : start + max(0, limit)]], len(rows)

    def paths(self, scope: RerunScope) -> list[Path]:
        """The files a follow-up run takes: those that failed, or those never reached."""
        wanted: frozenset[FileState] = (
            frozenset(("failed",)) if scope == "failed" else frozenset(("not_started", "waiting"))
        )
        with self._lock:
            return [Path(f.path) for f in self._files if f.state in wanted and f.kind == "file"]


class BatchView:
    """One batch of a run's list, as `ingest.main` sees it (A8″): the loop's indexes are mapped to
    the rows by path, so a file the loop drops before it starts (not found, not supported) is said
    to have failed instead of shifting every later row. A folder batch's ``begin`` turns the
    folder's row into its files. ``end`` does nothing: the queue ends the run when the list is
    empty, not when one batch is."""

    def __init__(self, run: IngestRun, batch: Batch) -> None:
        self._run = run
        self._batch = batch
        self._ids: list[str] = []
        self._next = 0
        #: The batch ended before its end because the next file's row was removed: the queue
        #: starts a new batch from the rows still waiting (A8″, a removal found by driving it).
        self.cut_short = False

    def begin(self, paths: list[Path]) -> None:
        from doc_assistant.ingest.registry import pathkey

        if self._batch.kind == "folder":
            self._ids = self._run.expand(self._batch.ids[0], paths)
            return
        by_key = {pathkey(self._run.item(i).path): i for i in self._batch.ids}
        self._ids = []
        for p in paths:
            # The loop only ever drops files from a selection, never adds one; a path the batch
            # does not hold is not reported rather than guessed at.
            self._ids.append(by_key.pop(pathkey(p), ""))
        for item_id in by_key.values():
            self._run.mark(
                item_id,
                "failed",
                "Not read: the file was not found, or is not a kind this app reads.",
            )

    def checkpoint(self) -> bool:
        """The run's checkpoint, and one more: the loop asks before each file, in order, so a file
        whose row was removed after the batch began ends the batch there instead of being read."""
        if self._run.checkpoint():
            return True
        index, self._next = self._next, self._next + 1
        item_id = self._ids[index] if index < len(self._ids) else ""
        if item_id and self._run.item(item_id).state == "removed":
            self.cut_short = True
            return True
        return False

    def reading(self, index: int) -> None:
        item_id = self._ids[index]
        if item_id and self._run.item(item_id).state != "removed":
            self._run.mark(item_id, "reading")

    def record(self, index: int, done: DocumentResult) -> None:
        item_id = self._ids[index]
        if item_id and self._run.item(item_id).state != "removed":
            row = self._run.item(item_id)
            out = outcome_of(row.path, row.name, done)
            self._run.mark(item_id, out.state, out.detail)

    def end(self) -> None:
        return None

    def stop_requested(self) -> bool:
        return self._run.stop_requested()
