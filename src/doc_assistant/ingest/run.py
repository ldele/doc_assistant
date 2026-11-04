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

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FileState = Literal["waiting", "reading", "added", "unchanged", "failed", "not_started"]
"""Where one file of a run stands. ``not_started`` is a file a stopped run never reached."""

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
    "all": frozenset(("waiting", "reading", "added", "unchanged", "failed", "not_started")),
    "in_progress": frozenset(("reading",)),
    "waiting": frozenset(("waiting",)),
    "not_started": frozenset(("not_started",)),
    "finished": frozenset(("added", "unchanged")),
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
    """One row of the indexing screen."""

    path: str
    name: str
    state: FileState
    detail: str | None = None


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
        self._files: list[FileOutcome] = []

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
            self._files = [FileOutcome(str(p), p.name, "waiting") for p in paths]

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
            self._files[index] = FileOutcome(row.path, row.name, "reading")

    def record(self, index: int, done: DocumentResult) -> None:
        with self._lock:
            row = self._files[index]
            self._files[index] = outcome_of(row.path, row.name, done)

    def end(self) -> None:
        """The loop is over. Files a stop left behind become *not started*."""
        with self._lock:
            self._ended = True
            self._files = [
                FileOutcome(f.path, f.name, "not_started") if f.state == "waiting" else f
                for f in self._files
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
            for f in self._files:
                counts[f.state] += 1
            counts["total"] = len(self._files)
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
            return [Path(f.path) for f in self._files if f.state in wanted]
