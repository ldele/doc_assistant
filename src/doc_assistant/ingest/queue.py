"""One list for every ingestion, run in order on one thread (A8″, `docs/specs/ingestion-queue.md`).

An add, a selection, the library folder and a document's re-run each become rows of one
`IngestRun`, so nothing is refused because something else is running: what arrives during a run
joins the end of its list. One thread takes the rows in order: consecutive waiting files as one
batch through ``ingest_fn(files=…)``, the folder through ``ingest_fn(scope=…)`` (its row becomes
its files when it starts), and a re-run one document at a time through ``rerun_fn``.

What a run does once happens once, when the list is empty: ``on_drained`` is told the hashes the
re-runs replaced and whether any text changed, so the engine can forget them and reload (the API's
part, since the engine is the API's). Pause and stop act on the whole list, between documents.

Kept in memory (decision 1): a list left waiting when the engine stops is not lost, since its files
are already in the library folder, where Manage files lists them as not indexed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from doc_assistant.ingest.run import Batch, IngestRun, short_reason

log = structlog.get_logger(__name__)


def _quiet(done: int, total: int, current: str | None) -> None:
    """The per-batch progress sink: the list's own rows carry the position now (A8″)."""


#: ``ingest_fn(files=… | scope=…, run=…, on_progress=…) -> {"added", "skipped", "error"}``.
IngestFn = Callable[..., dict[str, int]]
#: ``rerun_fn(document_ids, parts) -> ReingestResult`` (`doc_assistant.reingest.rerun`).
RerunFn = Callable[[list[str], list[str]], Any]
#: ``on_drained(superseded_hashes, text_changed)``, once per run, after its last item.
DrainFn = Callable[[list[str], bool], None]


class IngestQueue:
    """The engine's one list: add to it from any thread; one worker runs it."""

    def __init__(
        self, *, ingest_fn: IngestFn, rerun_fn: RerunFn, on_drained: DrainFn | None = None
    ) -> None:
        self._ingest_fn = ingest_fn
        self._rerun_fn = rerun_fn
        self._on_drained = on_drained
        self._lock = threading.Lock()
        self._run: IngestRun | None = None
        self._thread: threading.Thread | None = None
        self._thread_run: IngestRun | None = None
        #: ``idle`` before any run; ``running`` until the last item and the reload are done; then
        #: ``done``, ``stopped`` or ``error``.
        self.state = "idle"
        self.error: str | None = None
        self.tally = {"added": 0, "skipped": 0, "error": 0}
        #: Each re-run row's per-part outcomes (`reingest.PartOutcome`), for the re-run dialog.
        self.outcomes: dict[str, list[Any]] = {}

    # --- reads ------------------------------------------------------------------------------

    @property
    def run(self) -> IngestRun | None:
        """The current run, or the last one once it has ended; ``None`` before any."""
        with self._lock:
            return self._run

    def ahead_of(self, item_id: str) -> int:
        """How many rows wait or run before ``item_id``: what a re-run asked now waits for."""
        run = self.run
        return run.ahead_of(item_id) if run is not None else 0

    # --- adding ---------------------------------------------------------------------------------

    def _open_run(self) -> IngestRun:
        """The run an add joins: the current one while its loop goes on, else a new one. Called
        under ``_lock``, which the loop holds too when it decides to end: the two never race."""
        if self._run is None or self._run.closed:
            self._run = IngestRun()
            self.state = "running"
            self.error = None
            self.tally = {"added": 0, "skipped": 0, "error": 0}
        return self._run

    def add_files(self, paths: list[Path], detail: str | None = None) -> list[str]:
        """Files to index, at the end of the list. Returns the ids of the rows added."""
        with self._lock:
            run = self._open_run()
            ids = run.append_files(paths, detail)
            self._ensure_worker(run)
        log.info("ingest_items_added", kind="file", count=len(ids))
        return ids

    def add_folder(self, source: Path, name: str) -> str | None:
        """The library folder, at the end of the list: its new and changed files when it starts."""
        with self._lock:
            run = self._open_run()
            item_id = run.append_folder(source, name)
            self._ensure_worker(run)
        log.info("ingest_items_added", kind="folder", count=1)
        return item_id

    def add_rerun(self, documents: list[tuple[str, str]], parts: list[str]) -> list[str]:
        """One re-run row per ``(document_id, name)``, at the end of the list."""
        ids: list[str] = []
        with self._lock:
            run = self._open_run()
            for document_id, name in documents:
                item_id = run.append_rerun(document_id, name, tuple(parts))
                if item_id is not None:
                    ids.append(item_id)
            self._ensure_worker(run)
        log.info("ingest_items_added", kind="rerun", count=len(ids))
        return ids

    def remove(self, item_id: str) -> bool:
        """Take a waiting row out; ``False`` once it has started, or if it is not in the list."""
        run = self.run
        return run.remove(item_id) if run is not None else False

    def _ensure_worker(self, run: IngestRun) -> None:
        """Start the loop for ``run`` unless it is already running. Called under ``_lock``."""
        if self._thread_run is run and self._thread is not None and self._thread.is_alive():
            return
        previous = self._thread
        self._thread_run = run
        self._thread = threading.Thread(
            target=self._loop, args=(run, previous), name="ingest", daemon=True
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the loop to end (tests, and a clean shutdown)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # --- the worker -----------------------------------------------------------------------------

    def _loop(self, run: IngestRun, previous: threading.Thread | None) -> None:
        # A run whose loop ended while this add arrived may still be reloading the engine: wait for
        # it, so two reloads never overlap and this run reads the engine it left.
        if previous is not None and previous.is_alive():
            previous.join()
        superseded: list[str] = []
        changed = False
        failed: str | None = None
        try:
            while not run.stop_requested():
                with self._lock:
                    batch = run.next_batch()
                    if batch is None and run.close_if_drained():
                        break
                if batch is None:
                    continue
                if batch.kind == "rerun":
                    if run.checkpoint():
                        break
                    hashes, text = self._rerun(run, batch)
                    superseded += hashes
                    changed = changed or text
                else:
                    changed = self._index(run, batch) or changed
        except Exception as e:  # a crashed item must still leave a readable list and status
            log.exception("ingest_queue_failed")
            failed = short_reason(e)
        finally:
            run.end()
            if self._on_drained is not None:
                try:
                    self._on_drained(superseded, changed)
                except Exception:
                    log.exception("ingest_drain_failed")
            with self._lock:
                if self._run is run:
                    self.error = failed
                    self.state = "error" if failed else ("stopped" if run.stopped else "done")

    def _index(self, run: IngestRun, batch: Batch) -> bool:
        """Index one batch of files, or the folder. Returns whether anything was added."""
        view = run.view(batch)
        if batch.kind == "folder":
            stats = self._ingest_fn(
                scope=run.item(batch.ids[0]).path, run=view, on_progress=_quiet
            )
        else:
            paths = [Path(run.item(i).path) for i in batch.ids]
            stats = self._ingest_fn(files=paths, run=view, on_progress=_quiet)
        for key in self.tally:
            self.tally[key] += int(stats.get(key, 0))
        if not run.stop_requested() and not view.cut_short:
            # A batch that returned without a stop has run every row it held; one still waiting
            # was never reached by the loop, and left waiting it would be taken again, for ever.
            # A batch cut short for a removal leaves its other rows waiting for the next batch.
            for item_id in batch.ids:
                try:
                    row = run.item(item_id)
                except KeyError:  # the folder's row, which became its files
                    continue
                if row.state in ("waiting", "reading"):
                    if row.kind == "folder":
                        run.mark(item_id, "done", "Nothing new to index")
                    else:
                        run.mark(item_id, "failed", "Not read: the run ended before this file.")
        return int(stats.get("added", 0)) > 0

    def _rerun(self, run: IngestRun, batch: Batch) -> tuple[list[str], bool]:
        """Re-run one document's parts. Returns the hashes it replaced and whether text changed."""
        row = run.item(batch.ids[0])
        run.mark(row.id, "reading")
        document_id = row.document_id or ""
        try:
            result = self._rerun_fn([document_id], list(row.parts))
        except Exception as e:
            run.mark(row.id, "failed", short_reason(e))
            return [], False
        outcomes = list(getattr(result, "outcomes", ()))
        self.outcomes[row.id] = outcomes
        # The row's name already says which parts; one part's outcome needs no prefix.
        if len(outcomes) == 1:
            detail: str | None = outcomes[0].detail
        else:
            detail = "; ".join(f"{o.part}: {o.detail}" for o in outcomes) or None
        bad = any(o.status == "error" for o in outcomes)
        run.mark(row.id, "failed" if bad else "done", detail)
        return list(getattr(result, "superseded", ())), "text" in row.parts
