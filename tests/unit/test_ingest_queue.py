"""The engine's one list (A8″, `docs/specs/ingestion-queue.md`).

What arrives during a run joins the end of its list and runs after what is ahead of it; a file
already waiting is not added twice; a waiting item can be removed; Stop ends the whole list; the
library folder becomes its files when it starts; a re-run is a row of the same list; the work a run
does once (``on_drained``) happens once, when the list is empty.

Offline: a fake indexing function drives the run the way `ingest.main` does (``begin``, then
``checkpoint`` / ``reading`` / ``record`` per file), and a fake re-run.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from doc_assistant.ingest.queue import IngestQueue
from doc_assistant.ingest.run import DocumentResult

TIMEOUT = 10


class _Ingest:
    """``ingest_fn``: records each batch; the first document can be held until released."""

    def __init__(self, folder: dict[str, list[Path]] | None = None) -> None:
        self.batches: list[list[str]] = []
        self.hold = threading.Event()
        self.hold.set()
        self.reading_first = threading.Event()
        self.folder = folder or {}

    def __call__(
        self,
        *,
        run: Any,
        on_progress: Any = None,
        files: list[Path] | None = None,
        scope: str | None = None,
    ) -> dict[str, int]:
        paths = list(files) if files is not None else list(self.folder.get(str(scope), []))
        self.batches.append([p.name for p in paths])
        run.begin(paths)
        added = 0
        for i, _ in enumerate(paths):
            if run.checkpoint():
                break
            run.reading(i)
            self.reading_first.set()
            self.hold.wait(TIMEOUT)
            run.record(i, DocumentResult("added", passages=3))
            added += 1
        run.end()
        return {"added": added, "skipped": 0, "error": 0}


@dataclass
class _Outcome:
    part: str
    status: str
    detail: str


@dataclass
class _Rerun:
    outcomes: tuple[_Outcome, ...]
    superseded: tuple[str, ...] = field(default=())


def _rerun(document_ids: list[str], parts: list[str]) -> _Rerun:
    return _Rerun(
        tuple(_Outcome(p, "ok", "re-extracted") for p in parts), (f"old-{document_ids[0]}",)
    )


def _queue(ingest: _Ingest, drains: list[tuple[list[str], bool]]) -> IngestQueue:
    return IngestQueue(
        ingest_fn=ingest, rerun_fn=_rerun, on_drained=lambda h, c: drains.append((h, c))
    )


def _states(q: IngestQueue) -> list[tuple[str, str]]:
    assert q.run is not None
    rows, _ = q.run.page(0, 1000, "all")
    return sorted((r.name, r.state) for r in rows)


def test_files_added_during_a_run_join_its_end_and_run_after(tmp_path: Path) -> None:
    ingest = _Ingest()
    ingest.hold.clear()
    drains: list[tuple[list[str], bool]] = []
    q = _queue(ingest, drains)
    q.add_files([tmp_path / "a.pdf", tmp_path / "b.pdf"])
    assert ingest.reading_first.wait(TIMEOUT)
    q.add_files([tmp_path / "c.pdf"])  # arrives while a.pdf is being read
    assert q.run is not None and q.run.counts()["total"] == 3, "one list, counted whole"
    ingest.hold.set()
    q.join(TIMEOUT)
    assert ingest.batches == [["a.pdf", "b.pdf"], ["c.pdf"]], "c waits for the batch ahead of it"
    assert _states(q) == [("a.pdf", "added"), ("b.pdf", "added"), ("c.pdf", "added")]
    assert drains == [([], True)], "the run's end work happens once"
    assert q.state == "done" and q.tally["added"] == 3


def test_a_file_already_waiting_is_not_added_twice(tmp_path: Path) -> None:
    ingest = _Ingest()
    ingest.hold.clear()
    q = _queue(ingest, [])
    q.add_files([tmp_path / "a.pdf", tmp_path / "b.pdf"])
    assert ingest.reading_first.wait(TIMEOUT)
    assert q.add_files([tmp_path / "b.pdf", tmp_path / "a.pdf"]) == []
    ingest.hold.set()
    q.join(TIMEOUT)
    assert ingest.batches == [["a.pdf", "b.pdf"]]


def test_a_waiting_item_can_be_removed_and_a_started_one_cannot(tmp_path: Path) -> None:
    ingest = _Ingest()
    ingest.hold.clear()
    q = _queue(ingest, [])
    [first] = q.add_files([tmp_path / "a.pdf"])
    assert ingest.reading_first.wait(TIMEOUT)
    [later] = q.add_files([tmp_path / "later.pdf"])
    assert not q.remove(first), "a.pdf is being read"
    assert q.remove(later)
    ingest.hold.set()
    q.join(TIMEOUT)
    assert ingest.batches == [["a.pdf"]]
    assert q.run is not None and q.run.counts()["total"] == 1, "a removed row is not counted"


def test_stop_ends_the_whole_list_and_the_next_add_starts_a_new_one(tmp_path: Path) -> None:
    ingest = _Ingest()
    ingest.hold.clear()
    drains: list[tuple[list[str], bool]] = []
    q = _queue(ingest, drains)
    q.add_files([tmp_path / "a.pdf", tmp_path / "b.pdf"])
    assert ingest.reading_first.wait(TIMEOUT)
    q.add_files([tmp_path / "c.pdf"])
    assert q.run is not None
    q.run.stop()
    ingest.hold.set()
    q.join(TIMEOUT)
    assert _states(q) == [("a.pdf", "added"), ("b.pdf", "not_started"), ("c.pdf", "not_started")]
    assert q.state == "stopped" and len(drains) == 1
    stopped = q.run
    q.add_files([tmp_path / "d.pdf"])
    q.join(TIMEOUT)
    assert q.run is not stopped and _states(q) == [("d.pdf", "added")]


def test_the_folder_becomes_its_files_when_it_starts(tmp_path: Path) -> None:
    folder = tmp_path / "library"
    ingest = _Ingest({str(folder): [folder / "x.pdf", folder / "y.pdf"]})
    q = _queue(ingest, [])
    q.add_folder(folder, "Library folder")
    q.join(TIMEOUT)
    assert _states(q) == [("x.pdf", "added"), ("y.pdf", "added")]


def test_a_re_run_is_a_row_of_the_same_list_and_its_hashes_reach_the_end(tmp_path: Path) -> None:
    ingest = _Ingest()
    ingest.hold.clear()
    drains: list[tuple[list[str], bool]] = []
    q = _queue(ingest, drains)
    q.add_files([tmp_path / "a.pdf"])
    assert ingest.reading_first.wait(TIMEOUT)
    [rerun] = q.add_rerun([("doc-1", "survey.pdf · Text & chunks")], ["text"])
    assert q.ahead_of(rerun) == 1, "it waits for the file being read"
    ingest.hold.set()
    q.join(TIMEOUT)
    assert q.run is not None
    row = q.run.item(rerun)
    assert (row.state, row.kind, row.detail) == ("done", "rerun", "re-extracted")
    assert drains == [(["old-doc-1"], True)]


def test_a_file_removed_after_its_batch_began_is_not_read(tmp_path: Path) -> None:
    """Found by driving the list: the batch held the file, so removing its row did nothing."""
    ingest = _Ingest()
    ingest.hold.clear()
    q = _queue(ingest, [])
    q.add_files([tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"])
    assert ingest.reading_first.wait(TIMEOUT)
    assert q.run is not None
    b = next(r.id for r in q.run.page(0, 10, "all")[0] if r.name == "b.pdf")
    assert q.remove(b), "b.pdf waits in the running batch, so it can still be taken out"
    ingest.hold.set()
    q.join(TIMEOUT)
    assert _states(q) == [("a.pdf", "added"), ("c.pdf", "added")]
    assert ingest.batches == [["a.pdf", "b.pdf", "c.pdf"], ["c.pdf"]], "c runs in a new batch"
