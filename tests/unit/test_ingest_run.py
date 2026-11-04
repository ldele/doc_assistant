"""`ingest.run`: each file's outcome, the screen's order and filters, and pause and stop.

Pure and offline. The controls are driven from a real second thread, the way the API drives them
while the ingest thread waits at a checkpoint.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from doc_assistant.ingest.run import (
    NO_TEXT,
    REASON_MAX_CHARS,
    UNCHANGED,
    DocumentResult,
    IngestRun,
    outcome_of,
    short_reason,
)


def _wait_for(predicate, timeout: float = 5.0) -> None:
    done = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        done.wait(0.01)
    pytest.fail("the condition never held")


# --- what a row says ----------------------------------------------------------------------------


def test_an_added_document_counts_its_passages() -> None:
    assert outcome_of("/p/a.pdf", "a.pdf", DocumentResult("added", passages=412)).detail == (
        "412 passages"
    )
    one = outcome_of("/p/a.pdf", "a.pdf", DocumentResult("added", passages=1))
    assert (one.state, one.detail) == ("added", "1 passage")


def test_a_document_with_no_passages_is_shown_as_failed_with_the_reason() -> None:
    # Recorded as added (its row exists), but no question will ever find it.
    row = outcome_of("/p/scan.pdf", "scan.pdf", DocumentResult("added", passages=0))
    assert (row.state, row.detail) == ("failed", NO_TEXT)


def test_skipped_means_unchanged_unless_there_is_a_reason() -> None:
    assert outcome_of("/p/a", "a", DocumentResult("skipped")).state == "unchanged"
    assert outcome_of("/p/a", "a", DocumentResult("skipped")).detail == UNCHANGED
    empty = outcome_of("/p/a", "a", DocumentResult("skipped", reason=NO_TEXT))
    assert (empty.state, empty.detail) == ("failed", NO_TEXT)


def test_an_error_keeps_its_reason() -> None:
    locked = "The file is password-protected."
    row = outcome_of("/p/a", "a", DocumentResult("error", reason=locked))
    assert (row.state, row.detail) == ("failed", locked)
    assert outcome_of("/p/a", "a", DocumentResult("error")).detail == "Could not read this file."


def test_a_reason_is_one_short_line() -> None:
    assert short_reason(ValueError("first line\nTraceback ...")) == "first line"
    assert short_reason(RuntimeError()) == "RuntimeError"
    long = short_reason(ValueError("x" * 1000))
    assert len(long) == REASON_MAX_CHARS and long.endswith("…")


# --- the ledger ---------------------------------------------------------------------------------


def _run_of(*names: str) -> IngestRun:
    run = IngestRun()
    run.begin([Path(f"/lib/{n}") for n in names])
    return run


def test_the_screen_lists_moving_then_failed_then_waiting_then_done() -> None:
    run = _run_of("a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf")
    run.record(0, DocumentResult("skipped"))  # unchanged
    run.record(1, DocumentResult("added", passages=3))
    run.record(2, DocumentResult("error", reason="bad"))
    run.reading(3)
    rows, total = run.page()
    assert total == 5
    assert [(r.name, r.state) for r in rows] == [
        ("d.pdf", "reading"),
        ("c.pdf", "failed"),
        ("e.pdf", "waiting"),
        ("b.pdf", "added"),
        ("a.pdf", "unchanged"),
    ]
    counts = run.counts()
    assert counts["total"] == 5 and counts["failed"] == 1 and counts["waiting"] == 1


def test_a_group_and_a_page_of_it() -> None:
    run = _run_of(*(f"{i:02}.md" for i in range(10)))
    for i in range(6):
        run.record(i, DocumentResult("added", passages=2))
    rows, total = run.page(offset=2, limit=3, group="finished")
    assert total == 6
    assert [r.name for r in rows] == ["02.md", "03.md", "04.md"]
    assert run.page(group="failed") == ([], 0)
    assert run.page(offset=50)[0] == []


def test_a_stop_leaves_the_rest_not_started_and_offers_them_again() -> None:
    run = _run_of("a.md", "b.md", "c.md", "d.md")
    run.record(0, DocumentResult("added", passages=1))
    run.record(1, DocumentResult("error", reason="bad"))
    run.stop()
    run.end()
    assert run.stopped
    assert run.counts()["not_started"] == 2
    assert run.paths("remaining") == [Path("/lib/c.md"), Path("/lib/d.md")]
    assert run.paths("failed") == [Path("/lib/b.md")]


def test_a_run_that_reaches_the_end_is_not_stopped() -> None:
    run = _run_of("a.md")
    run.record(0, DocumentResult("added", passages=1))
    run.end()
    # Ended but not yet reported done: what finished is being made searchable.
    assert not run.stopped and run.control() == "finishing"
    assert run.paths("remaining") == []


# --- the controls, across two threads -----------------------------------------------------------


def test_pause_waits_at_the_checkpoint_until_resumed() -> None:
    run = _run_of("a.md", "b.md")
    passed = threading.Event()
    run.pause()
    assert run.control() == "pausing"  # the document in flight has not finished yet

    def loop() -> None:
        stop = run.checkpoint()
        assert stop is False
        passed.set()

    worker = threading.Thread(target=loop, daemon=True)
    worker.start()
    _wait_for(lambda: run.control() == "paused")
    assert not passed.is_set()
    run.resume()
    worker.join(timeout=5)
    assert passed.is_set() and run.control() == "none"


def test_stop_wakes_a_paused_run_and_ends_it() -> None:
    run = _run_of("a.md")
    answer: list[bool] = []
    run.pause()
    worker = threading.Thread(target=lambda: answer.append(run.checkpoint()), daemon=True)
    worker.start()
    _wait_for(lambda: run.control() == "paused")
    run.stop()
    worker.join(timeout=5)
    assert answer == [True]
    assert run.control() == "stopping"
    run.end()
    assert run.control() == "stopping"  # through the time after the loop, until the run reports


def test_pause_after_stop_does_nothing() -> None:
    run = _run_of("a.md")
    run.stop()
    run.pause()
    assert run.checkpoint() is True  # returns at once: a stopped run never waits
