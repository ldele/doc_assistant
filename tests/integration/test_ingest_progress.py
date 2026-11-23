"""The ingest progress channel: `ingest.main` -> the run's rows -> GET /api/ingest/status.

Since A8″ the position is read from the engine's one list (`ingest/run.py`), which `ingest.main`
reports each file to (``begin`` / ``reading`` / ``record``); the fakes below drive it the same way.

Why this file exists: before this, `/api/ingest/status` reported `{running, 0, 0, 0}` for the whole
duration of a run and only filled in at the end, so the desktop app had nothing to show while a
document was being indexed — a user who added a file saw no evidence it was working. These tests
pin the *position* channel (`total`/`done`/`current`) and, just as importantly, pin that it stays
separate from the *outcome* channel (`added`/`skipped`/`errors`), which remains end-of-run only.

All fakes (cpc §13): no model load, no Chroma, no network.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.ingest.run import DocumentResult

Sink = Callable[[int, int, str | None], None] | None


class FakeController:
    """The minimum `create_app` needs; `chunk_count` is read by the settings/health views."""

    def __init__(self, count: int = 0) -> None:
        self._count = count

    def chunk_count(self) -> int:
        return self._count

    def release_keyword_index(self) -> None:
        pass


def _client(ingest_fn: Any) -> TestClient:
    app = create_app(
        controller=FakeController(),
        ingest_fn=ingest_fn,
        controller_factory=lambda: FakeController(),
    )
    return TestClient(app)


def test_status_carries_position_while_a_run_is_in_flight() -> None:
    """The whole point: mid-run, the endpoint says how far along it is and what it is on.

    The fake blocks after reporting, so the assertion happens while the run is genuinely running —
    not after it, where the old end-of-run fields would have answered anyway.
    """
    reported = threading.Event()
    release = threading.Event()

    def fake_ingest(
        *, scope: str | None = None, on_progress: Sink = None, run: Any = None
    ) -> dict[str, int]:
        names = [f"paper-{i}.pdf" for i in range(11)] + ["hodgkin_huxley_1952.pdf"]
        run.begin([Path(n) for n in names])
        for i in range(3):
            run.reading(i)
            run.record(i, DocumentResult("added", passages=4))
        run.reading(11)  # the file in flight
        reported.set()
        release.wait(timeout=5)
        return {"added": 12, "skipped": 0, "error": 0}

    with _client(fake_ingest) as c:
        assert c.post("/api/ingest").status_code == 202
        assert reported.wait(timeout=5), "the sink was never called"

        body = c.get("/api/ingest/status").json()
        assert body["state"] == "running"
        assert (body["done"], body["total"]) == (3, 12)
        assert body["current"] == "hodgkin_huxley_1952.pdf"
        # The outcome channel must NOT have been touched by a position report: reporting 3-of-12
        # as "3 added" would claim an outcome for documents that may still fail.
        assert (body["added"], body["skipped"], body["errors"]) == (0, 0, 0)

        release.set()


def test_nothing_is_in_flight_once_the_run_is_done() -> None:
    """`current` must clear, or the bar keeps naming a file that finished long ago."""

    def fake_ingest(
        *, scope: str | None = None, on_progress: Sink = None, run: Any = None
    ) -> dict[str, int]:
        run.begin([Path("a.pdf"), Path("b.pdf")])
        for i in range(2):
            run.reading(i)
            run.record(i, DocumentResult("added", passages=4))
        return {"added": 2, "skipped": 0, "error": 0}

    with _client(fake_ingest) as c:
        c.post("/api/ingest")
        body = _await_final(c)
        assert body["state"] == "done"
        assert body["current"] is None
        assert (body["done"], body["total"]) == (2, 2)
        assert body["added"] == 2


def test_a_failed_run_does_not_leave_a_file_in_flight() -> None:
    """A crash mid-document must clear `current` too — the error state is still a resting state."""

    def boom(
        *, scope: str | None = None, on_progress: Sink = None, run: object = None
    ) -> dict[str, int]:
        assert on_progress is not None
        on_progress(1, 4, "halfway.pdf")
        raise RuntimeError("disk full")

    with _client(boom) as c:
        c.post("/api/ingest")
        body = _await_final(c)
        assert body["state"] == "error"
        assert body["message"] == "disk full"
        assert body["current"] is None, "a failed run must not keep claiming a file is indexing"


def test_a_new_run_does_not_open_on_the_previous_run_s_position() -> None:
    """Stale position is worse than none: the bar would start part-full and jump backwards."""
    calls: list[int] = []

    def fake_ingest(
        *, scope: str | None = None, on_progress: Sink = None, run: Any = None
    ) -> dict[str, int]:
        calls.append(1)
        paths = [Path(f"p{i}.pdf") for i in range(9)] if len(calls) == 1 else []
        run.begin(paths)
        for i, _ in enumerate(paths):
            run.reading(i)
            run.record(i, DocumentResult("skipped"))
        return {"added": 0, "skipped": len(paths), "error": 0}

    with _client(fake_ingest) as c:
        c.post("/api/ingest")
        assert _await_final(c)["done"] == 9

        # The second run finds no file at all, so whatever the endpoint says is what a new list
        # starts from.
        c.post("/api/ingest")
        body = _await_final(c)
        assert (body["done"], body["total"]) == (0, 0)
        assert body["current"] is None


def _await_final(c: TestClient, tries: int = 100) -> dict[str, Any]:
    """Poll until the run leaves `running`. The worker is a real thread, so this cannot be sync."""
    for _ in range(tries):
        body = c.get("/api/ingest/status").json()
        if body["state"] != "running":
            return body  # type: ignore[no-any-return]
        threading.Event().wait(0.05)
    pytest.fail("ingest never left the running state")
