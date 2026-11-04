"""The indexing screen's routes: pause, resume and stop, the files page, and the follow-up runs.

The injected ingest function drives the real `IngestRun` the way `ingest.main` does (begin, a
checkpoint before each document, a record after it, end), over made-up outcomes, so the routes are
proved against the real controls without loading a model. The real loop is proved in
`tests/integration/ingest/test_ingest_run_controls.py`. All fakes (cpc §13): no Chroma, no network.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.ingest.run import DocumentResult, IngestRun

FILES = [Path(f"/lib/{n}") for n in ("a.pdf", "scan.pdf", "c.pdf", "d.pdf")]
OUTCOMES = {
    "a.pdf": DocumentResult("added", passages=12),
    "scan.pdf": DocumentResult("error", reason="No text found: a scan needs OCR."),
    "c.pdf": DocumentResult("skipped"),
    "d.pdf": DocumentResult("added", passages=3),
}


class FakeController:
    def chunk_count(self) -> int:
        return 0

    def release_keyword_index(self) -> None:
        pass


class FakeIngest:
    """Runs the loop's side of `IngestRun` over `FILES`, holding before each document until let.

    `step` lets one document through, so a test can ask for a pause or a stop while a document is
    genuinely in flight and see the request wait for it.
    """

    def __init__(self) -> None:
        self.step = threading.Semaphore(0)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        scope: str | None = None,
        files: list[Path] | None = None,
        on_progress: Any = None,
        run: IngestRun | None = None,
    ) -> dict[str, int]:
        assert run is not None, "the route must hand the run to the ingest"
        self.calls.append({"scope": scope, "files": files})
        todo = files if files is not None else FILES
        run.begin(todo)
        stats = {"added": 0, "skipped": 0, "error": 0}
        for i, path in enumerate(todo):
            if run.checkpoint():
                break
            on_progress(i, len(todo), path.name)
            run.reading(i)
            self.step.acquire(timeout=10)
            outcome = OUTCOMES[path.name]
            stats[outcome.result] += 1
            run.record(i, outcome)
        run.end()
        return stats


def _client(fake: FakeIngest) -> TestClient:
    app = create_app(
        controller=FakeController(),  # type: ignore[arg-type]
        ingest_fn=fake,
        controller_factory=FakeController,  # type: ignore[arg-type]
    )
    return TestClient(app)


def _until(c: TestClient, predicate, tries: int = 400) -> dict[str, Any]:
    for _ in range(tries):
        body = c.get("/api/ingest/status").json()
        if predicate(body):
            return body  # type: ignore[no-any-return]
        threading.Event().wait(0.01)
    pytest.fail(f"never reached the state; last: {body}")


def test_nothing_to_pause_or_list_before_any_run() -> None:
    with _client(FakeIngest()) as c:
        assert c.post("/api/ingest/pause").status_code == 409
        assert c.post("/api/ingest/stop").status_code == 409
        assert c.get("/api/ingest/files").json() == {"total": 0, "offset": 0, "files": []}
        status = c.get("/api/ingest/status").json()
        assert (status["control"], status["counts"]) == ("none", None)
        assert c.post("/api/ingest", json={"rerun": "failed"}).status_code == 400


def test_pause_waits_for_the_document_in_flight_then_holds_until_resume() -> None:
    fake = FakeIngest()
    with _client(fake) as c:
        c.post("/api/ingest")
        _until(c, lambda b: b["current"] == "a.pdf")
        assert c.post("/api/ingest/pause").json()["control"] == "pausing"
        fake.step.release()  # a.pdf finishes
        paused = _until(c, lambda b: b["control"] == "paused")
        assert paused["state"] == "running"  # still a run in flight: nothing else may start
        assert paused["counts"]["added"] == 1 and paused["counts"]["waiting"] == 3
        assert c.post("/api/ingest").status_code == 409

        assert c.post("/api/ingest/resume").json()["control"] == "none"
        for _ in range(3):
            fake.step.release()
        done = _until(c, lambda b: b["state"] != "running")
        assert done["state"] == "done" and done["counts"]["failed"] == 1


def test_stop_ends_after_the_document_in_flight_and_offers_the_rest() -> None:
    fake = FakeIngest()
    with _client(fake) as c:
        c.post("/api/ingest")
        _until(c, lambda b: b["current"] == "a.pdf")
        fake.step.release()
        _until(c, lambda b: b["current"] == "scan.pdf")
        assert c.post("/api/ingest/stop").json()["control"] == "stopping"
        fake.step.release()  # scan.pdf finishes (and fails)
        stopped = _until(c, lambda b: b["state"] != "running")
        assert stopped["state"] == "stopped"
        assert stopped["control"] == "none"
        counts = stopped["counts"]
        assert (counts["added"], counts["failed"], counts["not_started"]) == (1, 1, 2)

        # The screen's list: failed first, then not started, then done; each with its words.
        page = c.get("/api/ingest/files").json()
        assert [(f["name"], f["state"]) for f in page["files"]] == [
            ("scan.pdf", "failed"),
            ("c.pdf", "not_started"),
            ("d.pdf", "not_started"),
            ("a.pdf", "added"),
        ]
        assert page["files"][0]["detail"] == "No text found: a scan needs OCR."
        assert page["files"][3]["detail"] == "12 passages"
        one = c.get("/api/ingest/files", params={"group": "not_started", "limit": 1}).json()
        assert one["total"] == 2 and [f["name"] for f in one["files"]] == ["c.pdf"]

        # Index the remaining: the server takes the two from its own record of the run.
        for _ in range(2):
            fake.step.release()
        assert c.post("/api/ingest", json={"rerun": "remaining"}).status_code == 202
        _until(c, lambda b: b["state"] != "running")
        assert [p.name for p in fake.calls[-1]["files"]] == ["c.pdf", "d.pdf"]


def test_retry_takes_the_failed_files_and_refuses_when_there_are_none() -> None:
    fake = FakeIngest()
    for _ in range(4):
        fake.step.release()
    with _client(fake) as c:
        c.post("/api/ingest")
        _until(c, lambda b: b["state"] != "running")
        fake.step.release()
        assert c.post("/api/ingest", json={"rerun": "failed"}).status_code == 202
        _until(c, lambda b: b["state"] != "running")
        assert [p.name for p in fake.calls[-1]["files"]] == ["scan.pdf"]
        # That retry failed again, so a second retry has it; a run with nothing remaining has none.
        assert c.post("/api/ingest", json={"rerun": "remaining"}).status_code == 400
        both = c.post("/api/ingest", json={"rerun": "failed", "paths": ["a.pdf"]})
        assert both.status_code == 400


@pytest.mark.parametrize("limit", [0, 501])
def test_the_page_size_is_bounded(limit: int) -> None:
    with _client(FakeIngest()) as c:
        assert c.get("/api/ingest/files", params={"limit": limit}).status_code == 422
