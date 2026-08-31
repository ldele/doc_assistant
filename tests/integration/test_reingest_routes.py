"""The per-part re-ingest routes (ADR-048, ROADMAP 20/21).

Drives the API shell with an injected `reingest_fn`, so nothing here loads torch or touches a
vector store — the core's own behaviour is covered by `test_reingest_parts.py`. What is tested
here is what the *shell* owns: validation before the 202, the two 409s, and the position/outcome
split in the status object.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient


class _FakeController:
    def chunk_count(self) -> int:
        return 0


class _FakeOutcome:
    def __init__(self, part: str, status: str) -> None:
        self.document_id = "doc-1"
        self.filename = "paper.pdf"
        self.part = part
        self.status = status
        self.detail = f"{part} {status}"


class _FakeResult:
    def __init__(self, outcomes: list[_FakeOutcome]) -> None:
        self.outcomes = outcomes
        self.ok = sum(1 for o in outcomes if o.status == "ok")
        self.skipped = sum(1 for o in outcomes if o.status == "skipped")
        self.errors = sum(1 for o in outcomes if o.status == "error")


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(controller=_FakeController())  # type: ignore[arg-type]
    with TestClient(app) as c:
        yield c


def _install(client: TestClient, fn: Any) -> None:
    client.app.state.reingest_fn = fn  # type: ignore[attr-defined]


def _wait_for_done(client: TestClient, timeout: float = 5.0) -> dict[str, Any]:
    """Poll until the job leaves `running` — the route is 202 + poll by design."""
    deadline = threading.Event()
    for _ in range(int(timeout / 0.02)):
        body = client.get("/api/library/reingest/status").json()
        if body["state"] != "running":
            return body
        deadline.wait(0.02)
    raise AssertionError("re-ingest never left the running state")


def test_options_serve_the_registry_including_what_is_declined(client: TestClient) -> None:
    """The client must not hardcode a cost — that is the whole honesty of the control."""
    body = client.get("/api/library/reingest/options").json()
    assert [p["id"] for p in body["parts"]] == [
        "metadata",
        "crops",
        "figures",
        "references",
        "text",
    ]
    for part in body["parts"]:
        assert part["cost"], part["id"]
    assert [p["id"] for p in body["parts"] if p["moves_identity"]] == ["text"]
    # Named, not hidden: a user who cannot find the button is told there is no button.
    assert "Connections" in body["corpus_wide"]


def test_an_unknown_part_is_400_before_any_job_starts(client: TestClient) -> None:
    started: list[Any] = []
    _install(client, lambda *a, **k: started.append(a) or _FakeResult([]))

    r = client.post(
        "/api/library/documents/reingest",
        json={"document_ids": ["doc-1"], "parts": ["metadata", "nonsense"]},
    )
    assert r.status_code == 400
    assert "nonsense" in str(r.json()["detail"])
    assert started == [], "a bad body must not start a job"
    assert client.get("/api/library/reingest/status").json()["state"] == "idle"


def test_an_empty_body_is_refused_by_the_schema(client: TestClient) -> None:
    for body in (
        {"document_ids": [], "parts": ["metadata"]},
        {"document_ids": ["d"], "parts": []},
    ):
        assert client.post("/api/library/documents/reingest", json=body).status_code == 422


def test_a_run_reports_counts_and_per_part_outcomes(client: TestClient) -> None:
    _install(
        client,
        lambda ids, parts, on_progress=None: _FakeResult(
            [_FakeOutcome("metadata", "ok"), _FakeOutcome("figures", "skipped")]
        ),
    )
    r = client.post(
        "/api/library/documents/reingest",
        json={"document_ids": ["doc-1"], "parts": ["metadata", "figures"]},
    )
    assert r.status_code == 202
    assert r.json() == {"started": True, "total": 2}

    body = _wait_for_done(client)
    assert body["state"] == "done"
    assert (body["ok"], body["skipped"], body["errors"]) == (1, 1, 0)
    assert [o["part"] for o in body["outcomes"]] == ["metadata", "figures"]
    # A skip carries its reason all the way to the client.
    assert body["outcomes"][1]["detail"]
    assert "1 re-run" in body["message"] and "1 skipped" in body["message"]


def test_position_is_reported_separately_from_outcome(client: TestClient) -> None:
    """`total`/`done` are where the run is; ok/skipped/errors are what it produced. A partial
    outcome reported as a final one is the failure this split exists to prevent."""
    seen: list[tuple[int, int, int]] = []

    def runner(ids: list[str], parts: list[str], on_progress: Any = None) -> _FakeResult:
        on_progress(0, 4, "paper.pdf · metadata")
        body = client.get("/api/library/reingest/status").json()
        seen.append((body["done"], body["total"], body["ok"]))
        on_progress(2, 4, "paper.pdf · figures")
        body = client.get("/api/library/reingest/status").json()
        seen.append((body["done"], body["total"], body["ok"]))
        return _FakeResult([_FakeOutcome("metadata", "ok")])

    _install(client, runner)
    client.post(
        "/api/library/documents/reingest",
        json={"document_ids": ["doc-1"], "parts": ["metadata", "figures"]},
    )
    _wait_for_done(client)
    assert seen == [(0, 4, 0), (2, 4, 0)], seen


def test_a_second_run_is_refused_while_one_is_running(client: TestClient) -> None:
    release = threading.Event()

    def runner(ids: list[str], parts: list[str], on_progress: Any = None) -> _FakeResult:
        release.wait(5.0)
        return _FakeResult([_FakeOutcome("metadata", "ok")])

    _install(client, runner)
    body = {"document_ids": ["doc-1"], "parts": ["metadata"]}
    assert client.post("/api/library/documents/reingest", json=body).status_code == 202
    try:
        assert client.post("/api/library/documents/reingest", json=body).status_code == 409
    finally:
        release.set()
    _wait_for_done(client)


def test_a_run_is_refused_while_an_ingest_is_running(client: TestClient) -> None:
    """Both write the same chunk stores and the same rows; overlapping them races a re-extract
    against a corpus scan for the same file."""
    client.app.state.ingest_status.state = "running"  # type: ignore[attr-defined]
    try:
        r = client.post(
            "/api/library/documents/reingest",
            json={"document_ids": ["doc-1"], "parts": ["metadata"]},
        )
        assert r.status_code == 409
        assert "indexing" in r.json()["detail"]
    finally:
        client.app.state.ingest_status.state = "idle"  # type: ignore[attr-defined]


def test_a_crashed_run_leaves_a_readable_status(client: TestClient) -> None:
    def boom(ids: list[str], parts: list[str], on_progress: Any = None) -> _FakeResult:
        raise RuntimeError("simulated failure")

    _install(client, boom)
    client.post(
        "/api/library/documents/reingest",
        json={"document_ids": ["doc-1"], "parts": ["metadata"]},
    )
    body = _wait_for_done(client)
    assert body["state"] == "error"
    assert "simulated failure" in body["message"]
    assert body["current"] is None, "a finished run must not still name a file in flight"
