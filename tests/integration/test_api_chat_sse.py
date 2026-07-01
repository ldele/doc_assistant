"""Integration tests for the FastAPI/SSE boundary (PR-M2).

A **fake ChatController** yields canned ``TurnEvent``s — the API layer is exercised for
request→call mapping, SSE framing, and JSON serialization only; never a real pipeline,
LLM, or network (cpc §13). Uses FastAPI's ``TestClient``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.chat_controller import (
    ClaimView,
    Result,
    SourceView,
    Step,
    Token,
    TurnResult,
    UsageView,
)


def _turn_result() -> TurnResult:
    return TurnResult(
        answer="Hello world [1].",
        mode="ai",
        sources=[
            SourceView(1, "[1] a.pdf · p.1", "…", None, "d1:0", ["contested"]),
        ],
        flagged_claims=[ClaimView("c1", 1, "a claim", "unsupported")],
        usage=UsageView(10, 20, 30, None, True),
        standalone_query="hi",
        record_id="rec-1",
        provenance_card_md="P",
        claim_review_md="C",
        sources_md="S",
        usage_md="U",
        citation_note_md="",
    )


class FakeController:
    """Canned TurnEvent stream — no pipeline, no DB, no LLM."""

    def __init__(self) -> None:
        self.adjudicated: list[tuple[str, str, str | None]] = []

    def chunk_count(self) -> int:
        return 7

    def handle_message(self, session: object, text: str) -> Iterator[object]:
        yield Step("Searching documents", "Found 1 passage")
        yield Token("Hello ")
        yield Token("world [1].")
        yield Result(_turn_result())

    def adjudicate(self, claim_id: str, decision: str, edited_text: str | None = None) -> None:
        self.adjudicated.append((claim_id, decision, edited_text))

    def export_conversation(self, session: object, *, dev: bool) -> tuple[str, object]:
        return ("Nothing to export yet — ask a question first.", None)


def _client(controller: FakeController | None = None) -> TestClient:
    return TestClient(create_app(controller=controller or FakeController()))


def _parse_sse(lines: Iterator[str]) -> list[dict[str, str]]:
    """Parse an SSE byte/line stream into a list of ``{event, data}`` dicts."""
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):  # comment / heartbeat
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            current["event"] = value
        elif field == "data":
            current["data"] = current.get("data", "") + value
    if current:
        events.append(current)
    return events


# ============================================================
# Tests
# ============================================================


def test_health():
    r = _client().get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunk_count"] == 7
    assert "model" in body and "embedding_model" in body


def test_chat_streams_sse_events_in_order():
    with _client().stream("POST", "/api/chat", json={"text": "hi", "session_id": "s1"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = _parse_sse(resp.iter_lines())

    kinds = [e["event"] for e in events]
    assert "token" in kinds
    assert kinds.count("result") == 1
    assert kinds[-1] == "done"
    # token order preserved
    tokens = [e["data"] for e in events if e["event"] == "token"]
    assert "".join(tokens) == "Hello world [1]."
    # the result event carries a valid TurnResult payload
    result = next(e for e in events if e["event"] == "result")
    payload = json.loads(result["data"])
    assert payload["answer"] == "Hello world [1]." and payload["mode"] == "ai"
    assert payload["sources"][0]["markers"] == ["contested"]


def test_chat_creates_session_for_unknown_id():
    # An unknown session_id on /chat is fine — it starts a new conversation.
    with _client().stream("POST", "/api/chat", json={"text": "q", "session_id": "new"}) as resp:
        assert resp.status_code == 200


def test_adjudicate_maps_to_controller():
    fake = FakeController()
    client = _client(fake)
    r = client.post("/api/claims/claim-1/adjudicate", json={"decision": "accepted"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert fake.adjudicated == [("claim-1", "accepted", None)]


def test_adjudicate_rejects_bad_decision():
    r = _client().post("/api/claims/c1/adjudicate", json={"decision": "bogus"})
    assert r.status_code == 422  # pydantic Literal validation


def test_export_unknown_session_is_404():
    r = _client().post("/api/export", json={"session_id": "nope", "dev": False})
    assert r.status_code == 404


def test_figure_served_and_missing(tmp_path, monkeypatch):
    import apps.api.main as api_main

    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic — enough for a served-bytes check
    monkeypatch.setattr(api_main, "load_figure_image_paths", lambda ids: {"fig1": str(png)})
    client = _client()
    ok = client.get("/api/figures/fig1")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"

    monkeypatch.setattr(api_main, "load_figure_image_paths", lambda ids: {})
    assert client.get("/api/figures/missing").status_code == 404


def test_settings_read_and_post_validation():
    client = _client()
    body = client.get("/api/settings").json()
    assert "top_k" in body  # locked knobs still present in the read view
    assert "data_home" in body and "source_dir" in body and "chunk_count" in body
    # POST now wires the write path (set the source folder) — an empty body fails validation.
    assert client.post("/api/settings", json={}).status_code == 422
