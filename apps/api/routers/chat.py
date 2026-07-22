"""Chat router — the answer/turn surface.

Owns the SSE chat stream (and its sync-generator → event-loop bridge), the retrieval-only A/B
compare, claim adjudication, conversation export, and the per-answer source/figure artifact
lookups. The ``_sse``/``_event_stream`` helpers live here because only this surface produces the
``TurnEvent`` stream (PR-M2).
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from apps.api.models import (
    AdjudicateRequest,
    ChatRequest,
    CompareRequest,
    CompareResultPayload,
    ExportRequest,
    TurnResultPayload,
)
from apps.api.sessions import SessionStore
from doc_assistant.chat_controller import (
    ChatController,
    RagOverrides,
    Result,
    Session,
    Step,
    Token,
)
from doc_assistant.ingest.figures import load_figure_image_paths
from doc_assistant.provenance import get_record

router = APIRouter()


def _sse(event: object) -> ServerSentEvent | None:
    """Map one TurnEvent to an SSE event (None for unknown types)."""
    if isinstance(event, Token):
        return ServerSentEvent(event="token", data=event.text)
    if isinstance(event, Step):
        return ServerSentEvent(
            event="step", data=json.dumps({"name": event.name, "status": event.status})
        )
    if isinstance(event, Result):
        return ServerSentEvent(
            event="result",
            data=TurnResultPayload.from_turn_result(event.result).model_dump_json(),
        )
    return None


async def _event_stream(
    controller: ChatController,
    session: Session,
    text: str,
    overrides: RagOverrides | None = None,
    *,
    scope_folder_id: str | None = None,
) -> AsyncIterator[ServerSentEvent]:
    """Map the controller's sync ``TurnEvent`` generator to SSE events 1:1.

    ``handle_message`` is a **sync, blocking** generator — the LLM token stream does
    blocking network reads, and retrieval is heavy CPU. Iterated directly on the event
    loop it stalls SSE flushing, so the whole answer bursts out at the end. Instead, run it
    in a worker thread and hand events to the loop through a queue: the loop stays free to
    flush each token as the model produces it (M2 ADR-2; the threadpool note made good)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()
    done = object()

    def worker() -> None:
        try:
            for event in controller.handle_message(
                session, text, overrides=overrides, scope_folder_id=scope_folder_id
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event = await queue.get()
        if event is done:
            break
        sse = _sse(event)
        if sse is not None:
            yield sse
    yield ServerSentEvent(event="done", data="")


@router.post("/api/chat")
async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
    controller: ChatController = request.app.state.controller
    sessions: SessionStore = request.app.state.sessions
    session = sessions.get_or_create(body.session_id)  # unknown id → fresh conversation
    overrides = (
        RagOverrides(
            top_k=body.overrides.top_k,
            synthesis_mode=body.overrides.synthesis_mode,
            use_multi_query=body.overrides.use_multi_query,
            epistemics_markers_enabled=body.overrides.epistemics_markers_enabled,
            reviewer_evidence_chars=body.overrides.reviewer_evidence_chars,
        )
        if body.overrides is not None
        else None
    )
    return EventSourceResponse(
        _event_stream(
            controller, session, body.text, overrides, scope_folder_id=body.scope_folder_id
        )
    )


@router.post("/api/compare")
def compare_route(request: Request, body: CompareRequest) -> CompareResultPayload:
    """A/B-compare retrieval (U6): run ``text`` under the locked defaults (A) and the session
    override (B), returning both source sets + the diff + the honesty note. **$0** — retrieval
    only, no generation. Request-scoped overrides, no module-global mutation (ADR-010)."""
    controller: ChatController = request.app.state.controller
    overrides = (
        RagOverrides(
            top_k=body.overrides.top_k,
            synthesis_mode=body.overrides.synthesis_mode,
            use_multi_query=body.overrides.use_multi_query,
            epistemics_markers_enabled=body.overrides.epistemics_markers_enabled,
            reviewer_evidence_chars=body.overrides.reviewer_evidence_chars,
        )
        if body.overrides is not None
        else RagOverrides()
    )
    return CompareResultPayload.from_result(
        controller.compare_retrieval(body.text, overrides, body.scope_folder_id)
    )


@router.post("/api/claims/{claim_id}/adjudicate")
def adjudicate(request: Request, claim_id: str, body: AdjudicateRequest) -> dict[str, bool]:
    controller: ChatController = request.app.state.controller
    try:
        controller.adjudicate(claim_id, body.decision, edited_text=body.edited_text)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True}


@router.post("/api/export")
def export(request: Request, body: ExportRequest) -> FileResponse:
    controller: ChatController = request.app.state.controller
    sessions: SessionStore = request.app.state.sessions
    # get_or_create (not get): a past/reopened conversation's session_id isn't in the
    # in-memory store, but export sources from the durable transcript by id. An id with no
    # persisted turns falls through to the "nothing to export" 400 below.
    session = sessions.get_or_create(body.session_id)
    message, path = controller.export_conversation(session, dev=body.dev)
    if path is None:  # nothing to export yet
        raise HTTPException(status_code=400, detail=message)
    return FileResponse(str(path), media_type="text/markdown", filename=path.name)


@router.get("/api/figures/{figure_id}")
def figure(figure_id: str) -> FileResponse:
    path = load_figure_image_paths([figure_id]).get(figure_id)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="figure not found")
    return FileResponse(path, media_type="image/png")


@router.get("/api/source/{record_id}/{n}")
def source(record_id: str, n: int) -> dict[str, Any]:
    prov = get_record(record_id)
    if prov is None or n < 1 or n > len(prov.retrieved_chunks):
        raise HTTPException(status_code=404, detail="source not found")
    chunk = prov.retrieved_chunks[n - 1]
    return {
        "n": n,
        "filename": chunk.filename,
        "page": chunk.page,
        "section": chunk.section,
        "reranker_score": chunk.reranker_score,
        "excerpt": chunk.chunk_excerpt,
    }
