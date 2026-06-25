"""FastAPI desktop backend — a thin HTTP/SSE renderer over ``ChatController`` (PR-M2).

FastAPI is *just another shell* (like ``apps/cli.py``): it
maps ``TurnEvent`` → HTTP/SSE and HTTP requests → controller calls, and owns no
retrieval/provenance/claim logic. One ``ChatController`` is built per process in the
``lifespan`` (model load is expensive); the test path injects a fake controller via
``create_app(controller=...)`` so no real pipeline / LLM / network is touched (cpc §13).

Streaming: ``/api/chat`` returns ``text/event-stream`` — each ``TurnEvent`` becomes one
SSE event (``token`` / ``step`` / terminal ``result`` / ``done``). ``handle_message`` is a
**sync, blocking** generator (the LLM token stream); for a single-user local app we
iterate it directly on the event loop. A multi-client server would offload to a
threadpool (``anyio.to_thread``) — noted here, not needed for the desktop target.

Future ``SourceAdapter`` registry + ``/api/sources`` mount here (per the source-agnostic
companion goal) once a second concrete ingestion source exists — a seam, not an
abstraction to build now.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from apps.api.models import AdjudicateRequest, ChatRequest, ExportRequest, TurnResultPayload
from apps.api.sessions import SessionStore
from doc_assistant.chat_controller import ChatController, Result, Session, Step, Token
from doc_assistant.config import LLM_MODEL, LLM_PROVIDER, LOG_JSON, LOG_LEVEL
from doc_assistant.embeddings import get_active_model_name
from doc_assistant.figures import load_figure_image_paths
from doc_assistant.logging_config import configure_logging
from doc_assistant.provenance import get_record

# Local app only: the frontend speaks to the sidecar over 127.0.0.1. Explicit origins,
# never "*". The Tauri dev server (default :1420) + the packaged webview origin.
_ALLOWED_ORIGINS = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "tauri://localhost",
]


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
    controller: ChatController, session: Session, text: str
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
            for event in controller.handle_message(session, text):
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


def _settings_view() -> dict[str, Any]:
    """A read-only view of the locked + env-toggleable knobs (Phase-8 settings UI is later)."""
    from doc_assistant.config import (
        CANDIDATE_K,
        CHILD_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE,
        PARENT_CHUNK_OVERLAP,
        PARENT_CHUNK_SIZE,
        SYNTHESIS_MODE,
        TOP_K,
        USE_PARENT_CHILD,
    )

    return {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "embedding_model": get_active_model_name(),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "use_parent_child": USE_PARENT_CHILD,
        "synthesis_mode": SYNTHESIS_MODE,
        "parent_chunk": [PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP],
        "child_chunk": [CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP],
        "retrieval_weights": {"bm25": 0.4, "vector": 0.6},
    }


def create_app(controller: ChatController | None = None) -> FastAPI:
    """Build the FastAPI app. ``controller`` is injected in tests (a fake) and set on
    ``app.state`` eagerly so the test client needs no lifespan; in production it is
    ``None`` and a real ``ChatController`` is built once in ``lifespan`` (lazy — so
    importing this module / ``app = create_app()`` does not load models)."""
    # FastAPI is an app entrypoint (this runs for both `uvicorn ...:app` and `python -m
    # apps.api`, which imports `app`). The env-driven LOG_JSON is the "deployed/observed
    # context" signal (ADR-003 Decision 7). Idempotent, so the test's create_app() is safe.
    configure_logging(json=LOG_JSON, level=LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "controller", None) is None:
            app.state.controller = ChatController()
        yield

    app = FastAPI(title="doc_assistant desktop API", lifespan=lifespan)
    app.state.sessions = SessionStore()
    if controller is not None:
        app.state.controller = controller
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        controller: ChatController = request.app.state.controller
        return {
            "status": "ok",
            "chunk_count": controller.chunk_count(),
            "model": f"{LLM_PROVIDER}/{LLM_MODEL}",
            "embedding_model": get_active_model_name(),
        }

    @app.post("/api/chat")
    async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
        controller: ChatController = request.app.state.controller
        sessions: SessionStore = request.app.state.sessions
        session = sessions.get_or_create(body.session_id)  # unknown id → fresh conversation
        return EventSourceResponse(_event_stream(controller, session, body.text))

    @app.post("/api/claims/{claim_id}/adjudicate")
    def adjudicate(request: Request, claim_id: str, body: AdjudicateRequest) -> dict[str, bool]:
        controller: ChatController = request.app.state.controller
        try:
            controller.adjudicate(claim_id, body.decision, edited_text=body.edited_text)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True}

    @app.post("/api/export")
    def export(request: Request, body: ExportRequest) -> FileResponse:
        controller: ChatController = request.app.state.controller
        sessions: SessionStore = request.app.state.sessions
        session = sessions.get(body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session_id")
        message, path = controller.export_conversation(session, dev=body.dev)
        if path is None:  # nothing to export yet
            raise HTTPException(status_code=400, detail=message)
        return FileResponse(str(path), media_type="text/markdown", filename=path.name)

    @app.get("/api/figures/{figure_id}")
    def figure(figure_id: str) -> FileResponse:
        path = load_figure_image_paths([figure_id]).get(figure_id)
        if not path or not Path(path).exists():
            raise HTTPException(status_code=404, detail="figure not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/source/{record_id}/{n}")
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

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return _settings_view()

    @app.post("/api/settings")
    def post_settings() -> dict[str, Any]:
        # The env-toggleable write path lands with the Phase-8 settings UI.
        raise HTTPException(status_code=501, detail="settings write not yet wired (Phase 8)")

    return app


# Module-level app for `uvicorn apps.api.main:app` (the `just api` dev recipe). The real
# ChatController is built lazily in lifespan, so importing this module stays cheap.
app = create_app()
