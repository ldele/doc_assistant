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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from apps.api.models import (
    AdjudicateRequest,
    ChatRequest,
    CompareRequest,
    CompareResultPayload,
    ConversationDetailPayload,
    ConversationMetaUpdate,
    ConversationSummaryPayload,
    ExportRequest,
    IngestRequest,
    LibraryDocumentChunksPayload,
    LibraryDocumentMetaUpdate,
    LibraryDocumentPayload,
    SettingsUpdate,
    SourceFilePayload,
    SourcePatch,
    TurnResultPayload,
)
from apps.api.sessions import SessionStore
from doc_assistant import app_settings
from doc_assistant.chat_controller import (
    ChatController,
    RagOverrides,
    Result,
    Session,
    Step,
    Token,
)
from doc_assistant.config import DATA_PATH, LOG_JSON, LOG_LEVEL
from doc_assistant.db.session import session_scope
from doc_assistant.embeddings import get_active_model_name
from doc_assistant.ingest.figures import load_figure_image_paths
from doc_assistant.logging_config import configure_logging
from doc_assistant.provenance import get_record

log = structlog.get_logger(__name__)

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
    controller: ChatController,
    session: Session,
    text: str,
    overrides: RagOverrides | None = None,
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
            for event in controller.handle_message(session, text, overrides=overrides):
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
        BM25_WEIGHT,
        CANDIDATE_K,
        CHILD_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE,
        EPISTEMICS_MARKERS_ENABLED,
        PAID_PROVIDERS,
        PARENT_CHUNK_OVERLAP,
        PARENT_CHUNK_SIZE,
        REVIEWER_EVIDENCE_CHARS,
        SYNTHESIS_MODE,
        TOP_K,
        USE_MULTI_QUERY,
        USE_PARENT_CHILD,
    )
    from doc_assistant.llm import provider_available

    # ADR-011 (U1c): the *effective* provider/model — the persisted selection if the user has
    # switched, else the config default — never the import-time LLM_PROVIDER/LLM_MODEL constants
    # (those would go stale the moment a switch happens).
    eff_provider, eff_model = app_settings.effective_llm()

    return {
        "provider": eff_provider,
        "model": eff_model,
        "embedding_model": get_active_model_name(),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "use_parent_child": USE_PARENT_CHILD,
        "synthesis_mode": SYNTHESIS_MODE,
        # The locked defaults for the RAG-sandbox toggles (ADR-010 + the U1b amendment) — the
        # sandbox section needs these to render each control's un-overridden state correctly.
        "use_multi_query": USE_MULTI_QUERY,
        "epistemics_markers_enabled": EPISTEMICS_MARKERS_ENABLED,
        "reviewer_evidence_chars": REVIEWER_EVIDENCE_CHARS,
        "parent_chunk": [PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP],
        "child_chunk": [CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP],
        # ADR-010 "fix in passing": sourced from config, not a hardcoded literal — the
        # read-only display can no longer silently drift from the real BM25_WEIGHT.
        "retrieval_weights": {"bm25": BM25_WEIGHT, "vector": round(1 - BM25_WEIGHT, 3)},
        # ADR-011 (U1c) — the provider picker's option list: availability (fork E) + paid/local
        # labeling (fork D/KI-4), so a keyless provider renders disabled with its reason.
        "providers": [
            {
                "id": p,
                "available": provider_available(p),
                "paid": p in PAID_PROVIDERS,
            }
            for p in ("anthropic", "ollama")
        ],
    }


SUPPORTED_NOTE = "pdf · epub · html · docx · md (and similar text formats)"


@dataclass
class _IngestStatus:
    """Background-ingest progress, read by GET /api/ingest/status (guarded by a lock)."""

    state: str = "idle"  # idle | running | done | error
    source_dir: str | None = None
    added: int = 0
    skipped: int = 0
    errors: int = 0
    message: str | None = None


def _default_ingest(
    *, scope: str | None = None, files: list[Path] | None = None
) -> dict[str, int]:
    """Lazy wrapper so importing this module doesn't pull the heavy ingest -> torch chain.

    ``scope`` = the whole source dir (honoring exclusions); ``files`` = an explicit, pre-resolved
    selection (selective ingestion, S1). The two are mutually exclusive at the ``main()`` layer.
    """
    from doc_assistant.ingest import main as ingest_main

    return ingest_main(scope=scope, files=files)


def _full_settings(app: FastAPI) -> dict[str, Any]:
    """Read view: the locked knobs plus the data home, the user's source folder, chunk_count."""
    controller: ChatController = app.state.controller
    source = app_settings.get_source_dir()
    return {
        **_settings_view(),
        "data_home": str(DATA_PATH),
        "source_dir": str(source),
        "source_dir_exists": source.is_dir(),
        "supported_formats": SUPPORTED_NOTE,
        "chunk_count": controller.chunk_count(),
    }


def _ingest_status_dict(app: FastAPI) -> dict[str, Any]:
    st: _IngestStatus = app.state.ingest_status
    with app.state.ingest_lock:
        return {
            "state": st.state,
            "source_dir": st.source_dir,
            "added": st.added,
            "skipped": st.skipped,
            "errors": st.errors,
            "message": st.message,
        }


def create_app(
    controller: ChatController | None = None,
    *,
    ingest_fn: Callable[..., dict[str, int]] | None = None,
    controller_factory: Callable[[], ChatController] | None = None,
) -> FastAPI:
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
    app.state.ingest_status = _IngestStatus()
    app.state.ingest_lock = threading.Lock()
    # Test seams (cpc §13): default to the real ingest + a fresh ChatController; tests inject
    # fakes so /api/ingest runs no real ingest / model reload.
    app.state.ingest_fn = ingest_fn or _default_ingest
    app.state.controller_factory = controller_factory or ChatController
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
        # ADR-011 (U1c): the effective provider/model, so /api/health reflects a live switch.
        eff_provider, eff_model = app_settings.effective_llm()
        return {
            "status": "ok",
            "chunk_count": controller.chunk_count(),
            "model": f"{eff_provider}/{eff_model}",
            "embedding_model": get_active_model_name(),
        }

    @app.post("/api/chat")
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
        return EventSourceResponse(_event_stream(controller, session, body.text, overrides))

    @app.post("/api/compare")
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
        return CompareResultPayload.from_result(controller.compare_retrieval(body.text, overrides))

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
        # get_or_create (not get): a past/reopened conversation's session_id isn't in the
        # in-memory store, but export sources from the durable transcript by id. An id with no
        # persisted turns falls through to the "nothing to export" 400 below.
        session = sessions.get_or_create(body.session_id)
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

    @app.get("/api/conversations")
    def list_conversations_route() -> list[ConversationSummaryPayload]:
        """Past conversations for the history sidebar (feature-conversation-history.md).

        A read over the ``AnswerRecord`` store — the *live* session appears here too once it has
        a persisted turn (the frontend marks it as current). Rows predating the ``session_id``
        write-fix are ``NULL`` and excluded."""
        from doc_assistant.conversations import list_conversations

        return [ConversationSummaryPayload.from_summary(s) for s in list_conversations()]

    @app.get("/api/conversations/{session_id}")
    def get_conversation_route(session_id: str) -> ConversationDetailPayload:
        """Rehydrate one conversation as a read-only transcript, or 404 if unknown."""
        from doc_assistant.conversations import get_conversation

        detail = get_conversation(session_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return ConversationDetailPayload.from_detail(detail)

    @app.patch("/api/conversations/{session_id}")
    def update_conversation_route(
        session_id: str, body: ConversationMetaUpdate
    ) -> dict[str, bool]:
        """Set a conversation's management flags (pin / archive / soft-delete). Only the fields
        present in the body change; others are left as-is. Idempotent per field."""
        from doc_assistant.conversations import set_conversation_meta

        set_conversation_meta(
            session_id,
            pinned=body.pinned,
            archived=body.archived,
            deleted=body.deleted,
            title=body.title,
        )
        return {"ok": True}

    @app.get("/api/library/documents")
    def list_library_documents() -> list[LibraryDocumentPayload]:
        """Every ingested (non-archived) document for the Library browser (read-only, no model).

        A read over the SQLite ``Document`` store — feature-library-browser.md (L1)."""
        from doc_assistant.library import list_documents

        return [LibraryDocumentPayload.from_summary(s) for s in list_documents()]

    @app.get("/api/library/documents/{doc_id}")
    def get_library_document(request: Request, doc_id: str) -> LibraryDocumentChunksPayload:
        """One document's chunks grouped into parent blocks, or 404 if the document is unknown.

        Reads the live Chroma handle (``ChatController.rag.db``) via a metadata filter — no
        embeddings, no generation. A known doc with no stored chunks returns empty parents (not
        a 404)."""
        from doc_assistant.library import get_document_chunks

        controller: ChatController = request.app.state.controller
        view = get_document_chunks(doc_id, controller.rag.db)
        if view is None:
            raise HTTPException(status_code=404, detail="document not found")
        return LibraryDocumentChunksPayload.from_view(view)

    @app.patch("/api/library/documents/{doc_id}")
    def patch_library_document(doc_id: str, body: LibraryDocumentMetaUpdate) -> dict[str, bool]:
        """Set a document's user metadata overrides (title/authors/year). ADR-013 — first
        browse-time write path. 404 if the document is unknown; effective values are
        override ?? auto-extracted default."""
        from doc_assistant.library import get_document_details, set_document_meta

        if get_document_details(doc_id) is None:
            raise HTTPException(status_code=404, detail="document not found")
        set_document_meta(doc_id, title=body.title, authors=body.authors, year=body.year)
        return {"ok": True}

    @app.post("/api/library/documents/{doc_id}/reset-metadata")
    def reset_library_document_metadata(doc_id: str) -> dict[str, bool]:
        """Reset a document to its auto-extracted metadata (delete the override row). ADR-013."""
        from doc_assistant.library import clear_document_meta, get_document_details

        if get_document_details(doc_id) is None:
            raise HTTPException(status_code=404, detail="document not found")
        clear_document_meta(doc_id)
        return {"ok": True}

    @app.post("/api/library/documents/{doc_id}/reveal")
    def reveal_library_document(doc_id: str) -> dict[str, bool]:
        """Reveal the source file in the OS file manager (local desktop action, ADR-013).
        404 if the document is unknown or its source file can't be located (moved/deleted)."""
        from doc_assistant.library import reveal_document_source

        if not reveal_document_source(doc_id):
            raise HTTPException(status_code=404, detail="source file not found")
        return {"ok": True}

    @app.get("/api/settings")
    def get_settings(request: Request) -> dict[str, Any]:
        return _full_settings(request.app)

    @app.post("/api/settings")
    def post_settings(request: Request, body: SettingsUpdate) -> dict[str, Any]:
        controller: ChatController = request.app.state.controller
        # "Point at a folder": set the source documents dir (validated + persisted to the data
        # home). The data *home* (index/DB) stays managed by config; the user only chooses where
        # their documents live. Re-index via POST /api/ingest to load the new folder.
        if body.source_dir is not None:
            try:
                app_settings.set_source_dir(body.source_dir)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        # ADR-011 (U1c): switch the live provider/model. SettingsUpdate's own validator already
        # guarantees these travel together, but mypy can't see that invariant across fields, so
        # both are checked here too.
        if body.llm_provider is not None and body.llm_model is not None:
            try:
                controller.reconfigure(body.llm_provider, body.llm_model)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        return _full_settings(request.app)

    @app.post("/api/ingest", status_code=202)
    def ingest_start(request: Request, body: IngestRequest | None = None) -> dict[str, Any]:
        app_ = request.app
        source = app_settings.get_source_dir()

        # Selective ingestion (S1): an explicit `paths` selection is resolved + validated up front
        # so a bad path is a 400 *before* anything starts running (nothing partial). No body (or
        # null paths) keeps today's behavior — the whole source dir, now minus standing exclusions.
        selection: list[Path] | None = None
        if body is not None and body.paths is not None:
            from doc_assistant.ingest import registry

            with session_scope() as session:
                try:
                    selection = registry.resolve_selection(session, source, body.paths)
                except registry.InvalidSelection as e:
                    raise HTTPException(
                        status_code=400,
                        detail={"error": "invalid selection", "offenders": e.offenders},
                    ) from e

        status: _IngestStatus = app_.state.ingest_status
        with app_.state.ingest_lock:
            if status.state == "running":
                raise HTTPException(status_code=409, detail="ingest already running")
            status.state = "running"
            status.source_dir = str(source)
            status.added = status.skipped = status.errors = 0
            status.message = None

        def _worker() -> None:
            try:
                if selection is not None:
                    stats = app_.state.ingest_fn(files=selection)
                else:
                    stats = app_.state.ingest_fn(scope=str(source))
            except Exception as e:  # surface any ingest failure to the status view
                log.exception("ingest_failed", source=str(source))
                with app_.state.ingest_lock:
                    status.state = "error"
                    status.message = str(e)
                return
            # Reload the controller so the new corpus is live BEFORE reporting "done" — a client
            # that sees "done" then reads chunk_count must get the updated count (BM25 + Chroma
            # handles are built at construction). A reload failure is logged but doesn't fail the
            # ingest: the documents are persisted, a restart picks them up, so a successful ingest
            # never flips to "error".
            try:
                app_.state.controller = app_.state.controller_factory()
            except Exception:
                log.exception("controller_reload_failed_after_ingest")
            with app_.state.ingest_lock:
                status.added = int(stats.get("added", 0))
                status.skipped = int(stats.get("skipped", 0))
                status.errors = int(stats.get("error", 0))
                status.message = (
                    f"indexed {status.added} new, {status.skipped} unchanged, "
                    f"{status.errors} errors"
                )
                status.state = "done"

        threading.Thread(target=_worker, name="ingest", daemon=True).start()
        return _ingest_status_dict(app_)

    @app.get("/api/ingest/status")
    def ingest_status(request: Request) -> dict[str, Any]:
        return _ingest_status_dict(request.app)

    @app.get("/api/sources")
    def list_sources(request: Request) -> list[SourceFilePayload]:
        """Selective ingestion (S1): stat-only scan of the source dir → each file with a
        derived ingest status (new / changed / ingested / missing) + its ``excluded`` flag."""
        from doc_assistant.ingest import registry

        source = app_settings.get_source_dir()
        with session_scope() as session:
            return [SourceFilePayload.from_view(v) for v in registry.scan_sources(session, source)]

    @app.patch("/api/sources")
    def patch_source(body: SourcePatch, request: Request) -> SourceFilePayload:
        """Update one registry row's intent (v1: ``excluded``). 404 if rel_path is unknown."""
        from doc_assistant.ingest import registry

        source = app_settings.get_source_dir()
        with session_scope() as session:
            try:
                registry.set_source_meta(session, body.rel_path, excluded=body.excluded)
            except KeyError as e:
                raise HTTPException(
                    status_code=404, detail=f"unknown source: {body.rel_path}"
                ) from e
            view = registry.view_for(session, source, body.rel_path)
            if view is None:  # unreachable (set_source_meta would have raised) — narrows for mypy
                raise HTTPException(status_code=404, detail=f"unknown source: {body.rel_path}")
            return SourceFilePayload.from_view(view)

    return app


# Module-level app for `uvicorn apps.api.main:app` (the `just api` dev recipe). The real
# ChatController is built lazily in lifespan, so importing this module stays cheap.
app = create_app()
