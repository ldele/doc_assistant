"""FastAPI desktop backend — a thin HTTP/SSE renderer over ``ChatController`` (PR-M2).

FastAPI is *just another shell* (like ``apps/cli.py``): it maps ``TurnEvent`` → HTTP/SSE and HTTP
requests → controller calls, and owns no retrieval/provenance/claim logic. One ``ChatController``
is built per process in the ``lifespan`` (model load is expensive); the test path injects a fake
controller via ``create_app(controller=...)`` so no real pipeline / LLM / network runs (cpc §13).

**Layout (APIRouter split).** The routes live in ``apps/api/routers/*`` — one module per domain
(``health`` · ``chat`` · ``conversations`` · ``library`` · ``concepts`` · ``settings`` ·
``sources``). Cross-router glue — the ``app.state`` status dataclasses, their ``202 + poll``
serializers, the settings read view, and the lazy default job runners — lives in
``apps/api/services``. This module owns only ``create_app``: the lifespan (schema migration +
controller construction), the ``app.state`` wiring + test seams, CORS, and the router mounts.
``_settings_view`` / ``_default_rebuild_graph`` are re-exported below because tests import them by
name from here.

Streaming: ``/api/chat`` returns ``text/event-stream`` — each ``TurnEvent`` becomes one SSE event
(``token`` / ``step`` / terminal ``result`` / ``done``). ``handle_message`` is a **sync, blocking**
generator; ``chat._event_stream`` bridges it to the event loop through a worker thread + queue so
tokens flush as they are produced (M2 ADR-2). A multi-client server would offload differently —
noted there, not needed for the desktop target.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import (
    chat,
    concepts,
    conversations,
    health,
    library,
    settings,
    sources,
    taxonomy,
)
from apps.api.services import (
    _default_ingest,
    _default_rebuild_graph,
    _GraphRebuildStatus,
    _IngestStatus,
    _settings_view,
)
from apps.api.sessions import SessionStore
from doc_assistant.chat_controller import ChatController
from doc_assistant.config import LOG_JSON, LOG_LEVEL
from doc_assistant.db.migrations import init_db
from doc_assistant.logging_config import configure_logging

# ``_settings_view`` / ``_default_rebuild_graph`` are re-exported (listed in ``__all__``) so
# ``from apps.api.main import _settings_view`` / ``_default_rebuild_graph`` keeps working for the
# two tests that import them by name from here — the public seam is ``create_app``; these are the
# only internals a test reaches for. ``_default_rebuild_graph`` also serves as the rebuild default.
# ``init_db`` is imported so the startup-migration test's
# ``monkeypatch.setattr("apps.api.main.init_db", …)`` target is preserved.
__all__ = ["_default_rebuild_graph", "_settings_view", "app", "create_app"]

log = structlog.get_logger(__name__)

# Local app only: the frontend speaks to the sidecar over 127.0.0.1. Explicit origins,
# never "*". The Tauri dev server (default :1420) + the packaged webview origin.
_ALLOWED_ORIGINS = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "tauri://localhost",
]


def create_app(
    controller: ChatController | None = None,
    *,
    ingest_fn: Callable[..., dict[str, int]] | None = None,
    controller_factory: Callable[[], ChatController] | None = None,
    rebuild_graph_fn: Callable[[], str] | None = None,
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
        # KI-23: bring the live schema up to date before serving. `init_db` is idempotent and
        # purely additive (create_all + ALTER ... ADD COLUMN), and until now it ran ONLY from
        # `ingest` — so a user who pulled an update and just chatted never got new columns.
        # That was survivable while additive columns fed sidecars; ADR-025 F2 put one on the
        # answer path (`answer_records.retrieval_scope_json`), where a stale schema breaks every
        # turn. Logged loudly when it actually changes something, so a silent drift like
        # `concepts.graph_include` (missing here for ~2 weeks) can't repeat unnoticed.
        # E0.5a: a FAILED migration fails the boot, rather than serving a half-migrated schema.
        # KI-23 moved init_db here precisely because a stale answer-path column (e.g.
        # `answer_records.retrieval_scope_json`) breaks *every* turn at runtime — so swallowing the
        # error only defers a worse, more confusing failure to the first chat. Refuse to start
        # with a clear message instead. (Deliberately reverses the earlier "never let a migration
        # problem stop the app" stance — an unreachable-at-boot DB is not something to paper over.)
        try:
            added = init_db()
        except Exception as e:
            log.error("schema_migration_failed", error=str(e))
            raise RuntimeError(
                "database migration failed at startup; refusing to serve a stale schema (every "
                "turn would 500). Fix the DB or run `python -m doc_assistant.db.migrations`."
            ) from e
        if added:
            log.warning("schema_migrated_at_startup", columns=added)
        else:
            log.info("schema_current")
        if getattr(app.state, "controller", None) is None:
            app.state.controller = ChatController()
        yield

    app = FastAPI(title="doc_assistant desktop API", lifespan=lifespan)
    app.state.sessions = SessionStore()
    app.state.ingest_status = _IngestStatus()
    app.state.ingest_lock = threading.Lock()
    app.state.graph_rebuild_status = _GraphRebuildStatus()
    app.state.graph_rebuild_lock = threading.Lock()
    # Test seams (cpc §13): default to the real ingest + a fresh ChatController; tests inject
    # fakes so /api/ingest runs no real ingest / model reload.
    app.state.ingest_fn = ingest_fn or _default_ingest
    app.state.rebuild_graph_fn = rebuild_graph_fn or _default_rebuild_graph
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

    # Domain routers (APIRouter split). Include order is preserved from the pre-split file;
    # route-matching order that matters (e.g. `/api/concepts/gaps` before the parameterised
    # `/api/concepts/{concept_id}/…`) is kept *within* each router's own declaration order.
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(conversations.router)
    app.include_router(library.router)
    app.include_router(concepts.router)
    app.include_router(taxonomy.router)
    app.include_router(settings.router)
    app.include_router(sources.router)

    return app


# Module-level app for `uvicorn apps.api.main:app` (the `just api` dev recipe). The real
# ChatController is built lazily in lifespan, so importing this module stays cheap.
app = create_app()
