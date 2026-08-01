"""Settings router — user-settable runtime settings (source dir, provider switch, epistemics)."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.models.settings import SettingsUpdate
from apps.api.services import _full_settings
from doc_assistant import app_settings
from doc_assistant.chat_controller import ChatController

router = APIRouter()


@router.get("/api/settings")
def get_settings(request: Request) -> dict[str, Any]:
    return _full_settings(request.app)


@router.post("/api/settings")
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
    # ADR-027 D2 (E3): persist the answer-layer epistemics default. A plain bool — no
    # validation to fail; applies from the next turn (_resolve_turn_knobs re-reads it).
    if body.epistemics_markers_enabled is not None:
        app_settings.set_markers_enabled(body.epistemics_markers_enabled)
    return _full_settings(request.app)


@router.post("/api/settings/reindex-keywords")
def reindex_keywords(request: Request) -> dict[str, Any]:
    """Rebuild the on-disk keyword index and swap it into the live pipeline (ADR-037).

    **Not destructive, and that is why it needs no confirmation:** the index is derived data that
    the next launch would rebuild anyway once the corpus fingerprint moves. The button exists to
    save that restart after an ingest.

    Synchronous on purpose. It is 2.8 s on a 33k-chunk corpus and minutes at the 10k-document
    contract — bounded work with a definite end, unlike a full re-ingest, so a job runner with
    polling would be machinery for nothing. If it ever stops being bounded, it becomes a `202`
    + status route like the graph rebuild, not a longer spinner.

    **Since ADR-038 this is also the recovery action**, not only a convenience. With the legacy
    in-RAM arm retired, a failed index build means keyword matching is off, and this route is what
    turns it back on — so it deliberately runs when no index is live. The only 409 left is an empty
    corpus: there is nothing to index, and reporting success for work that could not happen is the
    failure this route exists to avoid.
    """
    controller: ChatController = request.app.state.controller
    try:
        chunks = controller.rag.rebuild_sparse_index()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except (OSError, sqlite3.Error) as e:
        raise HTTPException(status_code=500, detail=f"index rebuild failed: {e}") from e
    return {"chunks": chunks, **_full_settings(request.app)}
