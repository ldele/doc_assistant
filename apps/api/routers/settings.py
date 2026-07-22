"""Settings router — user-settable runtime settings (source dir, provider switch, epistemics)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.models import SettingsUpdate
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
