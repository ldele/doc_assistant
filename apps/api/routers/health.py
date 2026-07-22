"""Health router — liveness + the live model identity."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from doc_assistant import app_settings
from doc_assistant.chat_controller import ChatController
from doc_assistant.embeddings import get_active_model_name

router = APIRouter()


@router.get("/api/health")
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
