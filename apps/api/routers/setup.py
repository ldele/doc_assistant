"""Setup router — first-run readiness + the in-app API key (ADR-034).

Separate from ``routers/settings`` on purpose: this is the only place key material is accepted,
so the surface that touches a secret is one small module with no other job. Nothing here returns
a key; the read model carries a source label and a last-4 hint (``apps/api/models/setup.py``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.models.setup import ApiKeyUpdate
from apps.api.services import log, setup_state_dict
from doc_assistant import credentials, llm

router = APIRouter()

_PROVIDER = "anthropic"  # the one keyed provider today (credentials.keyed_providers())


@router.get("/api/setup")
def get_setup(request: Request, probe: bool = True) -> dict[str, Any]:
    """The first-run picture. ``?probe=false`` answers from local state only (no Ollama call)."""
    return setup_state_dict(request.app, probe=probe)


@router.post("/api/setup/anthropic-key")
def post_anthropic_key(request: Request, body: ApiKeyUpdate) -> dict[str, Any]:
    """Verify then store an Anthropic API key, and rebuild the live chat model.

    * **Rejected key → 400.** Storing one the API refuses would leave a broken install looking
      configured (inform-don't-corrupt, the same posture as ``set_source_dir``).
    * **Unverifiable key → stored, with the reason.** No network / a proxy / a timeout is not
      evidence the key is bad, and discarding what the user typed would be the worse failure
      (inform-don't-block).

    Verification is a free metadata call (``models.list``), never a completion, so first-run setup
    cannot bill the user (KI-4's discipline on the setup path).
    """
    status, detail = llm.verify_anthropic_key(body.key)
    if status == "invalid":
        raise HTTPException(status_code=400, detail=detail)
    try:
        credentials.set_stored_key(_PROVIDER, body.key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _refresh_chat_model(request)
    log.info("anthropic_key_saved", verification=status)
    return {
        "stored": True,
        "verification": status,
        "detail": detail,
        "setup": setup_state_dict(request.app),
    }


@router.delete("/api/setup/anthropic-key")
def delete_anthropic_key(request: Request) -> dict[str, Any]:
    """Forget the key saved in the app. A key in ``.env`` is untouched — this route owns only the
    app's own store, and pretending to remove a file the user manages would be a lying UI."""
    removed = credentials.clear_stored_key(_PROVIDER)
    _refresh_chat_model(request)
    return {"removed": removed, "setup": setup_state_dict(request.app)}


def _refresh_chat_model(request: Request) -> None:
    """Rebuild the generation model so a key change reaches the next turn without a restart.

    Best-effort by design: if the model cannot be built with the new credential state (e.g. the
    key was just cleared while Anthropic is the active provider), the *setup state* the caller
    gets back already reports that honestly — failing this request instead would leave the user
    unable to remove a key.
    """
    try:
        request.app.state.controller.refresh_chat_model()
    except Exception as e:
        log.warning("chat_model_refresh_failed", error=str(e))
