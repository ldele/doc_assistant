"""Setup router — first-run readiness, the in-app API keys and servers of your own (ADR-034,
ADR-060).

Separate from ``routers/settings`` on purpose: this is the only place key material is accepted,
so the surface that touches a secret is one small module with no other job. Nothing here returns
a key; the read model carries a source label and a last-4 hint (``apps/api/models/setup.py``).

Every judgement — what a provider is, where its text goes, who bills, what a server's answer
means — is ``doc_assistant.providers``/``readiness``; these handlers map it to HTTP.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.models.setup import ApiKeyUpdate, BillingUpdate, ServerUpdate
from apps.api.services import log, setup_state_dict
from doc_assistant import app_settings, credentials, llm, providers

router = APIRouter()


@router.get("/api/setup")
def get_setup(request: Request, probe: bool = True) -> dict[str, Any]:
    """The first-run picture. ``?probe=false`` answers from local state only (no network)."""
    return setup_state_dict(request.app, probe=probe)


def _known(provider_id: str) -> str:
    key = provider_id.strip().lower()
    if providers.kind_of(key) is None:
        raise HTTPException(status_code=404, detail=f"No provider '{provider_id}'.")
    return key


@router.post("/api/setup/providers/{provider_id}/models")
def fetch_models(provider_id: str) -> dict[str, Any]:
    """Ask the provider's server for its models now (opening its form, or *Refresh*; choice 8).

    A POST, though it reads: it calls out with the stored key and changes what the setup state
    remembers, and with a launch token set (S-6) only a method other than GET needs it, so a web
    page could otherwise trigger the call.

    Free on every provider: a model list bills nothing. Kept in memory with its time, so the setup
    state can say *Key accepted at 14:02* or *Key refused* without asking again.
    """
    return asdict(providers.fetch_models(_known(provider_id)))


@router.post("/api/setup/providers/{provider_id}/key")
def post_key(request: Request, provider_id: str, body: ApiKeyUpdate) -> dict[str, Any]:
    """Verify then store a key, and rebuild the live chat model.

    * **Refused key → 400.** Storing one the server refuses would leave a broken install looking
      configured (inform-don't-corrupt, the same posture as ``set_source_dir``).
    * **Unverifiable key → stored, with the reason.** No network / a proxy / a timeout is not
      evidence the key is bad, and discarding what the user typed would be the worse failure
      (inform-don't-block).

    Verification is a free call: Claude's ``models.list``, or a compatible server's model list (and
    OpenRouter's ``/key``), never a completion, so setup cannot bill the user (KI-4). A compatible
    key is bound to its provider's address and sent nowhere else.
    """
    key = _known(provider_id)
    kind = providers.kind_of(key)
    if kind == "ollama":
        raise HTTPException(status_code=400, detail="Ollama takes no key.")
    if kind == "claude":
        status, detail = llm.verify_anthropic_key(body.key)
        if status == "invalid":
            raise HTTPException(status_code=400, detail=detail)
        try:
            credentials.set_stored_key("anthropic", body.key)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        from doc_assistant import compatible

        point = providers.endpoint(key)
        listing = compatible.list_models(
            point.address, body.key.strip(), key_check=point.key_check
        )
        if listing.status == "refused":
            raise HTTPException(status_code=400, detail=f"{point.name} refused this key.")
        try:
            credentials.set_bound_key(key, body.key, point.address)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        providers.remember_list(key, body.key.strip(), listing)
        host = providers.host_of(point.address) or point.address
        status = "ok" if listing.status == "ok" else "unreachable"
        detail = (
            "Key accepted."
            if listing.status == "ok"
            else f"Could not reach {host} to check the key. Saved. {listing.detail}".strip()
        )
    _refresh_chat_model(request)
    log.info("api_key_saved", provider=key, verification=status)
    return {
        "stored": True,
        "verification": status,
        "detail": detail,
        "setup": setup_state_dict(request.app),
    }


@router.delete("/api/setup/providers/{provider_id}/key")
def delete_key(request: Request, provider_id: str) -> dict[str, Any]:
    """Forget the key saved in the app. A Claude key in ``.env`` is untouched — this route owns
    only the app's own store, and pretending to remove a file the user manages would be a lying
    UI."""
    key = _known(provider_id)
    if key == "anthropic":
        removed = credentials.clear_stored_key("anthropic")
    else:
        removed = credentials.clear_bound_key(key)
    providers.forget_list(key)
    _refresh_chat_model(request)
    return {"removed": removed, "setup": setup_state_dict(request.app)}


@router.post("/api/setup/servers")
def post_server(request: Request, body: ServerUpdate) -> dict[str, Any]:
    """*Test connection* on a server of your own: save it, then ask it for its models.

    ADR-034's three outcomes: a refused key → 400 and nothing stored (a new server is not added; a
    changed one keeps its old record); unreachable → saved, with the reason; reached → saved. A
    changed address drops the old key, because a key is sent only to the address it was saved
    with.
    """
    from doc_assistant import compatible

    try:
        address = providers.check_address(body.address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    existing = providers.server(body.id) if body.id else None
    if body.id and existing is None:
        raise HTTPException(status_code=404, detail=f"No server '{body.id}'.")
    typed = body.key.strip() if body.key and body.key.strip() else None
    kept = (
        credentials.get_bound_key(existing.id, address)
        if existing is not None and typed is None
        else None
    )
    key = typed or kept
    listing = compatible.list_models(address, key)
    if listing.status == "refused" and typed is not None:
        raise HTTPException(status_code=400, detail="The server refused this key. Nothing saved.")
    try:
        saved = providers.save_server(body.name, address, body.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if typed is not None:
        credentials.set_bound_key(saved.id, typed, saved.address)
    providers.remember_list(saved.id, key, listing)
    _refresh_chat_model(request)
    log.info("server_saved", server=saved.id, status=listing.status)
    return {"id": saved.id, "listing": asdict(listing), "setup": setup_state_dict(request.app)}


@router.delete("/api/setup/servers/{server_id}")
def delete_server(request: Request, server_id: str) -> dict[str, Any]:
    """Forget a server of your own, with its key and its tick. The one answering now is kept:
    removing it would leave the next question with no engine."""
    key = server_id.strip().lower()
    if app_settings.effective_llm()[0] == key:
        raise HTTPException(
            status_code=409,
            detail="This server answers your questions now. Choose another provider first.",
        )
    if not providers.remove_server(key):
        raise HTTPException(status_code=404, detail=f"No server '{server_id}'.")
    return {"removed": True, "setup": setup_state_dict(request.app)}


@router.post("/api/setup/providers/{provider_id}/billing")
def post_billing(request: Request, provider_id: str, body: BillingUpdate) -> dict[str, Any]:
    """Record the user's word that a server bills nobody (*This server does not bill me*).

    Only where Provenote cannot know: a server of your own, or an Ollama on another machine. A
    preset's billing is its vendor's, and Claude's is Anthropic's.
    """
    key = _known(provider_id)
    if not providers.can_tick(key):
        raise HTTPException(
            status_code=400, detail=f"Who bills {providers.display_name(key)} is not yours to set."
        )
    app_settings.set_not_billed(key, body.not_billed)
    return {"setup": setup_state_dict(request.app)}


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
