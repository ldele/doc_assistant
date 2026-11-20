"""Conversations router — the history sidebar (read, management flags, bulk cleanup).

Route-order note: the literal paths (``/export``, ``/bulk``) are declared **before**
``/{session_id}`` so a conversation can never be named out of existence by one of them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from apps.api.models.chat import ScopePayload
from apps.api.models.conversations import (
    ChatSetupPayload,
    ContinuePayload,
    ContinueRequest,
    ConversationBulkResult,
    ConversationBulkUpdate,
    ConversationDetailPayload,
    ConversationMetaUpdate,
    ConversationSummaryPayload,
    ConversationTurnPayload,
    ResumePayload,
)
from apps.api.sessions import SessionStore

router = APIRouter()


@router.get("/api/conversations")
def list_conversations_route(
    document_id: str | None = None, folder_id: str | None = None
) -> list[ConversationSummaryPayload]:
    """Past conversations for the history sidebar (feature-conversation-history.md).

    A read over the ``AnswerRecord`` store — the *live* session appears here too once it has
    a persisted turn (the frontend marks it as current). Rows predating the ``session_id``
    write-fix are ``NULL`` and excluded.

    Each row carries its labels; ``document_id`` / ``folder_id`` keep the conversations that
    cited that document, or a document in that folder, both together = both must hold (A7b,
    G4). The derivation is the read surface's; the controls that use it are A8's."""
    from doc_assistant.conversations import list_conversations

    return [
        ConversationSummaryPayload.from_summary(s)
        for s in list_conversations(document_id=document_id, folder_id=folder_id)
    ]


@router.post("/api/conversations/export")
def export_history_route() -> FileResponse:
    """The whole chat history as one markdown file.

    **Uncapped, unlike the list** (which stops at ~100 by design): an export that silently omits
    a conversation is worse than none, because the omission is invisible exactly when the file is
    being relied on. Soft-deleted conversations are excluded — the export mirrors what the sidebar
    shows.

    A 200 with an empty-state document rather than a 400 for "no conversations": a first-run user
    asking for a backup should get a file that says there is nothing yet, not an error."""
    from doc_assistant.conversations import export_all_conversations

    result = export_all_conversations()
    # A temp file, not EXPORT_DIR: this is a download, and writing every "export history" click
    # into the user's export folder would litter it with near-duplicates they never asked to keep.
    tmp = Path(tempfile.gettempdir()) / "provenote-chat-history.md"
    tmp.write_text(result.markdown, encoding="utf-8")
    return FileResponse(str(tmp), media_type="text/markdown", filename=tmp.name)


@router.post("/api/conversations/bulk")
def bulk_update_conversations_route(body: ConversationBulkUpdate) -> ConversationBulkResult:
    """Soft-delete (or restore) many conversations in one transaction.

    Same per-row semantics as the single PATCH — ``deleted_at`` stamped, ``AnswerRecord``
    provenance retained — so this is undone by the same route with ``deleted=false``."""
    from doc_assistant.conversations import set_conversations_deleted

    return ConversationBulkResult(
        updated=set_conversations_deleted(body.session_ids, deleted=body.deleted)
    )


@router.get("/api/conversations/scope")
def scope_preview_route(
    folder_ids: list[str] | None = Query(default=None),  # noqa: B008 - FastAPI's idiom
) -> ScopePayload | None:
    """The folders resolved now, as a turn would resolve them (A8a): each folder's count and
    the union, each document once. What the start screen's *Search in* count shows; null = the
    whole library (no folder given). An unknown folder counts 0 and is named, never widened."""
    from doc_assistant.chat_setup import current_scope

    view = current_scope(list(folder_ids) if folder_ids else None)
    return ScopePayload.from_view(view) if view is not None else None


@router.get("/api/conversations/{session_id}")
def get_conversation_route(session_id: str) -> ConversationDetailPayload:
    """Rehydrate one conversation as a read-only transcript, or 404 if unknown."""
    from doc_assistant.chat_setup import describe_setup
    from doc_assistant.conversations import get_conversation

    detail = get_conversation(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationDetailPayload.from_detail(detail, describe_setup(detail.setup))


@router.get("/api/conversations/{session_id}/setup")
def get_setup_route(session_id: str) -> ChatSetupPayload:
    """A chat's setup (A8a): stored, worked out from its last record for a chat from before
    A8, or the default for a chat with no turn yet. Reading writes nothing."""
    from doc_assistant.chat_setup import describe_setup, read_setup

    return ChatSetupPayload.from_view(describe_setup(read_setup(session_id)))


@router.post("/api/conversations/{session_id}/continue")
def continue_route(request: Request, session_id: str, body: ContinueRequest) -> ContinuePayload:
    """*Continue in a new chat* (A8a): set up ``body.session_id`` as continued from this chat,
    with its own mode and folders, carrying this chat's exchanges into the new chat's memory
    when ``carry`` (G3). This chat is not changed. A mode id no mode has is a 404."""
    from doc_assistant.chat_modes import get_mode
    from doc_assistant.chat_setup import continue_chat, describe_setup
    from doc_assistant.conversations import get_conversation

    if body.session_id == session_id:
        raise HTTPException(status_code=400, detail="a chat cannot continue into itself")
    if body.mode_id and get_mode(body.mode_id) is None:
        raise HTTPException(status_code=404, detail="No mode has this id.")
    continued = continue_chat(
        session_id,
        body.session_id,
        mode_id=body.mode_id,
        folder_ids=body.folder_ids,
        carry=body.carry,
    )
    sessions: SessionStore = request.app.state.sessions
    sessions.preload(body.session_id, continued.carried.messages)
    old = get_conversation(session_id)
    carried_ids = set(continued.carried.record_ids)
    return ContinuePayload(
        session_id=body.session_id,
        setup=ChatSetupPayload.from_view(describe_setup(continued.setup)),
        carried=[
            ConversationTurnPayload.from_turn(t)
            for t in (old.turns if old is not None else [])
            if t.record_id in carried_ids
        ],
    )


@router.post("/api/conversations/{session_id}/resume")
def resume_route(request: Request, session_id: str) -> ResumePayload:
    """*Continue this chat* (A8a, open question 1, superseding the resume spec's decision 1):
    the chat's memory starts with its own exchanges, read from its records and fitted by A5
    like any earlier message. A chat already live in this process keeps the memory it has."""
    from doc_assistant.chat_setup import carried_exchanges

    sessions: SessionStore = request.app.state.sessions
    live = sessions.get(session_id)
    if live is not None and live.history:
        return ResumePayload(session_id=session_id, carried_exchanges=live.carried_messages // 2)
    carried = carried_exchanges(session_id)
    sessions.preload(session_id, carried.messages)
    return ResumePayload(session_id=session_id, carried_exchanges=carried.exchanges)


@router.patch("/api/conversations/{session_id}")
def update_conversation_route(session_id: str, body: ConversationMetaUpdate) -> dict[str, bool]:
    """Set a conversation's management flags (pin / archive / soft-delete) and, on a chat with no
    turn yet, its chat mode (A7; ``""`` is the default). Only the fields present in the body
    change; others are left as-is. Idempotent per field. An unknown mode id is a 404, and a mode
    for a chat that has turns a 409 (A8a, G2): both change nothing."""
    from doc_assistant.chat_modes import ModeNotFound, get_mode, set_conversation_mode
    from doc_assistant.chat_setup import chat_is_set
    from doc_assistant.conversations import set_conversation_meta

    if body.mode_id is not None:
        if body.mode_id and get_mode(body.mode_id) is None:
            raise HTTPException(status_code=404, detail="No mode has this id.")
        if chat_is_set(session_id):
            # A8a (G2): an open chat never changes mode; a new chat continued from it does.
            raise HTTPException(
                status_code=409,
                detail="This chat keeps its mode. Continue in a new chat to answer in another.",
            )
        try:
            set_conversation_mode(session_id, body.mode_id or None)
        except ModeNotFound as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
    set_conversation_meta(
        session_id,
        pinned=body.pinned,
        archived=body.archived,
        deleted=body.deleted,
        title=body.title,
    )
    return {"ok": True}
