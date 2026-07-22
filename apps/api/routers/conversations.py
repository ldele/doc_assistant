"""Conversations router — the history sidebar (read + management flags)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.models import (
    ConversationDetailPayload,
    ConversationMetaUpdate,
    ConversationSummaryPayload,
)

router = APIRouter()


@router.get("/api/conversations")
def list_conversations_route() -> list[ConversationSummaryPayload]:
    """Past conversations for the history sidebar (feature-conversation-history.md).

    A read over the ``AnswerRecord`` store — the *live* session appears here too once it has
    a persisted turn (the frontend marks it as current). Rows predating the ``session_id``
    write-fix are ``NULL`` and excluded."""
    from doc_assistant.conversations import list_conversations

    return [ConversationSummaryPayload.from_summary(s) for s in list_conversations()]


@router.get("/api/conversations/{session_id}")
def get_conversation_route(session_id: str) -> ConversationDetailPayload:
    """Rehydrate one conversation as a read-only transcript, or 404 if unknown."""
    from doc_assistant.conversations import get_conversation

    detail = get_conversation(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationDetailPayload.from_detail(detail)


@router.patch("/api/conversations/{session_id}")
def update_conversation_route(session_id: str, body: ConversationMetaUpdate) -> dict[str, bool]:
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
