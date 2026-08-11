"""Conversations router — the history sidebar (read, management flags, bulk cleanup).

Route-order note: the literal paths (``/export``, ``/bulk``) are declared **before**
``/{session_id}`` so a conversation can never be named out of existence by one of them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from apps.api.models.conversations import (
    ConversationBulkResult,
    ConversationBulkUpdate,
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
