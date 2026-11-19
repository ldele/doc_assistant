"""Chat-modes router (A7, `docs/specs/chat-modes.md`): list the modes, and create, change and
delete a person's own. A conversation's mode is set on the conversations router's PATCH. All logic
is `doc_assistant.chat_modes`'s; this maps its errors to HTTP (a format error 400, an unknown id
404, a built-in mode 409)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.models.modes import (
    ChatModeCreate,
    ChatModeDeleted,
    ChatModePayload,
    ChatModeSaved,
    ChatModeUpdate,
)

router = APIRouter()


def _http_error(e: Exception, mode_id: str | None = None) -> HTTPException:
    from doc_assistant.chat_modes import BUILT_IN, ModeNotFound

    if isinstance(e, ModeNotFound):
        return HTTPException(status_code=404, detail=str(e))
    if mode_id is not None and any(m.id == mode_id for m in BUILT_IN):
        return HTTPException(status_code=409, detail=str(e))
    return HTTPException(status_code=400, detail=str(e))


@router.get("/api/modes")
def list_modes_route() -> list[ChatModePayload]:
    """Every chat mode: the built-in ones first, then a person's own by name."""
    from doc_assistant.chat_modes import list_modes

    return [ChatModePayload.from_mode(m) for m in list_modes()]


@router.post("/api/modes")
def create_mode_route(body: ChatModeCreate) -> ChatModeSaved:
    """Save a person's own mode. Flagged text is saved with warnings; a format error is a 400."""
    from doc_assistant.chat_modes import ModeError, create_mode

    try:
        return ChatModeSaved.from_saved(create_mode(body.name, body.framing))
    except ModeError as e:
        raise _http_error(e) from e


@router.patch("/api/modes/{mode_id}")
def update_mode_route(mode_id: str, body: ChatModeUpdate) -> ChatModeSaved:
    """Rename or edit a person's own mode; a built-in mode answers 409."""
    from doc_assistant.chat_modes import ModeError, update_mode

    try:
        return ChatModeSaved.from_saved(update_mode(mode_id, name=body.name, framing=body.framing))
    except ModeError as e:
        raise _http_error(e, mode_id) from e


@router.delete("/api/modes/{mode_id}")
def delete_mode_route(mode_id: str) -> ChatModeDeleted:
    """Delete a person's own mode. The conversations using it go back to the default from their
    next turn, and the answer says how many."""
    from doc_assistant.chat_modes import ModeError, delete_mode, deleted_mode_note, get_mode

    mode = get_mode(mode_id)
    try:
        count = delete_mode(mode_id)
    except ModeError as e:
        raise _http_error(e, mode_id) from e
    name = mode.name if mode is not None else mode_id
    return ChatModeDeleted(conversations=count, message=deleted_mode_note(name, count))
