"""Chat-mode wire models (A7, `docs/specs/chat-modes.md`).

A mode as the app shows it (name, framing, built in or a person's own, evaluated or not), the
bodies that create and change a person's own, and what a save or a delete answers with: the
check's warnings (a save is never refused for its meaning) and how many conversations a delete
sent back to the default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from doc_assistant.chat_modes import ChatMode, SavedMode


class ChatModePayload(BaseModel):
    id: str
    name: str
    framing: str
    built_in: bool
    # Built in: a gate ran on the eval panel. A person's own: not evaluated.
    evaluated: bool

    @classmethod
    def from_mode(cls, m: ChatMode) -> ChatModePayload:
        return cls(
            id=m.id, name=m.name, framing=m.framing, built_in=m.built_in, evaluated=m.evaluated
        )


class ChatModeCreate(BaseModel):
    name: str
    framing: str


class ChatModeUpdate(BaseModel):
    """Only the fields sent change."""

    name: str | None = None
    framing: str | None = None


class ChatModeSaved(BaseModel):
    """A saved mode and the check's warnings (empty when nothing was flagged)."""

    mode: ChatModePayload
    warnings: list[str]

    @classmethod
    def from_saved(cls, saved: SavedMode) -> ChatModeSaved:
        return cls(mode=ChatModePayload.from_mode(saved.mode), warnings=list(saved.warnings))


class ChatModeDeleted(BaseModel):
    """How many conversations used the deleted mode, and the sentence that says so."""

    conversations: int
    message: str
