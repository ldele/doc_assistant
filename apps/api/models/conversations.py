"""Conversation-history wire models (feature-conversation-history.md — read-only).

The sidebar list, its management-flag PATCH body, and the rehydrated transcript. A replayed
turn is deliberately *degraded* against a live one (no markers, no figures, no claims — none of
that is persisted); it reuses ``ScopePayload`` from ``chat`` so a reopened scoped answer still
reports the scope it actually ran under.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from apps.api.models._common import _as_utc
from apps.api.models.chat import ScopePayload

if TYPE_CHECKING:
    from doc_assistant.conversations import (
        ConversationDetail,
        ConversationSummary,
        ConversationTurn,
    )


class ConversationSummaryPayload(BaseModel):
    """One conversation in the sidebar list."""

    session_id: str
    title: str
    turn_count: int
    started_at: datetime
    last_at: datetime
    pinned: bool = False
    archived: bool = False

    @classmethod
    def from_summary(cls, s: ConversationSummary) -> ConversationSummaryPayload:
        return cls(
            session_id=s.session_id,
            title=s.title,
            turn_count=s.turn_count,
            started_at=_as_utc(s.started_at),
            last_at=_as_utc(s.last_at),
            pinned=s.pinned,
            archived=s.archived,
        )


class ConversationMetaUpdate(BaseModel):
    """PATCH body for a conversation's management flags — only the fields sent are changed.
    ``deleted`` toggles the soft-delete (True hides + retains records; False restores). ``title``
    sets a custom title (blank reverts to the derived first-question title)."""

    pinned: bool | None = None
    archived: bool | None = None
    deleted: bool | None = None
    title: str | None = None


class ConversationBulkUpdate(BaseModel):
    """POST body for the sidebar's "delete selected" — the same soft-delete as the single-row
    PATCH, applied to a list in one transaction. ``deleted=False`` restores, so a mis-click is
    undoable by the same route that made it."""

    session_ids: list[str]
    deleted: bool = True


class ConversationBulkResult(BaseModel):
    """How many conversations the bulk action touched (deduped ids)."""

    updated: int


class ConversationSourcePayload(BaseModel):
    n: int
    citation: str
    excerpt: str


class ConversationTurnPayload(BaseModel):
    record_id: str
    question: str
    answer: str
    sources: list[ConversationSourcePayload]
    # ADR-025 F2 — replayed from the record, so a reopened scoped answer still says it was scoped.
    scope: ScopePayload | None = None

    @classmethod
    def from_turn(cls, t: ConversationTurn) -> ConversationTurnPayload:
        return cls(
            record_id=t.record_id,
            question=t.question,
            answer=t.answer,
            sources=[
                ConversationSourcePayload(n=s.n, citation=s.citation, excerpt=s.excerpt)
                for s in t.sources
            ],
            scope=(ScopePayload(**t.retrieval_scope) if t.retrieval_scope is not None else None),
        )


class ConversationDetailPayload(BaseModel):
    """A reopened conversation — its title + ordered turns (read-only transcript)."""

    session_id: str
    title: str
    turns: list[ConversationTurnPayload]

    @classmethod
    def from_detail(cls, d: ConversationDetail) -> ConversationDetailPayload:
        return cls(
            session_id=d.session_id,
            title=d.title,
            turns=[ConversationTurnPayload.from_turn(t) for t in d.turns],
        )
