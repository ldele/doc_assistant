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
from apps.api.models.modes import ChatModePayload

if TYPE_CHECKING:
    from doc_assistant.chat_setup import SetupView
    from doc_assistant.conversations import (
        ConversationDetail,
        ConversationSummary,
        ConversationTurn,
    )


class LabelDocumentPayload(BaseModel):
    document_id: str
    title: str


class LabelFolderPayload(BaseModel):
    folder_id: str
    name: str


class ConversationLabelsPayload(BaseModel):
    """The documents a conversation's answers cited, and the folders holding them (ADR-066
    choice 7, A7b): derived from the records on every read, never stored."""

    documents: list[LabelDocumentPayload] = []
    folders: list[LabelFolderPayload] = []


class SetupFolderPayload(BaseModel):
    """A folder a chat searches, by its current name; ``name`` null = since deleted."""

    folder_id: str
    name: str | None = None


class ConversationSummaryPayload(BaseModel):
    """One conversation in the sidebar list."""

    session_id: str
    title: str
    turn_count: int
    started_at: datetime
    last_at: datetime
    pinned: bool = False
    archived: bool = False
    labels: ConversationLabelsPayload = ConversationLabelsPayload()
    # A8a: what the rail's Group reads; `folders` empty = the whole library.
    mode_name: str = "Research assistant"
    folders: list[SetupFolderPayload] = []

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
            labels=ConversationLabelsPayload(
                documents=[
                    LabelDocumentPayload(document_id=d.document_id, title=d.title)
                    for d in s.labels.documents
                ],
                folders=[
                    LabelFolderPayload(folder_id=f.folder_id, name=f.name)
                    for f in s.labels.folders
                ],
            ),
            mode_name=s.mode_name,
            folders=[SetupFolderPayload(folder_id=f.folder_id, name=f.name) for f in s.folders],
        )


class ConversationMetaUpdate(BaseModel):
    """PATCH body for a conversation's management flags — only the fields sent are changed.
    ``deleted`` toggles the soft-delete (True hides + retains records; False restores). ``title``
    sets a custom title (blank reverts to the derived first-question title)."""

    pinned: bool | None = None
    archived: bool | None = None
    deleted: bool | None = None
    title: str | None = None
    # A7: the conversation's chat mode from its next turn — a mode's id; "" sets the default.
    mode_id: str | None = None


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
    # A7 — the chat mode the turn's answer prompt used, and whether it is evaluated; null on a turn
    # no answer prompt took part in (evidence only, a records answer) and on older records.
    chat_mode: str | None = None
    chat_mode_evaluated: bool | None = None

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
            scope=(
                ScopePayload.from_view(t.retrieval_scope)
                if t.retrieval_scope is not None
                else None
            ),
            chat_mode=t.chat_mode,
            chat_mode_evaluated=t.chat_mode_evaluated,
        )


class ContinuedFromPayload(BaseModel):
    """The chat this one was continued from, as the line at the top names it (A8a)."""

    session_id: str
    title: str | None = None
    mode_name: str
    scope: ScopePayload | None = None


class ChatSetupPayload(BaseModel):
    """A chat's setup (A8a): its mode and folders, fixed for the chat. ``scope`` is the
    folders resolved now (null = the whole library), so the line at the top of the chat names
    each folder with its count and says which one holds nothing (ADR-025 F2). ``stored``: written
    on the chat; ``derived``: a chat from before A8, worked out from its last record."""

    mode: ChatModePayload
    folder_ids: list[str] | None = None
    scope: ScopePayload | None = None
    stored: bool = False
    derived: bool = False
    continued_from: ContinuedFromPayload | None = None
    carried_exchanges: int = 0

    @classmethod
    def from_view(cls, v: SetupView) -> ChatSetupPayload:
        origin = v.continued_from
        return cls(
            mode=ChatModePayload.from_mode(v.setup.mode),
            folder_ids=v.setup.folder_ids,
            scope=ScopePayload.from_view(v.scope) if v.scope is not None else None,
            stored=v.setup.stored,
            derived=v.setup.derived,
            continued_from=(
                ContinuedFromPayload(
                    session_id=origin.session_id,
                    title=origin.title,
                    mode_name=origin.mode_name,
                    scope=ScopePayload.from_view(origin.scope) if origin.scope else None,
                )
                if origin is not None
                else None
            ),
            carried_exchanges=v.setup.carried_exchanges,
        )


class ContinueRequest(BaseModel):
    """POST body for *Continue in a new chat* (A8a): the new chat's id and setup, and whether
    to carry the old chat's exchanges into its memory (G3, ticked by default)."""

    session_id: str
    mode_id: str | None = None
    folder_ids: list[str] | None = None
    carry: bool = True


class ContinuePayload(BaseModel):
    """The new chat: its setup, and the old chat's turns it carried, as they were answered."""

    session_id: str
    setup: ChatSetupPayload
    carried: list[ConversationTurnPayload] = []


class ResumePayload(BaseModel):
    """*Continue this chat* (A8a, open question 1): how many of its own exchanges its memory
    now holds, read from its records."""

    session_id: str
    carried_exchanges: int


class ConversationDetailPayload(BaseModel):
    """A reopened conversation — its title + ordered turns (read-only transcript), and the chat
    mode its next turn uses (A7)."""

    session_id: str
    title: str
    turns: list[ConversationTurnPayload]
    mode: ChatModePayload
    # A8a: the chat's setup, for the line at the top of a reopened chat.
    setup: ChatSetupPayload | None = None

    @classmethod
    def from_detail(
        cls, d: ConversationDetail, setup: SetupView | None = None
    ) -> ConversationDetailPayload:
        return cls(
            session_id=d.session_id,
            title=d.title,
            turns=[ConversationTurnPayload.from_turn(t) for t in d.turns],
            mode=ChatModePayload.from_mode(d.mode),
            setup=ChatSetupPayload.from_view(setup) if setup is not None else None,
        )
