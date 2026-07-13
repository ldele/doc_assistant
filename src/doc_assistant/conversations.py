"""Conversation history — a read layer over the ``AnswerRecord`` store.

Every answered chat turn already persists one ``AnswerRecord`` keyed by an indexed
``session_id`` (``db/models.py``); this module groups those rows into conversations and
rehydrates a single conversation as a **read-only** transcript. It writes nothing — the one
mutation history needs (passing ``session_id`` into ``record_answer``) lives in
``chat_controller``. Rows with ``session_id IS NULL`` (pre-2026-07-13, before that wiring) are
excluded, so history populates from the fix forward.

Design lock: `docs/specs/feature-conversation-history.md` (Decisions 1, 3, 4; grilled 2026-07-13).
The rehydrated citation panel is intentionally **degraded** — ``retrieved_chunks_json`` persists
only ``{filename, page, section, chunk_excerpt}`` (not markers, figures, or the reviewer's
``full_text``), so reopened turns carry the excerpt + citation but no marker chips or figures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select

from doc_assistant.db.models import AnswerRecord
from doc_assistant.db.session import session_scope

_TITLE_MAX = 80


@dataclass(frozen=True)
class ConversationSummary:
    """One row in the sidebar list — a whole conversation, not a turn."""

    session_id: str
    title: str
    turn_count: int
    started_at: datetime
    last_at: datetime


@dataclass(frozen=True)
class ConversationSource:
    """A rehydrated citation — enough for the read-only source panel (Decision 4)."""

    n: int
    citation: str
    excerpt: str


@dataclass(frozen=True)
class ConversationTurn:
    """One answered turn in a reopened conversation."""

    record_id: str
    question: str
    answer: str
    sources: list[ConversationSource]
    created_at: datetime


@dataclass(frozen=True)
class ConversationDetail:
    """A reopened conversation: its title + ordered turns."""

    session_id: str
    title: str
    turns: list[ConversationTurn]


def _truncate(text: str, limit: int = _TITLE_MAX) -> str:
    """Collapse whitespace and cap for a sidebar title."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed or "(untitled)"
    return collapsed[: limit - 1].rstrip() + "…"


def _sources_from_json(retrieved_chunks_json: str | None) -> list[ConversationSource]:
    """Rebuild the citation list from the persisted chunk JSON.

    ``n`` is the 1-indexed retrieval position (the same numbering the answer's ``[n]`` markers
    reference — the list is stored in retrieval order). The citation string reproduces
    ``pipeline.format_citation``'s format exactly (``[n] name \xb7 p.N \xb7 "section"``) from the
    persisted fields, so a reopened source reads identically to a live one.
    """
    items = json.loads(retrieved_chunks_json or "[]")
    sources: list[ConversationSource] = []
    for i, chunk in enumerate(items):
        idx = i + 1
        parts = [f"[{idx}] {chunk.get('filename') or 'unknown'}"]
        if chunk.get("page"):
            parts.append(f"p.{chunk['page']}")
        if chunk.get("section"):
            parts.append(f'"{chunk["section"]}"')
        sources.append(
            ConversationSource(
                n=idx,
                citation=" \xb7 ".join(parts),
                excerpt=chunk.get("chunk_excerpt") or "",
            )
        )
    return sources


def list_conversations(limit: int = 100) -> list[ConversationSummary]:
    """The most-recently-active conversations, newest first (Decision 10: no prune, cap ~100).

    Groups ``AnswerRecord`` by ``session_id`` (``NULL`` excluded); the title is the earliest
    turn's question (``original_query`` before its rewrite, else ``query``).
    """
    with session_scope() as session:
        agg = session.execute(
            select(
                AnswerRecord.session_id,
                func.count(AnswerRecord.id),
                func.min(AnswerRecord.created_at),
                func.max(AnswerRecord.created_at),
            )
            .where(AnswerRecord.session_id.is_not(None))
            .group_by(AnswerRecord.session_id)
            .order_by(func.max(AnswerRecord.created_at).desc())
            .limit(limit)
        ).all()
        if not agg:
            return []

        session_ids = [row[0] for row in agg]
        # One extra query for the earliest question per session (the title). Ascending by time,
        # first-seen-per-session wins — avoids an N+1 over the grouped sessions.
        title_rows = session.execute(
            select(
                AnswerRecord.session_id,
                AnswerRecord.query,
                AnswerRecord.original_query,
            )
            .where(AnswerRecord.session_id.in_(session_ids))
            .order_by(AnswerRecord.created_at.asc())
        ).all()
        titles: dict[str, str] = {}
        for sid, query, original_query in title_rows:
            if sid not in titles:
                titles[sid] = _truncate(original_query or query)

        return [
            ConversationSummary(
                session_id=sid,
                title=titles.get(sid, "(untitled)"),
                turn_count=count,
                started_at=started_at,
                last_at=last_at,
            )
            for sid, count, started_at, last_at in agg
        ]


def get_conversation(session_id: str) -> ConversationDetail | None:
    """Rehydrate one conversation as a read-only transcript, or ``None`` if unknown."""
    with session_scope() as session:
        rows = (
            session.execute(
                select(AnswerRecord)
                .where(AnswerRecord.session_id == session_id)
                .order_by(AnswerRecord.created_at.asc())
            )
            .scalars()
            .all()
        )
        if not rows:
            return None
        turns = [
            ConversationTurn(
                record_id=str(row.id),
                question=row.original_query or row.query,
                answer=row.answer,
                sources=_sources_from_json(row.retrieved_chunks_json),
                created_at=row.created_at,
            )
            for row in rows
        ]
        title = _truncate(rows[0].original_query or rows[0].query)
        return ConversationDetail(session_id=session_id, title=title, turns=turns)
