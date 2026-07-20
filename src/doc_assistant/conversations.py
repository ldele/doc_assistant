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
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from doc_assistant.db.models import AnswerRecord, ConversationMeta
from doc_assistant.db.session import session_scope
from doc_assistant.export import ExportSource, ExportTurn

_TITLE_MAX = 80


@dataclass(frozen=True)
class ConversationSummary:
    """One row in the sidebar list — a whole conversation, not a turn."""

    session_id: str
    title: str
    turn_count: int
    started_at: datetime
    last_at: datetime
    pinned: bool = False
    archived: bool = False


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
    # ADR-025 F2 — {folder_id, folder_name, doc_count} if the turn was folder-scoped, else None.
    # Replayed from the record so a reopened conversation cannot present a scoped answer as a
    # whole-library one; that would be the same silent lie, just deferred.
    retrieval_scope: dict[str, Any] | None = None


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


def set_conversation_meta(
    session_id: str,
    *,
    pinned: bool | None = None,
    archived: bool | None = None,
    deleted: bool | None = None,
    title: str | None = None,
) -> None:
    """Upsert a conversation's management flags — only the fields passed are changed.

    ``deleted=True`` **soft-deletes** (stamps ``deleted_at``, hiding it from the list while its
    ``AnswerRecord`` provenance is retained); ``deleted=False`` restores it. ``title`` sets a
    custom title (blank reverts to the derived first-question title). Creates the sidecar row on
    first action; an absent row means all-default (not pinned/archived/deleted, derived title)."""
    with session_scope() as session:
        meta = session.get(ConversationMeta, session_id)
        if meta is None:
            meta = ConversationMeta(session_id=session_id)
            session.add(meta)
        if pinned is not None:
            meta.pinned = pinned
        if archived is not None:
            meta.archived = archived
        if deleted is not None:
            meta.deleted_at = datetime.now(timezone.utc) if deleted else None
        if title is not None:
            meta.title_override = title.strip() or None


def list_conversations(limit: int = 100) -> list[ConversationSummary]:
    """The most-recently-active conversations (Decision 10: no prune, cap ~100). Soft-deleted
    conversations are excluded; pinned ones sort first, then newest-first within each group.

    Groups ``AnswerRecord`` by ``session_id`` (``NULL`` excluded); the title is the earliest
    turn's question (``original_query`` before its rewrite, else ``query``). Pin/archive flags
    come from the ``conversation_meta`` sidecar (absent row = defaults).
    """
    with session_scope() as session:
        deleted_subq = select(ConversationMeta.session_id).where(
            ConversationMeta.deleted_at.is_not(None)
        )
        agg = session.execute(
            select(
                AnswerRecord.session_id,
                func.count(AnswerRecord.id),
                func.min(AnswerRecord.created_at),
                func.max(AnswerRecord.created_at),
            )
            .where(AnswerRecord.session_id.is_not(None))
            .where(AnswerRecord.session_id.not_in(deleted_subq))
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

        flag_rows = session.execute(
            select(
                ConversationMeta.session_id,
                ConversationMeta.pinned,
                ConversationMeta.archived,
                ConversationMeta.title_override,
            ).where(ConversationMeta.session_id.in_(session_ids))
        ).all()
        flags = {sid: (bool(pinned), bool(archived)) for sid, pinned, archived, _ in flag_rows}
        overrides = {sid: title for sid, _, _, title in flag_rows if title}

        summaries = [
            ConversationSummary(
                session_id=sid,
                title=overrides.get(sid) or titles.get(sid, "(untitled)"),
                turn_count=count,
                started_at=started_at,
                last_at=last_at,
                pinned=flags.get(sid, (False, False))[0],
                archived=flags.get(sid, (False, False))[1],
            )
            for sid, count, started_at, last_at in agg
        ]
        # Pinned first; stable sort preserves the newest-first order within each group.
        summaries.sort(key=lambda c: not c.pinned)
        return summaries


def conversation_export_turns(session_id: str) -> list[ExportTurn]:
    """Build export turns for a whole conversation from its durable ``AnswerRecord`` rows.

    The persisted records are the *complete* transcript, so a reopened or resumed chat exports
    the same as a live one (the in-memory session may hold only the newest turns, or none).
    Telemetry is best-effort from the persisted columns; the reviewer verdict and figures are
    not stored on ``AnswerRecord`` (sibling tables), so they're omitted here — a clean
    user transcript, not the live dev bundle."""
    turns: list[ExportTurn] = []
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
        # Build inside the session scope — the ORM rows detach once it closes.
        for row in rows:
            chunks = json.loads(row.retrieved_chunks_json or "[]")
            sources = [
                ExportSource(
                    n=i + 1,
                    filename=chunk.get("filename"),
                    page=chunk.get("page"),
                    section=chunk.get("section"),
                    reranker_score=chunk.get("reranker_score"),
                    excerpt=chunk.get("chunk_excerpt"),
                )
                for i, chunk in enumerate(chunks)
            ]
            turns.append(
                ExportTurn(
                    question=row.original_query or row.query,
                    answer=row.answer,
                    standalone_query=row.query if row.original_query else None,
                    sources=sources,
                    token_input=row.token_input,
                    token_output=row.token_output,
                    latency_ms=row.latency_ms,
                    model_name=row.model_name,
                    embedding_model=row.embedding_model,
                    record_id=str(row.id),
                )
            )
    return turns


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
                retrieval_scope=(
                    json.loads(row.retrieval_scope_json) if row.retrieval_scope_json else None
                ),
            )
            for row in rows
        ]
        meta = session.get(ConversationMeta, session_id)
        title = (meta.title_override if meta and meta.title_override else None) or _truncate(
            rows[0].original_query or rows[0].query
        )
        return ConversationDetail(session_id=session_id, title=title, turns=turns)
