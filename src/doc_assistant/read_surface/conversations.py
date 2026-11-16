"""The conversation tools: past conversations and one conversation's turns (ADR-066 choices 3, 7).

Read from the answer records the chat already keeps; nothing new is stored. A conversation's
*labels* are the documents and folders its answers cited, derived from those records on every
read (choice 7), so they follow a renamed folder or a document added to one.

Whether an outside program may read conversations at all is the endpoint's switch (choice 9, A4);
these functions only read. A deleted conversation is never listed or returned.

**Scale.** Conversations grow with use, not with the library. Without a filter, one grouped query
counts them and only the page's records are read for labels. A filter by document or folder reads
the cited sources of every live conversation's records, which is the price of labels that are
derived rather than stored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from doc_assistant.db.models import (
    AnswerRecord,
    ConversationMeta,
    Document,
    Folder,
    document_folders,
)
from doc_assistant.db.session import session_scope
from doc_assistant.read_surface.shapes import (
    CitedPassage,
    ConversationItem,
    ConversationLabels,
    ConversationList,
    ConversationRecord,
    LabelDocument,
    LabelFolder,
    Turn,
)

MAX_PAGE = 100
DEFAULT_PAGE = 20
_TITLE_MAX = 80


def _cited(raw: str | None) -> list[dict[str, Any]]:
    try:
        items = json.loads(raw or "[]")
    except ValueError:
        return []
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _title(text: str | None) -> str:
    flat = " ".join((text or "").split())
    if not flat:
        return "(untitled)"
    return flat if len(flat) <= _TITLE_MAX else flat[: _TITLE_MAX - 1].rstrip() + "…"


def _labels(session: Session, doc_ids: Iterable[str]) -> ConversationLabels:
    """The cited documents, by title, and the folders that hold them, by name."""
    ids = sorted({d for d in doc_ids if d})
    if not ids:
        return ConversationLabels()
    docs = [
        LabelDocument(document_id=str(i), title=((t or "").strip() or f))
        for i, t, f in session.execute(
            select(Document.id, Document.title, Document.filename).where(Document.id.in_(ids))
        ).all()
    ]
    folders = [
        LabelFolder(folder_id=str(i), name=str(n))
        for i, n in session.execute(
            select(Folder.id, Folder.name)
            .where(
                Folder.id.in_(
                    select(document_folders.c.folder_id).where(
                        document_folders.c.document_id.in_(ids)
                    )
                )
            )
            .order_by(Folder.name)
        ).all()
    ]
    docs.sort(key=lambda d: d.title.casefold())
    return ConversationLabels(documents=docs, folders=folders)


def _cited_doc_ids(session: Session, session_ids: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {sid: set() for sid in session_ids}
    if not session_ids:
        return out
    for sid, raw in session.execute(
        select(AnswerRecord.session_id, AnswerRecord.retrieved_chunks_json).where(
            AnswerRecord.session_id.in_(session_ids)
        )
    ).all():
        out[str(sid)].update(str(c["doc_id"]) for c in _cited(raw) if c.get("doc_id"))
    return out


def list_conversations(
    *,
    document_id: str | None = None,
    folder_id: str | None = None,
    offset: int = 0,
    limit: int = DEFAULT_PAGE,
) -> ConversationList:
    """One page of past conversations, most recently active first, with their labels.

    ``document_id`` keeps conversations whose answers cited that document; ``folder_id`` those
    that cited any document in that folder. Both given: both must hold."""
    offset, limit = max(0, offset), max(1, min(limit, MAX_PAGE))
    with session_scope() as session:
        deleted = select(ConversationMeta.session_id).where(
            ConversationMeta.deleted_at.is_not(None)
        )
        rows = session.execute(
            select(
                AnswerRecord.session_id,
                func.count(AnswerRecord.id),
                func.min(AnswerRecord.created_at),
                func.max(AnswerRecord.created_at),
            )
            .where(AnswerRecord.session_id.is_not(None))
            .where(AnswerRecord.session_id.not_in(deleted))
            .group_by(AnswerRecord.session_id)
            .order_by(func.max(AnswerRecord.created_at).desc(), AnswerRecord.session_id)
        ).all()
        filtering = document_id is not None or folder_id is not None
        cited: dict[str, set[str]] = {}
        if filtering:
            cited = _cited_doc_ids(session, [str(r[0]) for r in rows])
            in_folder: set[str] | None = None
            if folder_id is not None:
                in_folder = {
                    str(d)
                    for d in session.execute(
                        select(document_folders.c.document_id).where(
                            document_folders.c.folder_id == folder_id
                        )
                    ).scalars()
                }
            rows = [
                r
                for r in rows
                if (document_id is None or document_id in cited[str(r[0])])
                and (in_folder is None or bool(in_folder & cited[str(r[0])]))
            ]
        total = len(rows)
        page = rows[offset : offset + limit]
        ids = [str(r[0]) for r in page]
        if not filtering:
            cited = _cited_doc_ids(session, ids)
        titles = _titles(session, ids)
        items = [
            ConversationItem(
                conversation_id=sid,
                title=titles.get(sid, "(untitled)"),
                turns=int(n),
                started_at=started,
                last_at=last,
                labels=_labels(session, cited.get(sid, set())),
            )
            for sid, n, started, last in ((str(r[0]), r[1], r[2], r[3]) for r in page)
        ]
    return ConversationList(conversations=items, total=total, offset=offset, limit=limit)


def _titles(session: Session, session_ids: list[str]) -> dict[str, str]:
    """Each conversation's title: the person's own, else its first question."""
    if not session_ids:
        return {}
    titles: dict[str, str] = {}
    for sid, query, original in session.execute(
        select(AnswerRecord.session_id, AnswerRecord.query, AnswerRecord.original_query)
        .where(AnswerRecord.session_id.in_(session_ids))
        .order_by(AnswerRecord.created_at.asc())
    ).all():
        titles.setdefault(str(sid), _title(original or query))
    for sid, override in session.execute(
        select(ConversationMeta.session_id, ConversationMeta.title_override).where(
            ConversationMeta.session_id.in_(session_ids)
        )
    ).all():
        if override:
            titles[str(sid)] = str(override)
    return titles


def get_conversation(conversation_id: str) -> ConversationRecord | None:
    """One conversation's turns in order, each answer with the passages it was given.

    ``None`` for an unknown or deleted conversation."""
    with session_scope() as session:
        meta = session.get(ConversationMeta, conversation_id)
        if meta is not None and meta.deleted_at is not None:
            return None
        records = list(
            session.execute(
                select(AnswerRecord)
                .where(AnswerRecord.session_id == conversation_id)
                .order_by(AnswerRecord.created_at.asc())
            )
            .scalars()
            .all()
        )
        if not records:
            return None
        cited = [(rec, _cited(rec.retrieved_chunks_json)) for rec in records]
        doc_ids = {str(c["doc_id"]) for _, chunks in cited for c in chunks if c.get("doc_id")}
        labels = _labels(session, doc_ids)
        names = {d.document_id: d.title for d in labels.documents}
        turns: list[Turn] = []
        for rec, chunks in cited:
            scope = json.loads(rec.retrieval_scope_json) if rec.retrieval_scope_json else None
            turns.append(
                Turn(
                    question=rec.original_query or rec.query,
                    answer=rec.answer,
                    asked_at=rec.created_at,
                    cited=[
                        CitedPassage(
                            n=i + 1,
                            document_id=c.get("doc_id"),
                            title=names.get(str(c.get("doc_id")))
                            or str(c.get("filename") or "unknown"),
                            page=c.get("page") if isinstance(c.get("page"), int) else None,
                            section=c.get("section"),
                            excerpt=str(c.get("chunk_excerpt") or ""),
                        )
                        for i, c in enumerate(chunks)
                    ],
                    folder=(scope or {}).get("folder_name") if isinstance(scope, dict) else None,
                )
            )
        title = (meta.title_override if meta and meta.title_override else None) or _title(
            records[0].original_query or records[0].query
        )
    return ConversationRecord(
        conversation_id=conversation_id, title=title, turns=turns, labels=labels
    )
