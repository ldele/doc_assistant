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
import re
from collections.abc import Iterable
from pathlib import Path
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
    TurnScope,
)
from doc_assistant.retrieval_scope import read_scope

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
    return labels_for(session, {"": set(doc_ids)}).get("", ConversationLabels())


def labels_for(session: Session, cited: dict[str, set[str]]) -> dict[str, ConversationLabels]:
    """Each conversation's labels from the documents its answers cited (choice 7).

    ``cited`` maps a conversation to its cited document ids (:func:`cited_documents`). Two
    queries for the whole map, whatever its size: the documents by title, and the folders that
    hold them by name. A document no longer in the library drops out of the labels."""
    every = sorted({d for ids in cited.values() for d in ids if d})
    if not every:
        return {sid: ConversationLabels() for sid in cited}
    titles = {
        str(i): ((t or "").strip() or str(f))
        for i, t, f in session.execute(
            select(Document.id, Document.title, Document.filename).where(Document.id.in_(every))
        ).all()
    }
    held: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for doc_id, folder_id, name in session.execute(
        select(document_folders.c.document_id, Folder.id, Folder.name)
        .join(Folder, Folder.id == document_folders.c.folder_id)
        .where(document_folders.c.document_id.in_(every))
    ).all():
        held.setdefault(str(doc_id), set()).add(str(folder_id))
        names[str(folder_id)] = str(name)
    out: dict[str, ConversationLabels] = {}
    for sid, ids in cited.items():
        docs = [LabelDocument(document_id=d, title=titles[d]) for d in sorted(ids) if d in titles]
        docs.sort(key=lambda d: d.title.casefold())
        folder_ids = {f for d in ids for f in held.get(d, set())}
        folders = sorted(
            (LabelFolder(folder_id=f, name=names[f]) for f in folder_ids),
            key=lambda f: (f.name, f.folder_id),
        )
        out[sid] = ConversationLabels(documents=docs, folders=folders)
    return out


def cited_documents(session: Session, session_ids: list[str]) -> dict[str, set[str]]:
    """Each conversation's cited document ids, read from its answer records (one query)."""
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


def keep_cited(
    session: Session,
    cited: dict[str, set[str]],
    *,
    document_id: str | None = None,
    folder_id: str | None = None,
) -> set[str]:
    """The conversations in ``cited`` that pass the filters: ``document_id`` keeps those whose
    answers cited that document, ``folder_id`` those that cited any document in that folder, and
    both together must both hold. No filter keeps them all. The one filter the read surface and
    the app's chat list share (A7b, G4)."""
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
    return {
        sid
        for sid, ids in cited.items()
        if (document_id is None or document_id in ids)
        and (in_folder is None or bool(in_folder & ids))
    }


def list_conversations(
    *,
    text: str | None = None,
    document_id: str | None = None,
    folder_id: str | None = None,
    offset: int = 0,
    limit: int = DEFAULT_PAGE,
) -> ConversationList:
    """One page of past conversations, most recently active first, with their labels.

    ``text`` keeps conversations where a question contains it (case aside; the person's own
    title counts too). ``document_id`` keeps conversations whose answers cited that document;
    ``folder_id`` those that cited any document in that folder. Filters given together must all
    hold."""
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
        wanted = (text or "").strip().casefold()
        if wanted:
            asked = _asking(session, [str(r[0]) for r in rows], wanted)
            rows = [r for r in rows if str(r[0]) in asked]
        filtering = document_id is not None or folder_id is not None
        cited: dict[str, set[str]] = {}
        if filtering:
            cited = cited_documents(session, [str(r[0]) for r in rows])
            kept = keep_cited(session, cited, document_id=document_id, folder_id=folder_id)
            rows = [r for r in rows if str(r[0]) in kept]
        total = len(rows)
        page = rows[offset : offset + limit]
        ids = [str(r[0]) for r in page]
        if not filtering:
            cited = cited_documents(session, ids)
        titles = _titles(session, ids)
        labels = labels_for(session, {sid: cited.get(sid, set()) for sid in ids})
        items = [
            ConversationItem(
                conversation_id=sid,
                title=titles.get(sid, "(untitled)"),
                turns=int(n),
                started_at=started,
                last_at=last,
                labels=labels[sid],
            )
            for sid, n, started, last in ((str(r[0]), r[1], r[2], r[3]) for r in page)
        ]
    return ConversationList(conversations=items, total=total, offset=offset, limit=limit)


def _asking(session: Session, session_ids: list[str], wanted: str) -> set[str]:
    """The conversations among ``session_ids`` with a question, or a title, containing ``wanted``
    (already case-folded). Python-side, so case folding matches beyond ASCII."""
    if not session_ids:
        return set()
    hits = {
        str(sid)
        for sid, query, original in session.execute(
            select(AnswerRecord.session_id, AnswerRecord.query, AnswerRecord.original_query).where(
                AnswerRecord.session_id.in_(session_ids)
            )
        ).all()
        if wanted in (original or query or "").casefold() or wanted in (query or "").casefold()
    }
    for sid, title in session.execute(
        select(ConversationMeta.session_id, ConversationMeta.title_override).where(
            ConversationMeta.session_id.in_(session_ids)
        )
    ).all():
        if title and wanted in str(title).casefold():
            hits.add(str(sid))
    return hits


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


#: Words of a cited excerpt matched to place it in its document. Enough to be unique in practice
#: (a whole sentence's opening); fewer than 4 is too short to trust, and the page stays unknown.
_PLACE_WORDS = 12
_MIN_PLACE_WORDS = 4


class _Pager:
    """Places a cited excerpt on a page of its document, reading each document's text once.

    An answer record keeps a passage's excerpt and document, not its position, so the page is
    found by locating the excerpt's opening words in the extracted text and reading the page
    marker before it (`library.source_view.page_for_offset`). On the trial copy of the public
    papers this placed 3,869 of 4,000 cited passages (2026-11-17); the rest keep ``None``."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._texts: dict[str, str] = {}

    def _text(self, document_id: str) -> str:
        if document_id not in self._texts:
            from doc_assistant.stored_paths import confine

            row = self._session.get(Document, document_id)
            path = confine(row.source_cache, "cache") if row is not None else None
            try:
                self._texts[document_id] = (
                    Path(path).read_text(encoding="utf-8") if path and Path(path).exists() else ""
                )
            except OSError:
                self._texts[document_id] = ""
        return self._texts[document_id]

    def page(self, chunk: dict[str, Any]) -> int | None:
        stored = chunk.get("page")
        if isinstance(stored, int):
            return stored
        doc_id = chunk.get("doc_id")
        words = re.findall(r"\S+", str(chunk.get("chunk_excerpt") or ""))[:_PLACE_WORDS]
        if not doc_id or len(words) < _MIN_PLACE_WORDS:
            return None
        text = self._text(str(doc_id))
        found = re.search(r"\s+".join(map(re.escape, words)), text) if text else None
        if found is None:
            return None
        from doc_assistant.library.source_view import page_for_offset

        return page_for_offset(text, found.start())


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
        pager = _Pager(session)
        turns: list[Turn] = []
        for rec, chunks in cited:
            scope = read_scope(rec.retrieval_scope_json)
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
                            page=pager.page(c),
                            section=c.get("section"),
                            excerpt=str(c.get("chunk_excerpt") or ""),
                        )
                        for i, c in enumerate(chunks)
                    ],
                    # G6: `folder` keeps its meaning, set only for exactly one folder.
                    folder=(
                        scope.folders[0].folder_name
                        if scope is not None and scope.concept is None and len(scope.folders) == 1
                        else None
                    ),
                    scope=(
                        TurnScope(
                            folders=[f.folder_name for f in scope.folders],
                            concept=scope.concept.name if scope.concept is not None else None,
                            documents=scope.doc_count,
                        )
                        if scope is not None
                        else None
                    ),
                )
            )
        title = (meta.title_override if meta and meta.title_override else None) or _title(
            records[0].original_query or records[0].query
        )
    return ConversationRecord(
        conversation_id=conversation_id, title=title, turns=turns, labels=labels
    )
