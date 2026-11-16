"""`search_library`: ranked passages for a question (ADR-066 choices 1, 3, 9).

Passages, not an answer: the caller's own model writes the answer, so nothing here calls a model
or bills one. Retrieval is the chat's own (hybrid, reranked, at the locked settings), handed in by
the engine that holds it, so this module loads no model of its own.

Each passage carries its document, page and a stable id, its text, the same text inside the
chat's fence (`pipeline.fence_passage`), and what the text check found in it (`source_checks`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document as Chunk
from sqlalchemy import select

from doc_assistant.config import TOP_K
from doc_assistant.db.models import Document, DocumentMeta
from doc_assistant.db.session import session_scope
from doc_assistant.library.documents import count_documents, effective_metadata
from doc_assistant.read_surface.shapes import Passage, SearchResult
from doc_assistant.source_checks import addressed_to_ai

#: The engine's retrieval: ``(question, top_k, scope=…)`` to ``(chunk, score)`` pairs, best first.
#: `RAGPipeline.retrieve_with_scores` is the one the app passes.
Retrieve = Callable[..., list[tuple[Chunk, float]]]

#: The most passages one call returns. A wire-size bound; the default is the chat's own top-k.
MAX_TOP_K = 20

NO_DOCUMENTS = "The library holds no documents yet."
NO_FOLDER = "No folder has this id."
EMPTY_SCOPE = "The folder or documents named hold no indexed documents."
NOTHING_FOUND = "No passage matched the question."


def _scope(folder_id: str | None, document_ids: list[str] | None) -> frozenset[str] | None:
    """The ``doc_hash`` set to search within, ``None`` for the whole library.

    A folder and a list together: the documents of the list that are in the folder."""
    if folder_id is None and not document_ids:
        return None
    from doc_assistant.library.folders import folder_doc_hashes

    hashes: set[str] | None = set(folder_doc_hashes(folder_id)) if folder_id is not None else None
    if document_ids:
        with session_scope() as session:
            named = {
                str(h)
                for h in session.execute(
                    select(Document.doc_hash).where(
                        Document.id.in_(document_ids), Document.is_archived.is_(False)
                    )
                ).scalars()
            }
        hashes = named if hashes is None else hashes & named
    return frozenset(hashes or ())


def _folder_exists(folder_id: str) -> bool:
    from doc_assistant.library.folders import get_folder

    return get_folder(folder_id) is not None


def _records(document_ids: set[str]) -> dict[str, tuple[str, str | None, int | None]]:
    """``document id -> (title, authors, year)``, the person's edits applied."""
    if not document_ids:
        return {}
    with session_scope() as session:
        docs = list(
            session.execute(select(Document).where(Document.id.in_(document_ids))).scalars()
        )
        overrides = {
            m.document_id: m
            for m in session.execute(
                select(DocumentMeta).where(DocumentMeta.document_id.in_(document_ids))
            ).scalars()
        }
        out: dict[str, tuple[str, str | None, int | None]] = {}
        for d in docs:
            title, authors, year = effective_metadata(d, overrides.get(d.id))
            out[str(d.id)] = ((title or "").strip() or d.filename, authors, year)
        return out


def _passage(n: int, chunk: Chunk, score: float, records: dict[str, Any]) -> Passage:
    from doc_assistant.chat_controller import _chunk_key
    from doc_assistant.pipeline import fence_passage

    meta = chunk.metadata
    doc_id = meta.get("document_id")
    title, authors, year = records.get(
        str(doc_id), (str(meta.get("filename") or "unknown"), None, None)
    )
    page = meta.get("page")
    return Passage(
        passage_id=_chunk_key(meta),
        document_id=str(doc_id) if doc_id is not None else None,
        title=title,
        authors=authors,
        year=year,
        page=page if isinstance(page, int) else None,
        section=meta.get("section"),
        score=float(score),
        text=chunk.page_content,
        fenced=fence_passage(chunk, n),
        addressed_to_ai=addressed_to_ai(chunk.page_content),
    )


def search_library(
    retrieve: Retrieve,
    question: str,
    *,
    top_k: int = TOP_K,
    folder_id: str | None = None,
    document_ids: list[str] | None = None,
) -> SearchResult:
    """Ranked passages for ``question``, from the whole library or within a folder or documents.

    Never widens a scope that holds nothing: an unknown folder, or a list of documents that are
    not indexed, returns no passages and says why. An empty library is answered without calling
    retrieval at all."""
    top_k = max(1, min(top_k, MAX_TOP_K))
    asked = question.strip()

    def empty(note: str) -> SearchResult:
        return SearchResult(question=asked, passages=[], count=0, note=note, folder_id=folder_id)

    if count_documents() == 0:
        return empty(NO_DOCUMENTS)
    if folder_id is not None and not _folder_exists(folder_id):
        return empty(NO_FOLDER)
    if not asked:
        return empty(NOTHING_FOUND)
    scope = _scope(folder_id, document_ids)
    if scope is not None and not scope:
        return empty(EMPTY_SCOPE)
    found = retrieve(asked, top_k, scope=scope)
    records = _records(
        {str(c.metadata["document_id"]) for c, _ in found if c.metadata.get("document_id")}
    )
    passages = [_passage(i + 1, c, s, records) for i, (c, s) in enumerate(found)]
    return SearchResult(
        question=asked,
        passages=passages,
        count=len(passages),
        note=None if passages else NOTHING_FOUND,
        folder_id=folder_id,
    )
