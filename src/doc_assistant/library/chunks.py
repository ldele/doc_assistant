"""Chunk browser (Library space L1, docs/specs/feature-library-browser.md).

A read-only view of what the two-tier retriever stored, grouped into parent blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope

log = structlog.get_logger(__name__)


# ============================================================
# Chunk browser (Library space L1 — docs/specs/feature-library-browser.md)
# ============================================================
# A read-only view of the chunks the two-tier retriever stores for a document:
# parent blocks (the parent_text the LLM reads / a citation shows), each carrying
# its embedded child chunks. Reads the live Chroma handle via a metadata filter —
# no embeddings, no BM25, no generation, no writes. Markers + figures are L1b.


@dataclass
class ChunkChild:
    """One embedded child chunk within a parent block."""

    child_index: int
    text: str
    retrievable: bool  # False only when keep_for_retrieval metadata is explicitly False


@dataclass
class ParentBlock:
    """One parent block (the unit the LLM reads) + its ordered child chunks."""

    parent_index: int
    parent_text: str
    children: list[ChunkChild]


@dataclass
class DocumentChunkView:
    """A document's header + its chunks grouped into parent blocks (L1 browser detail).

    NULL metadata (``title``/``authors``/``year`` are often absent on real corpora) stays None;
    the renderer omits it rather than showing a blank label.
    """

    id: str
    filename: str
    format: str
    title: str | None
    authors: str | None
    year: int | None
    chunk_count: int | None
    health: str | None
    parents: list[ParentBlock]
    child_count: int


def group_children(chunks: list[dict[str, Any]]) -> list[ParentBlock]:
    """Group flat child chunks into ordered parent blocks — **pure**, the browser's core.

    Each input dict is one child chunk: ``parent_index`` (int), ``child_index`` (int),
    ``parent_text`` (str), ``text`` (the child's own text), ``keep_for_retrieval`` (bool | None).
    A chunk missing ``parent_index`` or ``child_index`` is dropped (logged count) — it cannot be
    placed. Parents are ordered by ``parent_index``; children within a parent by ``child_index``;
    ``parent_text`` is taken from each parent's first-seen child.
    """
    by_parent: dict[int, list[dict[str, Any]]] = {}
    parent_text: dict[int, str] = {}
    dropped = 0
    for chunk in chunks:
        p_idx = chunk.get("parent_index")
        c_idx = chunk.get("child_index")
        if p_idx is None or c_idx is None:
            dropped += 1
            continue
        by_parent.setdefault(int(p_idx), []).append(chunk)
        parent_text.setdefault(int(p_idx), str(chunk.get("parent_text") or ""))
    if dropped:
        log.info("library_chunks_dropped", count=dropped, reason="missing parent/child index")

    blocks: list[ParentBlock] = []
    for p_idx in sorted(by_parent):
        children = [
            ChunkChild(
                child_index=int(c["child_index"]),
                text=str(c.get("text") or ""),
                retrievable=c.get("keep_for_retrieval") is not False,
            )
            for c in sorted(by_parent[p_idx], key=lambda c: int(c["child_index"]))
        ]
        blocks.append(
            ParentBlock(parent_index=p_idx, parent_text=parent_text[p_idx], children=children)
        )
    return blocks


def get_document_chunks(doc_id: str, chroma: Any) -> DocumentChunkView | None:
    """One document's header + its chunks grouped into parent blocks, or ``None`` if unknown.

    ``chroma`` is the live handle (``ChatController.rag.db``) — a metadata-filtered ``get``, no
    embeddings, no generation. A document that exists but has zero stored chunks returns a view
    with ``parents=[]`` (honest empty-state), not ``None`` (which means "unknown document").
    """
    with session_scope() as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            return None
        # Read the scalar fields inside the session (avoids a detached lazy-load after close).
        d_id = str(doc.id)
        d_filename = doc.filename
        d_format = doc.format
        d_title = doc.title
        d_authors = doc.authors
        d_year = doc.year
        d_chunk_count = doc.chunk_count
        d_health = doc.extraction_health

    result = chroma.get(where={"document_id": doc_id}, include=["documents", "metadatas"])
    documents: list[str] = result.get("documents") or []
    metadatas: list[dict[str, Any]] = result.get("metadatas") or []
    chunks: list[dict[str, Any]] = [
        {
            "parent_index": (meta or {}).get("parent_index"),
            "child_index": (meta or {}).get("child_index"),
            "parent_text": (meta or {}).get("parent_text"),
            "text": text,
            "keep_for_retrieval": (meta or {}).get("keep_for_retrieval"),
        }
        for text, meta in zip(documents, metadatas, strict=True)
    ]
    parents = group_children(chunks)
    return DocumentChunkView(
        id=d_id,
        filename=d_filename,
        format=d_format,
        title=d_title,
        authors=d_authors,
        year=d_year,
        chunk_count=d_chunk_count,
        health=d_health,
        parents=parents,
        child_count=len(chunks),
    )
