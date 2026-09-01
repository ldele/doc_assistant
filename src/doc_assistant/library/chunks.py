"""Chunk browser (Library space L1, docs/specs/feature-library-browser.md).

A read-only view of what the two-tier retriever stored, grouped into parent blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from doc_assistant.db.models import Document, DocumentMeta
from doc_assistant.db.session import session_scope
from doc_assistant.library.documents import effective_metadata

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
        # The user's overrides win over what extraction found (ADR-013). This header is the title
        # a reader sees on the document's own page, and it showed the extracted value while the
        # Library grid beside it showed the corrected one (2026-08-19).
        meta = session.get(DocumentMeta, doc_id)
        d_title, d_authors, d_year = effective_metadata(doc, meta)
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


# ============================================================
# Where a chunk sits in its source (ROADMAP 19)
# ============================================================


@dataclass(frozen=True)
class ChunkContext:
    """A cited passage shown *in place* — the surrounding text, with the passage marked.

    "Which document" is what a citation already says; this is the "where in it" half. The window
    is read straight out of the cached markdown at the chunk's recorded span, so the reader sees
    the passage with what came before and after it rather than a floating excerpt.

    The offsets exist because ingest records them (`char_start`/`char_end`, ROADMAP 19) — nothing
    here re-derives a position per query, which is the ingest-once-amortises rule. A chunk whose
    span was never resolvable simply has no context to show, and the caller says so; it is not
    approximated, because a window centred on the wrong paragraph is worse than no window.
    """

    document_id: str
    filename: str
    #: The passage itself, exactly as the span names it.
    text: str
    #: Cached markdown immediately before and after it, trimmed to a word boundary.
    before: str
    after: str
    char_start: int
    char_end: int
    #: Total characters in the cached markdown — with `char_start`, the "34% of the way in" the
    #: citation cannot otherwise say.
    doc_chars: int
    #: The page of the *original* this passage starts on. Stored when the chunk carried one (a
    #: figure), otherwise derived from the cache's page markers (ADR-050 D2) — so it is populated
    #: for text parents too, which never stored one. `None` only for a cache without markers.
    page: int | None
    #: True when the window was cut at the start/end of the document rather than at `window`.
    at_document_start: bool
    at_document_end: bool


def _trim_to_word(text: str, *, from_start: bool) -> str:
    """Drop a partial word at the cut edge, so a window never opens mid-token."""
    if from_start:
        _, sep, rest = text.partition(" ")
        return rest if sep else text
    head, sep, _ = text.rpartition(" ")
    return head if sep else text


def _chunk_metadata(chunk_key: str, chroma: Any) -> tuple[str, dict[str, Any]] | None:
    """Resolve an epistemics-format ``chunk_key`` to ``(document_id, metadata)``.

    The two key shapes are the two segmentations (`chat_controller._chunk_key`):
    ``{document_id}:{chunk_index}`` for a flat/baseline chunk and ``{document_id}:p{parent_index}``
    for a parent-child parent — which is what a chat citation carries, since the parent is the unit
    the LLM reads.
    """
    document_id, _, tail = chunk_key.rpartition(":")
    if not document_id or not tail:
        return None
    if tail.startswith("p"):
        field, raw = "parent_index", tail[1:]
    else:
        field, raw = "chunk_index", tail
    try:
        index = int(raw)
    except ValueError:
        return None
    try:
        got = chroma.get(
            where={"$and": [{"document_id": document_id}, {field: index}]},
            include=["metadatas"],
            limit=1,
        )
    except Exception as e:  # a store that will not answer is a "no context", not a 500
        log.warning("chunk_context_lookup_failed", chunk_key=chunk_key, error=str(e))
        return None
    metas = [m for m in (got.get("metadatas") or []) if m]
    return (document_id, metas[0]) if metas else None


def get_chunk_context(chunk_key: str, chroma: Any, *, window: int = 700) -> ChunkContext | None:
    """The cached markdown around a cited chunk, or ``None`` when it cannot be placed.

    ``None`` covers every honest failure — unknown key, a chunk whose span never resolved, a cache
    file that is gone — and the caller reports it as "we cannot show where this sits" rather than
    guessing at a position.
    """
    from pathlib import Path

    found = _chunk_metadata(chunk_key, chroma)
    if found is None:
        return None
    document_id, meta = found

    # A parent-child citation is a *parent*: its span is the parent's, which is the passage the
    # answer was actually drawn from. A flat chunk carries its own.
    start = (
        meta.get("parent_char_start") if "parent_char_start" in meta else meta.get("char_start")
    )
    end = meta.get("parent_char_end") if "parent_char_end" in meta else meta.get("char_end")
    if start is None or end is None:
        return None

    cache = meta.get("source_cache")
    if not cache or not Path(str(cache)).exists():
        return None
    try:
        text = Path(str(cache)).read_text(encoding="utf-8")
    except OSError as e:
        log.warning("chunk_context_unreadable", file=str(cache), error=str(e))
        return None

    start, end = int(start), int(end)
    if not (0 <= start < end <= len(text)):
        return None  # a span the cache no longer supports: say nothing rather than slice wrongly

    # The page, when the chunk did not store one — which on the parent-child path is every chunk
    # that is not a figure (615 of 39,705 carry one; ADR-050 D2). Derived from the cache's own
    # `<!-- page:N -->` markers, which costs nothing here: text and offset are already read.
    from doc_assistant.library.source_view import page_for_offset

    page = int(meta["page"]) if meta.get("page") is not None else page_for_offset(text, start)

    before_at = max(0, start - window)
    after_to = min(len(text), end + window)
    return ChunkContext(
        document_id=document_id,
        filename=str(meta.get("filename") or ""),
        text=text[start:end],
        before=_trim_to_word(text[before_at:start], from_start=True),
        after=_trim_to_word(text[end:after_to], from_start=False),
        char_start=start,
        char_end=end,
        doc_chars=len(text),
        page=page,
        at_document_start=before_at == 0,
        at_document_end=after_to == len(text),
    )
