"""Source viewer (ROADMAP 18, ADR-050) — the document itself, beside its library entry.

Renders one PDF page at a time on demand (ADR-050 D1), and derives *which* page a cited chunk
sits on from the cache's ``<!-- page:N -->`` markers rather than from chunk metadata (D2 — the
live parent-child store carries ``page`` on 1.5% of its chunks, all of them figures).

Read-only: nothing here writes to the registry, the chunk store, or the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest.chunking import PAGE_MARKER

log = structlog.get_logger(__name__)

#: Rendering resolution. 110 dpi is a ~935x1210 px image for US Letter — sharp on a split pane at
#: 1x and acceptable at 2x, for ~170 KB and ~19 ms a page (measured 2026-09-01 over the six
#: longest documents; 150 dpi costs 1.5x the bytes for detail a half-width pane cannot show).
#: A structural default, not a tunable: it trades legibility against transfer size, and neither
#: side is an eval-harness question.
PAGE_RENDER_DPI = 110

#: The zoom range the renderer will honour, in dpi. The floor keeps a thumbnail legible; the
#: ceiling is what stops a query parameter from being a work generator — at 400 dpi a Letter page
#: is ~3400x4400 px, and the cost of a render is quadratic in dpi (19 ms at 96, 75 ms at 200).
#: A request outside the range is clamped, never refused: the viewer asked for a picture, and the
#: sharpest one this will draw is a better answer than an error.
MIN_RENDER_DPI = 72
MAX_RENDER_DPI = 400

#: What the viewer can render as pages. Everything else is a document without pages and is shown
#: as its extracted text instead (ADR-050 D3) — not as a failure.
PAGEABLE_FORMATS = frozenset({"pdf"})


def clamp_dpi(dpi: int) -> int:
    """The render resolution, held inside the range this will draw.

    Pure, and the only place the bound is expressed — the route clamps by calling this rather than
    repeating the numbers, so a caller cannot reach the renderer with an unbounded value.
    """
    return max(MIN_RENDER_DPI, min(MAX_RENDER_DPI, dpi))


@dataclass(frozen=True)
class SourceDocumentView:
    """What the viewer needs before it can render anything — including why it cannot.

    ``reason`` is populated **only** when ``available`` is False, and it names the path, because
    "the file is not available" is not actionable and "the drive holding it is not connected" is.
    """

    document_id: str
    filename: str
    format: str
    page_count: int | None
    #: The file is on disk right now. False is a sentence to show, never a broken pane (D4).
    available: bool
    #: True when this format renders as page images at all. A non-pageable document is still a
    #: perfectly good document — see ADR-050 D3.
    pageable: bool
    path: str | None = None
    reason: str | None = None


def page_for_offset(text: str, offset: int) -> int | None:
    """The page a character offset falls on — the last ``<!-- page:N -->`` at or before it.

    **Pure**, and the whole of ADR-050 D2. Applying this rule at read time is what makes a
    page-level jump free for the parent-child store, which never persisted a page.

    It resolves on the passage's **start**, where `ingest.chunking.extract_chunk_metadata`
    resolves the flat store's stored page on the chunk's *end*. The divergence is deliberate and
    only shows on a passage that straddles a page break: the ingest rule then labels it with the
    page it finishes on, and a viewer that opened there would show the reader a page whose top is
    already past the sentence they clicked. The passage begins where it begins.

    ``None`` when no marker precedes the offset — an unmarked cache, or an offset ahead of the
    first page. The caller opens at page 1 and claims nothing about position.
    """
    if offset < 0:
        return None
    last: int | None = None
    for match in PAGE_MARKER.finditer(text):
        if match.start() > offset:
            break
        last = int(match.group(1))
    return last


def _chunk_span(meta: dict[str, Any]) -> tuple[str | None, int | None]:
    """``(cache_path, char_start)`` for a chunk — the parent's span when it has one.

    Mirrors `library.chunks.get_chunk_context`: a parent-child citation *is* a parent, so its
    span is the parent's. Values arrive from Chroma as strings.
    """
    raw = meta.get("parent_char_start") if "parent_char_start" in meta else meta.get("char_start")
    if raw is None:
        return (meta.get("source_cache"), None)
    try:
        return (meta.get("source_cache"), int(raw))
    except (TypeError, ValueError):
        return (meta.get("source_cache"), None)


@dataclass(frozen=True)
class ChunkLocation:
    """Where a cited chunk is: which document, and which page of it.

    Both halves come from one lookup because a caller opening the viewer needs both, and only
    the server can turn a ``chunk_key`` into a document id — the key's shape is a contract
    (`chat_controller._chunk_key`), and parsing it in the client would be a second copy of it.

    ``page`` is nullable where ``document_id`` is not: a chunk always belongs to a document, but
    it cannot always be placed on a page (ADR-050 D2).
    """

    document_id: str
    page: int | None


def locate_chunk(chunk_key: str, chroma: Any) -> ChunkLocation | None:
    """Resolve a cited chunk to its document and page, or ``None`` for a key nothing knows.

    Prefers a **stored** page when the chunk has one: a figure chunk's page is detection output,
    not a reconstruction, and is the better answer (ADR-050 Consequences) — and it is the only
    page a figure has, since a figure carries no text span to derive one from. Everything else
    is derived from the cache's markers.
    """
    from doc_assistant.library.chunks import _chunk_metadata

    found = _chunk_metadata(chunk_key, chroma)
    if found is None:
        return None
    document_id, meta = found
    return ChunkLocation(document_id=document_id, page=_page_from_meta(meta))


def page_for_chunk(chunk_key: str, chroma: Any) -> int | None:
    """Which page of the original a cited chunk sits on, or ``None`` when it cannot be said."""
    found = locate_chunk(chunk_key, chroma)
    return found.page if found is not None else None


def _page_from_meta(meta: dict[str, Any]) -> int | None:
    """The page a chunk's metadata implies — stored if it has one, else derived from the cache."""
    stored = meta.get("page")
    if stored is not None:
        try:
            return int(stored)
        except (TypeError, ValueError):
            pass  # fall through to derivation rather than trust an unparseable value

    cache, start = _chunk_span(meta)
    if not cache or start is None:
        return None
    path = Path(str(cache))
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("source_view_cache_unreadable", file=str(path), error=str(e))
        return None
    return page_for_offset(text, start)


def _unavailable_reason(source_original: str) -> str:
    """Why the file is not there — the drive, or the file (ADR-050 D4).

    A whole root being unreachable is a different event from one file moving, and the pane says
    which. The distinction is `RootView.available`'s reason for existing.
    """
    from doc_assistant.ingest.registry import _root_available, root_containing

    path = Path(source_original)
    try:
        with session_scope() as session:
            root = root_containing(session, path)
            if root is not None and not _root_available(root):
                return f"The drive holding this document is not connected ({root.path})."
    except Exception as e:  # a registry that will not answer must not break the pane
        log.warning("source_view_root_lookup_failed", path=source_original, error=str(e))
    return f"The file is not where the library expects it ({source_original})."


def get_source_view(document_id: str) -> SourceDocumentView | None:
    """The viewer's header for one document, or ``None`` when the document is unknown.

    An unknown *document* is a 404; an unreachable *file* is not — it is a document the app knows
    everything about except where its bytes currently are, which is exactly what `available` and
    `reason` carry.
    """
    from doc_assistant.library.documents import resolve_source_path

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        filename, fmt = doc.filename, (doc.format or "")
        page_count, source_original = doc.page_count, doc.source_original

    path = resolve_source_path(source_original, filename)
    pageable = fmt.lower() in PAGEABLE_FORMATS
    if path is None:
        return SourceDocumentView(
            document_id=document_id,
            filename=filename,
            format=fmt,
            page_count=page_count,
            available=False,
            pageable=pageable,
            reason=_unavailable_reason(source_original),
        )
    return SourceDocumentView(
        document_id=document_id,
        filename=filename,
        format=fmt,
        page_count=page_count,
        available=True,
        pageable=pageable,
        path=str(path),
    )


class PageUnavailable(RuntimeError):
    """The page cannot be rendered, with a reason meant for a person to read."""


def render_page(document_id: str, page: int, *, dpi: int = PAGE_RENDER_DPI) -> bytes:
    """One page of a document as PNG bytes — rendered on demand, cached nowhere (ADR-050 D1).

    Raises `PageUnavailable` for every honest failure: an unknown document, a format without
    pages, a file that is not on disk, a page number outside the document. The caller turns that
    into a 404 carrying the same sentence.

    ``page`` is 1-based, matching both the ``<!-- page:N -->`` markers and what a reader sees
    printed on the page. ``dpi`` is clamped rather than validated (`clamp_dpi`): it is a zoom
    level, and the honest response to "sharper than we draw" is the sharpest we draw.
    """
    dpi = clamp_dpi(dpi)
    view = get_source_view(document_id)
    if view is None:
        raise PageUnavailable("document not found")
    if not view.pageable:
        fmt = (view.format or "").upper()
        # Phrased around the format name rather than "a {fmt} document", which produced
        # "a epub document" — the article cannot agree with a value from the database.
        raise PageUnavailable(
            f"a document in {fmt} format has no pages to render"
            if fmt
            else "this document has no pages to render"
        )
    if not view.available or view.path is None:
        raise PageUnavailable(view.reason or "the source file is not available")

    import pymupdf

    doc = pymupdf.open(view.path)  # type: ignore[no-untyped-call]
    try:
        if page < 1 or page > doc.page_count:
            raise PageUnavailable(f"page {page} is outside this document (1-{doc.page_count})")
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        data: bytes = pix.tobytes("png")  # type: ignore[no-untyped-call]
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return data
