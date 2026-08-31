"""Library-browser wire models (feature-library-browser.md — L1).

One document as the grid sees it (effective title/authors/year per ADR-013), the metadata
override PATCH body, the delete outcome (ADR-014), and the drill-in chunk view grouped into
parent blocks.

Sibling modules carry the rest of the Library surface: ``connections`` (the E4 exploration
bundle), ``folders`` (ADR-025 F1) and ``keywords`` (tag families). They are split apart because
the ``library`` router is split the same way — one file per sub-domain on both sides.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from apps.api.models._common import _as_utc

if TYPE_CHECKING:
    from doc_assistant.library import (
        ChunkContext,
        DocumentChunkView,
        DocumentFigureView,
        DocumentSummary,
        FigureView,
        ParentBlock,
    )


class LibraryDocumentPayload(BaseModel):
    """One document in the Library list (mirrors ``library.DocumentSummary``).

    ``title``/``authors``/``year`` are effective values (user override ?? auto); ``customized``
    is True when a user override is in force (ADR-013)."""

    id: str
    filename: str
    title: str | None
    authors: str | None
    year: int | None
    customized: bool
    format: str
    health: str | None
    chunk_count: int | None
    page_count: int | None
    folders: list[str]
    folder_ids: list[str]
    tags: list[str]
    keywords: list[str]
    added_at: datetime | None
    #: The file's real location — the delete dialog names it before offering to bin it (ADR-046).
    source_path: str | None = None

    @classmethod
    def from_summary(cls, s: DocumentSummary) -> LibraryDocumentPayload:
        return cls(
            id=s.id,
            filename=s.filename,
            title=s.title,
            authors=s.authors,
            year=s.year,
            customized=s.customized,
            format=s.format,
            health=s.health,
            chunk_count=s.chunk_count,
            page_count=s.page_count,
            folders=list(s.folders),
            folder_ids=list(s.folder_ids),
            tags=list(s.tags),
            keywords=list(s.keywords),
            added_at=_as_utc(s.added_at) if s.added_at is not None else None,
            source_path=s.source_path,
        )


class LibraryDocumentMetaUpdate(BaseModel):
    """PATCH body for a document's user metadata overrides (ADR-013).

    The editor sends the whole small form; each field is the desired *effective* value. A blank
    string (or a value equal to the auto-extracted default) clears that field's override."""

    title: str | None = None
    authors: str | None = None
    year: int | None = None


class DeleteResultPayload(BaseModel):
    """Outcome of a document delete (ADR-014)."""

    filename: str
    trashed_file: bool
    chunks_removed: int


class LibraryChildPayload(BaseModel):
    child_index: int
    text: str
    retrievable: bool


class LibraryParentPayload(BaseModel):
    parent_index: int
    parent_text: str
    children: list[LibraryChildPayload]

    @classmethod
    def from_block(cls, b: ParentBlock) -> LibraryParentPayload:
        return cls(
            parent_index=b.parent_index,
            parent_text=b.parent_text,
            children=[
                LibraryChildPayload(
                    child_index=c.child_index, text=c.text, retrievable=c.retrievable
                )
                for c in b.children
            ],
        )


class LibraryDocumentChunksPayload(BaseModel):
    """A document's header + its chunks grouped into parent blocks (browser detail)."""

    id: str
    filename: str
    format: str
    title: str | None
    authors: str | None
    year: int | None
    chunk_count: int | None
    health: str | None
    parents: list[LibraryParentPayload]
    child_count: int

    @classmethod
    def from_view(cls, v: DocumentChunkView) -> LibraryDocumentChunksPayload:
        return cls(
            id=v.id,
            filename=v.filename,
            format=v.format,
            title=v.title,
            authors=v.authors,
            year=v.year,
            chunk_count=v.chunk_count,
            health=v.health,
            parents=[LibraryParentPayload.from_block(b) for b in v.parents],
            child_count=v.child_count,
        )


class LibraryFigurePayload(BaseModel):
    """One figure in the per-document figure panel (L1b).

    Carries ``retrievable`` + ``not_retrievable_reason`` rather than only the raw columns: the
    panel's job is to show which figures the assistant can actually see, and a list that looked
    identical either way would hide that.
    """

    id: str
    page: int
    kind: str | None
    caption: str | None
    description: str | None
    extraction_method: str | None
    has_image: bool
    retrievable: bool
    not_retrievable_reason: str | None

    @classmethod
    def from_view(cls, v: FigureView) -> LibraryFigurePayload:
        return cls(
            id=v.id,
            page=v.page,
            kind=v.kind,
            caption=v.caption,
            description=v.description,
            extraction_method=v.extraction_method,
            has_image=v.has_image,
            retrievable=v.retrievable,
            not_retrievable_reason=v.not_retrievable_reason,
        )


class LibraryDocumentFiguresPayload(BaseModel):
    """A document's figures, addressed separately from its text chunks.

    Figures are a different kind of object from prose and get their own panel; the counts let
    the header state the corpus truth ("10 figures, 2 searchable") without the client
    recomputing it from the list.
    """

    id: str
    filename: str
    title: str | None
    figures: list[LibraryFigurePayload]
    total: int
    retrievable_count: int
    captioned_count: int
    missing_image_count: int

    @classmethod
    def from_view(cls, v: DocumentFigureView) -> LibraryDocumentFiguresPayload:
        return cls(
            id=v.id,
            filename=v.filename,
            title=v.title,
            figures=[LibraryFigurePayload.from_view(f) for f in v.figures],
            total=v.total,
            retrievable_count=v.retrievable_count,
            captioned_count=v.captioned_count,
            missing_image_count=v.missing_image_count,
        )


# ============================================================
# Per-part re-ingest (ADR-048, ROADMAP 20/21)
# ============================================================


class ReingestPartPayload(BaseModel):
    """One re-runnable part, as the registry declares it.

    Served rather than hardcoded in the client on purpose: `cost` is the thing that makes the
    control honest, and a copy of it in the UI would drift from `docs/performance.md` silently.
    Mirrors ``doc_assistant.reingest.ReingestPart``.
    """

    id: str
    label: str
    blurb: str
    cost: str
    moves_identity: bool


class ReingestOptionsPayload(BaseModel):
    """What the re-run control offers, and what it deliberately does not.

    ``corpus_wide`` names the passes that have no per-document form (ADR-048) so the UI can say
    why there is no button, rather than leaving a user to conclude the feature is missing.
    """

    parts: list[ReingestPartPayload]
    corpus_wide: list[str]


class ReingestRequest(BaseModel):
    """POST body: which documents, which parts. One document is row 20; many is row 21."""

    document_ids: list[str] = Field(min_length=1)
    parts: list[str] = Field(min_length=1)


class ReingestOutcomePayload(BaseModel):
    """What one part did to one document. ``status`` is ok | skipped | error, and a skip always
    carries its reason in ``detail`` — 'nothing happened' is the failure this feature cannot
    afford."""

    document_id: str
    filename: str
    part: str
    status: str
    detail: str


class ChunkContextPayload(BaseModel):
    """Where a cited chunk sits in its source (ROADMAP 19).

    Mirrors ``doc_assistant.library.ChunkContext``. `page` is often null and that is not a gap in
    this payload: the parent-child path — which is what a chat citation comes from — does not
    record a page on the parent. The character position is what is always available, so the client
    leads with that and shows a page only when there is one.
    """

    document_id: str
    filename: str
    text: str
    before: str
    after: str
    char_start: int
    char_end: int
    doc_chars: int
    page: int | None
    at_document_start: bool
    at_document_end: bool

    @classmethod
    def from_context(cls, c: ChunkContext) -> ChunkContextPayload:
        return cls(
            document_id=c.document_id,
            filename=c.filename,
            text=c.text,
            before=c.before,
            after=c.after,
            char_start=c.char_start,
            char_end=c.char_end,
            doc_chars=c.doc_chars,
            page=c.page,
            at_document_start=c.at_document_start,
            at_document_end=c.at_document_end,
        )
