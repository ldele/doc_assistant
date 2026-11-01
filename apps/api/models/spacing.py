"""Letter-spacing wire models — the document page's block and Settings' ignore list (ADR-056).

One document's letter-spaced runs, as its last extraction recorded them, with what the user's
ignore-list entries now ask for; and the ignore list itself, by document. Mirrors
`doc_assistant.library.spacing`; the TypeScript side is `core/types/spacing.ts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from doc_assistant.library import DocumentSpacing, IgnoreListEntry


class SpacingRowPayload(BaseModel):
    """One letter-spaced run of a document, however many pages repeat it.

    ``state`` is what extraction wrote: ``closed`` (the rule closed it), ``kept`` (left as
    extracted by the ignore list), ``yours`` (the user's spacing), ``maths`` (left alone by the
    rule), or ``unknown`` for an entry that names no recorded run. ``pending`` is what the ignore
    list asks for when that differs from ``state``: it applies when the text is re-run.
    ``written`` is the text now in the document; ``rule`` is how the rule closes the run.
    ``crop`` is the box (x0, y0, x1, y1, points) to render beside it, on ``pages[0]``."""

    spaced: str
    rule: str
    written: str
    kind: str
    label: str
    state: str
    pending: str | None
    chosen: str | None
    pages: list[int]
    crop: list[float] | None


class DocumentSpacingPayload(BaseModel):
    """A document's letter-spacing block. ``pageable`` False: not a PDF, nothing to show."""

    document_id: str
    filename: str
    pageable: bool
    keep_all: bool
    rows: list[SpacingRowPayload]

    @classmethod
    def from_view(cls, v: DocumentSpacing) -> DocumentSpacingPayload:
        return cls(
            document_id=v.document_id,
            filename=v.filename,
            pageable=v.pageable,
            keep_all=v.keep_all,
            rows=[SpacingRowPayload(**r.__dict__) for r in v.rows],
        )


class SpacingRunRequest(BaseModel):
    """One run of a document, named by its spaced form (as ``SpacingRowPayload.spaced``)."""

    spaced: str


class SpacingEditRequest(BaseModel):
    """One run spaced the user's way; ``chosen`` may only add or remove spaces."""

    spaced: str
    chosen: str


class SpacingKeepAllRequest(BaseModel):
    """Keep every run of a document as extracted (True), or stop doing so."""

    keep: bool


class IgnoreListEntryPayload(BaseModel):
    """One document's entries. ``document_id`` is null when no library document matches."""

    file: str
    sha256: str
    size: int
    document_id: str | None
    keep_all: bool
    kept: list[str]
    spacing: dict[str, str]

    @classmethod
    def from_entry(cls, e: IgnoreListEntry) -> IgnoreListEntryPayload:
        return cls(**e.__dict__)


class IgnoreListPayload(BaseModel):
    entries: list[IgnoreListEntryPayload]


class IgnoreListRemoveRequest(BaseModel):
    """Remove one document's entries (its PDF's ``sha256``), or every entry when null."""

    sha256: str | None = None


class IgnoreListRemovedPayload(BaseModel):
    """What was removed: ``previous`` is the whole list as it was, for Undo to send back."""

    removed: int
    documents: list[str]
    previous: dict[str, Any]


class IgnoreListRestoreRequest(BaseModel):
    previous: dict[str, Any]
