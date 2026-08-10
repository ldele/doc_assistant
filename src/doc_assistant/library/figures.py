"""Per-document figure browser (Library space L1b — `docs/specs/feature-library-browser.md`).

A document's figures, addressable **separately from its text chunks**: a figure is a different
kind of object (an image with a caption) and belongs in its own panel, not interleaved with
prose the chunk browser already shows.

Read-only, and a sidecar reader — it never writes. The rows come from the 4b `Figure` sidecar
(`scripts/extract_figures`), the descriptions from the 4c VLM pass (`scripts/describe_figures`).

**It reports why a figure is not retrievable, not just that it isn't.** A figure enters retrieval
only once it has a VLM description (`ingest.figure_units` filters on exactly that), so a browser
that listed figures without that distinction would show a page of images and leave the user to
guess which of them the assistant can actually see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from sqlalchemy import select

from doc_assistant.db.models import Document, Figure
from doc_assistant.db.session import session_scope

log = structlog.get_logger(__name__)


@dataclass
class FigureView:
    """One figure as the library shows it."""

    id: str
    page: int
    kind: str | None
    caption: str | None
    description: str | None
    extraction_method: str | None
    has_image: bool
    # True when this figure is a retrievable `chunk_type='figure'` chunk. The rule is
    # `ingest.figure_units`': a description is what makes a figure findable, because the
    # caption alone is already in the prose chunks.
    retrievable: bool
    # Why it is not retrievable, in the user's words rather than the enum's. None when it is.
    not_retrievable_reason: str | None


@dataclass
class DocumentFigureView:
    """A document's header + its figures, with the counts that make the state legible."""

    id: str
    filename: str
    title: str | None
    figures: list[FigureView]
    total: int
    retrievable_count: int
    captioned_count: int
    missing_image_count: int


# `vlm_call_skipped_reason` is an audit enum written by `scripts/describe_figures`. The browser
# is a user surface, so each value is translated once, here — a raw `caption_sufficient` in the
# UI reads as a defect rather than the deliberate cost gate it is.
_SKIP_REASONS = {
    "caption_sufficient": "Caption already describes it — no description needed",
    "budget_exhausted": "Per-document description budget reached",
    "image_missing": "Rendered image is missing on disk",
}
_NO_IMAGE = "Caption only — no image region was detected to describe"
_NOT_DESCRIBED = "Not described yet — run the figure description pass"
# A failed call records `error: <Type>: <message>` (`ingest.figures.describe_error_reason`).
# That string is an *audit* record — up to 400 characters of pydantic — so the panel says what
# the user can act on and leaves the diagnosis in the DB. An unknown enum still passes through
# verbatim: a new reason should be visible, not swallowed.
_ERROR_PREFIX = "error:"
_FAILED = "Description attempt failed — it is retried by the next description pass"


def _reason(description: str | None, skipped: str | None, has_image: bool) -> str | None:
    """Why this figure is absent from retrieval, or None when it is present.

    Order matters: a described figure is retrievable whatever else is recorded, and a
    caption-only row can never be described, so that beats a stale skip reason.
    """
    if description and description.strip():
        return None
    if not has_image:
        return _NO_IMAGE
    if skipped:
        if skipped.startswith(_ERROR_PREFIX):
            return _FAILED
        return _SKIP_REASONS.get(skipped, skipped)
    return _NOT_DESCRIBED


def list_document_figures(document_id: str) -> DocumentFigureView | None:
    """A document's figures in page order, or ``None`` if the document does not exist.

    ``None`` is distinct from a document with zero figures: the first is a bad id (404), the
    second is an ordinary document with no detectable figures (an empty panel). Collapsing them
    would make a deleted document look like a text-only one.
    """
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        rows = session.execute(
            select(Figure)
            .where(Figure.document_id == document_id)
            .order_by(Figure.page, Figure.id)
        ).scalars()

        figures: list[FigureView] = []
        for row in rows:
            has_image = bool(row.image_path) and Path(str(row.image_path)).exists()
            description = row.vlm_description
            reason = _reason(description, row.vlm_call_skipped_reason, has_image)
            figures.append(
                FigureView(
                    id=str(row.id),
                    page=int(row.page),
                    kind=row.kind,
                    caption=row.caption,
                    description=description,
                    extraction_method=row.extraction_method,
                    has_image=has_image,
                    retrievable=reason is None,
                    not_retrievable_reason=reason,
                )
            )

        return DocumentFigureView(
            id=str(doc.id),
            filename=str(doc.filename),
            title=doc.title,
            figures=figures,
            total=len(figures),
            retrievable_count=sum(1 for f in figures if f.retrievable),
            captioned_count=sum(1 for f in figures if f.caption and f.caption.strip()),
            # Counts a row that *should* have an image but whose PNG is gone — a real
            # broken state (the figure dir was cleared, the doc re-hashed), distinct from a
            # caption-only row that never had one.
            missing_image_count=sum(
                1 for f in figures if not f.has_image and f.extraction_method != "caption_only"
            ),
        )
