"""Reference-list wire models — the Library document view's References block.

One document's bibliography as the paper carries it: every extracted reference in one list,
with the ones already in the library carrying the ``document_id`` that makes them a link.
Distinct from ``connections`` (semantic neighbours + incoming citations) on purpose — that
bundle describes the document's neighbourhood, this one describes what the paper cites.

The list is capped server-side; ``total``/``shown`` carry the honest counts, and the cap
never drops a reference the user owns (see ``library.citations.document_references``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from doc_assistant.library import DocumentReferences


class ReferencePayload(BaseModel):
    """One extracted reference.

    ``document_id`` set ⇒ the reference resolved to a document in the library and the UI
    links it. ``title``/``authors``/``year``/``doi`` are regex-extraction output parsed from
    ``raw_text``, not metadata the library vouches for — ``library_title`` is the owned
    document's own title and is the label to trust when the two disagree.
    """

    raw_text: str | None
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    document_id: str | None
    filename: str | None
    library_title: str | None


class DocReferencesPayload(BaseModel):
    """A document's reference list + the counts that keep a capped list honest.

    ``in_library`` is counted over **all** references, not just the shown ones, so the
    header's "N of your documents" claim stays true under the cap.
    """

    references: list[ReferencePayload]
    total: int
    in_library: int
    shown: int

    @classmethod
    def from_view(cls, v: DocumentReferences) -> DocReferencesPayload:
        return cls(
            references=[
                ReferencePayload(
                    raw_text=r.raw_text,
                    title=r.title,
                    authors=r.authors,
                    year=r.year,
                    doi=r.doi,
                    document_id=r.target_document_id,
                    filename=r.target_filename,
                    library_title=r.library_title,
                )
                for r in v.references
            ],
            total=v.total,
            in_library=v.in_library,
            shown=v.shown,
        )
