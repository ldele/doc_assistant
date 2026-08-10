"""Document-connections wire models (ADR-027 D1 — ROADMAP E4 exploration surface).

One document's **neighbourhood**: semantic neighbours plus the documents that cite it. The
outgoing side — the paper's own reference list — is served by ``references`` instead
(2026-08-10): a reader wants one bibliography in one place, and duplicating its resolved
half here made the two blocks disagree about what the paper cites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from doc_assistant.library import DocConnections


class RelatedDocPayload(BaseModel):
    """One semantic neighbour (mirrors ``library.SimilarDoc``). ``score`` = cosine."""

    document_id: str
    filename: str
    title: str | None
    score: float


class CitedByPayload(BaseModel):
    """One in-corpus document citing the subject (deduped; mirrors ``library.CitedByDoc``)."""

    document_id: str
    filename: str
    n_citations: int


class DocConnectionsPayload(BaseModel):
    """A document's exploration bundle (E4): semantic neighbours + the documents citing it.

    What this document *cites* is the References block's payload (``references``), not
    this one."""

    related: list[RelatedDocPayload]
    cited_by: list[CitedByPayload]

    @classmethod
    def from_bundle(cls, b: DocConnections) -> DocConnectionsPayload:
        return cls(
            related=[
                RelatedDocPayload(
                    document_id=r.target_document_id,
                    filename=r.target_filename,
                    title=r.target_title,
                    score=r.score,
                )
                for r in b.related
            ],
            cited_by=[
                CitedByPayload(
                    document_id=d.document_id,
                    filename=d.filename,
                    n_citations=d.n_citations,
                )
                for d in b.cited_by
            ],
        )
