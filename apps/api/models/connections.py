"""Document-connections wire models (ADR-027 D1 — ROADMAP E4 exploration surface).

One document's neighbourhood: semantic neighbours, resolved in-corpus citation edges in both
directions, and the extracted-but-unresolved external references. ``external_refs`` is capped
server-side and ``external_total`` carries the full count, so the panel says "showing N of M"
rather than truncating silently.
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


class CorpusCitationPayload(BaseModel):
    """One resolved in-corpus citation edge (subject → target). ``title``/``year`` come from
    the parsed citation string, not the target's own metadata."""

    document_id: str
    filename: str | None
    title: str | None
    year: int | None


class CitedByPayload(BaseModel):
    """One in-corpus document citing the subject (deduped; mirrors ``library.CitedByDoc``)."""

    document_id: str
    filename: str
    n_citations: int


class ExternalRefPayload(BaseModel):
    """One extracted-but-unresolved reference (not in the library). Regex-parsed from the
    paper's bibliography — titles/years are extraction output, shown as such."""

    title: str
    authors: str | None
    year: int | None
    doi: str | None


class DocConnectionsPayload(BaseModel):
    """A document's exploration bundle (E4): related papers + citation edges + external refs.
    ``external_refs`` is capped server-side; ``external_total`` is the full titled count so
    the panel can say "showing N of M" (no silent truncation)."""

    related: list[RelatedDocPayload]
    cites: list[CorpusCitationPayload]
    cited_by: list[CitedByPayload]
    external_refs: list[ExternalRefPayload]
    external_total: int

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
            cites=[
                CorpusCitationPayload(
                    document_id=str(c.target_document_id),
                    filename=c.target_filename,
                    title=c.target_title,
                    year=c.target_year,
                )
                for c in b.cites
            ],
            cited_by=[
                CitedByPayload(
                    document_id=d.document_id,
                    filename=d.filename,
                    n_citations=d.n_citations,
                )
                for d in b.cited_by
            ],
            external_refs=[
                ExternalRefPayload(
                    title=str(e.target_title),
                    authors=e.target_authors,
                    year=e.target_year,
                    doi=e.target_doi,
                )
                for e in b.external_refs
            ],
            external_total=b.external_total,
        )
