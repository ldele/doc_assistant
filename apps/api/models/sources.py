"""Selective-ingestion wire models (feature-selective-ingestion.md, S1/S2).

The source-file registry — what is on disk, what is indexed, what the user has excluded — plus
the ingest trigger body. Distinct from the *citation* sources in ``chat``: these are files in
the watched folder, not retrieved passages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from doc_assistant.ingest.registry import SourceView as RegistrySourceView


class IngestRequest(BaseModel):
    """Optional POST /api/ingest body. Absent / null ``paths`` = ingest the whole source dir
    (minus standing exclusions); a list = ingest exactly that selection (overriding exclusions)."""

    paths: list[str] | None = None


class SourcePatch(BaseModel):
    """PATCH /api/sources body. v1 sets ``excluded`` only (``doc_type`` is the dormant column)."""

    rel_path: str
    excluded: bool | None = None


class SourceFilePayload(BaseModel):
    """One selective-ingestion registry row — mirrors ``ingest.registry.SourceView``.

    Named ``SourceFile`` (not ``SourceView``) to avoid colliding with the citation-source
    ``SourceView``. ``doc_type`` is always ``null`` in v1 (the dormant column).
    """

    rel_path: str
    format: str
    size: int
    mtime: float
    status: str
    excluded: bool
    doc_type: str | None

    @classmethod
    def from_view(cls, v: RegistrySourceView) -> SourceFilePayload:
        return cls(
            rel_path=v.rel_path,
            format=v.format,
            size=v.size,
            mtime=v.mtime,
            status=v.status,
            excluded=v.excluded,
            doc_type=v.doc_type,
        )
