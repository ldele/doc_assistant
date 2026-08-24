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
    from doc_assistant.library.add import FileVerdict


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


class InspectRequest(BaseModel):
    """POST /api/documents/inspect body (AD2). Absolute paths from the drop or the picker.

    Directories are allowed and expand recursively server-side — the client never walks a folder,
    because the recursion rule belongs with `registry.scan_sources`, not in two places.
    """

    paths: list[str]


class FileVerdictPayload(BaseModel):
    """What the review sheet renders for one candidate. Mirrors ``library.add.FileVerdict``.

    ``advisory`` is passed through verbatim from ``get_format_status`` for unsupported files, so
    the UI never rewrites the sentence that names the conversion target.
    """

    path: str
    name: str
    verdict: str
    size: int | None
    sha256: str | None
    advisory: str | None
    duplicate_of: str | None
    selected_by_default: bool

    @classmethod
    def from_verdict(cls, v: FileVerdict) -> FileVerdictPayload:
        return cls(
            path=v.path,
            name=v.name,
            verdict=v.verdict,
            size=v.size,
            sha256=v.sha256,
            advisory=v.advisory,
            duplicate_of=v.duplicate_of,
            selected_by_default=v.selected_by_default,
        )


class InspectResponse(BaseModel):
    """Verdicts plus the counts the sheet's header states.

    Already sorted: every non-``add`` verdict precedes every ``add`` (grill branch 7), so a
    paginated first page always carries the warnings, duplicates and unsupported files, and
    "and N more" only ever means clean ones.
    """

    files: list[FileVerdictPayload]
    counts: dict[str, int]


class AddRequest(BaseModel):
    """POST /api/documents/add body (AD3).

    ``mode`` is ADR-046's placement choice. ``reference`` is decided but unbuilt (AD3b) and is
    refused with a 501 rather than quietly copying — a silent fallback would put the user's files
    somewhere they did not choose.
    """

    paths: list[str]
    mode: str = "copy"


class AddOutcomePayload(BaseModel):
    """What happened to one file. Per-file, never a batch total."""

    path: str
    name: str
    ok: bool
    rel_path: str | None
    error: str | None


class AddResultPayload(BaseModel):
    """Applied, failed, and — the honest half — never attempted.

    The run stops at the first failure (grill branch 6), so files after it were not skipped by
    choice; naming them separately is what lets the UI offer *Keep the N* / *Undo all* truthfully.
    """

    added: list[AddOutcomePayload]
    failed: AddOutcomePayload | None
    not_attempted: list[str]
    stopped_early: bool


class UndoAddRequest(BaseModel):
    """POST /api/documents/undo-add body — the ``rel_path``s a just-completed add reported."""

    rel_paths: list[str]
