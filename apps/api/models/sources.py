"""Selective-ingestion wire models (feature-selective-ingestion.md, S1/S2).

The source-file registry — what is on disk, what is indexed, what the user has excluded — plus
the ingest trigger body. Distinct from the *citation* sources in ``chat``: these are files in
the watched folder, not retrieved passages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    #: ADR-046 (AD3b) — which root `rel_path` is relative to. Defaults to the library root, so a
    #: client written before multi-root keeps working unchanged.
    root_id: str = "library"


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
    #: ADR-046 (AD3b). `rel_path` is relative to this root and is no longer unique on its own;
    #: `key` is the pair the client should send back for selection or PATCH.
    root_id: str = "library"
    root_kind: str = "library"
    #: False when the root is unreachable right now (unplugged drive, offline share). Its files
    #: read ``missing``, and this is what lets the UI say *why* instead of implying deletion.
    root_available: bool = True
    key: str = ""

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
            root_id=v.root_id,
            root_kind=v.root_kind,
            root_available=v.root_available,
            key=f"{v.root_id}:{v.rel_path}",
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

    ``mode`` is ADR-046's placement choice, and **both modes work since AD3b**. Typed as a
    ``Literal`` so FastAPI rejects anything else with its own 422 before the handler runs — the
    library's `AddMode` is the same two values, so this is one contract rather than a hand-rolled
    check that could drift from it.
    """

    paths: list[str]
    mode: Literal["copy", "reference"] = "copy"
    #: The folder the whole batch belongs under, when the caller knows — an import from a
    #: catalogue that told it where its files live (ADR-049). Absent for a drop or a picker, where
    #: the per-parent rule applies. Ignored for `copy`.
    reference_root: str | None = None


class AddOutcomePayload(BaseModel):
    """What happened to one file. Per-file, never a batch total."""

    path: str
    name: str
    ok: bool
    #: Where it landed, relative to its own root — for display.
    rel_path: str | None
    #: The identifier to send back to `/api/documents/undo-add` (AD3b: `rel_path` alone no
    #: longer identifies a row, because two roots may hold the same relative path).
    key: str | None = None
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
    """POST /api/documents/undo-add body — the ``key``s a just-completed add reported.

    Still named ``rel_paths`` on the wire for compatibility; a bare rel_path is accepted and read
    as the library root, so a pre-AD3b client is unaffected.
    """

    rel_paths: list[str]


class CatalogueScanRequest(BaseModel):
    """POST /api/catalogue/zotero/scan body. Every field optional — the defaults are the ask.

    `data_dir` absent means "look where Zotero puts it", which is right for almost everyone and
    saves a folder picker. `base_dir` is Zotero's Linked Attachment Base Directory: a preference,
    not a database value, so it cannot be discovered and has to be offered.
    """

    data_dir: str | None = None
    base_dir: str | None = None
    include_snapshots: bool = False


class CatalogueScanResponse(BaseModel):
    """What a catalogue holds, before anything is staged.

    `skipped` is a reason -> count map, passed through verbatim: the client shows the reasons
    rather than a total, because "412 found" beside nothing else looks like a broken import when
    the library has 500 entries.
    """

    #: Human name of the catalogue, for the sentence the dialog writes.
    label: str
    #: The folder the files live under — shown so the user can confirm it is the right library.
    root: str
    #: Absolute paths, ready to hand to the existing review sheet.
    paths: list[str]
    found: int
    skipped: dict[str, int]
    #: How many of `paths` the catalogue could describe. The reason to import from Zotero at all.
    with_metadata: int
