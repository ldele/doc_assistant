"""Wire models for the library summary (ROADMAP 112): mirrors ``library.summary``.

``core/types/summary.ts`` mirrors these names. Counts only, never a ratio: a collection of 0
documents has nothing to divide by, and the screen says "0" rather than a percentage of nothing.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from apps.api.models._common import _as_utc

if TYPE_CHECKING:
    from doc_assistant.library.summary import CollectionSummary


class IndexingCountsPayload(BaseModel):
    no_text: int
    flagged_marginal: int
    flagged_broken: int
    not_classified: int
    #: ``None`` in a folder's scope: files that are not documents belong to no folder.
    files_not_in_library: int | None
    files_excluded: int | None
    files_seen_at: datetime | None
    #: Failures in this session's last index run; ``None`` when there was none this session.
    last_run_failed: int | None = None


class MetadataCountsPayload(BaseModel):
    missing_title: int
    missing_authors: int
    missing_year: int
    missing_doi: int
    implausible_year: int
    set_by_you: int
    from_a_catalogue: int
    #: Documents with any of title, authors, year, DOI; 0 means nothing has read metadata yet.
    with_any_field: int


class OnRecordPayload(BaseModel):
    on_record: int
    none_on_record: int
    rows: int = 0


class FigureCountsPayload(BaseModel):
    on_record: int
    none_on_record: int
    rows: int
    described: int
    pdfs: int


class SpacingCountsPayload(BaseModel):
    recorded: int
    pdfs: int


class FoundCountsPayload(BaseModel):
    references: OnRecordPayload
    figures: FigureCountsPayload
    letter_spacing: SpacingCountsPayload
    keywords: OnRecordPayload


class TypeCountPayload(BaseModel):
    format: str
    documents: int


class DecadeCountPayload(BaseModel):
    decade: int
    documents: int


class YearCountsPayload(BaseModel):
    by_decade: list[DecadeCountPayload]
    no_year: int
    oldest: int | None
    newest: int | None


class FolderCountsPayload(BaseModel):
    in_a_folder: int | None = None
    in_none: int | None = None
    also_elsewhere: int | None = None


class AddedRangePayload(BaseModel):
    first: datetime | None
    last: datetime | None


class CollectionSummaryPayload(BaseModel):
    """The *Summary* layout of the Library pane, for the library or one folder."""

    scope: Literal["library", "folder"]
    folder_id: str | None
    folder_name: str | None
    counted_at: datetime
    documents: int
    passages: int
    pages: int
    paged_documents: int
    need_attention: int
    indexing: IndexingCountsPayload
    metadata: MetadataCountsPayload
    found: FoundCountsPayload
    types: list[TypeCountPayload]
    years: YearCountsPayload
    folders: FolderCountsPayload
    added: AddedRangePayload

    @classmethod
    def from_summary(
        cls, s: CollectionSummary, *, last_run_failed: int | None = None
    ) -> CollectionSummaryPayload:
        payload = cls.model_validate(asdict(s))
        # Naive DB timestamps are UTC; tag them so a browser does not read them as local time.
        payload.counted_at = _as_utc(payload.counted_at)
        ix = payload.indexing
        ix.last_run_failed = last_run_failed
        if ix.files_seen_at is not None:
            ix.files_seen_at = _as_utc(ix.files_seen_at)
        if payload.added.first is not None:
            payload.added.first = _as_utc(payload.added.first)
        if payload.added.last is not None:
            payload.added.last = _as_utc(payload.added.last)
        return payload
