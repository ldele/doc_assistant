"""Extraction-record wire models: what indexing found in one document (0.9.0).

Mirrors `doc_assistant.ingest.record` (spec `docs/specs/ingestion-record.md`); the TypeScript side
is `core/types/extractionRecord.ts`. A section's counts are nullable because a section the record
never filled (a pass not run, a non-PDF's tables) carries only its ``reason``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

PassState = Literal["found", "partial", "none_found", "skipped", "error", "not_run"]


class RecordStates(BaseModel):
    references: PassState
    figures: PassState
    tables: PassState


class RecordWarningPayload(BaseModel):
    id: str
    text: str


class TextSection(BaseModel):
    pages: int = 0
    chars: int = 0
    empty_pages: list[int] = []


class MetadataSection(BaseModel):
    title: bool = False
    authors: bool = False
    year: bool = False
    doi: bool = False


class ReferencesSection(BaseModel):
    heading: bool | None = None
    entries: int | None = None
    without_year: int | None = None
    matched: int | None = None
    notes: list[str] = []
    seconds: float | None = None
    reason: str | None = None
    carried_from: str | None = None


class FiguresSection(BaseModel):
    found: int | None = None
    cropped: int | None = None
    with_caption: int | None = None
    caption_only: int | None = None
    seconds: float | None = None
    reason: str | None = None
    carried_from: str | None = None


class TablesSection(BaseModel):
    extracted: int = 0
    pages_naming_a_table: list[int] = []
    pages_without_one: list[int] = []
    seconds: float | None = None
    reason: str | None = None


class ExtractionRecordPayload(BaseModel):
    """The newest record. ``stale``: it describes an earlier reading of the file."""

    created_at: str
    run_kind: Literal["index", "rerun"]
    states: RecordStates
    stale: bool
    text: TextSection
    metadata: MetadataSection
    references: ReferencesSection
    figures: FiguresSection
    tables: TablesSection
    warnings: list[RecordWarningPayload]
    summary: str

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> ExtractionRecordPayload:
        return cls.model_validate(rec)


class DocumentExtractionRecordPayload(BaseModel):
    """``record`` is null for a document no run has recorded (indexed before 0.9.0)."""

    document_id: str
    record: ExtractionRecordPayload | None
