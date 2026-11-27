// TypeScript mirror of the desktop-API payloads (apps/api/models/extraction_record.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// What indexing found in a document (0.9.0, docs/specs/ingestion-record.md):
// GET /api/library/documents/{id}/extraction-record.

/** `not_run`: no recorded run looked for it (a document indexed before 0.9.0). */
export type PassState = 'found' | 'partial' | 'none_found' | 'skipped' | 'error' | 'not_run'

export interface RecordStates {
  references: PassState
  figures: PassState
  tables: PassState
}

/** One warning, read from the record by the backend; `id` is stable, `text` is shown. */
export interface RecordWarning {
  id: string
  text: string
}

export interface TextSection {
  pages: number
  chars: number
  empty_pages: number[]
}

export interface MetadataSection {
  title: boolean
  authors: boolean
  year: boolean
  doi: boolean
}

/** Counts are null for a section the record never filled; `reason` then says why. */
export interface ReferencesSection {
  heading: boolean | null
  entries: number | null
  without_year: number | null
  matched: number | null
  notes: string[]
  seconds: number | null
  reason: string | null
  /** Set when this run did not run the pass: the date of the record it came from. */
  carried_from: string | null
}

export interface FiguresSection {
  found: number | null
  cropped: number | null
  with_caption: number | null
  caption_only: number | null
  seconds: number | null
  reason: string | null
  carried_from: string | null
}

export interface TablesSection {
  extracted: number
  pages_naming_a_table: number[]
  pages_without_one: number[]
  seconds: number | null
  reason: string | null
}

export interface ExtractionRecord {
  created_at: string
  run_kind: 'index' | 'rerun'
  states: RecordStates
  /** It describes an earlier reading of the file. */
  stale: boolean
  text: TextSection
  metadata: MetadataSection
  references: ReferencesSection
  figures: FiguresSection
  tables: TablesSection
  warnings: RecordWarning[]
  /** One line for the queue's finished row: "38 references · 6 figures · 2 tables". */
  summary: string
}

/** `record` is null for a document no run has recorded. */
export interface DocumentExtractionRecord {
  document_id: string
  record: ExtractionRecord | null
}
