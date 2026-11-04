// TypeScript mirror of the desktop-API payloads (apps/api/models/sources.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Selective ingestion — files on disk, NOT the citation sources of an answer
// (those are SourceView in ./chat). Mirrors apps/api/models/sources.py.

// GET /api/ingest/status and the POST /api/ingest 202 body. Mirrors _IngestStatus.
export interface IngestStatus {
  /** `stopped` ends a run like `done` does (what finished is live); it says the person stopped it. */
  state: 'idle' | 'running' | 'done' | 'stopped' | 'error'
  source_dir: string | null
  added: number
  skipped: number
  errors: number
  message: string | null
  /**
   * Position, written per document while the run is in flight. `total` is known before the first
   * document, so this is a real fraction, not a spinner dressed up as one.
   *
   * Deliberately separate from `added`/`skipped`/`errors`, which stay **end-of-run outcomes** and
   * are 0 for the whole duration. Never render `added` as live progress: a document counted at
   * position 3 may still fail at position 4.
   */
  total: number
  done: number
  /** The file being indexed right now, or null when nothing is — before the first, after the last. */
  current: string | null
  /**
   * What pause or stop has been asked of a running run (ROADMAP 31). `pausing` and `stopping` wait
   * for the document in flight; `paused` means the run is waiting between two documents. Always
   * `none` once the run has ended.
   */
  control: IngestControl
  /** Each state's count of files for the current or last run, or null before any run this session. */
  counts: IngestCounts | null
}

/** `finishing`: the loop is over and what finished is being made searchable; nothing can be
 *  paused or stopped any more. A stopped run stays `stopping` through that time. */
export type IngestControl = 'none' | 'pausing' | 'paused' | 'stopping' | 'finishing'

/** One file of a run. Mirrors apps/api/models/sources.py::IngestFilePayload. */
export type IngestFileState =
  | 'waiting'
  | 'reading'
  | 'added'
  | 'unchanged'
  | 'failed'
  | 'not_started'

export interface IngestCounts {
  waiting: number
  reading: number
  added: number
  unchanged: number
  failed: number
  not_started: number
  total: number
}

/** The indexing screen's filters. `finished` is a file now in the library, added or unchanged. */
export type IngestGroup = 'all' | 'in_progress' | 'waiting' | 'not_started' | 'finished' | 'failed'

export interface IngestFile {
  path: string
  name: string
  state: IngestFileState
  /** What happened, in a sentence: a failure's reason, "412 passages", "Already in your library". */
  detail: string | null
}

/**
 * GET /api/ingest/plan: what a Re-index of the library folder would read. Mirrors
 * apps/api/models/sources.py::IngestPlanPayload. The seconds are null until this computer has
 * indexed a document: the estimate is measured, never assumed.
 */
export interface IngestPlan {
  new: number
  new_bytes: number
  changed: number
  changed_bytes: number
  unchanged: number
  excluded: number
  seconds_all: number | null
  seconds_new: number | null
  seconds_per_document: number | null
  measured_documents: number
}

/** GET /api/ingest/files. Mirrors apps/api/models/sources.py::IngestFilesPayload. */
export interface IngestFilesPage {
  /** How many files the asked-for group holds. */
  total: number
  offset: number
  files: IngestFile[]
}
// Selective ingestion (feature-selective-ingestion.md, S2). GET /api/sources lists every file
// under the source dir with a derived ingest status; PATCH /api/sources sets `excluded`; POST
// /api/ingest {paths} ingests a selection. Mirrors apps/api/models/sources.py::SourceFilePayload.
// `doc_type` is always null in v1 (the backend's dormant column).
export interface SourceFile {
  rel_path: string
  format: string
  size: number
  mtime: number
  status: 'new' | 'changed' | 'ingested' | 'missing'
  excluded: boolean
  doc_type: string | null
}
