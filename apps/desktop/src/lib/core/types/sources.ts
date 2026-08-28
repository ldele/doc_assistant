// TypeScript mirror of the desktop-API payloads (apps/api/models/sources.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Selective ingestion — files on disk, NOT the citation sources of an answer
// (those are SourceView in ./chat). Mirrors apps/api/models/sources.py.

// GET /api/ingest/status and the POST /api/ingest 202 body. Mirrors _IngestStatus.
export interface IngestStatus {
  state: 'idle' | 'running' | 'done' | 'error'
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
