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
