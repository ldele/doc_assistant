// TypeScript mirror of the desktop-API payloads (apps/api/models.py, PR-M2/M3).
// Keep in sync with the pydantic models — this is the wire contract.

export interface SourceView {
  n: number
  citation: string
  excerpt: string
  figure_id: string | null
  chunk_key: string | null
  markers: string[]
}

export interface ClaimView {
  claim_id: string
  n: number
  text: string
  badge: string
}

export interface UsageView {
  turn_input: number
  turn_output: number
  session_total: number
  cost_usd: number | null
  is_local: boolean
}

export interface TurnResult {
  answer: string
  mode: 'ai' | 'human'
  sources: SourceView[]
  flagged_claims: ClaimView[]
  usage: UsageView
  standalone_query: string
  record_id: string | null
  provenance_card_md: string
  claim_review_md: string
  sources_md: string
  usage_md: string
  citation_note_md: string
  download_path: string | null
}

export interface Health {
  status: string
  chunk_count: number
  model: string
  embedding_model: string
}

// GET/POST /api/settings — the locked engine knobs (read-only) plus the one user-settable
// knob, the source documents folder, and the live corpus size. Mirrors _full_settings() in
// apps/api/main.py.
export interface Settings {
  provider: string
  model: string
  embedding_model: string
  top_k: number
  candidate_k: number
  use_parent_child: boolean
  synthesis_mode: string
  parent_chunk: [number, number]
  child_chunk: [number, number]
  retrieval_weights: { bm25: number; vector: number }
  data_home: string
  source_dir: string
  source_dir_exists: boolean
  supported_formats: string
  chunk_count: number
}

// GET /api/ingest/status and the POST /api/ingest 202 body. Mirrors _IngestStatus.
export interface IngestStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  source_dir: string | null
  added: number
  skipped: number
  errors: number
  message: string | null
}

export type Decision = 'accepted' | 'rejected' | 'edited'
