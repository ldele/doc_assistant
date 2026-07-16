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

// Conversation history (feature-conversation-history.md). GET /api/conversations returns the
// sidebar list; GET /api/conversations/{sid} rehydrates one chat as a read-only transcript.
// Mirrors apps/api/models.py::Conversation*Payload.
export interface ConversationSummary {
  session_id: string
  title: string
  turn_count: number
  started_at: string // ISO 8601
  last_at: string // ISO 8601
  pinned: boolean
  archived: boolean
}

// A rehydrated citation — degraded vs a live SourceView (no markers/figures; not persisted).
export interface ConversationSource {
  n: number
  citation: string
  excerpt: string
}

export interface ConversationTurn {
  record_id: string
  question: string
  answer: string
  sources: ConversationSource[]
}

export interface ConversationDetail {
  session_id: string
  title: string
  turns: ConversationTurn[]
}

// GET/POST /api/settings — the locked engine knobs (read-only) plus the one user-settable
// knob, the source documents folder, and the live corpus size. Mirrors _full_settings() in
// apps/api/main.py.
export interface Settings {
  // provider/model are the *effective* values (ADR-011, U1c) — the persisted switch if one was
  // made, else the config default. Never stale after a live provider switch.
  provider: string
  model: string
  embedding_model: string
  top_k: number
  candidate_k: number
  use_parent_child: boolean
  synthesis_mode: string
  use_multi_query: boolean
  epistemics_markers_enabled: boolean
  reviewer_evidence_chars: number
  parent_chunk: [number, number]
  child_chunk: [number, number]
  retrieval_weights: { bm25: number; vector: number }
  providers: ProviderOption[]
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

// Session-scoped, non-persistent RAG-sandbox overrides (ADR-010, + the U1b niche-knob
// amendment). `undefined`/`null` (a field or the whole object) = use the locked default.
// Mirrors apps/api/models.py::RagOverrides.
export interface RagOverrides {
  top_k?: number | null
  synthesis_mode?: 'ai' | 'human' | null
  use_multi_query?: boolean | null
  epistemics_markers_enabled?: boolean | null
  reviewer_evidence_chars?: number | null
}

// One entry in the settings view's provider picker (ADR-011, U1c). `available: false` means the
// provider's credential is missing (e.g. no ANTHROPIC_API_KEY) — the UI disables that option.
export interface ProviderOption {
  id: string
  available: boolean
  paid: boolean
}

// Library browser (feature-library-browser.md, L1 — read-only). GET /api/library/documents lists
// ingested docs; GET /api/library/documents/{id} returns its chunks as parent blocks. Mirrors
// apps/api/models.py::Library*Payload. NULL metadata (title/authors/year, health) stays null.
export interface LibraryDocument {
  id: string
  filename: string
  title: string | null // effective: user override ?? auto-extracted (ADR-013)
  authors: string | null // effective
  year: number | null // effective
  customized: boolean // a user metadata override is in force
  format: string
  health: string | null
  chunk_count: number | null
  page_count: number | null
  folders: string[]
  tags: string[]
  keywords: string[]
  added_at: string | null // ISO 8601
}

// Selective ingestion (feature-selective-ingestion.md, S2). GET /api/sources lists every file
// under the source dir with a derived ingest status; PATCH /api/sources sets `excluded`; POST
// /api/ingest {paths} ingests a selection. Mirrors apps/api/models.py::SourceFilePayload.
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

export interface LibraryChild {
  child_index: number
  text: string
  retrievable: boolean
}

export interface LibraryParent {
  parent_index: number
  parent_text: string
  children: LibraryChild[]
}

export interface LibraryDocumentChunks {
  id: string
  filename: string
  format: string
  title: string | null
  authors: string | null
  year: number | null
  chunk_count: number | null
  health: string | null
  parents: LibraryParent[]
  child_count: number
}

// A/B-compare (feature-ab-compare-sandbox.md, U6 — retrieval diff). POST /api/compare returns both
// retrieved source sets (A = locked defaults, B = session override) + a diff + an honest note.
// Mirrors apps/api/models.py::Compare*Payload.
export interface CompareEff {
  top_k: number
  use_multi_query: boolean
}

export interface CompareSource {
  rank: number
  filename: string
  page: number | null
  section: string | null
  score: number
  excerpt: string
  citation: string
  identity: string
}

export interface CompareRow {
  identity: string
  source_a: CompareSource | null
  source_b: CompareSource | null
  status: 'in_both' | 'only_in_a' | 'only_in_b'
  rank_delta: number | null
}

export interface CompareResult {
  query: string
  sources_a: CompareSource[]
  sources_b: CompareSource[]
  rows: CompareRow[]
  eff_a: CompareEff
  eff_b: CompareEff
  note: string
}
