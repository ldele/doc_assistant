// TypeScript mirror of the desktop-API payloads (apps/api/models/chat.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Chat-turn wire types — the answer path.
// Mirrors apps/api/models/chat.py.

// ADR-027 D3 — one source's always-on epistemic assessment. Mirrors
// apps/api/models/chat.py::SourceEpistemicsPayload. `coverage` null = "not assessed".
export interface SourceEpistemics {
  coverage: 'corroborated' | 'unique' | 'contested' | null
  superseded: boolean
  n_claims: number
  year: number | null
}
// ADR-027 D3 — strip-level freshness. Mirrors apps/api/models/chat.py::SourceEvalSummaryPayload.
export interface SourceEvalSummary {
  graph_version: string | null
  stale: boolean
}
export interface SourceView {
  n: number
  citation: string
  excerpt: string
  figure_id: string | null
  chunk_key: string | null
  markers: string[]
  // ADR-027 D3 — always-on per-source evaluation + the rerank score (strip signals).
  reranker_score: number
  evaluation: SourceEpistemics | null
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
  // ADR-025 F2 — set when the turn searched only one folder; null = the whole library.
  // Whenever this is set the UI MUST say so: an answer drawn from a subset that doesn't
  // announce it is the failure the folders feature exists to prevent.
  scope: TurnScope | null
  // ADR-027 D3 — strip-level freshness for the always-on source-evaluation strip (per-source
  // evaluation rides on each source). null = no epistemics sidecar / 0-doc → no strip.
  source_eval: SourceEvalSummary | null
}
// The retrieval scope a turn ran under. `folder_name` is null when the folder was deleted
// before the turn ran — the turn then searched nothing rather than everything.
// Mirrors apps/api/models/chat.py::ScopePayload.
export interface TurnScope {
  folder_id: string
  folder_name: string | null
  doc_count: number
}
export type Decision = 'accepted' | 'rejected' | 'edited'
// Session-scoped, non-persistent RAG-sandbox overrides (ADR-010, + the U1b niche-knob
// amendment). `undefined`/`null` (a field or the whole object) = use the locked default.
// Mirrors apps/api/models/chat.py::RagOverrides.
export interface RagOverrides {
  top_k?: number | null
  synthesis_mode?: 'ai' | 'human' | null
  use_multi_query?: boolean | null
  epistemics_markers_enabled?: boolean | null
  reviewer_evidence_chars?: number | null
}
