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
  // A2 — the kinds of text addressed to an AI the passage carries; [] when none matched.
  // Mirrors apps/api/models/chat.py::SourceViewPayload.addressed_to_ai.
  addressed_to_ai: string[]
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
  /** The cost line for a provider Provenote has no price for (ADR-060), e.g. "Billed by Groq;
   *  Provenote does not know its price." `null` on Claude (priced) and on a local model. */
  billing: string | null
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
  // A2 — the web addresses in this turn's passages: an answer link outside them shows as text.
  source_links: string[]
  // A8a (G2) — set only on a `/mode <name>` reply: `continue` opens *Continue in a new chat*
  // prefilled with `mode_id`; `set` sets the start screen's mode chip.
  mode_offer?: ModeOffer | null
  // A8b — every claim of the answer with its marker and cited passages (G4); a records answer's
  // lookup; what the answer carried (the record's `context_json`), its mode and prompt version.
  claims?: TurnClaim[]
  records?: RecordsView | null
  context?: TurnContext | null
  chat_mode?: string | null
  prompt_version?: string | null
}
// A8b — one claim of an answer. `marker`: `ok` cites a well-ranked passage, `weak` cites one that
// ranked low (or has no score), `unsupported` cites nothing. Mirrors TurnClaimPayload.
export interface TurnClaim {
  claim_id: string
  n: number
  text: string
  marker: string
  sources: number[]
  decision: string
  edited_text: string | null
}
// A8b — a rule-answered turn's lookup (A6). Mirrors RecordsPayload.
export interface RecordsView {
  lookup: string
  args: Record<string, unknown>
  rows: number
  items: { kind: 'document' | 'folder' | 'concept'; id: string; label: string }[]
  more: number
  via: string
  asked: boolean
}
// A8b — the reviewer's recorded verdict (a reopened chat). Mirrors ReviewPayload.
export interface ReviewView {
  reviewer_kind: string
  faithfulness: number | null
  citation_density: number | null
  hedging_adequacy: number | null
  unsupported_claims: number | null
  failure_tag: string | null
  error: string | null
}
// A5's record of what a turn's calls carried (`context.ContextRecord`), plus A8a's carried
// exchanges. Every part may be absent on an older record.
export interface TurnContext {
  window: number
  window_source: 'set' | 'server' | 'documented' | 'floor'
  rewrite?: {
    estimated_tokens: number
    history_used: number
    history_dropped: number
    answer_shortened: boolean
    skipped: boolean
  } | null
  answer?: {
    estimated_tokens: number
    passages_used: number
    passages_dropped: number
    top_passage_clipped: boolean
  } | null
  carried?: { exchanges: number; kept: number; dropped: number } | null
}
export interface ModeOffer {
  action: 'continue' | 'set'
  mode_id: string
}
// The retrieval scope a turn ran under: several folders searched as one set, or one concept's
// documents (A7b). A folder's `folder_name` is null when it was unknown or deleted; its
// `doc_count` is its own count, 0 when it added nothing. `doc_count` on the scope is the union.
// Mirrors apps/api/models/chat.py::ScopePayload.
export interface ScopeFolder {
  folder_id: string
  folder_name: string | null
  doc_count: number
}
export interface ScopeConcept {
  concept_id: string
  name: string
  graph_stale: boolean
}
export interface TurnScope {
  folders: ScopeFolder[]
  concept: ScopeConcept | null
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
