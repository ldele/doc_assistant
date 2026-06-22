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

export type Decision = 'accepted' | 'rejected' | 'edited'
