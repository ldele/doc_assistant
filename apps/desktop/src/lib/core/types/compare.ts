// TypeScript mirror of the desktop-API payloads (apps/api/models/compare.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// A/B-compare retrieval diff (U6). Mirrors apps/api/models/compare.py.

// A/B-compare (feature-ab-compare-sandbox.md, U6 — retrieval diff). POST /api/compare returns both
// retrieved source sets (A = locked defaults, B = session override) + a diff + an honest note.
// Mirrors apps/api/models/compare.py::Compare*Payload.
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
  // ADR-025 F2 — the folder BOTH sides were retrieved under; null = the whole library.
  scope_label: string | null
}
