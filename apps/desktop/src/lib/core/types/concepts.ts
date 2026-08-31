// TypeScript mirror of the desktop-API payloads (apps/api/models/concepts.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Concept graph + gaps (ADR-017). ONE id space: every id here is a concept UUID.
// Mirrors apps/api/models/concepts.py.

// Concept graph (docs/specs/feature-concept-graph.md, ADR-017; PR-G1 serves it, PR-G2a renders it).
// Mirrors apps/api/models/concepts.py::ConceptGraph*Payload / Gap / staleness — ONE id space: every id here
// is a Concept UUID; `label` rides only on the node (the UUID-vs-label mismatch was KI-15).
export interface ConceptGraphNode {
  id: string
  label: string
  doc_ids: string[]
  degree: number
  community: number
}
// `relation` is the deferred Node-B stance annotation — `null` on every edge until that pass runs,
// so a renderer must not imply agreement/disagreement it does not have. Weights span a narrow
// 2.377–2.949 (nearly flat) — do not lean on them for thickness.
export interface ConceptGraphEdge {
  source: string
  target: string
  provenance: string[]
  weight: number
  n_cooccurrence_chunks: number
  relation: string | null
}
// `id` is POSITIONAL, not identity — it renumbers when the vocabulary changes. Never persist a
// preference against it.
export interface ConceptCommunity {
  id: number
  label: string
  node_ids: string[]
  size: number
}
// A detected corpus gap (ADR-004), anchored to `concept_id`. `status` is the raw row value; a
// user's triage lives in its own override sidecar (ADR-017 C1, PR-G2b), so it is not resolved here.
export type GapKind =
  | 'isolated'
  | 'single_source'
  | 'thin_bridge'
  | 'under_connected'
  | 'unsourced_claim'
  | 'citation_missing'
  | 'suggested_link'
  | 'suggested_concept'
  | 'thin_area'

export interface Gap {
  concept_id: string
  tier: string
  determinism: string
  kind: GapKind
  fact_ids: string[]
  rating: number | null
  status: string
}
export type GapStatus = 'surfaced' | 'promoted' | 'dismissed'
// One gap for the first-class gap list (E5). Mirrors apps/api/models/concepts.py::GapListItemPayload — a
// gap with its concept `label` resolved server-side and the *effective* triage status (a user
// override wins; ADR-017 C1). Distinct from `Gap` (which rides inside the graph payload and joins
// labels by node id).
export interface GapListItem {
  concept_id: string
  label: string
  kind: GapKind
  tier: string
  determinism: string
  fact_ids: string[]
  rating: number | null
  status: GapStatus
}
// The skeleton is a build artifact and the Manage-keywords view writes Concept rows live, so drift
// is structural, not a defect: the UI reports it and offers a rebuild (never auto-rebuilds).
export interface GraphStaleness {
  stale: boolean
  n_concepts_in_db: number
  n_concepts_in_skeleton: number
  added_labels: string[]
  removed_ids: string[]
  /**
   * Documents the graph cites that the library can no longer resolve — the *corpus* half of
   * staleness, where the two above watch the vocabulary. A document whose identity moved leaves a
   * reference nothing can render, and before this the view printed the raw id in the title slot.
   */
  missing_document_ids: string[]
  n_documents_in_skeleton: number
}
export interface ConceptGraph {
  graph_version: string
  nodes: ConceptGraphNode[]
  edges: ConceptGraphEdge[]
  communities: ConceptCommunity[]
  gaps: Gap[]
  staleness: GraphStaleness
}
// `chunk_keys` are ADR-4 composite `"{document_id}:p{parent_index}"` — the navigation payload that
// takes the ego view from a concept down to the chunks that mention it.
export interface ConceptPresence {
  document_id: string
  chunk_keys: string[]
  n_mentions: number
}
// Rebuild is a 202 + poll job (ADR-017 B1), mirroring /api/ingest. `graph_version` is set once the
// worker finishes; `message` carries the error text when `state === 'error'`.
export interface GraphRebuildStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  graph_version: string | null
  message: string | null
}
