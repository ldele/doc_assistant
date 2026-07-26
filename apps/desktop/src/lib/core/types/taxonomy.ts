// TypeScript mirror of the desktop-API payloads (apps/api/models/taxonomy.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Curated field forest (ADR-028). Mirrors apps/api/models/taxonomy.py.

// ============================================================
// Taxonomy (ADR-028 increment 2a) — the curated field forest + coverage.
// Mirrors apps/api/models/taxonomy.py TaxonomyViewPayload / FieldDetailPayload / HierarchyEdgeRequest.
// `*_direct` = attached straight to this field; `*_rollup` = the distinct set under this field or
// any narrower descendant (set-semantics, ADR-028 D6). Every count is 0 until members attach.
// ============================================================
export interface TaxonomyField {
  id: string
  label: string
  parent_ids: string[]
  child_ids: string[]
  n_concepts_direct: number
  n_documents_direct: number
  n_concepts_rollup: number
  n_documents_rollup: number
  // Of the direct members, the share that arrived as an auto-proposal (ADR-028 D8). The
  // direct/rollup counts themselves stay origin-inclusive.
  n_concepts_proposed: number
  n_documents_proposed: number
}
export interface TaxonomyView {
  fields: TaxonomyField[]
  roots: string[]
  n_concepts_total: number
  n_documents_total: number
  n_unassigned_concepts: number
}
// An attachable/attached thing by id+label. Used for the attach picker's vocabulary, which has no
// link and therefore no origin — so it is NOT the same type as an attached member below.
export interface LabelledOption {
  id: string
  label: string
}
// `origin`: 'curated' (a user edit or the ANZSRC seed) | 'proposed' (an auto-fill awaiting
// accept-or-delete). A proposal must never render as the user's own placement (increment 3b).
export interface FieldMember extends LabelledOption {
  origin: 'curated' | 'proposed'
}
export interface FieldDetail {
  id: string
  label: string
  concepts: FieldMember[]
  documents: FieldMember[]
  n_concepts_rollup: number
  n_documents_rollup: number
}
// source --type--> target (narrower -> broader). `in_field` also attaches a concept to a field.
export interface HierarchyEdgeRequest {
  source_id: string
  target_id: string
  type: 'is_a' | 'in_field'
}
