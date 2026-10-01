// TypeScript mirror of the desktop-API payloads (apps/api/models/keywords.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Tag families + zero-LLM detection proposals.
// Mirrors apps/api/models/keywords.py.

// Tag families (feature-tag-families.md, PR-1). A family is a curated Concept whose aliases are
// member Keyword names (ADR-015); `doc_count` is the union of docs carrying any member keyword.
// Mirrors apps/api/models/keywords.py::KeywordFamilyPayload.
export interface KeywordFamily {
  id: string
  canonical: string
  aliases: string[]
  doc_count: number
  // ADR-018 curation: whether this family's concept is part of the *graph* vocabulary. Boolean on
  // the wire though the column is nullable — a NULL reads as false server-side, because opt-in is
  // what keeps the families feature from re-flooding the graph as it grows. Since ADR-054 it is
  // also what makes the row a *concept* (taken on) rather than a *term*.
  graph_include: boolean
  // ADR-054: the members marked broad (counted beside presence) and exact (always mean the
  // concept). A member in neither list is unclassified, which reads as exact.
  broad: string[]
  exact: string[]
  // The canonical as the library writes it when it writes it in a case — shown instead of it.
  written?: string | null
}
/** A member's mark (ADR-054). `null` clears it; an unclassified member counts as exact. */
export type MemberBreadth = 'exact' | 'broad' | null
// What deleting one family would remove. Read-only — the confirmation states it before the delete.
// Mirrors apps/api/models/keywords.py::KeywordFamilyDeletionPayload.
export interface KeywordFamilyDeletion {
  id: string
  canonical: string
  written?: string | null
  is_concept: boolean
  aliases: number
  has_definition: boolean
  definition_candidates: number
  placements: number
  triage: number
  presence_documents: number
}
// Detection (PR-2). A zero-LLM proposal — nothing has been written; accepting one calls the
// create-family API above. Mirrors apps/api/models/keywords.py::KeywordFamilyProposalPayload.
export interface KeywordFamilyProposal {
  canonical: string
  members: string[]
  tier: 'morphological' | 'embedding'
  confidence: number
}
