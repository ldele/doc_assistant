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
}
// Detection (PR-2). A zero-LLM proposal — nothing has been written; accepting one calls the
// create-family API above. Mirrors apps/api/models/keywords.py::KeywordFamilyProposalPayload.
export interface KeywordFamilyProposal {
  canonical: string
  members: string[]
  tier: 'morphological' | 'embedding'
  confidence: number
}
