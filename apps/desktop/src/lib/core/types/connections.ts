// TypeScript mirror of the desktop-API payloads (apps/api/models/connections.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Document connections (ADR-027 D1, E4).
// Mirrors apps/api/models/connections.py.

// Document connections (ADR-027 D1 — E4 exploration surface). GET
// /api/library/documents/{id}/connections. Mirrors apps/api/models/connections.py::DocConnectionsPayload.
//
// This document's NEIGHBOURHOOD — not its bibliography. What the paper cites lives in
// `references.ts` / the References block; keeping the resolved half here too made the two
// blocks disagree about the same document.
export interface RelatedDoc {
  document_id: string
  filename: string
  title: string | null
  score: number
}
export interface CitedByDoc {
  document_id: string
  filename: string
  n_citations: number
}
export interface DocConnections {
  related: RelatedDoc[]
  cited_by: CitedByDoc[]
}
