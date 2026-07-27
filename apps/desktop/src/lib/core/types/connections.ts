// TypeScript mirror of the desktop-API payloads (apps/api/models/connections.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Document connections (ADR-027 D1, E4).
// Mirrors apps/api/models/connections.py.

// Document connections (ADR-027 D1 — E4 exploration surface). GET
// /api/library/documents/{id}/connections. Mirrors apps/api/models/connections.py::DocConnectionsPayload.
// external_refs is capped server-side; external_total is the full titled count so the panel
// can say "showing N of M" honestly.
export interface RelatedDoc {
  document_id: string
  filename: string
  title: string | null
  score: number
}
export interface CorpusCitation {
  document_id: string
  filename: string | null
  title: string | null
  year: number | null
}
export interface CitedByDoc {
  document_id: string
  filename: string
  n_citations: number
}
export interface ExternalRef {
  title: string
  authors: string | null
  year: number | null
  doi: string | null
}
export interface DocConnections {
  related: RelatedDoc[]
  cites: CorpusCitation[]
  cited_by: CitedByDoc[]
  external_refs: ExternalRef[]
  external_total: number
}
