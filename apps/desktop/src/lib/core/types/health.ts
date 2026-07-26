// TypeScript mirror of the desktop-API payloads (apps/api/models/health.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Service health. Mirrors the /api/health payload (apps/api/routers/health.py).

export interface Health {
  status: string
  chunk_count: number
  model: string
  embedding_model: string
}
