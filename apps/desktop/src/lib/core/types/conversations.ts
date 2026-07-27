// TypeScript mirror of the desktop-API payloads (apps/api/models/conversations.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Conversation history (read-only replay).
// Mirrors apps/api/models/conversations.py.

import type { SourceView, TurnScope } from './chat'

// Conversation history (feature-conversation-history.md). GET /api/conversations returns the
// sidebar list; GET /api/conversations/{sid} rehydrates one chat as a read-only transcript.
// Mirrors apps/api/models/conversations.py::Conversation*Payload.
export interface ConversationSummary {
  session_id: string
  title: string
  turn_count: number
  started_at: string // ISO 8601
  last_at: string // ISO 8601
  pinned: boolean
  archived: boolean
}
// A rehydrated citation — degraded vs a live SourceView (no markers/figures; not persisted).
export interface ConversationSource {
  n: number
  citation: string
  excerpt: string
}
export interface ConversationTurn {
  record_id: string
  question: string
  answer: string
  sources: ConversationSource[]
  // ADR-025 F2 — replayed from the record so a reopened scoped answer still says it was scoped.
  scope: TurnScope | null
}
export interface ConversationDetail {
  session_id: string
  title: string
  turns: ConversationTurn[]
}
