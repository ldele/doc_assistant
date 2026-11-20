// TypeScript mirror of the desktop-API payloads (apps/api/models/conversations.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Conversation history (read-only replay).
// Mirrors apps/api/models/conversations.py.

import type { TurnScope } from './chat'
import type { ChatMode } from './modes'

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
  // A7b (ADR-066 choice 7) — the documents its answers cited and the folders holding them,
  // derived from the records on every read. The list's controls that use them are A8's.
  labels: ConversationLabels
  // A8a — what the rail's Group reads: the chat's mode, and the folders it searches (empty = the
  // whole library). A folder's `name` is null when it has since been deleted.
  mode_name: string
  folders: { folder_id: string; name: string | null }[]
}
export interface ConversationLabels {
  documents: { document_id: string; title: string }[]
  folders: { folder_id: string; name: string }[]
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
  // A7 — the chat mode the turn's answer prompt used; null when no answer prompt took part.
  chat_mode: string | null
  chat_mode_evaluated: boolean | null
}
export interface ConversationDetail {
  session_id: string
  title: string
  turns: ConversationTurn[]
  // A7 — the chat mode the conversation's next turn uses.
  mode: ChatMode
  // A8a — the chat's setup, for its chat line.
  setup: ChatSetup | null
}
// A8a — a chat's setup: its mode and folders, fixed for the chat. `scope` is the folders resolved
// now (null = the whole library). `stored`: written on the chat; `derived`: a chat from before A8,
// worked out from its last record. Mirrors apps/api/models/conversations.py::ChatSetupPayload.
export interface ChatSetup {
  mode: ChatMode
  folder_ids: string[] | null
  scope: TurnScope | null
  stored: boolean
  derived: boolean
  continued_from: ContinuedFrom | null
  carried_exchanges: number
}
export interface ContinuedFrom {
  session_id: string
  title: string | null
  mode_name: string
  scope: TurnScope | null
}
// POST /api/conversations/{sid}/continue — *Continue in a new chat*.
export interface ContinueRequest {
  session_id: string
  mode_id: string | null
  folder_ids: string[] | null
  carry: boolean
}
export interface ContinueResult {
  session_id: string
  setup: ChatSetup
  carried: ConversationTurn[]
}
// POST /api/conversations/{sid}/resume — *Continue this chat* carries its own exchanges.
export interface ResumeResult {
  session_id: string
  carried_exchanges: number
}
