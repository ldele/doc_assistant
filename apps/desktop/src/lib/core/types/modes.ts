// TypeScript mirror of the desktop-API payloads (apps/api/models/modes.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Chat modes (A7, docs/specs/chat-modes.md). GET /api/modes lists them; POST, PATCH and DELETE
// /api/modes/{id} change a person's own. A conversation's mode is set by PATCH
// /api/conversations/{sid} with `mode_id` ("" = the default). The composer's picker is A8's.

export interface ChatMode {
  id: string
  name: string
  framing: string
  built_in: boolean
  // Built in: gated on the eval panel. A person's own: not evaluated.
  evaluated: boolean
}
// A save is never refused for its meaning: flagged text comes back with warnings.
export interface ChatModeSaved {
  mode: ChatMode
  warnings: string[]
}
export interface ChatModeDeleted {
  conversations: number
  message: string
}
