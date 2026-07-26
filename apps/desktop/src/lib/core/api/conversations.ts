// Thin `conversations` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/conversations.py and apps/api/models/conversations.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE } from './_base'
import type {
  ConversationDetail,
  ConversationSummary,
  TurnResult,
} from '../types'

/** List past conversations for the history sidebar (feature-conversation-history.md). */
export async function listConversations(): Promise<ConversationSummary[]> {
  const r = await fetch(`${API_BASE}/api/conversations`)
  if (!r.ok) throw new Error(`conversations failed: ${r.status}`)
  return (await r.json()) as ConversationSummary[]
}
/** Rehydrate one conversation as a read-only transcript. */
export async function getConversation(sessionId: string): Promise<ConversationDetail> {
  const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}`)
  if (!r.ok) throw new Error(`conversation failed: ${r.status}`)
  return (await r.json()) as ConversationDetail
}
/** Set a conversation's management flags (pin / archive / soft-delete). Only the fields passed
 *  change. `deleted: true` hides it (records retained); `deleted: false` restores it. */
export async function updateConversationMeta(
  sessionId: string,
  patch: { pinned?: boolean; archived?: boolean; deleted?: boolean; title?: string },
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!r.ok) throw new Error(`update conversation failed: ${r.status}`)
}
export async function exportConversation(sessionId: string, dev: boolean): Promise<void> {
  const r = await fetch(`${API_BASE}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, dev }),
  })
  if (!r.ok) throw new Error(`export failed: ${r.status}`)
  const blob = await r.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `doc_assistant-${sessionId}-${dev ? 'debug' : 'transcript'}.md`
  a.click()
  URL.revokeObjectURL(url)
}

export type { TurnResult }
