// Thin `conversations` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/conversations.py and apps/api/models/conversations.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, apiFetch } from './_base'
import type {
  ChatSetup,
  ContinueRequest,
  ContinueResult,
  ConversationDetail,
  ConversationSummary,
  ResumeResult,
  TurnResult,
  TurnScope,
} from '../types'

/** List past conversations for the history sidebar (feature-conversation-history.md). With a
 *  filter, only the chats that cited that document or a document in that folder (A7b, A8a's
 *  *Only chats that cited*). */
export async function listConversations(
  filter: { document_id?: string; folder_id?: string } = {},
): Promise<ConversationSummary[]> {
  const params = new URLSearchParams()
  if (filter.document_id) params.set('document_id', filter.document_id)
  if (filter.folder_id) params.set('folder_id', filter.folder_id)
  const qs = params.toString()
  const r = await apiFetch(`${API_BASE}/api/conversations${qs ? `?${qs}` : ''}`)
  if (!r.ok) throw new Error(`conversations failed: ${r.status}`)
  return (await r.json()) as ConversationSummary[]
}
/** The folders resolved now (A8a): each folder's count and the union, each document once.
 *  `null` = the whole library. What *Search in* counts. */
export async function previewScope(folderIds: string[] | null): Promise<TurnScope | null> {
  const params = new URLSearchParams()
  for (const id of folderIds ?? []) params.append('folder_ids', id)
  const qs = params.toString()
  const r = await apiFetch(`${API_BASE}/api/conversations/scope${qs ? `?${qs}` : ''}`)
  if (!r.ok) throw new Error(`scope failed: ${r.status}`)
  return (await r.json()) as TurnScope | null
}
/** A chat's setup (A8a): stored, worked out from its last record, or the default. */
export async function getChatSetup(sessionId: string): Promise<ChatSetup> {
  const r = await apiFetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}/setup`,
  )
  if (!r.ok) throw new Error(`setup failed: ${r.status}`)
  return (await r.json()) as ChatSetup
}
/** *Continue in a new chat* (A8a): the new chat, set up, with the old one's exchanges carried. */
export async function continueInNewChat(
  fromSessionId: string,
  body: ContinueRequest,
): Promise<ContinueResult> {
  const r = await apiFetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(fromSessionId)}/continue`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
  )
  if (!r.ok) throw new Error(`continue failed: ${r.status}`)
  return (await r.json()) as ContinueResult
}
/** *Continue this chat* (A8a): its memory starts with its own exchanges. */
export async function resumeChat(sessionId: string): Promise<ResumeResult> {
  const r = await apiFetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}/resume`,
    { method: 'POST' },
  )
  if (!r.ok) throw new Error(`resume failed: ${r.status}`)
  return (await r.json()) as ResumeResult
}
/** Rehydrate one conversation as a read-only transcript. */
export async function getConversation(sessionId: string): Promise<ConversationDetail> {
  const r = await apiFetch(`${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}`)
  if (!r.ok) throw new Error(`conversation failed: ${r.status}`)
  return (await r.json()) as ConversationDetail
}
/** Set a conversation's management flags (pin / archive / soft-delete). Only the fields passed
 *  change. `deleted: true` hides it (records retained); `deleted: false` restores it. */
export async function updateConversationMeta(
  sessionId: string,
  patch: { pinned?: boolean; archived?: boolean; deleted?: boolean; title?: string },
): Promise<void> {
  const r = await apiFetch(`${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!r.ok) throw new Error(`update conversation failed: ${r.status}`)
}
export async function exportConversation(sessionId: string, dev: boolean): Promise<void> {
  const r = await apiFetch(`${API_BASE}/api/export`, {
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

/** Soft-delete (or restore) many conversations in one request. Returns how many were touched.
 *  One call, not N: "delete selected" is a single user action and half of it landing is worse
 *  than none. Restore with `deleted: false` — the same route undoes a mis-click. */
export async function bulkUpdateConversations(
  sessionIds: string[],
  deleted: boolean = true,
): Promise<number> {
  const r = await apiFetch(`${API_BASE}/api/conversations/bulk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_ids: sessionIds, deleted }),
  })
  if (!r.ok) throw new Error(`bulk conversation update failed: ${r.status}`)
  return ((await r.json()) as { updated: number }).updated
}

/** Download the whole chat history as one markdown file. Uncapped — unlike the sidebar list,
 *  which stops at ~100. Offered *before* bulk delete: soft-deleted rows stay in the database,
 *  but that is not a restore path a person can act on; a file is. */
export async function exportAllConversations(): Promise<void> {
  const r = await apiFetch(`${API_BASE}/api/conversations/export`, { method: 'POST' })
  if (!r.ok) throw new Error(`history export failed: ${r.status}`)
  const blob = await r.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'provenote-chat-history.md'
  a.click()
  URL.revokeObjectURL(url)
}

export type { TurnResult }
