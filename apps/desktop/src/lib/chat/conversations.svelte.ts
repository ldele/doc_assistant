// Conversation-history list + management (feature-conversation-history.md).
//
// A `.svelte.ts` rune module. Scope is deliberately narrow: the sidebar's **list** and the three
// management flags (pin / archive / rename). Every one is write-then-refetch, and every read is a
// sidecar — a failure keeps the prior list rather than breaking the chat (inform, don't block).
//
// What is NOT here, and why: `viewing` / `viewedConvo` / `resumedHistory`, `openConversation`,
// `resumeConversation` and the soft-delete confirm all read or write live chat state (sessionId,
// turns, activeCitation) or the mobile drawer. They are chat-view orchestration, not conversation
// management, and live in App.svelte so that coupling stays visible in one place.

import { listConversations, updateConversationMeta } from '../core/api'
import type { ConversationSummary } from '../core/types'

export const conversations = $state({
  list: [] as ConversationSummary[],
})

export async function refreshConversations(): Promise<void> {
  try {
    conversations.list = await listConversations()
  } catch {
    // keep the prior list
  }
}

async function setMeta(
  sid: string,
  patch: Parameters<typeof updateConversationMeta>[1],
  what: string,
): Promise<void> {
  try {
    await updateConversationMeta(sid, patch)
    await refreshConversations()
  } catch (e) {
    console.error(`${what} failed`, e)
  }
}

export function pinConversation(sid: string, pinned: boolean): Promise<void> {
  return setMeta(sid, { pinned }, 'pin')
}
export function archiveConversation(sid: string, archived: boolean): Promise<void> {
  return setMeta(sid, { archived }, 'archive')
}
export function renameConversation(sid: string, title: string): Promise<void> {
  return setMeta(sid, { title }, 'rename')
}
