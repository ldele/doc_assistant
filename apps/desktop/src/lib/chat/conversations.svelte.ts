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

  /**
   * Has a fetch ever come back? **"Empty" and "not yet known" are different answers**, and the
   * list starts empty either way — so without this the sidebar told a user with 88 conversations
   * that they had none, for as long as the request took. Set on completion whether the request
   * succeeded or failed: a failure keeps the prior list, and "we looked and could not tell you"
   * is still an answer, where "still looking" is not.
   */
  loaded: false,
})

export async function refreshConversations(): Promise<void> {
  try {
    conversations.list = await listConversations()
  } catch {
    // keep the prior list
  } finally {
    conversations.loaded = true
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

/** Archive (or unarchive) several conversations, then refresh the list once.
 *
 * Not `sids.map(archiveConversation)`: each of those refreshes the whole list on completion, so N
 * archives would fire N list reloads and the sidebar would rerender under the user mid-action. The
 * PATCHes still run concurrently — they are independent rows — and one refresh follows them all.
 * A failure is logged per conversation and does not abandon the rest: a partial archive the user
 * can see and redo beats an all-or-nothing that silently stops halfway.
 */
export async function archiveConversations(sids: string[], archived: boolean): Promise<void> {
  if (sids.length === 0) return
  await Promise.all(
    sids.map(async (sid) => {
      try {
        await updateConversationMeta(sid, { archived })
      } catch (e) {
        console.error('archive failed', sid, e)
      }
    }),
  )
  await refreshConversations()
}
