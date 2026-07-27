// Live chat-turn state (the answer path) + the two DOM refs the composer and transcript need.
//
// A `.svelte.ts` rune module. This is the domain step 4 flagged as most coupled, so the split is
// deliberately narrow: **state and the DOM-mechanics helpers live here; orchestration stays in
// App.svelte.** `send` needs the folder scope, `newConversation`/`resumeConversation`/`doExport`
// read and write conversation-view state (`viewing`, `viewedConvo`, `resumedHistory`) — all of
// which span domains, so they stay where that coupling is visible.
//
// The DOM refs are here rather than in the pane because *both* sides need them: `ChatPane` binds
// them, and App's own handlers (`newConversation`, `resumeConversation`) still call
// `chat.taEl?.focus()`. Keeping them module-level is what let the pane be extracted at all.

import type { CompareResult, RagOverrides, TurnResult } from '../core/types'

export interface TurnState {
  id: number
  question: string
  answer: string
  result: TurnResult | null
  streaming: boolean
  error: string | null
}

export function freshSessionId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

export const chat = $state({
  /** The sidebar's "current" marker and the citation-source derivation read this, so a fresh id
   *  from ↻ New must trigger updates. */
  sessionId: freshSessionId(),
  turns: [] as TurnState[],
  input: '',
  sending: false,

  /** ADR-010: the RAG-sandbox overrides for this app session. In-memory only — a fresh launch
   *  always starts from {} (locked defaults), never persisted to disk. */
  overrides: {} as RagOverrides,

  /** A/B-compare (U6): a per-turn retrieval diff. $0 — retrieval only, no answer generation.
   *  An ephemeral card, not a chat turn. */
  compareResult: null as CompareResult | null,
  comparing: false,

  /** Which citation panel is open — keyed by a turn *key* (a live turn's id as a string, or a
   *  past turn's record_id) so a click resolves against the right turn in either mode. */
  activeCitation: null as { turnKey: string; n: number } | null,

  // DOM refs, bound by ChatPane.
  convoEl: null as HTMLElement | null,
  taEl: null as HTMLTextAreaElement | null,
})

/** Monotonic turn id. Deliberately not reactive — nothing renders it; it only mints keys. */
let nextId = 0
export function nextTurnId(): number {
  return nextId++
}
export function resetTurnIds(): void {
  nextId = 0
}

/** Follow a streaming answer only while the reader is already at the bottom, so a long response
 *  never yanks them away from something they scrolled up to re-read. Not reactive: it is read
 *  inside the autoscroll effect, and making it $state would re-trigger that effect on every
 *  scroll event. */
let pinned = true
export function setPinned(v: boolean): void {
  pinned = v
}

// A reader who has scrolled up to re-read is no longer pinned; snap back on once they return to
// the bottom (small slack so it engages just before the exact edge).
export function onConvoScroll(): void {
  if (!chat.convoEl) return
  pinned = chat.convoEl.scrollHeight - chat.convoEl.scrollTop - chat.convoEl.clientHeight < 80
}

// Grow the composer with its content up to a cap, then let it scroll. Reset to the base height
// after a send (measuring `scrollHeight` on empty content would keep the tall size).
export function autogrow(): void {
  if (!chat.taEl) return
  chat.taEl.style.height = 'auto'
  chat.taEl.style.height = `${Math.min(chat.taEl.scrollHeight, 160)}px`
}
export function resetComposer(): void {
  if (chat.taEl) chat.taEl.style.height = 'auto'
}

// Keep the newest content in view as tokens stream in / a turn is added / a chat is opened — but
// only when the reader is pinned to the bottom.
//
// `viewing` is passed as a getter, not a value: it is conversation-view state owned by App, and
// the effect must re-run when it changes (opening a past chat scrolls to the bottom too). Called
// once from App's script — `$effect` needs an active effect context and cannot run at module top
// level.
export function useChatAutoscroll(viewing: () => string | null): void {
  $effect(() => {
    const last = chat.turns[chat.turns.length - 1]
    void last?.answer
    void chat.turns.length
    void viewing()
    if (pinned && chat.convoEl) chat.convoEl.scrollTop = chat.convoEl.scrollHeight
  })
}
