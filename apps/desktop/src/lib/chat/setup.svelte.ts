// A chat's setup in the window (A8a, docs/specs/chat-interface-pass.md): the start screen's
// choice, the New chat defaults, the live chat's stored setup and the *Continue in a new chat*
// dialog's choice. A `.svelte.ts` rune module: state and the calls that fill it; the rules are
// `setup.ts` (tested). App orchestrates the switches between chats, which touch the chat state.

import {
  getChatSetup,
  getNewChatDefaults,
  listModes,
  saveNewChatDefaults,
} from '../core/api'
import type { ChatMode, ChatSetup, ConversationTurn, NewChatDefaults } from '../core/types'
import { RESEARCH_ASSISTANT, continuePrefill, startFromDefaults, type FolderChoice } from './setup'

export const setup = $state({
  /** Every chat mode, built in first, then a person's own (A7). */
  modes: [] as ChatMode[],
  /** The New chat defaults as last read (W2); null until read. */
  defaults: null as NewChatDefaults | null,

  /** The start screen's choice: what the chat's first turn stores. */
  startModeId: RESEARCH_ASSISTANT,
  startFolders: null as FolderChoice,

  /** The live chat's setup once it has a turn (or once it was continued from another). */
  live: null as ChatSetup | null,
  /** The exchanges the live chat carried from the chat it was continued from, as answered. */
  carried: [] as ConversationTurn[],

  /** The *New chat defaults* window. */
  defaultsOpen: false,
  /** The *Continue in a new chat* dialog: which chat it continues, and its choice. */
  continueFrom: null as string | null,
  continueModeId: RESEARCH_ASSISTANT,
  continueFolders: null as FolderChoice,
  continueCarry: true,
})

export async function loadModes(): Promise<void> {
  try {
    setup.modes = await listModes()
  } catch (e) {
    console.error('modes failed', e)
  }
}

/** Put the start screen back to the defaults (a New chat). */
export function resetStart(): void {
  // Every default folder gone: the backend keeps the gone ids, which search nothing, and the
  // start screen says so; it never falls back to *All documents* (ADR-025 F2).
  const start = startFromDefaults(setup.defaults)
  setup.startModeId = start.modeId
  setup.startFolders = start.folders
}

/** Read the defaults; with `apply`, the start screen opens from them. */
export async function loadDefaults(apply: boolean): Promise<void> {
  try {
    setup.defaults = await getNewChatDefaults()
    if (apply) resetStart()
  } catch (e) {
    console.error('new chat defaults failed', e)
  }
}

export async function saveDefaults(modeId: string, folders: FolderChoice): Promise<boolean> {
  try {
    setup.defaults = await saveNewChatDefaults(modeId, folders)
    return true
  } catch (e) {
    console.error('save new chat defaults failed', e)
    return false
  }
}

/** Read the live chat's setup from the backend (after its first turn, or a switch to it). */
export async function refreshLiveSetup(sessionId: string): Promise<void> {
  try {
    const s = await getChatSetup(sessionId)
    // Only a setup the chat has: a chat with no stored setup keeps the start screen's choice
    // for its next turn, rather than a default the person did not pick.
    if (s.stored || s.derived) setup.live = s
  } catch (e) {
    console.error('chat setup failed', e)
  }
}

/** Open *Continue in a new chat* from `sessionId`, prefilled with its setup, or a mode `/mode`
 *  named (G2). */
export function openContinue(sessionId: string, from: ChatSetup | null, modeId: string | null = null): void {
  const prefill = continuePrefill(from, modeId)
  setup.continueFrom = sessionId
  setup.continueModeId = prefill.modeId
  setup.continueFolders = prefill.folders
  setup.continueCarry = prefill.carry
}

export function closeContinue(): void {
  setup.continueFrom = null
}

/** The name of a mode by id, for words on screen. */
export function modeName(id: string): string {
  return setup.modes.find((m) => m.id === id)?.name ?? (id === RESEARCH_ASSISTANT ? 'Research assistant' : id)
}
