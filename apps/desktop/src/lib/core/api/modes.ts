// Thin `modes` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/modes.py and apps/api/models/modes.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, apiFetch } from './_base'
import type { ChatMode } from '../types'

/** Every chat mode: the built-in ones first, then a person's own (A7). */
export async function listModes(): Promise<ChatMode[]> {
  const r = await apiFetch(`${API_BASE}/api/modes`)
  if (!r.ok) throw new Error(`modes failed: ${r.status}`)
  return (await r.json()) as ChatMode[]
}
