// Thin `setup` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/setup.py and apps/api/models/setup.py (ADR-034); see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type { ApiKeyResult, SetupState } from '../types/setup'

/** The first-run picture: which providers can answer, what is indexed, what is left to do.
 *  `probe: false` answers from local state only (no Ollama call) — use it when a refresh must
 *  not pay for a network round trip. */
export async function getSetup(probe = true): Promise<SetupState> {
  const r = await fetch(`${API_BASE}/api/setup?probe=${probe ? 'true' : 'false'}`)
  if (!r.ok) throw new Error(await errorDetail(r, 'read setup state'))
  return (await r.json()) as SetupState
}

/** Save an Anthropic API key. The backend verifies it with a free metadata call first, so a 400
 *  here means the API itself rejected the key; a 200 with `verification: 'unreachable'` means it
 *  was stored but could not be checked (offline, or behind a proxy). */
export async function saveAnthropicKey(key: string): Promise<ApiKeyResult> {
  const r = await fetch(`${API_BASE}/api/setup/anthropic-key`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'save API key'))
  return (await r.json()) as ApiKeyResult
}

/** Forget the key saved in the app. A key in `.env` is untouched (the backend does not own it). */
export async function clearAnthropicKey(): Promise<SetupState> {
  const r = await fetch(`${API_BASE}/api/setup/anthropic-key`, { method: 'DELETE' })
  if (!r.ok) throw new Error(await errorDetail(r, 'remove API key'))
  return ((await r.json()) as { setup: SetupState }).setup
}
