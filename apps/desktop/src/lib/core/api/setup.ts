// Thin `setup` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/setup.py and apps/api/models/setup.py (ADR-034, ADR-060); see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { ApiKeyResult, ModelList, ServerResult, ServerUpdate, SetupState } from '../types/setup'

/** The first-run picture: which providers can answer, what is indexed, what is left to do.
 *  `probe: false` answers from local state only (no network) — use it when a refresh must
 *  not pay for a round trip. */
export async function getSetup(probe = true): Promise<SetupState> {
  const r = await apiFetch(`${API_BASE}/api/setup?probe=${probe ? 'true' : 'false'}`)
  if (!r.ok) throw new Error(await errorDetail(r, 'read setup state'))
  return (await r.json()) as SetupState
}

const provider = (id: string): string => `${API_BASE}/api/setup/providers/${encodeURIComponent(id)}`

/** Save a provider's API key. The backend verifies it with a free call first (a model list), so
 *  a 400 here means the server refused the key; a 200 with `verification: 'unreachable'` means it
 *  was stored but could not be checked (offline, or behind a proxy). */
export async function saveProviderKey(id: string, key: string): Promise<ApiKeyResult> {
  const r = await apiFetch(`${provider(id)}/key`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'save API key'))
  return (await r.json()) as ApiKeyResult
}

/** Forget the key saved in the app. A Claude key in `.env` is untouched (the backend does not own it). */
export async function clearProviderKey(id: string): Promise<SetupState> {
  const r = await apiFetch(`${provider(id)}/key`, { method: 'DELETE' })
  if (!r.ok) throw new Error(await errorDetail(r, 'remove API key'))
  return ((await r.json()) as { setup: SetupState }).setup
}

/** Ask the provider's server for its models now. Free; the backend keeps the answer in memory.
 *  A POST: it calls out with the stored key, so it carries the launch token like every write. */
export async function fetchProviderModels(id: string): Promise<ModelList> {
  const r = await apiFetch(`${provider(id)}/models`, { method: 'POST' })
  if (!r.ok) throw new Error(await errorDetail(r, 'list models'))
  return (await r.json()) as ModelList
}

/** *Test connection*: save a server of your own and ask it for its models. A 400 means the
 *  address cannot be used or the server refused the key, and nothing was saved. */
export async function testServer(body: ServerUpdate): Promise<ServerResult> {
  const r = await apiFetch(`${API_BASE}/api/setup/servers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'reach the server'))
  return (await r.json()) as ServerResult
}

/** Forget a server of your own. The one answering now is refused (409). */
export async function removeServer(id: string): Promise<SetupState> {
  const r = await apiFetch(`${API_BASE}/api/setup/servers/${encodeURIComponent(id)}`, { method: 'DELETE' })
  if (!r.ok) throw new Error(await errorDetail(r, 'remove the server'))
  return ((await r.json()) as { setup: SetupState }).setup
}

/** Record or clear the user's word that a server bills nobody. */
export async function setNotBilled(id: string, notBilled: boolean): Promise<SetupState> {
  const r = await apiFetch(`${provider(id)}/billing`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ not_billed: notBilled }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'save who bills'))
  return ((await r.json()) as { setup: SetupState }).setup
}
