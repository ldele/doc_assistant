// Thin `settings` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/settings.py and apps/api/models/settings.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type {
  Settings,
} from '../types'

export async function getSettings(): Promise<Settings> {
  const r = await fetch(`${API_BASE}/api/settings`)
  if (!r.ok) throw new Error(`settings failed: ${r.status}`)
  return (await r.json()) as Settings
}
/** Persist the source documents folder. 400 carries the backend's reason (e.g. not a directory). */
export async function setSourceDir(sourceDir: string): Promise<Settings> {
  const r = await fetch(`${API_BASE}/api/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_dir: sourceDir }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'save settings'))
  return (await r.json()) as Settings
}
/** Switch the live LLM provider/model (ADR-011, U1c) — takes effect on the next turn, no
 *  restart. 400 carries the backend's reason (unknown provider, or one with no key configured). */
export async function setLlmProvider(provider: string, model: string): Promise<Settings> {
  const r = await fetch(`${API_BASE}/api/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ llm_provider: provider, llm_model: model }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'switch provider'))
  return (await r.json()) as Settings
}
/** Persist the answer-layer epistemics default (ADR-027 D2, E3) — whether contested/superseded
 *  chips may appear on answer sources. Takes effect on the next turn; the RAG-sandbox session
 *  override still wins per-turn. Never affects the always-on source-evaluation strip (D3). */
export async function setMarkersEnabled(enabled: boolean): Promise<Settings> {
  const r = await fetch(`${API_BASE}/api/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ epistemics_markers_enabled: enabled }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'save epistemics setting'))
  return (await r.json()) as Settings
}

// ADR-037 — rebuild the on-disk keyword index and swap it into the live pipeline. Bounded work
// (seconds here, minutes at 10k documents) and non-destructive: the index is derived data the
// next launch would rebuild anyway, so there is no confirmation step. Returns the refreshed
// settings, so the caller needs no second GET.
export async function reindexKeywords(): Promise<Settings> {
  const r = await fetch(`${API_BASE}/api/settings/reindex-keywords`, { method: 'POST' })
  if (!r.ok) throw new Error(await errorDetail(r, 'rebuild the keyword index'))
  return (await r.json()) as Settings
}
