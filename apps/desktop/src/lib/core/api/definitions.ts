// Thin `definitions` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/definitions.py and apps/api/models/definitions.py (ADR-053).
// Every mutation answers with the concept's refreshed candidates, so callers replace their copy.

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { ConceptDefinitions, ConceptSignals, ConceptUsage, VocabularyMatch } from '../types'

const base = (conceptId: string): string =>
  `${API_BASE}/api/concepts/${encodeURIComponent(conceptId)}/definitions`

async function post(url: string, what: string, body: unknown = null): Promise<ConceptDefinitions> {
  const r = await apiFetch(url, {
    method: 'POST',
    headers: body === null ? undefined : { 'Content-Type': 'application/json' },
    body: body === null ? undefined : JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await errorDetail(r, what))
  return (await r.json()) as ConceptDefinitions
}

/** One concept's candidates, the chosen one first. 404 for an unknown id or a taxonomy field. */
export async function getDefinitions(conceptId: string): Promise<ConceptDefinitions> {
  const r = await apiFetch(base(conceptId))
  if (!r.ok) throw new Error(await errorDetail(r, 'definitions'))
  return (await r.json()) as ConceptDefinitions
}
/** How the library uses a concept: a plain sentence from each of the documents that use it most. */
export async function getUsage(conceptId: string): Promise<ConceptUsage> {
  const r = await apiFetch(`${API_BASE}/api/concepts/${encodeURIComponent(conceptId)}/usage`)
  if (!r.ok) throw new Error(await errorDetail(r, 'usage'))
  return (await r.json()) as ConceptUsage
}
/** What the text says about a row's name: forms it gives beside one of its own, a cited author,
 *  and where a label no sentence uses does occur. Read-only; a term has them as a concept does. */
export async function getSignals(conceptId: string): Promise<ConceptSignals> {
  const r = await apiFetch(`${API_BASE}/api/concepts/${encodeURIComponent(conceptId)}/signals`)
  if (!r.ok) throw new Error(await errorDetail(r, 'signals'))
  return (await r.json()) as ConceptSignals
}
/** Save the user's own definition — chosen unless `choose` is false. Replaces nothing. */
export function addDefinition(
  conceptId: string,
  text: string,
  choose: boolean = true,
): Promise<ConceptDefinitions> {
  return post(base(conceptId), 'save definition', { text, choose })
}
/** Look in the library for sentences that define it ($0, a few seconds). Suggestions only. */
export function extractDefinitions(conceptId: string): Promise<ConceptDefinitions> {
  return post(`${base(conceptId)}/extract`, 'look in the library')
}
export function chooseDefinition(conceptId: string, definitionId: string): Promise<ConceptDefinitions> {
  return post(`${base(conceptId)}/${encodeURIComponent(definitionId)}/choose`, 'choose definition')
}
export function dismissDefinition(conceptId: string, definitionId: string): Promise<ConceptDefinitions> {
  return post(`${base(conceptId)}/${encodeURIComponent(definitionId)}/dismiss`, 'dismiss definition')
}
export function restoreDefinition(conceptId: string, definitionId: string): Promise<ConceptDefinitions> {
  return post(`${base(conceptId)}/${encodeURIComponent(definitionId)}/restore`, 'restore definition')
}
/** Take back the most recent choose / dismiss / restore. */
export function undoDefinition(conceptId: string): Promise<ConceptDefinitions> {
  return post(`${base(conceptId)}/undo`, 'undo')
}
/** Concepts whose label contains `q`, across the whole vocabulary. */
export async function searchVocabulary(q: string, limit: number = 20): Promise<VocabularyMatch[]> {
  const params = new URLSearchParams({ q, limit: String(limit) })
  const r = await apiFetch(`${API_BASE}/api/concepts/search?${params}`)
  if (!r.ok) throw new Error(await errorDetail(r, 'concept search'))
  return (await r.json()) as VocabularyMatch[]
}
