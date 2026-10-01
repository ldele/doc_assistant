// Thin `keywords` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/keywords.py and apps/api/models/keywords.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type {
  KeywordFamily,
  KeywordFamilyDeletion,
  KeywordFamilyProposal,
  MemberBreadth,
} from '../types'

export async function listKeywordFamilies(): Promise<KeywordFamily[]> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families`)
  if (!r.ok) throw new Error(await errorDetail(r, 'keyword families'))
  return (await r.json()) as KeywordFamily[]
}
export async function createKeywordFamily(
  canonical: string,
  members: string[] = [],
): Promise<KeywordFamily> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canonical, members }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'create keyword family'))
  return (await r.json()) as KeywordFamily
}
export async function renameKeywordFamily(
  familyId: string,
  canonical: string,
): Promise<KeywordFamily> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canonical }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'rename keyword family'))
  return (await r.json()) as KeywordFamily
}
/** Put a family's concept on the graph, or take it off (ADR-018's curation flag).
 *
 * The graph itself does not change until it is rebuilt — the skeleton is a derived sidecar. That
 * is not hidden: the graph view re-reads the live vocabulary on every render and says how many
 * concepts it is behind, with the rebuild beside it.
 */
export async function setFamilyGraphInclude(
  familyId: string,
  include: boolean,
): Promise<KeywordFamily> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ graph_include: include }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'set graph vocabulary'))
  return (await r.json()) as KeywordFamily
}
/** Assign a keyword to a family, moving it off any other family it belonged to (ADR-015). */
export async function addFamilyMember(familyId: string, keyword: string): Promise<KeywordFamily> {
  const r = await fetch(
    `${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}/members`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keyword }),
    },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'add family member'))
  return (await r.json()) as KeywordFamily
}
export async function removeFamilyMember(
  familyId: string,
  keyword: string,
): Promise<KeywordFamily> {
  const r = await fetch(
    `${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}/members/${encodeURIComponent(keyword)}`,
    { method: 'DELETE' },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'remove family member'))
  return (await r.json()) as KeywordFamily
}
/** Mark one member exact or broad, or clear the mark with `null` (ADR-054).
 *
 * An exact form counts as presence; a broad one is counted beside it. Like the graph flag, this
 * moves the vocabulary: the graph follows at its next rebuild.
 */
export async function setFamilyMemberBreadth(
  familyId: string,
  keyword: string,
  breadth: MemberBreadth,
): Promise<KeywordFamily> {
  const r = await fetch(
    `${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}/members/${encodeURIComponent(keyword)}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ breadth }),
    },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'mark family member'))
  return (await r.json()) as KeywordFamily
}
/** What deleting this family would remove — asked before the delete, so the confirmation can say. */
export async function describeFamilyDeletion(familyId: string): Promise<KeywordFamilyDeletion> {
  const r = await fetch(
    `${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}/deletion`,
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'describe family deletion'))
  return (await r.json()) as KeywordFamilyDeletion
}
export async function deleteKeywordFamily(familyId: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'delete keyword family'))
}
/** Zero-LLM detection pass (PR-2): propose family groupings for un-familied keywords via
 *  morphological + bge-embedding clustering. Nothing is written — accept a proposal via
 *  createKeywordFamily. May take a few seconds (runs the embedder over the candidate pool). */
export async function detectKeywordFamilies(): Promise<KeywordFamilyProposal[]> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families/detect`, { method: 'POST' })
  if (!r.ok) throw new Error(await errorDetail(r, 'detect keyword families'))
  return (await r.json()) as KeywordFamilyProposal[]
}
