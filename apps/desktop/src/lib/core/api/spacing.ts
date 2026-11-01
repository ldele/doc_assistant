// Thin `spacing` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/library/spacing.py and apps/api/models/spacing.py (ADR-056).

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { DocumentSpacing, IgnoreList, IgnoreListRemoved } from '../types/spacing'

function docBase(docId: string): string {
  return `${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/spacing`
}

async function post<T>(url: string, body: unknown, what: string): Promise<T> {
  const r = await apiFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await errorDetail(r, what))
  return (await r.json()) as T
}

/** The runs a document's extraction recorded, and what the ignore list asks for now.
 *  `null` for an unknown document (a 404). */
export async function getDocumentSpacing(docId: string): Promise<DocumentSpacing | null> {
  const r = await apiFetch(docBase(docId))
  if (r.status === 404) return null
  if (!r.ok) throw new Error(await errorDetail(r, 'load the letter-spacing'))
  return (await r.json()) as DocumentSpacing
}

/** Keep one run as extracted. Takes effect when the document's text is re-run. */
export function keepRunSpaced(docId: string, spaced: string): Promise<DocumentSpacing> {
  return post(`${docBase(docId)}/keep`, { spaced }, 'keep this run spaced')
}

/** Space one run your way; refused (with a sentence) when it would change a letter. */
export function setRunSpacing(docId: string, spaced: string, chosen: string): Promise<DocumentSpacing> {
  return post(`${docBase(docId)}/edit`, { spaced, chosen }, 'save this spacing')
}

/** Give one run back to the rule. */
export function removeRunEntry(docId: string, spaced: string): Promise<DocumentSpacing> {
  return post(`${docBase(docId)}/remove`, { spaced }, 'remove this entry')
}

/** Keep every run of a document as extracted, or stop. */
export function setKeepAllSpaced(docId: string, keep: boolean): Promise<DocumentSpacing> {
  return post(`${docBase(docId)}/keep-all`, { keep }, 'change this document')
}

/** Every document the ignore list names, for Settings. */
export async function getIgnoreList(): Promise<IgnoreList> {
  const r = await apiFetch(`${API_BASE}/api/library/spacing`)
  if (!r.ok) throw new Error(await errorDetail(r, 'load the ignore list'))
  return (await r.json()) as IgnoreList
}

/** Remove one document's entries (`sha256`), or every entry (`null`). */
export function removeIgnoreEntries(sha256: string | null): Promise<IgnoreListRemoved> {
  return post(`${API_BASE}/api/library/spacing/remove`, { sha256 }, 'remove the entries')
}

/** Undo: put back the list a removal returned. */
export function restoreIgnoreList(previous: Record<string, unknown>): Promise<IgnoreList> {
  return post(`${API_BASE}/api/library/spacing/restore`, { previous }, 'undo the removal')
}
