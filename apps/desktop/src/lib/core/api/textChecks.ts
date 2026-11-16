// Thin `text-check` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/library/text_checks.py and apps/api/models/text_checks.py (A2b).

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { DocumentTextCheck } from '../types/textChecks'

/** What the text check found in a document. `null` for an unknown document (a 404). */
export async function getDocumentTextCheck(docId: string): Promise<DocumentTextCheck | null> {
  const r = await apiFetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/text-check`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(await errorDetail(r, 'load the text check'))
  return (await r.json()) as DocumentTextCheck
}
