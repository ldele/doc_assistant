// Thin `summary` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/library/summary.py and apps/api/models/summary.py (ROADMAP 112).

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { LibraryDocument } from '../types'
import type { CollectionSummary, SummaryIssue } from '../types/summary'

/** The summary of the whole library, or of one folder. `null` for a folder that is gone (404). */
export async function getCollectionSummary(
  folderId: string | null = null,
): Promise<CollectionSummary | null> {
  const query = folderId ? `?folder_id=${encodeURIComponent(folderId)}` : ''
  const r = await apiFetch(`${API_BASE}/api/library/summary${query}`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(await errorDetail(r, 'library summary'))
  return (await r.json()) as CollectionSummary
}

/** The documents one of the summary's counts names — what its *Show the N* opens. */
export async function listIssueDocuments(
  issue: SummaryIssue,
  folderId: string | null = null,
): Promise<LibraryDocument[]> {
  const params = new URLSearchParams({ issue })
  if (folderId) params.set('folder_id', folderId)
  const r = await apiFetch(`${API_BASE}/api/library/documents?${params.toString()}`)
  if (!r.ok) throw new Error(await errorDetail(r, 'documents for the summary'))
  return (await r.json()) as LibraryDocument[]
}
