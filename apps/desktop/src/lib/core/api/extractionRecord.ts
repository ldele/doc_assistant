// Thin `extraction-record` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/library/extraction_record.py and apps/api/models/extraction_record.py.

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { DocumentExtractionRecord } from '../types/extractionRecord'

/** What indexing found in a document. `null` for an unknown document (a 404). */
export async function getExtractionRecord(docId: string): Promise<DocumentExtractionRecord | null> {
  const r = await apiFetch(
    `${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/extraction-record`,
  )
  if (r.status === 404) return null
  if (!r.ok) throw new Error(await errorDetail(r, 'load what indexing found'))
  return (await r.json()) as DocumentExtractionRecord
}
