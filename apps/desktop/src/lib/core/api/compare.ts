// Thin `compare` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/compare.py and apps/api/models/compare.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type {
  CompareResult,
  RagOverrides,
} from '../types'

/** A/B-compare retrieval (U6): the query under the locked defaults vs the session override.
 *  $0 — retrieval only, no generation. `overrides` rides this one request. */
export async function compareRetrieval(
  text: string,
  overrides?: RagOverrides,
  scopeFolderId?: string | null,
): Promise<CompareResult> {
  const r = await fetch(`${API_BASE}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // ADR-025 F2 — the scope rides both sides, so the diff isolates the knob, not the corpus.
    body: JSON.stringify({
      text,
      overrides: overrides ?? null,
      scope_folder_id: scopeFolderId ?? null,
    }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'compare'))
  return (await r.json()) as CompareResult
}
