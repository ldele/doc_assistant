// Thin `libraries` client for the desktop API — fetch + parsing only. Pairs with
// apps/api/routers/libraries.py and apps/api/models/libraries.py (ADR-061).

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { LibraryInspection } from '../types'

/** Read what a folder holds as a library, before the app opens it. Writes nothing. */
export async function inspectLibrary(path: string): Promise<LibraryInspection> {
  const r = await apiFetch(`${API_BASE}/api/libraries/inspect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'read that folder'))
  return (await r.json()) as LibraryInspection
}
