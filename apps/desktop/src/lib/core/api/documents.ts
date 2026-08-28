// Thin `documents` client — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/sources.py (AD2).

import { API_BASE, errorDetail } from './_base'
import type { InspectResponse } from '../types/documents'

/**
 * Ask what would happen to each candidate path. **Mutates nothing** — inspect and apply are two
 * endpoints precisely so that stays true (spec constraint 2).
 *
 * Directories are sent as-is: the server expands them, because the recursion rule belongs with
 * `registry.scan_sources` rather than in a second implementation here.
 */
export async function inspectDocuments(paths: string[]): Promise<InspectResponse> {
  const r = await fetch(`${API_BASE}/api/documents/inspect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'inspect documents'))
  return (await r.json()) as InspectResponse
}

/**
 * ADR-046 placement, **both modes built since AD3b**. `copy` puts the file in the Provenote
 * folder and the app owns it; `reference` registers it where it already lives and the app must
 * never delete it.
 */
export type AddMode = 'copy' | 'reference'

export interface AddOutcome {
  path: string
  name: string
  ok: boolean
  /** Where it landed, relative to its own root — for display. */
  rel_path: string | null
  /**
   * The identifier to send back for undo or indexing. Since AD3b a bare `rel_path` no longer
   * identifies a row: two roots may hold the same relative path, so always round-trip this.
   */
  key: string | null
  error: string | null
}

export interface AddResult {
  added: AddOutcome[]
  failed: AddOutcome | null
  /** Files after the failure. They were never touched — not skipped by choice. */
  not_attempted: string[]
  stopped_early: boolean
}

/**
 * Copy the chosen files into the library folder and register them. **Does not index** — that is
 * a separate call to `/api/ingest` with an explicit `paths` list, so the system has one ingest
 * path rather than two.
 *
 * 409 while an ingest is running.
 */
export async function addDocuments(paths: string[], mode: AddMode = 'copy'): Promise<AddResult> {
  const r = await fetch(`${API_BASE}/api/documents/add`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths, mode }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'add documents'))
  return (await r.json()) as AddResult
}

/**
 * Reverse a just-completed add. Drops the registry rows; deletes the file **only** when the app
 * made the copy itself — a referenced original is never touched.
 *
 * Takes the `key` of each outcome, not its `rel_path`.
 */
export async function undoAddDocuments(keys: string[]): Promise<number> {
  const r = await fetch(`${API_BASE}/api/documents/undo-add`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rel_paths: keys }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'undo add'))
  return ((await r.json()) as { undone: number }).undone
}

/** Index exactly these paths through the existing ingest endpoint (spec constraint 4). */
export async function indexPaths(paths: string[]): Promise<void> {
  const r = await fetch(`${API_BASE}/api/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'index documents'))
}
