// Thin `sources` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/sources.py and apps/api/models/sources.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type {
  IngestStatus,
  SourceFile,
} from '../types'

/** Kick off a background re-index. No `paths` = the whole saved source folder (minus exclusions);
 *  a `paths` list (rel_paths) = ingest exactly that selection. 409 if one is already running,
 *  400 if any path is invalid (the detail names the offenders). */
export async function startIngest(paths?: string[]): Promise<IngestStatus> {
  const init: RequestInit =
    paths === undefined
      ? { method: 'POST' }
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ paths }),
        }
  const r = await fetch(`${API_BASE}/api/ingest`, init)
  if (!r.ok) throw new Error(await errorDetail(r, 'start ingest'))
  return (await r.json()) as IngestStatus
}
/** Selective ingestion (S2): stat-only scan of the source folder → every file with a derived
 *  ingest status + its `excluded` flag. Cheap ($0/offline); safe to call on every panel open. */
export async function getSources(): Promise<SourceFile[]> {
  const r = await fetch(`${API_BASE}/api/sources`)
  if (!r.ok) throw new Error(`sources failed: ${r.status}`)
  return (await r.json()) as SourceFile[]
}
/** Set a file's `excluded` flag (an excluded file is skipped by a whole-folder index; an explicit
 *  selection still overrides it). Returns the updated row. 404 if the rel_path is unknown. */
export async function patchSource(relPath: string, excluded: boolean): Promise<SourceFile> {
  const r = await fetch(`${API_BASE}/api/sources`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rel_path: relPath, excluded }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'update source'))
  return (await r.json()) as SourceFile
}
export async function getIngestStatus(): Promise<IngestStatus> {
  const r = await fetch(`${API_BASE}/api/ingest/status`)
  if (!r.ok) throw new Error(`ingest status failed: ${r.status}`)
  return (await r.json()) as IngestStatus
}
