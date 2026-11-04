// Thin `sources` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/sources.py and apps/api/models/sources.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, apiFetch, errorDetail } from './_base'
import type {
  IngestFilesPage,
  IngestGroup,
  IngestPlan,
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
  const r = await apiFetch(`${API_BASE}/api/ingest`, init)
  if (!r.ok) throw new Error(await errorDetail(r, 'start ingest'))
  return (await r.json()) as IngestStatus
}
/** Selective ingestion (S2): stat-only scan of the source folder → every file with a derived
 *  ingest status + its `excluded` flag. Cheap ($0/offline); safe to call on every panel open. */
export async function getSources(): Promise<SourceFile[]> {
  const r = await apiFetch(`${API_BASE}/api/sources`)
  if (!r.ok) throw new Error(`sources failed: ${r.status}`)
  return (await r.json()) as SourceFile[]
}
/** Set a file's `excluded` flag (an excluded file is skipped by a whole-folder index; an explicit
 *  selection still overrides it). Returns the updated row. 404 if the rel_path is unknown. */
export async function patchSource(relPath: string, excluded: boolean): Promise<SourceFile> {
  const r = await apiFetch(`${API_BASE}/api/sources`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rel_path: relPath, excluded }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'update source'))
  return (await r.json()) as SourceFile
}
export async function getIngestStatus(): Promise<IngestStatus> {
  const r = await apiFetch(`${API_BASE}/api/ingest/status`)
  if (!r.ok) throw new Error(`ingest status failed: ${r.status}`)
  return (await r.json()) as IngestStatus
}
/** What a Re-index of the library folder would read, and a rough duration. Stat-only, $0. */
export async function getIngestPlan(): Promise<IngestPlan> {
  const r = await apiFetch(`${API_BASE}/api/ingest/plan`)
  if (!r.ok) throw new Error(await errorDetail(r, 'plan the re-index'))
  return (await r.json()) as IngestPlan
}
/** Index only the library folder's files never indexed (the Re-index confirmation's checkbox). */
export async function startIngestNewOnly(): Promise<IngestStatus> {
  const r = await apiFetch(`${API_BASE}/api/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ only_new: true }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'start indexing'))
  return (await r.json()) as IngestStatus
}
/** Start a run over files taken from the last one, which the server reads from its own record of
 *  it: `failed` retries the failures, `remaining` indexes what a stop left (ROADMAP 31). */
export async function rerunIngest(rerun: 'failed' | 'remaining'): Promise<IngestStatus> {
  const r = await apiFetch(`${API_BASE}/api/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rerun }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'start indexing'))
  return (await r.json()) as IngestStatus
}
/** Pause, resume or stop the run in flight. Pause and stop take effect once the document being
 *  indexed finishes. 409 when no run is in progress. Returns the status after the request. */
export async function controlIngest(action: 'pause' | 'resume' | 'stop'): Promise<IngestStatus> {
  const r = await apiFetch(`${API_BASE}/api/ingest/${action}`, { method: 'POST' })
  if (!r.ok) throw new Error(await errorDetail(r, `${action} indexing`))
  return (await r.json()) as IngestStatus
}
/** One page of the current or last run's files, in the screen's order. Empty before any run. */
export async function getIngestFiles(
  group: IngestGroup = 'all',
  offset = 0,
  limit = 200,
): Promise<IngestFilesPage> {
  const q = new URLSearchParams({ group, offset: String(offset), limit: String(limit) })
  const r = await apiFetch(`${API_BASE}/api/ingest/files?${q}`)
  if (!r.ok) throw new Error(`ingest files failed: ${r.status}`)
  return (await r.json()) as IngestFilesPage
}
