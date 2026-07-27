// Thin `taxonomy` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/taxonomy.py and apps/api/models/taxonomy.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type {
  FieldDetail,
  HierarchyEdgeRequest,
  TaxonomyView,
} from '../types'

/** The field forest + per-field coverage + corpus totals. 200 with zeros on an unseeded/empty
 *  forest (the ANZSRC trunk is bundled data — an empty forest is a legit state, not a 404). */
export async function getTaxonomy(): Promise<TaxonomyView> {
  const r = await fetch(`${API_BASE}/api/taxonomy`)
  if (!r.ok) throw new Error(await errorDetail(r, 'taxonomy'))
  return (await r.json()) as TaxonomyView
}
/** One field's directly-attached concepts + documents + rollup counts. 404 when `fieldId` is not a
 *  domain node (a wrong id or a concept id) — as distinct from a real-but-empty field. */
export async function getFieldDetail(fieldId: string): Promise<FieldDetail> {
  const r = await fetch(`${API_BASE}/api/taxonomy/fields/${encodeURIComponent(fieldId)}`)
  if (!r.ok) throw new Error(await errorDetail(r, 'field detail'))
  return (await r.json()) as FieldDetail
}
/** Add one curated hierarchy edge (`source --type--> target`). Attaching a concept to a field is
 *  just an `in_field` edge from the concept to the domain node — same endpoint. Surfaces the
 *  backend's 409 (cycle) / 404 (missing id) / 400 (bad type) `detail` message. */
export async function addHierarchyEdge(body: HierarchyEdgeRequest): Promise<void> {
  const r = await fetch(`${API_BASE}/api/taxonomy/hierarchy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'add hierarchy edge'))
}
/** Remove a curated hierarchy edge by its unique key. Idempotent (`removed` 0 if absent). The
 *  client's first DELETE with a JSON body — `fetch` allows it; set the JSON content-type. */
export async function removeHierarchyEdge(body: HierarchyEdgeRequest): Promise<{ removed: number }> {
  const r = await fetch(`${API_BASE}/api/taxonomy/hierarchy`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'remove hierarchy edge'))
  return (await r.json()) as { removed: number }
}
/** Attach a document to a taxonomy field (a domain node). Idempotent per pair. 404 for an unknown
 *  document; 400 when the target is not a domain field (`NotADomainError`). */
export async function attachDocumentField(docId: string, fieldId: string): Promise<void> {
  const r = await fetch(
    `${API_BASE}/api/taxonomy/documents/${encodeURIComponent(docId)}/fields/${encodeURIComponent(fieldId)}`,
    { method: 'POST' },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'attach document to field'))
}
/** Remove a document→field link. Idempotent (`removed` 0 if absent). The reject half of the
 *  auto-propose loop (ADR-028 D8) — and equally undoes a curated attach. */
export async function detachDocumentField(
  docId: string,
  fieldId: string,
): Promise<{ removed: number }> {
  const r = await fetch(
    `${API_BASE}/api/taxonomy/documents/${encodeURIComponent(docId)}/fields/${encodeURIComponent(fieldId)}`,
    { method: 'DELETE' },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'detach document from field'))
  return (await r.json()) as { removed: number }
}

/** Pull a human message out of a FastAPI error body, falling back to the status code. `detail` is
 *  usually a string, but the selective-ingest 400 sends `{error, offenders}` — render that too. */
