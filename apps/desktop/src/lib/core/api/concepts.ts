// Thin `concepts` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/concepts.py and apps/api/models/concepts.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type {
  ConceptGraph,
  ConceptPresence,
  GapListItem,
  GapStatus,
  GraphRebuildStatus,
} from '../types'

// Concept graph (docs/specs/feature-concept-graph.md, ADR-017). Read-only render model + a 202+poll
// rebuild, mirroring the ingest job. A 404 is the NORMAL first run — skeleton.json is gitignored, so
// a fresh clone has none; the caller renders an empty state offering a rebuild, not an error.
export async function getConceptGraph(): Promise<ConceptGraph | null> {
  const r = await fetch(`${API_BASE}/api/concepts/graph`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(await errorDetail(r, 'concept graph'))
  return (await r.json()) as ConceptGraph
}
/** The first-class gap list (E5, ADR-004): every detected gap with its concept label + effective
 *  triage status. Pure sidecar read, $0. Empty list = no gaps built (0-doc / pre-build), not an error. */
export async function getGapList(): Promise<GapListItem[]> {
  const r = await fetch(`${API_BASE}/api/concepts/gaps`)
  if (!r.ok) throw new Error(await errorDetail(r, 'gap list'))
  return (await r.json()) as GapListItem[]
}
/** Record (or reset) a user's triage verdict on one gap (ADR-017 C1). `surfaced` resets to the
 *  detector's default. Keyed on (concept_id, kind) so it survives the deterministic rebuild. */
export async function triageGap(
  conceptId: string,
  kind: string,
  status: GapStatus,
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/concepts/gaps/triage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ concept_id: conceptId, kind, status }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'triage gap'))
}
/** Where one concept appears, down to the chunk keys — the ego view's navigation payload. Served
 *  per-concept (not bulk in the graph) so one neighbourhood's chunks load at a time. */
export async function getConceptPresence(conceptId: string): Promise<ConceptPresence[]> {
  const r = await fetch(`${API_BASE}/api/concepts/${encodeURIComponent(conceptId)}/presence`)
  if (!r.ok) throw new Error(await errorDetail(r, 'concept presence'))
  return (await r.json()) as ConceptPresence[]
}
/** Trigger a rebuild (202). Deterministic, ~7s, zero-LLM; poll getGraphRebuildStatus until done. */
export async function rebuildConceptGraph(): Promise<GraphRebuildStatus> {
  const r = await fetch(`${API_BASE}/api/concepts/graph/rebuild`, { method: 'POST' })
  if (!r.ok) throw new Error(await errorDetail(r, 'rebuild concept graph'))
  return (await r.json()) as GraphRebuildStatus
}
export async function getGraphRebuildStatus(): Promise<GraphRebuildStatus> {
  const r = await fetch(`${API_BASE}/api/concepts/graph/rebuild/status`)
  if (!r.ok) throw new Error(`graph rebuild status failed: ${r.status}`)
  return (await r.json()) as GraphRebuildStatus
}

// Taxonomy (docs/specs/feature-taxonomy-view.md, ADR-028 increment 2b). The curated field forest +
// coverage read model (2a) and its placement write-path. $0, zero-LLM — deterministic sidecar
// reads/writes. Edits *placement* only (concept→field `in_field` edges + document→field); concept
// vocabulary + field structure stay elsewhere (ADR-017 A1 / ADR-019 D11).
