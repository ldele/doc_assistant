// Taxonomy-view state (docs/specs/feature-taxonomy-view.md, ADR-028 2b).
//
// A `.svelte.ts` **rune** module (reactive `$state`), deliberately sitting beside the plain
// `taxonomy.ts` next to it: that one is pure tree-shaping and is unit-tested under `npm test`,
// this one needs the Svelte compiler and cannot be. The extension is the marker — keep the pure
// logic in `taxonomy.ts` so it stays testable (apps/desktop/CLAUDE.md).
//
// LibraryTaxonomy.svelte is a dumb renderer; this module owns the data. Every mutation is
// write-then-refetch: the server owns counts + acyclicity, so the tree is never patched by hand.
//
// Cross-domain bit that stays in App.svelte: opening the modal also lazy-loads the *document*
// list (for the attach picker). App wraps `openTaxonomy` to do that first.

import {
  addHierarchyEdge,
  attachDocumentField,
  getConceptGraph,
  getFieldDetail,
  getTaxonomy,
  removeHierarchyEdge,
} from '../core/api'
import type { FieldDetail, HierarchyEdgeRequest, LabelledOption, TaxonomyView } from '../core/types'
import { graph } from '../graph/graph.svelte'

export const taxonomy = $state({
  open: false,
  view: null as TaxonomyView | null,
  fieldDetail: null as FieldDetail | null,
  /** The attach picker's vocabulary (concept id → label). */
  concepts: [] as LabelledOption[],
  /** Preselected concept when opened from a graph node's "Place" action. */
  focusConceptId: null as string | null,
  loading: false,
  error: null as string | null,
})

// NB: a default-valued param, not `focusConceptId?: string` — the `<script lang="ts">` transform
// strips the type annotation but leaves the `?`, emitting invalid JS. svelte-check type-checks the
// source and misses it; it only breaks at runtime. Keep optional params defaulted.
export async function openTaxonomy(focusConceptId: string | null = null): Promise<void> {
  taxonomy.open = true
  taxonomy.focusConceptId = focusConceptId
  taxonomy.error = null
  taxonomy.fieldDetail = null
  void ensureTaxonomyConcepts()
  taxonomy.loading = true
  try {
    taxonomy.view = await getTaxonomy()
  } catch (e) {
    taxonomy.error = e instanceof Error ? e.message : String(e)
  } finally {
    taxonomy.loading = false
  }
}

export function closeTaxonomy(): void {
  taxonomy.open = false
  taxonomy.focusConceptId = null
}

// The attach picker's vocabulary = the graph nodes (spec ledger #7 — 2a serves no concept list).
// Reuse the already-loaded graph when present, else fetch it. On failure the picker stays empty.
export async function ensureTaxonomyConcepts(): Promise<void> {
  if (taxonomy.concepts.length > 0) return
  try {
    const g = graph.data ?? (await getConceptGraph())
    taxonomy.concepts = (g?.nodes ?? []).map((n) => ({ id: n.id, label: n.label }))
  } catch {
    // leave empty — attach-concept just has nothing to offer
  }
}

export async function selectTaxonomyField(fieldId: string): Promise<void> {
  try {
    taxonomy.fieldDetail = await getFieldDetail(fieldId)
  } catch (e) {
    taxonomy.error = e instanceof Error ? e.message : String(e)
  }
}

// Write-then-refetch: re-pull the view + the open field's detail after every mutation rather than
// patching the tree by hand (mirrors the folder handlers).
export async function reloadTaxonomy(): Promise<void> {
  try {
    taxonomy.view = await getTaxonomy()
  } catch {
    // keep the prior view
  }
  const id = taxonomy.fieldDetail?.id
  if (id !== undefined) {
    try {
      taxonomy.fieldDetail = await getFieldDetail(id)
    } catch {
      // keep the prior detail
    }
  }
}

async function mutate(op: () => Promise<unknown>): Promise<void> {
  taxonomy.error = null
  try {
    await op()
  } catch (e) {
    taxonomy.error = e instanceof Error ? e.message : String(e)
    return
  }
  await reloadTaxonomy()
}

export function taxonomyAddEdge(body: HierarchyEdgeRequest): Promise<void> {
  return mutate(() => addHierarchyEdge(body))
}
export function taxonomyRemoveEdge(body: HierarchyEdgeRequest): Promise<void> {
  return mutate(() => removeHierarchyEdge(body))
}
export function taxonomyAttachDocument(docId: string, fieldId: string): Promise<void> {
  return mutate(() => attachDocumentField(docId, fieldId))
}
