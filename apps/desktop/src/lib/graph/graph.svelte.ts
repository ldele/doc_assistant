// Concept-graph state (feature-concept-graph.md PR-G2a, ADR-017).
//
// A `.svelte.ts` **rune** module — the extension is the signal: this file holds reactive `$state`,
// unlike the plain `gaps.ts`/`forceLayout.ts` beside it, which are pure and unit-tested under
// `npm test`. Rune modules need the Svelte compiler, so they are NOT runnable by `node:test`;
// keep pure logic in the plain modules so it stays testable (apps/desktop/CLAUDE.md).
//
// State is exported as ONE `$state` object rather than separate `let`s: an imported binding cannot
// be reassigned by a consumer, so `graph.selectedId = x` is the only shape that works across a
// module boundary.
//
// What deliberately did NOT move here: `selectGraphConcept` (it also closes the mobile drawer —
// shell state) and the nav-history observer (it watches mode + library + graph at once). Those are
// cross-domain orchestration and live in App.svelte, where the coupling stays visible.

import { getConceptGraph, getGraphRebuildStatus, rebuildConceptGraph } from '../core/api'
import type { ConceptGraph, GraphRebuildStatus } from '../core/types'

export const graph = $state({
  /** `null` after a load = "never built" (a 404 — the normal first run); the view renders a
   *  build affordance rather than an error. */
  data: null as ConceptGraph | null,
  loading: false,
  error: null as string | null,
  rebuildState: 'idle' as GraphRebuildStatus['state'],
  /** Selection + the under-connected lens are shared: the sidebar's GraphIndex rail and
   *  ConceptGraph's ego panel both read them, so they must agree. */
  selectedId: null as string | null,
  showUnderConnected: false,
})

// Deliberately not reactive: a one-shot "has the lazy load already run" latch. Making it $state
// would invite a render to depend on it and re-trigger the fetch.
let loaded = false
export function graphLoaded(): boolean {
  return loaded
}

/** Drop the lazy-load latch so the next entry into the Graph refetches.
 *
 * For writes that change what the graph would *say* without changing the graph: curating a
 * concept into the vocabulary (ADR-018) moves `staleness`, which is derived server-side at read
 * time, so a graph already loaded in this session would keep reporting the pre-toggle count.
 * Dropping the latch rather than refetching keeps the cost where the user actually goes looking. */
export function invalidateGraph(): void {
  loaded = false
}

export async function loadConceptGraph(): Promise<void> {
  graph.loading = true
  graph.error = null
  try {
    graph.data = await getConceptGraph()
  } catch (e) {
    graph.error = e instanceof Error ? e.message : String(e)
  } finally {
    graph.loading = false
    loaded = true
  }
}

// Kick a rebuild and poll the status route until it settles, then refetch the graph. Deterministic
// and ~7s; the view stays usable throughout (inform, don't block).
export async function rebuildGraph(): Promise<void> {
  if (graph.rebuildState === 'running') return
  graph.rebuildState = 'running'
  try {
    await rebuildConceptGraph()
  } catch (e) {
    graph.rebuildState = 'error'
    graph.error = e instanceof Error ? e.message : String(e)
    return
  }
  const poll = async (): Promise<void> => {
    try {
      const st = await getGraphRebuildStatus()
      graph.rebuildState = st.state
      if (st.state === 'running') {
        setTimeout(() => void poll(), 700)
        return
      }
      if (st.state === 'error') {
        graph.error = st.message ?? 'rebuild failed'
        return
      }
      await loadConceptGraph() // 'done' → pull the fresh graph
    } catch (e) {
      graph.rebuildState = 'error'
      graph.error = e instanceof Error ? e.message : String(e)
    }
  }
  void poll()
}

// `$effect` needs an active effect context, so it cannot run at module top level — this is called
// once from App.svelte's script during component init. Hygiene only, and intra-domain: a rebuild
// can drop the selected concept (mirrors the chatScopeFolderId guard, which stays in App because
// it spans chat + folders).
export function useGraphHygiene(): void {
  $effect(() => {
    if (
      graph.data &&
      graph.selectedId !== null &&
      !graph.data.nodes.some((n) => n.id === graph.selectedId)
    ) {
      graph.selectedId = null
    }
  })
}
