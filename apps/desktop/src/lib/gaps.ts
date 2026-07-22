// Gap taxonomy + presentation ordering (ADR-004 / RG-014). Pure module — no Svelte, no imports of
// sibling value modules — so it is unit-testable under `npm test` (node:test) and shared by both
// ConceptGraph.svelte (the node-badge lens) and GapList.svelte (the E5 triage list).
//
// Why the ranking/hidden-by-default lives here, not on the server: it is a *presentation* decision
// (RG-014). The strong signal is `single_source` — the corroboration thesis — so it leads;
// `under_connected` measures graph degree, is dominated by vocabulary sparsity at this corpus size,
// and stays out of the lens until the user opts in. `tone` picks the reserved semantic colour
// (single_source is the danger-toned thesis; softer kinds warn). Stance is an EDGE property (Node B,
// deferred), so a node gap badge and an edge stance never collide (B9).
import type { GapKind } from './types'

export interface GapMeta {
  rank: number
  tone: 'danger' | 'warn'
  label: string
  blurb: string
  hiddenByDefault?: boolean
}

export const GAP_META: Record<GapKind, GapMeta> = {
  single_source: {
    rank: 0,
    tone: 'danger',
    label: 'Single source',
    blurb: 'Appears in only one document — no independent corroboration.',
  },
  unsourced_claim: {
    rank: 1,
    tone: 'warn',
    label: 'Unsourced claims',
    blurb: 'Carries claims the corpus does not cite (count is approximate).',
  },
  citation_missing: {
    rank: 2,
    tone: 'warn',
    label: 'Citation missing',
    blurb: 'A citation could not be resolved to a source.',
  },
  thin_bridge: {
    rank: 3,
    tone: 'warn',
    label: 'Thin bridge',
    blurb: 'Connects two areas through a single fragile edge.',
  },
  isolated: {
    rank: 3,
    tone: 'warn',
    label: 'Isolated',
    blurb: 'Has no edges to the rest of the graph.',
  },
  thin_area: {
    rank: 4,
    tone: 'warn',
    label: 'Thin area',
    blurb: 'Sits in a sparsely covered region of the corpus.',
  },
  suggested_link: {
    rank: 5,
    tone: 'warn',
    label: 'Suggested link',
    blurb: 'A plausible missing connection.',
  },
  suggested_concept: {
    rank: 5,
    tone: 'warn',
    label: 'Suggested concept',
    blurb: 'A concept the corpus implies but does not name.',
  },
  under_connected: {
    rank: 6,
    tone: 'warn',
    label: 'Under-connected',
    blurb:
      'Low graph degree — noisy at a small vocabulary; grows more meaningful as the graph fills in.',
    hiddenByDefault: true,
  },
}

export function gapRank(kind: GapKind): number {
  return GAP_META[kind]?.rank ?? 9
}

/** True when a kind is kept out of the lens until the user opts in (`under_connected`). */
export function isHiddenByDefault(kind: GapKind): boolean {
  return GAP_META[kind]?.hiddenByDefault === true
}

/** Whether a gap of `kind` is visible under the current lens (the `under_connected` opt-in). */
export function gapVisible(kind: GapKind, showUnderConnected: boolean): boolean {
  return showUnderConnected || !isHiddenByDefault(kind)
}

/**
 * Order gaps for display (RG-014): strong list-shaped kinds first (single_source, then
 * unsourced_claim, …), ties broken by concept label. Pure and total — an unknown kind sorts last.
 * `keyOf` extracts the sort inputs so this works for both `Gap` and `GapListItem`.
 */
export function orderGaps<T>(
  items: readonly T[],
  keyOf: (item: T) => { kind: GapKind; label: string },
): T[] {
  return [...items].sort((a, b) => {
    const ka = keyOf(a)
    const kb = keyOf(b)
    const r = gapRank(ka.kind) - gapRank(kb.kind)
    return r !== 0 ? r : ka.label.localeCompare(kb.label)
  })
}
