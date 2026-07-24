// Taxonomy tree-shaping (docs/specs/feature-taxonomy-view.md, ADR-028 increment 2b, T2). Pure by
// design so it is exercised by `npm test` (node:test) — LibraryTaxonomy is a dumb renderer, App owns
// the data. Turns the flat field list the API serves into ordered display rows.
//
// The taxonomy is a DAG, not a tree: a field may have several parents (ADR-028 D1), so it renders
// under EACH of them — the DAG is displayed as a tree with repeats, nothing orphaned or deduped. A
// diamond (a field reached by two paths) terminates on its own; the guard below is only for a
// corrupt DB that somehow holds a cycle (the backend's 409 on write is the real invariant).
import type { TaxonomyField, TaxonomyView } from './types'

export interface TaxonomyRow {
  field: TaxonomyField
  depth: number // 0 at the roots (divisions), +1 per level
  hasChildren: boolean
}

/** Flatten the field DAG into ordered rows (root order, then each field's `child_ids` order). A
 *  poly-parented field appears once per parent, its subtree expanded each time. The guard is the
 *  set of ancestors on the CURRENT path (not a global visited set — that would suppress the second
 *  expansion and break render-under-both); it fires only if a child is already its own ancestor,
 *  i.e. a cycle, and truncates there instead of recursing forever. */
export function buildForest(view: TaxonomyView): TaxonomyRow[] {
  const byId = new Map<string, TaxonomyField>()
  for (const f of view.fields) byId.set(f.id, f)

  const rows: TaxonomyRow[] = []
  const path = new Set<string>() // ancestors of the node being visited; backtracked on exit

  const walk = (id: string, depth: number): void => {
    const field = byId.get(id)
    if (field === undefined) return // a dangling id in roots/child_ids — skip, never throw
    if (path.has(id)) return // this id is its own ancestor: a cycle — truncate (corrupt DB only)
    rows.push({ field, depth, hasChildren: field.child_ids.length > 0 })
    path.add(id)
    for (const childId of field.child_ids) walk(childId, depth + 1)
    path.delete(id)
  }

  for (const rootId of view.roots) walk(rootId, 0)
  return rows
}
