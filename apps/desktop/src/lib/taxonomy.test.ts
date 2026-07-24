// Tests for the taxonomy tree-shaper (docs/specs/feature-taxonomy-view.md, T2 / DoD 1).
// Runner: node's built-in `node:test` with native TypeScript stripping — no new dependency.
// `npm test` from apps/desktop.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import { buildForest } from './taxonomy.ts'
import type { TaxonomyField, TaxonomyView } from './types.ts'

const field = (id: string, child_ids: string[], parent_ids: string[] = []): TaxonomyField => ({
  id,
  label: id.toUpperCase(),
  parent_ids,
  child_ids,
  n_concepts_direct: 0,
  n_documents_direct: 0,
  n_concepts_rollup: 0,
  n_documents_rollup: 0,
})

const view = (roots: string[], fields: TaxonomyField[]): TaxonomyView => ({
  fields,
  roots,
  n_concepts_total: 0,
  n_documents_total: 0,
  n_unassigned_concepts: 0,
})

test('buildForest: root-then-child order, with depth + hasChildren', () => {
  const rows = buildForest(
    view(['r'], [
      field('r', ['a', 'b']),
      field('a', ['c'], ['r']),
      field('b', [], ['r']),
      field('c', [], ['a']),
    ]),
  )
  assert.deepEqual(
    rows.map((x) => [x.field.id, x.depth, x.hasChildren]),
    [
      ['r', 0, true],
      ['a', 1, true],
      ['c', 2, false],
      ['b', 1, false],
    ],
  )
})

test('buildForest: a poly-parented field renders under both parents, subtree expanded both times', () => {
  // r → {a, b}; both a and b → d; d → e. `d` (and its child `e`) must appear under BOTH a and b.
  const rows = buildForest(
    view(['r'], [
      field('r', ['a', 'b']),
      field('a', ['d'], ['r']),
      field('b', ['d'], ['r']),
      field('d', ['e'], ['a', 'b']),
      field('e', [], ['d']),
    ]),
  )
  const ids = rows.map((x) => x.field.id)
  assert.deepEqual(ids, ['r', 'a', 'd', 'e', 'b', 'd', 'e'])
  // A global-visited implementation would render d (and e) once — assert both appear twice.
  assert.equal(ids.filter((i) => i === 'd').length, 2)
  assert.equal(ids.filter((i) => i === 'e').length, 2)
})

test('buildForest: the ancestor-path guard truncates a cycle instead of hanging', () => {
  // r → a → b → a: the back-edge b→a closes a loop. The walk must terminate (a corrupt DB only —
  // the backend 409s on any cycle-forming write), skipping the back-edge.
  const rows = buildForest(
    view(['r'], [
      field('r', ['a']),
      field('a', ['b'], ['r']),
      field('b', ['a'], ['a']),
    ]),
  )
  assert.deepEqual(
    rows.map((x) => x.field.id),
    ['r', 'a', 'b'],
  )
})

test('buildForest: empty forest → []', () => {
  assert.deepEqual(buildForest(view([], [])), [])
})
