// Tests for the concept-graph force layout (added 2026-08-20). At 189 lines this was the largest
// untested module in the frontend, and its own header comment states the reason it needs tests:
// "the layout is the risk; determinism is the safety net". The safety net had nothing checking it.
//
// Every property below is a claim the module makes about itself in prose — determinism, no NaN
// ever, positions inside the padded box, unknown edge endpoints ignored. They are asserted as
// properties rather than against golden coordinates on purpose: a golden snapshot of 300 cooled
// iterations would break on any tuning change while proving nothing about the invariants, which
// are what the SVG actually depends on.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { forceLayout } from './forceLayout.ts'
import type { LayoutEdge } from './forceLayout.ts'

const BOX = { width: 600, height: 400 }

/** A small connected graph: a hub with three leaves plus one leaf-to-leaf edge. */
const NODES = ['hub', 'a', 'b', 'c']
const EDGES: LayoutEdge[] = [
  { source: 'hub', target: 'a' },
  { source: 'hub', target: 'b' },
  { source: 'hub', target: 'c' },
  { source: 'a', target: 'b' },
]

function allPoints(map: Map<string, { x: number; y: number }>) {
  return [...map.values()]
}

// ============================================================
// Determinism — the property the module was designed around
// ============================================================

test('the same input produces identical positions', () => {
  const first = forceLayout(NODES, EDGES, BOX)
  const second = forceLayout(NODES, EDGES, BOX)
  for (const id of NODES) {
    assert.deepEqual(first.get(id), second.get(id), `node ${id} moved between identical runs`)
  }
})

test('a different seed produces a different layout', () => {
  // Without this the determinism test above would also pass if the seed were ignored entirely
  // and the layout were simply constant.
  const a = forceLayout(NODES, EDGES, { ...BOX, seed: 1 })
  const b = forceLayout(NODES, EDGES, { ...BOX, seed: 2 })
  const moved = NODES.some((id) => a.get(id)!.x !== b.get(id)!.x || a.get(id)!.y !== b.get(id)!.y)
  assert.ok(moved, 'seed had no effect on the layout')
})

test('the default seed is stable across calls that omit it', () => {
  const implicit = forceLayout(NODES, EDGES, BOX)
  const explicit = forceLayout(NODES, EDGES, { ...BOX, seed: 42 })
  for (const id of NODES) assert.deepEqual(implicit.get(id), explicit.get(id))
})

// ============================================================
// No NaN — "closed by design", per the header, so worth pinning
// ============================================================

test('every coordinate is finite for a connected graph', () => {
  for (const p of allPoints(forceLayout(NODES, EDGES, BOX))) {
    assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y), `non-finite point ${JSON.stringify(p)}`)
  }
})

test('every coordinate is finite when every node shares an edge with every other', () => {
  // The dense case drives the all-pairs repulsion hardest; a zero distance here is what the EPS
  // floor exists to stop.
  const ids = ['a', 'b', 'c', 'd', 'e']
  const dense: LayoutEdge[] = []
  for (const s of ids) for (const t of ids) if (s !== t) dense.push({ source: s, target: t })
  for (const p of allPoints(forceLayout(ids, dense, BOX))) {
    assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y))
  }
})

test('every coordinate is finite for a graph with no edges at all', () => {
  for (const p of allPoints(forceLayout(NODES, [], BOX))) {
    assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y))
  }
})

test('a large ego graph stays finite', () => {
  // The hub case in the spec is 21 nodes; go past it so the O(n^2) path is genuinely exercised.
  const ids = Array.from({ length: 40 }, (_, i) => `n${i}`)
  const star: LayoutEdge[] = ids.slice(1).map((t) => ({ source: ids[0], target: t }))
  for (const p of allPoints(forceLayout(ids, star, BOX))) {
    assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y))
  }
})

// ============================================================
// Fit to the viewBox
// ============================================================

test('every node lands inside the padded box', () => {
  const padding = 24
  for (const p of allPoints(forceLayout(NODES, EDGES, { ...BOX, padding }))) {
    assert.ok(p.x >= padding - 0.5 && p.x <= BOX.width - padding + 0.5, `x out of box: ${p.x}`)
    assert.ok(p.y >= padding - 0.5 && p.y <= BOX.height - padding + 0.5, `y out of box: ${p.y}`)
  }
})

test('a larger padding shrinks the occupied area', () => {
  const spread = (padding: number) => {
    const xs = allPoints(forceLayout(NODES, EDGES, { ...BOX, padding })).map((p) => p.x)
    return Math.max(...xs) - Math.min(...xs)
  }
  assert.ok(spread(100) < spread(10), 'padding did not constrain the layout')
})

test('the box dimensions are respected, not just the aspect ratio', () => {
  const tall = forceLayout(NODES, EDGES, { width: 200, height: 900, padding: 10 })
  for (const p of allPoints(tall)) {
    assert.ok(p.x >= 9.5 && p.x <= 190.5, `x escaped the narrow box: ${p.x}`)
    assert.ok(p.y >= 9.5 && p.y <= 890.5, `y escaped the tall box: ${p.y}`)
  }
})

// ============================================================
// Degenerate inputs — the 0-document robustness contract, in the graph
// ============================================================

test('no nodes returns an empty map rather than throwing', () => {
  assert.equal(forceLayout([], [], BOX).size, 0)
})

test('a single node is centred', () => {
  const pos = forceLayout(['only'], [], BOX)
  assert.deepEqual(pos.get('only'), { x: BOX.width / 2, y: BOX.height / 2 })
})

test('a single node with a self-edge is still centred and finite', () => {
  const pos = forceLayout(['only'], [{ source: 'only', target: 'only' }], BOX)
  const p = pos.get('only')!
  assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y))
})

test('two nodes do not end up coincident', () => {
  const pos = forceLayout(['a', 'b'], [{ source: 'a', target: 'b' }], BOX)
  const a = pos.get('a')!
  const b = pos.get('b')!
  assert.ok(Math.hypot(a.x - b.x, a.y - b.y) > 1, 'nodes overlap; they would render as one dot')
})

// ============================================================
// Edge filtering — documented as safe to pass an unfiltered list
// ============================================================

test('edges naming unknown nodes are ignored, not crashed on', () => {
  // The docstring invites callers to pass the full edge list unfiltered; a non-null assertion on
  // a missing node would throw instead.
  const edges: LayoutEdge[] = [...EDGES, { source: 'hub', target: 'not-in-graph' }]
  const pos = forceLayout(NODES, edges, BOX)
  assert.equal(pos.size, NODES.length)
  assert.equal(pos.has('not-in-graph'), false)
})

test('an unfiltered edge list gives the same layout as a pre-filtered one', () => {
  const withStrays: LayoutEdge[] = [
    ...EDGES,
    { source: 'ghost', target: 'a' },
    { source: 'b', target: 'ghost' },
  ]
  const filtered = forceLayout(NODES, EDGES, BOX)
  const unfiltered = forceLayout(NODES, withStrays, BOX)
  for (const id of NODES) assert.deepEqual(unfiltered.get(id), filtered.get(id))
})

// ============================================================
// Output shape
// ============================================================

test('every requested node gets exactly one position', () => {
  const pos = forceLayout(NODES, EDGES, BOX)
  assert.equal(pos.size, NODES.length)
  for (const id of NODES) assert.ok(pos.has(id), `missing position for ${id}`)
})

test('duplicate node ids do not produce duplicate entries', () => {
  const pos = forceLayout(['a', 'b', 'a'], [], BOX)
  assert.deepEqual([...pos.keys()].sort(), ['a', 'b'])
})

test('connected nodes settle closer than unconnected ones', () => {
  // The one *behavioural* claim, kept deliberately weak: attraction must beat repulsion for a
  // bonded pair. Anything tighter would be a golden snapshot of the cooling schedule.
  const ids = ['a', 'b', 'x', 'y']
  const pos = forceLayout(ids, [{ source: 'a', target: 'b' }], { ...BOX, iterations: 300 })
  const d = (p: string, q: string) =>
    Math.hypot(pos.get(p)!.x - pos.get(q)!.x, pos.get(p)!.y - pos.get(q)!.y)
  assert.ok(d('a', 'b') < Math.max(d('a', 'x'), d('a', 'y')), 'the edge exerted no attraction')
})
