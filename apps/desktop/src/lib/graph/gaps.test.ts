import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  GAP_META,
  gapRank,
  isHiddenByDefault,
  gapVisible,
  orderGaps,
  visibleConceptGaps,
  conceptIndexRows,
  filterGapRows,
  graphCoverage,
} from './gaps.ts'
import type { ConceptGraphNode, Gap, GapKind } from '../core/types/index.ts'

test('every GapKind has taxonomy metadata', () => {
  const kinds: GapKind[] = [
    'isolated',
    'single_source',
    'thin_bridge',
    'under_connected',
    'unsourced_claim',
    'citation_missing',
    'suggested_link',
    'suggested_concept',
    'thin_area',
  ]
  for (const k of kinds) {
    assert.ok(GAP_META[k], `missing meta for ${k}`)
    assert.ok(GAP_META[k].label.length > 0)
  }
})

test('single_source is the strongest (lowest-rank) signal (RG-014)', () => {
  assert.equal(gapRank('single_source'), 0)
  assert.ok(gapRank('single_source') < gapRank('unsourced_claim'))
  assert.ok(gapRank('unsourced_claim') < gapRank('under_connected'))
})

test('single_source carries the danger tone; softer kinds warn', () => {
  assert.equal(GAP_META.single_source.tone, 'danger')
  assert.equal(GAP_META.under_connected.tone, 'warn')
})

test('under_connected is hidden by default, others are not', () => {
  assert.equal(isHiddenByDefault('under_connected'), true)
  assert.equal(isHiddenByDefault('single_source'), false)
})

test('gapVisible respects the under-connected opt-in', () => {
  assert.equal(gapVisible('under_connected', false), false)
  assert.equal(gapVisible('under_connected', true), true)
  assert.equal(gapVisible('single_source', false), true)
})

test('orderGaps sorts strong kinds first, then by label', () => {
  const items = [
    { kind: 'under_connected' as GapKind, label: 'Zebra' },
    { kind: 'single_source' as GapKind, label: 'Beta' },
    { kind: 'single_source' as GapKind, label: 'Alpha' },
    { kind: 'unsourced_claim' as GapKind, label: 'Gamma' },
  ]
  const ordered = orderGaps(items, (it) => it)
  assert.deepEqual(
    ordered.map((o) => o.label),
    ['Alpha', 'Beta', 'Gamma', 'Zebra'],
  )
})

test('orderGaps is pure — it does not mutate the input', () => {
  const items = [
    { kind: 'under_connected' as GapKind, label: 'Z' },
    { kind: 'single_source' as GapKind, label: 'A' },
  ]
  const before = items.map((i) => i.label).join(',')
  orderGaps(items, (it) => it)
  assert.equal(items.map((i) => i.label).join(','), before)
})

function mkGap(conceptId: string, kind: GapKind, status = 'surfaced'): Gap {
  return { concept_id: conceptId, tier: 'A', determinism: 'det', kind, fact_ids: [], rating: null, status }
}
function mkNode(id: string, label: string): ConceptGraphNode {
  return { id, label, doc_ids: [], degree: 0, community: 0 }
}

test('visibleConceptGaps drops dismissed, gates under_connected, sorts strongest-first', () => {
  const gaps = [
    mkGap('c1', 'under_connected'),
    mkGap('c1', 'single_source', 'dismissed'),
    mkGap('c1', 'thin_bridge'),
    mkGap('c1', 'unsourced_claim', 'promoted'),
  ]
  assert.deepEqual(
    visibleConceptGaps(gaps, false).map((g) => g.kind),
    ['unsourced_claim', 'thin_bridge'],
  )
  // Opt-in surfaces under_connected, at the tail (weakest rank).
  assert.deepEqual(
    visibleConceptGaps(gaps, true).map((g) => g.kind),
    ['unsourced_claim', 'thin_bridge', 'under_connected'],
  )
})

test('conceptIndexRows filters by label query, case-insensitively', () => {
  const nodes = [mkNode('c1', 'Embeddings'), mkNode('c2', 'BM25'), mkNode('c3', 'Brain connectivity')]
  const rows = conceptIndexRows(nodes, new Map(), 'bR', false, false)
  assert.deepEqual(
    rows.map((r) => r.node.label),
    ['Brain connectivity'],
  )
})

test('conceptIndexRows gapsOnly keeps only concepts with a visible gap', () => {
  const nodes = [mkNode('c1', 'Alpha'), mkNode('c2', 'Beta'), mkNode('c3', 'Gamma')]
  const byConcept = new Map<string, Gap[]>([
    ['c2', [mkGap('c2', 'single_source')]],
    // c3's only gap is hidden by the default lens — gapsOnly must respect the lens.
    ['c3', [mkGap('c3', 'under_connected')]],
  ])
  const rows = conceptIndexRows(nodes, byConcept, '', true, false)
  assert.deepEqual(
    rows.map((r) => r.node.label),
    ['Beta'],
  )
})

test('conceptIndexRows orders gapped concepts first by rank, then A–Z', () => {
  const nodes = [mkNode('c1', 'Zebra'), mkNode('c2', 'Alpha'), mkNode('c3', 'Mid'), mkNode('c4', 'Beta')]
  const byConcept = new Map<string, Gap[]>([
    ['c1', [mkGap('c1', 'unsourced_claim')]],
    ['c3', [mkGap('c3', 'single_source')]],
  ])
  const rows = conceptIndexRows(nodes, byConcept, '', false, false)
  // single_source (rank 0) first, then unsourced_claim (rank 1), then gapless A–Z.
  assert.deepEqual(
    rows.map((r) => r.node.label),
    ['Mid', 'Zebra', 'Alpha', 'Beta'],
  )
})

test('conceptIndexRows is pure — inputs unmutated', () => {
  const nodes = [mkNode('c1', 'B'), mkNode('c2', 'A')]
  const byConcept = new Map<string, Gap[]>([['c1', [mkGap('c1', 'single_source')]]])
  conceptIndexRows(nodes, byConcept, '', false, false)
  assert.deepEqual(
    nodes.map((n) => n.label),
    ['B', 'A'],
  )
  assert.equal(byConcept.get('c1')?.length, 1)
})

// --- filterGapRows: the Gaps-tab filter box (ui-checklist §2) ---------------------------------

const gapRow = (label: string, kind: GapKind): { kind: GapKind; label: string } => ({ kind, label })
const rowKey = (r: { kind: GapKind; label: string }): { kind: GapKind; label: string } => r

test('filterGapRows matches the concept label, case-insensitively', () => {
  const rows = [gapRow('Transformers', 'single_source'), gapRow('Retrieval', 'thin_bridge')]
  assert.deepEqual(
    filterGapRows(rows, rowKey, 'TRANS').map((r) => r.label),
    ['Transformers'],
  )
})

test('filterGapRows also matches the gap KIND, which is what people actually type', () => {
  // "single" is the list's own vocabulary for a class of problem, not a concept name — filtering
  // on the label alone would return nothing and read as "no such gaps".
  const rows = [gapRow('Transformers', 'single_source'), gapRow('Retrieval', 'thin_bridge')]
  assert.deepEqual(
    filterGapRows(rows, rowKey, 'single').map((r) => r.label),
    ['Transformers'],
  )
})

test('filterGapRows returns everything for an empty or whitespace query', () => {
  const rows = [gapRow('A', 'single_source'), gapRow('B', 'isolated')]
  assert.equal(filterGapRows(rows, rowKey, '').length, 2)
  assert.equal(filterGapRows(rows, rowKey, '   ').length, 2)
})

test('filterGapRows returns empty rather than everything when nothing matches', () => {
  const rows = [gapRow('A', 'single_source')]
  assert.deepEqual(filterGapRows(rows, rowKey, 'zzz'), [])
})

test('filterGapRows is pure — the input array is not mutated or aliased', () => {
  const rows = [gapRow('B', 'single_source'), gapRow('A', 'isolated')]
  const out = filterGapRows(rows, rowKey, '')
  assert.notEqual(out, rows)
  assert.deepEqual(
    rows.map((r) => r.label),
    ['B', 'A'],
  )
})

// --- graph coverage ---------------------------------------------------------------------------

const cov = (over: Record<string, number> = {}) =>
  graphCoverage({
    n_documents_in_skeleton: 30,
    n_documents_in_library: 98,
    n_concepts_in_db: 13,
    ...over,
  })

test('coverage states the fraction and the rule that produces it', () => {
  assert.equal(
    cov(),
    'Covers 30 of your 98 documents — a document appears once it mentions one of the 13 concepts on your graph.',
  )
})

test('coverage never implies the rest are waiting for a rebuild', () => {
  // The whole point: 68 uncited documents mention none of the graph's concepts, so a rebuild
  // changes nothing for them. The sentence must not say "missing", "not yet" or "rebuild".
  const s = cov().toLowerCase()
  for (const word of ['missing', 'not yet', 'rebuild', 'pending']) {
    assert.ok(!s.includes(word), `coverage should not say "${word}": ${s}`)
  }
})

test('a graph covering the whole library says nothing', () => {
  assert.equal(cov({ n_documents_in_skeleton: 98 }), '')
})

test('an empty library says nothing', () => {
  assert.equal(cov({ n_documents_in_skeleton: 0, n_documents_in_library: 0 }), '')
})

test('one concept is not "one of the 1 concepts"', () => {
  assert.match(cov({ n_concepts_in_db: 1 }), /mentions the one concept on your graph\.$/)
})
