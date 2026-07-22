import { test } from 'node:test'
import assert from 'node:assert/strict'
import { GAP_META, gapRank, isHiddenByDefault, gapVisible, orderGaps } from './gaps.ts'
import type { GapKind } from './types.ts'

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
