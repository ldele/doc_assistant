// Tests for the Connections presentation helpers (REVIEW 2026-08-12 §2b R1).
//
// The behaviour worth pinning is negative: nothing here produces a similarity *number*. The panel
// deliberately shows a rank because the underlying score is not a meaningful distance — 750 edges,
// median 0.918, against a 0.5 threshold.

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { ordinal, rankLabel, RELATED_CAVEAT } from './connections.ts'

test('ordinal handles the common cases', () => {
  assert.equal(ordinal(1), '1st')
  assert.equal(ordinal(2), '2nd')
  assert.equal(ordinal(3), '3rd')
  assert.equal(ordinal(4), '4th')
  assert.equal(ordinal(21), '21st')
  assert.equal(ordinal(22), '22nd')
  assert.equal(ordinal(23), '23rd')
})

test('ordinal handles the 11/12/13 exceptions the last-digit rule gets wrong', () => {
  assert.equal(ordinal(11), '11th')
  assert.equal(ordinal(12), '12th')
  assert.equal(ordinal(13), '13th')
  assert.equal(ordinal(111), '111th')
  assert.equal(ordinal(112), '112th')
})

test('ordinal returns empty rather than nonsense for junk input', () => {
  assert.equal(ordinal(0), '')
  assert.equal(ordinal(-1), '')
  assert.equal(ordinal(NaN), '')
  assert.equal(ordinal(Infinity), '')
})

test('rankLabel is 1-based over a nearest-first list', () => {
  assert.equal(rankLabel(0), '1st')
  assert.equal(rankLabel(1), '2nd')
  assert.equal(rankLabel(9), '10th')
})

test('the caveat tells the user to read order, not distance', () => {
  assert.match(RELATED_CAVEAT, /order, not the distance/)
  // And it must not itself quote a number that would be over-read.
  assert.doesNotMatch(RELATED_CAVEAT, /\d\.\d/)
})
