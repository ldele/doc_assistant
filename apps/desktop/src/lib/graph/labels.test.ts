import assert from 'node:assert/strict'
import { test } from 'node:test'

import { broadSummary, formsChangedNotice, shownLabel, shownName } from './labels.ts'

test('a concept the library writes in a case is shown as written (ADR-053 decision 3)', () => {
  assert.equal(shownLabel({ label: 'din', written: 'dIN' }), 'dIN')
  assert.equal(shownLabel({ label: 'cre', written: 'Cre' }), 'Cre')
})

test('without a written form the stored label is shown', () => {
  assert.equal(shownLabel({ label: 'beta' }), 'beta')
  assert.equal(shownLabel({ label: 'beta', written: null }), 'beta')
})

test('a payload with its own field names follows the same rule (ADR-054)', () => {
  assert.equal(shownName('cre', 'Cre'), 'Cre')
  assert.equal(shownName('beta', null), 'beta')
  assert.equal(shownName('beta'), 'beta')
})

test('the broad count is stated beside presence, with the forms that carry it', () => {
  assert.equal(broadSummary(7, ['distillation']), '7 more through the broad form distillation')
  assert.equal(
    broadSummary(3, ['contrastive', 'nce']),
    '3 more through the broad forms contrastive, nce',
  )
})

test('nothing beside presence says nothing', () => {
  assert.equal(broadSummary(0, []), '')
  assert.equal(broadSummary(0, ['distillation']), '')
  assert.equal(broadSummary(2, []), '')
})

test('the graph names the concepts whose forms changed since it was built', () => {
  assert.equal(formsChangedNotice([]), '')
  assert.equal(
    formsChangedNotice(['Cre']),
    'Name or forms changed since this graph was built: Cre.',
  )
  assert.equal(
    formsChangedNotice(['Cre', 'dIN', 'pose']),
    'Name or forms changed since this graph was built: Cre, dIN, pose.',
  )
})

test('a graph with no record of its forms says it cannot tell, and claims no change', () => {
  const notice = formsChangedNotice([], false)
  assert.match(notice, /cannot tell whether they changed/)
  assert.doesNotMatch(notice, /changed since/)
  // Names are not listed: with no record there is no comparison to have produced them.
  assert.equal(formsChangedNotice(['Cre'], false), notice)
})

test('past three names the notice counts the rest instead of listing them', () => {
  assert.equal(
    formsChangedNotice(['a', 'b', 'c', 'd', 'e']),
    'Name or forms changed since this graph was built: a, b, c and 2 more.',
  )
})
