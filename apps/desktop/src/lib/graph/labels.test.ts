import assert from 'node:assert/strict'
import { test } from 'node:test'

import { shownLabel } from './labels.ts'

test('a concept the library writes in a case is shown as written (ADR-053 decision 3)', () => {
  assert.equal(shownLabel({ label: 'din', written: 'dIN' }), 'dIN')
  assert.equal(shownLabel({ label: 'cre', written: 'Cre' }), 'Cre')
})

test('without a written form the stored label is shown', () => {
  assert.equal(shownLabel({ label: 'beta' }), 'beta')
  assert.equal(shownLabel({ label: 'beta', written: null }), 'beta')
})
