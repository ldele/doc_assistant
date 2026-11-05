// Tests for list multi-select (selection.ts; ROADMAP 78, 80).
// Runner: node's built-in `node:test` with native TypeScript stripping. `npm test` from apps/desktop.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import { isSelectClick, pick, selectAllLabel, selectAllOrNone } from './selection.ts'

const order = ['a', 'b', 'c', 'd', 'e']

test('a plain click toggles one row and becomes the anchor', () => {
  assert.deepEqual(pick(order, [], null, 'c', false), { selected: ['c'], anchor: 'c' })
  assert.deepEqual(pick(order, ['c'], 'c', 'c', false), { selected: [], anchor: 'c' })
  assert.deepEqual(pick(order, ['a'], 'a', 'd', false), { selected: ['a', 'd'], anchor: 'd' })
})

test('a Shift-click selects every row between the anchor and the clicked row', () => {
  assert.deepEqual(pick(order, ['b'], 'b', 'd', true).selected, ['b', 'c', 'd'])
  // Upwards too, appended in list order.
  assert.deepEqual(pick(order, ['d'], 'd', 'b', true).selected, ['d', 'b', 'c'])
})

test('a Shift-click keeps what was selected outside the range', () => {
  assert.deepEqual(pick(order, ['a', 'c'], 'c', 'e', true).selected, ['a', 'c', 'd', 'e'])
})

test('a Shift-click on a selected row unselects the range, and only the range', () => {
  const r = pick(order, ['a', 'b', 'c', 'd'], 'b', 'd', true)
  assert.deepEqual(r.selected, ['a'])
  assert.equal(r.anchor, 'd')
})

test('a Shift-click with no anchor, or an anchor no longer shown, toggles the row alone', () => {
  assert.deepEqual(pick(order, [], null, 'c', true).selected, ['c'])
  assert.deepEqual(pick(order, ['x'], 'x', 'c', true).selected, ['x', 'c'])
})

test('a Shift-click from outside select mode ranges from the open row', () => {
  // Chat: nothing selected yet, the open conversation is the anchor.
  assert.deepEqual(pick(order, [], 'b', 'e', true).selected, ['b', 'c', 'd', 'e'])
})

test('Shift, Ctrl and Cmd each ask for selection; a bare click does not', () => {
  const none = { shiftKey: false, ctrlKey: false, metaKey: false }
  assert.equal(isSelectClick(none), false)
  assert.equal(isSelectClick({ ...none, shiftKey: true }), true)
  assert.equal(isSelectClick({ ...none, ctrlKey: true }), true)
  assert.equal(isSelectClick({ ...none, metaKey: true }), true)
})

test('the helper says Select all on an empty selection and Select none otherwise', () => {
  assert.equal(selectAllLabel([]), 'Select all')
  assert.equal(selectAllLabel(['b']), 'Select none')
  assert.deepEqual(selectAllOrNone(order, []), order)
  assert.deepEqual(selectAllOrNone(order, ['b']), [])
  // Select none also drops a row a filter hides: nothing stays selected behind the user's back.
  assert.deepEqual(selectAllOrNone(['a', 'b'], ['z']), [])
})
