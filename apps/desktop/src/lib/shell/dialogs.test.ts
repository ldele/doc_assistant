// Tests for the list of open dialogs (dialogs.ts; ADR-063 choice 8).
// Runner: node's built-in `node:test` with native TypeScript stripping. `npm test` from apps/desktop.
import { test, beforeEach } from 'node:test'
import assert from 'node:assert/strict'

import {
  BASE_LAYER,
  closeDialog,
  isFront,
  layerOf,
  mayRestoreTo,
  openCount,
  openDialog,
  resetDialogs,
  wrapTab,
} from './dialogs.ts'

beforeEach(() => resetDialogs())

test('Esc belongs to the dialog in front: a question asked from a dialog takes it', () => {
  const settings = openDialog()
  assert.equal(isFront(settings), true)
  const confirm = openDialog()
  assert.equal(isFront(confirm), true)
  assert.equal(isFront(settings), false)
  closeDialog(confirm)
  assert.equal(isFront(settings), true)
  closeDialog(settings)
  assert.equal(openCount(), 0)
  assert.equal(isFront(settings), false)
})

test('a dialog opened from another sits two layers above it', () => {
  const a = openDialog()
  const b = openDialog()
  assert.equal(layerOf(a), BASE_LAYER)
  assert.equal(layerOf(b), BASE_LAYER + 2)
})

test('a parent that closes before its question leaves the question in front', () => {
  const settings = openDialog()
  const indexing = openDialog()
  closeDialog(settings)
  assert.equal(isFront(indexing), true)
  assert.equal(openCount(), 1)
  closeDialog(settings) // closing twice is harmless
  assert.equal(openCount(), 1)
})

test('focus goes back to the opener only when nothing else holds the front', () => {
  const button = { isConnected: true }
  assert.equal(mayRestoreTo(button, null), true)
  assert.equal(mayRestoreTo({ isConnected: false }, null), false)
  assert.equal(mayRestoreTo(null, null), false)
  // A dialog is still open: only if the opener is inside it.
  assert.equal(mayRestoreTo(button, (o) => o === button), true)
  assert.equal(mayRestoreTo(button, () => false), false)
})

test('Tab wraps at both ends and leaves the middle to the browser', () => {
  const els = ['a', 'b', 'c']
  assert.equal(wrapTab(els, 'c', false, 'frame'), 'a')
  assert.equal(wrapTab(els, 'a', true, 'frame'), 'c')
  assert.equal(wrapTab(els, 'b', false, 'frame'), null)
  assert.equal(wrapTab(els, 'b', true, 'frame'), null)
  // Focus on the frame itself counts as the start: Tab goes on to the first control by itself,
  // Shift+Tab would leave the dialog, so it goes to the last.
  assert.equal(wrapTab(els, 'frame', false, 'frame'), null)
  assert.equal(wrapTab(els, 'frame', true, 'frame'), 'c')
  assert.equal(wrapTab([], 'x', false, 'frame'), null)
})
