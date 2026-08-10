// The Chunks block's per-document session memory: the reader opened it here, so coming back
// via the top ← → arrows should find it open — without the collapsed-by-default block losing
// the property it exists for (opening a document costs zero chunk bytes).
import { test } from 'node:test'
import assert from 'node:assert/strict'

import { forgetChunkState, rememberChunksOpen, wereChunksOpen } from './chunkmemory.ts'

test('a document is closed until the reader opens it', () => {
  forgetChunkState()
  assert.equal(wereChunksOpen('doc-a'), false)
  rememberChunksOpen('doc-a', true)
  assert.equal(wereChunksOpen('doc-a'), true)
})

test('the memory is per document — opening one does not open the next', () => {
  // The regression that would undo the whole block: a global "chunks are open" flag would
  // restore the eager render on every document the reader navigates to.
  forgetChunkState()
  rememberChunksOpen('doc-a', true)
  assert.equal(wereChunksOpen('doc-b'), false)
})

test('collapsing forgets it, so back/forward respects the last thing the reader did', () => {
  forgetChunkState()
  rememberChunksOpen('doc-a', true)
  rememberChunksOpen('doc-a', false)
  assert.equal(wereChunksOpen('doc-a'), false)
})

test('no document open (null id) is never remembered as open', () => {
  forgetChunkState()
  assert.equal(wereChunksOpen(null), false)
})
