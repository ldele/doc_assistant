// Tests for the letter-spacing block's decisions (ADR-056 points 9 and 10).
//
// The block and the editor both read a run as gaps between its glyphs. If that reading is wrong
// the ticks land in the wrong place and the editor writes a spacing the backend refuses, so the
// cases are real runs from the library (Cajal's title and running head) and the edges around them.

import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  gapsOf,
  glyphsOf,
  markRemovedSpaces,
  removedCount,
  removedNote,
  rowCaption,
  shownText,
  splitRows,
  writeWithGaps,
} from './spacing.ts'

const TITLE = 'S A N T I A G O R A M Ó N Y C A J A L'
const HEAD = 'N E U R O N S : S T R U C T U R E A N D C O N N E X I O N S'

test('the glyphs of a run are its single characters', () => {
  assert.deepEqual(glyphsOf('N E U'), ['N', 'E', 'U'])
})

test("Cajal's title closes with 15 spaces removed and its word breaks kept", () => {
  const written = 'SANTIAGO RAMÓN Y CAJAL'
  assert.equal(removedCount(TITLE, written), 15)
  const marked = markRemovedSpaces(TITLE, written)
  assert.equal(marked.map((g) => (g.spaceBefore ? ' ' : '') + g.char).join(''), written)
  assert.equal(marked.filter((g) => g.spaceBefore).length, 3)
})

test('the running head keeps its word breaks and closes the colon onto its word', () => {
  assert.equal(removedCount(HEAD, 'NEURONS: STRUCTURE AND CONNEXIONS'), 26)
})

test('a run left as extracted has no ticks', () => {
  assert.equal(removedCount('N E U', 'N E U'), 0)
  assert.ok(markRemovedSpaces('N E U', 'N E U').every((g) => !g.removedBefore))
})

test('a written form that changes a glyph is not read as gaps', () => {
  assert.equal(gapsOf('N E U', 'NEW'), null)
  assert.equal(gapsOf('N E U', ' NEU'), null)
  assert.equal(gapsOf('N E U', 'NEUX'), null)
  // shown as written, unmarked, rather than misplaced ticks
  assert.deepEqual(
    markRemovedSpaces('N E U', 'NEW').map((g) => g.removedBefore),
    [false, false, false],
  )
})

test('the editor writes back exactly the gaps it shows', () => {
  const gaps = gapsOf(TITLE, 'SANTIAGO RAMÓN Y CAJAL')
  assert.ok(gaps)
  assert.equal(writeWithGaps(TITLE, gaps), 'SANTIAGO RAMÓN Y CAJAL')
  const broken = [...gaps]
  broken[2] = true // break after "SAN"
  assert.equal(writeWithGaps(TITLE, broken), 'SAN TIAGO RAMÓN Y CAJAL')
})

test('rows sit where their pending change puts them', () => {
  const rows = [
    { id: 'a', state: 'closed', pending: null },
    { id: 'b', state: 'closed', pending: 'kept' },
    { id: 'c', state: 'kept', pending: 'closed' },
    { id: 'd', state: 'yours', pending: null },
    { id: 'e', state: 'maths', pending: null },
  ]
  const { closed, kept } = splitRows(rows)
  assert.deepEqual(closed.map((r) => r.id), ['a', 'c', 'e'])
  assert.deepEqual(kept.map((r) => r.id), ['b', 'd'])
})

test('the caption says how many spaces went and where', () => {
  assert.equal(
    rowCaption({ spaced: TITLE, written: 'SANTIAGO RAMÓN Y CAJAL', pages: [1] }),
    '15 spaces removed · page 1',
  )
  assert.equal(
    rowCaption({ spaced: 'N E U', written: 'N E U', pages: [2, 4] }),
    'as extracted · 2 pages',
  )
  assert.equal(rowCaption({ spaced: 'A B', written: 'AB', pages: [] }), '1 space removed · no page recorded')
})

test('a row shows the text the next re-run will write', () => {
  const base = { spaced: 'N E U', rule: 'NEU', written: 'NEU', chosen: null }
  assert.equal(shownText({ ...base, state: 'closed', pending: null }), 'NEU')
  assert.equal(shownText({ ...base, state: 'closed', pending: 'kept' }), 'N E U')
  assert.equal(shownText({ ...base, state: 'closed', pending: 'yours', chosen: 'NE U' }), 'NE U')
  assert.equal(shownText({ ...base, written: 'N E U', state: 'kept', pending: 'closed' }), 'NEU')
})

test('a removal says what happens next, and never counts zero documents', () => {
  assert.equal(removedNote(1, 1), "Removed 1 entry. It takes effect when that document's text is re-run.")
  assert.equal(removedNote(3, 2), "Removed 3 entries. They take effect when the 2 documents' text is re-run.")
  assert.equal(removedNote(1, 0), 'Removed 1 entry. No document in this library is affected.')
})
