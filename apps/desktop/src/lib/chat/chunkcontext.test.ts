// Tests for what the "where does this sit?" panel SAYS (ROADMAP 19).
//
// The load-bearing ones are about absence. A citation panel that invents a page number, or shows
// "0% of the way in" for a document it could not measure, is the same failure the ingest side
// refuses — a confident answer that is not backed by anything.

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { elision, position, tidy, whereLabel } from './chunkcontext.ts'
import type { ChunkContext } from '../core/types/library.ts'

const ctx = (over: Partial<ChunkContext> = {}): ChunkContext => ({
  document_id: 'd',
  filename: 'paper.pdf',
  text: 'the cited passage',
  before: 'what came before',
  after: 'what came after',
  char_start: 2500,
  char_end: 2700,
  doc_chars: 10000,
  page: null,
  at_document_start: false,
  at_document_end: false,
  ...over,
})

test('position is the fraction of the document before the passage', () => {
  assert.equal(position(ctx()), 0.25)
  assert.equal(position(ctx({ char_start: 0 })), 0)
})

test('an unmeasurable document has NO position, not zero', () => {
  // A zero-length cache is not "0% in" — it is unmeasurable, and 0 would render as a real answer.
  assert.equal(position(ctx({ doc_chars: 0 })), null)
  assert.equal(position(null), null)
  assert.equal(whereLabel(ctx({ doc_chars: 0 })), 'Position unknown')
})

test('position is clamped, so a stale span cannot render past 100%', () => {
  assert.equal(position(ctx({ char_start: 99999 })), 1)
})

test('the label leads with the position, because that is what is always known', () => {
  assert.equal(whereLabel(ctx()), '25% of the way in')
})

test('a page is shown only when there IS one', () => {
  // Most chat citations are parent-child parents, which record no page. "Page ?" would be worse
  // than saying nothing about pages.
  assert.equal(whereLabel(ctx({ page: 7 })), 'Page 7 · 25% of the way in')
  assert.doesNotMatch(whereLabel(ctx({ page: null })), /page/i)
})

test('an ellipsis marks cut-off text, never a window that reached the edge', () => {
  assert.deepEqual(elision(ctx()), { before: true, after: true })
  assert.deepEqual(
    elision(ctx({ at_document_start: true, at_document_end: true })),
    { before: false, after: false },
    'a complete window must not claim there is more to read',
  )
})

test('tidy collapses extraction noise without touching the words', () => {
  const raw = 'First line.\n\n\n\n<!-- page:12 -->\n\nSecond line.   \nThird.'
  const out = tidy(raw)
  assert.match(out, /First line\./)
  assert.match(out, /Second line\./)
  assert.match(out, /Third\./)
  assert.doesNotMatch(out, /page:12/, 'page markers are extraction plumbing, not prose')
  assert.doesNotMatch(out, /\n{3,}/, 'runs of blank lines make a small panel unreadable')
})

test('tidy leaves ordinary prose completely alone', () => {
  const prose = 'A sentence.\n\nAnother sentence.'
  assert.equal(tidy(prose), prose)
})
