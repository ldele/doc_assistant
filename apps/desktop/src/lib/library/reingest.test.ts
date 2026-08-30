// Tests for the re-run control's arithmetic (ADR-048).
//
// The cost statement is the load-bearing one. The parts span four orders of magnitude, so the
// summary must quote the ceiling rather than an average — an average describes none of them, and
// the user pressing the button is deciding whether they can afford the worst case.

import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  costSummary,
  fraction,
  needsConfirmation,
  orderedParts,
  outcomesByDocument,
} from './reingest.ts'
import type { ReingestOptions, ReingestOutcome, ReingestPart, ReingestStatus } from '../core/types/library.ts'

const part = (id: string, cost: string, moves = false): ReingestPart => ({
  id,
  label: id,
  blurb: '',
  cost,
  moves_identity: moves,
})

// Registry order is cheapest-first, exactly as the server serves it.
const OPTIONS: ReingestOptions = {
  parts: [
    part('metadata', 'instant'),
    part('figures', 'a few seconds'),
    part('references', 'about 10 seconds'),
    part('text', '30 seconds to a few minutes', true),
  ],
  corpus_wide: ['Connections'],
}

test('selection is reported in registry order, not click order', () => {
  const picked = orderedParts(OPTIONS, new Set(['text', 'metadata']))
  assert.deepEqual(
    picked.map((p) => p.id),
    ['metadata', 'text'],
  )
})

test('no options loaded yet is an empty selection, not a crash', () => {
  assert.deepEqual(orderedParts(null, new Set(['text'])), [])
})

test('only an identity-moving part asks for confirmation', () => {
  assert.equal(needsConfirmation(orderedParts(OPTIONS, new Set(['metadata', 'references']))), false)
  assert.equal(needsConfirmation(orderedParts(OPTIONS, new Set(['metadata', 'text']))), true)
})

test('the cost quoted is the ceiling, never an average', () => {
  const summary = costSummary(orderedParts(OPTIONS, new Set(['metadata', 'text'])), 1)
  assert.match(summary, /30 seconds to a few minutes/)
  assert.doesNotMatch(summary, /instant/, 'quoting the cheapest part would understate the run')
})

test('a single cheap part is quoted plainly, with no "slowest part" hedge', () => {
  const summary = costSummary(orderedParts(OPTIONS, new Set(['metadata'])), 1)
  assert.match(summary, /instant/)
  assert.doesNotMatch(summary, /slowest/)
})

test('the document count is stated separately, because row 21 multiplies it', () => {
  assert.match(costSummary(orderedParts(OPTIONS, new Set(['text'])), 1), /^1 document/)
  const many = costSummary(orderedParts(OPTIONS, new Set(['text'])), 12)
  assert.match(many, /^12 documents/)
  assert.match(many, /each/, 'a multi-document run must say the cost is per document')
})

test('the sentence is built around the cost phrase, not prefixed onto it', () => {
  // The costs are whole phrases already ('instant', 'about 10 seconds'), so a naive
  // `about ${cost}` produced "about instant, per document." — caught by driving the real dialog.
  assert.equal(costSummary(orderedParts(OPTIONS, new Set(['metadata'])), 1), '1 document · instant.')
  assert.doesNotMatch(costSummary(orderedParts(OPTIONS, new Set(['references'])), 1), /about about/)
})

test('an empty selection says so rather than quoting a cost', () => {
  assert.equal(costSummary([], 3), 'Nothing selected.')
  assert.equal(costSummary(orderedParts(OPTIONS, new Set(['text'])), 0), 'No documents selected.')
})

test('outcomes group by document, keeping the order they ran in', () => {
  const outcomes: ReingestOutcome[] = [
    { document_id: 'b', filename: 'b.pdf', part: 'metadata', status: 'ok', detail: '' },
    { document_id: 'a', filename: 'a.pdf', part: 'metadata', status: 'ok', detail: '' },
    { document_id: 'b', filename: 'b.pdf', part: 'text', status: 'skipped', detail: 'why' },
  ]
  const grouped = outcomesByDocument(outcomes)
  assert.deepEqual(
    grouped.map((g) => g.documentId),
    ['b', 'a'],
  )
  assert.equal(grouped[0].parts.length, 2)
  assert.equal(grouped[0].filename, 'b.pdf')
})

const status = (over: Partial<ReingestStatus> = {}): ReingestStatus => ({
  state: 'running',
  total: 4,
  done: 1,
  current: null,
  ok: 0,
  skipped: 0,
  errors: 0,
  message: null,
  outcomes: [],
  ...over,
})

test('an uncounted run has NO fraction — null, never zero', () => {
  // Zero draws a bar that says "no progress"; null draws none. The difference is between a
  // measurement and an absence of one, and it is the same rule the ingest bar follows.
  assert.equal(fraction(status({ total: 0 })), null)
  assert.equal(fraction(null), null)
  assert.equal(fraction(status({ state: 'done', total: 4, done: 4 })), null)
})

test('a running count is a real fraction, clamped', () => {
  assert.equal(fraction(status({ done: 1, total: 4 })), 0.25)
  assert.equal(fraction(status({ done: 9, total: 4 })), 1)
})
