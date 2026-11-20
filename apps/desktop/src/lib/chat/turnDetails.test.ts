import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import {
  RECORDS_TAG,
  anythingLeftOut,
  carriedRows,
  decisionLines,
  recordsLine,
  reviewLine,
} from './turnDetails.ts'

const ctx = {
  window: 8192,
  window_source: 'server',
  rewrite: { history_used: 6, history_dropped: 3, answer_shortened: false, skipped: false },
  answer: { passages_used: 8, passages_dropped: 2, top_passage_clipped: true },
  carried: { exchanges: 4, kept: 2, dropped: 2 },
}

test('the carried card shows A5\'s counts and the carried exchanges, as recorded', () => {
  assert.deepEqual(carriedRows(ctx, 'Research assistant', 'promptv1-later'), [
    { key: 'Earlier messages', value: '6 of 9 · the 3 oldest left out' },
    { key: 'Carried', value: '2 of 4 carried exchanges · the 2 oldest left out' },
    { key: 'Passages', value: '8 of 10 · 2 left out · the best one shortened' },
    { key: 'Window', value: "8,192 tokens · reported by the model's server" },
    { key: 'Mode', value: 'Research assistant · prompt version promptv1' },
  ])
  assert.ok(anythingLeftOut(ctx))
})

test('a first turn with nothing left out, on the floor window', () => {
  const first = {
    window: 8192,
    window_source: 'floor',
    answer: { passages_used: 3, passages_dropped: 0, top_passage_clipped: false },
  }
  assert.deepEqual(carriedRows(first, null, null), [
    { key: 'Earlier messages', value: 'none' },
    { key: 'Passages', value: '3 of 3' },
    { key: 'Window', value: "8,192 tokens · the server does not state it, so the app's floor was used" },
  ])
  assert.ok(anythingLeftOut(first)) // the floor is said, like the fitting note says it
  assert.ok(!anythingLeftOut({ ...first, window_source: 'documented' }))
  assert.deepEqual(carriedRows(null, 'Critical reading', null), [{ key: 'Mode', value: 'Critical reading' }])
})

test('the records card says what it looked up', () => {
  assert.equal(
    recordsLine({ lookup: 'list_documents', args: { folder: 'Thesis ch.2' }, rows: 14 }),
    'list_documents · folder Thesis ch.2 · 14 rows',
  )
  assert.equal(recordsLine({ lookup: 'count_only', args: {}, rows: 1 }), 'count_only · 1 row')
  assert.equal(RECORDS_TAG, "Answered from the library's records · no model, no passages")
})

test('a reopened chat shows the decisions and the reviewer verdict as recorded', () => {
  assert.deepEqual(
    decisionLines([
      { n: 1, decision: 'pending', edited_text: null },
      { n: 2, decision: 'accepted', edited_text: null },
      { n: 3, decision: 'edited', edited_text: 'It may cause slowing.' },
      { n: 4, decision: 'rejected', edited_text: null },
    ]),
    ['#2 accepted', '#3 edited: “It may cause slowing.”', '#4 rejected'],
  )
  assert.equal(
    reviewLine({ faithfulness: 4, citation_density: 3, hedging_adequacy: 5, unsupported_claims: 1, error: null }),
    'Reviewer, as recorded: faithfulness 4/5 · citation 3/5 · hedging 5/5 · 1 claim it found unsupported',
  )
  assert.equal(
    reviewLine({ faithfulness: null, citation_density: null, hedging_adequacy: null, unsupported_claims: null, error: 'timeout' }),
    'Reviewer, as recorded: no verdict (timeout)',
  )
})
