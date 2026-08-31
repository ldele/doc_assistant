import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import { nothingToAdd, skippedSentence } from './catalogue.ts'

test('the dominant reason leads', () => {
  const sentence = skippedSentence({
    'in the Zotero trash': 3,
    'a web-page snapshot': 412,
    'not downloaded to this computer': 88,
  })
  assert.equal(
    sentence,
    '412 a web-page snapshot · 88 not downloaded to this computer · 3 in the Zotero trash',
  )
})

test('ties are ordered by reason, so the sentence is stable between runs', () => {
  assert.equal(skippedSentence({ beta: 2, alpha: 2 }), '2 alpha · 2 beta')
})

test('a zero count is not worth a clause', () => {
  assert.equal(skippedSentence({ 'in the Zotero trash': 0, 'a web-page snapshot': 4 }), '4 a web-page snapshot')
})

test('no reasons at all is the empty string, not "undefined"', () => {
  assert.equal(skippedSentence({}), '')
})

test('an empty library and a fully filtered one say different things', () => {
  // The two need different next steps from the user; one message for both would hide which
  // happened.
  assert.equal(nothingToAdd({}), 'That Zotero library has no documents in it yet.')
  assert.equal(
    nothingToAdd({ 'a web-page snapshot': 12 }),
    'Nothing to add from that library — 12 a web-page snapshot.',
  )
})
