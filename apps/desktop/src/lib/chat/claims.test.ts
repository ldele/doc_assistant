import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import { CITED_TOOLTIP, CLAIM_CITE_BODY, WEAK_TOOLTIP, claimKind, claimWords, locateClaims, passagesFor } from './claims.ts'
import { CITE_BODY } from './citations.ts'

test('claims read citations exactly as citations.ts does', () => {
  assert.equal(CLAIM_CITE_BODY, CITE_BODY)
})

const claim = (marker: string, sources: number[], text = 'x') => ({ marker, sources, text })

test('colours say only what the marker measures: lilac cited, amber weak, uncited plain (G4)', () => {
  assert.equal(claimKind(claim('ok', [1])), 'cited')
  assert.equal(claimKind(claim('weak', [2])), 'weak')
  assert.equal(claimKind(claim('weak', [])), 'weak')
  assert.equal(claimKind(claim('unsupported', [])), null)
  assert.equal(claimKind(claim('ok', [])), null)
  assert.equal(WEAK_TOOLTIP, 'Cites a passage that ranked low for this question')
  assert.ok(!/reviewer|verified|true/i.test(WEAK_TOOLTIP + CITED_TOOLTIP))
})

test('a claim is read as shown: no citations, emphasis, list or link markup', () => {
  assert.deepEqual(claimWords('**Release** falls with age [1].'), ['Release', 'falls', 'with', 'age', '.'])
  assert.deepEqual(claimWords('- See [the review](http://x.org) [Sources 2, 4].'), ['See', 'the', 'review', '.'])
  assert.deepEqual(claimWords('1. `D2` binding drops [3]'), ['D2', 'binding', 'drops'])
})

// The answer as the screen shows it: the citation buttons' text is left out of the flat text.
const flat = 'Release falls with age . Density drops too . This causes slowing.'

test('claims are matched by text, in order', () => {
  const got = locateClaims(flat, [
    'Release falls with age [1].',
    'Density drops too [2].',
    'This causes slowing.',
  ])
  assert.deepEqual(
    got.map((r) => (r ? flat.slice(r.start, r.end) : null)),
    ['Release falls with age .', 'Density drops too .', 'This causes slowing.'],
  )
})

test('an unmatched claim is not highlighted and the next is still found', () => {
  const got = locateClaims(flat, ['A sentence the text does not hold [1].', 'Density drops too [2].'])
  assert.equal(got[0], null)
  assert.equal(got[1] && flat.slice(got[1].start, got[1].end), 'Density drops too .')
})

test('a sentence written twice is matched to its own place each time', () => {
  const twice = 'It falls . Then more. It falls .'
  const got = locateClaims(twice, ['It falls [1].', 'Then more.', 'It falls [2].'])
  assert.deepEqual(got.map((r) => r?.start), [0, 11, 22])
})

test('an empty claim and an empty answer match nothing', () => {
  assert.deepEqual(locateClaims(flat, ['[1]']), [null])
  assert.deepEqual(locateClaims('', ['Release falls.']), [null])
})

test('a claim opens every passage it cites, in its own order (walk item 25)', () => {
  const sources = [{ n: 1 }, { n: 2 }, { n: 3 }]
  assert.deepEqual(passagesFor(sources, [3, 1]).map((s) => s.n), [3, 1])
  assert.deepEqual(passagesFor(sources, [2, 2, 1]).map((s) => s.n), [2, 1])
  assert.deepEqual(passagesFor(sources, [4, 2]).map((s) => s.n), [2])
  assert.deepEqual(passagesFor(sources, []), [])
})
