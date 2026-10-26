import { test } from 'node:test'
import assert from 'node:assert/strict'

import type { ConceptSignals, FormOffer, SignalExample } from '../core/types'
import {
  citedLine,
  exampleSource,
  hasNewOffer,
  hasSignals,
  markedSegments,
  offerLine,
  offerMeta,
  placeLabel,
  placeMeta,
  readScope,
  silentLine,
} from './signals.ts'

const example = (text: string, marks: [number, number][], extra: Partial<SignalExample> = {}): SignalExample => ({
  text,
  marks,
  document_id: 'd1',
  document_title: 'Stimulation in practice',
  chunk_key: 'd1:p0',
  ...extra,
})

const offer = (extra: Partial<FormOffer> = {}): FormOffer => ({
  form: 'deep brain stimulation',
  beside: 'DBS',
  kind: 'long',
  documents: 2,
  known: false,
  initials: true,
  example: example('deep brain stimulation (DBS)', [[0, 22], [24, 27]]),
  ...extra,
})

const signals = (extra: Partial<ConceptSignals> = {}): ConceptSignals => ({
  concept_id: 'c',
  available: true,
  documents_read: 3,
  documents_matched: 3,
  mentions: 4,
  documents: 2,
  offers: [],
  cited: null,
  silent: false,
  places: [],
  ...extra,
})

test('marks cut the text where the server placed them, and only there', () => {
  const text = 'Communicability (CO) measures communication.'
  const segs = markedSegments(example(text, [[17, 19], [0, 15]]))
  assert.deepEqual(segs, [
    { text: 'Communicability', mark: true },
    { text: ' (', mark: false },
    { text: 'CO', mark: true },
    { text: ') measures communication.', mark: false },
  ])
})

test('whitespace is collapsed for reading, and the verbatim text is not changed', () => {
  const ex = example('  Tools and resources\n\nNeuroscience   1 of 21  ', [[12, 35]])
  const segs = markedSegments(ex)
  assert.deepEqual(segs, [
    { text: 'Tools and ', mark: false },
    { text: 'resources Neuroscience', mark: true },
    { text: ' 1 of 21', mark: false },
  ])
  assert.equal(ex.text, '  Tools and resources\n\nNeuroscience   1 of 21  ')
})

test('a mark that overlaps another or runs past the text is dropped, not guessed', () => {
  const segs = markedSegments(example('abcdef', [[0, 3], [2, 4], [5, 9]]))
  assert.deepEqual(segs, [
    { text: 'abc', mark: true },
    { text: 'def', mark: false },
  ])
  assert.deepEqual(markedSegments(example('plain text', [])), [{ text: 'plain text', mark: false }])
})

test('dropped characters leave the text and the marks stay on their words', () => {
  // `_**Moyle et al., 2021**_` with the name marked: the drop set is emphasis.ts's markers.
  const text = '( _**Moyle et al., 2021**_ ).'
  const drop = new Set([2, 3, 4, 23, 24, 25])
  assert.deepEqual(markedSegments(example(text, [[5, 10]]), drop), [
    { text: '( ', mark: false },
    { text: 'Moyle', mark: true },
    { text: ' et al., 2021 ).', mark: false },
  ])
})

test('an offer says which of the row’s forms it stands beside, and whether it is new', () => {
  assert.equal(offerLine(offer()), '“DBS” is spelled out as “deep brain stimulation”')
  assert.equal(offerLine(offer({ kind: 'short', form: 'DBS', beside: 'deep brain stimulation' })), '“deep brain stimulation” is abbreviated as “DBS”')
  assert.equal(offerMeta(offer()), 'in 2 documents')
  assert.equal(offerMeta(offer({ documents: 1, known: true })), 'in 1 document · already one of its forms')
  assert.equal(hasNewOffer(signals({ offers: [offer({ known: true })] })), false)
  assert.equal(hasNewOffer(signals({ offers: [offer({ known: true }), offer()] })), true)
})

test('a cited author shows the count the rule rests on', () => {
  assert.equal(citedLine({ et_al: 18, mentions: 19, example: null }), '18 of 19 mentions are followed by “et al.”')
  assert.equal(citedLine({ et_al: 1, mentions: 1, example: null }), '1 of 1 mention is followed by “et al.”')
})

test('a label no sentence uses is described by its places, never called noise', () => {
  const place = { place: 'table' as const, mentions: 223, documents: 1, example: example('| AP10k |', [[2, 7]]) }
  assert.equal(placeLabel('table'), 'In tables')
  assert.equal(placeLabel('figure'), 'In a figure’s caption or description')
  assert.equal(placeMeta(place), '223 times in 1 document')
  assert.match(silentLine(signals({ silent: true, places: [place] })), /occurs here instead/)
  assert.match(silentLine(signals({ silent: true })), /not found anywhere else/)
  for (const kind of ['sentence', 'table', 'heading', 'figure', 'references', 'elsewhere'] as const) {
    assert.doesNotMatch(placeLabel(kind).toLowerCase(), /noise|junk/)
  }
})

test('the section shows only when there is something to show, and only after a read', () => {
  assert.equal(hasSignals(null), false)
  assert.equal(hasSignals(signals()), false)
  assert.equal(hasSignals(signals({ offers: [offer()] })), true)
  assert.equal(hasSignals(signals({ silent: true })), true)
  assert.equal(hasSignals(signals({ available: false, silent: true })), false)
})

test('a read that stopped short says so; a whole read says nothing', () => {
  assert.equal(readScope(signals()), null)
  assert.equal(
    readScope(signals({ documents_read: 200, documents_matched: 312 })),
    'Read in the 200 documents that mention it most, of 312 the keyword index found.',
  )
})

test('an example names its document, or the library when it has no title', () => {
  assert.equal(exampleSource(example('x', [])), 'From “Stimulation in practice”')
  assert.equal(exampleSource(example('x', [], { document_title: null })), 'From your library')
})
