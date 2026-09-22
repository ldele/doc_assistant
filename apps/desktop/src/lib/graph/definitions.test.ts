import { test } from 'node:test'
import assert from 'node:assert/strict'

import type { ConceptDefinitions, DefinitionCandidate } from '../core/types'
import { displayText, gradeLabel, groupCandidates, offGraphMatches, sourceLabel, usageSource } from './definitions.ts'

const cand = (
  id: string,
  status: DefinitionCandidate['status'],
  extra: Partial<DefinitionCandidate> = {},
): DefinitionCandidate => ({
  id,
  text: `text ${id}`,
  source: 'passage',
  status,
  grade: 'some',
  reasons: [],
  document_id: null,
  document_title: null,
  chunk_key: null,
  ...extra,
})

const view = (candidates: DefinitionCandidate[]): ConceptDefinitions => ({
  concept_id: 'c',
  label: 'c',
  chosen_id: candidates.find((c) => c.status === 'chosen')?.id ?? null,
  candidates,
  can_undo: false,
  thin: false,
  extracted: true,
})

test('candidates split into chosen, suggested and dismissed, keeping the server order', () => {
  const g = groupCandidates(
    view([cand('a', 'chosen'), cand('b', 'suggested'), cand('c', 'dismissed'), cand('d', 'suggested')]),
  )
  assert.equal(g.chosen?.id, 'a')
  assert.deepEqual(g.suggested.map((c) => c.id), ['b', 'd'])
  assert.deepEqual(g.dismissed.map((c) => c.id), ['c'])
})

test('no view and no candidates are both an empty state, not a throw', () => {
  assert.deepEqual(groupCandidates(null), { chosen: null, suggested: [], dismissed: [] })
  assert.deepEqual(groupCandidates(view([])), { chosen: null, suggested: [], dismissed: [] })
})

test('a grade reads as evidence, and an unknown one passes through', () => {
  assert.equal(gradeLabel('strong'), 'Strong evidence')
  assert.equal(gradeLabel('user'), 'Your words')
  assert.equal(gradeLabel('new-kind'), 'new-kind')
})

test('the source line names the document when there is one', () => {
  assert.equal(sourceLabel(cand('a', 'suggested', { document_title: 'A survey' })), 'From “A survey”')
  assert.equal(sourceLabel(cand('a', 'suggested')), 'From your library')
  assert.equal(sourceLabel(cand('a', 'suggested', { source: 'user' })), 'Written by you')
})

test('a usage example names its document and how often that document uses the word', () => {
  const u = { text: 't', document_id: 'd', document_title: 'A survey', chunk_key: 'd:p0', doc_mentions: 12 }
  assert.equal(usageSource(u), '“A survey” · uses it 12 times')
  assert.equal(usageSource({ ...u, document_title: null, doc_mentions: 1 }), 'A document in your library · uses it once')
})

test('display collapses whitespace without touching the words', () => {
  assert.equal(
    displayText('  Knowledge\n distillation   refers to\tit. '),
    'Knowledge distillation refers to it.',
  )
})

test('the search shows only what the graph index does not already list', () => {
  const m = [
    { id: 'virus', label: 'virus', on_graph: true, has_definition: false },
    { id: 'viral', label: 'viral', on_graph: false, has_definition: false },
  ]
  assert.deepEqual(
    offGraphMatches(m, new Set(['virus'])).map((x) => x.id),
    ['viral'],
  )
})
