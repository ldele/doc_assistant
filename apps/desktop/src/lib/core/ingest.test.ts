import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  currentLabel,
  fraction,
  ingestLabel,
  isCompletion,
  pendingStatus,
} from './ingest.ts'
import type { IngestStatus } from './types/sources.ts'

function status(over: Partial<IngestStatus> = {}): IngestStatus {
  return {
    state: 'running',
    source_dir: '/library',
    added: 0,
    skipped: 0,
    errors: 0,
    message: null,
    total: 0,
    done: 0,
    current: null,
    ...over,
  }
}

test('fraction is null, not zero, when the batch has not been counted', () => {
  // The distinction the bar depends on: "no fraction yet" must render as indeterminate, while 0
  // would render as an empty bar — indistinguishable from a run that is stuck at the first file.
  assert.equal(fraction(status({ total: 0, done: 0 })), null)
  assert.equal(fraction(null), null)
  assert.equal(fraction(status({ total: 4, done: 0 })), 0)
})

test('fraction is a real ratio once there is a total', () => {
  assert.equal(fraction(status({ total: 4, done: 1 })), 0.25)
  assert.equal(fraction(status({ total: 12, done: 12 })), 1)
})

test('fraction never leaves 0..1 even if the backend contradicts itself', () => {
  // Defensive: a bar wider than its track is a visual break, and the UI cannot re-derive the truth.
  assert.equal(fraction(status({ total: 4, done: 9 })), 1)
  assert.equal(fraction(status({ total: 4, done: -2 })), 0)
})

test('a running run reports position, never an outcome', () => {
  // `added` is 0 for the whole run by design; if the label ever read it, it would claim a result
  // for documents that have not been processed.
  assert.equal(ingestLabel(status({ total: 12, done: 4, added: 0 })), 'Indexing 4 of 12')
})

test('an uncounted run says so instead of showing 0 of 0', () => {
  assert.equal(ingestLabel(status({ total: 0, done: 0 })), 'Preparing to index…')
})

test('a finished run that added nothing does not read as a failure', () => {
  assert.equal(ingestLabel(status({ state: 'done', added: 0, total: 9, done: 9 })), 'Nothing new to index')
  assert.equal(ingestLabel(status({ state: 'done', added: 3, total: 3, done: 3 })), 'Indexed 3 new')
})

test('a failed run is named as failed', () => {
  assert.equal(ingestLabel(status({ state: 'error', message: 'disk full' })), 'Indexing failed')
})

test('a missing status still says something rather than rendering blank', () => {
  assert.equal(ingestLabel(null), 'Indexing…')
})

test('the optimistic pending status carries no fraction', () => {
  // It exists so the bar can appear on the click that started the run; claiming a position it
  // has not been told would be inventing one.
  const p = pendingStatus('/library')
  assert.equal(p.state, 'running')
  assert.equal(fraction(p), null)
  assert.equal(ingestLabel(p), 'Preparing to index…')
  assert.equal(p.source_dir, '/library')
})

test('the file in flight distinguishes starting, between, and finishing', () => {
  // All three report `current: null`, and calling them all "starting…" put "Now: starting…" next
  // to "2 of 2" in the panel, which reads as a run that is stuck.
  assert.equal(currentLabel(status({ total: 3, done: 0, current: null })), 'starting…')
  assert.equal(currentLabel(status({ total: 3, done: 1, current: null })), '…')
  assert.equal(currentLabel(status({ total: 3, done: 3, current: null })), 'finishing…')
})

test('a named file is shown verbatim', () => {
  assert.equal(currentLabel(status({ total: 3, done: 1, current: 'hebb_1949.pdf' })), 'hebb_1949.pdf')
})

test('an uncounted run still says it is starting rather than rendering blank', () => {
  assert.equal(currentLabel(status({ total: 0, done: 0, current: null })), 'starting…')
  assert.equal(currentLabel(null), '…')
})

test('a completion is a transition, never a state', () => {
  // The guard that keeps the app from refreshing on every launch: a `done` left behind by a run
  // that ended hours ago is the state the very first poll sees, and it is not an event.
  const running = status({ state: 'running' })
  const done = status({ state: 'done', added: 2 })

  assert.equal(isCompletion(null, done), false, 'first poll after launch is not a completion')
  assert.equal(isCompletion(done, done), false, 'a resting done stays resting')
  assert.equal(isCompletion(running, running), false, 'still in flight')
  assert.equal(isCompletion(running, done), true)
})

test('a failed run counts as a completion too', () => {
  // It also changes what is on disk — a partial run may have indexed some documents before it
  // died — so the surfaces that show the corpus have to re-read either way.
  assert.equal(isCompletion(status({ state: 'running' }), status({ state: 'error' })), true)
})

test('a run that goes straight to idle still completes', () => {
  assert.equal(isCompletion(status({ state: 'running' }), status({ state: 'idle' })), true)
})
