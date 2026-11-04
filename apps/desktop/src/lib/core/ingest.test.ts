import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  currentLabel,
  fraction,
  groupChips,
  indexLocked,
  ingestLabel,
  isCompletion,
  pendingStatus,
  rerunLabel,
  runBanner,
  runFooter,
  runTitle,
} from './ingest.ts'
import type { IngestCounts, IngestStatus } from './types/sources.ts'

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
    control: 'none',
    counts: null,
    ...over,
  }
}

function counts(over: Partial<IngestCounts> = {}): IngestCounts {
  return {
    waiting: 0,
    reading: 0,
    added: 0,
    unchanged: 0,
    failed: 0,
    not_started: 0,
    total: 0,
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
  assert.equal(ingestLabel(status({ total: 12, done: 4, added: 0 })), 'Indexing… 4/12 done')
})

test('an uncounted run says so instead of showing 0 of 0', () => {
  assert.equal(ingestLabel(status({ total: 0, done: 0 })), 'Indexing… preparing')
  assert.equal(runTitle(status({ total: 0, done: 0 })), 'Preparing to index…')
})

test('a finished run that added nothing does not read as a failure', () => {
  assert.equal(
    ingestLabel(status({ state: 'done', added: 0, total: 9, done: 9 })),
    'Indexing done · nothing new',
  )
  assert.equal(
    ingestLabel(status({ state: 'done', added: 3, total: 3, done: 3 })),
    'Indexing done · 3 new',
  )
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
  assert.equal(ingestLabel(p), 'Indexing… preparing')
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

test('a panel that started nothing still locks on a run in flight', () => {
  // Settings reopened mid-run has no request of its own out. Reading only its own flag would show
  // an idle form, and pressing it would ask for a second run the API refuses with a 409.
  assert.equal(indexLocked(false, status({ state: 'running' })), true)
})

test('a request not yet answered locks before any status says so', () => {
  assert.equal(indexLocked(true, null), true)
  assert.equal(indexLocked(true, status({ state: 'done' })), true)
})

test('a resting status leaves the controls free', () => {
  // Includes a failed run: the form is how the user tries again.
  assert.equal(indexLocked(false, null), false)
  assert.equal(indexLocked(false, status({ state: 'idle' })), false)
  assert.equal(indexLocked(false, status({ state: 'done' })), false)
  assert.equal(indexLocked(false, status({ state: 'error' })), false)
})

// ---- the indexing screen ----------------------------------------------------------------------

test('the title counts failed files as done, and names a pause and a stop', () => {
  assert.equal(runTitle(status({ total: 30, done: 14 })), 'Indexing 14 of 30')
  assert.equal(runTitle(status({ total: 30, done: 15, control: 'paused' })), 'Paused at 15 of 30')
  // Pausing is not paused yet: the document in flight is still being read.
  assert.equal(runTitle(status({ total: 30, done: 14, control: 'pausing' })), 'Indexing 14 of 30')
  assert.equal(runTitle(status({ state: 'stopped', total: 30, done: 15 })), 'Stopped at 15 of 30')
  assert.equal(runTitle(status({ state: 'done', total: 30, done: 30 })), 'Finished 30 of 30')
  assert.equal(runTitle(null), 'Indexing')
})

test('the status bar always names the job first, then m/n done', () => {
  // The user, 2026-11-04: the bar at the bottom must say it is an indexing job.
  for (const s of [
    status({ total: 4, done: 2 }),
    status({ total: 4, done: 2, control: 'paused' }),
    status({ state: 'stopped', total: 4, done: 2 }),
    status({ state: 'done', total: 4, done: 4, added: 4 }),
    status({ state: 'error' }),
  ]) {
    assert.match(ingestLabel(s), /^Indexing/)
  }
  assert.equal(ingestLabel(status({ total: 4, done: 2, control: 'paused' })), 'Indexing paused · 2/4 done')
  assert.equal(ingestLabel(status({ state: 'stopped', total: 4, done: 2 })), 'Indexing stopped · 2/4 done')
})

test('a stop is a completion: the library refreshes when a stopped run ends', () => {
  assert.equal(isCompletion(status(), status({ state: 'stopped' })), true)
})

test('each kind of run offers its own filters, with counts', () => {
  const c = counts({ total: 30, reading: 1, waiting: 15, added: 10, unchanged: 2, failed: 2 })
  const running = groupChips(status({ counts: c }))
  assert.deepEqual(
    running.map((g) => `${g.label} ${g.count}`),
    ['All 30', 'In progress 1', 'Waiting 15', 'Finished 12', 'Failed 2'],
  )
  const stopped = groupChips(
    status({ state: 'stopped', counts: counts({ total: 30, not_started: 15, added: 13, failed: 2 }) }),
  )
  assert.deepEqual(
    stopped.map((g) => g.group),
    ['all', 'not_started', 'finished', 'failed'],
  )
  const done = groupChips(status({ state: 'done', counts: counts({ total: 3, added: 3 }) }))
  assert.deepEqual(done.map((g) => g.group), ['all', 'finished', 'failed'])
  assert.deepEqual(groupChips(status()), []) // no run this session: no list to filter
})

test('the banner says what a pause or a stop is waiting for, and what a stop left', () => {
  assert.match(runBanner(status({ control: 'pausing' })) ?? '', /^Pausing once/)
  assert.match(runBanner(status({ control: 'paused' })) ?? '', /Searching and chatting work/)
  assert.match(runBanner(status({ control: 'stopping' })) ?? '', /Nothing finished is lost/)
  assert.equal(runBanner(status()), null)
  assert.equal(
    runBanner(status({ state: 'stopped', counts: counts({ added: 12, unchanged: 1, not_started: 15 }) })),
    '13 documents are in your library. 15 were not started.',
  )
  assert.equal(
    runBanner(status({ state: 'stopped', counts: counts({ added: 1, not_started: 1 }) })),
    '1 document is in your library. 1 was not started.',
  )
})

test('closing the screen never loses anything, and the footer says so', () => {
  assert.match(runFooter(status()), /keeps running/)
  assert.match(runFooter(status({ control: 'paused' })), /stays paused/)
  assert.match(runFooter(status({ state: 'done' })), /can be searched now/)
})

test('a follow-up is offered only when there is something for it to take', () => {
  const stopped = status({ state: 'stopped', counts: counts({ not_started: 15, failed: 2 }) })
  assert.equal(rerunLabel('remaining', stopped), 'Index the remaining 15')
  assert.equal(rerunLabel('failed', stopped), 'Retry the 2 that failed')
  const done = status({ state: 'done', counts: counts({ failed: 1 }) })
  assert.equal(rerunLabel('remaining', done), null) // a run that reached the end left nothing
  assert.equal(rerunLabel('failed', done), 'Retry the one that failed')
  assert.equal(rerunLabel('failed', status({ state: 'done', counts: counts() })), null)
  // Nothing to re-run while a run is in flight: the API would refuse it (409).
  assert.equal(rerunLabel('failed', status({ counts: counts({ failed: 3 }) })), null)
})

test('after the loop and before the run reports, the screen says what it is doing', () => {
  assert.match(runBanner(status({ control: 'finishing' })) ?? '', /searchable/)
  // A stopped loop has marked what it left; the filter for it appears before the run reports.
  const chips = groupChips(
    status({ control: 'stopping', counts: counts({ total: 26, added: 15, not_started: 11 }) }),
  )
  assert.deepEqual(
    chips.map((g) => g.group),
    ['all', 'in_progress', 'waiting', 'not_started', 'finished', 'failed'],
  )
})
