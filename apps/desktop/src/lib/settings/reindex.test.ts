import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  confirmDisabled,
  confirmLabel,
  estimateLine,
  filesToRead,
  formatDuration,
  measuredLine,
  planLines,
} from './reindex.ts'
import type { IngestPlan } from '../core/types/sources.ts'

function plan(over: Partial<IngestPlan> = {}): IngestPlan {
  return {
    new: 12,
    new_bytes: 0,
    changed: 3,
    changed_bytes: 0,
    unchanged: 240,
    excluded: 0,
    seconds_all: 15 * 25,
    seconds_new: 12 * 25,
    seconds_per_document: 25,
    measured_documents: 15,
    ...over,
  }
}

test('only new files leaves the changed ones out of the count and the time', () => {
  assert.equal(filesToRead(plan(), false), 15)
  assert.equal(filesToRead(plan(), true), 12)
  assert.equal(estimateLine(plan(), false), 'It should take about 6 minutes.')
  assert.equal(estimateLine(plan(), true), 'It should take about 5 minutes.')
})

test('durations are rough on purpose', () => {
  assert.equal(formatDuration(null), null)
  assert.equal(formatDuration(30), 'under a minute')
  assert.equal(formatDuration(70), 'about a minute')
  assert.equal(formatDuration(13 * 60), 'about 13 minutes')
  assert.equal(formatDuration(23 * 60), 'about 25 minutes') // fives past twenty
  assert.equal(formatDuration(60 * 60), 'about 1 hour')
  assert.equal(formatDuration(127 * 60), 'about 2 hours 5 min')
})

test('no measurement means no number, and it says so', () => {
  const unmeasured = plan({ seconds_all: null, seconds_new: null, seconds_per_document: null, measured_documents: 0 })
  assert.match(estimateLine(unmeasured, false), /^No estimate yet/)
  assert.equal(measuredLine(unmeasured), null)
  assert.match(measuredLine(plan()) ?? '', /25 s per document on the last run \(15 documents\)/)
})

test('each kind of file says what the run does with it', () => {
  assert.deepEqual(planLines(plan({ excluded: 2 }), false), [
    '12 new files: indexed',
    '3 files changed since indexing: read again',
    '240 files already indexed: skipped',
    '2 excluded files: skipped',
  ])
  assert.equal(planLines(plan(), true)[1], '3 files changed since indexing: keep the version already indexed')
  assert.deepEqual(planLines(plan({ new: 1, changed: 0, unchanged: 0 }), false), ['1 new file: indexed'])
})

test('the button counts what it will do, and only new with none new cannot start', () => {
  assert.equal(confirmLabel(plan(), false), 'Index 15 files')
  assert.equal(confirmLabel(plan({ new: 1, changed: 0 }), true), 'Index 1 file')
  const nothing = plan({ new: 0, changed: 0 })
  assert.equal(confirmLabel(nothing, false), 'Re-index anyway') // inform, never block
  assert.equal(confirmDisabled(nothing, false), false)
  assert.equal(confirmDisabled(nothing, true), true)
  assert.equal(estimateLine(nothing, true), 'No new files to index.')
})
