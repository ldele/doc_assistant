// Tests for the update-check presentation helpers (ADR-044).
//
// The load-bearing one is that `unknown` never renders as reassurance. A user whose machine is
// offline and a user who is genuinely current take different next actions, so the two sentences
// must not be confusable — and the failing case must never be the reassuring one.

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { asOf, describeUpdate, shouldNotify } from './updates.ts'
import type { UpdateStatus } from '../core/types/updates.ts'

function status(over: Partial<UpdateStatus> = {}): UpdateStatus {
  return {
    state: 'current',
    current_version: '0.5.0',
    latest_version: '0.5.0',
    release_url: 'https://github.com/ldele/doc_assistant/releases/latest',
    checked_at: new Date().toISOString(),
    reason: null,
    auto_check_enabled: false,
    ...over,
  }
}

test('an available update names the version and offers the link', () => {
  const s = describeUpdate(status({ state: 'update_available', latest_version: '0.6.0' }))
  assert.match(s.headline, /0\.6\.0/)
  assert.equal(s.showLink, true)
  assert.equal(s.tone, 'info')
})

test('the available-update line says the user installs it themselves', () => {
  // ADR-044's boundary, stated where the user actually reads it rather than only in the ADR.
  const s = describeUpdate(status({ state: 'update_available', latest_version: '0.6.0' }))
  assert.match(s.detail, /never installs/i)
})

test('being current says so, with how old the answer is', () => {
  const s = describeUpdate(status({ state: 'current' }))
  assert.match(s.headline, /up to date/i)
  assert.match(s.detail, /Checked/)
  assert.equal(s.showLink, false)
})

test('a failed check never reads as up to date', () => {
  const s = describeUpdate(
    status({
      state: 'unknown',
      latest_version: null,
      reason: 'could not reach the update server',
    }),
  )
  assert.doesNotMatch(s.headline, /up to date/i)
  assert.match(s.headline, /Couldn't check/i)
  assert.equal(s.detail, 'could not reach the update server')
  assert.equal(s.showLink, false)
})

test('an unknown state with no reason still says something true', () => {
  const s = describeUpdate(status({ state: 'unknown', latest_version: null, reason: null }))
  assert.match(s.detail, /No check has run yet/)
})

test('update_available without a version cannot offer a link', () => {
  // A malformed payload must not produce a "get the update" button pointing at nothing.
  const s = describeUpdate(status({ state: 'update_available', latest_version: null }))
  assert.equal(s.showLink, false)
})

test('only a confirmed newer release notifies outside Settings', () => {
  assert.equal(shouldNotify(status({ state: 'update_available', latest_version: '0.6.0' })), true)
  assert.equal(shouldNotify(status({ state: 'current' })), false)
  // Offline must not nag forever about a check it cannot run.
  assert.equal(shouldNotify(status({ state: 'unknown', latest_version: null })), false)
})

test('asOf reads in the units a person would use', () => {
  const now = new Date('2026-08-12T12:00:00Z')
  assert.equal(asOf('2026-08-12T11:59:40Z', now), 'Checked just now.')
  assert.equal(asOf('2026-08-12T11:59:00Z', now), 'Checked 1 minute ago.')
  assert.equal(asOf('2026-08-12T11:30:00Z', now), 'Checked 30 minutes ago.')
  assert.equal(asOf('2026-08-12T09:00:00Z', now), 'Checked 3 hours ago.')
  assert.equal(asOf('2026-08-10T12:00:00Z', now), 'Checked 2 days ago.')
})

test('asOf is empty rather than wrong when there is no usable stamp', () => {
  assert.equal(asOf(null), '')
  assert.equal(asOf('not-a-date'), '')
})
