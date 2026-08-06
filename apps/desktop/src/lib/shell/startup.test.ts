// Tests for the readiness-gate timing (KI-39).
// Runner: node's built-in `node:test` with native TypeScript stripping — no new dependency.
// `npm test` from apps/desktop.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  FIRST_LAUNCH_HINT_MS,
  STALLED_MS,
  backoffDelayMs,
  startupPhase,
} from './startup.ts'

test('backoff starts fast, eases off, and stays bounded', () => {
  assert.equal(backoffDelayMs(0), 1_000) // a warm start answers in a few seconds
  assert.equal(backoffDelayMs(9), 1_000)
  assert.equal(backoffDelayMs(10), 2_000)
  assert.equal(backoffDelayMs(19), 2_000)
  assert.equal(backoffDelayMs(20), 5_000)
  // Bounded forever: a late backend must still be noticed promptly, so this never grows.
  assert.equal(backoffDelayMs(10_000), 5_000)
})

test('backoff is monotonic — a later attempt never polls more eagerly', () => {
  let prev = 0
  for (let a = 0; a < 60; a++) {
    const d = backoffDelayMs(a)
    assert.ok(d >= prev, `attempt ${a} (${d}ms) polled sooner than attempt ${a - 1} (${prev}ms)`)
    prev = d
  }
})

test('the first minute of a cold start is spent polling, not giving up', () => {
  // The regression this file exists for: the old gate stopped at 60 polls x 1s = 60s, and RG-012
  // measured health at ~30s on a FAST box. Walk the real schedule and prove we are still going
  // well past the point where the old one had already declared the backend dead.
  let elapsed = 0
  let attempt = 0
  while (elapsed < 60_000) {
    elapsed += backoffDelayMs(attempt)
    attempt++
  }
  assert.ok(attempt < 60, `reached 60s in ${attempt} polls — should be far fewer than the old 60`)
  // And there is no attempt count at which we stop: the schedule is defined for any attempt.
  assert.equal(typeof backoffDelayMs(attempt + 1_000), 'number')
})

test('what we SAY changes with elapsed time', () => {
  assert.equal(startupPhase(0), 'connecting')
  assert.equal(startupPhase(FIRST_LAUNCH_HINT_MS - 1), 'connecting')
  assert.equal(startupPhase(FIRST_LAUNCH_HINT_MS), 'slow')
  assert.equal(startupPhase(STALLED_MS - 1), 'slow')
  assert.equal(startupPhase(STALLED_MS), 'stalled')
  assert.equal(startupPhase(60 * 60 * 1000), 'stalled')
})

test('the hint fires before the RG-012 cold-start measurement, and stalled well after', () => {
  // RG-012 (2026-08-06) measured /api/health at ~30s on a fast, warm-cache box. The user must
  // already have been told why it is waiting by then...
  assert.ok(FIRST_LAUNCH_HINT_MS < 30_000, 'the hint must appear before a NORMAL cold start ends')
  // ...and a normal cold start must never be labelled a fault.
  assert.ok(STALLED_MS > 30_000, 'a normal cold start must not read as stalled')
})
