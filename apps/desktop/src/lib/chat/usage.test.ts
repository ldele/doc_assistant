import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import { usageLabel } from './usage.ts'

const u = (over: Partial<Parameters<typeof usageLabel>[0] & object> = {}) => ({
  turn_input: 800,
  turn_output: 404,
  is_local: false,
  cost_usd: 0.0031,
  ...over,
})

// The count is grouped with `toLocaleString`, so the separator is the *reader's*, not ours — it
// is an apostrophe on the machine this was written on. Building the expectation the same way keeps
// the test about the label and not about where the developer happens to live.
const N = (1204).toLocaleString()

test('a metered turn shows what it cost', () => {
  assert.equal(usageLabel(u()), `${N} tokens · $0.0031`)
})

test('a local turn that reports counts shows them', () => {
  assert.equal(usageLabel(u({ is_local: true, cost_usd: null })), `${N} tokens · local`)
})

test('a local turn with no counts says they were not reported, not that they were zero', () => {
  // The defect this closes: Ollama returns no usage, so the counters sit at 0 and the line read
  // "0 tokens · local" — a measurement of nothing where nothing was measured.
  assert.equal(
    usageLabel(u({ turn_input: 0, turn_output: 0, is_local: true, cost_usd: null })),
    'local · tokens not reported',
  )
})

test('a metered turn that somehow used nothing still reports zero honestly', () => {
  // Only the *local* case is unknown. A metered zero is a real measurement and must not be
  // relabelled — that would hide a genuine "this turn called nothing".
  assert.equal(usageLabel(u({ turn_input: 0, turn_output: 0, cost_usd: 0 })), '0 tokens · $0.0000')
})

test('a priced turn with no price says so rather than implying free', () => {
  assert.equal(usageLabel(u({ cost_usd: null })), `${N} tokens · n/a`)
})

test('no usage at all renders nothing', () => {
  assert.equal(usageLabel(null), '')
})
