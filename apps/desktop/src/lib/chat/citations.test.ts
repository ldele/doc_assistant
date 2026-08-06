// Tests for the frontend inline-citation parser.
// Runner: node's built-in `node:test` with native TypeScript stripping — no new dependency.
// `npm test` from apps/desktop.
//
// The vectors are NOT written here: they live in `tests/fixtures/citation_vectors.json` at the
// repo root and are asserted identically by `tests/unit/test_synthesis.py` against the backend
// parser. One file, two implementations — so a change made on one side and not the other fails
// that side's suite. See KI-35 for what unpinned divergence cost.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { CITE_ANYWHERE, CITE_EXACT, CITE_SPLIT, citationNumbers } from './citations.ts'

const FIXTURE = fileURLToPath(
  new URL('../../../../../tests/fixtures/citation_vectors.json', import.meta.url),
)
const vectors: { name: string; text: string; expected: number[] }[] = JSON.parse(
  readFileSync(FIXTURE, 'utf-8'),
).vectors

test('the shared contract vectors parse identically to the backend', () => {
  assert.ok(vectors.length >= 18, 'vectors were removed rather than added to')
  for (const v of vectors) {
    assert.deepEqual(citationNumbers(v.text), v.expected, v.name)
  }
})

test('CITE_ANYWHERE agrees with citationNumbers on every vector', () => {
  // linkifyCitations uses CITE_ANYWHERE to decide whether a text node is worth walking; if it
  // disagreed with the extractor, citations would silently fail to become clickable.
  for (const v of vectors) {
    assert.equal(CITE_ANYWHERE.test(v.text), v.expected.length > 0, v.name)
  }
})

test('CITE_SPLIT keeps the token, and each kept part is exactly one citation', () => {
  const parts = 'DPR [1] beats BM25 [2].'.split(CITE_SPLIT).filter(Boolean)
  assert.deepEqual(parts, ['DPR ', '[1]', ' beats BM25 ', '[2]', '.'])
  assert.deepEqual(parts.filter((p) => CITE_EXACT.test(p)), ['[1]', '[2]'])
})

test('a multi-number token yields one button per number', () => {
  // "[Sources 2, 4]" must render as [2][4] — the numbers, not the model's phrasing.
  assert.deepEqual('[Sources 2, 4]'.match(/\d+/g), ['2', '4'])
  assert.ok(CITE_EXACT.test('[Sources 2, 4]'))
})

test('the regexes are stateless across calls (no lastIndex leak)', () => {
  // A `g`-flagged regex reused via .test() alternates true/false. These must not.
  assert.equal(CITE_ANYWHERE.test('cite [1]'), true)
  assert.equal(CITE_ANYWHERE.test('cite [1]'), true)
  assert.equal(CITE_EXACT.test('[1]'), true)
  assert.equal(CITE_EXACT.test('[1]'), true)
  // citationNumbers uses a module-level /g regex internally; it must reset per call.
  assert.deepEqual(citationNumbers('a [1] b [2]'), [1, 2])
  assert.deepEqual(citationNumbers('a [1] b [2]'), [1, 2])
})
