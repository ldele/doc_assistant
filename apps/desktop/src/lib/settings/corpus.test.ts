// Tests for the Corpus panel's presentation helpers (ADR-037).
//
// The load-bearing one is `describeIndex`: the panel exists to answer "will this still work when
// my library is much bigger?", and the two keyword-index arms have *opposite* answers. Printing
// the reassuring sentence while the legacy in-memory arm is live would be worse than printing
// nothing.

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { builtAgo, describeIndex, formatBytes, perDocument } from './corpus.ts'
import type { CorpusFacts } from '../core/types/settings.ts'

function facts(over: Partial<CorpusFacts> = {}): CorpusFacts {
  return {
    documents: 97,
    chunks: 33105,
    disk: {
      vector_store_bytes: 416_000_000,
      baseline_store_bytes: 120_000_000,
      keyword_index_bytes: 41_000_000,
      document_store_bytes: 7_400_000,
      extraction_cache_bytes: 0,
      total_bytes: 584_400_000,
    },
    keyword_index: { mode: 'on_disk', bytes: 41_000_000, built_at: null },
    ...over,
  }
}

test('formatBytes scales and keeps one decimal only where it reads better', () => {
  assert.equal(formatBytes(0), '0 B')
  assert.equal(formatBytes(999), '999 B')
  assert.equal(formatBytes(1024), '1.0 KB')
  assert.equal(formatBytes(41_000_000), '39 MB')
  assert.equal(formatBytes(2_600_000_000), '2.4 GB')
})

test('formatBytes renders an absent size as a dash, not as zero', () => {
  // "0 B" would read as "this file is empty"; the truth is "there is no file".
  assert.equal(formatBytes(null), '—')
  assert.equal(formatBytes(undefined), '—')
})

test('perDocument answers the scale question, and says nothing at zero documents', () => {
  assert.equal(perDocument(584_400_000, 97), '5.7 MB per document')
  assert.equal(perDocument(0, 0), '')
})

test('builtAgo is relative while that is informative, absolute after a day', () => {
  const now = new Date('2026-07-30T12:00:00Z')
  assert.equal(builtAgo('2026-07-30T11:58:00Z', now), '2 min ago')
  assert.equal(builtAgo('2026-07-30T11:59:40Z', now), 'just now')
  assert.equal(builtAgo('2026-07-30T09:00:00Z', now), '3 h ago')
  assert.notEqual(builtAgo('2026-07-20T09:00:00Z', now), '') // a date, locale-formatted
})

test('builtAgo never prints a negative age from clock skew', () => {
  const now = new Date('2026-07-30T12:00:00Z')
  assert.equal(builtAgo('2026-07-30T12:05:00Z', now), 'just now')
})

test('builtAgo tolerates a missing or unparseable timestamp', () => {
  assert.equal(builtAgo(null), '')
  assert.equal(builtAgo('not a date'), '')
})

test('describeIndex tells the truth about memory for the on-disk arm', () => {
  const d = describeIndex(facts({ keyword_index: { mode: 'on_disk', bytes: 41_000_000, built_at: null } }))
  assert.match(d.label, /on disk, 39 MB/)
  assert.match(d.memory, /does not grow/)
  assert.equal(d.rebuildable, true)
})

test('describeIndex does NOT claim flat memory on the legacy in-memory arm', () => {
  // The whole reason `mode` crosses the wire instead of a pre-rendered sentence.
  const d = describeIndex(facts({ keyword_index: { mode: 'in_memory', bytes: null, built_at: null } }))
  assert.match(d.label, /legacy/)
  assert.match(d.memory, /grows with your library/)
  assert.equal(d.rebuildable, false, 'there is no on-disk index to rebuild')
})

test('describeIndex handles an empty corpus without offering a rebuild', () => {
  const d = describeIndex(facts({ keyword_index: { mode: 'disabled', bytes: null, built_at: null } }))
  assert.match(d.memory, /Add documents/)
  assert.equal(d.rebuildable, false)
})
