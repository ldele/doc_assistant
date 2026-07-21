// Tests for the global-search match layer (docs/specs/feature-app-shell-search-collapse.md, A8).
// Runner: node's built-in `node:test` with native TypeScript stripping — no new dependency.
// `npm test` from apps/desktop.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import { searchEverything } from './search.ts'
import type { ConversationSummary, LibraryDocument } from './types.ts'

const chat = (
  session_id: string,
  title: string,
  last_at: string,
  extra: Partial<ConversationSummary> = {},
): ConversationSummary =>
  ({
    session_id,
    title,
    turn_count: 1,
    started_at: last_at,
    last_at,
    pinned: false,
    archived: false,
    ...extra,
  }) as ConversationSummary

const doc = (
  id: string,
  fields: Partial<LibraryDocument> = {},
): LibraryDocument =>
  ({
    id,
    filename: `${id}.pdf`,
    title: null,
    authors: null,
    year: null,
    keywords: [],
    ...fields,
  }) as unknown as LibraryDocument

const CHATS = [
  chat('s1', 'Transformers and attention', '2026-07-20T10:00:00Z'),
  chat('s2', 'Retrieval augmented generation', '2026-07-21T10:00:00Z'),
  chat('s3', 'Old archived chat', '2026-07-01T10:00:00Z', { archived: true }),
]
const DOCS = [
  doc('a', { title: 'Attention Is All You Need', authors: 'Vaswani', keywords: ['transformer'] }),
  doc('b', { title: 'Dense Passage Retrieval', authors: 'Karpukhin', keywords: ['retrieval', 'dpr'] }),
  doc('c', { filename: 'sbert.pdf', keywords: ['sentence embeddings'] }),
]

// --- empty query (A4) ------------------------------------------------------------------------- //

test('empty query returns recent non-archived chats and NO documents', () => {
  const r = searchEverything('', CHATS, DOCS)
  assert.equal(r.mode, 'recent')
  assert.deepEqual(
    r.chats.map((c) => c.session_id),
    ['s2', 's1'], // recency desc, archived s3 excluded
  )
  assert.deepEqual(r.docs, [])
  assert.equal(r.docsTotal, 0)
})

test('empty query never dumps the whole document list', () => {
  const many = Array.from({ length: 76 }, (_, i) => doc(`d${i}`, { title: `Paper ${i}` }))
  const r = searchEverything('   ', CHATS, many)
  assert.equal(r.docs.length, 0)
})

test('recentCount caps the empty-query chat list', () => {
  const lots = Array.from({ length: 20 }, (_, i) =>
    chat(`c${i}`, `Chat ${i}`, `2026-07-${String(i + 1).padStart(2, '0')}T10:00:00Z`),
  )
  const r = searchEverything('', lots, [], { recentCount: 6 })
  assert.equal(r.chats.length, 6)
})

// --- query match semantics (A1) --------------------------------------------------------------- //

test('a chat matches on title, case-insensitively', () => {
  const r = searchEverything('RETRIEVAL', CHATS, DOCS)
  assert.deepEqual(
    r.chats.map((c) => c.session_id),
    ['s2'],
  )
})

test('an archived chat can still be found by an explicit query', () => {
  const r = searchEverything('archived', CHATS, DOCS)
  assert.deepEqual(
    r.chats.map((c) => c.session_id),
    ['s3'],
  )
})

test('a document matches on title, filename, authors, OR any keyword', () => {
  assert.deepEqual(searchEverything('attention', CHATS, DOCS).docs.map((d) => d.id), ['a']) // title
  assert.deepEqual(searchEverything('sbert', CHATS, DOCS).docs.map((d) => d.id), ['c']) // filename
  assert.deepEqual(searchEverything('karpukhin', CHATS, DOCS).docs.map((d) => d.id), ['b']) // authors
  assert.deepEqual(searchEverything('dpr', CHATS, DOCS).docs.map((d) => d.id), ['b']) // keyword
})

test('a query hitting both groups returns chats and docs together', () => {
  const r = searchEverything('transformer', CHATS, DOCS)
  assert.equal(r.mode, 'query')
  assert.deepEqual(r.chats.map((c) => c.session_id), ['s1']) // "Transformers and attention"
  assert.deepEqual(r.docs.map((d) => d.id), ['a']) // keyword "transformer"
})

test('docs sort by label ascending; chats by recency descending', () => {
  const chats = [
    chat('older', 'ml topic', '2026-07-10T00:00:00Z'),
    chat('newer', 'ml topic two', '2026-07-20T00:00:00Z'),
  ]
  const docs = [doc('z', { title: 'Zeta ml' }), doc('a', { title: 'Alpha ml' })]
  const r = searchEverything('ml', chats, docs)
  assert.deepEqual(r.chats.map((c) => c.session_id), ['newer', 'older'])
  assert.deepEqual(r.docs.map((d) => d.id), ['a', 'z'])
})

// --- caps + totals (A3) ----------------------------------------------------------------------- //

test('per-group cap truncates but the total reports the real count', () => {
  const docs = Array.from({ length: 12 }, (_, i) => doc(`p${i}`, { title: `Neuro paper ${i}` }))
  const r = searchEverything('neuro', CHATS, docs, { perGroup: 8 })
  assert.equal(r.docs.length, 8)
  assert.equal(r.docsTotal, 12) // "+4 more" is derivable and honest
})

// --- honest-empty + robustness (DoD 4) -------------------------------------------------------- //

test('a query matching nothing returns empty groups, not a throw', () => {
  const r = searchEverything('zzzznomatch', CHATS, DOCS)
  assert.equal(r.chats.length, 0)
  assert.equal(r.docs.length, 0)
  assert.equal(r.chatsTotal, 0)
  assert.equal(r.docsTotal, 0)
})

test('zero chats and zero docs never throw (0-document robustness)', () => {
  assert.doesNotThrow(() => searchEverything('anything', [], []))
  assert.doesNotThrow(() => searchEverything('', [], []))
})
