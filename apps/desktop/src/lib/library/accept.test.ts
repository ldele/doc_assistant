// Tests for the AD1 accept-surface helpers.
//
// These are the only part of the accept surface this repo can test today: the Tauri boundary needs
// a native window, and the components need a harness that does not exist. So the logic that can be
// pure is pure, and it is all asserted here.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  basename,
  dedupePaths,
  previewNames,
  remainderLabel,
  sourceKeyName,
  summarise,
} from './accept.ts'

const WIN = 'C:\\Users\\Lucas\\Zotero\\storage\\ABC123\\cajal-1899.pdf'
const NIX = '/home/lucas/papers/cajal-1899.pdf'

// ============================================================
// basename — Tauri reports native separators, so both must work
// ============================================================

test('basename handles Windows and POSIX separators', () => {
  assert.equal(basename(WIN), 'cajal-1899.pdf')
  assert.equal(basename(NIX), 'cajal-1899.pdf')
})

test('basename survives a trailing separator on a dropped folder', () => {
  assert.equal(basename('C:\\Users\\Lucas\\papers\\'), 'papers')
  assert.equal(basename('/home/lucas/papers/'), 'papers')
})

test('basename returns the input when there is no separator', () => {
  assert.equal(basename('cajal-1899.pdf'), 'cajal-1899.pdf')
})

test('basename does not throw on an empty string', () => {
  assert.equal(basename(''), '')
})

// ============================================================
// dedupePaths — one gesture can yield the same path twice
// ============================================================

test('duplicate paths collapse to one', () => {
  assert.deepEqual(dedupePaths([WIN, WIN, NIX]), [WIN, NIX])
})

test('dedupe keeps first-appearance order', () => {
  // The order is the order the user chose them in; AD2 sorts verdicts, not this.
  assert.deepEqual(dedupePaths(['b', 'a', 'b', 'c']), ['b', 'a', 'c'])
})

test('dedupe drops empty entries rather than passing them downstream', () => {
  assert.deepEqual(dedupePaths(['', 'a', '']), ['a'])
})

test('dedupe of nothing is nothing, not a throw', () => {
  assert.deepEqual(dedupePaths([]), [])
})

test('paths differing only in case are kept apart', () => {
  // Windows paths are case-insensitive, but the backend resolves and compares them; guessing here
  // would silently drop a real file on a case-sensitive volume.
  assert.equal(dedupePaths(['C:\\A\\x.pdf', 'C:\\a\\x.pdf']).length, 2)
})

// ============================================================
// summarise
// ============================================================

test('an empty drop says so instead of claiming zero files are ready', () => {
  assert.equal(summarise([]), 'Nothing to add')
})

test('a single file is named, not counted', () => {
  assert.equal(summarise([WIN]), '1 file ready: cajal-1899.pdf')
})

test('several files are counted with a plural', () => {
  assert.equal(summarise([WIN, NIX]), '2 files ready')
})

// ============================================================
// previewNames / remainderLabel — the label is bounded, the batch is not
// ============================================================

test('the preview shows at most `limit` names', () => {
  const paths = ['a/1.pdf', 'a/2.pdf', 'a/3.pdf', 'a/4.pdf', 'a/5.pdf']
  assert.deepEqual(previewNames(paths), ['1.pdf', '2.pdf', '3.pdf'])
})

test('the remainder reports what the preview hid', () => {
  const paths = ['a/1.pdf', 'a/2.pdf', 'a/3.pdf', 'a/4.pdf', 'a/5.pdf']
  assert.equal(remainderLabel(paths), 'and 2 more')
})

test('one hidden file reads as "and 1 more", not "and 1 mores"', () => {
  assert.equal(remainderLabel(['1', '2', '3', '4']), 'and 1 more')
})

test('nothing hidden renders no remainder at all', () => {
  // The caller must never print "and 0 more"; an empty string is what makes that impossible.
  assert.equal(remainderLabel(['1', '2']), '')
  assert.equal(remainderLabel(['1', '2', '3']), '')
})

test('the preview never bounds the batch — only the label', () => {
  const paths = Array.from({ length: 500 }, (_, i) => `a/${i}.pdf`)
  assert.equal(previewNames(paths).length, 3)
  assert.equal(remainderLabel(paths), 'and 497 more')
  assert.equal(summarise(paths), '500 files ready')
})

test('a limit of zero is honoured without producing a negative remainder', () => {
  assert.deepEqual(previewNames(['a', 'b'], 0), [])
  assert.equal(remainderLabel(['a', 'b'], 0), 'and 2 more')
})


// `sourceKeyName` — a duplicate's match, named for a reader rather than for the API.
//
// The review sheet used to show nothing at all here: the server always sets `advisory` on a
// duplicate, so the branch that named the matched file was unreachable. Reaching it exposed the
// other half of the problem — `duplicate_of` is a `source_key`, so naming it raw would have shown
// the user a uuid.

test('a library key shows the file, not the "library:" prefix', () => {
  assert.equal(sourceKeyName('library:cajal-1899.pdf'), 'cajal-1899.pdf')
})

test('a referenced key never shows its uuid', () => {
  const key = 'b0d8a4e8-55be-429c-96fc-a8bd34980891:papers/rag.pdf'
  const shown = sourceKeyName(key)
  assert.equal(shown, 'rag.pdf')
  assert.ok(!shown.includes('b0d8a4e8'), 'a root uuid must never reach the user')
})

test('a bare rel_path — the pre-AD3b shorthand — still names its file', () => {
  assert.equal(sourceKeyName('papers/rag.pdf'), 'rag.pdf')
  assert.equal(sourceKeyName('rag.pdf'), 'rag.pdf')
})

test('a filename containing a colon does not lose its head', () => {
  // Only the FIRST colon splits, and a name with no root prefix still resolves to itself.
  assert.equal(sourceKeyName('library:notes: draft.pdf'), 'notes: draft.pdf')
})

test('a windows-separated rel_path is handled like a posix one', () => {
  assert.equal(sourceKeyName(String.raw`library:sub\dir\paper.pdf`), 'paper.pdf')
})
