import assert from 'node:assert/strict'
import { test } from 'node:test'

import { deleteFileDetail, removeOnlyDetail, shortenPath, targetName } from './deletetarget.ts'

test('the heading names the document, and is never blank', () => {
  assert.equal(targetName({ title: 'Cajal 1899', filename: 'cajal.pdf' }), 'Cajal 1899')
  assert.equal(targetName({ title: null, filename: 'cajal.pdf' }), 'cajal.pdf')
  // A whitespace-only title is the same failure as an empty one: a confirm with no subject.
  assert.equal(targetName({ title: '   ', filename: 'cajal.pdf' }), 'cajal.pdf')
})

test('a short path is shown whole', () => {
  assert.equal(shortenPath('C:\\lib\\a.pdf'), 'C:\\lib\\a.pdf')
})

test('a long path keeps both ends, because both ends identify the file', () => {
  const long = 'C:\\Users\\Lucas\\Zotero\\storage\\ABC123XYZ789\\cajal-textura-1899.pdf'
  const out = shortenPath(long, 40)
  assert.equal(out.length, 40)
  assert.ok(out.startsWith('C:\\Users'), 'the drive and user must survive')
  assert.ok(out.endsWith('1899.pdf'), 'the filename must survive — it says WHICH file')
  assert.ok(out.includes('…'))
})

test('shortenPath degenerates safely rather than throwing', () => {
  assert.equal(shortenPath('anything', 1), '…')
  assert.equal(shortenPath('anything', 0), '…')
})

test('the destructive option names the destination', () => {
  const detail = deleteFileDetail({ source_path: 'C:\\lib\\a.pdf' })
  assert.ok(detail && detail.includes('C:\\lib\\a.pdf'))
  assert.ok(detail.includes('Recycle Bin'))
})

test('with no path recorded there is no sentence — so the caller must not offer the option', () => {
  // ADR-046 makes naming the destination part of the decision. Offering "also delete the file"
  // without being able to say which file is the version the ADR rejected.
  assert.equal(deleteFileDetail({ source_path: null }), null)
  assert.equal(deleteFileDetail({ source_path: '   ' }), null)
})

test('the safe option says the file stays, and counts chunks only when there are some', () => {
  assert.equal(
    removeOnlyDetail({ chunk_count: 12 }),
    'The file stays where it is. Its 12 chunks leave the search index.',
  )
  assert.equal(
    removeOnlyDetail({ chunk_count: 1 }),
    'The file stays where it is. Its 1 chunk leaves the search index.',
    'the verb has to agree, or the dialog looks machine-generated while asking for trust',
  )
  // "removing its 0 chunks" reads as a malfunction on a document that never indexed.
  assert.equal(removeOnlyDetail({ chunk_count: 0 }), 'The file stays where it is.')
  assert.equal(removeOnlyDetail({ chunk_count: null }), 'The file stays where it is.')
})
