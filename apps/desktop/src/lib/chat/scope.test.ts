import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import { GRAPH_STALE_NOTE, scopeIcon, scopeSentences, scopeText } from './scope.ts'

const folder = (name: string | null, n: number, id = name ?? 'gone') => ({
  folder_id: id,
  folder_name: name,
  doc_count: n,
})

test('one folder is named with its count', () => {
  const s = { folders: [folder('Thesis', 5)], concept: null, doc_count: 5 }
  assert.equal(scopeText(s), 'Searched Thesis only: 5 documents.')
  assert.equal(scopeIcon(s), 'folder')
})

test('every folder is named, and the count is the union', () => {
  const s = { folders: [folder('A', 9), folder('B', 7)], concept: null, doc_count: 14 }
  assert.equal(scopeText(s), 'Searched A, B only: 14 documents.')
  const names = scopeSentences(s)[0].filter((p) => p.strong).map((p) => p.text)
  assert.deepEqual(names, ['A', 'B'])
})

test('a folder that held nothing is named as left out', () => {
  const s = { folders: [folder('A', 1), folder('C', 0)], concept: null, doc_count: 1 }
  assert.equal(scopeText(s), 'Searched A only: 1 document. C held no documents and was left out.')
})

test('a deleted folder is not given a name', () => {
  const s = { folders: [folder('A', 2), folder(null, 0)], concept: null, doc_count: 2 }
  assert.equal(
    scopeText(s),
    'Searched A only: 2 documents. A folder that no longer exists was left out.',
  )
})

test('when every folder is bad, nothing was searched', () => {
  const s = { folders: [folder(null, 0)], concept: null, doc_count: 0 }
  assert.equal(scopeText(s), 'Nothing was searched. A folder that no longer exists was left out.')
})

test('a concept scope names the concept, and says when the graph is out of date', () => {
  const fresh = { folders: [], concept: { name: 'synapse', graph_stale: false }, doc_count: 6 }
  assert.equal(scopeText(fresh), 'Searched the documents where synapse is present: 6 documents.')
  assert.equal(scopeIcon(fresh), 'tag')
  const stale = { ...fresh, concept: { name: 'synapse', graph_stale: true } }
  assert.equal(
    scopeText(stale),
    `Searched the documents where synapse is present: 6 documents. ${GRAPH_STALE_NOTE}`,
  )
})
