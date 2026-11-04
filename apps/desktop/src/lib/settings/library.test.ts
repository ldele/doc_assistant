import assert from 'node:assert/strict'
import { test } from 'node:test'

import { goBackLabel, inspectionSummary, libraryName, sourcesLine } from './library.ts'
import type { LibraryInspection } from '../core/types/libraries.ts'

function found(over: Partial<LibraryInspection> = {}): LibraryInspection {
  return {
    path: 'C:\\Projects\\pv-idx\\data',
    is_library: true,
    documents: 36,
    chunks: 5700,
    embedding_model: 'bge-base',
    sources_found: 36,
    cache_named_folder: null,
    is_current: false,
    problems: [],
    notes: [],
    can_open: true,
    ...over,
  }
}

test('the summary says how much a library holds and whether it can be searched', () => {
  // Numbers in the reader's own format: this machine may print 5'700 or 5 700.
  const n = (5700).toLocaleString()
  assert.equal(inspectionSummary(found()), `36 documents · ${n} chunks, searchable with bge-base`)
  assert.equal(
    inspectionSummary(found({ documents: 1, chunks: 0 })),
    "1 document, no chunks for this app's model (bge-base)",
  )
  assert.equal(inspectionSummary(found({ is_library: false })), null) // its problem says why
})

test('the sources line counts what was found, and is quiet for an empty library', () => {
  assert.equal(sourcesLine(found({ sources_found: 35 })), '35 of 36 source files found where the library says.')
  assert.equal(sourcesLine(found({ documents: 0, sources_found: 0 })), null)
})

test('the controls name a library by its folder, or as the app’s own', () => {
  assert.equal(libraryName(null), "the app's own library")
  assert.equal(libraryName('D:\\libs\\a'), 'D:\\libs\\a')
  assert.equal(goBackLabel(null), "Go back to the app's own library")
  assert.equal(goBackLabel('D:\\libs\\a'), 'Go back to the previous library')
})
