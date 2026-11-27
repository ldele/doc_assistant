import { test } from 'node:test'
import assert from 'node:assert/strict'

import type { ExtractionRecord } from '../core/types/extractionRecord.ts'
import { factLines, fixFor, recordView, whenLine } from './extractionRecord.ts'

function rec(over: Partial<ExtractionRecord> = {}): ExtractionRecord {
  return {
    created_at: '2026-11-27T16:52:00',
    run_kind: 'index',
    states: { references: 'found', figures: 'found', tables: 'partial' },
    stale: false,
    text: { pages: 28, chars: 90000, empty_pages: [] },
    metadata: { title: true, authors: false, year: true, doi: true },
    references: { heading: true, entries: 65, without_year: 1, matched: 4, notes: [], seconds: 0.1, reason: null, carried_from: null },
    figures: { found: 8, cropped: 8, with_caption: 5, caption_only: 2, seconds: 0.5, reason: null, carried_from: null },
    tables: { extracted: 1, pages_naming_a_table: [6, 8], pages_without_one: [8], seconds: 1.3, reason: null },
    warnings: [],
    summary: '',
    ...over,
  }
}

test('each warning offers the fix that addresses it, and empty pages offer none', () => {
  assert.deepEqual(fixFor({ id: 'references_none', text: '' }), { kind: 'rerun', parts: ['references'] })
  assert.deepEqual(fixFor({ id: 'figures_uncaptioned', text: '' }), { kind: 'rerun', parts: ['figures'] })
  assert.deepEqual(fixFor({ id: 'tables_missed', text: '' }), { kind: 'rerun', parts: ['text'] })
  assert.deepEqual(fixFor({ id: 'metadata_missing', text: '' }), { kind: 'metadata' })
  assert.equal(fixFor({ id: 'text_empty_pages', text: '' }), null)
})

test('never recorded, stale and current are three different views', () => {
  assert.equal(recordView(undefined).kind, 'loading')
  assert.equal(recordView(null).kind, 'never')
  assert.equal(recordView({ document_id: 'd', record: null }).kind, 'never')
  assert.equal(recordView({ document_id: 'd', record: rec({ stale: true }) }).kind, 'stale')
  assert.equal(recordView({ document_id: 'd', record: rec() }).kind, 'current')
})

test('the dated line says whether indexing or a re-run wrote it, in UTC read as local', () => {
  assert.match(whenLine(rec()), /^Recorded when this document was indexed, 27 Nov 2026 at \d\d:\d\d\.$/)
  assert.match(whenLine(rec({ run_kind: 'rerun' })), /was last re-run/)
})

test('the fact lines follow the drawing and a pass that did not run says why', () => {
  assert.deepEqual(factLines(rec()), [
    { label: 'References', text: '65 references found · 1 without a year · 4 of them in your library' },
    { label: 'Figures', text: '8 figures found · 5 with a caption · 2 found by their caption only' },
    { label: 'Tables', text: '1 extracted · 2 pages name a table' },
    { label: 'Metadata', text: 'title, year, DOI found · no authors' },
  ])
  const epub = factLines(
    rec({
      states: { references: 'none_found', figures: 'skipped', tables: 'skipped' },
      references: { ...rec().references, heading: false, entries: 0 },
      figures: { ...rec().figures, reason: 'figure detection is PDF-only' },
      tables: { ...rec().tables, reason: 'tables are read from PDFs only' },
    }),
  )
  assert.deepEqual(epub.slice(0, 3).map((l) => l.text), [
    'none found',
    'figure detection is PDF-only',
    'tables are read from PDFs only',
  ])
})
