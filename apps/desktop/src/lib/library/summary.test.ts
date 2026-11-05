import { test } from 'node:test'
import assert from 'node:assert/strict'

import type { CollectionSummary } from '../core/types/summary.ts'
import {
  contextLine,
  decadeLabel,
  foundRows,
  indexingRows,
  meterShare,
  metadataRows,
  plural,
  tiles,
  typeBars,
  whatLine,
  yearBars,
  type MeterRow,
} from './summary.ts'

function summary(over: Partial<CollectionSummary> = {}): CollectionSummary {
  return {
    scope: 'library',
    folder_id: null,
    folder_name: null,
    counted_at: '2026-11-05T13:02:00+00:00',
    documents: 0,
    passages: 0,
    pages: 0,
    paged_documents: 0,
    need_attention: 0,
    indexing: {
      no_text: 0,
      flagged_marginal: 0,
      flagged_broken: 0,
      not_classified: 0,
      files_not_in_library: 0,
      files_excluded: 0,
      files_seen_at: null,
      last_run_failed: null,
    },
    metadata: {
      missing_title: 0,
      missing_authors: 0,
      missing_year: 0,
      missing_doi: 0,
      implausible_year: 0,
      set_by_you: 0,
      from_a_catalogue: 0,
      with_any_field: 0,
    },
    found: {
      references: { on_record: 0, none_on_record: 0, rows: 0 },
      figures: { on_record: 0, none_on_record: 0, rows: 0, described: 0, pdfs: 0 },
      letter_spacing: { recorded: 0, pdfs: 0 },
      keywords: { on_record: 0, none_on_record: 0, rows: 0 },
    },
    types: [],
    years: { by_decade: [], no_year: 0, oldest: null, newest: null },
    folders: { in_a_folder: 0, in_none: 0, also_elsewhere: null },
    added: { first: null, last: null },
    ...over,
  }
}

const seen = '2026-11-05T12:00:00Z'

test('plural, decade and context labels', () => {
  assert.equal(plural(1, 'file', 'files'), '1 file')
  assert.equal(plural(41180, 'file', 'files'), '41,180 files')
  assert.equal(decadeLabel(1980), '1980s')
  assert.equal(
    contextLine(summary({ documents: 60, passages: 12870, pages: 1288, paged_documents: 57 })),
    '12,870 passages · 1,288 pages in 57 paged documents',
  )
  assert.equal(contextLine(summary({ documents: 2, passages: 1, pages: 9, paged_documents: 2 })), '1 passage · 9 pages')
})

test('need attention leads, and is the server number', () => {
  const t = tiles(summary({ documents: 0, need_attention: 6 }))
  assert.deepEqual(
    t.map((x) => x.key),
    ['need', 'documents'],
  )
  assert.equal(t[0].value, '6')
  assert.equal(t[0].alert, true)
  assert.ok(t.every((tile) => !tile.value.includes('%')))
  assert.equal(tiles(summary())[0].alert, false)
})

test('at 0 documents only the files row shows, and it names its fix', () => {
  const empty = indexingRows(summary())
  assert.deepEqual(
    empty.map((r) => r.key),
    ['files'],
  )
  assert.equal(empty[0].title, 'Your documents folder has not been looked at yet')
  assert.equal(empty[0].target, null)

  const waiting = indexingRows(
    summary({ need_attention: 6, indexing: { ...summary().indexing, files_not_in_library: 6, files_seen_at: seen } }),
  )
  assert.equal(waiting[0].title, '6 files in your documents folder are not in the library')
  assert.equal(waiting[0].fix, 'Index them in Settings > Documents')
  assert.deepEqual(waiting[0].target, { kind: 'files' })
})

test('files that failed in this run open the indexing screen', () => {
  const rows = indexingRows(
    summary({
      indexing: { ...summary().indexing, files_not_in_library: 6, files_seen_at: seen, last_run_failed: 2 },
    }),
  )
  assert.equal(rows[0].detail, "2 failed in this session's run; the others were never indexed.")
  assert.equal(rows[0].fix, 'See why on the indexing screen')
  assert.deepEqual(rows[0].target, { kind: 'indexing' })
  const all = indexingRows(
    summary({
      indexing: { ...summary().indexing, files_not_in_library: 1, files_seen_at: seen, last_run_failed: 1 },
    }),
  )
  assert.equal(all[0].title, '1 file in your documents folder is not in the library')
  assert.equal(all[0].detail, "It failed in this session's run.")
})

test('a zero row is plain text: no fix, no target', () => {
  const rows = indexingRows(summary({ documents: 3 }))
  const noText = rows.find((r) => r.key === 'no_text')
  assert.ok(noText)
  assert.equal(noText.title, '0 documents gave no text')
  assert.equal(noText.fix, null)
  assert.equal(noText.target, null)
  assert.equal(noText.tone, 'none')
})

test('document issues open the grid and name the re-run', () => {
  const rows = indexingRows(
    summary({ documents: 4, indexing: { ...summary().indexing, no_text: 1, flagged_marginal: 2, files_seen_at: seen } }),
  )
  const noText = rows.find((r) => r.key === 'no_text')
  const flagged = rows.find((r) => r.key === 'flagged')
  assert.deepEqual(noText?.target, { kind: 'issue', issue: 'no_text' })
  assert.equal(noText?.fix, 'Fix: Re-run… > Text & chunks')
  assert.equal(flagged?.title, '2 extractions were flagged')
  assert.deepEqual(flagged?.target, { kind: 'issue', issue: 'flagged' })
})

test('a folder never counts files, and points at the library', () => {
  const rows = indexingRows(
    summary({
      scope: 'folder',
      folder_id: 'f',
      folder_name: 'Thesis',
      documents: 2,
      indexing: { ...summary().indexing, files_not_in_library: null, files_excluded: null },
    }),
  )
  assert.deepEqual(
    rows.map((r) => r.key),
    ['no_text', 'flagged', 'folder-files'],
  )
  assert.deepEqual(rows[2].target, { kind: 'library' })
})

test('metadata never read is said once, not as four rows of defects', () => {
  const s = summary({
    documents: 140,
    metadata: { ...summary().metadata, missing_title: 140, missing_authors: 140, missing_year: 140, missing_doi: 140 },
  })
  const rows = metadataRows(s)
  assert.equal(rows.length, 1)
  assert.equal(rows[0].kind, 'line')
  assert.equal(rows[0].kind === 'line' && rows[0].title, 'Not read for 140 documents')
})

test('metadata meters: N of M, and the missing ones to open', () => {
  const s = summary({
    documents: 5,
    metadata: { ...summary().metadata, missing_year: 2, missing_doi: 5, with_any_field: 5 },
  })
  const rows = metadataRows(s) as MeterRow[]
  assert.deepEqual(
    rows.map((r) => [r.label, r.have, r.of, r.tail, r.issue]),
    [
      ['Title', 5, 5, null, null],
      ['Authors', 5, 5, null, null],
      ['Year', 3, 5, '2 missing', 'missing_year'],
      ['DOI', 0, 5, '5 missing', 'missing_doi'],
    ],
  )
})

test('references and figures not looked for are said once; otherwise meters', () => {
  const never = foundRows(
    summary({
      documents: 4,
      found: { ...summary().found, figures: { on_record: 0, none_on_record: 3, rows: 0, described: 0, pdfs: 3 } },
    }),
  )
  assert.deepEqual(
    never.map((r) => r.key),
    ['references-not-read', 'figures-not-read', 'keywords-not-read'],
  )
  const some = foundRows(
    summary({
      documents: 5,
      found: {
        references: { on_record: 3, none_on_record: 2, rows: 40 },
        figures: { on_record: 2, none_on_record: 1, rows: 9, described: 1, pdfs: 3 },
        letter_spacing: { recorded: 1, pdfs: 3 },
        keywords: { on_record: 5, none_on_record: 0, rows: 0 },
      },
    }),
  ) as MeterRow[]
  assert.deepEqual(
    some.map((r) => [r.label, r.have, r.of, r.tail]),
    [
      ['References', 3, 5, '2 none'],
      ['Figures', 2, 3, '1 none'],
      ['Spacing', 1, 3, '1 spaced'],
      ['Keywords', 5, 5, null],
    ],
  )
})

test('meter share has no division by zero', () => {
  assert.equal(meterShare(0, 0), 0)
  assert.equal(meterShare(3, 4), 0.75)
})

test('types are ranked; years keep decade order with no year apart', () => {
  const s = summary({
    types: [
      { format: 'md', documents: 1 },
      { format: 'pdf', documents: 4 },
    ],
    years: {
      by_decade: [
        { decade: 1950, documents: 2 },
        { decade: 1990, documents: 8 },
      ],
      no_year: 4,
      oldest: 1951,
      newest: 1999,
    },
  })
  assert.deepEqual(
    typeBars(s).map((b) => [b.label, b.value, b.share]),
    [
      ['PDF', 4, 1],
      ['MD', 1, 0.25],
    ],
  )
  const y = yearBars(s)
  assert.deepEqual(
    y.decades.map((b) => [b.label, b.share]),
    [
      ['1950s', 0.25],
      ['1990s', 1],
    ],
  )
  assert.deepEqual(y.noYear && [y.noYear.label, y.noYear.value, y.noYear.share], ['No year', 4, 0.5])
  assert.equal(yearBars(summary()).noYear, null)
  assert.deepEqual(typeBars(summary()), [])
})

test('what line', () => {
  assert.equal(
    whatLine(summary({ folders: { in_a_folder: 180, in_none: 132, also_elsewhere: null } })),
    '180 in a folder and 132 in none.',
  )
  assert.equal(
    whatLine(summary({ scope: 'folder', folders: { in_a_folder: null, in_none: null, also_elsewhere: 1 } })),
    '1 of these is also in another folder.',
  )
})
