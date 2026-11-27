// The library summary's words and shapes (ROADMAP 112, docs/wireframes/library-summary.md, rev 2).
//
// Pure: the counts come from the API (library.summary owns what each one means); this module only
// turns them into what the Summary layout shows, so the zero, plural and "not read" rules are
// tested in one place. Never a percentage: a collection of 0 documents has nothing to divide by.
//
// A count with documents behind it is **navigation**, not an action (ADR-063): the screen makes
// the whole row the target. `target` is where a row leads; `null` makes it plain text.

import type { CollectionSummary, SummaryIssue } from '../core/types/summary'

/** Where a row leads: some documents, the documents folder's files, the indexing screen, or the
 *  library's own summary. The summary opens a fix; it never runs one. */
export type SummaryTarget =
  | { kind: 'issue'; issue: SummaryIssue }
  | { kind: 'files' }
  | { kind: 'indexing' }
  | { kind: 'library' }

/** A *From indexing* row: what happened and how many, then where to fix it. */
export interface IndexingRow {
  key: string
  title: string
  detail: string
  /** The fix, named at the row's end; `null` on a row with nothing to open. */
  fix: string | null
  tone: 'warn' | 'crit' | 'none'
  target: SummaryTarget | null
}

/** "N of M" with a thin bar, and what is left to open. */
export interface MeterRow {
  kind: 'meter'
  key: string
  label: string
  have: number
  of: number
  /** What the row opens, e.g. `12 missing`; `null` when there is nothing to open. */
  tail: string | null
  issue: SummaryIssue | null
}

/** Said once instead of rows: nothing has looked, so a zero is not a defect. */
export interface NotReadLine {
  kind: 'line'
  key: string
  title: string
  detail: string
}

/** One count under a pass (0.9.0, the extraction record): what it names, how many, and the
 *  documents it opens. A count of 0 opens nothing and is plain text. `sub` indents it under the
 *  pass it belongs to. */
export interface CountRow {
  kind: 'count'
  key: string
  label: string
  count: number
  issue: SummaryIssue | null
  sub: boolean
}

export type CardRow = MeterRow | NotReadLine | CountRow

export interface Bar {
  key: string
  label: string
  value: number
  /** Length as a fraction of the longest bar in the chart. */
  share: number
}

const NUMBER = new Intl.NumberFormat('en-US')

/** `41180` → `41,180`. */
export function fmt(n: number): string {
  return NUMBER.format(n)
}

/** `1 document` / `3 documents`, with the count formatted. */
export function plural(n: number, one: string, many: string): string {
  return `${fmt(n)} ${n === 1 ? one : many}`
}

/** `1980` → `1980s`. */
export function decadeLabel(decade: number): string {
  return `${decade}s`
}

/** The context line under the title: passages and pages, which say how big, not what is wrong. */
export function contextLine(s: CollectionSummary): string {
  const pages =
    s.paged_documents < s.documents
      ? `${plural(s.pages, 'page', 'pages')} in ${plural(s.paged_documents, 'paged document', 'paged documents')}`
      : plural(s.pages, 'page', 'pages')
  return `${plural(s.passages, 'passage', 'passages')} · ${pages}`
}

/** The two tiles, *Need attention* first. Its number is the server's, never re-summed here. */
export function tiles(s: CollectionSummary): { key: string; value: string; caption: string; alert: boolean }[] {
  return [
    {
      key: 'need',
      value: fmt(s.need_attention),
      caption: s.scope === 'library' ? 'Need attention: files or documents' : 'Need attention: documents',
      alert: s.need_attention > 0,
    },
    {
      key: 'documents',
      value: fmt(s.documents),
      caption: s.documents === 1 ? 'Document' : 'Documents',
      alert: false,
    },
  ]
}

const FIX_TEXT = 'Fix: Re-run… > Text & chunks'

/** The *From indexing* rows: known for every document, so a zero is said in words, not hidden. */
export function indexingRows(s: CollectionSummary): IndexingRow[] {
  const rows: IndexingRow[] = []
  const ix = s.indexing

  if (s.scope === 'library' && ix.files_not_in_library !== null) {
    const n = ix.files_not_in_library
    const failed = Math.min(ix.last_run_failed ?? 0, n)
    if (n > 0) {
      const detail =
        failed === n
          ? `${n === 1 ? 'It' : 'They'} failed in this session's run.`
          : failed > 0
            ? `${plural(failed, 'failed', 'failed')} in this session's run; the others were never indexed.`
            : 'Never indexed, or indexing failed; no run this session says why.'
      rows.push({
        key: 'files',
        title: `${plural(n, 'file', 'files')} in your documents folder ${n === 1 ? 'is' : 'are'} not in the library`,
        detail,
        fix: failed > 0 ? 'See why on the indexing screen' : 'Index them in Settings > Documents',
        tone: 'warn',
        target: failed > 0 ? { kind: 'indexing' } : { kind: 'files' },
      })
    } else {
      rows.push({
        key: 'files',
        title: ix.files_seen_at
          ? 'Every file in your documents folder is in the library'
          : 'Your documents folder has not been looked at yet',
        detail: ix.files_seen_at
          ? 'As of the last look at the folder, from Settings or an index run.'
          : 'Settings > Documents lists its files, and indexing reads them.',
        fix: null,
        tone: 'none',
        target: null,
      })
    }
  }

  if (s.documents > 0 || s.scope === 'folder') {
    const nt = ix.no_text
    rows.push({
      key: 'no_text',
      title: `${plural(nt, 'document', 'documents')} gave no text`,
      detail:
        nt > 0
          ? 'Probably scans with no text layer: nothing in them can be searched or cited.'
          : 'Every document gave text when it was indexed.',
      fix: nt > 0 ? FIX_TEXT : null,
      tone: nt > 0 ? 'crit' : 'none',
      target: nt > 0 ? { kind: 'issue', issue: 'no_text' } : null,
    })

    const fl = ix.flagged_marginal + ix.flagged_broken
    const older =
      ix.not_classified > 0
        ? ` ${plural(ix.not_classified, 'document was', 'documents were')} indexed before this check existed.`
        : ''
    rows.push({
      key: 'flagged',
      title: `${plural(fl, 'extraction was', 'extractions were')} flagged`,
      detail:
        (fl > 0
          ? 'Marked marginal or broken when indexed: little text per page, or very short passages. The reason is not kept.'
          : 'None was marked marginal or broken when indexed.') + older,
      fix: fl > 0 ? FIX_TEXT : null,
      tone: fl > 0 ? 'warn' : 'none',
      target: fl > 0 ? { kind: 'issue', issue: 'flagged' } : null,
    })

    rows.push(...textCheckRows(s))
  }

  if (s.scope === 'folder') {
    rows.push({
      key: 'folder-files',
      title: 'Files that are not in the library belong to no folder',
      detail: "The library's summary counts them.",
      fix: "Open the library's summary",
      tone: 'none',
      target: { kind: 'library' },
    })
  }
  return rows
}

/** A2b's rows: documents whose text addresses an AI, and documents the check has not read.
 *  A mark informs: nothing was removed, and a paper about prompt injection is marked too. */
export function textCheckRows(s: CollectionSummary): IndexingRow[] {
  const t = s.found.text_checks
  const rows: IndexingRow[] = []
  if (t.marked > 0) {
    rows.push({
      key: 'addressed_to_ai',
      title: `${plural(t.marked, 'document contains', 'documents contain')} text addressed to an AI`,
      detail:
        'Text written to an AI assistant rather than to a reader, such as planted instructions. A paper about AI that quotes such text is marked too. Nothing was removed.',
      fix: 'Read it on each document’s page',
      tone: 'warn',
      target: { kind: 'issue', issue: 'addressed_to_ai' },
    })
  } else if (t.checked > 0) {
    rows.push({
      key: 'addressed_to_ai',
      title: `No text addressed to an AI in the ${plural(t.checked, 'document', 'documents')} checked`,
      detail: 'The check looks for planted instructions and the like. It can miss a new phrasing.',
      fix: null,
      tone: 'none',
      target: null,
    })
  }
  if (t.not_checked > 0) {
    rows.push({
      key: 'text_not_checked',
      title: `${plural(t.not_checked, 'document was', 'documents were')} not checked for text addressed to an AI`,
      detail: 'Indexed before this check existed, or the text changed since. The check takes a moment per document.',
      fix: 'Fix: Re-run… > Text checks',
      tone: 'none',
      target: { kind: 'issue', issue: 'text_not_checked' },
    })
  }
  return rows
}

function meter(key: SummaryIssue, label: string, have: number, of: number, left: number, word: string): MeterRow {
  return {
    kind: 'meter',
    key,
    label,
    have,
    of,
    tail: left > 0 ? `${fmt(left)} ${word}` : null,
    issue: left > 0 ? key : null,
  }
}

/** The *Metadata* card: four meters, or one line when no document has any field yet. */
export function metadataRows(s: CollectionSummary): CardRow[] {
  const m = s.metadata
  const n = s.documents
  if (n > 0 && m.with_any_field === 0) {
    return [
      {
        kind: 'line',
        key: 'metadata-not-read',
        title: `Not read for ${plural(n, 'document', 'documents')}`,
        detail:
          'Indexing reads it for new or changed documents only. For documents indexed earlier, Re-run… > Metadata reads it from the first page.',
      },
    ]
  }
  return [
    meter('missing_title', 'Title', n - m.missing_title, n, m.missing_title, 'missing'),
    meter('missing_authors', 'Authors', n - m.missing_authors, n, m.missing_authors, 'missing'),
    meter('missing_year', 'Year', n - m.missing_year, n, m.missing_year, 'missing'),
    meter('missing_doi', 'DOI', n - m.missing_doi, n, m.missing_doi, 'missing'),
  ]
}

/** The *Found in each document* card. A kind with nothing on record anywhere says "not looked for"
 *  once: nothing records that a re-run looked, so its zero is not a finding. */
function count(key: string, label: string, n: number, issue: SummaryIssue | null, sub = true): CountRow {
  return { kind: 'count', key, label, count: n, issue: n > 0 ? issue : null, sub }
}

/** The references, figures and tables rows once any document has an extraction record (0.9.0,
 *  docs/wireframes/extraction-record.md S4): each pass split by what its record says. */
function recordedRows(s: CollectionSummary): CardRow[] {
  const f = s.found
  const r = f.record
  const n = s.documents
  const pdfs = f.figures.pdfs
  const rows: CardRow[] = [
    { kind: 'meter', key: 'references', label: 'References', have: r.references_listed, of: n, tail: null, issue: null },
    count('references_heading_only', 'A references heading, but no entries', r.references_heading_only, 'references_heading_only'),
    count('references_none', 'No reference list found', r.references_none, 'references_none'),
    count('references_few', 'Only a few references found', r.references_few, 'references_few'),
    count('references_not_looked_for', 'Not looked for yet', r.references_not_looked_for, 'references_not_looked_for'),
  ]
  if (pdfs > 0) {
    rows.push(
      meter('no_figures', 'Figures', f.figures.on_record, pdfs, f.figures.none_on_record, 'none'),
      count(
        'figures_uncaptioned',
        r.figures_uncaptioned > 0 ? `${plural(r.figures_uncaptioned, 'figure', 'figures')} without a caption, in` : 'Figures without a caption',
        r.figures_uncaptioned_documents,
        'figures_uncaptioned',
      ),
      count('tables', r.tables_extracted > 0 ? `Tables: ${plural(r.tables_extracted, 'table', 'tables')} extracted, in` : 'Tables extracted', r.tables_documents, null, false),
      count('tables_missed', "Name a table that wasn't extracted", r.tables_missed, 'tables_missed'),
      count('tables_not_looked_for', 'Tables not looked for yet', r.tables_not_looked_for, null),
    )
  }
  return rows
}

export function foundRows(s: CollectionSummary): CardRow[] {
  const f = s.found
  const n = s.documents
  const rows: CardRow[] = []

  if (f.record.recorded > 0) {
    rows.push(...recordedRows(s))
  } else if (n > 0 && f.references.on_record === 0) {
    rows.push({
      kind: 'line',
      key: 'references-not-read',
      title: `References: not looked for in ${plural(n, 'document', 'documents')}`,
      detail: 'Indexed before indexing read references. Re-run… > References reads them.',
    })
  } else {
    rows.push(meter('no_references', 'References', f.references.on_record, n, f.references.none_on_record, 'none'))
  }

  const pdfs = f.figures.pdfs
  if (f.record.recorded > 0) {
    // the figures and tables rows are in `recordedRows`
  } else if (pdfs > 0 && f.figures.on_record === 0) {
    rows.push({
      kind: 'line',
      key: 'figures-not-read',
      title: `Figures: not looked for in ${plural(pdfs, 'PDF', 'PDFs')}`,
      detail: 'Indexed before indexing read figures. Re-run… > Figures reads them.',
    })
  } else if (pdfs > 0) {
    rows.push(meter('no_figures', 'Figures', f.figures.on_record, pdfs, f.figures.none_on_record, 'none'))
  }

  if (f.letter_spacing.pdfs > 0) {
    const r = f.letter_spacing.recorded
    rows.push({
      kind: 'meter',
      key: 'letter_spaced',
      label: 'Spacing',
      have: r,
      of: f.letter_spacing.pdfs,
      tail: r > 0 ? `${fmt(r)} spaced` : null,
      issue: r > 0 ? 'letter_spaced' : null,
    })
  }

  if (n > 0 && f.keywords.on_record === 0) {
    rows.push({
      kind: 'line',
      key: 'keywords-not-read',
      title: `Keywords: none on record for ${plural(n, 'document', 'documents')}`,
      detail: 'Keywords come from the keyword pass, not from indexing.',
    })
  } else {
    rows.push(meter('no_keywords', 'Keywords', f.keywords.on_record, n, f.keywords.none_on_record, 'none'))
  }
  return rows
}

/** Bar length as a share of `of`, for a meter. 0 when there is nothing to measure against. */
export function meterShare(have: number, of: number): number {
  return of > 0 ? Math.min(1, Math.max(0, have / of)) : 0
}

function withShares(items: { key: string; label: string; value: number }[]): Bar[] {
  const top = items.reduce((m, it) => Math.max(m, it.value), 0)
  return items.map((it) => ({ ...it, share: top > 0 ? it.value / top : 0 }))
}

/** *Types*: ranked bars, largest first (the server's order, re-sorted here so it cannot drift). */
export function typeBars(s: CollectionSummary): Bar[] {
  const ranked = [...s.types].sort((a, b) => b.documents - a.documents || a.format.localeCompare(b.format))
  return withShares(ranked.map((t) => ({ key: t.format, label: t.format.toUpperCase(), value: t.documents })))
}

/** *Years*: decades in time order (an ordered scale keeps its order), and *No year* apart, on the
 *  same scale so its length compares with theirs. Both empty when no document has a year. */
export function yearBars(s: CollectionSummary): { decades: Bar[]; noYear: Bar | null } {
  const items = s.years.by_decade.map((d) => ({ key: String(d.decade), label: decadeLabel(d.decade), value: d.documents }))
  const all = withShares([...items, { key: 'none', label: 'No year', value: s.years.no_year }])
  const noYear = all[all.length - 1]
  return { decades: all.slice(0, -1), noYear: s.years.no_year > 0 ? noYear : null }
}

function day(iso: string | null): string | null {
  if (!iso) return null
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return null
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
}

/** The line under *What is in it*: folder membership and the dates documents were added. */
export function whatLine(s: CollectionSummary): string {
  const parts: string[] = []
  if (s.scope === 'library' && s.folders.in_a_folder !== null && s.folders.in_none !== null) {
    parts.push(`${fmt(s.folders.in_a_folder)} in a folder and ${fmt(s.folders.in_none)} in none.`)
  } else if (s.folders.also_elsewhere !== null) {
    parts.push(`${plural(s.folders.also_elsewhere, 'of these is', 'of these are')} also in another folder.`)
  }
  const first = day(s.added.first)
  const last = day(s.added.last)
  if (first && last) parts.push(first === last ? `Added on ${first}.` : `Added between ${first} and ${last}.`)
  return parts.join(' ')
}

/** `14:02` for the counted-at line, in the reader's clock. */
export function countedAt(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
}
