// What indexing found, shaped for the document page (0.9.0, docs/specs/ingestion-record.md,
// docs/wireframes/extraction-record.md S2/S3). Pure, so it is tested under node:test; the block
// (`screens/document-page/WhatIndexingFound.svelte`) only renders what this returns. Every warning
// sentence is the backend's (`ingest.record.WARNINGS`): nothing here re-phrases one.

import type { DocumentExtractionRecord, ExtractionRecord, RecordWarning } from '../core/types/extractionRecord.ts'

/** What a warning's button does: re-run some parts (ticked in the dialog), or open the metadata
 *  editor. `null`: nothing in the app fixes it (empty pages are the file's own). */
export type WarningFix = { kind: 'rerun'; parts: string[] } | { kind: 'metadata' } | null

export function fixFor(w: RecordWarning): WarningFix {
  if (w.id.startsWith('references_')) return { kind: 'rerun', parts: ['references'] }
  if (w.id === 'figures_uncaptioned') return { kind: 'rerun', parts: ['figures'] }
  if (w.id === 'tables_missed' || w.id === 'stale') return { kind: 'rerun', parts: ['text'] }
  if (w.id === 'metadata_missing') return { kind: 'metadata' }
  return null
}

/** What the block shows: never recorded, recorded for an earlier reading, or the record itself. */
export type RecordView =
  | { kind: 'loading' }
  | { kind: 'never' }
  | { kind: 'stale'; when: string }
  | { kind: 'current'; record: ExtractionRecord }

export function recordView(d: DocumentExtractionRecord | null | undefined): RecordView {
  if (d === undefined || d === null) return { kind: d === undefined ? 'loading' : 'never' }
  if (d.record === null) return { kind: 'never' }
  if (d.record.stale) return { kind: 'stale', when: dateOf(d.record.created_at) }
  return { kind: 'current', record: d.record }
}

const DATE = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
const TIME = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit' })

/** A naive server timestamp is UTC (the API's rule for stored times). */
function parse(iso: string): Date {
  return new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`)
}

export function dateOf(iso: string): string {
  return DATE.format(parse(iso))
}

/** "Recorded when this document was indexed, 27 Nov 2026 at 17:52." */
export function whenLine(r: ExtractionRecord): string {
  const d = parse(r.created_at)
  const how = r.run_kind === 'index' ? 'was indexed' : 'was last re-run'
  return `Recorded when this document ${how}, ${DATE.format(d)} at ${TIME.format(d)}.`
}

function n(count: number | null | undefined, one: string, many: string): string {
  const c = count ?? 0
  return `${c} ${c === 1 ? one : many}`
}

/** One line per pass and one for metadata, in the drawing's order. A pass that did not run says
 *  why, in the record's own words. */
export function factLines(r: ExtractionRecord): { label: string; text: string }[] {
  const out: { label: string; text: string }[] = []
  const refs = r.references
  const rs = r.states.references
  if (rs === 'found' || rs === 'partial') {
    const parts = [`${n(refs.entries, 'reference', 'references')} found`]
    if (refs.without_year) parts.push(`${refs.without_year} without a year`)
    if (refs.matched) parts.push(`${refs.matched} of them in your library`)
    out.push({ label: 'References', text: parts.join(' · ') })
  } else if (rs === 'none_found') {
    out.push({ label: 'References', text: refs.heading ? 'a references heading, but no entries' : 'none found' })
  } else {
    out.push({ label: 'References', text: refs.reason ?? 'not looked for yet' })
  }

  const fig = r.figures
  const fs = r.states.figures
  if (fs === 'found') {
    const parts = [`${n(fig.found, 'figure', 'figures')} found`, `${fig.with_caption ?? 0} with a caption`]
    if (fig.caption_only) parts.push(`${fig.caption_only} found by their caption only`)
    out.push({ label: 'Figures', text: parts.join(' · ') })
  } else if (fs === 'none_found') {
    out.push({ label: 'Figures', text: 'none found' })
  } else {
    out.push({ label: 'Figures', text: fig.reason ?? 'not looked for yet' })
  }

  const tab = r.tables
  const ts = r.states.tables
  if (ts === 'found' || ts === 'partial' || ts === 'none_found') {
    const parts = [`${tab.extracted} extracted`]
    if (tab.pages_naming_a_table.length) parts.push(`${n(tab.pages_naming_a_table.length, 'page names', 'pages name')} a table`)
    out.push({ label: 'Tables', text: parts.join(' · ') })
  } else {
    out.push({ label: 'Tables', text: tab.reason ?? 'not looked for yet' })
  }

  const m = r.metadata
  const fields: [string, boolean][] = [['title', m.title], ['authors', m.authors], ['year', m.year]]
  const have = fields.filter(([, ok]) => ok).map(([k]) => k)
  const missing = fields.filter(([, ok]) => !ok).map(([k]) => k)
  if (m.doi) have.push('DOI')
  const metaParts: string[] = []
  if (have.length) metaParts.push(`${have.join(', ')} found`)
  if (missing.length) metaParts.push(`no ${missing.join(', no ')}`)
  out.push({ label: 'Metadata', text: metaParts.join(' · ') })
  return out
}
