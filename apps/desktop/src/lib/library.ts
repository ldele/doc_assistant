// Client-side library navigation model (feature-library-redesign.md, L4 Phase A).
// Collections are computed from the LibraryDocument payload the API already ships
// (Decision 5: Phase A filters client-side) — no backend involvement; Phase B wires
// server-side folders + the folder-tree endpoint.
import type { LibraryDocument } from './types'

export type DateBucket = 'today' | 'week' | 'month' | 'earlier'

// The active collection shown in the main-pane grid. 'folder' is reachable in Phase A only
// if the payload already carries folder names (the current corpus has none — empty-stated).
export type LibraryCollection =
  | { kind: 'all' }
  | { kind: 'type'; value: string }
  | { kind: 'date'; value: DateBucket }
  | { kind: 'folder'; value: string }
  | { kind: 'keyword'; value: string }

export const DATE_BUCKET_LABELS: Record<DateBucket, string> = {
  today: 'Today',
  week: 'This week',
  month: 'This month',
  earlier: 'Earlier',
}
const DATE_BUCKET_ORDER: DateBucket[] = ['today', 'week', 'month', 'earlier']

// Bucket an ISO added_at relative to now (Decision 3b). An unknown date buckets nowhere —
// the doc stays reachable via All documents; claiming "Earlier" for an unknown date would
// break L1's honest-empty rule.
export function dateBucket(addedAt: string | null, now: Date): DateBucket | null {
  if (!addedAt) return null
  const t = new Date(addedAt).getTime()
  if (Number.isNaN(t)) return null
  const dayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const day = 86_400_000
  if (t >= dayStart) return 'today'
  if (t >= dayStart - 6 * day) return 'week'
  if (t >= dayStart - 29 * day) return 'month'
  return 'earlier'
}

export function docsFor(
  documents: LibraryDocument[],
  c: LibraryCollection,
  now: Date,
): LibraryDocument[] {
  switch (c.kind) {
    case 'all':
      return documents
    case 'type':
      return documents.filter((d) => d.format === c.value)
    case 'date':
      return documents.filter((d) => dateBucket(d.added_at, now) === c.value)
    case 'folder':
      return documents.filter((d) => d.folders.includes(c.value))
    case 'keyword':
      return documents.filter((d) => d.keywords.includes(c.value))
  }
}

export function collectionLabel(c: LibraryCollection): string {
  switch (c.kind) {
    case 'all':
      return 'All documents'
    case 'type':
      return c.value.toUpperCase()
    case 'date':
      return DATE_BUCKET_LABELS[c.value]
    case 'folder':
    case 'keyword':
      return c.value
  }
}

export function sameCollection(a: LibraryCollection, b: LibraryCollection): boolean {
  return (
    a.kind === b.kind &&
    (a as { value?: string }).value === (b as { value?: string }).value
  )
}

// Row label: prefer "Title — First Author" over the raw filename (which stays the tooltip).
// Moved here from Sidebar.svelte — the breadcrumb and search use the combined label.
export function docLabel(d: LibraryDocument): string {
  if (!d.title) return d.filename
  if (!d.authors) return d.title
  const first = authorLabel(d)
  return first ? `${d.title} · ${first}` : d.title
}

// Just the author part, or '' when unknown. Up to three authors show in full (books, small
// collaborations); four or more collapse to "First Author et al." The tile renders this on its
// own byline (the breadcrumb/search still use docLabel). Space-only author strings that don't
// split cleanly stay as one name and are ellipsis-truncated by the tile CSS (user can fix in
// the edit modal). Splits on ; , & and the word "and".
export function authorLabel(d: LibraryDocument): string {
  if (!d.authors) return ''
  const names = d.authors
    .split(/\s*(?:;|,|&|\band\b)\s*/)
    .map((n) => n.trim())
    .filter(Boolean)
  if (names.length === 0) return ''
  if (names.length <= 3) return names.join(', ')
  return `${names[0]} et al.`
}

// Library sort order — a client-side sort over the already-filtered collection. Directions match
// the useful default per key (dates newest-first, names A→Z).
export type LibrarySort = 'title-az' | 'author-az' | 'pub-desc' | 'added-desc'

export function sortDocs(docs: LibraryDocument[], sort: LibrarySort): LibraryDocument[] {
  const copy = [...docs]
  const title = (d: LibraryDocument) => (d.title ?? d.filename).toLowerCase()
  // No-author docs sort last under Author A→Z (￿ is above any real letter).
  const author = (d: LibraryDocument) => (authorLabel(d) || '￿').toLowerCase()
  switch (sort) {
    case 'title-az':
      return copy.sort((a, b) => title(a).localeCompare(title(b)))
    case 'author-az':
      return copy.sort((a, b) => author(a).localeCompare(author(b)) || title(a).localeCompare(title(b)))
    case 'pub-desc':
      return copy.sort((a, b) => (b.year ?? -Infinity) - (a.year ?? -Infinity))
    case 'added-desc':
      return copy.sort((a, b) => (b.added_at ?? '').localeCompare(a.added_at ?? ''))
  }
}

// Search filter (Decision 5a): same fields the old rail search matched.
export function filterDocs(documents: LibraryDocument[], query: string): LibraryDocument[] {
  const q = query.trim().toLowerCase()
  if (q === '') return documents
  return documents.filter(
    (d) =>
      docLabel(d).toLowerCase().includes(q) ||
      d.filename.toLowerCase().includes(q) ||
      (d.authors ?? '').toLowerCase().includes(q),
  )
}

export interface Group<T extends string = string> {
  value: T
  count: number
}

export function typeGroups(documents: LibraryDocument[]): Group[] {
  const m = new Map<string, number>()
  for (const d of documents) m.set(d.format, (m.get(d.format) ?? 0) + 1)
  return [...m.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value))
}

export function dateGroups(documents: LibraryDocument[], now: Date): Group<DateBucket>[] {
  const m = new Map<DateBucket, number>()
  for (const d of documents) {
    const b = dateBucket(d.added_at, now)
    if (b) m.set(b, (m.get(b) ?? 0) + 1)
  }
  return DATE_BUCKET_ORDER.filter((b) => m.has(b)).map((b) => ({
    value: b,
    count: m.get(b) ?? 0,
  }))
}

// Flat folder groups from the payload's folder names. Phase A renders these only if a
// populated corpus already carries them; hierarchy (expandable tree) is Phase B.
export function folderGroups(documents: LibraryDocument[]): Group[] {
  const m = new Map<string, number>()
  for (const d of documents) for (const f of d.folders) m.set(f, (m.get(f) ?? 0) + 1)
  return [...m.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => a.value.localeCompare(b.value))
}

// Top keywords by doc count (capped — the auto-extracted vocabulary is long-tailed).
// A keyword with 0 docs can't appear by construction (Decision 8).
export function keywordGroups(documents: LibraryDocument[], cap = 12): Group[] {
  const m = new Map<string, number>()
  for (const d of documents) for (const k of d.keywords) m.set(k, (m.get(k) ?? 0) + 1)
  return [...m.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value))
    .slice(0, cap)
}
