// Client-side library navigation model (feature-library-redesign.md, L4 Phase A).
// Collections are computed from the LibraryDocument payload the API already ships
// (Decision 5: Phase A filters client-side) — no backend involvement; Phase B wires
// server-side folders + the folder-tree endpoint.
import type { KeywordFamily, LibraryDocument, LibraryFolder } from './types'

export type DateBucket = 'today' | 'week' | 'month' | 'earlier'

// The active collection shown in the main-pane grid.
// 'folder' carries a folder **id**, not a name (ADR-025 F1, spec D2): root folder names are not
// unique (the DB constraint is (name, parent_folder_id) and SQLite treats NULL parents as
// distinct), so a name cannot key a filter.
// Keywords are NOT a collection kind — they are a multi-select facet (see keywordFacets below),
// orthogonal to the single-select structural nav. Type/Date/Folder stay mutually exclusive.
export type LibraryCollection =
  | { kind: 'all' }
  | { kind: 'type'; value: string }
  | { kind: 'date'; value: DateBucket }
  | { kind: 'folder'; value: string }

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
      return documents.filter((d) => d.folder_ids.includes(c.value))
  }
}

// `folderNames` resolves a folder id → its display name. A folder deleted in another view falls
// back to a neutral label rather than showing a raw uuid.
export function collectionLabel(c: LibraryCollection, folderNames?: Map<string, string>): string {
  switch (c.kind) {
    case 'all':
      return 'All documents'
    case 'type':
      return c.value.toUpperCase()
    case 'date':
      return DATE_BUCKET_LABELS[c.value]
    case 'folder':
      return folderNames?.get(c.value) ?? 'Folder'
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

// NOTE: there is deliberately no `folderGroups(documents)` (retired in ADR-025 F1, spec D3).
// A folder derived from the document payload cannot exist while empty — and an empty folder you
// cannot see is a folder you cannot add documents to. The rail renders GET /api/library/folders
// instead, which carries every folder plus its own doc_count.
export function folderNameMap(folders: LibraryFolder[]): Map<string, string> {
  return new Map(folders.map((f) => [f.id, f.name]))
}

// ---------------------------------------------------------------------------
// Keyword facets — multi-select filtering over the keyword chips (AND semantics).
// ---------------------------------------------------------------------------

export interface KeywordFacet {
  value: string
  count: number // docs in the current (faceted) result set that carry this keyword
  selected: boolean
  available: boolean // adding this keyword still leaves a result (always true when selected)
}

// `keywordsOf` extracts a document's facet units. Defaults to its raw keywords; a caller passes
// `familyUnitsOf(...)` (below) to filter/count over family-collapsed units instead. The default
// makes the no-families path byte-identical to the pre-PR-1 behavior.
type KeywordsOf = (d: LibraryDocument) => string[]
const rawKeywordsOf: KeywordsOf = (d) => d.keywords

// AND semantics: a document must carry *every* selected keyword. Empty selection = no filter.
// This is the narrowing that makes "grey out unavailable" meaningful — under OR, adding a
// keyword only ever broadens, so nothing is ever unavailable.
export function facetFilter(
  documents: LibraryDocument[],
  selected: string[],
  keywordsOf: KeywordsOf = rawKeywordsOf,
): LibraryDocument[] {
  if (selected.length === 0) return documents
  return documents.filter((d) => {
    const units = new Set(keywordsOf(d))
    return selected.every((k) => units.has(k))
  })
}

// Facet chips for the current pool. `base` is the collection+search result *before* keyword
// facets; `selected` the active keywords. Each chip's `count` is how many of the current
// (faceted) results also carry it — i.e. what the grid would narrow to if you added it — so a
// chip greys out (`available: false`) exactly when adding it would empty the grid. Selected chips
// always appear (pinned first) and stay `available` so they can be removed even when the
// selection is over-constrained. Pure + deterministic (ties break alphabetically), so it is
// unit-testable and stable across renders.
export function keywordFacets(
  base: LibraryDocument[],
  selected: string[],
  keywordsOf: KeywordsOf = rawKeywordsOf,
): KeywordFacet[] {
  const sel = new Set(selected)
  const faceted = facetFilter(base, selected, keywordsOf)
  const counts = new Map<string, number>()
  for (const d of faceted)
    for (const k of new Set(keywordsOf(d))) counts.set(k, (counts.get(k) ?? 0) + 1)
  // Universe = every unit in the pre-facet pool, plus the selected ones (which may no longer
  // appear in `base` after a collection switch but must still render as removable chips).
  const universe = new Set<string>(selected)
  for (const d of base) for (const k of keywordsOf(d)) universe.add(k)

  const facets: KeywordFacet[] = [...universe].map((value) => {
    const isSelected = sel.has(value)
    const count = counts.get(value) ?? 0
    return { value, count, selected: isSelected, available: isSelected || count > 0 }
  })
  facets.sort((a, b) => {
    if (a.selected !== b.selected) return a.selected ? -1 : 1
    if (a.available !== b.available) return a.available ? -1 : 1
    return b.count - a.count || a.value.localeCompare(b.value)
  })
  return facets
}

// ---------------------------------------------------------------------------
// Tag families (feature-tag-families.md, PR-1) — synonym-collapse over keyword facets.
// A family is a curated Concept whose alias keywords collapse into one facet unit
// (ADR-015). This is a pre-facet grouping step: `facetFilter`/`keywordFacets` above stay
// unchanged and operate on whatever `keywordsOf` hands them.
// ---------------------------------------------------------------------------

// Case-insensitive keyword-name -> canonical-label map (a family's own canonical maps to
// itself too, so re-mapping an already-canonical value is a no-op). Un-familied keywords are
// absent — callers fall back to the raw keyword name.
export function familyCanonicalMap(families: KeywordFamily[]): Map<string, string> {
  const m = new Map<string, string>()
  for (const f of families) {
    m.set(f.canonical.toLowerCase(), f.canonical)
    for (const alias of f.aliases) m.set(alias.toLowerCase(), f.canonical)
  }
  return m
}

// Canonical label -> its family, for "N forms" display (the overlay looks up a facet's value
// here to show its member keywords). Absent = an un-familied, single-keyword facet.
export function familyByCanonical(families: KeywordFamily[]): Map<string, KeywordFamily> {
  return new Map(families.map((f) => [f.canonical, f]))
}

// A `keywordsOf` accessor that collapses each doc's keywords through `canonicalOf` (deduping),
// for use with `facetFilter`/`keywordFacets`. When `canonicalOf` is empty (no families exist)
// this is behaviorally identical to the raw keyword list.
export function familyUnitsOf(canonicalOf: Map<string, string>): KeywordsOf {
  return (d) => {
    const units = new Set<string>()
    for (const k of d.keywords) units.add(canonicalOf.get(k.toLowerCase()) ?? k)
    return [...units]
  }
}

// Re-point a live keyword selection after a family write (PR-2.5 D5). The Manage view is opened
// *from* the facet overlay — i.e. exactly where a selection is live — so grouping `llm` into a
// family left the grid filtering on a unit that no longer exists: empty results behind a chip that
// still looked selectable (`keywordFacets`' universe includes `selected` by design, so it renders).
//
// Two moves, one for each direction of a family write: map each selection through the *new*
// canonical map (create/rename/add-member), then drop anything that is no longer a unit of any
// document (delete/remove-member). `documents` must be the whole library, not the active
// collection — a selection that is merely out-of-collection must stay, so it can still be removed.
export function remapSelection(
  selected: string[],
  canonicalOf: Map<string, string>,
  documents: LibraryDocument[],
): string[] {
  if (selected.length === 0) return selected
  const unitsOf = familyUnitsOf(canonicalOf)
  const live = new Set<string>()
  for (const d of documents) for (const u of unitsOf(d)) live.add(u)
  const remapped = selected.map((k) => canonicalOf.get(k.toLowerCase()) ?? k)
  return [...new Set(remapped)].filter((k) => live.has(k))
}

// Float the selected units to the front of a tile's chip row (PR-2.6). The reason a document is
// in the grid must be visible even when its chips would otherwise fall past the `+N` cap — so this
// is ordering, not filtering: nothing is dropped, and an empty or non-matching selection returns
// the input untouched (that is what keeps the no-families path byte-identical).
//
// It takes *units*, not raw keywords, which is the whole of defect D6: the grid matched
// `activeKeywords.includes(rawKeyword)`, but a family selection holds the **canonical**
// (`Large language model`) while tiles held the raw forms (`llm`/`llms`), so the match could never
// fire — measured live at 0 of 25 chips highlighted against 19 for a plain keyword.
export function orderedUnits(units: string[], active: string[]): string[] {
  if (active.length === 0) return units
  const selected = new Set(active)
  const hit = units.filter((u) => selected.has(u))
  if (hit.length === 0) return units
  return [...hit, ...units.filter((u) => !selected.has(u))]
}
