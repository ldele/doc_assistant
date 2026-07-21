// Global-search match layer (docs/specs/feature-app-shell-search-collapse.md, A1/A8). Pure by
// design so it is exercised by `npm test` (node:test) rather than eyeballed in the overlay — the
// GlobalSearch component is a dumb renderer, App owns the data + navigation.
//
// This is a *navigation* search (A1): it matches conversation TITLES and document
// title/filename/authors/keywords, so you can jump to a chat or a paper. It deliberately does NOT
// search message bodies or chunk text — that is the composer's job (corpus retrieval), and those
// bodies are not client-side anyway. Keeping the two distinct is the same honesty rule the
// integrity layer enforces: a box that looked like corpus search but wasn't would mislead.
import type { ConversationSummary, LibraryDocument } from './types'

// 'recent' = the empty-query state (A4: recent chats, no docs). 'query' = an actual search, with
// per-group caps + overflow totals so the overlay can render an honest "+N more" (A3).
export interface SearchResults {
  mode: 'recent' | 'query'
  chats: ConversationSummary[]
  docs: LibraryDocument[]
  chatsTotal: number // matches before the per-group cap
  docsTotal: number
}

export interface SearchOpts {
  perGroup?: number // cap per group in the query state (A3)
  recentCount?: number // how many recent chats in the empty state (A4)
}

const DEFAULT_PER_GROUP = 8
const DEFAULT_RECENT = 6

// NaN-safe ISO → epoch, mirroring the sidebar's own `ts()` so recency ordering is identical.
function ts(iso: string): number {
  const t = new Date(iso).getTime()
  return Number.isNaN(t) ? 0 : t
}

// The document's display title, or the filename when it has none. Kept self-contained (not
// `library.docLabel`) so this module has no runtime cross-module import — node:test strips the
// type-only `./types` import but would fail to resolve an extensionless value import (the same
// reason library.ts imports only types). Title-only is also the right sort key: papers sort by
// title, not by "Title · Author".
function docTitle(d: LibraryDocument): string {
  return d.title ?? d.filename
}

function docMatches(d: LibraryDocument, q: string): boolean {
  return (
    docTitle(d).toLowerCase().includes(q) ||
    d.filename.toLowerCase().includes(q) ||
    (d.authors ?? '').toLowerCase().includes(q) ||
    d.keywords.some((k) => k.toLowerCase().includes(q))
  )
}

// The one entry point. `chats`/`docs` are the full client-side lists; ordering of the result is
// owned here (chats by recency desc, docs by label asc) so the overlay stays presentation-only.
export function searchEverything(
  query: string,
  chats: ConversationSummary[],
  docs: LibraryDocument[],
  opts: SearchOpts = {},
): SearchResults {
  const perGroup = opts.perGroup ?? DEFAULT_PER_GROUP
  const recentCount = opts.recentCount ?? DEFAULT_RECENT
  const q = query.trim().toLowerCase()

  // Empty query → recent, non-archived chats only (A4). Never dump the whole document list.
  if (q === '') {
    const recent = chats
      .filter((c) => !c.archived)
      .sort((a, b) => ts(b.last_at) - ts(a.last_at))
      .slice(0, recentCount)
    return { mode: 'recent', chats: recent, docs: [], chatsTotal: recent.length, docsTotal: 0 }
  }

  const chatHits = chats
    .filter((c) => c.title.toLowerCase().includes(q))
    .sort((a, b) => ts(b.last_at) - ts(a.last_at))
  const docHits = docs
    .filter((d) => docMatches(d, q))
    .sort((a, b) => docTitle(a).toLowerCase().localeCompare(docTitle(b).toLowerCase()))

  return {
    mode: 'query',
    chats: chatHits.slice(0, perGroup),
    docs: docHits.slice(0, perGroup),
    chatsTotal: chatHits.length,
    docsTotal: docHits.length,
  }
}
