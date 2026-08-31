// Presentation logic for "where does this passage sit?" (ROADMAP 19) — pure, so `node:test` runs it.
//
// The component fetches and renders; everything here is what the panel *says* about a position,
// which is the half worth guarding. The rule inherited from the ingest side: a position is either
// known or it is not. Nothing here interpolates, rounds a missing value into a present one, or
// turns an absent page into a plausible-looking number.

import type { ChunkContext } from '../core/types'

/** How far into the document the passage begins, 0-1. `null` when the document has no length to
 *  divide by — a zero-length cache is not "0% in", it is unmeasurable. */
export function position(ctx: ChunkContext | null): number | null {
  if (!ctx || ctx.doc_chars <= 0) return null
  return Math.min(1, Math.max(0, ctx.char_start / ctx.doc_chars))
}

/** The one-line "where" a citation cannot otherwise say.
 *
 * Leads with the position because it is always known, and appends the page only when there is
 * one — the parent-child path records no page on a parent, so most chat citations have none.
 * Saying "page ?" would be worse than saying nothing about pages at all. */
export function whereLabel(ctx: ChunkContext | null): string {
  const p = position(ctx)
  if (!ctx || p === null) return 'Position unknown'
  // Rounded to a whole percent: the precision beyond that is real but meaningless to a reader,
  // and a figure like "23.7%" implies the span is exact to the character in the ORIGINAL file,
  // which it is not — it is exact in the extracted markdown.
  const pct = Math.round(p * 100)
  const where = `${pct}% of the way in`
  return ctx.page != null ? `Page ${ctx.page} · ${where}` : where
}

/** Whether an ellipsis belongs before / after the window — i.e. whether text was cut off.
 *  Not the same as "the window is short": a window that reached the document's edge is complete,
 *  and marking it with an ellipsis would claim there is more to read when there is not. */
export function elision(ctx: ChunkContext | null): { before: boolean; after: boolean } {
  if (!ctx) return { before: false, after: false }
  return { before: !ctx.at_document_start, after: !ctx.at_document_end }
}

/** Collapse the runs of blank lines and page markers that make extracted markdown unreadable in a
 *  small panel, without touching the words. Purely cosmetic — the underlying offsets are untouched
 *  and the passage itself keeps its own text. */
export function tidy(text: string): string {
  return text
    .replace(/<!--\s*page:\d+\s*-->/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+\n/g, '\n')
}
