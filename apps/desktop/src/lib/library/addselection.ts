// Pure helpers for the review sheet's ticks (AD2) — no runes, no fetch, so `node --test` reaches
// them. The sheet keeps the set of paths the user unticked; everything it shows is derived here.
//
// **Only a clean `add` can be ticked.** The server starts exactly those ticked
// (`FileVerdict.selected_by_default`) and nothing else: a duplicate would be re-registered and
// refused, and an unsupported or unreadable file cannot be added at all. So the user's choice is
// subtractive — untick what you do not want — and a tick can never reach a file the server ruled
// out. The 2026-08-20 plan's wireframe (§6.2) drew the ☑ on every addable row; the sheet shipped
// without it, and the 0.7.0 walkthrough asked for it.

import type { FileVerdict } from '../core/types/documents'

type Row = Pick<FileVerdict, 'path' | 'selected_by_default'>

/** True when the row carries a checkbox at all. */
export function tickable(f: Row): boolean {
  return f.selected_by_default
}

/** The rows that will be added: every tickable row the user has not unticked, in server order. */
export function selectedRows<T extends Row>(files: readonly T[], unticked: ReadonlySet<string>): T[] {
  return files.filter((f) => tickable(f) && !unticked.has(f.path))
}

/** The unticked set after *Select all* (empty) or *Select none* (every tickable row). */
export function tickAll(files: readonly Row[], on: boolean): Set<string> {
  return on ? new Set() : new Set(files.filter(tickable).map((f) => f.path))
}

/**
 * The line under a capped list. The server sorts every exception first, so the rows past the cap
 * are only ever clean additions; the line says how many of those are still ticked rather than
 * repeating "all ready to add" after a *Select none* made it false.
 */
export function hiddenLine(hidden: readonly Row[], unticked: ReadonlySet<string>): string {
  const n = hidden.length
  if (n === 0) return ''
  const ticked = selectedRows(hidden, unticked).length
  if (ticked === n) return `and ${n} more, all ready to add`
  if (ticked === 0) return `and ${n} more, none ticked`
  return `and ${n} more, ${ticked} ticked`
}
