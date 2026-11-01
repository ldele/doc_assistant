// The letter-spacing block's decisions (ADR-056 points 9 and 10), pure so they can be tested.
//
// A run reaches the app twice: `spaced`, as PyMuPDF read it (one space between every two glyphs),
// and `written`, the same glyphs with fewer spaces, as the text now holds them. Everything the
// block shows comes from comparing the two: a tick where a space was removed, a word break where
// one was kept. The editor works on the same gaps and turns them back into a spacing, so it can
// only ever move spaces (the backend refuses anything else as well).

/** One glyph of a run as shown: the character, whether a space sits before it in the text, and
 *  whether a space PyMuPDF read before it was removed (the tick). */
export interface MarkedGlyph {
  char: string
  spaceBefore: boolean
  removedBefore: boolean
}

/** The glyphs of a run's spaced form. */
export function glyphsOf(spaced: string): string[] {
  return spaced.split(' ').filter((g) => g.length > 0)
}

/** Whether `written` holds a space before each glyph after the first, or `null` when `written`
 *  is not `spaced` with spaces moved (then nothing here can mark it). */
export function gapsOf(spaced: string, written: string): boolean[] | null {
  const glyphs = glyphsOf(spaced)
  const gaps: boolean[] = []
  let at = 0
  for (let i = 0; i < glyphs.length; i++) {
    let space = false
    while (written[at] === ' ') {
      space = true
      at++
    }
    if (!written.startsWith(glyphs[i], at)) return null
    if (i > 0) gaps.push(space)
    else if (space) return null
    at += glyphs[i].length
  }
  return at === written.length ? gaps : null
}

/** The run as written, with a tick before each glyph whose preceding space was removed. */
export function markRemovedSpaces(spaced: string, written: string): MarkedGlyph[] {
  const glyphs = glyphsOf(spaced)
  const gaps = gapsOf(spaced, written)
  if (gaps === null) {
    return [...written].map((char) => ({ char, spaceBefore: false, removedBefore: false }))
  }
  return glyphs.map((char, i) => ({
    char,
    spaceBefore: i > 0 && gaps[i - 1],
    removedBefore: i > 0 && !gaps[i - 1],
  }))
}

/** How many spaces the text no longer has, of those PyMuPDF read. */
export function removedCount(spaced: string, written: string): number {
  return markRemovedSpaces(spaced, written).filter((g) => g.removedBefore).length
}

/** A run written with these gaps: a space where `gaps[i]` is true, none where it is false. */
export function writeWithGaps(spaced: string, gaps: boolean[]): string {
  const glyphs = glyphsOf(spaced)
  return glyphs.map((g, i) => (i > 0 && gaps[i - 1] ? ' ' : '') + g).join('')
}

/** The rows of the block, split as it shows them: what the text holds closed now, and what is
 *  kept as extracted or spaced your way. A row with a `pending` change sits where that change
 *  will put it, so the list reads as the document will be after the next re-run. */
export function splitRows<T extends { state: string; pending: string | null }>(
  rows: T[],
): { closed: T[]; kept: T[] } {
  const closed: T[] = []
  const kept: T[] = []
  for (const r of rows) {
    const s = r.pending ?? r.state
    if (s === 'closed' || s === 'maths') closed.push(r)
    else kept.push(r)
  }
  return { closed, kept }
}

/** The caption under a row: spaces removed and where, in words. */
export function rowCaption(row: { spaced: string; written: string; pages: number[] }): string {
  const n = removedCount(row.spaced, row.written)
  const where =
    row.pages.length === 0
      ? 'no page recorded'
      : row.pages.length === 1
        ? `page ${row.pages[0]}`
        : `${row.pages.length} pages`
  const spaces = n === 0 ? 'as extracted' : `${n} ${n === 1 ? 'space' : 'spaces'} removed`
  return `${spaces} · ${where}`
}

/** What a removal says, in words: `entries` removed, of which `documents` are in this library. */
export function removedNote(entries: number, documents: number): string {
  const said = `Removed ${entries} ${entries === 1 ? 'entry' : 'entries'}.`
  if (documents === 0) return `${said} No document in this library is affected.`
  const whose = documents === 1 ? "that document's" : `the ${documents} documents'`
  return `${said} ${entries === 1 ? 'It takes' : 'They take'} effect when ${whose} text is re-run.`
}

/** The text a row shows: what the document holds for the run after the next re-run, which is
 *  what it holds now when nothing is pending. */
export function shownText(row: {
  spaced: string
  rule: string
  written: string
  state: string
  pending: string | null
  chosen: string | null
}): string {
  const target = row.pending ?? row.state
  if (target === 'kept' || target === 'maths') return row.spaced
  if (target === 'yours') return row.chosen ?? row.written
  if (target === 'closed') return row.rule
  return row.written
}
