// Pure helpers for the delete dialog (ADR-046 §2) — no runes, no fetch, so `node --test` reaches
// them. The component next door renders these strings; it decides nothing.
//
// **Why this is not just copy-writing.** ADR-046 makes naming the destination part of the
// decision: "the accepted risk of a per-delete choice is a mis-click; showing the destination is
// what makes the click informed". A dialog that said "also delete the file" without saying *which*
// file would be the version of this feature the ADR rejected.

import type { LibraryDocument } from '../core/types/library'

/**
 * What to call the document in the dialog heading — the effective title, else the filename.
 *
 * Never blank: a confirm whose subject line is empty is a confirm the user cannot check.
 */
export function targetName(doc: Pick<LibraryDocument, 'title' | 'filename'>): string {
  const title = doc.title?.trim()
  return title && title.length > 0 ? title : doc.filename
}

/**
 * A long path shortened from the middle, keeping both ends.
 *
 * The ends are what identify a file — the drive/user at the front, the filename at the back — so a
 * plain truncation is the one form that must not be used here: it hides exactly the half that says
 * *which* file is about to be binned. The full value still goes in a `title`.
 */
export function shortenPath(path: string, max = 56): string {
  if (max <= 1) return '…'
  if (path.length <= max) return path
  // -1 for the ellipsis itself; the tail keeps the filename, so give it the larger half.
  const budget = max - 1
  const tail = Math.ceil(budget / 2)
  const head = budget - tail
  return `${path.slice(0, head)}…${path.slice(path.length - tail)}`
}

/**
 * The sentence under "Also delete the file", or `null` when there is no path to name.
 *
 * `null` is a real case — a row whose `source_path` was never recorded — and the caller must then
 * *not* offer the destructive option, rather than offer it with a vague label. Refusing to name
 * the target is refusing the guarantee the ADR asked for.
 */
export function deleteFileDetail(doc: Pick<LibraryDocument, 'source_path'>): string | null {
  const path = doc.source_path?.trim()
  if (!path) return null
  return `Moves ${shortenPath(path)} to your Recycle Bin.`
}

/**
 * The sentence under "Remove from library" — what leaves, and what does not.
 *
 * States the chunk count only when there is one, because "removing its 0 chunks" reads as a
 * malfunction on a document that never indexed.
 */
export function removeOnlyDetail(doc: Pick<LibraryDocument, 'chunk_count'>): string {
  const n = doc.chunk_count ?? 0
  // The verb agrees too: "Its 1 chunk leave the search index" is the kind of sentence that makes a
  // confirm dialog look machine-generated at exactly the moment it is asking for trust.
  const chunks = n > 0 ? ` Its ${n.toLocaleString()} ${n === 1 ? 'chunk leaves' : 'chunks leave'} the search index.` : ''
  return `The file stays where it is.${chunks}`
}
