// TypeScript mirror of the desktop-API payloads (apps/api/models/spacing.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// A document's letter-spacing (ADR-056): GET /api/library/documents/{id}/spacing, and the
// ignore list behind it, GET /api/library/spacing.

/** What extraction wrote for a run: the rule closed it, the ignore list kept it, the user's
 *  spacing, maths the rule left alone, or `unknown` for an entry that names no recorded run. */
export type SpacingState = 'closed' | 'kept' | 'yours' | 'maths' | 'unknown'

/** One letter-spaced run of a document, however many pages repeat it.
 *
 * `written` is the text now in the document; `rule` is how the rule closes the run; `pending` is
 * what the ignore list asks for when that differs from `state`, which applies when the text is
 * re-run. `crop` is the box (x0, y0, x1, y1, in points) to render on `pages[0]`. */
export interface SpacingRow {
  spaced: string
  rule: string
  written: string
  kind: string
  label: string
  state: SpacingState
  pending: SpacingState | null
  chosen: string | null
  pages: number[]
  crop: number[] | null
}

/** A document's letter-spacing block. `pageable` false: not a PDF, nothing to show. */
export interface DocumentSpacing {
  document_id: string
  filename: string
  pageable: boolean
  keep_all: boolean
  rows: SpacingRow[]
}

/** One document's entries in the ignore list. `document_id` is null when no library document
 *  matches the file. */
export interface IgnoreListEntry {
  file: string
  sha256: string
  size: number
  document_id: string | null
  keep_all: boolean
  kept: string[]
  spacing: Record<string, string>
}

export interface IgnoreList {
  entries: IgnoreListEntry[]
}

/** What a removal took out; `previous` is the whole list as it was, for Undo to send back. */
export interface IgnoreListRemoved {
  removed: number
  documents: string[]
  previous: Record<string, unknown>
}
