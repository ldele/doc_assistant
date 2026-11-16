// TypeScript mirror of the desktop-API payloads (apps/api/models/text_checks.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// A document's text check (A2b): GET /api/library/documents/{id}/text-check.

/** One paragraph of the document that addresses an AI. `page` is null for a format without pages. */
export interface TextFinding {
  page: number | null
  labels: string[]
  excerpt: string
}

/** `stale`: the text or the patterns changed since the check; `not_checked`: never checked. */
export type TextCheckState = 'current' | 'stale' | 'not_checked'

export interface DocumentTextCheck {
  document_id: string
  state: TextCheckState
  checked_at: string | null
  /** Every marked paragraph; `findings` lists the first of them. */
  n_marked: number
  findings: TextFinding[]
}
