// TypeScript mirror of the desktop-API payloads (apps/api/models/references.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// A document's reference list. GET /api/library/documents/{id}/references.
// Distinct from `connections` on purpose: that bundle is the document's *neighbourhood*
// (semantic neighbours + who cites it), this one is the paper's own bibliography.

/** One extracted reference.
 *
 * `document_id` set ⇒ the reference resolved to a document in the library, and the panel
 * renders it as a link. `title`/`authors`/`year`/`doi` are regex-extraction output parsed
 * from `raw_text` — `library_title` is the owned document's own title and is the label to
 * trust when the two disagree. Every field can be null: an unparseable reference still
 * belongs in the list (dropping it would misrepresent the bibliography), rendered from
 * `raw_text`. */
export interface DocumentReference {
  raw_text: string | null
  title: string | null
  authors: string | null
  year: number | null
  doi: string | null
  document_id: string | null
  filename: string | null
  library_title: string | null
}

/** A document's reference list + the counts that keep a capped list honest.
 *
 * `in_library` is counted over ALL references, not just the shown ones. */
export interface DocReferences {
  references: DocumentReference[]
  total: number
  in_library: number
  shown: number
}
