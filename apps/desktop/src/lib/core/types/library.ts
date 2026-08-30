// TypeScript mirror of the desktop-API payloads (apps/api/models/library.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Library browser — documents and their chunk drill-in.
// Mirrors apps/api/models/library.py.

// Library browser (feature-library-browser.md, L1 — read-only). GET /api/library/documents lists
// ingested docs; GET /api/library/documents/{id} returns its chunks as parent blocks. Mirrors
// apps/api/models/library.py::Library*Payload. NULL metadata (title/authors/year, health) stays null.
export interface LibraryDocument {
  id: string
  filename: string
  title: string | null // effective: user override ?? auto-extracted (ADR-013)
  authors: string | null // effective
  year: number | null // effective
  customized: boolean // a user metadata override is in force
  format: string
  health: string | null
  chunk_count: number | null
  page_count: number | null
  folders: string[] // display names
  folder_ids: string[] // the key — a root folder name is not unique (ADR-025 F1, spec D2)
  tags: string[]
  keywords: string[]
  added_at: string | null // ISO 8601
  /**
   * Where the file actually is. The delete dialog must *name* it before offering to bin it
   * (ADR-046 §2) — "…from C:\Users\…\Zotero\storage\ABC123", not an abstract
   * "the file". Null only for a row whose source path was never recorded.
   */
  source_path: string | null
}
export interface LibraryChild {
  child_index: number
  text: string
  retrievable: boolean
}
export interface LibraryParent {
  parent_index: number
  parent_text: string
  children: LibraryChild[]
}
export interface LibraryDocumentChunks {
  id: string
  filename: string
  format: string
  title: string | null
  authors: string | null
  year: number | null
  chunk_count: number | null
  health: string | null
  parents: LibraryParent[]
  child_count: number
}

/** One figure in the per-document figure panel (Library L1b).
 *
 * `retrievable` is the field that matters: a figure enters retrieval only once it has a
 * description, so a panel that showed every row identically would display images the
 * assistant cannot actually see. `not_retrievable_reason` says why, already phrased for a
 * reader (the backend translates the audit enum). */
export interface LibraryFigure {
  id: string
  page: number
  kind: string | null
  caption: string | null
  description: string | null
  extraction_method: string | null
  has_image: boolean
  retrievable: boolean
  not_retrievable_reason: string | null
}

/** A document's figures, addressed separately from its text chunks. */
export interface LibraryDocumentFigures {
  id: string
  filename: string
  title: string | null
  figures: LibraryFigure[]
  total: number
  retrievable_count: number
  captioned_count: number
  missing_image_count: number
}
