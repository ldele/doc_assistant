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
