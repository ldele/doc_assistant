// TypeScript mirror of the desktop-API payloads (apps/api/models/folders.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Library folders (ADR-025 F1). Mirrors apps/api/models/folders.py.

// A Library folder (ADR-025 F1, docs/specs/feature-corpus-folders.md). Organises the Library
// and, since F2, a chat turn's retrieval scope. `parent_id` is always null in v1
// (folders are flat, spec D1); `doc_count` excludes archived documents, matching the grid.
// Mirrors apps/api/models/folders.py::LibraryFolderPayload.
export interface LibraryFolder {
  id: string
  name: string
  description: string | null
  parent_id: string | null
  doc_count: number
}
