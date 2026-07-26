// Thin `library` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/library.py and apps/api/models/library.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE } from './_base'
import type {
  LibraryDocument,
  LibraryDocumentChunks,
} from '../types'

/** List ingested documents for the Library browser (feature-library-browser.md, read-only). */
export async function listLibraryDocuments(): Promise<LibraryDocument[]> {
  const r = await fetch(`${API_BASE}/api/library/documents`)
  if (!r.ok) throw new Error(`library documents failed: ${r.status}`)
  return (await r.json()) as LibraryDocument[]
}
/** One document's chunks grouped into parent blocks (each expandable to its children). 404 unknown. */
export async function getLibraryDocument(docId: string): Promise<LibraryDocumentChunks> {
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}`)
  if (!r.ok) throw new Error(`library document failed: ${r.status}`)
  return (await r.json()) as LibraryDocumentChunks
}
/** Set a document's user metadata overrides (title/authors/year). The editor sends the whole form;
 *  each effective value blank/equal-to-default clears that field's override (ADR-013). */
export async function updateDocumentMeta(
  docId: string,
  patch: { title?: string; authors?: string; year?: number | null },
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!r.ok) throw new Error(`update document metadata failed: ${r.status}`)
}
/** Reset a document's metadata to the auto-extracted defaults (drop the override). */
export async function resetDocumentMeta(docId: string): Promise<void> {
  const r = await fetch(
    `${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/reset-metadata`,
    { method: 'POST' },
  )
  if (!r.ok) throw new Error(`reset document metadata failed: ${r.status}`)
}
/** Reveal a document's source file in the OS file manager (local desktop action). 404 if the file
 *  can't be located (moved/deleted since ingest). */
export async function revealDocument(docId: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/reveal`, {
    method: 'POST',
  })
  if (!r.ok) throw new Error(`reveal document failed: ${r.status}`)
}
export interface DeleteResult {
  filename: string
  trashed_file: boolean
  chunks_removed: number
}
/** Safe-delete a document: source file → Recycle Bin, then drop its DB row + index chunks
 *  (ADR-014). Throws on failure (e.g. 409 when the file couldn't be moved to the Recycle Bin). */
export async function deleteDocument(docId: string): Promise<DeleteResult> {
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(`delete document failed: ${r.status}`)
  return (await r.json()) as DeleteResult
}

// Folders (ADR-025 F1, docs/specs/feature-corpus-folders.md). Manual Library organisation over
// the previously dormant Folder schema. Since F2 a folder can also scope a chat turn's
// retrieval — see `streamChat`'s `scopeFolderId`.
