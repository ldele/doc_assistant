// Thin `library` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/library.py and apps/api/models/library.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type {
  ChunkContext,
  DocReferences,
  LibraryDocumentFigures,
  LibraryDocument,
  LibraryDocumentChunks,
  ReingestOptions,
  ReingestStatus,
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
/**
 * Remove a document from the library; bin its source file only if `deleteFile` is true.
 *
 * Defaults to the safe branch, matching the server (ADR-046 §2 amended ADR-014, which used to bin
 * the source unconditionally). The caller is asserting it showed the user the real path before
 * passing true — `LibraryDocument.source_path` is there so it can.
 */
export async function deleteDocument(docId: string, deleteFile = false): Promise<DeleteResult> {
  const q = deleteFile ? '?delete_file=true' : ''
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}${q}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(`delete document failed: ${r.status}`)
  return (await r.json()) as DeleteResult
}

// Folders (ADR-025 F1, docs/specs/feature-corpus-folders.md). Manual Library organisation over
// the previously dormant Folder schema. Since F2 a folder can also scope a chat turn's
// retrieval — see `streamChat`'s `scopeFolderId`.

/** One document's figures (Library L1b) — a pure sidecar read, separate from its text chunks.
 *  404 for an unknown document; a document with no figures returns an empty list. */
export async function getDocumentFigures(docId: string): Promise<LibraryDocumentFigures> {
  const r = await fetch(
    `${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/figures`,
  )
  if (!r.ok) throw new Error(`document figures failed: ${r.status}`)
  return (await r.json()) as LibraryDocumentFigures
}

/** One document's reference list — the paper's own bibliography, including the entries that
 *  resolved to nothing. 404 for an unknown document; never-extracted references come back as
 *  an empty list. */
export async function getDocumentReferences(docId: string): Promise<DocReferences> {
  const r = await fetch(
    `${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/references`,
  )
  if (!r.ok) throw new Error(`document references failed: ${r.status}`)
  return (await r.json()) as DocReferences
}

/** What a re-run can do, and what it declines to do (ADR-048). */
export async function getReingestOptions(): Promise<ReingestOptions> {
  const r = await fetch(`${API_BASE}/api/library/reingest/options`)
  if (!r.ok) throw new Error(await errorDetail(r, 'load re-run options'))
  return (await r.json()) as ReingestOptions
}

/** Start a re-run. One document is ROADMAP 20; a selection is 21 — same call either way.
 *  202 + poll, like every other background job here. */
export async function startReingest(documentIds: string[], parts: string[]): Promise<void> {
  const r = await fetch(`${API_BASE}/api/library/documents/reingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_ids: documentIds, parts }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'start the re-run'))
}

export async function getReingestStatus(): Promise<ReingestStatus> {
  const r = await fetch(`${API_BASE}/api/library/reingest/status`)
  if (!r.ok) throw new Error(await errorDetail(r, 'read the re-run status'))
  return (await r.json()) as ReingestStatus
}

/** Where a cited chunk sits in its source. `null` when it cannot be placed (a 404) — an
 *  unresolved span or a cache that is gone. Never an approximation. */
export async function getChunkContext(chunkKey: string, window = 700): Promise<ChunkContext | null> {
  const r = await fetch(
    `${API_BASE}/api/library/chunk-context?key=${encodeURIComponent(chunkKey)}&window=${window}`,
  )
  if (r.status === 404) return null
  if (!r.ok) throw new Error(await errorDetail(r, 'locate this passage'))
  return (await r.json()) as ChunkContext
}
