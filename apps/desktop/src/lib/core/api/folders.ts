// Thin `folders` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/folders.py and apps/api/models/folders.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE, errorDetail } from './_base'
import type {
  LibraryFolder,
} from '../types'

export async function listFolders(): Promise<LibraryFolder[]> {
  const r = await fetch(`${API_BASE}/api/library/folders`)
  if (!r.ok) throw new Error(await errorDetail(r, 'folders'))
  return (await r.json()) as LibraryFolder[]
}
/** Create a folder. Idempotent on the case-folded name — the existing folder comes back. */
export async function createFolder(name: string): Promise<LibraryFolder> {
  const r = await fetch(`${API_BASE}/api/library/folders`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'create folder'))
  return (await r.json()) as LibraryFolder
}
export async function renameFolder(folderId: string, name: string): Promise<LibraryFolder> {
  const r = await fetch(`${API_BASE}/api/library/folders/${encodeURIComponent(folderId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'rename folder'))
  return (await r.json()) as LibraryFolder
}
/** Delete the folder only — the documents in it are never touched (spec D6). */
export async function deleteFolder(folderId: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/library/folders/${encodeURIComponent(folderId)}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'delete folder'))
}
/** Bulk-add documents to a folder. Idempotent; unknown ids are skipped server-side. */
export async function addDocumentsToFolder(
  folderId: string,
  documentIds: string[],
): Promise<LibraryFolder> {
  const r = await fetch(
    `${API_BASE}/api/library/folders/${encodeURIComponent(folderId)}/documents`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document_ids: documentIds }),
    },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'add documents to folder'))
  return (await r.json()) as LibraryFolder
}
export async function removeDocumentFromFolder(
  folderId: string,
  documentId: string,
): Promise<LibraryFolder> {
  const r = await fetch(
    `${API_BASE}/api/library/folders/${encodeURIComponent(folderId)}/documents/${encodeURIComponent(documentId)}`,
    { method: 'DELETE' },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'remove document from folder'))
  return (await r.json()) as LibraryFolder
}

// Tag families (feature-tag-families.md, PR-1). A family collapses near-duplicate keywords
// (`llm`/`llms`) into one filterable entry — errors carry the backend's `detail` (400/404).
