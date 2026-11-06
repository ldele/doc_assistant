// Library folders (ADR-025 F1, docs/specs/feature-corpus-folders.md; ADR-064 choice 5): the folder
// list, its writes, and the *Manage folders* dialog's state.
//
// A `.svelte.ts` rune module. The rail renders this list rather than deriving groups from the
// documents, so a folder with no members is still visible and therefore fillable (spec D3). Every
// write is write-then-refetch: the server is the authority on names and counts, so the list is
// never patched by hand. A membership change also refetches the document list, whose
// `folder_ids` drive the grid, which is why this module reads `documents.svelte.ts` (and not the
// other way round).
//
// Three doors open the dialog through `openManageFolders*`: *Manage…* on the Library rail, a
// tile's *Add to folder…*, and *Add to folder…* on the select bar when there is no folder yet.
// Opening it closes the mobile drawer (`shell.sidebarOpen`), as every navigation does.
//
// What stays in App.svelte: deleting a folder that is the Library's open collection also returns
// the Library to All documents (navigation), and the chat-scope guard reads this list to write
// chat state. Both are glue.

import {
  addDocumentsToFolder,
  createFolder,
  deleteFolder,
  listFolders,
  removeDocumentFromFolder,
  renameFolder,
} from '../core/api'
import { loadSnapshot, saveSnapshot, windowStorage } from '../core/lastSession'
import type { LibraryFolder } from '../core/types'
import { shell } from '../shell/shell.svelte'
import { exitSelectMode, library, refreshDocuments } from './documents.svelte'
import { docLabel } from './library'

export const folders = $state({
  list: [] as LibraryFolder[],
  /** *Manage folders*: whether it is open, the folder it shows, and the document query it
   *  opens with (a tile's *Add to folder…* opens it on that one document). */
  manageOpen: false,
  manageId: null as string | null,
  manageQuery: '',
  /** A write's 400 (a blank or taken name), shown in Manage folders without blocking anything
   *  else. */
  error: null as string | null,
})

// Folders are a sidecar read, same inform-don't-block rule as documents.
export async function refreshFolders(): Promise<void> {
  try {
    folders.list = await listFolders()
    saveSnapshot(windowStorage(), 'folders', $state.snapshot(folders.list))
  } catch {
    // keep the prior list
  }
}

/** Show the folder list kept from the last session until the engine answers (`lastSession.ts`).
 *  Called once, during App's init, so the copy is on screen at the first paint. */
export function restoreFolders(): boolean {
  const last = loadSnapshot<LibraryFolder[]>(windowStorage(), 'folders')
  if (!last) return false
  folders.list = last.data
  return true
}

async function folderWrite(op: () => Promise<unknown>, alsoDocuments = false): Promise<void> {
  folders.error = null
  try {
    await op()
  } catch (e) {
    folders.error = e instanceof Error ? e.message : String(e)
    return
  }
  await refreshFolders()
  if (alsoDocuments) await refreshDocuments()
}

export function createLibraryFolder(name: string): void {
  void folderWrite(() => createFolder(name))
}

export function renameLibraryFolder(folderId: string, name: string): void {
  void folderWrite(() => renameFolder(folderId, name), true)
}

/** The folder half of a delete; App's `deleteLibraryFolder` first moves the Library off it. */
export function deleteLibraryFolder(folderId: string): void {
  // A deleted folder can't stay the Manage selection.
  if (folders.manageId === folderId) folders.manageId = null
  void folderWrite(() => deleteFolder(folderId), true)
}

export function addDocsToFolder(folderId: string, documentIds: string[]): void {
  void folderWrite(() => addDocumentsToFolder(folderId, documentIds), true)
}

export function removeDocFromFolder(folderId: string, documentId: string): void {
  void folderWrite(() => removeDocumentFromFolder(folderId, documentId), true)
}

/** The select bar's *Add to folder…* → a folder: adds the selection, and ends select mode. */
export function addSelectionToFolder(folderId: string): void {
  if (library.selected.length > 0) addDocsToFolder(folderId, library.selected)
  exitSelectMode()
}

export function openManageFolders(): void {
  folders.manageQuery = ''
  folders.manageOpen = true
  folders.error = null
  shell.sidebarOpen = false
  void refreshFolders()
}

/** A tile's *Add to folder…*: the same view, opened pre-filtered to that one document so the
 *  picker shows it alone once a folder is chosen. */
export function openManageFoldersForDoc(id: string): void {
  const doc = library.list.find((d) => d.id === id)
  openManageFolders()
  folders.manageQuery = doc ? docLabel(doc) : ''
}

export function closeManageFolders(): void {
  folders.manageOpen = false
}
