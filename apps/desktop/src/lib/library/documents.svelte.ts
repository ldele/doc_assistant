// Library documents (feature-library-browser.md; ADR-064 choice 5): the document list, the two
// dialogs a tile's ⋯ menu opens on one document (*Edit metadata*, *Delete this document?*), its
// *Reveal in file explorer*, and the Library's select mode.
//
// A `.svelte.ts` rune module: ONE `$state` object, since an imported binding cannot be reassigned
// across modules. Every read is a sidecar and a failure keeps the prior list (inform, don't
// block); every write is write-then-refetch, so the server stays the authority on the list.
//
// What stays in App.svelte, and why: confirming a delete also leaves the document's page when it
// is the one open (navigation) and re-reads the engine's health for the status bar, so it is glue.
// The grid's order (`visibleDocs`, which a Shift-click ranges over) is the Library's facet
// pipeline, also App's; the door passes it in.

import {
  listLibraryDocuments,
  resetDocumentMeta,
  revealDocument,
  updateDocumentMeta,
} from '../core/api'
import { loadSnapshot, saveSnapshot, windowStorage } from '../core/lastSession'
import type { LibraryDocument } from '../core/types'
import { pick } from '../shell/selection'

export const library = $state({
  list: [] as LibraryDocument[],
  /** Has a fetch ever come back? Read by the Library to tell "empty" from "not yet known", and
   *  by every lazy load as its latch. `$state`, not a plain flag: a non-reactive latch read by a
   *  component leaves the loading line up forever (svelte-check catches exactly this). */
  loaded: false,

  /** *Edit metadata* (ADR-013): the document it edits; `null` while closed. */
  editingId: null as string | null,
  /** *Delete this document?* (ADR-014, ADR-046): the document it asks about, and whether the
   *  delete is in flight. */
  deletingId: null as string | null,
  deleteBusy: false,

  /** Select mode (batch add-to-folder, ROADMAP 80). Bulk delete is deliberately not here —
   *  deferred pending its own ADR (ui-checklist). */
  selectMode: false,
  selected: [] as string[],
  /** The document the next Shift-click ranges from. */
  anchor: null as string | null,
  /** The select bar's *Add to folder…* menu. */
  addMenuOpen: false,
})

export async function refreshDocuments(): Promise<void> {
  try {
    library.list = await listLibraryDocuments()
    library.loaded = true
    saveSnapshot(windowStorage(), 'documents', $state.snapshot(library.list))
  } catch {
    // keep the prior list
  }
}

/** Show the document list kept from the last session until the engine answers
 *  (`lastSession.ts`). `loaded` stays false, so the first real fetch still runs and replaces it.
 *  Called once, during App's init, so the copy is on screen at the first paint. */
export function restoreDocuments(): boolean {
  const last = loadSnapshot<LibraryDocument[]>(windowStorage(), 'documents')
  if (!last) return false
  library.list = last.data
  return true
}

// --- Edit metadata (ADR-013, the first browse-time write path). Save and Reset write, then
// re-fetch the list so the tile shows the new effective values.

export function openEditMetadata(id: string): void {
  library.editingId = id
}

export function closeEditMetadata(): void {
  library.editingId = null
}

export async function saveMetadata(patch: {
  title: string
  authors: string
  year: number | null
}): Promise<void> {
  if (library.editingId === null) return
  try {
    await updateDocumentMeta(library.editingId, patch)
    await refreshDocuments()
  } catch {
    // inform-don't-block: a write failure leaves the prior values in place
  }
  library.editingId = null
}

export async function resetMetadata(): Promise<void> {
  if (library.editingId === null) return
  try {
    await resetDocumentMeta(library.editingId)
    await refreshDocuments()
  } catch {
    // keep the prior list on failure
  }
  library.editingId = null
}

// --- Delete this document? The confirm is App's (see the header).

export function openDeleteDocument(id: string): void {
  library.deletingId = id
}

export function closeDeleteDocument(): void {
  library.deletingId = null
}

/** *Reveal in file explorer*: the server opens the OS file manager. */
export async function revealDoc(id: string): Promise<void> {
  try {
    await revealDocument(id)
  } catch {
    // the source file may have moved since ingest — surface nothing, never crash the UI
  }
}

// --- Select mode

/** A click that selects, in the order the grid shows (ROADMAP 80; the same `pick` as the chat
 *  list). A Shift- or Ctrl/Cmd-click outside select mode starts it with that document. */
export function pickDocument(order: string[], id: string, shift: boolean): void {
  if (!library.selectMode) enterSelectMode()
  const r = pick(order, library.selected, library.anchor, id, shift)
  library.selected = r.selected
  library.anchor = r.anchor
}

export function enterSelectMode(): void {
  library.selectMode = true
  library.selected = []
  library.anchor = null
}

export function exitSelectMode(): void {
  library.selectMode = false
  library.selected = []
  library.anchor = null
  library.addMenuOpen = false
}
