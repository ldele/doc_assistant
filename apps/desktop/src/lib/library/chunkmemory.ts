// Which documents the reader had the Chunks block open on, for THIS app session.
//
// **Ids only — never the chunk payload.** Restoring the *state* costs one re-fetch; caching the
// text would mean holding up to 1.85 MB per visited document, which is the exact cost the
// collapsed-by-default block exists to avoid. A set of ids is bounded by the number of documents
// the reader opened chunks on, so it needs no eviction.
//
// **Session-scoped on purpose — not localStorage.** Remembering "open" across launches would
// restore the eager render on a fresh start, where the reader has not asked for anything yet.
// Within a session it is the opposite: they *did* ask, so coming back via the top ← → arrows
// should land where they left off.

const openOn = new Set<string>()

/** Record whether this document's Chunks block is open. Called on every toggle. */
export function rememberChunksOpen(docId: string, open: boolean): void {
  if (open) openOn.add(docId)
  else openOn.delete(docId)
}

/** Did the reader leave this document's Chunks block open earlier in the session? */
export function wereChunksOpen(docId: string | null): boolean {
  return docId !== null && openOn.has(docId)
}

/** Drop the whole memory. Test seam; nothing in the app calls it. */
export function forgetChunkState(): void {
  openOn.clear()
}
