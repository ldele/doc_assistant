// What the window shows before its engine answers: the library, folders, chat list and concept
// graph as they were the last time each loaded. The packaged backend takes tens of seconds to
// start, and the window used to sit empty until it answered (the 0.7.0 walkthrough). Each list is
// saved here when it loads for real, read back at the next launch, and replaced as soon as the
// engine answers; the view says it is a copy. Pure and storage-agnostic, tested under `npm test`.
//
// Kept in the window's own storage (`localStorage`, per user, on this machine), beside its other
// preferences — the same trust zone as the data folder these lists come from.

export const SNAPSHOT_VERSION = 1

/** A copy larger than this is not kept: it would crowd the window's storage (a few MB), and a
 *  library big enough to reach it (thousands of documents) simply starts as it did before. */
export const MAX_SNAPSHOT_CHARS = 2_000_000

export type SnapshotKey = 'documents' | 'folders' | 'conversations' | 'graph'

export interface Snapshot<T> {
  data: T
  savedAt: string
}

/** The two storage calls this module needs, so a test can pass a plain object. */
export interface SnapshotStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export function storageKey(key: SnapshotKey): string {
  return `provenote.lastSession.${key}`
}

/** The window's storage, or `null` where it cannot be reached (a locked-down profile). */
export function windowStorage(): SnapshotStorage | null {
  try {
    return globalThis.localStorage ?? null
  } catch {
    return null
  }
}

/** Keep a copy of `data`. Never throws: a full or unavailable store just means no copy, and a
 *  copy that outgrew the cap is removed, so an old, smaller one cannot outlive it. */
export function saveSnapshot<T>(
  storage: SnapshotStorage | null,
  key: SnapshotKey,
  data: T,
  now: Date = new Date(),
): void {
  if (!storage) return
  try {
    const text = JSON.stringify({ v: SNAPSHOT_VERSION, savedAt: now.toISOString(), data })
    if (text.length > MAX_SNAPSHOT_CHARS) storage.removeItem(storageKey(key))
    else storage.setItem(storageKey(key), text)
  } catch {
    // quota, a refused write, or data that does not serialise: no copy this time
  }
}

/** The copy kept for `key`, or `null` when there is none, it is from another version of the
 *  app's format, or it does not parse. */
export function loadSnapshot<T>(storage: SnapshotStorage | null, key: SnapshotKey): Snapshot<T> | null {
  if (!storage) return null
  try {
    const raw = storage.getItem(storageKey(key))
    if (!raw) return null
    const parsed = JSON.parse(raw) as { v?: unknown; savedAt?: unknown; data?: unknown }
    if (parsed.v !== SNAPSHOT_VERSION || typeof parsed.savedAt !== 'string' || parsed.data == null) return null
    return { data: parsed.data as T, savedAt: parsed.savedAt }
  } catch {
    return null
  }
}
