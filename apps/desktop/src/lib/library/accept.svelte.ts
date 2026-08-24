// Accept-surface state (AD1) — what the user has handed the app but not yet added.
//
// A leaf rune module: it imports the Tauri boundary and the pure helpers, and nothing else from
// the app. That keeps the drop target usable from the Library pane, the empty state (AD4) and the
// toolbar button without threading props through three components.
//
// **This is a staging area, not a mutation.** Nothing here copies, registers or indexes anything;
// it holds paths until AD2's review sheet turns them into verdicts and AD3 applies them. That
// boundary is spec constraint 2 ("nothing before the review sheet is shown and confirmed"), and
// keeping the state module incapable of writing is how it stays true by construction.

import { canPickFiles, canReceiveDrops, isTauri, onDragHover, onFilesDropped, pickPaths } from '../core/tauri'
import { dedupePaths, type NativePath } from './accept'

export const accept = $state({
  /** Paths the user has handed over, awaiting review. Empty is the resting state. */
  pending: [] as NativePath[],

  /** True while a native drag is over the window — drives the drop overlay. */
  dragging: false,

  /** True while the OS picker is open, so the button can't be double-fired. */
  picking: false,
})

/** Whether this build can accept documents at all. False in a plain browser. */
export function canAccept(): boolean {
  return isTauri() && (canReceiveDrops() || canPickFiles())
}

/** Why the surface is unavailable, in words a user can act on. `null` when it is available. */
export function unavailableReason(): string | null {
  if (!isTauri()) return 'Adding documents works in the desktop app.'
  if (!canReceiveDrops() && !canPickFiles()) return 'The file picker is not available in this build.'
  return null
}

/** Stage paths for review. Merges with anything already pending and de-duplicates the result. */
export function stagePaths(paths: readonly NativePath[]): void {
  if (paths.length === 0) return
  accept.pending = dedupePaths([...accept.pending, ...paths])
}

export function clearPending(): void {
  accept.pending = []
}

/** Open the OS picker. No-op outside Tauri; `pickPaths` already resolves to `null` there. */
export async function pickDocuments(opts: { directory?: boolean } = {}): Promise<void> {
  if (accept.picking) return
  accept.picking = true
  try {
    const chosen = await pickPaths(opts)
    if (chosen) stagePaths(chosen)
  } finally {
    accept.picking = false
  }
}

/**
 * Wire the window-level drag-drop listeners. Call once from the app root; the returned function
 * unsubscribes. Safe to call in a browser — both subscriptions become no-ops.
 */
export function watchDrops(): () => void {
  const stopDrop = onFilesDropped((paths) => {
    accept.dragging = false
    stagePaths(paths)
  })
  const stopHover = onDragHover((over) => {
    accept.dragging = over
  })
  return () => {
    stopDrop()
    stopHover()
  }
}
