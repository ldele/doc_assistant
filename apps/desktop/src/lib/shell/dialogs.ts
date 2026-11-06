// The open dialogs, oldest first (shell/Dialog.svelte; ADR-063 choice 8). One list for the whole
// app, so the rules that span dialogs hold without each one knowing the others:
//
// - *Esc* closes the dialog in front and only it. A confirm asked from inside a dialog answers the
//   question; the dialog under it stays. Each dialog used to guard its own Escape against the
//   questions it might open, and a guard forgotten closed both.
// - A dialog opened from another sits above it, at the next layer, however the two are nested.
// - When a dialog closes, focus goes back to what opened it, unless something else has it now
//   (the WAI-ARIA dialog pattern).
//
// Pure apart from the focus helpers at the end, so `node --test` reaches the list.

/** The z-index of the backdrop of the first dialog; its frame sits one above. */
export const BASE_LAYER = 40

interface Open {
  id: number
  frame: HTMLElement | null
}

const open: Open[] = []
let made = 0

/** Adds a dialog in front of the others and returns its id. */
export function openDialog(): number {
  made += 1
  open.push({ id: made, frame: null })
  return made
}

/** Takes a dialog off the list, wherever it is (a parent can close before its question). */
export function closeDialog(id: number): void {
  const i = open.findIndex((d) => d.id === id)
  if (i >= 0) open.splice(i, 1)
}

/** Records the element a dialog draws, so focus can tell which dialog holds it. */
export function setFrame(id: number, frame: HTMLElement | null): void {
  const d = open.find((x) => x.id === id)
  if (d) d.frame = frame
}

/** True when this dialog is the one in front, the one *Esc* closes. */
export function isFront(id: number): boolean {
  return open.length > 0 && open[open.length - 1].id === id
}

/** The z-index of a dialog's backdrop: two layers per dialog under it. */
export function layerOf(id: number): number {
  const i = open.findIndex((d) => d.id === id)
  return BASE_LAYER + 2 * Math.max(0, i)
}

/** How many dialogs are open. */
export function openCount(): number {
  return open.length
}

/** For tests: forget every dialog. */
export function resetDialogs(): void {
  open.length = 0
  made = 0
}

/** The element of the dialog in front, if it has drawn one. */
export function frontFrame(): HTMLElement | null {
  return open.length > 0 ? open[open.length - 1].frame : null
}

/**
 * Whether focus may go back to `opener` when a dialog closes: only when no dialog is left, or the
 * opener sits inside the one now in front (a confirm asked from a dialog hands focus back to the
 * button that asked). A dialog that opened another and then closed itself must not pull focus out
 * of the new one.
 */
export function mayRestoreTo(opener: { isConnected: boolean } | null, frontContains: ((o: unknown) => boolean) | null): boolean {
  if (!opener || !opener.isConnected) return false
  return frontContains === null || frontContains(opener)
}

/** What *Tab* may land on inside a dialog. */
export const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * Where *Tab* should go to stay inside a modal dialog (`aria-modal` tells assistive technology the
 * rest is inert, so focus must not walk out into it): the first control after the last, the last
 * before the first. Focus on the dialog's own frame (where it starts when nothing inside asks for
 * it) counts as the start. Null when the browser's own move stays inside.
 */
export function wrapTab<T>(
  focusable: readonly T[],
  active: unknown,
  backwards: boolean,
  frame: unknown,
): T | null {
  if (focusable.length === 0) return null
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  if (backwards && (active === first || active === frame)) return last
  if (!backwards && active === last) return first
  return null
}
