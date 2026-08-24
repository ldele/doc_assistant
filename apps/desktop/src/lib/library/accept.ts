// Pure helpers for the add-documents accept surface (AD1).
//
// Everything here is a plain function over plain data so it runs under the repo's `node --test`
// runner with no harness — which matters, because the surrounding surface is Svelte components and
// a Tauri boundary, neither of which this project can test today.
//
// **This module deliberately does NOT decide which files are acceptable.** The verdicts (supported
// / unsupported / duplicate / no-text-layer) are computed server-side in AD2, from
// `extractors.is_supported` and `get_format_status` — the single source of truth for the format
// list. Filtering here would duplicate that list in a second language and let the two drift, which
// is the shape of the bug ADR-013 produced (one rule, two copies, one of them stale).

/** A path as the Tauri drag-drop event and the dialog plugin both report it: absolute, native. */
export type NativePath = string

/** Last path segment, for Windows (`\`) and POSIX (`/`) alike — Tauri reports native separators. */
export function basename(path: NativePath): string {
  const cleaned = path.replace(/[\\/]+$/, '')
  const cut = Math.max(cleaned.lastIndexOf('/'), cleaned.lastIndexOf('\\'))
  return cut === -1 ? cleaned : cleaned.slice(cut + 1)
}

/**
 * Drop the duplicates a single gesture can produce, keeping first-appearance order.
 *
 * Selecting a file *and* its parent folder in one dialog, or dropping overlapping selections, both
 * yield the same path twice. Order is kept because it is the order the user chose them in, and the
 * review sheet (AD2) shows them in that order before it sorts verdicts to the top.
 */
export function dedupePaths(paths: readonly NativePath[]): NativePath[] {
  const seen = new Set<NativePath>()
  const out: NativePath[] = []
  for (const p of paths) {
    if (p && !seen.has(p)) {
      seen.add(p)
      out.push(p)
    }
  }
  return out
}

/** One line naming what was accepted. Plural-correct, and it never invents a number. */
export function summarise(paths: readonly NativePath[]): string {
  const n = paths.length
  if (n === 0) return 'Nothing to add'
  if (n === 1) return `1 file ready: ${basename(paths[0])}`
  return `${n} files ready`
}

/**
 * The first few names, for a summary that shows *what* arrived rather than only how many.
 *
 * `limit` bounds the label, never the batch — the batch is uncapped by decision (grill branch 7),
 * and a label that silently truncated would read as "this is all of it", which is the failure this
 * project keeps writing down. The remainder is reported by `remainderLabel`.
 */
export function previewNames(paths: readonly NativePath[], limit = 3): string[] {
  return paths.slice(0, Math.max(0, limit)).map(basename)
}

/** `""` when nothing was hidden, so the caller never renders "and 0 more". */
export function remainderLabel(paths: readonly NativePath[], limit = 3): string {
  const hidden = paths.length - Math.max(0, limit)
  if (hidden <= 0) return ''
  return hidden === 1 ? 'and 1 more' : `and ${hidden} more`
}
