// What an add accepts, as words — the formats line and the size limit the uploader states up front.
//
// Pure and tested (`formats.test.ts`). The numbers come from `GET /api/documents/accepts`, never
// from this file: the limits are env-configurable on the backend, and a stated limit the backend
// does not enforce would be wrong in the one place users read before they add anything.

import type { Accepts } from '../core/types/documents'

/** `['.pdf', '.html', '.htm', '.md']` → `'PDF · HTML · MD'` — `.htm` is HTML under another name. */
export function formatLine(extensions: readonly string[]): string {
  const labels: string[] = []
  for (const ext of extensions) {
    const label = ext.replace(/^\./, '').toUpperCase()
    if (label === 'HTM' && extensions.includes('.html')) continue
    if (!labels.includes(label)) labels.push(label)
  }
  return labels.join(' · ')
}

/** Bytes as a person reads them, in the binary units the file system shows: 1073741824 → `1 GB`. */
export function sizeLabel(bytes: number): string {
  const units: [string, number][] = [
    ['GB', 1024 ** 3],
    ['MB', 1024 ** 2],
    ['KB', 1024],
  ]
  for (const [unit, scale] of units) {
    if (bytes >= scale) {
      const n = bytes / scale
      const shown = Number.isInteger(n) ? String(n) : n.toFixed(1)
      return `${shown} ${unit}`
    }
  }
  return `${bytes} bytes`
}

/** The limit mid-sentence: `up to 1 GB per file`. Never lower-case a whole line — the unit is `GB`. */
export function limitPhrase(accepts: Pick<Accepts, 'max_file_bytes'>): string {
  return `up to ${sizeLabel(accepts.max_file_bytes)} per file`
}

/** The limit as its own line: `Up to 1 GB per file`. */
export function limitLine(accepts: Pick<Accepts, 'max_file_bytes'>): string {
  const phrase = limitPhrase(accepts)
  return phrase.charAt(0).toUpperCase() + phrase.slice(1)
}
