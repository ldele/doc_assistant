// Pure presentation helpers for the Settings "Corpus" panel (ADR-037). Plain `.ts`, not
// `.svelte.ts`, so `npm test` can run it (apps/desktop/CLAUDE.md — the extension is the marker).
//
// The backend sends facts; this file turns them into the words a user reads. That split is
// deliberate: the honest sentence about memory differs per keyword-index arm, and a sentence is
// not something the API should be shipping.

import type { CorpusFacts } from '../core/types/settings'

/** Bytes as a short human string. Binary units, because that is what a file manager shows. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  // One decimal below 10 (2.4 GB reads better than 2 GB); none above (417 MB, not 417.3 MB).
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`
}

/** Disk per document, for the "what will 10x cost me" question the panel exists to answer.
 *  Empty string at zero documents rather than a division by zero or a meaningless "0 B". */
export function perDocument(totalBytes: number, documents: number): string {
  if (documents <= 0) return ''
  return `${formatBytes(Math.round(totalBytes / documents))} per document`
}

/** A relative "built 2 minutes ago" for the index timestamp; absolute past a day, where relative
 *  stops being informative. `null` in, empty string out — the caller renders nothing. */
export function builtAgo(iso: string | null, now: Date = new Date()): string {
  if (!iso) return ''
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return ''
  const seconds = Math.round((now.getTime() - then.getTime()) / 1000)
  if (seconds < 0) return 'just now' // a clock skew must not print "in -3 minutes"
  if (seconds < 60) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} h ago`
  return then.toLocaleDateString()
}

/** What the panel says about the keyword index and about memory.
 *
 * The sentence is the point of the whole panel, and it differs per state. `on_disk` (ADR-036) keeps
 * memory flat and is the healthy case. `unavailable` is the one that matters: since ADR-038 retired
 * the in-RAM fallback, a failed index build means keyword matching is **off** and the app is
 * answering on the vector arm alone — so this says so plainly and offers the rebuild that fixes it,
 * rather than letting a degraded install look identical to a healthy one. */
export function describeIndex(facts: CorpusFacts): {
  label: string
  memory: string
  rebuildable: boolean
  degraded: boolean
} {
  const { mode, bytes, built_at } = facts.keyword_index
  const ago = builtAgo(built_at)
  if (mode === 'on_disk') {
    const size = bytes === null ? '' : `, ${formatBytes(bytes)}`
    return {
      label: `on disk${size}${ago ? `, built ${ago}` : ''}`,
      memory: 'Memory does not grow with your library — both search indexes live on disk.',
      rebuildable: true,
      degraded: false,
    }
  }
  if (mode === 'unavailable') {
    return {
      label: 'unavailable',
      memory:
        'Keyword search is off — answers are using meaning-based search only, so exact terms ' +
        'may be missed. Rebuild to restore it.',
      // Rebuildable precisely *because* it is broken: this is the recovery action (ADR-038).
      rebuildable: true,
      degraded: true,
    }
  }
  return {
    label: 'none yet',
    memory: 'Add documents to build a keyword index.',
    rebuildable: false,
    degraded: false,
  }
}
