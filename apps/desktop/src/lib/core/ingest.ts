// Pure helpers for the ingest indicator — no runes, no fetch, so `node --test` can reach them.
//
// The state and the poller live in `ingest.svelte.ts` next door; everything here is a plain
// function over a plain payload. The split is the house rule (apps/desktop/CLAUDE.md): the
// extension is the marker, and only the type is imported from a sibling so node's type-stripping
// leaves nothing behind at runtime.

import type { IngestStatus } from './types/sources'

/** Poll cadence. Fast enough to feel live, slow enough not to hammer a local API for nothing. */
export const RUNNING_MS = 1000
export const IDLE_MS = 4000
/** How long a finished run stays on screen before the bar goes quiet. Errors never expire. */
export const RESULT_LINGER_MS = 8000
/**
 * Transient failures are normal right after a run: the sidecar rebuilds the controller *before*
 * reporting "done", so a poll can land mid-reload. One miss is not lost contact.
 */
export const MISS_TOLERANCE = 5

/**
 * The fraction complete, or `null` when it cannot honestly be computed.
 *
 * `null` is not 0. A run whose batch has not been counted yet has *no* fraction, and rendering it
 * as an empty bar would claim "0% done" — indistinguishable from a run that is stuck. The caller
 * shows an indeterminate bar for `null` instead.
 */
export function fraction(s: IngestStatus | null): number | null {
  if (!s || s.total <= 0) return null
  return Math.min(1, Math.max(0, s.done / s.total))
}

/**
 * One line naming what is happening, in words a person can act on.
 *
 * Never invents a number: with no `total` yet it says it is preparing rather than showing "0 of 0",
 * which reads as "nothing to do" when the truth is "not counted yet". `added` is only ever spoken
 * once the run is `done` — mid-run it is 0 by design, and reporting it as progress would claim an
 * outcome for documents that may still fail.
 */
export function ingestLabel(s: IngestStatus | null): string {
  if (!s) return 'Indexing…'
  switch (s.state) {
    case 'running':
      return s.total > 0 ? `Indexing ${s.done} of ${s.total}` : 'Preparing to index…'
    case 'error':
      return 'Indexing failed'
    case 'done':
      // A run that walked a whole corpus and found everything already indexed added nothing —
      // saying "Indexed 0 new" is true but reads as a failure, so name the outcome instead.
      return s.added === 0 ? 'Nothing new to index' : `Indexed ${s.added} new`
    default:
      return 'Idle'
  }
}

/**
 * What to say for the file in flight when the backend reports none.
 *
 * `current` is null at three different moments and they do not mean the same thing: before the
 * first document, between two documents, and on the final report once the loop has ended. Saying
 * "starting…" for all three put "Now: starting…" next to "2 of 2", which reads as stuck.
 */
export function currentLabel(s: IngestStatus | null): string {
  if (!s) return '…'
  if (s.current) return s.current
  if (s.total > 0 && s.done >= s.total) return 'finishing…'
  if (s.done === 0) return 'starting…'
  return '…'
}

/**
 * Whether a poll just witnessed a run *finish* — the moment the corpus changed.
 *
 * Both halves matter. Without `wasRunning`, the `done` left behind by a run that ended hours ago
 * would read as a completion on the very first poll after launch, lighting the bar and firing a
 * refresh on every app start. Without the second half, a run still in flight would count. Only the
 * transition is the event; the state alone is not.
 */
export function isCompletion(
  previous: IngestStatus | null,
  next: IngestStatus,
): boolean {
  return previous?.state === 'running' && next.state !== 'running'
}

/** An optimistic "running" status, so the bar can appear on the gesture that started the run. */
export function pendingStatus(sourceDir: string | null = null): IngestStatus {
  return {
    state: 'running',
    source_dir: sourceDir,
    added: 0,
    skipped: 0,
    errors: 0,
    message: null,
    // 0 keeps `fraction` at null, so the bar renders indeterminate until the backend counts.
    total: 0,
    done: 0,
    current: null,
  }
}
