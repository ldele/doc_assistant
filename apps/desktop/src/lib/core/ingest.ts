// Pure helpers for the ingest indicator — no runes, no fetch, so `node --test` can reach them.
//
// The state and the poller live in `ingest.svelte.ts` next door; everything here is a plain
// function over a plain payload. The split is the house rule (apps/desktop/CLAUDE.md): the
// extension is the marker, and only the type is imported from a sibling so node's type-stripping
// leaves nothing behind at runtime.

import type { IngestFileState, IngestGroup, IngestStatus } from './types/sources'

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
 * The status bar's line: it always opens with *Indexing*, so a glance at the bottom of any screen
 * says which job the bar belongs to, then the standard "m/n done" (the user, 2026-11-04).
 *
 * Never invents a number: with no `total` yet it says it is preparing rather than showing "0/0",
 * which reads as "nothing to do" when the truth is "not counted yet". `added` is only ever spoken
 * once the run is `done` — mid-run it is 0 by design, and reporting it as progress would claim an
 * outcome for documents that may still fail.
 */
/**
 * How many files of the run failed: the per-file count once the run has one, else the run's own
 * tally. Read it only once the run has ended — while it runs the label shows position, not outcome.
 */
export function failedCount(s: IngestStatus | null): number {
  if (!s) return 0
  return s.counts?.failed ?? s.errors ?? 0
}

/**
 * Whether an ended run's chip stays on the status bar until dismissed, instead of going quiet
 * after `RESULT_LINGER_MS`. An error and a run with failed files stay: a failure that vanishes on a
 * timer is a failure nobody reads. A stopped run stays too: it still offers *Index the remaining*.
 */
export function staysUntilDismissed(s: IngestStatus): boolean {
  if (s.state === 'error' || s.state === 'stopped') return true
  return s.state === 'done' && failedCount(s) > 0
}

export function ingestLabel(s: IngestStatus | null): string {
  if (!s) return 'Indexing…'
  const doneOf = `${s.done}/${s.total} done`
  // A file that failed is never folded into "done" or "nothing new": the status bar is what stays
  // on screen after the run, so it has to say the run left something behind (the user, 2026-11-04).
  const failed = failedCount(s)
  const failedTail = failed > 0 ? ` · ${failed} failed` : ''
  switch (s.state) {
    case 'running':
      if (s.total <= 0) return 'Indexing… preparing'
      return s.control === 'paused' ? `Indexing paused · ${doneOf}` : `Indexing… ${doneOf}`
    case 'stopped':
      return `Indexing stopped · ${doneOf}${failedTail}`
    case 'error':
      return 'Indexing failed'
    case 'done':
      // A run that walked a whole corpus and found everything already indexed added nothing —
      // saying "0 new" is true but reads as a failure, so name the outcome instead. When files did
      // fail, the failures are the outcome, and "nothing new" would hide them.
      if (failed > 0) {
        return s.added === 0
          ? `Indexing done · ${failed} failed`
          : `Indexing done · ${s.added} new${failedTail}`
      }
      return s.added === 0 ? 'Indexing done · nothing new' : `Indexing done · ${s.added} new`
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

/**
 * Whether a panel's indexing controls are locked: while its own request is out, or while any run is
 * in flight.
 *
 * The second half reads the shared status rather than a flag set by the click. A panel reopened
 * mid-run started nothing, so a flag would offer it a second run the API can only refuse (409).
 */
export function indexLocked(requesting: boolean, s: IngestStatus | null): boolean {
  return requesting || s?.state === 'running'
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
    control: 'none',
    counts: null,
  }
}

// ---- The indexing screen (ROADMAP 31; docs/wireframes/indexing.md) ----------------------------

/**
 * The screen's title. The count includes failed files, as the status bar's does: 12 finished and
 * 2 failed read "14 of 30", because both are done with.
 */
export function runTitle(s: IngestStatus | null): string {
  if (!s) return 'Indexing'
  const at = `${s.done} of ${s.total}`
  switch (s.state) {
    case 'running':
      if (s.total <= 0) return 'Preparing to index…'
      return s.control === 'paused' ? `Paused at ${at}` : `Indexing ${at}`
    case 'stopped':
      return `Stopped at ${at}`
    case 'done':
      return `Finished ${at}`
    case 'error':
      return 'Indexing failed'
    default:
      return 'Indexing'
  }
}

/** One filter above the list: what it shows, and how many files that is. */
export interface GroupChip {
  group: IngestGroup
  label: string
  count: number
}

/**
 * The filters a run offers. A running run (paused or not) has files in progress and waiting; a
 * stopped one has files it never started; a finished one has neither, so neither is offered.
 */
export function groupChips(s: IngestStatus | null): GroupChip[] {
  const c = s?.counts
  if (!s || !c) return []
  const all: GroupChip = { group: 'all', label: 'All', count: c.total }
  const finished: GroupChip = { group: 'finished', label: 'Finished', count: c.added + c.unchanged }
  const failed: GroupChip = { group: 'failed', label: 'Failed', count: c.failed }
  if (s.state === 'running') {
    const chips: GroupChip[] = [
      all,
      { group: 'in_progress', label: 'In progress', count: c.reading },
      { group: 'waiting', label: 'Waiting', count: c.waiting },
      finished,
      failed,
    ]
    // A stopped loop marks what it left before the run reports its end.
    if (c.not_started > 0) chips.splice(3, 0, { group: 'not_started', label: 'Not started', count: c.not_started })
    return chips
  }
  if (s.state === 'stopped') {
    return [all, { group: 'not_started', label: 'Not started', count: c.not_started }, finished, failed]
  }
  return [all, finished, failed]
}

/** A row's status, in the screen's words. */
export const FILE_STATE_LABEL: Record<IngestFileState, string> = {
  waiting: 'Waiting',
  reading: 'Reading',
  added: 'Added',
  unchanged: 'Unchanged',
  failed: 'Failed',
  not_started: 'Not started',
}

/** What a row in progress says, since the backend has no detail for it yet. */
export const READING_DETAIL = 'Reading the text. Its passages are stored next.'

/**
 * The line under the header: what a request is waiting for, or what a pause or stop left. `null`
 * when there is nothing to say (a run going normally, or a finished run).
 */
export function runBanner(s: IngestStatus | null): string | null {
  if (!s) return null
  const c = s.counts
  if (s.state === 'running') {
    switch (s.control) {
      case 'pausing':
        return `Pausing once ${s.current ?? 'the document in progress'} finishes.`
      case 'paused':
        return 'Paused. Searching and chatting work as usual; no new document is read until you resume.'
      case 'stopping':
        return `Stopping after ${s.current ?? 'the document in progress'}. Nothing finished is lost.`
      case 'finishing':
        return 'Making the documents that finished searchable…'
      default:
        return null
    }
  }
  if (s.state === 'stopped' && c) {
    const inLibrary = c.added + c.unchanged
    const docs = inLibrary === 1 ? '1 document is' : `${inLibrary} documents are`
    const left = c.not_started === 1 ? '1 was' : `${c.not_started} were`
    return `${docs} in your library. ${left} not started.`
  }
  return null
}

/** The footer: whether closing the screen loses anything (it never does). */
export function runFooter(s: IngestStatus | null): string {
  if (s?.state === 'running' && s.control === 'paused') {
    return 'You can close this screen. The run stays paused until you resume. Quitting the app ends the run; what finished stays in your library.'
  }
  if (s?.state === 'running') {
    return 'You can close this screen. Indexing keeps running, and the status bar shows where it is.'
  }
  return 'What finished is in your library and can be searched now.'
}

/** The follow-up a stopped or finished run offers, or null when there is none to offer. */
export function rerunLabel(scope: 'failed' | 'remaining', s: IngestStatus | null): string | null {
  const c = s?.counts
  if (!s || !c || s.state === 'running') return null
  if (scope === 'remaining') {
    return s.state === 'stopped' && c.not_started > 0 ? `Index the remaining ${c.not_started}` : null
  }
  if (c.failed === 0) return null
  return c.failed === 1 ? 'Retry the one that failed' : `Retry the ${c.failed} that failed`
}
