// Live ingest position, shared by every surface that needs to say "something is indexing".
//
// A leaf rune module, like `library/accept.svelte.ts`: it imports the API client and the pure
// helpers next door, and nothing else from the app — so the status bar, the add sheet and the
// Manage files list all read one run rather than each polling their own and disagreeing about it.
//
// **Why this exists at all.** `POST /api/ingest` is a 202, and the status endpoint used to report
// `{running, 0, 0, 0}` for the whole run, so a user who added a document saw nothing happen and
// could not tell the app from a hung one. The backend now reports position per document
// (`total`/`done`/`current`); this module is what reads it.
//
// The pure parts (`fraction`, `ingestLabel`, the cadences) live in `./ingest` so `node --test` can
// reach them — a `.svelte.ts` module needs the compiler and cannot be unit-tested here.

import { untrack } from 'svelte'

import { controlIngest, getIngestStatus, rerunIngest } from './api/sources'
import {
  IDLE_MS,
  MISS_TOLERANCE,
  RESULT_LINGER_MS,
  RUNNING_MS,
  isCompletion,
  pendingStatus,
  staysUntilDismissed,
} from './ingest'
import type { IngestStatus } from './types/sources'

export {
  currentLabel,
  failedCount,
  fraction,
  groupChips,
  indexLocked,
  ingestLabel,
  isCompletion,
  rerunLabel,
  runBanner,
  runFooter,
  runTitle,
} from './ingest'

export const ingestRun = $state({
  /** The last status actually read, or null when nothing has been read yet. */
  status: null as IngestStatus | null,

  /**
   * True while the indexing screen is open (ROADMAP 31). Owned here so the status bar, Settings and
   * the screen itself all open and close the same one.
   */
  panelOpen: false,

  /**
   * Whether the bar should show this run at all. True while running, and for a short linger after
   * it finishes so the result is actually seen. An `error` stays until dismissed — a failure that
   * vanishes on a timer is a failure nobody reads.
   */
  visible: false,

  /** True when contact with the indexer was lost mid-run — stated, never silently hidden. */
  lostContact: false,

  /**
   * Incremented once each time a run this watcher was following reaches `done` or `error`.
   *
   * A counter rather than a boolean or a callback list: an `$effect` that reads it re-runs on
   * every completion, which is exactly the subscription shape Svelte 5 already gives us. It exists
   * because the corpus changes when a run ends, and the surfaces that show the corpus — the
   * library grid, the chunk count — otherwise refresh on the 202 and so read the corpus *before*
   * the run that changed it. That is the second half of "I could not tell it worked": the document
   * had indexed, and the list still did not show it.
   */
  completedRuns: 0,
})

let timer: ReturnType<typeof setTimeout> | null = null
let linger: ReturnType<typeof setTimeout> | null = null
let stopped = true
let misses = 0

function clearLinger(): void {
  if (linger !== null) {
    clearTimeout(linger)
    linger = null
  }
}

function schedule(ms: number): void {
  if (stopped) return
  if (timer !== null) clearTimeout(timer)
  timer = setTimeout(() => void tick(), ms)
}

/** A run we were watching has just ended. */
function onFinished(status: IngestStatus): void {
  clearLinger()
  // An error, a run with failed files, and a run the person stopped stay until dismissed
  // (`staysUntilDismissed`): a failure that vanishes on a timer is a failure nobody reads.
  if (staysUntilDismissed(status)) return
  // The chip goes quiet; an open screen stays open. Someone reading which files failed must not
  // have the list taken away on a timer.
  linger = setTimeout(() => {
    ingestRun.visible = false
  }, RESULT_LINGER_MS)
}

async function tick(): Promise<void> {
  if (stopped) return
  // Nothing to learn from a hidden window, and polling one spends battery on a bar nobody can see.
  // The next visibilitychange re-arms immediately.
  if (typeof document !== 'undefined' && document.hidden) {
    schedule(IDLE_MS)
    return
  }
  try {
    const next = await getIngestStatus()
    const finished = isCompletion(ingestRun.status, next)
    misses = 0
    ingestRun.lostContact = false
    ingestRun.status = next

    if (next.state === 'running') {
      ingestRun.visible = true
      clearLinger()
    } else if (finished) {
      ingestRun.visible = true
      ingestRun.completedRuns += 1
      onFinished(next)
    }
    schedule(next.state === 'running' ? RUNNING_MS : IDLE_MS)
  } catch {
    if (++misses >= MISS_TOLERANCE && ingestRun.status?.state === 'running') {
      ingestRun.lostContact = true
    }
    schedule(IDLE_MS)
  }
}

/**
 * Start watching. Call once from the app root; the returned function unsubscribes.
 *
 * Polls on an idle heartbeat even when nothing is running, so a run started somewhere else — the
 * Manage files list, the CLI, a second window — still lights the bar. That is the point of the status
 * bar being ambient rather than owned by whichever screen kicked the run off.
 */
export function watchIngest(): () => void {
  stopped = false
  misses = 0
  void tick()

  const onVisibility = (): void => {
    if (typeof document !== 'undefined' && !document.hidden) schedule(0)
  }
  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', onVisibility)
  }

  return () => {
    stopped = true
    if (timer !== null) clearTimeout(timer)
    timer = null
    clearLinger()
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }
}

/**
 * Tell the watcher a run was just requested, so the bar appears on the same gesture that started
 * it rather than up to `IDLE_MS` later. Safe to call when no watcher is running.
 */
export function noteIngestStarted(sourceDir: string | null = null): void {
  ingestRun.visible = true
  ingestRun.lostContact = false
  clearLinger()
  ingestRun.status = pendingStatus(sourceDir ?? ingestRun.status?.source_dir ?? null)
  schedule(0)
}

/**
 * Call `onEnd` when a run this component saw in flight ends, whichever surface started it.
 *
 * For a panel that reports a result. A panel can close and reopen mid-run, so it follows the shared
 * status rather than a poll of its own, and `isCompletion` keeps the `done` an earlier run left
 * behind from reading as this run's result. Call it from a component's script: `$effect` needs one.
 */
export function useRunEnd(onEnd: (status: IngestStatus) => void): void {
  let seen: IngestStatus | null = null
  $effect(() => {
    const next = ingestRun.status
    const ended = next !== null && isCompletion(seen, next)
    seen = next
    if (ended) untrack(() => onEnd(next))
  })
}

/** Open the indexing screen: from the status bar, or from Settings with *View indexing*. */
export function openIndexing(): void {
  ingestRun.panelOpen = true
  schedule(0)
}

/**
 * Pause, resume or stop the run in flight. The answer is shown at once, so *Pausing…* appears on
 * the click rather than on the next poll; the poll then follows the run as usual.
 */
export async function controlRun(action: 'pause' | 'resume' | 'stop'): Promise<void> {
  ingestRun.status = await controlIngest(action)
  schedule(0)
}

/** Index what a stop left, or retry what failed, from the server's own record of the last run. */
export async function rerunLast(scope: 'failed' | 'remaining'): Promise<void> {
  const started = await rerunIngest(scope)
  noteIngestStarted(started.source_dir)
}

/** Dismiss a finished/failed run from the bar. Never interrupts a run — it only stops showing it. */
export function dismissIngest(): void {
  clearLinger()
  ingestRun.visible = false
  ingestRun.panelOpen = false
}
