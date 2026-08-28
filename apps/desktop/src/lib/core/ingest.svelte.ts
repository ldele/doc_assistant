// Live ingest position, shared by every surface that needs to say "something is indexing".
//
// A leaf rune module, like `library/accept.svelte.ts`: it imports the API client and the pure
// helpers next door, and nothing else from the app — so the status bar, the add sheet and the
// Sources panel all read one run rather than each polling their own and disagreeing about it.
//
// **Why this exists at all.** `POST /api/ingest` is a 202, and the status endpoint used to report
// `{running, 0, 0, 0}` for the whole run, so a user who added a document saw nothing happen and
// could not tell the app from a hung one. The backend now reports position per document
// (`total`/`done`/`current`); this module is what reads it.
//
// The pure parts (`fraction`, `ingestLabel`, the cadences) live in `./ingest` so `node --test` can
// reach them — a `.svelte.ts` module needs the compiler and cannot be unit-tested here.

import { getIngestStatus } from './api/sources'
import {
  IDLE_MS,
  MISS_TOLERANCE,
  RESULT_LINGER_MS,
  RUNNING_MS,
  isCompletion,
  pendingStatus,
} from './ingest'
import type { IngestStatus } from './types/sources'

export { currentLabel, fraction, ingestLabel, isCompletion } from './ingest'

export const ingestRun = $state({
  /** The last status actually read, or null when nothing has been read yet. */
  status: null as IngestStatus | null,

  /** True while the detail panel is open. Owned here so any surface can close it. */
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
function onFinished(state: IngestStatus['state']): void {
  clearLinger()
  if (state === 'error') return // stays until dismissed
  linger = setTimeout(() => {
    ingestRun.visible = false
    ingestRun.panelOpen = false
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
      onFinished(next.state)
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
 * Sources panel, the CLI, a second window — still lights the bar. That is the point of the status
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
export function noteIngestStarted(): void {
  ingestRun.visible = true
  ingestRun.lostContact = false
  clearLinger()
  ingestRun.status = pendingStatus(ingestRun.status?.source_dir ?? null)
  schedule(0)
}

/** Dismiss a finished/failed run from the bar. Never interrupts a run — it only stops showing it. */
export function dismissIngest(): void {
  clearLinger()
  ingestRun.visible = false
  ingestRun.panelOpen = false
}
