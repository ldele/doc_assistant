// Readiness-gate timing: how long to wait between `/api/health` polls, and what to tell the user
// while waiting. Pure and dependency-free so `node:test` can run it (the polling loop that USES it
// lives in App.svelte, which node cannot load).
//
// Why this exists (KI-39, 2026-08-06). The gate used to poll 60 times at 1 s and then set
// `status = 'down'` **permanently** — the `$effect` reads no reactive state before its `await`, so
// it runs once per mount with no retry and no control to trigger one. Against a PyInstaller
// **onefile** sidecar that extracts ~1.5 GB to %TEMP% before uvicorn binds. RG-012 measured health
// at ~30 s — half that budget — on an idle VM with an NVMe disk and the file cache warm from the
// install that had just written those bytes. On a slower machine, or with an antivirus scanning a
// 1.5 GB extraction, exceeding 60 s is plausible, and the result was not "slow" but terminal.
//
// So: **there is no terminal state here.** A backend that has not answered yet is not a backend
// that never will. We keep polling, and we change what we *say* as time passes.

/** A first launch unpacking a large bundle is normal; say so rather than showing a silent dot. */
export const FIRST_LAUNCH_HINT_MS = 20_000

/** Past this, something is likely actually wrong — show it as a fault, but KEEP POLLING so the
 *  app self-heals the moment the backend arrives, with no relaunch. */
export const STALLED_MS = 90_000

/**
 * Delay before poll `attempt` (0-based).
 *
 * 1 s while a warm start is still plausible, easing to 5 s so a genuinely slow first launch is not
 * spinning a request every second for minutes. Bounded — never grows without limit, because the
 * whole point is that a late backend still gets noticed promptly.
 */
export function backoffDelayMs(attempt: number): number {
  if (attempt < 10) return 1_000
  if (attempt < 20) return 2_000
  return 5_000
}

export type StartupPhase = 'connecting' | 'slow' | 'stalled'

/** What to tell the user, given how long the backend has been silent. */
export function startupPhase(elapsedMs: number): StartupPhase {
  if (elapsedMs >= STALLED_MS) return 'stalled'
  if (elapsedMs >= FIRST_LAUNCH_HINT_MS) return 'slow'
  return 'connecting'
}
