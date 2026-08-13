// Pure display logic for the update check (ADR-044). No fetch, no runes — kept plain `.ts` so
// `node:test` can run it (apps/desktop/CLAUDE.md: `.svelte.ts` needs the compiler).
//
// The one rule these functions encode: an `unknown` state never gets a reassuring sentence.
// "Couldn't check" and "you're up to date" must not be confusable, because the user's next
// action differs — go look manually, or do nothing.

import type { UpdateStatus } from '../core/types/updates'

export interface UpdateSentence {
  /** The headline the section (or banner) shows. */
  headline: string
  /** Supporting line — the reason for 'unknown', the version for the others. Never reassuring
   *  when the check did not complete. */
  detail: string
  /** Whether to offer the "Get the update" link. Only ever true for a confirmed newer release. */
  showLink: boolean
  /** Tone for styling: 'info' is the actionable case, 'muted' the quiet ones. */
  tone: 'info' | 'ok' | 'muted'
}

export function describeUpdate(status: UpdateStatus): UpdateSentence {
  if (status.state === 'update_available' && status.latest_version) {
    return {
      headline: `Version ${status.latest_version} is available`,
      detail: `You're running ${status.current_version}. Download and install it yourself — Provenote never installs anything for you.`,
      showLink: true,
      tone: 'info',
    }
  }
  if (status.state === 'current') {
    return {
      headline: `Provenote ${status.current_version} is up to date`,
      detail: asOf(status.checked_at),
      showLink: false,
      tone: 'ok',
    }
  }
  return {
    headline: "Couldn't check for updates",
    // The backend's reason is written for a person and is safe to show verbatim; the fallback
    // covers a payload that somehow arrived without one.
    detail: status.reason ?? 'No check has run yet.',
    showLink: false,
    tone: 'muted',
  }
}

/** "Checked just now" / "Checked 3 hours ago" / "" when we have no stamp.
 *
 *  A verdict with no age is a verdict pretending to be live, so every 'current' answer carries
 *  one — that is what makes a day-old "up to date" honest rather than misleading.
 */
export function asOf(checkedAt: string | null, now: Date = new Date()): string {
  if (!checkedAt) return ''
  const then = new Date(checkedAt)
  if (Number.isNaN(then.getTime())) return ''
  const minutes = Math.floor((now.getTime() - then.getTime()) / 60000)
  if (minutes < 1) return 'Checked just now.'
  if (minutes < 60) return `Checked ${minutes} minute${minutes === 1 ? '' : 's'} ago.`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `Checked ${hours} hour${hours === 1 ? '' : 's'} ago.`
  const days = Math.floor(hours / 24)
  return `Checked ${days} day${days === 1 ? '' : 's'} ago.`
}

/** Whether the shell should surface a banner outside Settings.
 *
 *  Only a confirmed newer release earns interrupting the user. 'unknown' deliberately does not:
 *  a machine that is simply offline would otherwise nag forever about a check it cannot run,
 *  which is friction with no action attached to it (the "inform, don't block" house rule).
 */
export function shouldNotify(status: UpdateStatus): boolean {
  return status.state === 'update_available' && Boolean(status.latest_version)
}
