// Pure helpers for the Re-index confirmation (the user, 2026-11-04): what the run would read, a
// rough duration, and the button's words. No runes and no fetch, so `node --test` reaches them.
//
// The duration is the backend's: seconds per document measured on this computer's last run, times
// the files to read (`doc_assistant.ingest.estimate`). Before any run there is no number, and the
// dialog says so rather than inventing one.

import type { IngestPlan } from '../core/types/sources'

/** How many files the run reads: new ones, plus changed ones unless only new files are asked for. */
export function filesToRead(plan: IngestPlan, onlyNew: boolean): number {
  return onlyNew ? plan.new : plan.new + plan.changed
}

/**
 * A duration in the words a person estimates in. Rough on purpose: minutes are rounded to five
 * past twenty, since a per-document mean is honest about a batch and never about one file.
 */
export function formatDuration(seconds: number | null): string | null {
  if (seconds === null) return null
  if (seconds < 45) return 'under a minute'
  let minutes = Math.round(seconds / 60)
  if (minutes <= 1) return 'about a minute'
  if (minutes >= 20) minutes = Math.round(minutes / 5) * 5
  if (minutes < 60) return `about ${minutes} minutes`
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  const hours = h === 1 ? '1 hour' : `${h} hours`
  return m === 0 ? `about ${hours}` : `about ${hours} ${m} min`
}

function plural(n: number, one: string, many: string): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`
}

/** One line per kind of file, in the order the run treats them. Empty kinds are left out. */
export function planLines(plan: IngestPlan, onlyNew: boolean): string[] {
  const lines: string[] = []
  if (plan.new > 0) lines.push(`${plural(plan.new, 'new file', 'new files')}: indexed`)
  if (plan.changed > 0) {
    const what = onlyNew ? 'keep the version already indexed' : 'read again'
    lines.push(`${plural(plan.changed, 'file changed', 'files changed')} since indexing: ${what}`)
  }
  if (plan.unchanged > 0) {
    lines.push(`${plural(plan.unchanged, 'file', 'files')} already indexed: skipped`)
  }
  if (plan.excluded > 0) lines.push(`${plural(plan.excluded, 'excluded file', 'excluded files')}: skipped`)
  return lines
}

/** The estimate, or why there is none. */
export function estimateLine(plan: IngestPlan, onlyNew: boolean): string {
  if (filesToRead(plan, onlyNew) === 0) {
    return onlyNew ? 'No new files to index.' : 'Nothing to read: every file is already indexed.'
  }
  const duration = formatDuration(onlyNew ? plan.seconds_new : plan.seconds_all)
  if (duration === null) {
    return 'No estimate yet: Provenote measures its speed on this computer the next time it indexes a document.'
  }
  return `It should take ${duration}.`
}

/** Where the estimate comes from, so it can be weighed; null when there is none. */
export function measuredLine(plan: IngestPlan): string | null {
  const per = plan.seconds_per_document
  if (per === null || plan.measured_documents <= 0) return null
  const each = per < 10 ? per.toFixed(1) : Math.round(per).toString()
  const over = plural(plan.measured_documents, 'document', 'documents')
  return `Measured on this computer: ${each} s per document on the last run (${over}). Documents differ a lot, so this is rough.`
}

/** The confirm button: what it will do, in a count. */
export function confirmLabel(plan: IngestPlan, onlyNew: boolean): string {
  const n = filesToRead(plan, onlyNew)
  if (n === 0) return onlyNew ? 'Nothing to index' : 'Re-index anyway'
  return `Index ${plural(n, 'file', 'files')}`
}

/** Only new files with none new has nothing to send: the server would refuse it (400). */
export function confirmDisabled(plan: IngestPlan, onlyNew: boolean): boolean {
  return onlyNew && plan.new === 0
}
