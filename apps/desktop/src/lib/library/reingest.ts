// Re-run control logic (ADR-048, ROADMAP 20/21) — pure, so it is testable under `node:test`.
//
// The component owns rendering and the fetch; everything here is the arithmetic that decides what
// the control *says*, which is the half worth guarding. The cost statement is the reason the parts
// are separate at all: they differ by four orders of magnitude, so a summary that quietly averaged
// them, or dropped the expensive one, would put the control back to four equivalent checkboxes.

import type { ReingestOptions, ReingestOutcome, ReingestPart, ReingestStatus } from '../core/types'

/** Selected parts, in registry order — never the order the user happened to click them in.
 *  The backend runs `text` last regardless; the control has to *say* the same order it does. */
export function orderedParts(options: ReingestOptions | null, selected: Set<string>): ReingestPart[] {
  return (options?.parts ?? []).filter((p) => selected.has(p.id))
}

/** Whether the run needs a confirmation rather than a plain button.
 *  True when any chosen part moves document identity — today that is `text` alone. */
export function needsConfirmation(parts: ReingestPart[]): boolean {
  return parts.some((p) => p.moves_identity)
}

/** What this run is about to spend, said before it runs.
 *
 * Deliberately quotes the **most expensive** part rather than summing or averaging: the parts span
 * milliseconds to minutes, so an average is a number that describes none of them, and a user
 * deciding whether to press the button cares about the ceiling. The document count is stated
 * separately because row 21 multiplies it. */
export function costSummary(parts: ReingestPart[], documentCount: number): string {
  if (parts.length === 0) return 'Nothing selected.'
  if (documentCount === 0) return 'No documents selected.'
  const dearest = parts[parts.length - 1] // registry order is cheapest-first
  const docs = documentCount === 1 ? '1 document' : `${documentCount} documents`
  const many = documentCount > 1
  // The cost strings are already whole phrases ('instant', 'about 10 seconds'), so the sentence is
  // built around them rather than prefixed onto them — 'about instant' was the first attempt.
  if (parts.length === 1) {
    return `${docs} · ${dearest.cost}${many ? ' each' : ''}.`
  }
  return `${docs} · the slowest part is ${dearest.cost}${many ? ', each' : ''}.`
}

/** One line per document for the finished-run report, collapsing its parts.
 *  Keeps the order the API returned, which is the order they ran in. */
export function outcomesByDocument(
  outcomes: ReingestOutcome[],
): { documentId: string; filename: string; parts: ReingestOutcome[] }[] {
  const order: string[] = []
  const byDoc = new Map<string, ReingestOutcome[]>()
  for (const o of outcomes) {
    if (!byDoc.has(o.document_id)) {
      byDoc.set(o.document_id, [])
      order.push(o.document_id)
    }
    byDoc.get(o.document_id)!.push(o)
  }
  return order.map((id) => ({
    documentId: id,
    filename: byDoc.get(id)![0].filename,
    parts: byDoc.get(id)!,
  }))
}

/** Progress as a fraction, or `null` when there is nothing honest to draw.
 *
 * `null` rather than 0 for an uncounted run — the same rule the ingest bar follows. A bar sitting
 * at zero says "no progress"; there is a difference between that and "we do not know yet". */
export function fraction(status: ReingestStatus | null): number | null {
  if (!status || status.state !== 'running' || status.total <= 0) return null
  return Math.min(1, Math.max(0, status.done / status.total))
}
