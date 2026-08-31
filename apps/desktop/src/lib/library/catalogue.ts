// Pure helpers for importing from an outside catalogue (ROADMAP 17, ADR-049).
//
// Kept out of the Svelte component for the reason the module CLAUDE.md gives: `.svelte` and
// `.svelte.ts` cannot run under `node:test`, and the sentence a user reads when an import finds
// nothing is exactly the kind of thing that should be pinned rather than eyeballed.

/**
 * Turn a reason -> count map into "12 web-page snapshots · 3 in the Zotero trash".
 *
 * **Reasons, never a bare total.** "0 documents" out of a 500-item library reads as a broken
 * import; "412 web-page snapshots · 88 not downloaded to this computer" reads as a working filter
 * and tells the user what to change. Ordered by count so the dominant reason leads.
 */
export function skippedSentence(skipped: Record<string, number>): string {
  return Object.entries(skipped)
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([reason, count]) => `${count} ${reason}`)
    .join(' · ')
}

/**
 * What to say when a catalogue was read but nothing was staged.
 *
 * A separate function from the sentence above because the two failures are different: "your
 * library is empty" and "everything in it was filtered out" need different next steps from the
 * user, and collapsing them into one message would hide which one happened.
 */
export function nothingToAdd(skipped: Record<string, number>): string {
  const reasons = skippedSentence(skipped)
  return reasons
    ? `Nothing to add from that library — ${reasons}.`
    : 'That Zotero library has no documents in it yet.'
}
