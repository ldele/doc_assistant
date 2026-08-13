// Build-time feature visibility. One flag per surface that is *built but not ready to be met by
// a first-time user* — the code stays, the entry point goes.
//
// Why a flag and not a deletion: these are working features whose problem is that a user would
// draw a false conclusion from them today. Deleting the code would throw away the work and the
// tests; leaving the door open ships the false conclusion. A flag is the honest middle, and it
// keeps the cost of reversing the call at one line.

/**
 * The Graph workspace (concept graph + gap list).
 *
 * **Hidden for 0.6 by user decision (2026-08-12).** Not because it is broken — because the page
 * is empty until the concept skeleton is built, and an empty page reads as a *failure* rather than
 * as "nothing here yet". Its placement is also explicitly undecided (it is entangled with the
 * Project ADR), so shipping the tab would be shipping a location we intend to move.
 *
 * Everything behind it still works and is still tested: `/api/concepts/*` stays mounted, the gap
 * list keeps its triage writes, and `GraphIndex`/`ConceptGraph`/`GapList` are untouched. Flip this
 * to `true` to bring the tab back — nothing else needs to change.
 *
 * See `docs/REVIEW_2026-08-12_release-readiness.md` §2b R4.
 */
export const GRAPH_TAB_ENABLED = false
