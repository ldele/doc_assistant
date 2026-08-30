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
 * **On since 2026-08-29** (ROADMAP row 22, user decision "before moving on to next release").
 * Hidden for 0.6 on 2026-08-12 because the page was empty until the concept skeleton was built
 * and an empty page reads as a *failure* rather than as "nothing here yet"
 * (`docs/REVIEW_2026-08-12_release-readiness.md` §2b R4, which asked for either a real empty
 * state or the hide). Both halves are answered now: the skeleton is built (13 concepts, 19 edges,
 * 6 communities, not stale — measured 2026-08-28), and R4's actual remedy shipped with the flip —
 * `ConceptGraph`/`GraphIndex` now separate *never built* from *built and empty*, so neither state
 * can be read as a broken page.
 *
 * **The caveat that came with the flip, and how it was closed the same day.** Graph vocabulary is
 * curated, not automatic (ADR-018): a concept joins the graph only at `graph_include=true`, and
 * every in-app path (`create_keyword_family`, `promote_keyword`) creates concepts with it **off**.
 * At the moment of the flip the only writer was `scripts/curate_concepts.py`, so this machine had
 * 13 of its 593 concepts in the graph and a fresh install had an empty page nothing in the app
 * could fill. That is now the Manage-keywords toggle (ROADMAP row 23 — the follow-up ADR-018 had
 * already specified), and the empty state's primary action is the door to it.
 *
 * The second reason for hiding is also still open: placement is entangled with the Project ADR, so
 * this tab may yet move. That is what keeps the flag here instead of deleting it — reversing or
 * relocating the call stays one line.
 */
export const GRAPH_TAB_ENABLED = true
