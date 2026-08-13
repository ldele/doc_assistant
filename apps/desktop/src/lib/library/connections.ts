// Presentation helpers for the Connections block. Pure module — no Svelte, no sibling value
// imports — so `node:test` can run it (apps/desktop/CLAUDE.md).
//
// WHY THIS EXISTS (docs/REVIEW_2026-08-12_release-readiness.md §2b R1). The panel used to print
// `score.toFixed(2)` beside each related paper, which reads as "these two papers are 92% similar".
// It is not that. `doc_vectors` mean-pools every chunk of a document into one vector, so two papers
// from the same field collapse together almost regardless of what they say: on the 97-document
// corpus there are **750 edges with a median score of 0.918**, against a 0.5 threshold that
// therefore excludes almost nothing.
//
// The ORDER those scores produce is still useful — the nearest neighbour really is the nearest.
// The DISTANCE is not. So the UI shows a rank and never the number: a rank cannot be over-read the
// way "0.92" can, and it is exactly as much as the data supports.

/** English ordinal for a 1-based position: 1 → "1st", 2 → "2nd", 3 → "3rd", 11 → "11th". */
export function ordinal(n: number): string {
  if (!Number.isFinite(n) || n < 1) return ''
  const i = Math.floor(n)
  // 11/12/13 are the exceptions the naive last-digit rule gets wrong.
  const teens = i % 100
  if (teens >= 11 && teens <= 13) return `${i}th`
  switch (i % 10) {
    case 1:
      return `${i}st`
    case 2:
      return `${i}nd`
    case 3:
      return `${i}rd`
    default:
      return `${i}th`
  }
}

/** Badge text for the related-paper at 0-based `index` in a list already sorted nearest-first. */
export function rankLabel(index: number): string {
  return ordinal(index + 1)
}

/**
 * The one-line caveat shown under the Related-papers heading.
 *
 * It is a sentence and not a tooltip on purpose: a user who never hovers is exactly the user who
 * would otherwise read the ordering as a measured claim about how alike two papers are.
 */
export const RELATED_CAVEAT =
  'Ranked by nearest-first, not scored. Similarity is measured across each document as a whole, ' +
  'so papers from the same field sit close together — read the order, not the distance.'
