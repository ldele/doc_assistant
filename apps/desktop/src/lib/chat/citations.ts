// The inline-citation contract, frontend side. Pure and dependency-free so `node:test` can run
// it (the DOM walk that USES it stays in Markdown.svelte, which node cannot load).
//
// Kept deliberately identical to the backend's `synthesis._CITATION_TOKEN_RE` /
// `cited_source_numbers`. Both are pinned by ONE fixture, `tests/fixtures/citation_vectors.json`,
// read by `citations.test.ts` here and by `tests/unit/test_synthesis.py` there — so a change made
// on one side and not the other fails that side's suite.
//
// Why the tolerance exists (2026-07-14): the prompt documents bare `[n]`, but models emit
// unambiguous variants — [Source 2], [Sources 3, 4], [2, 4], [2 and 4] — and dropping those
// renders a correctly-cited answer as uncited. We *read* them; the answer text is never rewritten
// (surface-don't-mutate). Forms no parser can resolve ([karp2020dense], `Source 6 [file.pdf]`)
// stay unresolved on purpose: the backend audit surfaces them as malformed.
//
// KI-35 is why the pin exists: a THIRD implementation (the RG-012 packaging gate) used
// `\[\d+\]`, scored a passing cited turn as FAIL, and that false verdict was filed as an app bug.

/** One citation token: an optional source/ref label, then a separated list of integers. */
export const CITE_BODY = String.raw`\[\s*(?:sources?|refs?)?\s*\d+(?:\s*(?:,|;|&|and)\s*\d+)*\s*\]`

/** Does this text contain a citation token anywhere? (Non-global — no `lastIndex` state.) */
export const CITE_ANYWHERE = new RegExp(CITE_BODY, 'i')

/** Split a string on citation tokens, keeping the tokens (capturing group). */
export const CITE_SPLIT = new RegExp(`(${CITE_BODY})`, 'i')

/** Is this string *entirely* one citation token? */
export const CITE_EXACT = new RegExp(`^${CITE_BODY}$`, 'i')

const CITE_GLOBAL = new RegExp(CITE_BODY, 'gi')
const DIGITS = /\d+/g

/**
 * Every source number cited in `text`, in order of appearance, repeats kept.
 *
 * Mirrors the backend's `cited_source_numbers`. Range is NOT checked here — a citation of `[17]`
 * against 10 sources parses as 17 and is judged out-of-range downstream, exactly as the backend's
 * `audit_citations` does. Tokenising and adjudicating are separate jobs.
 */
export function citationNumbers(text: string): number[] {
  const out: number[] = []
  for (const token of text.match(CITE_GLOBAL) ?? []) {
    for (const digits of token.match(DIGITS) ?? []) out.push(Number(digits))
  }
  return out
}
