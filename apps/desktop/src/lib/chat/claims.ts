// Claims in the answer text (A8b, row 42, docs/specs/chat-interface-pass.md G4). Pure, so the
// matching and the colour rules are pinned by `claims.test.ts`; Markdown.svelte wraps the spans.
//
// What a colour means is only what the retrieval marker measures:
//   - `cited` (lilac): the sentence cites a passage that ranked well (`ok` with a citation);
//   - `weak` (amber): it cites one that ranked low for the question, or one with no score;
//   - no colour: it cites nothing (`unsupported`), or it was not found in the text.
// The marker never checks that a sentence says what its passage says, so no colour claims that.

/** The citation forms `citations.ts` recognises, the same text (a sibling import would need the
 *  `.ts` extension, which the app's type check refuses); `claims.test.ts` pins the two equal. */
export const CLAIM_CITE_BODY = String.raw`\[\s*(?:sources?|refs?)?\s*\d+(?:\s*(?:,|;|&|and)\s*\d+)*\s*\]`

export type ClaimKind = 'cited' | 'weak'

export const WEAK_TOOLTIP = 'Cites a passage that ranked low for this question'
export const CITED_TOOLTIP = 'Cites a passage: click to open it'

/** The fields of a claim this reads (`TurnClaim`). */
export interface ClaimLike {
  marker: string
  sources: number[]
  text: string
}

/** How a claim is coloured, or `null` for plain (G4). */
export function claimKind(c: ClaimLike): ClaimKind | null {
  if (c.marker === 'ok' && c.sources.length > 0) return 'cited'
  if (c.marker === 'weak') return 'weak'
  return null
}

const CITES = new RegExp(CLAIM_CITE_BODY, 'gi')

/** A claim's words as they read on screen: no citation markers (they render as buttons and are
 *  left out of the text matched), no markdown emphasis, code ticks, heading or list markers, and
 *  a link's text without its address. */
export function claimWords(text: string): string[] {
  const plain = text
    .replace(CITES, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)/gm, '')
    .replace(/[*_`~]/g, '')
  return plain.split(/\s+/).filter(Boolean)
}

function escape(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Where each claim sits in `flat` (the answer's text as shown, citation buttons left out), in
 *  order: each search starts where the last match ended, so a sentence written twice is matched
 *  to its own place. A claim that is not found is `null`, and the next one is searched from the
 *  same place: nothing is guessed. */
export function locateClaims(flat: string, claims: string[]): ({ start: number; end: number } | null)[] {
  let from = 0
  return claims.map((text) => {
    const words = claimWords(text)
    if (words.length === 0) return null
    const re = new RegExp(words.map(escape).join('\\s*'), 'g')
    re.lastIndex = from
    const m = re.exec(flat)
    if (!m) return null
    from = m.index + m[0].length
    return { start: m.index, end: from }
  })
}

/** The passages a clicked claim opens, in the order the claim cites them (0.8.0 walk, item 25:
 *  opening only the first hid the rest). A number cited twice opens once; one the turn does not
 *  hold (a degraded past turn) is left out rather than drawn empty. */
export function passagesFor<T extends { n: number }>(available: readonly T[], cited: readonly number[]): T[] {
  const seen = new Set<number>()
  const out: T[] = []
  for (const n of cited) {
    if (seen.has(n)) continue
    seen.add(n)
    const hit = available.find((s) => s.n === n)
    if (hit) out.push(hit)
  }
  return out
}
