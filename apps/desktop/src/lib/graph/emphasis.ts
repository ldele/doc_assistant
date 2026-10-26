// Which characters of a quoted passage are markdown emphasis, not text. The extractor writes a
// paper's bold and italic as `**…**` and `_…_` (a citation often comes out `_**Fox, 2018**_`),
// and a quote shown as plain text displayed those markers as characters. Display only: the
// passage stays verbatim on the wire and in the store (ADR-043). Tested under `npm test`.

const isWordChar = (c: string | undefined): boolean => c !== undefined && /[\p{L}\p{N}]/u.test(c)

/** Indices of the emphasis delimiters in `text`: every `*` in a run of two or more (`**bold**`),
 *  and every `_` that is not inside a word (`_italic_`, `_**x**_`). `snake_case`, `x_1` and a
 *  single `*` (a footnote, a p-value star) keep theirs. */
export function emphasisMarks(text: string): Set<number> {
  const drop = new Set<number>()
  for (const m of text.matchAll(/\*{2,}/g)) {
    for (let i = m.index; i < m.index + m[0].length; i++) drop.add(i)
  }
  // The nearest character either side that is not itself a delimiter.
  const neighbour = (i: number, step: number): string | undefined => {
    let j = i + step
    while (j >= 0 && j < text.length && (drop.has(j) || text[j] === '_')) j += step
    return text[j]
  }
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== '_') continue
    if (!(isWordChar(neighbour(i, -1)) && isWordChar(neighbour(i, 1)))) drop.add(i)
  }
  return drop
}

/** `text` without its emphasis delimiters. */
export function withoutEmphasis(text: string): string {
  const drop = emphasisMarks(text)
  if (drop.size === 0) return text
  let out = ''
  for (let i = 0; i < text.length; i++) if (!drop.has(i)) out += text[i]
  return out
}
