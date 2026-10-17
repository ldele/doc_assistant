// Pure display logic for what the text says about a concept's name (ROADMAP 93, ADR-053
// decision 4). Tested under `npm test`; ConceptDefinition.svelte only renders what this returns.
//
// Only the signals the user's answers supported are on the wire
// (tests/eval/baselines/term_signals_2026-10-14.md): short and long forms with their sentence, a
// cited author, and where a label no sentence uses does occur. Every one is offered — nothing
// here adds a form; that is done in Manage keywords. A place is described, never called noise.
import type { CitedAuthor, ConceptSignals, FormOffer, SignalExample, SignalPlace, SignalPlaceKind } from '../core/types'

export interface Segment {
  text: string
  mark: boolean
}

/** The example cut at its marks, whitespace collapsed for reading. Display only: the text stays
 *  verbatim on the wire (ADR-043). The server places the marks; one that overlaps another or
 *  falls outside the text is dropped rather than guessed at. */
export function markedSegments(example: SignalExample): Segment[] {
  const text = example.text
  const marks = [...example.marks].sort((a, b) => a[0] - b[0])
  const raw: Segment[] = []
  let at = 0
  for (const [start, end] of marks) {
    if (start < at || end <= start || end > text.length) continue
    if (start > at) raw.push({ text: text.slice(at, start), mark: false })
    raw.push({ text: text.slice(start, end), mark: true })
    at = end
  }
  if (at < text.length) raw.push({ text: text.slice(at), mark: false })
  const out = raw.map((s) => ({ ...s, text: s.text.replace(/\s+/g, ' ') }))
  if (out.length > 0) {
    out[0].text = out[0].text.trimStart()
    out[out.length - 1].text = out[out.length - 1].text.trimEnd()
  }
  return out.filter((s) => s.text.length > 0)
}

const plural = (n: number, one: string, many: string): string => (n === 1 ? `1 ${one}` : `${n} ${many}`)

/** Whether there is anything to show: an offer, a cited author, or a label no sentence uses. */
export function hasSignals(s: ConceptSignals | null): boolean {
  return !!s && s.available && (s.offers.length > 0 || s.cited !== null || s.silent)
}

/** One offer in words: which of the row's forms, and what the library gives beside it. */
export function offerLine(o: FormOffer): string {
  return o.kind === 'long'
    ? `“${o.beside}” is spelled out as “${o.form}”`
    : `“${o.beside}” is abbreviated as “${o.form}”`
}

export function offerMeta(o: FormOffer): string {
  const parts = [`in ${plural(o.documents, 'document', 'documents')}`]
  if (o.known) parts.push('already one of its forms')
  return parts.join(' · ')
}

/** Whether any offer is a form the row does not have yet — then the panel says where to add it. */
export function hasNewOffer(s: ConceptSignals | null): boolean {
  return !!s && s.offers.some((o) => !o.known)
}

/** "18 of 19 mentions are followed by “et al.”" — the count the rule (half or more) rests on. */
export function citedLine(c: CitedAuthor): string {
  const noun = c.mentions === 1 ? 'mention' : 'mentions'
  const verb = c.et_al === 1 ? 'is' : 'are'
  return `${c.et_al} of ${c.mentions} ${noun} ${verb} followed by “et al.”`
}

const PLACE_LABELS: Record<SignalPlaceKind, string> = {
  sentence: 'In sentences, written another way',
  table: 'In tables',
  heading: 'In headings',
  figure: 'In a figure’s caption or description',
  references: 'In reference lists',
  elsewhere: 'In lines that are not sentences',
}

const PLACE_HINTS: Partial<Record<SignalPlaceKind, string>> = {
  sentence: 'another case, or other characters between its words — not counted as a use',
  elsewhere: 'a list, a short line, a page header or footer, code',
}

export function placeLabel(kind: SignalPlaceKind): string {
  return PLACE_LABELS[kind] ?? kind
}

export function placeHint(kind: SignalPlaceKind): string | null {
  return PLACE_HINTS[kind] ?? null
}

export function placeMeta(p: SignalPlace): string {
  return `${plural(p.mentions, 'time', 'times')} in ${plural(p.documents, 'document', 'documents')}`
}

/** The sentence that opens the section for a label no sentence uses. */
export function silentLine(s: ConceptSignals): string {
  return s.places.length > 0
    ? 'No sentence in your library uses it. It occurs here instead:'
    : 'No sentence in your library uses it, and it is not found anywhere else in the text.'
}

/** Set only when the read stopped short: every count above is over the documents read. */
export function readScope(s: ConceptSignals | null): string | null {
  if (!s || !s.available || s.documents_read >= s.documents_matched) return null
  return `Read in the ${s.documents_read} documents that mention it most, of ${s.documents_matched} the keyword index found.`
}

/** Where an example comes from. */
export function exampleSource(e: SignalExample): string {
  return e.document_title ? `From “${e.document_title}”` : 'From your library'
}
