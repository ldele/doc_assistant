// How a concept's name is shown, on every screen that shows one (ADR-053 decision 3, ADR-054).
//
// The library's own spelling wins when it writes the word in a case — `dIN`, not the stored
// `din`; `Cre`, not `cre` — because the lower-cased label loses a difference the text keeps
// (`Din` is the join in "vitamin Din"). The label itself is never changed: every write and every
// match still uses `label`; this is display only. The server decides the spelling
// (`written_forms.shown_labels`); these two functions only apply it, so the graph, the vocabulary
// search, the gap list, the taxonomy view and Manage keywords cannot disagree.

export interface Labelled {
  label: string
  written?: string | null
}

export function shownLabel(node: Labelled): string {
  return node.written ?? node.label
}

/** The same rule for a payload that names its fields otherwise (`canonical`, `source_label`). */
export function shownName(label: string, written: string | null = null): string {
  return written ?? label
}

/** "7 more through the broad form `distillation`" — what sits beside a concept's documents
 *  (ADR-054). Empty when nothing does, so a caller renders nothing. The count is never added to
 *  the concept's own: a broad form also matches other things. */
export function broadSummary(broadDocuments: number, broadForms: readonly string[]): string {
  if (broadDocuments <= 0 || broadForms.length === 0) return ''
  const forms = broadForms.join(', ')
  const noun = broadForms.length === 1 ? 'the broad form' : 'the broad forms'
  return `${broadDocuments} more through ${noun} ${forms}`
}

// How many concepts a notice names before it counts the rest: a banner listing twenty names is
// not read, and the count still says how far behind the graph is.
const NAMED_IN_NOTICE = 3

/** The graph's notice for concepts whose name or forms changed since it was built (ADR-054).
 *  Their counts are still the old forms' counts, and no node or edge says so. Empty when nothing
 *  changed, so a caller renders nothing.
 *
 *  `recorded` is false for a graph built before it kept that record: it cannot tell, and the
 *  notice says so instead of claiming a change — the rebuild it sits beside starts the record. */
export function formsChangedNotice(names: readonly string[], recorded: boolean = true): string {
  if (!recorded)
    return "This graph was built before it recorded each concept's name and forms, so it cannot tell whether they changed."
  if (names.length === 0) return ''
  const listed = names.slice(0, NAMED_IN_NOTICE).join(', ')
  const more = names.length - NAMED_IN_NOTICE
  const who = more > 0 ? `${listed} and ${more} more` : listed
  return `Name or forms changed since this graph was built: ${who}.`
}
