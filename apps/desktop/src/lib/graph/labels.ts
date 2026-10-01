// How a concept is shown in the Graph tab (ADR-053 decision 3).
//
// The library's own spelling wins when it writes the word in a case — `dIN`, not the stored
// `din`; `Cre`, not `cre` — because the lower-cased label loses a difference the text keeps
// (`Din` is the join in "vitamin Din"). The label itself is never changed: every write and every
// match still uses `label`; this is display only.

export interface Labelled {
  label: string
  written?: string | null
}

export function shownLabel(node: Labelled): string {
  return node.written ?? node.label
}
