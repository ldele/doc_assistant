// The words a turn's retrieval scope is stated in, above the answer (ADR-025 F2, A7b). Pure, so
// the wording can be pinned by tests; the backend's `retrieval_scope.scope_summary` says the same.

/** The fields of `TurnScope` this reads. Kept structural so the test needs no wire type. */
export interface ScopeLike {
  folders: { folder_id: string; folder_name: string | null; doc_count: number }[]
  concept: { name: string; graph_stale: boolean } | null
  doc_count: number
}

/** A run of text in a scope line; `strong` marks a name. */
export interface ScopePart {
  text: string
  strong?: boolean
}

export const GRAPH_STALE_NOTE = 'The concept graph is out of date; rebuild it to refresh this list.'

function documents(n: number): string {
  return `${n} document${n === 1 ? '' : 's'}`
}

/** The scope as sentences, each a list of parts. The first sentence says what was searched; the
 *  rest name each folder that added nothing (G1), or that the concept graph is out of date. */
export function scopeSentences(scope: ScopeLike): ScopePart[][] {
  if (scope.concept) {
    const out: ScopePart[][] = [
      [
        { text: 'Searched the documents where ' },
        { text: scope.concept.name, strong: true },
        { text: ` is present: ${documents(scope.doc_count)}.` },
      ],
    ]
    if (scope.concept.graph_stale) out.push([{ text: GRAPH_STALE_NOTE }])
    return out
  }
  const searched = scope.folders.filter((f) => f.folder_name !== null && f.doc_count > 0)
  const dropped = scope.folders.filter((f) => f.folder_name === null || f.doc_count === 0)
  const out: ScopePart[][] = []
  if (searched.length) {
    const first: ScopePart[] = [{ text: 'Searched ' }]
    searched.forEach((f, i) => {
      if (i > 0) first.push({ text: ', ' })
      first.push({ text: f.folder_name ?? '', strong: true })
    })
    first.push({ text: ` only: ${documents(scope.doc_count)}.` })
    out.push(first)
  } else {
    out.push([{ text: 'Nothing was searched.', strong: true }])
  }
  for (const f of dropped) {
    out.push(
      f.folder_name === null
        ? [{ text: 'A folder that no longer exists was left out.' }]
        : [{ text: f.folder_name, strong: true }, { text: ' held no documents and was left out.' }],
    )
  }
  return out
}

/** The scope chip's icon: a concept's, or a folder's. */
export function scopeIcon(scope: ScopeLike): 'tag' | 'folder' {
  return scope.concept ? 'tag' : 'folder'
}

/** The sentences as plain text, one line: what a test or a screen reader compares. */
export function scopeText(scope: ScopeLike): string {
  return scopeSentences(scope)
    .map((s) => s.map((p) => p.text).join(''))
    .join(' ')
}
