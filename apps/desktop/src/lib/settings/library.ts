// Pure helpers for opening another library (ADR-061): the check's answer in words, and the names
// the controls use. No runes and no fetch, so `node --test` reaches them.

import type { LibraryInspection } from '../core/types/libraries'

/** A library as the controls name it: the folder, or the app's own when there is none. */
export function libraryName(dataHome: string | null): string {
  return dataHome ?? "the app's own library"
}

/** The one-line answer: how much it holds, and whether this app can search it. */
export function inspectionSummary(i: LibraryInspection): string | null {
  if (!i.is_library) return null
  const docs = `${i.documents.toLocaleString()} document${i.documents === 1 ? '' : 's'}`
  if (i.chunks === 0) return `${docs}, no chunks for this app's model (${i.embedding_model})`
  return `${docs} · ${i.chunks.toLocaleString()} chunks, searchable with ${i.embedding_model}`
}

/** Where the sources are, when the library has any documents. */
export function sourcesLine(i: LibraryInspection): string | null {
  if (!i.is_library || i.documents === 0) return null
  return `${i.sources_found.toLocaleString()} of ${i.documents.toLocaleString()} source files found where the library says.`
}

/** What *Go back* returns to after a switch that did not answer (ADR-061 choice 7). */
export function goBackLabel(previous: string | null): string {
  return previous === null ? "Go back to the app's own library" : 'Go back to the previous library'
}
