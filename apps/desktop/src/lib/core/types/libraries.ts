// TypeScript mirror of apps/api/models/libraries.py — the check before the app opens another
// library folder (ADR-061). Keep in sync with the pydantic model (apps/desktop/CLAUDE.md).

/** POST /api/libraries/inspect: what a folder holds as a library, read without writing. */
export interface LibraryInspection {
  path: string
  is_library: boolean
  documents: number
  chunks: number
  embedding_model: string
  sources_found: number
  /** The library folder its stored cache paths name, when it is not this one (a copy or a move). */
  cache_named_folder: string | null
  is_current: boolean
  /** Sentences that stop *Open*; empty means it may open. */
  problems: string[]
  /** Sentences to read before *Open*. */
  notes: string[]
  can_open: boolean
}
