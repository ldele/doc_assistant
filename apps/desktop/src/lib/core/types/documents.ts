// Wire types for the add-documents review (AD2). Mirrors apps/api/models/sources.py.

/** Why a candidate file is or is not going to be added. */
export type AddVerdict = 'add' | 'unsupported' | 'duplicate' | 'unreadable'

export interface FileVerdict {
  path: string
  name: string
  verdict: AddVerdict
  size: number | null
  sha256: string | null
  /** For `unsupported`, `get_format_status`'s own sentence — render it verbatim. */
  advisory: string | null
  /**
   * For `duplicate`, the `registry.source_key` of the registered file it matches —
   * `"<root_id>:<rel_path>"` since AD3b, so it carries a raw uuid for a referenced root. It is an
   * identifier, not a label: show the file's name, not the key.
   */
  duplicate_of: string | null
  selected_by_default: boolean
}

export interface InspectResponse {
  /** Already sorted server-side: every non-`add` verdict precedes every `add`. */
  files: FileVerdict[]
  /** Per-verdict counts plus `total`. */
  counts: Record<string, number>
}

/**
 * What reading an outside catalogue (Zotero today) found — ROADMAP 17, ADR-049.
 *
 * Deliberately stops at *paths*: the review sheet, the duplicate rule and the copy-or-reference
 * choice are the ones the app already has, and an import is just another way of reaching them.
 */
export interface CatalogueScan {
  /** Human name of the catalogue, for the sentence the dialog writes. */
  label: string
  /** The folder the files live under — shown so the user can confirm it is the right library. */
  root: string
  /** Absolute paths, ready to stage. */
  paths: string[]
  found: number
  /**
   * Reason -> count for everything the catalogue held that was not staged. Shown, not summed: a
   * bare "37 found" out of a 500-item library reads as a broken import.
   */
  skipped: Record<string, number>
  /** How many of `paths` the catalogue could describe — the reason to import rather than browse. */
  with_metadata: number
}
