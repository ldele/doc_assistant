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
