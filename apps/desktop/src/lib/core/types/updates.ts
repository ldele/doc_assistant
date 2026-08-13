// TypeScript mirror of the update-check payloads (apps/api/models/updates.py, ADR-044).
// Keep in sync with the pydantic models — this is the wire contract; a change to the model and
// a change here belong in the same commit (apps/desktop/CLAUDE.md).

/**
 * Three states, not two. `unknown` covers "never checked" and "the check failed" — the same
 * thing to a user deciding whether to go look. What it must never collapse into is `current`:
 * saying "up to date" because the network was down is the failure ADR-044 exists to prevent,
 * so the UI reads `reason` in that case instead of inventing reassurance.
 */
export type UpdateState = 'current' | 'update_available' | 'unknown'

export interface UpdateStatus {
  state: UpdateState
  current_version: string
  /** The newest published release, when one was seen. Null whenever `state` is 'unknown'. */
  latest_version: string | null
  /** Where "Get the update" goes. The app never downloads or installs anything (ADR-044). */
  release_url: string
  /** ISO stamp of the last completed check — how old this answer is, not when it was rendered. */
  checked_at: string | null
  /** Plain-words explanation, set only for 'unknown'. Safe to show verbatim. */
  reason: string | null
  /** Whether the *automatic* daily check is on. A manual check runs regardless. */
  auto_check_enabled: boolean
}
