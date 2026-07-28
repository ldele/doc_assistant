// TypeScript mirror of the first-run setup payloads (apps/api/models/setup.py, ADR-034).
// Keep in sync with the pydantic models — this is the wire contract; a change to the model and a
// change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// No field here can hold key material: the backend serves a `key_source` label and a last-4
// `key_hint`, never the key.

/** One provider's readiness, plus the action that would fix it. Mirrors ProviderReadinessModel. */
export interface ProviderReadiness {
  id: string
  /** Metered (Anthropic) vs local/free (Ollama) — drives the cost wording, never a block. */
  paid: boolean
  /** Credential present (or not needed). Local state only, never a network answer. */
  configured: boolean
  /** Probe result, or `null` when not probed (`?probe=false`) or not probeable (Anthropic). */
  reachable: boolean | null
  /** Configured and not known-unreachable — i.e. worth selecting. */
  ready: boolean
  /** One user-facing sentence: what is wrong, or what is right. */
  detail: string
  /** The next thing the user should do. `null` when ready. */
  action: string | null
  /** Where the live key comes from: `.env`/environment, or saved in this app. */
  key_source: 'env' | 'app' | null
  /** Last 4 characters of the live key, for display. */
  key_hint: string | null
  /** Models the probe found installed (Ollama). */
  models: string[]
}

/** One outstanding (or completed) first-run task. Mirrors SetupStepModel. */
export interface SetupStep {
  id: string
  title: string
  detail: string
  done: boolean
  action: string | null
}

/** GET /api/setup — the whole first-run picture. Mirrors SetupStateModel. */
export interface SetupState {
  providers: ProviderReadiness[]
  active_provider: string
  active_model: string
  active_ready: boolean
  chunk_count: number
  document_count: number
  ollama_host: string
  steps: SetupStep[]
  ready: boolean
}

/** POST /api/setup/anthropic-key — `unreachable` means stored but not verifiable (offline/proxy). */
export interface ApiKeyResult {
  stored: boolean
  verification: 'ok' | 'unreachable'
  detail: string
  setup: SetupState
}
