// TypeScript mirror of the first-run setup payloads (apps/api/models/setup.py, ADR-034, ADR-060).
// Keep in sync with the pydantic models — this is the wire contract; a change to the model and a
// change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// No field here can hold key material: the backend serves a `key_source` label and a last-4
// `key_hint`, never the key.

/** One provider's readiness, plus the action that would fix it. Mirrors ProviderReadinessModel. */
export interface ProviderReadiness {
  id: string
  /** Billed, or not known: what the CLI credit guard stops on (ADR-060). Never a block here. */
  paid: boolean
  /** Credential present (or not needed). Local state only, never a network answer. */
  configured: boolean
  /** Probe result, or `null` when not probed (`?probe=false`) or not probeable (a cloud server). */
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
  /** Models the server reported, from a probe or the last list fetched. */
  models: string[]
  /** What the screens call the provider (`id` is the code's word). */
  name: string
  /** Where the text goes, as a list to sit in. */
  group: 'local' | 'cloud' | 'server'
  /** One user-facing sentence: where the questions and passages go. */
  destination: string
  /** Which form the panel draws. */
  kind: 'claude' | 'ollama' | 'preset' | 'server'
  /** Where the requests go. A preset's is fixed; a server's is the user's. */
  address: string
  reach: 'this_computer' | 'local_network' | 'elsewhere'
  /** Who bills, as the panel says it (*Billed by Groq*, *Billing not known*, ...). */
  billing: string
  billing_kind: 'vendor' | 'nobody' | 'unknown' | 'your_word'
  /** Whether *This server does not bill me* applies. */
  can_tick: boolean
  /** Whether it cannot answer without a key (a server's key is optional). */
  takes_key: boolean
  /** Where to get a key. */
  key_page: string | null
  /** What the last model list said about the live key. */
  key_status: 'accepted' | 'refused' | null
  /** When `models` was fetched from the server (ISO 8601), if it was. */
  models_fetched_at: string | null
  /** Context windows the server's list documents, by model. */
  windows: Record<string, number>
  /** Ollama: models served from elsewhere, with the host they run on. */
  remote: Record<string, string>
  /** Lines the form carries: the window, forwarding, an open address. */
  notes: string[]
  /** Everything an answer sends to it. */
  sends: string[]
}

/** One outstanding (or completed) first-run task: one row of Getting started. Mirrors SetupStepModel. */
export interface SetupStep {
  id: string
  title: string
  detail: string
  done: boolean
  /** The row's status, in the row's own words (*Engine online*, *Indexed*, *Not set*). */
  status: string
  action: string | null
  /** The chosen engine does not answer, which reads apart from a step nobody has done yet. */
  offline: boolean
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
  /** What the screens call the provider answering now. */
  active_name: string
  /** Where the text for the model answering now goes, worked out per model. */
  active_destination: string
  /** Who bills the model answering now. */
  active_billing: string
}

/** POST /api/setup/providers/{id}/key — `unreachable` means stored but not verifiable. */
export interface ApiKeyResult {
  stored: boolean
  verification: 'ok' | 'unreachable'
  detail: string
  setup: SetupState
}

/** What a server said when asked for its models. Mirrors ModelListModel. */
export interface ModelList {
  status: 'ok' | 'refused' | 'needs_key' | 'unreachable' | 'error'
  models: string[]
  detail: string
  fetched_at: string
  windows: Record<string, number>
}

/** POST /api/setup/servers — what *Test connection* found. Mirrors ServerResult. */
export interface ServerResult {
  id: string
  listing: ModelList
  setup: SetupState
}

/** The body of *Test connection*. `id` absent adds a server. Mirrors ServerUpdate. */
export interface ServerUpdate {
  id: string | null
  name: string
  address: string
  key: string | null
}
