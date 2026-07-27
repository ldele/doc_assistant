// TypeScript mirror of the desktop-API payloads (apps/api/models/settings.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// Settings — the locked engine knobs (read-only) plus the user-settable fields.
// Mirrors apps/api/models/settings.py + the services._settings_view read model.

// GET/POST /api/settings — the locked engine knobs (read-only) plus the one user-settable
// knob, the source documents folder, and the live corpus size. Mirrors _full_settings() in
// apps/api/main.py.
export interface Settings {
  // provider/model are the *effective* values (ADR-011, U1c) — the persisted switch if one was
  // made, else the config default. Never stale after a live provider switch.
  provider: string
  model: string
  embedding_model: string
  top_k: number
  candidate_k: number
  use_parent_child: boolean
  synthesis_mode: string
  use_multi_query: boolean
  // ADR-027 D2 (E3): the *effective* persisted answer-layer epistemics default (the persisted
  // toggle if set, else the config default) — same effective-value rule as provider/model.
  // Doubles as the RAG-sandbox baseline; U1b's per-turn override still wins for the session.
  epistemics_markers_enabled: boolean
  reviewer_evidence_chars: number
  parent_chunk: [number, number]
  child_chunk: [number, number]
  retrieval_weights: { bm25: number; vector: number }
  providers: ProviderOption[]
  data_home: string
  source_dir: string
  source_dir_exists: boolean
  supported_formats: string
  chunk_count: number
}
// One entry in the settings view's provider picker (ADR-011, U1c). `available: false` means the
// provider's credential is missing (e.g. no ANTHROPIC_API_KEY) — the UI disables that option.
export interface ProviderOption {
  id: string
  available: boolean
  paid: boolean
}
