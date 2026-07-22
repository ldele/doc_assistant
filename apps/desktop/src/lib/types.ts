// TypeScript mirror of the desktop-API payloads (apps/api/models.py, PR-M2/M3).
// Keep in sync with the pydantic models — this is the wire contract.

// ADR-027 D3 — one source's always-on epistemic assessment. Mirrors
// apps/api/models.py::SourceEpistemicsPayload. `coverage` null = "not assessed".
export interface SourceEpistemics {
  coverage: 'corroborated' | 'unique' | 'contested' | null
  superseded: boolean
  n_claims: number
  year: number | null
}

// ADR-027 D3 — strip-level freshness. Mirrors apps/api/models.py::SourceEvalSummaryPayload.
export interface SourceEvalSummary {
  graph_version: string | null
  stale: boolean
}

export interface SourceView {
  n: number
  citation: string
  excerpt: string
  figure_id: string | null
  chunk_key: string | null
  markers: string[]
  // ADR-027 D3 — always-on per-source evaluation + the rerank score (strip signals).
  reranker_score: number
  evaluation: SourceEpistemics | null
}

export interface ClaimView {
  claim_id: string
  n: number
  text: string
  badge: string
}

export interface UsageView {
  turn_input: number
  turn_output: number
  session_total: number
  cost_usd: number | null
  is_local: boolean
}

export interface TurnResult {
  answer: string
  mode: 'ai' | 'human'
  sources: SourceView[]
  flagged_claims: ClaimView[]
  usage: UsageView
  standalone_query: string
  record_id: string | null
  provenance_card_md: string
  claim_review_md: string
  sources_md: string
  usage_md: string
  citation_note_md: string
  download_path: string | null
  // ADR-025 F2 — set when the turn searched only one folder; null = the whole library.
  // Whenever this is set the UI MUST say so: an answer drawn from a subset that doesn't
  // announce it is the failure the folders feature exists to prevent.
  scope: TurnScope | null
  // ADR-027 D3 — strip-level freshness for the always-on source-evaluation strip (per-source
  // evaluation rides on each source). null = no epistemics sidecar / 0-doc → no strip.
  source_eval: SourceEvalSummary | null
}

// The retrieval scope a turn ran under. `folder_name` is null when the folder was deleted
// before the turn ran — the turn then searched nothing rather than everything.
// Mirrors apps/api/models.py::ScopePayload.
export interface TurnScope {
  folder_id: string
  folder_name: string | null
  doc_count: number
}

export interface Health {
  status: string
  chunk_count: number
  model: string
  embedding_model: string
}

// Conversation history (feature-conversation-history.md). GET /api/conversations returns the
// sidebar list; GET /api/conversations/{sid} rehydrates one chat as a read-only transcript.
// Mirrors apps/api/models.py::Conversation*Payload.
export interface ConversationSummary {
  session_id: string
  title: string
  turn_count: number
  started_at: string // ISO 8601
  last_at: string // ISO 8601
  pinned: boolean
  archived: boolean
}

// A rehydrated citation — degraded vs a live SourceView (no markers/figures; not persisted).
export interface ConversationSource {
  n: number
  citation: string
  excerpt: string
}

export interface ConversationTurn {
  record_id: string
  question: string
  answer: string
  sources: ConversationSource[]
  // ADR-025 F2 — replayed from the record so a reopened scoped answer still says it was scoped.
  scope: TurnScope | null
}

export interface ConversationDetail {
  session_id: string
  title: string
  turns: ConversationTurn[]
}

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

// GET /api/ingest/status and the POST /api/ingest 202 body. Mirrors _IngestStatus.
export interface IngestStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  source_dir: string | null
  added: number
  skipped: number
  errors: number
  message: string | null
}

export type Decision = 'accepted' | 'rejected' | 'edited'

// Session-scoped, non-persistent RAG-sandbox overrides (ADR-010, + the U1b niche-knob
// amendment). `undefined`/`null` (a field or the whole object) = use the locked default.
// Mirrors apps/api/models.py::RagOverrides.
export interface RagOverrides {
  top_k?: number | null
  synthesis_mode?: 'ai' | 'human' | null
  use_multi_query?: boolean | null
  epistemics_markers_enabled?: boolean | null
  reviewer_evidence_chars?: number | null
}

// One entry in the settings view's provider picker (ADR-011, U1c). `available: false` means the
// provider's credential is missing (e.g. no ANTHROPIC_API_KEY) — the UI disables that option.
export interface ProviderOption {
  id: string
  available: boolean
  paid: boolean
}

// Library browser (feature-library-browser.md, L1 — read-only). GET /api/library/documents lists
// ingested docs; GET /api/library/documents/{id} returns its chunks as parent blocks. Mirrors
// apps/api/models.py::Library*Payload. NULL metadata (title/authors/year, health) stays null.
export interface LibraryDocument {
  id: string
  filename: string
  title: string | null // effective: user override ?? auto-extracted (ADR-013)
  authors: string | null // effective
  year: number | null // effective
  customized: boolean // a user metadata override is in force
  format: string
  health: string | null
  chunk_count: number | null
  page_count: number | null
  folders: string[] // display names
  folder_ids: string[] // the key — a root folder name is not unique (ADR-025 F1, spec D2)
  tags: string[]
  keywords: string[]
  added_at: string | null // ISO 8601
}

// A Library folder (ADR-025 F1, docs/specs/feature-corpus-folders.md). Organises the Library
// and, since F2, a chat turn's retrieval scope. `parent_id` is always null in v1
// (folders are flat, spec D1); `doc_count` excludes archived documents, matching the grid.
// Mirrors apps/api/models.py::LibraryFolderPayload.
export interface LibraryFolder {
  id: string
  name: string
  description: string | null
  parent_id: string | null
  doc_count: number
}

// Tag families (feature-tag-families.md, PR-1). A family is a curated Concept whose aliases are
// member Keyword names (ADR-015); `doc_count` is the union of docs carrying any member keyword.
// Mirrors apps/api/models.py::KeywordFamilyPayload.
export interface KeywordFamily {
  id: string
  canonical: string
  aliases: string[]
  doc_count: number
}

// Detection (PR-2). A zero-LLM proposal — nothing has been written; accepting one calls the
// create-family API above. Mirrors apps/api/models.py::KeywordFamilyProposalPayload.
export interface KeywordFamilyProposal {
  canonical: string
  members: string[]
  tier: 'morphological' | 'embedding'
  confidence: number
}

// Selective ingestion (feature-selective-ingestion.md, S2). GET /api/sources lists every file
// under the source dir with a derived ingest status; PATCH /api/sources sets `excluded`; POST
// /api/ingest {paths} ingests a selection. Mirrors apps/api/models.py::SourceFilePayload.
// `doc_type` is always null in v1 (the backend's dormant column).
export interface SourceFile {
  rel_path: string
  format: string
  size: number
  mtime: number
  status: 'new' | 'changed' | 'ingested' | 'missing'
  excluded: boolean
  doc_type: string | null
}

export interface LibraryChild {
  child_index: number
  text: string
  retrievable: boolean
}

export interface LibraryParent {
  parent_index: number
  parent_text: string
  children: LibraryChild[]
}

export interface LibraryDocumentChunks {
  id: string
  filename: string
  format: string
  title: string | null
  authors: string | null
  year: number | null
  chunk_count: number | null
  health: string | null
  parents: LibraryParent[]
  child_count: number
}

// A/B-compare (feature-ab-compare-sandbox.md, U6 — retrieval diff). POST /api/compare returns both
// retrieved source sets (A = locked defaults, B = session override) + a diff + an honest note.
// Mirrors apps/api/models.py::Compare*Payload.
export interface CompareEff {
  top_k: number
  use_multi_query: boolean
}

export interface CompareSource {
  rank: number
  filename: string
  page: number | null
  section: string | null
  score: number
  excerpt: string
  citation: string
  identity: string
}

export interface CompareRow {
  identity: string
  source_a: CompareSource | null
  source_b: CompareSource | null
  status: 'in_both' | 'only_in_a' | 'only_in_b'
  rank_delta: number | null
}

export interface CompareResult {
  query: string
  sources_a: CompareSource[]
  sources_b: CompareSource[]
  rows: CompareRow[]
  eff_a: CompareEff
  eff_b: CompareEff
  note: string
  // ADR-025 F2 — the folder BOTH sides were retrieved under; null = the whole library.
  scope_label: string | null
}

// Concept graph (docs/specs/feature-concept-graph.md, ADR-017; PR-G1 serves it, PR-G2a renders it).
// Mirrors apps/api/models.py::ConceptGraph*Payload / Gap / staleness — ONE id space: every id here
// is a Concept UUID; `label` rides only on the node (the UUID-vs-label mismatch was KI-15).
export interface ConceptGraphNode {
  id: string
  label: string
  doc_ids: string[]
  degree: number
  community: number
}

// `relation` is the deferred Node-B stance annotation — `null` on every edge until that pass runs,
// so a renderer must not imply agreement/disagreement it does not have. Weights span a narrow
// 2.377–2.949 (nearly flat) — do not lean on them for thickness.
export interface ConceptGraphEdge {
  source: string
  target: string
  provenance: string[]
  weight: number
  n_cooccurrence_chunks: number
  relation: string | null
}

// `id` is POSITIONAL, not identity — it renumbers when the vocabulary changes. Never persist a
// preference against it.
export interface ConceptCommunity {
  id: number
  label: string
  node_ids: string[]
  size: number
}

// A detected corpus gap (ADR-004), anchored to `concept_id`. `status` is the raw row value; a
// user's triage lives in its own override sidecar (ADR-017 C1, PR-G2b), so it is not resolved here.
export type GapKind =
  | 'isolated'
  | 'single_source'
  | 'thin_bridge'
  | 'under_connected'
  | 'unsourced_claim'
  | 'citation_missing'
  | 'suggested_link'
  | 'suggested_concept'
  | 'thin_area'

export interface Gap {
  concept_id: string
  tier: string
  determinism: string
  kind: GapKind
  fact_ids: string[]
  rating: number | null
  status: string
}

// The skeleton is a build artifact and the Manage-keywords view writes Concept rows live, so drift
// is structural, not a defect: the UI reports it and offers a rebuild (never auto-rebuilds).
export interface GraphStaleness {
  stale: boolean
  n_concepts_in_db: number
  n_concepts_in_skeleton: number
  added_labels: string[]
  removed_ids: string[]
}

export interface ConceptGraph {
  graph_version: string
  nodes: ConceptGraphNode[]
  edges: ConceptGraphEdge[]
  communities: ConceptCommunity[]
  gaps: Gap[]
  staleness: GraphStaleness
}

// `chunk_keys` are ADR-4 composite `"{document_id}:p{parent_index}"` — the navigation payload that
// takes the ego view from a concept down to the chunks that mention it.
export interface ConceptPresence {
  document_id: string
  chunk_keys: string[]
  n_mentions: number
}

// Rebuild is a 202 + poll job (ADR-017 B1), mirroring /api/ingest. `graph_version` is set once the
// worker finishes; `message` carries the error text when `state === 'error'`.
export interface GraphRebuildStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  graph_version: string | null
  message: string | null
}
