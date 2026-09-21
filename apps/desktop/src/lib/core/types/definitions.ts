// TypeScript mirror of the desktop-API payloads (apps/api/models/definitions.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// A concept's definition is chosen from candidates that keep their source (ADR-053). Nothing
// overwrites anything: a passage from the library, the user's own words and (later) a model's text
// sit side by side, and the user picks one. `grade` + `reasons` are the evidence, never a model's
// rating of itself.

export interface DefinitionCandidate {
  id: string
  /** Verbatim for a passage (whitespace included — collapse it for display only). */
  text: string
  source: 'passage' | 'user' | 'model'
  status: 'suggested' | 'chosen' | 'dismissed'
  /** 'strong' | 'some' | 'thin' for a passage, 'user' for the user's own words. */
  grade: string
  reasons: string[]
  document_id: string | null
  document_title: string | null
  /** Opens the passage where it was written (the same path a chat citation takes). */
  chunk_key: string | null
}

export interface ConceptDefinitions {
  concept_id: string
  label: string
  chosen_id: string | null
  candidates: DefinitionCandidate[]
  can_undo: boolean
  /** Nothing chosen and no definition-shaped passage: the library has little to go on. */
  thin: boolean
  /** Any passage candidate on record — else the panel offers to look, not implies it looked. */
  extracted: boolean
}

/** One concept a label search found — the whole vocabulary, not only the graph. */
export interface VocabularyMatch {
  id: string
  label: string
  on_graph: boolean
  has_definition: boolean
}
