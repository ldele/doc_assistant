<!-- status: design-locked · created: 2026-07-23 · owner: Code · plan: docs/decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md (increment 2) -->

# Feature spec — Taxonomy increment 2a: the curation backend (read model + read/write API)

Build contract for **increment 2a** of [ADR-028](../decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md).
Increment 1 seeded the field trunk (23 divisions + 213 groups) and built the `knowledge/taxonomy.py`
write seam, but the taxonomy is **disconnected** — no concepts or documents attach to any field, and no
surface reads or edits it. This increment is the **serve + edit backend**: the read model a UI renders and
the HTTP write endpoints that drive the seam. The **frontend taxonomy view is 2b** (a separate PR — it
needs the running app + interacts with the app-shell placement track). Backend only, **$0, zero-LLM**.

## Why this slice
The graph shipped its read model first (PR-G1) before any renderer; the same order applies. 2a is
fork-free and verifiable on this box; 2b (the Svelte view) needs live-app verification and is where the
"dedicated taxonomy view, deep-linked like Manage-keywords" placement (ADR-019 D11) is realised.

## Scope
In: `knowledge/taxonomy_view.py` (read model), `apps/api/routers/taxonomy.py` (read + write routes),
payloads in `apps/api/models.py`, router registration, tests. Out: the Svelte view (2b), auto-propose
(increment 3, $0/Ollama, KI-4), coverage-based gap detectors (RG-015), the About/Settings CC-BY UI (T4).

## Design decisions
- **Attaching a concept to a field is an `in_field` hierarchy edge** (`concept →in_field→ domain`), written
  through the existing `taxonomy.add_hierarchy_edge` — no new primitive. So "attach concept" and "add a
  field→field edge" are the *same* endpoint (`POST /hierarchy`), disambiguated by the endpoint `kind`s.
- **Coverage uses set-semantics rollup (ADR-028 D6).** A field's coverage = the **distinct** set of
  concepts/documents for which it is an ancestor (itself or any narrower descendant), deduped by id. No
  fractional counts, no forced primary parent. Computed over the `load_taxonomy` DiGraph.
- **Honest zero-state.** Nothing is attached yet, so every rollup count is currently 0 — served as real
  zeros with a total, never hidden or faked (robustness contract; inform-don't-block).
- **Read-only vocabulary boundary holds (ADR-017 A1 / ADR-019 D11).** These routes edit the *taxonomy*
  (hierarchy edges + document_field), never the concept vocabulary itself — no concept create/rename/delete.

## Items

### V1 — `knowledge/taxonomy_view.py` (read model, pure over a session read)
- `load_taxonomy_view() -> TaxonomyView` — the field forest + coverage:
  - `fields: list[TaxonomyField]` — every `kind="domain"` node as `{id, code, label, parent_ids,
    child_ids, n_concepts_direct, n_documents_direct, n_concepts_rollup, n_documents_rollup}`. `code` is
    recovered from `source="anzsrc"` nodes via the seed map (or None for a hand-added field).
  - `roots: list[str]` — field ids with no `in_field` parent (the divisions).
  - `n_concepts_total`, `n_documents_total`, `n_unassigned_concepts` (concepts with no `in_field` edge).
  - **Rollup** = union over `descendants(field) ∪ {field}` of directly-attached concept/doc id sets, deduped.
- `load_field_detail(field_id) -> FieldDetail | None` — one field's directly-attached concepts
  (`[(id,label)]`) + documents (`[(id,title)]`) + its rollup counts; `None` if the id is not a domain node.
- Pure read (its own `session_scope`); reuses `taxonomy.load_taxonomy`. No LLM, no network.

### V2 — `apps/api/routers/taxonomy.py` (thin shell over `taxonomy.py` + the view)
- `GET /api/taxonomy` → the `TaxonomyView` payload. Empty `fields` (pre-seed) returns `200` with a total of
  0, not 404 — the trunk is bundled data, so "unseeded" is a legitimate empty state, not a missing artifact.
- `POST /api/taxonomy/hierarchy` `{source_id, target_id, type}` → add an edge. `201`; **`409` on a cycle**
  (`TaxonomyCycleError`); `400` on a bad `type`; `404` if either id is not a concept/domain row.
- `DELETE /api/taxonomy/hierarchy` `{source_id, target_id, type}` → remove an edge; `{removed: n}`.
- `POST /api/taxonomy/documents/{document_id}/fields/{field_id}` → `attach_document_field`; `404`/`400`
  (`NotADomainError`) when the field id is not a domain node; idempotent.
- New domain ⇒ new router module + `include_router` (apps/api/CLAUDE.md).

### V3 — payloads (`apps/api/models.py`) + `types.ts` mirror
`TaxonomyFieldPayload`, `TaxonomyViewPayload` (+ `from_view`), `FieldDetailPayload`, `HierarchyEdgeRequest`.
Mirror in `apps/desktop/src/lib/types.ts` (the wire contract) even though 2a ships no renderer — 2b consumes it.

## DoD / guard tests (each fails against today's code)
1. `GET /api/taxonomy` on the seeded corpus → 236 fields, 23 roots, `n_concepts_total=26`,
   `n_unassigned_concepts=26` (nothing attached yet), all rollups 0.
2. Attach a concept to a group (`POST /hierarchy` `in_field`) → that group's `n_concepts_direct=1`, and its
   **division's `n_concepts_rollup=1`** (rollup crosses the group→division edge).
3. A cycle-forming `POST /hierarchy` → `409`; a bad `type` → `400`.
4. `POST /hierarchy` with a non-existent id → `404` (no partial write).
5. `attach_document_field` to a non-domain target via the route → `400`; to a domain → `201`, idempotent.
6. `load_taxonomy_view` rollup dedups a concept sitting under two fields (counts once at a common ancestor).
7. Unit: `n_unassigned_concepts` counts only `kind="concept"` rows with no `in_field` edge.

## Gate
`ruff` / `ruff format` / `mypy --strict src` / `bandit` / `pytest` (unit + `tests/integration/test_api_taxonomy.py`
via the `create_app(controller=…)` seam — no model load, no network); `docs_check`/`integrity_check` 0/0. Live
`GET /api/taxonomy` against the seeded DB.

## Out of scope → 2b and beyond
- **2b — the Svelte taxonomy view**: render the forest, edit edges, attach concepts/docs; deep-linked like
  Manage-keywords (ADR-019 D11). Placement w.r.t. the app-shell nav track is settled there.
- **Increment 3 — auto-propose** `in_field` parents where a concept is unassigned ($0/Ollama, KI-4).
- Coverage-based gap detectors (RG-015); the CC-BY attribution UI (T4).
