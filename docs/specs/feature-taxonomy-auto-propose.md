<!-- status: design-locked · created: 2026-07-25 · owner: Code · plan: docs/decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md (increment 3) -->

# Feature spec — Taxonomy increment 3: auto-propose placements ($0/Ollama, propose-only)

Build contract for **increment 3** of [ADR-028](../decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md)
(its **Decision 8**, which is ADR-019 E1 — *"the LLM proposes only where the link IS NULL, never
overwriting"* — applied to `concept_hierarchy` and `document_field`).

Increments 1/2a/2b built the substrate (ANZSRC trunk, write seam), the read model + write API, and the
Svelte placement view. Placement is therefore possible but **entirely manual**: on the primary corpus
every concept and every document sits unplaced, so the coverage math ADR-028 D6 exists for reads 0
everywhere. This increment adds the first-pass filler: one quarantined local-LLM pass that **proposes**
an `in_field` parent for each unplaced concept and a field for each unclassified document, written as
`origin="proposed"` rows the user accepts or deletes.

## Why this slice

The manual surface must exist before an auto-fill has anywhere to land (it does — 2b), and a proposal
must be **distinguishable from a user's own edit** or "never auto-written" is a slogan rather than a
property. `document_field.origin` already ships that distinction; `concept_hierarchy` does not, so the
column is part of this increment, not a later polish. Quality is explicitly *not* claimed here — it is
RG-015-gated (ADR-028's own ⚠ "auto-propose placement will be accurate enough" is unmeasured), which is
exactly why nothing is written as curated.

## Scope

**In:** `concept_hierarchy.origin` (additive migration) · promote/never-demote semantics in
`knowledge/taxonomy.py` + the unplaced-set accessors · new `knowledge/taxonomy_propose.py` (the LLM
pass) · `scripts/propose_taxonomy.py` (dry-run-default runner) · `origin` surfaced through
`taxonomy_view` → API payloads → `types.ts` · tests.
**Out:** the UI badge + accept/reject affordance on proposals (**increment 3b**, frontend, needs the
live app) · coverage-based gap detectors (RG-015) · MeSH/ACM subtree grafting (ADR-028 D7) · the CC-BY
attribution UI (T4, already partly landed in `AboutDialog`) · `is_a` proposals (this pass proposes
`in_field` only — a concept→broader-concept chain is a different judgment and has no seeded candidates).

## Design decisions

- **A proposal is an ordinary row with `origin="proposed"`**, not a separate proposal table. Reason: the
  distinction the ADR needs is *provenance of one link*, and `DocumentField.origin` already established
  exactly this vocabulary (`"curated"` wins, `"proposed"` is overridable). A parallel table would double
  the read model and let the two drift.
- **Curation always wins, and accepting a proposal is just a curated write of the same edge.**
  `add_hierarchy_edge(..., origin="curated")` over an existing `proposed` row **promotes** it in place
  (the accept primitive — no new endpoint); over an existing `curated` row it is a no-op; a `proposed`
  write never demotes a `curated` row. Mirrors `attach_document_field`'s existing rule.
- **Two-stage narrowing: division (23) → group within that division (~10).** Reason: ANZSRC is a
  2-level trunk and the candidate list is 236 fields; a small local model chooses far better from ~23
  then ~10 labelled options than from 236 at once, and the intermediate answer is *itself* a valid
  placement target (both levels are `kind="domain"` nodes). If stage 2 abstains or fails to parse, the
  **division-level placement stands** — coarse but honest, and the rollup makes it useful.
- **Abstention is a first-class outcome.** Both stages may answer "none"; an abstaining item gets **no
  proposal** rather than a forced field. A wrong placement costs more than a missing one (it inflates
  coverage, which is the number this layer exists to make trustworthy).
- **The candidate lists are read from the DB, never from the ANZSRC file.** The pass works on whatever
  domain nodes exist (hand-added fields included) and degrades to "no candidates ⇒ no calls" on an
  unseeded DB.
- **Scope defaults to the graph vocabulary (`graph_include=1`), with `--all-concepts` to widen.**
  Reason: ADR-018 made `graph_include` the boundary between the curated concept map and the (breadth-first)
  keyword families, and the taxonomy augments the *graph* (ADR-019 D1). On a box carrying the 2026-07-05
  promotion flood that is 13 concepts rather than 357 — but the skipped count is **always printed**, never
  silently dropped.
- **Confinement mirrors `gap_suggest` / Node B:** the module takes an already-built `LLMClient` and makes
  no provider decision; it never writes, never creates a `Concept`, and a per-item transport/parse failure
  is logged and skipped instead of sinking the run. **Zero LLM calls without `--apply`** (the `build_gaps`
  polarity — and the reason a dry run cannot silently bill: `assert_provider_intent` no-ops when
  `apply=False`, so "no `--apply`" must mean "no calls").

## Items

### P1 — `concept_hierarchy.origin` (schema)
- `ConceptHierarchy.origin: str` — `"curated"` | `"proposed"`, non-null, default `"curated"`.
- `db/migrations.py` `_ADDITIVE_COLUMNS` += `("concept_hierarchy", "origin", "VARCHAR NOT NULL DEFAULT
  'curated'", None)` — the literal DEFAULT backfills existing rows in the same ALTER (KI-25 discipline);
  every pre-increment edge is a seed/user edge, i.e. curated. Unindexed: read with the row, never filtered on.

### P2 — `knowledge/taxonomy.py` (seam)
- `HIERARCHY_ORIGINS = frozenset({"curated", "proposed"})`; `add_hierarchy_edge(..., origin="curated")`
  validates it, and on an existing row: `proposed` + curated write → **promote** (flip to `curated`);
  otherwise return untouched.
- `unplaced_concepts(session, *, graph_only=True) -> list[Concept]` — `kind="concept"` rows with **no**
  `in_field` edge (any origin), through `presence_nodes`' kind guard; `graph_only` filters `graph_include`.
- `unclassified_documents(session) -> list[Document]` — documents with no `document_field` row.
- Both return the honest empty list at 0 docs / 0 concepts (robustness contract).

### P3 — `knowledge/taxonomy_propose.py` (the pass, quarantined)
- `division_candidates(graph)` / `group_candidates(graph, division_id)` — pure reads over the
  `load_taxonomy` DiGraph (`in_field` edges between `kind="domain"` nodes; a division = a domain with no
  domain parent).
- `build_choice_messages(item, candidates, *, level)` → the numbered-options JSON prompt;
  `parse_choice(text, n_candidates)` → `Choice(index | None, confidence)`, or `None` when unparseable
  (tolerant of a code fence, a bare integer, `{"choice": 3}`, `"none"`; out-of-range index ⇒ unparseable).
- `propose_placements(items, graph, client, *, temperature=0.0, max_tokens=256) -> list[PlacementProposal]`
  where `items: Sequence[ProposalItem]` (`kind` ∈ {`concept`, `document`}, `id`, `label`, `context`).
  Returns `PlacementProposal(item_kind, item_id, item_label, field_id, field_label, division_id,
  division_label, confidence, evidence)`; `evidence` records the exact LLM inputs + both stage answers
  (ADR-004's "expose the LLM inputs, rate the output" mandate). Empty `items` or no divisions ⇒ `[]`,
  checked **before** any call.

### P4 — `scripts/propose_taxonomy.py` (runner)
- `python -m scripts.propose_taxonomy` → dry run: report the unplaced/unclassified counts + the call
  budget, **zero LLM calls**, zero writes.
- `--apply` → run the pass and write `origin="proposed"` rows (`add_hierarchy_edge(origin="proposed")` /
  `attach_document_field(origin="proposed")`). `--concepts-only` / `--documents-only`, `--limit N`,
  `--all-concepts`, `--provider`/`--model` (default `TAXONOMY_PROPOSE_LLM_PROVIDER=ollama` /
  `_MODEL=qwen3.5:9b` — was `llama3.1:8b` until RG-015 measured the three local instruments on the
  97-doc corpus, KI-4) routed through `llm.assert_provider_intent` **before** any client exists.
- Concept context = up to 3 titles of documents the concept is present in (`concept_presence`, when the
  skeleton has been built) + its aliases + definition; document context = authors + year.
- Prints one line per proposal (`item → division / group  conf`) and the abstain/skip counts.

### P5 — surface the origin (read model → wire)
- `TaxonomyField` += `n_concepts_proposed` / `n_documents_proposed` (direct, proposed-origin only);
  `FieldDetail`'s member tuples += `origin`; `FieldMemberPayload` += `origin`; `types.ts` mirrors both.
  Reason: 3b's badge needs it, and until then the API must not present a machine guess as a user edit.

## DoD / guard tests (each fails against today's code)
1. **Migration:** a `concept_hierarchy` predating `origin` gains it and every existing row reads
   `"curated"` (asserted absent first — non-vacuous).
2. **Promotion:** `add_hierarchy_edge(origin="proposed")` then `origin="curated"` on the same triple →
   one row, `origin="curated"`. Reverse order → stays `"curated"` (no demotion). No duplicate rows either way.
3. **Unplaced sets:** a concept with an `in_field` edge is excluded; a `kind="domain"` node is never
   returned; `graph_only=True` excludes `graph_include=0`; a document with a `document_field` row is excluded.
4. **Two-stage placement (fake client):** a scripted client answering `{"choice": 2}` then `{"choice": 1}`
   yields a proposal whose `field_id` is the **group**, with the division recorded in `evidence`.
5. **Stage-2 fallback:** stage 1 answers a division, stage 2 answers `"none"` → the proposal's `field_id`
   is the **division** (coarse placement stands).
6. **Stage-1 abstain / unparseable:** `"none"` or garbage at stage 1 → **no proposal** for that item, and
   the run continues to the next item (one bad item never sinks the batch).
7. **Zero-state:** empty `items`, or a graph with no domain nodes, → `[]` with **zero** `client.complete`
   calls (asserted on a call-counting fake).
8. **Transport failure** on one item is logged and skipped; the other items still produce proposals.
9. **API:** a proposed concept attachment shows `origin="proposed"` in `GET /api/taxonomy/fields/{id}`
   and in the field's `n_concepts_proposed`; a curated one shows `"curated"`.

## Gate
`ruff` / `ruff format` / `just typecheck` / `bandit` / full `pytest` (unit + the taxonomy API
integration file) / `svelte-check` (types-only change) / `docs_check --strict` / `integrity_check`.
**Live, $0:** on this box the taxonomy substrate has never been migrated or seeded (its DB predates
increment 1 — 47 docs / 357 concepts / 13 in the graph vocabulary), so the live run is: back up
`library.db` → boot migration + `seed_taxonomy --apply` → `propose_taxonomy` dry run → `--apply` with
local Ollama → verify the proposals through `GET /api/taxonomy` + one field detail.

## Out of scope → 3b and beyond
- **3b (frontend):** proposal badge + accept/reject in `LibraryTaxonomy.svelte` (accept = the existing
  curated `POST /hierarchy`; reject = the existing `DELETE`), and an "unplaced" queue to work through.
- **RG-015:** measure placement precision on a sample before any coverage-based gap detector trusts it.
- `is_a` proposals; MeSH/ACM grafting; per-document *multi*-field proposals (this pass proposes one).
