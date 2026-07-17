# Spec — Tag families (keyword synonym-collapse over the concept vocabulary)

**Status:** **PR-1 + PR-2 BUILT 2026-07-17 (staged, not committed)**; PR-3 still DESIGN-LOCKED (grilled
2026-07-16, `grill-me`; ledger at foot; parked — not scheduled). Architectural decisions in **ADR-015**.
Collapses near-duplicate keywords (`llm`/`llms`, `connectome`/`connectomics`) into user-curated **families** so
the Library keyword filter treats each family as one entry. *Takes advantage of* the curated
`Concept`/`ConceptAlias` vocabulary — it is **not** the concept-graph/epistemics UI (a separate later track
over the same rows; ADR-015 §C).

**Owner:** Claude Code. **Three PRs, never bundle:** ~~PR-1 families end-to-end (manual)~~ **BUILT** →
~~PR-2 detection~~ **BUILT** → PR-3 LLM assist (parked).

---

## The decision (user, 2026-07-16)

A family = a canonical tag with several member keywords the user can adjust. Filtering a family returns every
doc carrying **any** member (union). Families are curated in a dedicated **"Manage keywords" view**; a
deterministic **detection** pass proposes them; an optional LLM pass comes later. See ADR-015 for the
grouping-vs-presence and family↔graph rationale.

## Grounding (read from the code, 2026-07-16 — not assumed)

- **The vocabulary + primitives exist.** `db/models.py`: `Concept` (`label`, `source` `"keyword"|"manual"`,
  `definition`, `folder_id`) + `ConceptAlias` (`concept_id`, `alias`, unique per pair).
  `concept_skeleton.py`: `promote_keyword(name)` (Keyword → Concept + seed alias), `add_concept(label,
  definition, aliases)`, `load_glossary()` (label + definition + aliases), `load_concepts()` (`[(id, label)]`
  + `{id: [alias]}`), `list_keyword_candidates()`. All zero-LLM, idempotent. **No new schema, no migration.**
- **An alias is a free-text string** — nothing structurally links `ConceptAlias` to a `Keyword` row; the
  family↔keyword mapping is by **string match** (alias casefold == keyword name casefold). This is exactly
  what keyword-name grouping needs.
- **The filter already ships the data it needs.** `LibraryDocument.keywords` is client-side; the overlay's
  `keywordFacets`/`facetFilter` (`lib/library.ts`) compute counts/availability over it. Family-awareness is a
  pre-facet grouping step, not a new data path.
- **Nothing in the frontend touches concepts yet** — this is the first client surface (ADR-015).
- **Corpus state:** ~60 keywords surface on this box; **0 `Concept` rows** (the vocabulary was never seeded
  here). → PR-1 must be demoable by *creating* a family by hand (not by assuming existing concepts).

## Data model (no new tables)

A **family** is a `Concept` whose `ConceptAlias` rows carry the member `Keyword` names (the canonical `label`
is also an implicit member). A `Keyword` belongs to at most one family (enforced in the Manage view — assigning
a keyword already in another family moves it). Un-familied keywords remain standalone facet units.

**Filter grouping (client-side, pure):** build `keyword_name → canonical_label` from the family map; collapse
each doc's `keywords` to their canonical (or themselves), dedupe; `keywordFacets`/`facetFilter` then operate on
these units. A family's count = union of member docs; grey-out/AND semantics are unchanged (they apply to the
unit). Extends the existing pure helpers; no backend filter change.

## Carve

### PR-1 — families end-to-end, manual (build first) — ✅ BUILT 2026-07-17 (staged)

The whole mechanism with hand-curation; no detection yet. Demoable: create `large language models`, assign
`llm`/`llms`, watch the overlay collapse them and filter the union.

- **Backend (thin shells over `concept_skeleton`/`library`, host-only writes per KI-5):**
  - `GET /api/library/keyword-families` → `[{ id, canonical, aliases: [str], doc_count }]` (aliases = member
    keyword names; `doc_count` = union over `document_keywords`).
  - Family CRUD: create (`add_concept`), rename canonical, add/remove member keyword (alias add/remove),
    delete. New `library.py` functions + `concept_skeleton` helpers where a primitive is missing (e.g.
    remove-alias, delete-concept, list-with-doc-counts).
  - Wire models in `apps/api/models.py`; `types.ts` mirror.
- **Frontend:**
  - `LibraryManageKeywords.svelte` (new) — a view/modal listing the keyword pool + current families; create a
    family, rename, add/remove members, split/merge, delete. Opened from a "Manage keywords…" link in the
    filter overlay.
  - `lib/library.ts` — a family-aware grouping (`keyword_name → canonical`) feeding `keywordFacets`/
    `facetFilter`; the overlay + strip render a family as an **atomic entry** (canonical + "N forms" with the
    aliases visible on hover/subtitle, union count). Un-familied keywords unchanged.
- **DoD:** `svelte-check` 0; `ruff`/`ruff format`/`mypy --strict src`/`bandit`; pytest for the new library/API
  functions (incl. the string-match grouping + move-on-reassign + union-count cases); default path (no
  families) byte-identical to today's overlay; both themes; mobile no-overflow; a11y (dialog, labels);
  preview-harness `$0` — create a family live, confirm the overlay collapses `llm`/`llms` into one entry that
  filters the union; DEVLOG + ui-checklist + this spec's status flip with SHA.

### PR-2 — detection — ✅ BUILT 2026-07-17 (staged)

- **`src/` pure core** `keyword_families.py` (or extend `keywords.py`): `Keyword` list → proposed families via
  **Tier 1 morphological** (casefold + conservative singularizer/suffix-normalizer; `llms`→`llm`,
  `connectomes`→`connectome`) + **Tier 2 `bge` embedding** cosine clustering (semantic near-synonyms;
  `connectome`≈`connectomics`), with edit-distance as a supporting signal. Deterministic, `$0`, no LLM.
- **Surface:** `POST /api/library/keyword-families/detect` → proposals (grouped candidates, per-tier
  confidence) the Manage view reviews (accept/edit/reject → PR-1's CRUD); plus `scripts/detect_keyword_families.py`
  (dry-run default) for host batch (house enrichment pattern).
- **DoD:** unit tests on the pure tiers (toy keyword sets, incl. the two example cases); the embedding tier
  behind the API's loaded `bge` (no new model load); nothing auto-applies; preview-harness — Detect surfaces
  `llm`/`llms` (Tier 1) and `connectome`/`connectomics` (Tier 2) as reviewable proposals.
- **Built as:** a new `keyword_families.py` (not an extension of `keywords.py` — kept the extraction module's
  concern separate, matching the `concept_semantics.py`/`concept_curation.py` split precedent). Tier 2 groups
  transitively via union-find (a chain proposes one family, not overlapping pairs) rather than returning flat
  pairs. `library.detect_family_candidates(embed_fn, embedding_threshold)` is the impure boundary (loads
  keyword names, subtracts already-familied ones, injects `embed_fn`). The API route wraps
  `controller.rag.embeddings.embed_documents` to satisfy "no new model load"; the CLI script has no `--apply` at
  all (report-only, per the DoD). Verified live on the real 76-doc corpus (both the CLI and the app's Detect
  button found the same proposal, `pvpo`≈`avpv pvpo` @ 0.77 confidence) — full details in `docs/DEVLOG.md`.

### PR-3 — LLM confirm/merge pass (parked)

An optional gated LLM step to confirm noisy Tier-2 proposals or suggest merges. Deferred; prove on Ollama
first (KI-4). Not scheduled.

## Decisions (grill ledger, 2026-07-16 — full rationale in ADR-015)

| # | Decision | Reason / reopens if |
|---|---|---|
| T1 | Family = `Concept` + `ConceptAlias`, aliases = keyword names | Take advantage of the vocabulary; not the graph UI |
| T2 | **Keyword-name grouping** (union of `document_keywords`) | Light consumer; reuses shipped data. Reopens if presence-recall is needed (graph's job) |
| T3 | Detection = **tiered morphological + `bge` embedding** (edit-distance support), no auto-apply | Examples span plural + derivational; `bge` already loaded, $0 |
| T5 | Curation in a dedicated **"Manage keywords" view** | Curation ≠ filtering; room for proposals + editing + future graph curation |
| T6 | Overlay renders a family as an **atomic entry** (canonical + N forms, union count) | It *is* the synonym-collapse goal; per-alias selection is over-engineering |
| T7 | Read + CRUD + detect endpoints (thin shells); docs payload unchanged | Mechanical |
| T8 | **Foundation-first** carve: PR-1 manual → PR-2 detection → PR-3 LLM | De-risks the model + overlay before detection |

## Parked (reopen triggers)

- **LLM assist** → PR-3 (prove on Ollama first — KI-4).
- **Presence-recall** (docs mentioning a term the extractor never tagged) → closes when the concept-graph
  presence layer gets its UI.
- **User-pinned favorites** → this work carries them as *promote-to-`Concept`* (keyword-overlay grill).
- **Concept-graph + epistemics UI** → separate track, same vocabulary; explicitly **not** built here.

## Grill ledger

Grilled 2026-07-16 (`grill-me`), 8 branches resolved, 4 parked. Walked data-mechanism → filter-target →
curation-home → detection → overlay-render → wire → LLM → carve in dependency order. Grounded in the live
`concept_skeleton`/`Concept` code (promote/add/glossary primitives, string-match aliases, 0 concepts on the
corpus). User steer that shaped it: *"not re-using but taking advantage of"* + the concept-graph/epistemics
UI is a separate later track. Decisions routed to ADR-015 (architecture) + this spec (contract).
