# Spec — Tag families (keyword synonym-collapse over the concept vocabulary)

**Status:** **PR-1 ✅ SHIPPED `0c3b0d4` · PR-2 ✅ SHIPPED `0af43db`** (both committed 2026-07-17) ·
**PR-2.5 · PR-2.6 · PR-2.7 ✅ ALL BUILT 2026-07-20** (D1–D6 + F1–F4).
**PR-3 (LLM assist) still parked** from a post-commit review — see the
carve below; both are **defect-driven, not new scope**. PR-3 still DESIGN-LOCKED (grilled 2026-07-16,
`grill-me`; ledger at foot; parked — not scheduled). Architectural decisions in **ADR-015**.
Collapses near-duplicate keywords (`llm`/`llms`, `connectome`/`connectomics`) into user-curated **families** so
the Library keyword filter treats each family as one entry. *Takes advantage of* the curated
`Concept`/`ConceptAlias` vocabulary — it is **not** the concept-graph/epistemics UI (a separate later track
over the same rows; ADR-015 §C).

**Owner:** Claude Code. **Never bundle:** ~~PR-1 families end-to-end (manual)~~ **SHIPPED** →
~~PR-2 detection~~ **SHIPPED** → ~~PR-2.5 hardening~~ **BUILT** → ~~PR-2.6 family-aware tiles~~ **BUILT** →
~~PR-2.7 Manage view at scale~~ **BUILT** → PR-3 LLM assist (parked).

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

### PR-1 — families end-to-end, manual (build first) — ✅ SHIPPED 2026-07-17 `0c3b0d4`

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

### PR-2 — detection — ✅ SHIPPED 2026-07-17 `0af43db`

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

### PR-2.5 — hardening the write paths (defect-driven) — ✅ BUILT 2026-07-20 (staged)

Post-commit review of `0c3b0d4`+`0af43db` (agent review + live drive on the real 76-doc corpus). The read
path is sound — facet math, union-find determinism, thin-shell discipline, and the no-families default path
all reviewed **clean**, and the feature demonstrably works live (family `Large language model` → 14 docs,
union correct, AND recount correct, Detect reproduced `pvpo`≈`avpv pvpo` @ 0.77). The defects are all in the
**under-guarded write paths**, and none were caught by the 977 passing tests.

**Why this jumps the queue:** D1/D2 sit on the *natural* post-PR-2 flow. `_canonical_and_members` always
proposes an **existing keyword** as canonical (`llm`), so Detect → Accept → **Rename** (to get the spec's own
demo label "large language models") is the obvious next click — and that click is currently a data-corruption
trap whose blast radius escapes the feature.

| # | Defect | Where | Repro (verified, not inferred) |
|---|--------|-------|--------------------------------|
| **D1** | Rename onto an existing canonical creates **duplicate `Concept` labels** → create route 500s forever (no UI recovery) **and** `promote_keyword` throws `MultipleResultsFound` repo-wide (breaks `scripts/seed_concepts.py`). `Concept.label` has no unique constraint (`db/models.py:441`); `rename_concept` explicitly defers the check to callers and **no caller does**. | `concept_skeleton.py:948` · `library.py:529` · `main.py:514` (catches only `ValueError`) | create `llm` → create `vector search` → PATCH `vector search`→`llm` = 200 → `GET` lists `['llm','llm']` → next create `llm` = **HTTP 500** |
| **D2** | Rename **silently drops the family's own canonical keyword**. `create_keyword_family` → `add_concept` never seeds an alias for the label (unlike `promote_keyword`), so the canonical is only *implicit*; rename re-points it and the original keyword falls out and reappears as a standalone chip — the exact duplicate the feature exists to remove. | `library.py:508,529` · `_build_family` `:478` | `create("llm",["llms"])` → `doc_count=3` → `rename(id,"large language models")` → aliases=`['llms']`, **`doc_count=2`**, `llm` gone |
| **D3** | A keyword can belong to **two families**. `add_family_member` defers the canonical-collision case to the Manage view; the view only guards the *pick-list* path — "New family" canonical is unchecked free text. Downstream `familyCanonicalMap` (`library.ts:249`) resolves order-dependently, so the overlay, its tooltip, and the Manage view show **three inconsistent numbers**. | `library.py:541` · `LibraryManageKeywords.svelte:62` | `create("large language models",["llm","llms"])` then `create("llm",[])` → two families claim `llm` |
| **D4** | Stemmer's sibilant-`es` rule (`w[:-2]`) **over-strips**, so common plurals never match structurally. False *negatives* only (conservatism claim holds; **coverage** claim does not) — they silently degrade from a `confidence=1.0` structural match to a threshold-dependent Tier-2 fuzzy one. Existing tests pick `boxes`/`taxonomies` — the two inputs the rules get right. | `keyword_families.py:41-45` | verified via real `_stem`: `database/databas`, `size/siz`, `cache/cach`, `response/respons`, `analysis/analys` all **MISS**; `detect_family_proposals(["database","databases"])` → `[]` |
| **D5** | A live keyword selection **isn't remapped when families change** → grid goes empty while the stale chip still looks selectable (the universe includes `selected` by design). The Manage view is opened *from* the overlay, i.e. exactly where a selection is live. | `App.svelte:670` (`createFamily`) / `:702` (`deleteFamily`), neither touches `libraryKeywords` (`:126`) | select `llm` (2 docs) → Manage → create family with `llm`+`llms` → `facetFilter` returns `[]`, grid empty |

- **Fixes:** D1 → reject a colliding label in `library.rename_keyword_family` (**409**, not in the view — the
  invariant belongs at the library boundary; map it in the API shell). D2 → seed the canonical as a real alias
  on create (align `create_keyword_family` with `promote_keyword`'s alias-seeding), or make rename carry the
  old label into the alias set — **pick one and note the migration for the 26 pre-existing concepts on this
  box**. D3 → route the free-text canonical through the same move-on-reassign guard. D4 → only strip `es` when
  `w[:-2]` ends in a sibilant, else `w[:-1]` (or try both stems). D5 → map `libraryKeywords` through the new
  canonical map after each family write.
- **DoD:** the five repros above become **regression tests first** (they all pass today — that's the point);
  `test_rename_family_canonical` (`tests/integration/test_keyword_families.py:~199`) extended to assert
  `aliases` + `doc_count`, not just `canonical` (asserting only `canonical` is why D2 hid); stemmer tests gain
  the `-se`/`-ze`/`-che`/`-ie` class; first tests for `familyCanonicalMap`/`familyUnitsOf`/`facetFilter`
  composition (the grouping layer is currently **entirely unexercised**). Then the standard gates
  (`svelte-check` 0 · ruff/format · `mypy --strict src` · bandit · full suite) + a live $0 drive of
  Detect→Accept→Rename on the real corpus. No new scope, no ADR, no locked-setting touch.

### PR-2.6 — family-aware grid tiles (frontend-only; carries defect D6) — ✅ BUILT 2026-07-20 (staged)

Carved **with** D6 rather than into PR-2.5: both are the same root cause in the same file — `LibraryGrid`
never learned about families — and splitting them would touch it twice.

- **D6 (verified live):** selecting a **family** highlights **nothing** on the tiles. `LibraryGrid.svelte:129`
  renders raw `d.keywords` and matches `activeKeywords.includes(k)`; with a family selected `activeKeywords`
  holds the canonical (`Large language model`) while tiles hold raw forms (`llm`/`llms`), so the match can
  never fire. Measured on the real corpus: **family → 0 of 25 chips highlighted; control keyword `pretrained`
  → 19 highlighted.** `orderedKeywords` (`:47-51`) breaks identically — `active.length === 0` returns
  `d.keywords` unchanged, so the matching form isn't floated to the front and can hide behind `+N`.
- **Display:** tiles render a family as its **atomic canonical chip** (`Connectome`, not
  `connectomes`+`connectome`+`connectomics`), consistent with the overlay's T6 atomic-entry rule. Two wins
  beyond the fix: the tile's scarce keyword budget (`CHIP_CAP`, 2 reserved lines, `+N` overflow) stops being
  spent on duplicate forms of one concept, and casing stops disagreeing with the filter (tile `imagenet` vs
  overlay `ImageNet` — the vocabulary already supplies the display name).
- **Reuses the shipped pure helper** (`familyCanonicalOf`/`keywordsOf`) — presentation only, no new data path,
  `$0`, no backend change. **DoD:** family selection highlights + floats its members; default path (no
  families) byte-identical; `svelte-check` 0; both themes; mobile no-overflow; live $0 verify.

### PR-2.7 — Manage view at scale (frontend-only) — ✅ BUILT 2026-07-20 (staged)

User reviewed the shipped Manage view and the filter overlay and raised three things; all three are real and
all are presentation. **Carved as its own PR** (not folded into PR-2.5) to keep hardening purely about data
correctness — but note **F2 below *is* the D3 fix**, so the two PRs must not both implement it: PR-2.5 owns
the `library.py` boundary guard (the invariant), PR-2.7 owns the control that stops the user reaching it.

**Grounding measured on the live corpus 2026-07-17 (not assumed):** 60 keywords surface; **30 of them (exactly
50%) sit on a single document**. `FAMILIES (26)` is likewise misleading — only ~6 are real families; the rest
are 0-member concepts inherited from the earlier concept-graph seeding, several with **0 docs** (`BERT`).

| # | Finding | Fix |
|---|---------|-----|
| **F1** | **"Manage keywords…" is rendered inside the scrolling pane**, so it reads as the last list item and is lost under the scrollbar. | Move it to the overlay header (beside the title/search) or a fixed footer bar **outside** the scroll region. |
| **F2** | **"New family" canonical is unchecked free text** → no navigation to existing families at scale, and it's the D3 hole. | **Autocomplete the canonical against existing families.** One control = the user's navigation ask **and** the D3 guard. Typing an existing canonical offers *navigate to that family*, not *create a duplicate*. |
| **F3** | **Flat pools don't scale** — a 38-chip keyword pool and a flat `FAMILIES (26)` list are already awkward; at ~100 families they're unusable. | Searchable/filterable picker for both; hide the 0-member/0-doc inherited concepts from the families list (they are vocabulary, not families). |
| **F4** | **The 1-doc long tail is 50% of the list** and is where every ugly string lives. | **Demote, don't delete** (below). |

**F4 — the long tail, and why demotion is the right verb.** Traced every odd keyword back to its source doc:
they are **mostly real specialist vocabulary, not junk** — `16p11` is **16p11.2** (autism CNV) truncated at
the dot; `c57bl` is the **C57BL/6** mouse strain truncated at the slash (7 docs); `va1v`/`dl5`/`osns`/`upns`/
`mgns` are Drosophila antennal-lobe glomeruli and neuron classes, all from **one** olfactory paper;
`avpv`/`pvpo`/`mpoa` are hypothalamic nuclei from one mating paper; `rabv` = rabies-virus tracing. The truly
broken ones are a small, **clustered** minority: `mathrm` (LaTeX `\mathrm` leak), `professium`, `outflux` —
**all three from the same 1952 scanned Hodgkin-Huxley paper** — plus `102ff` (a "p. 102ff" citation
artifact), `fne-tune` (OCR of "fine-tune"), `neurosc` (truncation).

- **Principle:** a facet exists to *partition* a set; a 1-doc facet doesn't partition — selecting it yields
  one document, which search already does better. So rare keywords have near-zero **filtering** value
  regardless of whether they're junk or gold. That is a principled threshold that sweeps up the ugly strings
  for free, **without** having to classify them.
- **Build:** collapse keywords below a doc-count threshold (default **< 2 docs**) behind a
  **"Show rare (N)"** toggle in the overlay + Manage view. **The overlay's existing search box is the escape
  hatch** — a specialist still finds `va1v` by typing it. Nothing is destroyed; the tail is demoted and
  instantly reversible. Threshold is a **display default, not a locked retrieval setting** (no eval-harness
  gate — it governs presentation, not retrieval).
- **DoD:** `svelte-check` 0; default path unchanged for corpora with no rare tail; the toggle round-trips;
  search finds a demoted keyword; `$0` live verify on the real corpus (expect 60 → **30** default-visible);
  both themes; mobile no-overflow.

**Rejected:** (a) *deleting the rare tail* — it's mostly real vocabulary, and delete isn't reversible-by-
search. (b) *modelling suppression as a "hidden" family* (a `Concept` whose aliases are the junk) — abuses
ADR-015's model, which defines a family as **synonym collapse**; overloading it would corrupt the meaning of
every consumer that reads those rows. Suppression gets its own store (see the checklist row). (c) *folding F2
into PR-2.5* — the invariant belongs at the `library.py` boundary regardless; a view-level control is a
convenience on top, not a substitute (a second client would bypass it).

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

#### PR-2.5 — as built (2026-07-20)

All five defects fixed; the five repros are regression tests that **fail against the shipped code**.

| # | Fix, as built | Note |
|---|---------------|------|
| **D1** | `library.rename_keyword_family` raises `KeywordFamilyExists` (a `ValueError` subclass) on a **case-insensitive** collision; the API shell maps it to **409**. | Case-insensitive because the client's `familyCanonicalMap` lowercases its keys — two families differing only by case would collide there anyway. Renaming to your own label, or only changing its case, still works. |
| **D2** | Rename carries the **old label into the alias set** before re-pointing it. | The spec offered "seed the canonical as an alias on create" as the alternative. This one was chosen because it needs **no migration** for the 26 pre-existing concepts on this box: the label stays an *implicit* member exactly as `_build_family` already treats it, so nothing about existing rows changes. |
| **D3** | `create_keyword_family` routes its canonical through `add_family_member` before the members. | Reuses the move-on-reassign guard instead of restating it; being the label, the call adds no self-alias and only detaches the name from other families. |
| **D4** | `_stem` → `_stem_candidates(word) -> frozenset[str]`; Tier 1 groups on a **non-empty intersection**, via union-find (a name can now bridge buckets). | The `-es` plural is structurally ambiguous — `boxes`→`box` but `databases`→`database`, and both stems end in a sibilant — so no single-stem rule can be right. Emitting both trades an implausible false *positive* (a real keyword equal to an over-stripped stem, e.g. `cas` beside `cases`) for the silent false *negative* that is not reviewable at all. |
| **D5** | New pure `remapSelection(selected, canonicalOf, documents)` in `lib/library.ts`, called from `refreshFamilies` after every family write. | Maps the selection through the new canonical map, then drops units no document carries any more — covering both directions (create re-points, delete drops). Mapped against the **whole** library so an out-of-collection selection stays removable. |

**Frontend tests now exist** (`apps/desktop/src/lib/library.test.ts`, 10 tests, `npm test`): the grouping
layer the spec called "entirely unexercised". Runner is node's built-in `node:test` with native TS
stripping — **zero new dependencies**; test files are excluded from `tsconfig.json` so the app config
needn't carry `@types/node` + `allowImportingTsExtensions` for test-only imports.

**Sharp edge, unchanged but now reachable one more way:** move-on-reassign is **not undoable** —
detaching a keyword from another family deletes that `ConceptAlias` row, and deleting the new family
does not restore it. D3 extends that to the *canonical*, so naming a new family after a keyword
already claimed elsewhere silently strips it from the other family. That is ADR-015's stated
"a keyword belongs to at most one family", not a new behaviour, but it is worth knowing before
curating in bulk.

#### PR-2.6 — as built (2026-07-20)

`LibraryGrid` gains one optional prop, `keywordsOf`, defaulting to `(d) => d.keywords` — that
default is what makes the no-families path byte-identical. `App` passes the accessor it already
derives for the overlay (`familyUnitsOf(familyCanonicalMap(keywordFamilies))`), so tiles and facets
finally agree on what a unit *is*: **one data path, two renderers**, no new API and no backend
change.

Ordering moved out of the component into a pure `orderedUnits(units, active)` in `lib/library.ts`,
so the half of D6 that is easiest to get wrong is unit-tested rather than only eyeballed. The `+N`
overflow count and its tooltip now count **units**, not raw keywords — otherwise a tile holding
`llm`+`llms` would claim one more chip than it renders, and the family collapse would not actually
free the tile's chip budget.

Svelte 5 note: the `{@const}` binding the tile's units has to sit as the immediate child of
`{#each}` (it is a block-scoped declaration), not inside the `<span class="kws">` where it reads
most naturally.

**Live on the real 76-doc corpus ($0, no LLM)** — probe family `Pretrained model`
(`pretrained` + `huggingface`, both previously un-familied so the probe was exactly reversible;
deleted afterwards, 26 concepts / 17 aliases before and after):

| | Before (spec measurement) | After |
|---|---|---|
| Family selected → chips highlighted | **0 of 25** | **22 of 22** |
| Family selected → active chip floated first | not floated | **22 of 22 tiles** |
| Plain keyword (`cajal`, control) | 9 | **9 of 9**, floated — default path unchanged |

Tiles render the family as its atomic canonical chip (`Pretrained model`, not
`pretrained`+`huggingface`), so the vocabulary's casing now drives the tile too. Dark theme at
375 px: **0 px** horizontal overflow, active chip visually distinct (filled indigo vs translucent),
0 console errors.

#### PR-2.7 — as built (2026-07-20), and where the spec's grounding was wrong

| # | As built | Note |
|---|----------|------|
| **F1** | **Already satisfied — nothing to change.** `.kwlistfoot` is a flex sibling of the scrolling `.kwlist`, so "Manage keywords…" is already a pinned footer outside the scroll region. Verified live in the running app. | PR-1 (`0c3b0d4`) landed the same day the feedback was taken; the complaint predates it. Recorded rather than "fixed" so the next reader doesn't go looking. |
| **F2** | Typing a canonical that already exists flips **Create → "Go to family"** and shows a warning naming the existing family's doc count; a partial match offers up to 5 existing families as one-click navigation, which scrolls to and outlines that row. | The user's navigation ask and the D3 guard are the same control, as the spec predicted. The invariant still lives at the `library.py` boundary (PR-2.5) — this only stops the user reaching it. |
| **F3** | Search box over the keyword pool **and** the families list; the families heading now states an honest split. | See the correction below. |
| **F4** | 1-doc keywords collapsed behind **"Show rare (N)"** in both the overlay and the Manage pool; a **selected** facet is never demoted; **search bypasses the split entirely**, so a specialist typing `va1v` still finds it. | Threshold is `RARE_MAX_DOCS = 1` — presentation only, no eval gate. |

**The spec's F3 grounding was wrong, and the live data corrected it.** It expected "`FAMILIES (26)` is
misleading — only ~6 are real families; the rest are 0-member concepts … several with 0 docs".
Measured on the real corpus:

- **12** have ≥1 alias — genuine synonym collapses (`Connectome`, `Large language model`, …)
- **10** have 0 aliases but **>0 documents** (`ImageNet` 10, `Tractography` 10, `BM25` 6) — not
  synonym collapses, but they *do* partition the grid, so hiding them would remove working facets
- **4** have 0 aliases **and** 0 documents (`BERT`, `ColBERT`, `HyDE`, `Contrastive learning`) — inert

So only the last group is hidden (behind "Show glossary-only (4)"), and the heading carries the
split — `12 collapse synonyms · 10 single-label · 4 glossary-only hidden` — which addresses the
"misleading count" complaint without deleting anything usable.

**A trap found while building, in this PR's own rule.** A family created with **no members** starts
at 0 aliases / 0 docs — exactly the shape the glossary-only group hides — so creating one would look
like it silently failed. `submitCreate` now reveals that group when (and only when) the new family
has no members.

**Live on the real corpus ($0, no LLM):** overlay **55 facets → 25 shown, 30 demoted** (the spec
predicted 30 rare, exactly right); toggle round-trips 25 ↔ 55; searching `mathrm` (a demoted 1-doc
keyword) finds it. Manage pool **38 → 12**, round-trips. Families **22 ↔ 26**. F2 exact match shows
"Go to family" + the warning; partial `conn` suggests `Brain connectivity`, `Connectome`. Dark theme
at 375 px: 0 px horizontal overflow, 0 console errors.
