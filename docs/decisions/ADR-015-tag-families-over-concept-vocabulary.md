<!-- status: active · updated: 2026-07-16 · class: append-only -->

# ADR-015 — Tag families over the curated concept vocabulary (keyword-name grouping)

- **Status:** accepted
- **Date:** 2026-07-16
- **Deciders:** user + Claude Code

## Context

The Library keyword filter (shipped as the two-pane overlay, `ce8b112`) exposes the auto-extracted
`Keyword` pool. On the real corpus it surfaces near-duplicates as separate entries — `llm`/`llms`,
`connectome`/`connectomics`/`connectomes` — so a filter on one misses the others. The user asked for **tag
families**: one canonical tag with several member surface forms the user can adjust, plus a **detection
script** that proposes families and (later) an optional LLM pass.

The repo already carries a curated-vocabulary layer — `Concept` (label, `source`, `definition`) +
`ConceptAlias` (surface forms) — built for the Phase-7 **concept graph / epistemics** (edges, stance, gaps,
corroboration). Two of the machinery's primitives already exist: `promote_keyword(name)` (a `Keyword` → a
`Concept` + a seed alias) and `add_concept(label, definition, aliases)`. Nothing in the frontend touches this
vocabulary yet — tag-families would be its **first UI/write surface**.

The user's framing bounds the decision: *"not really re-using but taking advantage of … we will add the
concept graph and all of the epistemics to the UI later on."* So tag-families is the **light** consumer of the
vocabulary (collapse synonyms for filtering + curation); the concept graph is the **heavy** consumer (research
integrity) and gets its own UI later, over the same rows.

## Options

**A. Where family data lives**
1. A **new lightweight `TagAlias` table**, decoupled from `Concept`/`ConceptAlias`.
2. **Reuse `Concept` + `ConceptAlias`** — a family = a `Concept` (canonical `label`) + `ConceptAlias` rows
   whose alias strings are the member `Keyword` names.

**B. What a family filters on**
1. **Keyword-name grouping** — a family's document set = the union of its member keywords' existing
   `document_keywords` links. Client-side grouping over data already shipped; no chunk scan.
2. **Concept-presence** — text-match the aliases against document/chunk text (`concept_skeleton` matchers),
   as the concept graph does. Higher recall (catches docs that *mention* a term the extractor missed) but a
   heavier path (a text scan + the presence sidecar).

**C. Relationship to the concept graph** — conflate tag-families with concept-graph nodes, or keep them the
same rows with **distinct consumers** and a distinct (later) UI.

**D. Curation home** — a dedicated in-app "Manage keywords" view · inline edit in the filter overlay · CLI-only.

**E. Detection signals** — morphological-only · morphological + `bge` embedding · morphological + edit-distance.

**F. LLM assist** — in v1 · deferred.

**G. Carve** — foundation-first (manual end-to-end, then detection) · detection-first.

## Decision

- **A2 — reuse `Concept`/`ConceptAlias`** ("take advantage of" the vocabulary; no new table). A family is a
  `Concept` + alias rows; the aliases are `Keyword` names.
- **B1 — keyword-name grouping.** Filtering a family returns the union of its members' `document_keywords`.
  Presence-matching is explicitly **not** tag-families' job — that recall lift arrives free when the concept
  graph's presence layer gets its UI.
- **C — same rows, distinct consumers.** Tag-families and the concept-graph/epistemics share the `Concept`
  vocabulary but are separate UIs with separate purposes; curating a family also enriches the vocabulary the
  graph will later consume (a feature, not a coupling — the graph already gates its own edges by
  co-occurrence, so a larger vocabulary is coverage, not noise). Do **not** build the graph/epistemics UI here.
- **D — a dedicated "Manage keywords" view** (opened from the filter overlay) hosts proposal-review +
  per-family editing (rename canonical, add/remove member keyword, split/merge, delete). The filter overlay
  stays a filter.
- **E — tiered morphological + `bge` embedding** (edit-distance as a supporting signal), zero-LLM, `$0`; the
  pure detector lives in `src/`, called by a "Detect" endpoint/button and also exposable as a CLI runner.
  **Nothing auto-applies** — proposals are reviewed in the Manage view.
- **F — LLM assist deferred** (a later increment; prove on Ollama first per KI-4 cost-discipline).
- **G — foundation-first:** **PR-1** families end-to-end, manual (read + CRUD + Manage view + overlay
  atomic-family rendering) → **PR-2** detection → **PR-3** LLM (parked). Build PR-1 first.

## Consequences

- **Easy:** collapsing `llm`/`llms` into one filter entry with a hand-made family; the overlay renders a
  family as an atomic entry (canonical + N forms, union count). Reuses `promote_keyword`/`add_concept` and the
  existing `document_keywords` links — no new schema, no chunk scan, no migration (the `Concept`/`ConceptAlias`
  tables already exist).
- **Committed to:** the **first frontend write path to the `Concept` vocabulary** (family CRUD over
  `Concept`/`ConceptAlias`); a small read/CRUD/detect API surface (thin shells over `concept_skeleton`/
  `library`); a `bge`-backed detection core in `src/`. Curated tag-families become part of the vocabulary the
  future concept-graph UI reads.
- **Hard / deferred:** presence-recall (docs that mention a term the extractor never tagged) — closes with the
  concept-graph presence UI; the LLM confirm/merge pass (PR-3); user-pinned favorites, which ride this work as
  *promote-to-`Concept`* (a promoted concept = a de facto favorite, per the keyword-overlay grill).
- **Boundary risk to watch:** because tag-families and the concept graph share rows, a future graph UI must
  treat the vocabulary as shared state (a family created for filtering is a real `Concept`). The `source`
  field (`"keyword"` | `"manual"`) already distinguishes provenance if the graph ever needs to filter what it
  surfaces.
