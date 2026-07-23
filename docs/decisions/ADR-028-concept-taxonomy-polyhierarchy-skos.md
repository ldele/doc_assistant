<!-- status: active · updated: 2026-07-23 · class: append-only -->

# ADR-028 — Concept taxonomy amendment: a unified, typed, polyhierarchical SKOS graph

- **Status:** accepted
- **Date:** 2026-07-23
- **Deciders:** user + Claude Code (routed from the 2026-07-23 `grill-me` — 7 taxonomy branches
  resolved, the epistemic-health cluster parked; ledger in
  `docs/PLAN_2026-07-23_concept-graph-taxonomy-epistemics.md` §6)
- **Amends [ADR-019](ADR-019-concept-taxonomy-classification-layer.md):** supersedes its Decisions
  **C1** (separate domain entity), **D1** (single-parent tree), and **D6** (document attaches via a
  single FK); partially reverses **D9** (no concept is-a). ADR-019's Decisions 1 (augment), 2/3
  (ANZSRC editable seed), 7/8 (curated + auto-propose-on-NULL), 10 (gap detectors additive), and 11
  (a dedicated taxonomy view owns edits) **stand unchanged**.

## Context

[ADR-019](ADR-019-concept-taxonomy-classification-layer.md) decided *to build* a curated
classification layer over the derived concept graph, but chose a **tree of research fields only**:
concepts merely *attach* to a field, domains are a *separate table* (C1), each concept has *one*
parent (D1), documents attach via a *single FK* (D6), and there is *no concept-to-concept is-a* (D9).
It was accepted 2026-07-18 and never built.

Building it surfaced three requirements ADR-019's shape cannot hold, all confirmed with the user in the
2026-07-23 grill:

1. **The intended structure is one broad→specific spine**, not fields only: *domain (neuroscience) →
   topic (connectomics) → concept (neuron) → specific type (pyramidal neuron)*. The last hop is a
   concept-to-concept **is-a**, which D9 forbids.
2. **A concept legitimately belongs to more than one parent** — "in two domains at once" — which D1's
   single-parent tree cannot express. ADR-019 itself named this as the reopener toward polyhierarchy
   (its "reverses if (b)"); this ADR exercises it, on model grounds rather than on scale.
3. **The hierarchy must be user-editable and not locked to any one source scheme**, because (a) the
   user must be able to reshape it and (b) the best shape is not yet known.

Two facts from the codebase bound the schema. `concept_edges` (the derived Node-A/B graph) is
**dropped and rebuilt on every `build_concept_skeleton` run** (Enrichment-Layer Pattern), so any
*curated* structure stored there would be destroyed by a routine rebuild — the KI-17/KI-20 class of bug
the 2026-07-21 E0 batch just fixed. And `Concept` already carries orthogonal flags (`graph_include`
per ADR-018, `folder_id` per ADR-025), so it is an established home for additive per-node state.

External schemes were characterised in ADR-019 and re-verified 2026-07-23:
[SKOS](https://www.w3.org/TR/skos-primer/) is a W3C data model of `broader`/`narrower`/`related`
properties (not a structure — it permits multiple `broader` edges, i.e. polyhierarchy, and does **not**
enforce acyclicity);
[ANZSRC 2020 FoR](https://www.abs.gov.au/statistics/classifications/australian-and-new-zealand-standard-research-classification-anzsrc/latest-release)
is a 3-level (2/4/6-digit) research-field classification, **CC-BY**, 23 divisions · 213 groups;
[MeSH](https://hhs.github.io/meshrdf/tree-numbers) (~30k descriptors, public domain) and
[ACM CCS 2012](https://dl.acm.org/ccs) (~2k concepts, ships as SKOS/XML) are both natively
polyhierarchical but domain-siloed.

## Options

**Hierarchy scope.** (A) fields only, concepts attach, no concept is-a — ADR-019 as written, but
stops at "neuron", cannot express "pyramidal neuron". (B) one unified graph where fields *and* concepts
are nodes and `broader`/`narrower` spans every hop, edges *typed* to keep is-a distinct from field-of —
one store, one traversal; costs a type on the edge and a node-kind flag. (C) two separate structures, a
field taxonomy plus a concept is-a tree — clean separation but splits the one spine the user wants
across two graphs.

**Node model.** (i) one node table (`Concept` + a `kind` flag) — one id space, which the codebase
already favours ("concept UUIDs everywhere" was PR-G1's fix for the KI-15 id-space confusion); every
presence-assuming consumer must filter by kind. (ii) a separate `domain` table (ADR-019 C1) — a
presence-detector structurally cannot touch a domain, but a typed edge whose endpoints may be *either*
kind then needs a polymorphic FK, which SQL cannot express against two tables.

**Coverage counting under multiple parents.** (set) a member counts in every field it sits under, and
rollup to a common ancestor dedups by id — explainable integer counts. (fractional) a member under two
fields contributes 0.5 to each — avoids inflation but yields "7.5 papers", which reads as a bug.
(primary) count only a designated primary parent — simple, but discards the polyhierarchy just adopted.

**Document attachment.** (derived) a document's fields = the fields of the concepts its chunks mention —
free, but blind to the **25 of 47** documents that carry no concept presence (ADR-019's measured
motivation for D6). (explicit) a curated + auto-proposed `document_field` link — covers all documents;
under polyhierarchy it must be many-to-many, not the single FK D6 imagined.

**Seed depth.** (trunk) seed ANZSRC divisions+groups only (~236 rows), graft deeper subtrees on demand.
(full) import MeSH/ACM CCS wholesale for depth — but 30k/2k mostly-unused nodes is the "a facet that
partitions nothing" anti-pattern (`feature-tag-families.md` PR-2.7, which demotes 1-document facets) at
30,000× scale: a field with zero documents is noise, not structure.

## Decision

Adopt a **unified, typed, polyhierarchical SKOS-shaped concept graph**, layered over — not replacing —
the derived co-occurrence graph (ADR-019 Decision 1 stands). Concretely:

1. **One unified graph, typed edges (scope B).** Fields and concepts are all nodes; the
   domain→topic→concept→subtype spine is expressed as SKOS `broader`/`narrower` edges. *Deciding
   reason:* SKOS `broader` already expresses every hop, so the only addition needed is a **type** on
   the edge; typing keeps is-a and field-of honestly distinct rather than collapsing them into one
   ambiguous relation. **This reverses ADR-019 D9** — concept is-a is now modelled — but as a
   *distinctly typed* edge, so classification and is-a stay separable. *Reverses if:* typed-edge
   curation proves too fiddly in practice → fall back to two structures (C) or fields-only (A).

2. **Edge types: `is-a`, `in-field`, `related`; `part-of` parked.** `is-a` = concept→broader concept;
   `in-field` = the "belongs to a broader field" relation, covering both field→field and concept→field
   (the endpoint `kind`s disambiguate if a detector needs it); `related` = the existing associative
   Node-A/B edges, unchanged. *Deciding reason:* two hierarchical types deliver the is-a/field-of
   distinction; a curated `part-of` (meronymy) re-invites the exact confusion that made Node B unusable
   for structure (ADR-019 measured 189/221 of its taxonomy-shaped relations as meronymy with unreliable
   direction). *Reverses if:* coverage math needs `field→field` split from `concept→field` (→ split
   `in-field` into `subfield-of`/`classified-in`), or a concrete feature needs curated `part-of`.

3. **No maximum hierarchical depth; acyclicity is the structural invariant.** Depth is data-driven and
   variable; a curation-lint may *flag* an unusually deep is-a chain (advisory), but nothing caps it.
   *Deciding reason:* a hard depth cap is exactly the corpus-tuned magic number the robustness contract
   bans (`.claude/CONTEXT.md`), "depth" is multi-valued under polyhierarchy (min/max path to a root),
   and acyclicity alone guarantees traversal termination and well-defined roots. The invariant is
   enforced as a cycle-check on edge insert (`nx.is_directed_acyclic_graph` on the build-time NetworkX
   artifact). *Reverses if:* a traversal needs a hard bound for cost at the 10k-concept contract (a
   `RIGOR_TODO` measurement, not a design cap).

4. **Domains are `Concept` rows with a `kind` column (`'concept'` | `'domain'`) — node model (i);
   this supersedes ADR-019 C1.** *Deciding reason:* the typed hierarchy edges span domain↔concept, so
   both endpoints must share **one node-id space** — a separate domain table forces the edge table into
   a polymorphic FK SQL cannot express. C1's real benefit (presence-detectors never touching abstract
   domain nodes) is recovered by a **single canonical `presence_nodes()` accessor that filters
   `kind='concept'`** — one guard, centralised, not N scattered guard clauses. `kind` is an additive
   column (`db/migrations.py` `_ADDITIVE_COLUMNS`). *Reverses if:* domains acquire domain-only columns
   that pollute `Concept` → a separate table.

5. **The curated hierarchy lives in a new `concept_hierarchy` table that survives a skeleton rebuild;
   the associative `related` edges stay in the derived `concept_edges`.** `concept_hierarchy` =
   (`source_id`, `target_id` both FK `concepts.id`, `type` ∈ {`is_a`, `in_field`}), polyhierarchy-native
   (many rows per node). **This supersedes ADR-019 D1** (a single-parent tree). *Deciding reason:*
   `concept_edges` is dropped and rebuilt on every `build_concept_skeleton` run, so storing the user's
   hand-built hierarchy there would let a routine rebuild wipe it (the KI-17/KI-20 class); curated data
   lives beside `Concept`/`ConceptAlias`, which survive rebuilds. Additive via `create_all` (no
   rebuild-migration). *Reverses if:* — (forced by the Enrichment-Layer Pattern).

6. **Coverage uses set-semantics counting; documents attach via an explicit `document_field`
   many-to-many (this supersedes ADR-019 D6's single FK).** A field's coverage = the **distinct set**
   of members for which it is an ancestor; a member under two fields is a full member of both (the truth
   polyhierarchy captures), and rollup to a common ancestor dedups by id. No fractional counts, no
   forced primary parent, and no sideways leak (a gap in statistics does not close because a stats
   concept also hangs off ML). Documents attach via a curated + auto-proposed **many-to-many**
   `document_field`. *Deciding reason:* distinct-set counts are explainable and honest; explicit
   attachment covers the **25/47** concept-less documents a derived-only rule is blind to; polyhierarchy
   makes it many-to-many, not a single FK. *Reverses if:* per-field rollup is too slow at 10k docs →
   materialise counts (`RIGOR_TODO`); auto-propose accuracy is poor → coverage-based gaps stay gated
   behind RG-015.

7. **Seed the ANZSRC 2-level trunk only; graft MeSH/ACM CCS subtrees on demand; attribution is
   required.** Seed ~236 editable `Concept(kind='domain')` rows + `in_field` edges from a small bundled
   **CC-BY** data file via an idempotent `scripts/seed_taxonomy.py`; deeper depth is grafted only where
   the corpus has documents/concepts under it (a curation action, using MeSH/ACM as candidate sources).
   *Deciding reason:* bulk-importing 30k MeSH / 2k ACM nodes for a small corpus is the "facet that
   partitions nothing" anti-pattern at scale; seed the trunk that maps the corpus and grow depth where
   documents justify it. ANZSRC's CC-BY licence **requires visible attribution** — in About/Settings
   *and* the seed data-file header (a product obligation, not optional). *Reverses if:* most concepts
   pile under one coarse group → pull ANZSRC's 6-digit fields or graft standard subtrees earlier.

8. **Auto-propose a parent where an `in-field` edge is NULL (ADR-019 E1 pattern, extended).** When a
   concept has no field parent, the LLM *proposes* one from existing taxonomy nodes first (a new
   MeSH/ACM graft is the heavier secondary action), user-gated accept/edit, **never auto-written**,
   `$0`/Ollama (KI-4). Same mechanism as `document_field` auto-propose. *Deciding reason:* this is
   ADR-019 E1 ("LLM proposes only where the FK IS NULL, never overwriting") applied to the new
   `concept_hierarchy` and `document_field` links; no new pattern is invented.

The **epistemic-health detector layer** (per-concept contradiction/hedging/citation-density,
source-trust scoring, dual + non-paper staleness, content-type degradation, degree-based-detector
retirement) is **explicitly out of scope here** and parked to a dedicated future ADR (ADR-EH),
sequenced *after* this taxonomy is validated. It is blocked on measurement (per-concept LLM-pass cost)
and a heterogeneous corpus (source-trust); tracked in `RIGOR_TODO.md` **RG-023**, design in
`docs/PLAN_2026-07-23_concept-graph-taxonomy-epistemics.md` §4–6.

## Consequences

**Easier.** The user's broad→specific spine is now representable end to end, and a concept can honestly
sit in multiple domains. Coverage math over *documents* ("N papers under machine learning, 1 under
statistics") becomes trustworthy — closer to ADR-004's north star than degree-based `under_connected`.
The hierarchy is data (editable edges), not schema, so the user reshapes it freely and no source scheme
is locked in. Storing the curated hierarchy outside `concept_edges` means a `build_concept_skeleton`
rebuild can never destroy it.

**Harder.** Two edge-typed layers plus a node-kind flag now live over the same nodes; every
presence-assuming consumer must go through the `presence_nodes()` accessor (one guard to write once, but
one that must not be bypassed). Coverage math must implement set-union rollup with dedup. The seed
carries a standing CC-BY attribution obligation in the product UI. Auto-propose quality is only as good
as the LLM pass and is gated behind RG-015 before coverage-based gaps are trusted.

**Must revisit.** Seed depth (2 levels) is evidence-backed (a 5/5 mapping of the user's own example,
ADR-019) but not outcome-validated — RG-015. Whether documents and their concepts may *disagree* about
domain, and which wins for coverage, surfaces on first implementation (ADR-019 flagged this; set-union
counting makes disagreement additive rather than contradictory, but the display rule is unspecified).
The `related` associative layer stays association-only until Node B stance is regenerated (RTX box,
KI-4).

## Confidence

- ✓ **SKOS expresses the full spine and permits polyhierarchy without enforcing acyclicity** — verified
  at the W3C primer/reference 2026-07-23; acyclicity is therefore our invariant, not SKOS's.
- ✓ **The curated hierarchy must not live in `concept_edges`** — that table is dropped/rebuilt every
  skeleton run (`db/models.py` `ConceptEdge` docstring; Enrichment-Layer Pattern); this is a
  correctness constraint, not a preference.
- ✓ **Explicit document attachment is needed** — 22/47 documents carry concept presence (ADR-019,
  measured); derived-only coverage is blind to the other 25.
- ✓ **ANZSRC is redistributable at the needed granularity** — CC-BY confirmed at the ABS source;
  4611/3209 present (ADR-019).
- ⚠ **Seed depth of 2 levels is right** — supported by a 5/5 example mapping, not outcome-validated
  (RG-015).
- ⚠ **Auto-propose placement will be accurate enough to trust coverage gaps** — asserted, not measured
  (RG-015).
- ⚠ **A single unified typed graph will be curatable in practice** (vs two structures) — untested at any
  vocabulary size; the "reverses if" on Decision 1 is the fallback.
