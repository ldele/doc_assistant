<!-- status: active · updated: 2026-07-18 · class: append-only -->

# ADR-019 — Concept taxonomy: a curated classification layer over the derived concept graph

- **Status:** accepted (amended in part by [ADR-028](ADR-028-concept-taxonomy-polyhierarchy-skos.md),
  2026-07-23 — supersedes Decisions C1/D1/D6, partially reverses D9; the rest stands)
- **Date:** 2026-07-18
- **Deciders:** user + Claude Code (routed from the 2026-07-18 `grill-me` — 8 branches resolved,
  1 parked-then-resolved by a research pass; ledger in `.claude/SESSION.md`)

> **Scope.** This ADR decides **how concepts are organised into a hierarchy of fields**, and how that
> hierarchy relates to the graph that already exists. It does not change what the concept graph renders
> (`docs/specs/feature-concept-graph.md`), the gap layer's existing detectors (**ADR-004**), or the
> vocabulary-scoping flag (**ADR-018**). It **closes** the grouping question shelved by the library
> redesign on 2026-07-15 ("grouping-if-ever = manual tags/folders, own ADR"). This supersedes nothing.

## Context

The concept graph is flat. Every structural signal it has — co-occurrence edges, Louvain communities,
degree-based gap detectors — is *derived from text adjacency*, and none of it can express the one thing
this corpus most needs said: **which field a concept belongs to**.

**Measured facts this decision rests on** (live corpus, CPU box, 2026-07-18, all $0/offline):

- **The corpus is genuinely multi-domain.** 47 documents spanning IR/RAG, systems neuroscience, viral
  tracing / mouse genetics, and AI planning. A flat graph renders `pddl` and `BM25` as unrelated isolated
  nodes; it has no way to say *both are computer science, in different subfields*. On 2026-07-18 that
  representational gap caused a reviewer (Claude) to misread four correctly-curated neuroscience and
  planning concepts as extraction junk — the second occurrence of that error in two days
  (`docs/specs/feature-concept-graph.md` → Traps).
- **Half the library is structurally invisible.** Only **22 of 47** documents carry any
  `concept_presence` row. Anything computed over concepts alone cannot see the other 25.
- **The derived groupings cannot hold a user's intent.** Louvain communities are positional and
  **renumber when a concept is added** — the spec's own trap says never persist a user preference
  against one. There is no stable, nameable grouping in the system today.
- **The vocabulary miner cannot supply structure either.** **672 of 688 keywords (97.7%) appear in
  exactly one document**; the keyword extractor scores per-document salience, not cross-document
  vocabulary.
- **Two features already collided over `Concept` rows.** **ADR-015** named the risk; **ADR-018** paid for
  it (tag families and graph nodes are the same rows, so the graph could not be cleaned by deletion).
  Any third consumer of those rows starts with that history.
- **A grouping mechanism was already wanted and deferred.** The library redesign's **Collections rail
  renders but is permanently empty-stated**; folder-mirroring (Phase B) was shelved on 2026-07-15 because
  the source directory is flat and all-PDF, with the note *"grouping-if-ever = manual tags/folders (own
  ADR)"*.

**One candidate source was tested and rejected on evidence.** Node B's LLM enrichment appears to emit
hierarchy — **221 of 1254** annotated relations are taxonomy-shaped. Sampling shows it is not usable:
**189 of the 221 are `is a component of`, which is meronymy (part-of), not hyponymy (is-a)** — a different
relation with different inheritance semantics; directions are unreliable (`includes` is inverted
throughout, e.g. `print → programming languages`); and pairs are frequently cross-domain nonsense
(`dnns → pose`, `abstractions → neuronal relationships`). Those were extracted over the polluted
357-concept vocabulary, so the fair re-test is the clean one: over the current 13 concepts, **2 of 19**
edges are taxonomy-shaped and **1 of those 2 is wrong** (`dense retrieval is a component of knowledge
distillation`). Node B yields roughly one usable hierarchy edge per rebuild.

## Options

### A — Does the taxonomy replace the co-occurrence graph, or augment it?

1. **Augment — two layers.** *Pros:* the gap layer's value is *evidence* ("which documents, which
   chunks", `single_source`), which only presence/co-occurrence supplies — a pure taxonomy knows
   `pddl` sits under AI planning but not that you own one paper on it. The layers also have different
   lifecycles: a taxonomy is durable user-owned data, the skeleton is rebuilt in ~7 s. *Cons:* two
   structures to store, render and reason about.
2. **Replace — taxonomy only.** *Pros:* one structure; discards a derived layer that demonstrably
   produces noise. *Cons:* throws away working machinery and leaves the gap detectors with no evidence
   base.
3. **Replace, keeping presence.** *Pros:* kills the noisy half (edges, communities, Node B relations),
   keeps the evidence half. *Cons:* still discards `single_source`'s substrate mid-flight, on a signal
   RG-014 measured as the strongest one.

### B — Where does the backbone come from?

Researched 2026-07-18. No single external scheme spans this corpus at *full* depth — MeSH has
`Cre recombinase` but not `BM25`; ACM CCS the reverse. The upper levels are where schemes agree, so only
the **trunk** need come from a standard.

| Scheme | Size | Licence | ML/AI node? | Bio/neuro fit |
|---|---|---|---|---|
| **ANZSRC 2020 FoR** | 23 divisions · 213 groups | **CC BY** — the ABS applies CC BY 4.0 to its published data ([ABS copyright](https://www.abs.gov.au/website-privacy-copyright-and-disclaimer)); ANZSRC is recorded as CC BY 2.5 AU ([Wikipedia](https://en.wikipedia.org/wiki/Australian_and_New_Zealand_Standard_Research_Classification)) | **Yes** — 4602 Artificial intelligence, 4611 Machine learning | **Yes** — 31 Biological sciences, 32 Biomedical and clinical sciences (3209 Neurosciences), 3105 Genetics |
| OECD FOS | 6 · 42 | unverified | **No** — stops at "Computer and information sciences" | coarse |
| Scopus ASJC | 4 · 27 · 333 | Elsevier proprietary; the only permission claim found is secondhand and concerns the *title list*, not the taxonomy | partial | yes |
| arXiv | ~150 | descriptive metadata CC0 ([arXiv licences](https://info.arxiv.org/help/license/index.html)) | yes (cs.LG, cs.AI) | **weak** — `q-bio` is shallow; no home for deep brain stimulation or Cre recombinase |
| UNESCO (1988) | 24 · ~250 | CC BY 4.0 attaches to a *third-party SKOS rendering*, not the source | **No** — 1988 vintage, predates ML | yes |

A sixth option — **author the trunk by hand, no standard** — was considered: maximal fit, zero licence
surface, but it forgoes a shared vocabulary and every future field is an invention.

### C — Where do domain nodes live?

1. **A separate entity** (new table, self-referential `parent_id`; `Concept` gains an FK). *Pros:* a
   domain is abstract, zero-presence and seeded; a `Concept` is something that appears in text — they are
   different kinds. The presence-based gap detectors then simply do not apply to domains, with no guard
   clauses to forget. *Cons:* two node types on the wire; joins.
2. **`Concept` rows + `parent_id` + a kind flag.** *Pros:* one table, one id space — which PR-G1
   deliberately established ("one wire id space, concept UUIDs everywhere") after **KI-15** was caused by
   id-space confusion. *Cons:* `Concept` means two things; every presence-assuming consumer needs a kind
   guard.
3. **`Concept` rows, no flag — domains are just concepts with no presence.** *Cons:* a zero-presence
   concept is indistinguishable from a *failed* one that should have matched text and didn't — which is
   precisely the `isolated` gap signal. Corrupts gap semantics permanently.

### D — Tree or polyhierarchy?

1. **Tree** — nullable FK, exactly one parent. *Pros:* coverage counts are unambiguous. *Cons:* forces
   an arbitrary choice where a concept genuinely spans fields.
2. **Polyhierarchy** — join table, as MeSH does. *Pros:* truer to how knowledge works. *Cons:* coverage
   math needs a stated double-counting rule (does a paper under ML *and* neuroscience count twice? does a
   gap in statistics close because a stats concept also hangs off ML?), and tree rendering gets harder.

### E — How are assignments made durable and kept fresh?

1. **Curated, with auto-propose for `IS NULL` only.** *Pros:* two in-repo precedents —
   `backfill_graph_include` (ADR-018) touches only NULL rows, and ADR-013's `DocumentMeta` override
   sidecar makes user-entered values win and survive re-ingest. Self-maintaining; the user always wins.
2. **Curated only, no auto-fill.** *Cons:* every new concept needs manual placement — the friction that
   makes taxonomies go stale, and unassigned concepts are invisible to coverage math.
3. **Fully derived each rebuild.** *Cons:* destroys user edits on every rebuild — the same class of bug
   as the `--apply` footgun and **KI-17**.

## Decision

**Adopt a curated classification layer, seeded from ANZSRC 2020 and layered over — not replacing — the
derived concept graph.** Concretely:

1. **Augment (A1).** The taxonomy is a curated layer; the derived co-occurrence skeleton remains as the
   **evidence** layer. *Deciding reason:* gap findings are only actionable with evidence attached, and
   only presence/co-occurrence supplies it.
2. **Backbone = ANZSRC 2020 Fields of Research, seeded at divisions + groups (23 + 213 ≈ 236 rows)
   (B1).** *Deciding reason:* it is the only candidate that is **both** freely redistributable (CC BY)
   **and** able to express the granularity actually wanted. The user's own framing example maps onto its
   two top levels five for five — biology → **31**, math → **49**, computer science → **46**, machine
   learning → **4611**, neuroscience → **3209** — which is evidence for the seed depth, not a guess.
   OECD FOS is disqualified independently of its licence: it cannot represent "machine learning" at all.
3. **Seeded as ordinary editable rows, not a vendored immutable asset.** *Deciding reason:* the taxonomy
   is explicitly the user's to reshape; an immutable asset would put user edits in conflict with it on
   every upgrade, and it avoids new package-data plumbing in the frozen bundle.
4. **Domain nodes are a separate entity (C1)** — a new table with a self-referential parent. *Deciding
   reason:* this is the third feature to want a piece of `Concept`, and the first two collided
   (ADR-015's named boundary risk, paid for by ADR-018).
5. **A tree, via a nullable FK on `Concept` (D1).** *Deciding reason:* coverage math is what makes this
   earn its keep beyond navigation, and multi-parent makes counts ambiguous. Polyhierarchy remains
   additive later — the FK becomes the primary parent and a join table is added beside it.
6. **Documents attach to the tree as well as concepts** — an additive FK on `Document`. *Deciding
   reason:* only 22 of 47 documents carry concept presence, so a concept-only taxonomy is blind to half
   the library, and coverage math needs *documents* as its denominator.
7. **Assignments are curated; the LLM auto-proposes only where the FK `IS NULL`, never overwriting (E1).**
8. **The LLM may propose, never write — for both concepts and domains.** *Deciding reason:* this needs
   **no amendment to redesign Decision 1**, because the pattern already exists — a `Keyword` is "a
   candidate only … never auto-written as a Concept; the user promotes one". An LLM is simply a better
   candidate generator occupying the same slot.
9. **The hierarchy is a classification scheme** (broader/narrower *field*), **explicitly not an
   ontological is-a or part-of relation.** Node B's relation strings stay in the co-occurrence layer,
   unused for structure — see the Context measurement.
10. **Gap detectors are additive.** Domain-coverage kinds land beside the existing five; **nothing is
    retired until measured** (`.claude/RIGOR_TODO.md` **RG-015**).
11. **A dedicated taxonomy view owns all tree edits**; the concept graph stays read-only and deep-links
    to it, as it already deep-links to Manage-keywords.

**What would reverse this.** (a) If coverage-based gap detection measures *worse* precision than the
degree-based detectors it was meant to improve on (RG-015), the taxonomy falls back to a
navigation-and-organisation device only, and the gap layer stays as-is. (b) If the vocabulary grows past
roughly a few hundred concepts and single-parent placement is routinely arbitrary, D1 reopens toward
polyhierarchy. (c) If ANZSRC's licence terms change, the backbone is replaceable — it is seeded data, and
nothing but the seed depends on the source.

## Consequences

**Easier.** A stable, nameable, user-owned grouping exists for the first time — Louvain communities never
provided one. The Collections rail in the library redesign can finally be populated, closing the question
shelved on 2026-07-15. Coverage gaps ("12 papers under machine learning, 1 under statistics") become
expressible, which is materially closer to ADR-004's north star — *directions for exploration the user did
not think to look* — than degree-based `under_connected` ever was. And a multi-domain corpus stops
looking like a broken single-domain one, which is a documented, twice-repeated failure mode of reviewing
this vocabulary.

**Harder.** Two structures now exist over the same concepts, and the UI must make clear which one the
user is looking at. There is a new node kind (abstract, zero-presence) that most existing consumers were
not written for — the separate-entity choice contains the damage but does not eliminate the need to
audit them. The 25 concept-less documents need classifying from title/abstract to reach full coverage
(~25 Ollama calls, $0). And a new curation surface must be built and maintained.

**Obligations.** ANZSRC is CC BY: **attribution is required**, and this repo is public and ships an
installer, so the attribution must appear in the product (About/Settings) and in the seed data's header —
not only in this ADR. The exact licence version carried by the download must be cited when the seed is
built (CC BY 4.0 per current ABS policy vs CC BY 2.5 AU as recorded for ANZSRC; both attribution-only).

**Must revisit.** RG-015 (detector precision) gates any retirement of the existing five gap kinds. The
seed depth (two levels) is evidence-backed but not outcome-validated. Whether documents and their
concepts should be *allowed to disagree* about domain — and which wins for coverage math — is unspecified
here and will surface on first implementation.

## Confidence

- ✓ **Node B is unusable as a hierarchy source** — measured on both vocabularies (221/1254 taxonomy-shaped
  but 189 of them meronymy with inverted directions; 2/19 on the clean vocabulary, 1 wrong).
- ✓ **Half the library is invisible to concept-only structure** — 22/47 documents carry presence.
- ✓ **The keyword miner cannot supply cross-document structure** — 672/688 keywords are single-document.
- ✓ **ANZSRC is redistributable and expresses the needed granularity** — licence confirmed at the ABS
  primary source; 4611/3209 verified present.
- ⚠ **Domain-coverage detectors will outperform the degree-based ones** — *asserted, not measured.* This
  is the whole value case beyond navigation. Tracked by **`.claude/RIGOR_TODO.md` RG-015**, which also
  records why it is deferred rather than assumed: RG-014's ordering verdict was measured on a different
  corpus and **failed to transfer** across the ADR-018 rescope.
- ⚠ **Two levels (divisions + groups) is the right seed depth** — supported by a 5/5 mapping of the
  user's own example, but not validated against outcomes. Revisit once real assignment begins; folded
  into RG-015's audit.
- ⚠ **A single parent per concept will not feel arbitrary in practice** — untested at any vocabulary size
  beyond the current 13.
