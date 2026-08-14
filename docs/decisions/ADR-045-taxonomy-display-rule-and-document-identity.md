<!-- status: active · updated: 2026-08-12 · class: append-only -->

# ADR-045 — Show the most specific label, search the whole ancestry; a document's identity is not vocabulary

- **Status:** accepted (design; unbuilt)
- **Date:** 2026-08-12
- **Deciders:** user (product, 2026-08-12), Claude (Claude Code session 2026-08-12)
- **Amends [ADR-028](ADR-028-concept-taxonomy-polyhierarchy-skos.md):** fills the display rule its
  *Consequences → Must revisit* explicitly leaves unspecified. Nothing in ADR-028 is superseded;
  every decision there stands.
- **Relates to:** [ADR-019](ADR-019-concept-taxonomy-classification-layer.md) ·
  [ADR-015](ADR-015-tag-families-over-concept-vocabulary.md) (keywords and concepts are one table) ·
  [ADR-018](ADR-018-graph-vocabulary-scope.md) (`graph_include`) · DEVLOG 2026-08-12 (3)/(4)

## Context

ADR-028 built a unified, typed, polyhierarchical concept graph and then said, in its own
"Must revisit": *"the display rule is unspecified."* This ADR specifies it, because two separate
things now depend on it.

**1 · The keyword layer does not partition the corpus, and cannot.** Measured twice on 2026-08-12
over the live 97-document library: after repairing four extraction defects (D1/D2/D4/D5 —
DEVLOG 2026-08-12 (3) and (4)), keyword *content* is good — the PMC running header is gone,
overlapping shingles fell from 27% of slots to 0%, coverage went 82 → 96 of 97 documents. But
**97% of keywords still appear on exactly one document, before and after.** That is not a defect to
fix: per-document TF-IDF selects `df≈1` terms *by construction* — "distinctive to this document" is
what it means. A facet built on raw keywords will never group anything, no matter how clean the
keywords get.

**The taxonomy already solves this, and nobody wired it up.** `rag` has `df=1` and always will;
`machine learning` does not. If a document tagged `rag` is *findable* under `machine learning`, the
facet becomes the taxonomy rather than the keyword, and the partitioning problem dissolves without
touching extraction at all.

**2 · The machinery is built; the taxonomy is empty.** Live state, measured 2026-08-12:

| | |
|---|---|
| `Concept(kind='domain')` — the ANZSRC seed | **236** (213 carry a trunk parent) |
| `Concept(kind='concept')` | **357** |
| …of which carry **any** taxonomy parent | **13** |
| `concept_hierarchy` edges of type `is_a` | **0** |
| `document_field` links | 83 |

So `ConceptHierarchy`, `DocumentField`, `taxonomy.py` (acyclic writer, `presence_nodes`,
`unplaced_concepts`, NetworkX loader), `taxonomy_propose.py`, `scripts/seed_taxonomy.py`, the
`/api/taxonomy` routes and the `LibraryTaxonomy` view all exist and work. What does not exist is
**data**: 344 of 357 concepts are unplaced and the concept→concept spine has never been built.
Any display rule must therefore be correct *on today's empty taxonomy* as well as a full one.

**3 · The user's third ask is a different kind of thing.** A per-document key
`<topic>_<firstauthor>_<year>` (`rag_lewis_2020`), and a bibliographic spine
*document type → origin → key* (`article → journal → rag_lewis_2020`). The first is an identifier;
the second is bibliographic metadata. Neither is a research field, and the corpus filenames already
follow the `topic_author_year` convention (`transformer_vaswani_2017.pdf`, `rag_lewis_2020.pdf`).

## Options

**Display.** (a) Show every attached node — honest but unreadable: a document under
`rag` also shows `llm`, `machine learning`, `information and computing sciences`, and the specific
label drowns. (b) Show only the most specific attached nodes, expandable to ancestors — the
standard faceted-classification rule; costs an "expand" affordance. (c) Show a fixed depth (e.g.
always the division) — stable-looking, but depth is multi-valued under polyhierarchy and ADR-028
D3 already refused to cap it.

**Search/filter.** (i) Match only what is displayed — filtering by `machine learning` then misses
a paper tagged `rag`, which is the entire failure this ADR exists to fix. (ii) Match the ancestor
closure — one traversal, and the coarse filter finally does something. (iii) Match closure only for
explicitly "broad" nodes — needs a broadness flag nobody can define.

**Document identity.** (A) A `Concept` row per document (`kind='document'`) — uniform participation
in hierarchy, search and filter; but mints one permanent `df=1` node per document, i.e. re-creates
at the vocabulary level exactly the singleton problem this ADR is closing, and every facet surface
must then filter it back out. (B) A computed field derived from metadata — no vocabulary rows,
nothing to go stale, still displayable and searchable; loses uniform hierarchy participation.
(C) Both, with promotion on demand — defers the decision and doubles the surfaces.

**Bibliographic type/origin.** (α) Taxonomy nodes under `in_field` — cheapest to build, but
`in_field` then means both *"is a kind of document"* and *"belongs to a broader research field"*,
which is precisely the ambiguity ADR-028 D2 introduced typed edges to prevent. (β) A third
hierarchical edge type — keeps the distinction, at the cost of a third traversal to reason about.
(γ) Document metadata columns — the extended-metadata work the UI checklist already records as
owed, fillable from Crossref, and orthogonal to the topic spine.

## Decision

**1 · A document shows its most specific placed labels; ancestors are one expansion away
(option b).** The displayed set is the document's attached nodes minus any node that is an
**ancestor of another node in that same set**. Two attached nodes where neither is an ancestor of
the other are *both* shown — that is the polyhierarchy case working as intended: MESOSPIM placed
under both `optics` and `neuroscience` displays both, because neither subsumes the other.
*Deciding reason:* it is the only rule that makes the specific label visible without lying about
the general one — the general one is still there, one click away, and still governs search.
*Reverses if:* users routinely expand every document, which would mean the collapse is fighting them.

**2 · Search and filter match the full ancestor closure (option ii).** Filtering by a node matches
every document attached to that node **or to any of its descendants**. This is what makes a
coarse facet useful over a `df≈1` vocabulary, and it is the direct answer to the measured
partitioning failure: a corpus where 97% of keywords are singletons still filters cleanly by
`machine learning`, because the closure does the grouping the keywords cannot.
*Deciding reason:* the alternative — matching only what is displayed — makes every coarse filter
return almost nothing, which reads as a broken filter rather than a sparse taxonomy.
*Reverses if:* closure traversal becomes a hot path at the 10k-document contract → materialise a
closure table (a `RIGOR_TODO` measurement, not a design change).

**3 · Attach at the most specific applicable node; never also attach its ancestors.** Ancestors are
*derived* by Decision 2, never stored. A concept used only in neuroscience is attached to
`neuroscience`, not additionally to `biology`. *Deciding reason:* storing both makes the pair a
consistency hazard — deleting the specific edge silently leaves the general one, and coverage
double-counts. Under Decision 2 the ancestor link is free, so storing it buys nothing and costs
correctness.

**4 · Both rules must degrade to identity on an unplaced vocabulary.** A concept with no parent has
an empty ancestor set: it displays as itself and matches only itself. This is not a special case to
add later — **it is today's behaviour for 344 of 357 concepts**, so it is the path that will run
first and most. *Deciding reason:* the robustness contract's 0-document rule applied one layer up —
a feature whose correctness begins only once someone has curated a taxonomy would be dark on every
real install, including this one.

**5 · A document's identity key is a computed field, not a vocabulary row (option B).** The
`<topic>_<firstauthor>_<year>` key is **derived** from the document's own metadata and rendered
wherever the document is named; it is searchable; it is **not** a `Concept`/`Keyword` row and takes
no part in `concept_hierarchy`. *Deciding reason:* it is an identifier, not a classification — it
is `df=1` by definition for every document forever, so admitting it to the vocabulary would
manufacture 97 permanent singletons in the very facet this ADR is repairing, and every facet
surface would then need to filter them back out. A `Document` is already a first-class entity with
an id, title, authors and year; a second identity in a second table is duplication, not structure.
*Derivation:* filename when it already matches `topic_author_year` (most of this corpus does), else
first author + year from `document_meta`, with the topic word left empty rather than guessed — an
LLM-proposed topic word belongs to the P2 ingestion pass, gated the same way as every other
proposal (user-accepted, never auto-written). *Reverses if:* a real need appears to classify
*documents* into a hierarchy of documents, rather than to name them.

**6 · Bibliographic type and origin are document metadata, not taxonomy nodes (option γ).**
`article_type`, `journal`, `publisher` and friends are additive `Document`/`document_meta` columns —
the extended-metadata work `docs/ui-checklist.md` records as owed and as ~6 appends to
`_ADDITIVE_COLUMNS`, Crossref-fillable. *Deciding reason:* "is a journal article" is not "belongs
to a broader research field", and modelling it as `in_field` would give that edge two meanings —
the exact collapse ADR-028 D2 typed the edges to prevent. Faceting by document type is then a
metadata filter beside the taxonomy filter, not inside it. ⚠ **That decision needs its own ADR and
a fresh number**: the checklist calls it "ADR-016 owed", but ADR-016 is a retired number with no
file and numbers are never reused (`docs/decisions.md`). *Reverses if:* users want document type
and research field in one combined tree badly enough to accept a third edge type (option β).

## Consequences

**Easier.** The coarse filters finally work: a 97%-singleton keyword vocabulary becomes a usable
facet through its ancestry, with no change to extraction and no re-ingest. The specific label —
which is the one carrying the information — is what the user sees. Polyhierarchy displays honestly
instead of picking an arbitrary primary parent. And the immediate next action is unambiguous:
**place concepts**, not build anything, because the code is already there.

**Harder.** Two derived sets now exist per document (displayed = attached minus ancestors,
searchable = attached plus ancestors) and they must be computed from one source of truth or they
will drift. Ancestor closure over a DAG needs memoising per request at minimum. The "expand"
affordance is new UI. And the honesty burden moves: with 344 concepts unplaced, a coarse filter
returning few results means *the taxonomy is sparse*, not that the corpus is — the UI has to say
which, or it reproduces the empty-Graph-page problem this project just hid a tab over
(`docs/REVIEW_2026-08-12_release-readiness.md` §2b R4).

**Must revisit.** Placement accuracy and Decision 2's load-bearingness were tested the same day this
ADR was written, and **bulk auto-placement failed** (RG-015, "THE `--all-concepts` RUN", 2026-08-13):
over 344 uncurated keyword rows a strong local model placed `acdc` (a cardiac-MRI benchmark) under
**Music** and `alpha` (an EEG band) under **Analytical chemistry**, at the same confidence as its
correct answers. The decisive comparison is that the *same* pass over the **13 curated
`graph_include` concepts** was 13/13 plausible with a weaker model — **the variable is scope, not
capability.** So Decision 2 is safe only over a curated vocabulary, and ADR-018's `graph_include`
boundary is now a precondition of this ADR, not an unrelated flag. Nothing was placed in bulk; the
24 trial rows were deleted. The `is_a` spine (`rag → llm → machine learning`) has **zero** edges today, so
Decision 1's "most specific" is currently decided entirely by `in_field`; whether concept-to-concept
depth changes the display materially is untested. Whether a document and its concepts may disagree
about field — ADR-028's own open question — becomes visible the moment both are displayed.

## Confidence

- ✓ **The keyword layer cannot partition on its own** — measured twice on the live 97-document
  corpus, before and after four extraction fixes: 97% singletons both times (DEVLOG 2026-08-12).
- ✓ **The machinery exists and the data does not** — 13 of 357 concepts placed, 0 `is_a` edges,
  measured 2026-08-12 against `data/library.db`.
- ✓ **`in_field` would become ambiguous under option α** — ADR-028 D2 states the reason it is typed.
- ⚠ **Ancestor-closure filtering is affordable at the 10k-document contract** — asserted from the
  shallowness of a 2-level ANZSRC trunk, not measured. `RIGOR_TODO` if it becomes a hot path.
- ⚠ **Collapse-to-most-specific is the display users want** — a reasoned default from faceted
  classification practice, not validated on this UI.
- ⚠ **The `topic` word is derivable for most documents** — supported by the corpus filename
  convention, not counted. Where it is absent the key degrades to author+year rather than guessing.
