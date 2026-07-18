<!-- status: active · updated: 2026-07-18 · class: append-only -->

# ADR-018 — Graph vocabulary scope: an opt-in `graph_include` flag over the shared `Concept` table

- **Status:** accepted
- **Date:** 2026-07-18
- **Deciders:** user + Claude Code

> **Scope.** This ADR settles **which `Concept` rows the concept graph is built over**. It does not touch
> what the graph renders (`docs/specs/feature-concept-graph.md`), the UI boundaries it crosses
> (**ADR-017**), or the gap layer it surfaces (**ADR-004**). **ADR-015** named the collision this
> resolves — "the boundary risk to watch" — and reserved the graph track over the same rows.
> This supersedes nothing.

## Context

**ADR-015's boundary risk materialized.** Tag families and concept-graph nodes are the *same* `Concept`
rows, and the two features want opposite things from that table: families want **breadth** (every keyword
worth grouping), the graph wants a **small curated vocabulary** (a readable map, meaningful gaps). There
was no way to satisfy both, because neither had a way to say "this row is mine".

**Measured facts this decision rests on** (live corpus on the CPU box, 2026-07-18, all $0/offline):

- **The vocabulary is unfiltered at both ends.** `concept_skeleton.load_concepts()`
  (`src/doc_assistant/concept_skeleton.py:810`) selects **every** `Concept` row; so does
  `library.list_keyword_families()` (`src/doc_assistant/library.py:491`). One table, two consumers,
  zero scoping.
- **The vocabulary was flooded by a bulk promotion.** All **344** `source="keyword"` concepts share a
  single `created_at` date — **2026-07-05** — i.e. one run of `scripts/seed_concepts.py --promote-all`
  (`scripts/seed_concepts.py:102`). This ran against `promote_keyword`'s own stated contract: *"a Keyword
  is a candidate only — never auto-written as a Concept (Decision 1); the user promotes one"*
  (`concept_skeleton.py:829`). The deliberately curated vocabulary is the **13** `source="manual"` rows.
- **The graph is noise-dominated as a result, and it is not a staleness artifact.** `load_graph_view()`
  returns **357 nodes / 1534 edges / 302 gaps**, with `staleness=False` — the skeleton faithfully
  represents this vocabulary. Node labels include `'unlabeled'`, `'speckles'`, `'hyaline'`,
  `'x 13'`, `'13 intentionally omitted'`.
- **The gap layer inherits the noise, which breaks the feature's whole premise.** `single_source` is
  **224 of 302** gaps here. RG-014's verdict — the one PR-G2b's "strong kinds first" ordering is built on
  — found `single_source` to be the *strong, low-volume* signal (3 true positives at a 26-concept
  vocabulary). At 357 concepts it is the loudest kind by an order of magnitude. **A gap on a concept
  named `'speckles'` is not a research finding.**
- **Deletion is not available.** Dropping the 344 rows to clean the graph would delete **344 keyword
  families** from the shipped Manage-keywords view (PR-1/PR-2, `0c3b0d4`/`0af43db`) and cascade into
  `concept_presence`, `concept_edges`, and `gaps`.

**Note on corpora.** ADR-017 and PR-G2a were authored against a **76-doc / 26-concept** corpus on the RTX
box. This box carries **47 docs / 688 keywords / 357 concepts**. The two are different machines with
different data homes — a fact worth holding when a spec's "live-verified" numbers do not reproduce.

## Options

1. **An additive `graph_include` flag on `Concept`; the graph filters on it. (CHOSEN)**
   *Pros:* both features keep the rows they need — families stay whole, the graph gets a curated subset;
   additive and reversible (no data loss, flip a flag); gives curation an **explicit, inspectable surface**
   rather than an implicit side effect of a `source` value; the migration precedent is well-worn
   (`_ADDITIVE_COLUMNS`, `db/migrations.py:25`). *Cons:* one more column and one more concept to explain;
   needs a backfill; the opt-in **UI** is not built by this change (CLI/seed only for now).

2. **Filter the graph on the existing `source='manual'`.**
   *Pros:* zero schema change, ships in one line. *Cons:* overloads `source` — a **provenance** field,
   recording *where a row came from* — as a **curation control**, meaning *whether the user wants it*.
   Those diverge the moment a genuinely graph-worthy concept arrives via `promote_keyword`: it is
   permanently ineligible, and the only fix is to lie about its provenance. Rejected as a semantic
   overload that would need undoing later.

3. **Undo the bulk promotion (delete the 344).**
   *Pros:* one shared small vocabulary, no new concept. *Cons:* destructive and cascading; deletes 344
   keyword families from a shipped view to fix a different feature; loses the promotion work; and, being a
   data fix rather than a structural one, **the next `--promote-all` re-floods the graph**. Rejected.

4. **Split into two tables (`Concept` for the graph, a separate family entity).**
   *Pros:* the cleanest conceptual separation — no shared row at all. *Cons:* a real migration across
   `concept_presence`/`concept_edges`/`gaps`/families, invalidating ADR-015's deliberate "same rows"
   design and every consumer, to buy what a nullable column buys. Rejected as disproportionate; revisit
   only if the two vocabularies genuinely diverge in *shape*, not just membership.

## Decision

Add a nullable **`graph_include`** boolean to `Concept`. `concept_skeleton.load_concepts()` returns only
rows where it is true. `library.list_keyword_families()` stays **unfiltered** — families keep every row.

**Polarity: opt-in (default excluded).** New concepts do **not** silently enter the graph. This is the
load-bearing half of the decision: the failure being fixed is *the vocabulary growing into the graph
unbidden*, and the families feature is expected to keep adding rows indefinitely. An opt-out default would
let the identical regression recur on the next bulk operation; opt-in makes re-flooding **structurally
impossible** rather than merely discouraged.

**Which paths opt in**, following the same rule the backfill uses — deliberate glossary curation joins the
graph, organisational/candidate paths do not:

| Path | `graph_include` | Why |
|---|---|---|
| `add_concept()` | **True** (parameter, default true) | The direct-curation glossary path — a user naming a concept they care about. |
| `promote_keyword()` | **False** | A *candidate* promotion by its own contract; also the path `--promote-all` drives. |
| `create_keyword_family()` | **False** | Library organisation, not a claim that the concept belongs on the map. |
| Backfill (existing rows) | `source == "manual"` | Reproduces the same rule retroactively: the 13 curated in, the 344 bulk-promoted out. |

**Backfill** ships as an idempotent CLI runner (`scripts/backfill_graph_include.py`) per the
Enrichment-Layer Pattern, not as logic hidden in the migration — the migration adds the column, a runner
sets policy, and re-running it is a no-op.

**An empty flag set means an empty graph, and that is correct.** `load_concepts()` already documents
*"Empty vocabulary → empty graph (the curation prereq)"*; PR-G2a already renders an empty state offering a
rebuild. A fresh install therefore has no graph until something is curated — the pre-existing contract,
now actually enforced.

## Consequences

- **The graph on this box goes 357 → 13 nodes.** That is the point, but 13 is a *small* map, and the gap
  distribution must be **re-measured, not assumed**: PR-G2b's "strong kinds first" ordering rests on
  RG-014's verdict, which was calibrated at 26 concepts on a different corpus. Re-measure before building
  the triage surface — a rerun of the verdict, not a rubber stamp.
- **Curation has no UI yet.** Opting a concept in is CLI-only (`add_concept`, the backfill runner) until a
  follow-up PR adds the toggle. Its natural home is the **Manage-keywords** view, which keeps ADR-017 A1
  intact (the graph still never writes the vocabulary — the keywords view does).
- **`--promote-all` is now safe for the graph** but still violates `promote_keyword`'s documented
  candidate-only contract. Out of scope here; worth its own look.
- **Families are untouched** — `list_keyword_families()` still returns all 357, so the Manage-keywords
  view is byte-identical in behaviour. This is the guard test.
- **A rebuild is required** for the filter to take effect; the skeleton is a derived artifact
  (`build_concept_skeleton --apply --enrich` — **`--apply` alone wipes Node B's stance annotations**, see
  `tests/eval/baselines/superseded_year_rule_2026-07.md`).
- **Reversible.** Flipping rows back to true and rebuilding restores the old graph exactly; no row is
  deleted at any point.
