<!-- status: active · updated: 2026-07-08 · class: baseline -->

# Year-aware `superseded_trend` rule (SPRINT-003 / G3)

Records the exact comparison rule `concept_skeleton._aggregate_direction` uses to promote a
`contested` node to `direction="superseded_trend"`, per the sprint's DoD ("the exact rule is
locked in-sprint and recorded in the baseline"). This is a **rule-design** baseline (the rule
itself, worked out from the spec's constraints), not a corpus measurement — the real-corpus
year-coverage count and the resulting superseded/contested split are a **host-apply follow-up**
(see "Pending" below), not part of this sprint's deliverable (`docs/sprints/
SPRINT-003-year-aware-superseded.md` scopes the host `--apply` runs to the user, after review).

## The rule

For a node with >= 1 opposing (contradicting) document (i.e. already `coverage="contested"`):

1. If there is no supporting document at all (`ns == 0`), or no opposing document (never
   reached — this branch only runs when `nc >= 1`), stay `contested`. Nothing to compare a
   sole disputer against.
2. If **any** document in either the supporting set or the opposing set has no recorded
   `Document.year`, stay `contested`. Never guess a currency call on incomplete year data
   (fail-safe, per the spec's Decision 1 — this is a stricter reading than "compare what
   years we have": a partially-dated set is not evidence of a *trend*, so it defaults to the
   pre-G3 behaviour).
3. Otherwise, compare **median(opposing years)** vs **median(supporting years)**.
   `superseded_trend` only when the opposing median is **strictly newer**
   (`opp_median > sup_median`); equal medians (including the degenerate single-doc-per-side
   case where median == the one year) stay `contested`.

Parameter-free by construction — median-of-set vs median-of-set, no threshold, no window, no
new `config.py` knob. Absolute age is never an input (a concept isn't superseded for being
old; it's superseded when what disputes it is *demonstrably newer* than what supports it) —
matches `docs/specs/feature-7d-knowledge-currency.md` Decision 1.

## Why median, not mean

Mean is sensitive to one outlier year (a single very-old or very-new supporting/opposing doc
skews the whole aggregate); median is the natural "typical year on this side of the dispute"
and is what the sprint's "aggregate-newer" language points at. Both are permitted by the DoD
("median-vs-median or equivalent aggregate") — median was chosen for robustness, matching the
`min_degree` baseline's own preference for a distribution-relative statistic over a fixed
threshold (`gap_min_degree_2026-07.md`).

## Why not `>=`

`opp_median > sup_median` (strict), not `>=`. An opposing set whose median year **equals** the
supporting set's median is not demonstrably a *newer* trend — it's coincident evidence, which
is what `contested` already means. Strict `>` also makes the single-doc-per-side case
(`median([2020]) == median([2020])`) behave identically to "two docs published the same year,
one supports and one contradicts" — correctly `contested`, not a coin-flip `superseded_trend`.

## Fail-safe examples (guard-tested)

G3-only rule (median-vs-median, no confidence floor). **Superseded by G6 below** — the `{2018}` vs
`{2024}` row is the exact case G6 demotes; kept here as the historical G3-alone behavior.

| Supporting years | Opposing years | Result (G3 alone) | Test |
|---|---|---|---|
| `{2018}` vs `{2024}` (1 doc/side) | | `superseded_trend` | superseded by G6 — see below |
| `{2024}` | `{2018}` | `contested` | `test_older_or_equal_opposing_stays_contested` |
| `{2020}` | `{2020}` | `contested` | `test_equal_year_opposing_stays_contested` |
| `{2018}` | *(no year)* | `contested` | `test_missing_year_stays_contested_failsafe` |
| *(none)* | `{2024}` | `contested` | `test_sole_disputer_with_years_stays_contested` |
| *(meta has no `doc_years` key at all — pre-G3 skeleton.json)* | | `contested` | `test_pre_g3_skeleton_with_no_doc_years_key_is_byte_identical_behaviour` |

## G6 (SPRINT-006) — >= 2 dated docs per side confidence floor

`_aggregate_direction` now also requires `len(sup) >= MIN_DATED_DOCS_PER_SIDE (2)` and
`len(opp) >= 2` before treating median-vs-median as a meaningful aggregate — a median of one
document is not an aggregate. `MIN_DATED_DOCS_PER_SIDE` is a **named structural constant** in
`concept_skeleton.py`, not a `config.py` tunable (see the sprint doc for the rationale). All G3
fail-safes above are preserved verbatim; `epistemics.py` and `_graph_version` are unchanged.

| Supporting years | Opposing years | Result | Test |
|---|---|---|---|
| `{2017,2018}` | `{2023,2024}` (2 docs/side) | `superseded_trend` | `test_two_dated_per_side_newer_opposing_fires_superseded` |
| `{2018}` | `{2024}` (1 doc/side — the exact G3 fixture) | `contested` (demoted) | `test_single_disputer_one_supporter_now_stays_contested` |
| `{2018,2019}` (2 docs) | `{2024}` (1 doc) | `contested` (thin side gates it) | `test_thin_side_two_vs_one_stays_contested` |

Real-corpus before/after split + host-apply findings: see "Host apply" / "G3 before split" / "G6
after split" sections above.

## Cache invalidation

`concept_skeleton._graph_version` now hashes the sorted `doc_years` items alongside nodes/edges
/seed/resolution, so a `Document.year` backfill (metadata extraction discovering a year that
was previously null) changes `graph_version` even when no node or edge changed — a stale
`skeleton.json` doesn't silently keep serving pre-backfill `contested` calls. Guard test:
`test_graph_version_changes_when_doc_years_change`.

## Host apply — DONE (2026-07-08, same session as G6)

Real year coverage at build time: **45/47 documents have a year (96%)** — matches the
2026-07-08 backfill measurement.

**Planning gap found before this ran (see G6 addendum below for the full writeup):**
`build_concept_skeleton --apply` *alone* rebuilds `concept_edges` from scratch via
`cooccurrence_edges`/`add_citation_provenance`/`add_similarity_provenance` — none of which set
`relation`/`stance_by_doc` — so it silently **wipes the existing Node-B (LLM stance) annotations**
already on disk (verified empirically: a dry run showed `llm_relation 0` where the live
`skeleton.json` carried `node_b_calls: 17` / `n_llm_annotated_edges: 219`). Without stance data,
`n_contested_nodes`/`n_superseded_nodes` come out near-zero regardless of the year-aware rule —
not because the corpus lacks contested claims, but because the apply-alone command threw away the
evidence. **The correct host command is `build_concept_skeleton --apply --enrich`** (Node A + Node
B in one invocation, so Node A's fresh edges are re-annotated by Node B before anything is
written). Ran that: **46 LLM calls, 1254/1534 edges annotated, 1455 stance assertions, 55 contested
edges** (larger than the prior 17-call/219-edge run — presence/candidate pairs have grown since
that snapshot). $0, local Ollama (`llama3.1:8b`).

## G3 "before" split (year-aware rule, no G6 floor) — measured 2026-07-08

With the code as G3 shipped it (no `MIN_DATED_DOCS_PER_SIDE` gate):

| Metric | Value |
|---|---|
| Concept nodes weighted | 357 |
| Contested nodes | 226 |
| **Superseded-trend nodes** | **26** |

## G6 "after" split (>= 2 dated docs per side) — measured 2026-07-08

Same skeleton snapshot (no rebuild between before/after — see method note below), G6's guard
clause applied in `_aggregate_direction`:

| Metric | Value |
|---|---|
| Concept nodes weighted | 357 |
| Contested nodes | 226 (unchanged — the gate only changes `direction`, never `coverage`) |
| **Superseded-trend nodes** | **9** |

**17 of the 26 pre-gate fires (65%) were the thin single-doc-per-side case and are now demoted to
`contested`** — confirming the review finding that motivated this sprint: most fires on this
corpus were the thinnest possible evidence. This is a real, non-zero result — not the "premature
on this corpus" zero-fire outcome the sprint doc flagged as a possible reportable finding — but it
does confirm the underlying suspicion at a similar magnitude (a majority, not "some").

**Hand-audit of the 9 surviving fires** (all satisfy >= 2 dated docs per side by construction; spot
checked for a genuine year spread, not e.g. two docs sharing one year duplicated):

| Concept | Supporting years | Opposing years |
|---|---|---|
| abstractions | 2024, 2025, 2026 | 2025, 2026 |
| agent | 2021, 2022, 2024, 2025, 2026, 2026, 2026 | 2022, 2025, 2026, 2026, 2026 |
| psychology | 2020, 2024, 2026 | 2024, 2026 |
| pose estimation | 2019, 2019, 2020, 2021, 2022, 2022, 2022 | 2021, 2022 |
| rhythm | 2017, 2018, 2024, 2026 | 2017, 2024, 2026 |
| bank | 2021, 2022, 2024 | 2022, 2024 |
| political science | 2020, 2024 | 2024, 2026 |
| human pose | 2019, 2019, 2020, 2022 | 2019, 2021 |
| multi-agent | 2022, 2025, 2026 | 2025, 2026 |

All nine have a real multi-year spread on both sides (not one doc's year duplicated to clear the
floor) — genuine aggregate comparisons, not noise. **Aside, not a G6 finding:** three labels
(`psychology`, `bank`, `political science`) read as unusually generic/broad for a curated
technical vocabulary on this corpus — worth a look at the curated-vocabulary quality separately,
out of scope here (G6 gates evidence *count*, not label quality, per the sprint's own scope
boundary).

**Method note (why before/after used one skeleton build, not two):** `compute_epistemics` reads
`skeleton.json` directly and makes no LLM call, but `build_concept_skeleton --apply --enrich` does
(Node B is **not cached** — "idempotent" in this codebase means *reproducible*, not *skip-if-
unchanged*; re-running always re-calls the LLM, and temp-0 llama is documented as "near-stable, not
guaranteed" byte-for-byte). Running the host apply twice (once pre-G6-code, once post) would have
compared two *different* LLM-derived stance graphs, confounding the gate's effect with run-to-run
LLM variance. Instead: built the skeleton **once** (`--apply --enrich`), then toggled
`MIN_DATED_DOCS_PER_SIDE` in/out of `src/doc_assistant/concept_skeleton.py` (via a local `git
stash`) and re-ran only the free, LLM-free `compute_epistemics --apply` for each side — an
apples-to-apples comparison on the identical stance data.

**Not fixed here, flagged separately:** `build_epistemics` reported **0 chunks with a claim**
against this real skeleton, despite `load_doc_chunks()` returning all 6215 real chunks correctly
in isolation. Root cause: `epistemics.concepts_in_text` matches skeleton node **ids** literally
against chunk text, and the curated `concept_skeleton.py` uses `Concept.id` — a UUID — as the node
id (the retired `concept_graph.py` used `canonical_key(label)`, a real lowercase string, which is
why this worked before G1). A UUID never appears in chunk text, so the live answer-time marker
surfacing (PR-M1) has been silently dark on the real corpus since the G1 re-point — independent of
G3/G6, and out of scope for this sprint (`epistemics.py` is explicitly unchanged by G3/G6). Logged
as `.claude/KNOWN_ISSUES.md` KI-15.
