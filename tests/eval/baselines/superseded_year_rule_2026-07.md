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

| Supporting years | Opposing years | Result | Test |
|---|---|---|---|
| `{2018}` | `{2024}` | `superseded_trend` | `test_newer_opposing_makes_superseded` |
| `{2024}` | `{2018}` | `contested` | `test_older_or_equal_opposing_stays_contested` |
| `{2020}` | `{2020}` | `contested` | `test_equal_year_opposing_stays_contested` |
| `{2018}` | *(no year)* | `contested` | `test_missing_year_stays_contested_failsafe` |
| *(none)* | `{2024}` | `contested` | `test_sole_disputer_with_years_stays_contested` |
| *(meta has no `doc_years` key at all — pre-G3 skeleton.json)* | | `contested` | `test_pre_g3_skeleton_with_no_doc_years_key_is_byte_identical_behaviour` |

## Cache invalidation

`concept_skeleton._graph_version` now hashes the sorted `doc_years` items alongside nodes/edges
/seed/resolution, so a `Document.year` backfill (metadata extraction discovering a year that
was previously null) changes `graph_version` even when no node or edge changed — a stale
`skeleton.json` doesn't silently keep serving pre-backfill `contested` calls. Guard test:
`test_graph_version_changes_when_doc_years_change`.

## Pending (host apply — not this sprint's deliverable)

Per the sprint's scope: `build_concept_skeleton --apply` (now loads `Document.year` via the new
`load_doc_years()`) + `compute_epistemics --apply` on the real corpus, then record here:

- Real year coverage at build time (`len(doc_years)` vs `n_documents` — the CLI report's new
  "Documents with a year" line surfaces this without a manual SQL query).
- The resulting `n_contested_nodes` / `n_superseded_nodes` split from `build_epistemics`'s
  `EpistemicsResult`, to see whether the rule fires materially or stays near-zero (the original
  2026-07-07 park note's "low-yield veneer" concern) now that coverage is 96%
  (SESSION.md 2026-07-08).
