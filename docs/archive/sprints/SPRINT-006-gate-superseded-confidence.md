<!-- status: archived · updated: 2026-07-09 · class: disposable -->
<!-- Follow-on to SPRINT-003 (G3). G3 shipped the parameter-free median-vs-median superseded_trend
     rule as a *measurable instrument* and deliberately left the single-doc-per-side case firing
     (1 old supporter vs 1 new disputer -> superseded_trend). Review flagged that case as the
     weakest evidence in the rule: median-of-one is not an aggregate, and on a 47-doc corpus most
     fires are likely to be exactly these 1-v-1 coin-flips. This sprint gates the marker behind a
     structural confidence floor (>= 2 dated docs per side) and validates the before/after fire-set
     against the real corpus — the validation G3 made possible but did not itself perform. -->

# SPRINT-006 — gate-superseded-confidence

- **base:** main
- **depends-on:** SPRINT-003 (G3) — landed; `_aggregate_direction` + `skeleton.meta["doc_years"]`
  are in place. **Also depends on G3's host apply having run** (`build_concept_skeleton --apply`
  + `compute_epistemics --apply` on the real `data/library.db`), so the **ungated** superseded/
  contested split is recorded in `tests/eval/baselines/superseded_year_rule_2026-07.md` — that
  recorded fire-set is the *before* this sprint's gate is measured against. If that number is not
  yet captured, capture it first (it is a one-command host run, not a code change).
- **DoD:** `_aggregate_direction` promotes a contested node to `direction="superseded_trend"` **only
  when both sides carry at least two dated documents** (`len(sup) >= 2 AND len(opp) >= 2`) *in
  addition to* the existing strict `median(opp) > median(sup)` test. A node that clears the median
  test but has a thin side (1 supporter, or 1 disputer, or 1-v-1) now stays `contested` — the fire
  that G3 produced for the single-doc case is demoted, never a coin-flip on median-of-one. The
  floor is a **structural minimum, not a tunable knob**: median-vs-median is only a meaningful
  *aggregate* at n>=2 per side, so `2` is the definitional smallest value, justified in the baseline
  and **not** added to `config.py`'s locked-settings table (no eval-harness ceremony — a value > 2,
  or a year-gap/window threshold, WOULD be a tunable and is explicitly out of scope; if one proves
  unavoidable that is a further scope change, raised not silently added). The `2` lives as a named
  module constant (e.g. `MIN_DATED_DOCS_PER_SIDE`) with a docstring stating it is a confidence
  floor, not a retrieval knob. All G3 fail-safes are preserved verbatim: any missing year on either
  side, or no supporting docs, still returns `contested`; a year-less (pre-G3) `skeleton.json` still
  yields byte-identical `stable`/`contested`/`unique` behaviour; `epistemics.py` remains **unchanged**
  (it already consumes `.direction` at `epistemics.py:159`/`:422`). `_graph_version` is **unchanged**
  by this sprint (the floor is a read-time weight rule, not a serialized field — no cache-key churn).
  Guard tests cover: `>= 2`-per-side newer-opposing still fires; the exact 1-v-1 fixture that fired
  under G3 now stays `contested`; a `2-supporters-vs-1-disputer` thin side stays `contested`; the
  end-to-end marker still surfaces through `build_epistemics` for a `>= 2`-per-side fixture. **Host
  validation (the point of this sprint):** re-run the host apply and record in the baseline the
  *after* split (`n_superseded_nodes` under the gate) beside G3's *before*, plus a hand-audit note on
  the surviving fires (are they genuine multi-doc supersessions, not noise?). If the gate takes the
  fire-rate to zero, that is a **reportable finding**, not a silent pass — it means the feature is
  premature on the current corpus and the marker should stay dark until the corpus grows (record it,
  do not tune `2` down to force a fire). Full gate green (ruff / ruff format / mypy --strict / bandit
  0 HIGH / pytest); nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). concept_skeleton.py is the sole write target in src/;
     epistemics.py is read only for the unchanged consume-side call sites (:159/:422). -->
- docs/specs/feature-7d-knowledge-currency.md
- tests/eval/baselines/superseded_year_rule_2026-07.md
- src/doc_assistant/concept_skeleton.py
- src/doc_assistant/epistemics.py
- scripts/build_concept_skeleton.py

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/concept_skeleton.py
- tests/unit/test_concept_skeleton_weights.py
- tests/integration/test_compute_epistemics.py
- tests/eval/baselines/superseded_year_rule_2026-07.md

## contracts
<!-- type: target [ | when: <glob> ]  —  test = verify-loop reminder -->
- test: tests/unit/test_concept_skeleton_weights.py::two_dated_per_side_newer_opposing_fires_superseded
- test: tests/unit/test_concept_skeleton_weights.py::single_disputer_one_supporter_now_stays_contested
- test: tests/unit/test_concept_skeleton_weights.py::thin_side_two_vs_one_stays_contested
- test: tests/integration/test_compute_epistemics.py::superseded_marker_requires_two_per_side

## docs
<!-- must be touched when this lands (the docs gate enforces these appear in the diff) -->
- docs/DEVLOG.md
- docs/ROADMAP.md

<!--
Scope boundary (NOT machine-read):
- The ONLY behavioral change is one guard clause in `_aggregate_direction` (concept_skeleton.py:721):
  add `if len(sup) < 2 or len(opp) < 2: return "contested"` (equivalently a single combined guard).
  Because the pre-existing fail-safe already requires EVERY doc in sup|opp to be dated, `len(sup) >= 2`
  is exactly ">= 2 *dated* supporters" — no separate dated-count is needed. Update the docstring to
  state the floor + why (median-of-one is not an aggregate).
- The `2` is a NAMED CONSTANT with a rationale docstring, NOT a `config.py` locked/tunable setting.
  This is load-bearing for staying off the eval-harness treadmill: `2` is definitional (the minimum
  n for a median to aggregate), not empirically tuned. Anything other than the definitional minimum
  (>= 3, a min-year-gap, a window) is a tunable and OUT of scope — raise it as its own sprint.
- Do NOT touch `_graph_version`, serialization, `load_doc_years`, or `epistemics.py`. The floor is a
  read-time weight decision; the skeleton artifact and its cache key are unchanged. An unchanged
  epistemics.py is again part of the proof.
- G3's existing tests must all still pass EXCEPT the ones asserting the 1-v-1 single-doc case fires
  superseded (`test_newer_opposing_makes_superseded` uses 1-supporter-vs-1-disputer and WILL flip to
  contested under the floor). Update those fixtures to >= 2 dated docs per side to keep asserting the
  fire, and ADD the demotion assertions rather than deleting the coverage. Re-home the affected rows
  of the baseline's fail-safe table accordingly (the `{2018}` vs `{2024}` row is now `contested`).
- Host steps (KI-5, host not sandbox): re-run `build_concept_skeleton --apply` + `compute_epistemics
  --apply` and record the after-gate split beside G3's before in the baseline. This is the user's run
  after review; the sprint delivers code + tests + baseline scaffold, the host produces the numbers.
- OUT of scope: any LLM/Node-B change, `gap_suggest.py`, touching `epistemics.py`, and any change to
  the underlying stance extraction (the quality of `contradicts`/`supersedes` labels is a separate,
  larger question — this sprint gates on *evidence count*, not on stance-label confidence).
-->
