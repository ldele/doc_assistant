<!-- status: archived · updated: 2026-07-09 · class: disposable -->
<!-- UN-PARKED 2026-07-08. The park condition below — "un-park once the corpus carries real year
     metadata (`extract_doc_metadata --apply` → measure coverage)" — is now MET: the backfill ran on
     the RTX box (2026-07-08), giving 45/47 active docs a year (96%, range 2012-2026), far above the
     "likely too thin" premise this was parked on. G3 is now the next actionable build sprint (CPU-box,
     $0, deterministic). G4/SPRINT-004 stays active-but-hardware-blocked (needs a TLS-MITM proxy box).
     ORIGINAL PARK NOTE (2026-07-07, kept for the record): parked as a low-yield veneer on the
     assumption `Document.year` coverage was too thin for `superseded_trend` to fire — that assumption
     is now disproven by the 96% measurement. -->

# SPRINT-003 — year-aware-superseded

- **base:** main
- **depends-on:** SPRINT-001 (G1) — landed; the skeleton is now the single node-weight source for `epistemics.py`.
- **DoD:** `node_weights_for_epistemics` can emit `direction="superseded_trend"` when a contested node's **contradicting** documents are, in aggregate, **newer** than its **supporting** documents (relative polarity-over-time, per `feature-7d-knowledge-currency.md` Decision 1 — *absolute age is never an input*); a node contradicted by same-age-or-older docs stays `contested`; a node missing any required year stays `contested` (fail-safe — never superseded on incomplete year data). The comparison is **parameter-free** (median-vs-median or equivalent aggregate; the exact rule is locked in-sprint and recorded in the baseline — **no new eval-locked or tunable setting is introduced**). The skeleton artifact round-trips per-document years (`skeleton_to_dict` ⇄ `skeleton_from_dict`); a year-less (pre-sprint) `skeleton.json` still deserializes and yields **byte-identical** `stable`/`contested`/`unique` behaviour (back-compat); `_graph_version` changes when doc-years change so a rebuild busts the cache. `build_concept_skeleton` loads `Document.year` and attaches a doc→year map **once** at build time; corpus year coverage (`SELECT count(*) FROM documents WHERE year IS NOT NULL`) is measured and recorded in the baseline (if coverage is inadequate, `extract_doc_metadata --apply` is a documented host precondition). End-to-end: a fixture whose contradicting docs are newer produces a `superseded_trend` **marker** through `epistemics.build_epistemics` with **no change to `epistemics.py`** (it already consumes the direction at `epistemics.py:412`/`:159`/`:422`). Year data lives at the **skeleton/meta level, not on `ConceptNode`**, so the positional `ConceptNode(...)` fanout across the test suite is not broken. Full gate green (ruff / ruff format / mypy --strict / bandit 0 HIGH / pytest); nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). concept_skeleton.py IS the primary write target and must be
     read; epistemics.py is read only for the consume-side call site (:159/:412/:422 — it does NOT change). -->
- docs/specs/feature-7d-knowledge-currency.md
- src/doc_assistant/concept_skeleton.py
- src/doc_assistant/epistemics.py
- src/doc_assistant/db/models.py
- scripts/build_concept_skeleton.py
- scripts/extract_doc_metadata.py

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/concept_skeleton.py
- scripts/build_concept_skeleton.py
- tests/unit/test_concept_skeleton_weights.py
- tests/unit/test_concept_skeleton.py
- tests/integration/test_compute_epistemics.py
- tests/eval/baselines/superseded_year_rule_2026-07.md

## contracts
<!-- type: target [ | when: <glob> ]  —  test = verify-loop reminder; snap/map must co-change -->
- test: tests/unit/test_concept_skeleton_weights.py::newer_opposing_makes_superseded
- test: tests/unit/test_concept_skeleton_weights.py::older_or_equal_opposing_stays_contested
- test: tests/unit/test_concept_skeleton_weights.py::missing_year_stays_contested_failsafe
- test: tests/integration/test_compute_epistemics.py::superseded_row_and_marker

## docs
<!-- must be touched when this lands (the docs gate enforces these appear in the diff) -->
- docs/DEVLOG.md
- docs/ROADMAP.md

<!--
Scope boundary (NOT machine-read):
- The ONLY behavioral seam that unlocks the marker is the `direction` assignment at
  `concept_skeleton.py:675` (`direction = "contested" if nc >= 1 else "stable"`), gated on making
  per-document years reachable inside `node_weights_for_epistemics` (whose sole arg today is `skeleton`).
- Thread years at the SKELETON level (a `doc_years: dict[str, int]` on `ConceptSkeleton`/`meta`, default
  empty), NOT as a new positional field on `ConceptNode` — the Explore map flagged that every positional
  `ConceptNode(...)` constructor in tests/unit/test_concept_skeleton.py (+ the weights/epistemics fixtures)
  breaks if ConceptNode gains a field. Keeping year off ConceptNode keeps the blast radius to serialization.
- `node_weights_for_epistemics` reads each edge's `stance_by_doc` (doc_id, polarity) and looks the year up
  in the skeleton-level map — supporting vs opposing document-year sets, aggregate-compared.
- Parameter-free by design so the locked-settings eval ceremony does NOT apply. If a threshold proves
  unavoidable, that is a scope change — raise it, do not silently add a knob.
- OUT of scope: any LLM/Node-B change (this is deterministic, CPU-box, $0), `gap_suggest.py`, and touching
  `epistemics.py` (it already consumes the direction — an unchanged epistemics.py is part of the proof).
- Host steps (KI-5, run on host not sandbox): confirm year coverage, then `build_concept_skeleton --apply`
  + `compute_epistemics --apply` to bake years into the on-disk skeleton + regenerate the sidecar. These
  are the user's runs after review — the sprint delivers the code + tests + baseline, not the host apply.
- Back-compat invariant is load-bearing: an old year-less skeleton.json MUST still load and behave exactly
  as today (superseded unreachable), so no forced re-build is imposed on existing corpora.
-->
