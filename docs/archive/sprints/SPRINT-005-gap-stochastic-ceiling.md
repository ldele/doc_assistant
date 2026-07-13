<!-- status: archived · updated: 2026-07-08 · class: disposable -->
<!-- LANDED 2026-07-08 on the RTX/Ollama box. All DoD items met; gate green (773 passed, +18 tests).
     The deferred real-Ollama smoke run also completed here (12/12 under_connected concepts
     suggested, $0; tests/eval/baselines/gap_suggest_ollama_2026-07-08.md). SPRINT-004 (G4, the
     sibling contract this note originally said to archive first) was NOT touched — it needs a
     TLS-MITM proxy box this machine isn't, so it stays `status: active`, genuinely un-landed, not
     just un-archived-for-hygiene. sprint_check now sees exactly 1 active contract (SPRINT-004),
     which is the target state, not a violation of the archive-on-land convention. -->

# SPRINT-005 — gap-stochastic-ceiling

- **base:** main
- **depends-on:** G2 (SPRINT-002) — landed; the deterministic Tier-1 + Tier-2a floor + the `Gap`/`GapRow`/`build_gaps` seam it plugs into.
- **DoD:** new `src/doc_assistant/gap_suggest.py` implements the Tier-2a **stochastic ceiling** (ADR-004 Decision 4 / `feature-gap-detection.md:123-132`): `suggest_for_thin(gaps, skeleton, client: LLMClient, *, …) -> list[Gap]` routes each Tier-1 `under_connected` node through **one quarantined LLM call, handed only the concept and its present neighbours**, returning `suggested_link` / `suggested_concept` / `thin_area` `Gap`s carrying `determinism="stochastic"`, a `rating`, `status="surfaced"`, and the exact LLM inputs in `evidence`. The module **makes no provider decision** (takes an already-built `LLMClient`, the `concept_skeleton_enrich.py` precedent), **never creates a `Concept`/edge/skeleton node**, and **never auto-writes the graph** — it only emits `Gap`s for promotion (quarantine guard). `build_gaps(..., suggest=True)` replaces the `NotImplementedError` stub (`gaps.py:288-293`): deterministic detectors still run and their rows still rebuild via `_write_gap_rows`' `delete(...).where(determinism=="deterministic")` filter **left untouched**; stochastic rows are written by a **dedicated upsert path** that does NOT widen that delete — a re-run neither duplicates a suggestion nor downgrades an already `promoted`/`dismissed` stochastic row (`status` is the compounding arrow, must survive both a deterministic rebuild AND a re-suggest). `scripts/build_gaps.py`'s `--suggest` gains `--provider`/`--model` (default from new **Ollama-default** `GAP_SUGGEST_LLM_PROVIDER`/`GAP_SUGGEST_LLM_MODEL` config, mirroring `CONCEPT_SKELETON_LLM_*`), routes the apply through **`llm.assert_provider_intent(...)` before any client is constructed** (the `build_concept_skeleton._run_node_b` precedent), and makes **zero LLM calls on a dry run**. **No live paid API call in any test** (cpc §13): a scripted `LLMClient` (the `test_concept_skeleton_enrich.py::FakeClient` pattern) proves the logic offline — stochastic label + non-None rating + inputs-in-evidence, the quarantine (input skeleton byte-identical after the call), graceful degradation (one malformed/failed call doesn't sink the run), and a zero-under-connected → zero-calls guard; the integration `--suggest`-on path uses the scripted client with `DOC_ASSUME_YES=1` / `abort_seconds=0` so `assert_provider_intent` never blocks and nothing bills. **No `GapRow` / `_ADDITIVE_COLUMNS` change** — `rating`/`evidence_json`/`status` already exist (`create_all` from G2 covers the table). Full gate green on this box (ruff / ruff format / mypy --strict / bandit 0 HIGH / pytest); nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). concept_skeleton_enrich.py + build_concept_skeleton._run_node_b
     are the confined-LLM precedent to COPY (provider guard, token budget, per-item try/except, FakeClient
     test shape); read _run_node_b (build_concept_skeleton.py:153-198), not the whole build. -->
- docs/specs/feature-gap-detection.md
- docs/decisions/ADR-004-gap-detection-layer.md
- docs/specs/llm-provider-isolation.md
- src/doc_assistant/gaps.py
- src/doc_assistant/llm.py
- src/doc_assistant/concept_skeleton_enrich.py
- scripts/build_gaps.py
- scripts/build_concept_skeleton.py

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/gap_suggest.py
- src/doc_assistant/gaps.py
- scripts/build_gaps.py
- src/doc_assistant/config.py
- .env.example
- tests/unit/test_gap_suggest.py
- tests/integration/test_build_gaps.py

## contracts
<!-- type: target [ | when: <glob> ]  —  test = verify-loop reminder; snap/map must co-change -->
- test: tests/unit/test_gap_suggest.py::stochastic_label_rating_and_inputs
- test: tests/unit/test_gap_suggest.py::quarantine_never_mutates_the_skeleton
- test: tests/integration/test_build_gaps.py::suggest_on_writes_stochastic_without_a_paid_call
- test: tests/integration/test_build_gaps.py::promoted_stochastic_survives_rebuild_and_resuggest

## docs
<!-- must be touched when this lands (the docs gate enforces these appear in the diff) -->
- docs/DEVLOG.md
- docs/ROADMAP.md

<!--
Scope boundary (NOT machine-read):
- IN scope = the Tier-2a CEILING only: a within-corpus, quarantined LLM pass over Tier-1 `under_connected`
  nodes that only SUGGESTS (link/concept/thin_area), rated + never auto-written (ADR-004 Decisions 1/4/5).
  Reviewer answer-level `failure_tag` routing (models.py AnswerReview.failure_tag) is a spec-optional EXTRA —
  keep the core on `under_connected` nodes to bound scope; a failure_tag feed is a follow-on, not this sprint.
- OUT of scope = Tier-2b, the external "anti-blind-spot" reach (ADR-004 Decision 7 / feature-gap-detection.md
  :198-200). The idea-generator is REJECTED for it (ADR-004 option 3 — "a convex-hull filler, not a
  frontier-crosser"). gap_suggest.py must NOT reach outside the curated space.
- The `_write_gap_rows` deterministic delete-filter (`gaps.py:252`) is LOAD-BEARING and stays byte-identical.
  Stochastic rows get their OWN write path (upsert by suggestion identity, status-preserving). Do not route
  stochastic gaps through the deterministic replace — that would clobber the compounding arrow. Extend the
  existing `test_stochastic_rows_survive_a_deterministic_rebuild` (test_build_gaps.py:182-204) to also cover
  a re-suggest, and have the ceiling actually WRITE the stochastic row (today the test hand-seeds it).
- Provider isolation is mandatory (KI-4 credit-leak): Ollama-default config, `assert_provider_intent` before
  `make_client`, zero LLM calls in a dry run, NO paid call in tests. There is no `--allow-paid` flag; the
  abort window is `abort_seconds` + the `DOC_ASSUME_YES` bypass (llm.py:219-286).
- MACHINE REALITY: everything above is built + proven offline on THIS CPU box with a scripted `LLMClient`
  (no Ollama). The REAL `--suggest --apply` smoke against a local model — first real stochastic rows + a
  baseline note (provider/model, calls, suggestions surfaced, a manual spot-check of quality) — is a HOST
  step on the RTX/Ollama box, DEFERRED and NOT a gate for landing this code (cost-discipline: prove offline
  first; Node B was built the same way). Do NOT run a paid Anthropic `--provider anthropic` pass to validate.
-->
