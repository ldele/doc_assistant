<!-- status: active · updated: 2026-07-07 · class: disposable -->
<!-- QUEUED behind SPRINT-001 (G1). Both contracts sit `active` on disk, which is fine, but
     sprint_check ERRORs on >1 active contract when run bare. While G1 is in flight, scope the gate
     explicitly: `python -m cpc.sprint_check --root . --contract docs/sprints/SPRINT-001-retire-concept-graph.md`.
     Start G2 only after G1 lands (retirement gives the gap layer a single skeleton to define against). -->

# SPRINT-002 — gap-layer-deterministic

- **base:** main
- **depends-on:** SPRINT-001 (G1) — the KI-7 retirement must land first.
- **DoD:** `python -m scripts.build_gaps --apply` on the `data/` corpus writes deterministic `gaps` rows; an idempotent re-run is a no-op (row count + `graph_version` unchanged) and makes zero LLM calls; a degree-0 fixture concept → one `isolated` Gap, a sole-source concept → `single_source` (rating None, flagged-not-penalized), a cut edge → `thin_bridge`, a below-`min_degree` concept → `under_connected`; fixture `answer_claims` with `marker=="unsupported"` aggregate to `unsourced_claim` gaps carrying the contributing ids in `evidence`, cited claims produce none; every Gap carries `determinism="deterministic"`; the three gap types (`unsupported`/`contested`/`superseded_trend`) stay distinct; retrieval output byte-identical (public-eval fixture); `min_degree` set from the corpus degree distribution and recorded in a baseline note (not guessed); full gate green; nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set: the files the agent should load for this sprint, nothing else (cap 12 files / 2500 lines).
     concept_skeleton.py (1076) is NOT read whole — the gap layer consumes its documented public seam
     (Louvain communities + edge list), which feature-gap-detection.md specifies; reading the spec is
     enough. That keeps the read-set at ~1625 lines, well under the cap. -->
- docs/specs/feature-gap-detection.md
- docs/decisions/ADR-004-gap-detection-layer.md
- src/doc_assistant/synthesis.py
- src/doc_assistant/db/models.py
- src/doc_assistant/db/migrations.py
- scripts/build_concept_skeleton.py

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/gaps.py
- src/doc_assistant/db/models.py
- src/doc_assistant/db/migrations.py
- scripts/build_gaps.py
- tests/unit/test_gaps.py
- tests/unit/test_gaps_floor.py
- tests/integration/test_build_gaps.py
- tests/eval/baselines/gap_min_degree_2026-07.md

## contracts
<!-- type: target [ | when: <glob> ]  —  test = verify-loop reminder; snap/map must co-change -->
- test: tests/unit/test_gaps.py::single_source_flagged_not_penalized
- test: tests/unit/test_gaps_floor.py::unsourced_claim_aggregation
- test: tests/integration/test_build_gaps.py::idempotent_no_llm

## docs
<!-- must be touched when this lands (the docs gate enforces these appear in the diff) -->
- docs/DEVLOG.md
- docs/ROADMAP.md

<!--
Scope boundary (NOT machine-read): gap_suggest.py — the Tier-2a STOCHASTIC ceiling (quarantined LLM
suggestion pass) — is OUT of this sprint and deliberately absent from `affects` so the write-set gate
blocks it. Deterministic Tier-1 (isolated / single_source / thin_bridge / under_connected) + the
Tier-2a deterministic FLOOR (unsourced_claim / citation_missing) only. build_gaps.py may carry a
--suggest flag stub that RAISES "stochastic ceiling — separate sprint", so the seam exists without the
LLM path. GapRow is a NEW table → create_all handles it (no _ADDITIVE_COLUMNS entry; the
figures/chunk_epistemics precedent). Depends on SPRINT-001 (retirement) landing first so the gap layer
is defined against the single skeleton. Reads: concept_skeleton (Louvain communities + edges),
AnswerClaim.marker == synthesis.MARKER_UNSUPPORTED, Citation. min_degree: derive from the skeleton's own
degree distribution on this corpus (the same-domain-embedding brittle-absolute-threshold lesson —
CLAUDE.md / atlas), record in the baseline note, do not hardcode a guessed absolute.
-->
