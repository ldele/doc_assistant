<!-- status: active · updated: 2026-07-07 · class: disposable -->

# SPRINT-001 — retire-concept-graph

- **base:** main
- **DoD:** (1) `concept_graph.py` + `scripts/build_concept_graph.py` + `tests/unit/test_concept_graph.py` are deleted, and `grep -rEn "import .*concept_graph|from .*concept_graph|graph_from_dict|GRAPH_NAME" src/ scripts/` returns **zero** matches (no import/load sites remain; surviving docstring/log-event *string* mentions are fine and must be reworded to point at `build_concept_skeleton`). (2) `epistemics.py` sources node-weights from `concept_skeleton.node_weights_for_epistemics` and `wiki.py` sources doc-clusters from the skeleton's Louvain communities; `python -m scripts.compute_epistemics` and `python -m scripts.build_wiki` run against `skeleton.json` (their `build_concept_graph` help-text/hints repointed to `build_concept_skeleton`). (3) `EPISTEMICS_MARKERS_ENABLED` defaults `true`; `tests/unit/test_chat_controller.py::test_markers_disabled_by_default` is updated to the new default (renamed/repurposed to assert markers are ON by default) and a fixture turn surfaces a `contested`/`superseded_trend` chip when the skeleton holds an opposing-stance edge, none when it does not. (4) Full gate green: `ruff` · `ruff format` · `mypy --strict src` · `bandit` 0 HIGH · `pytest tests/unit tests/integration` (suite drops `test_concept_graph.py`, no new failures). (5) KI-7 → RESOLVED in `.claude/KNOWN_ISSUES.md`; ADR-005 status note updated to "default-on, markers rest on Node-B skeleton"; DEVLOG + baton entries added. Nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set: the files the agent should load for this sprint, nothing else (cap 12 files / 2500 lines).
     Trimmed to fit: concept_graph.py (1145) is unavoidable — it is being retired. concept_skeleton.py
     is NOT read whole (only its node_weights_for_epistemics/community seam — see the ordering note);
     wiki.py's cluster seam is line 380 only; the redesign spec is a reference, not a full read. -->
- src/doc_assistant/concept_graph.py
- src/doc_assistant/epistemics.py
- src/doc_assistant/config.py
- scripts/build_concept_skeleton.py
- docs/decisions/ADR-005-epistemics-markers-default-off.md

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/concept_skeleton.py
- src/doc_assistant/epistemics.py
- src/doc_assistant/wiki.py
- src/doc_assistant/config.py
- scripts/compute_epistemics.py
- scripts/build_wiki.py
- tests/unit/test_epistemics.py
- tests/unit/test_chat_controller.py
- tests/integration/test_compute_epistemics.py
- tests/integration/test_build_wiki.py

## contracts
<!-- type: target [ | when: <glob> ]  —  test = verify-loop reminder; snap/map must co-change.
     Idents verified to exist 2026-07-07 against the current test files. -->
- test: tests/unit/test_epistemics.py
- test: tests/unit/test_chat_controller.py::test_markers_absent_is_byte_identical
- test: tests/unit/test_chat_controller.py::test_markers_disabled_by_default
- test: tests/unit/test_concept_skeleton_weights.py::test_sole_source_is_unique_never_contested

## docs
<!-- must be touched when this lands (the docs gate enforces these appear in the diff) -->
- docs/DEVLOG.md
- .claude/KNOWN_ISSUES.md
- docs/ROADMAP.md
- docs/decisions/ADR-005-epistemics-markers-default-off.md

<!--
Ordering note (load-bearing — NOT machine-read): NodeWeight/compute_node_weights currently LIVE in
concept_graph.py and are imported back by concept_skeleton.py:40. Re-home the shared vocab first (into
concept_skeleton.py or a new concept_types.py — if the latter, add it to affects + record a one-line
ADR), fix the self-import, gate green; THEN re-point epistemics.py (sig types at 144/172-173/232 +
the load at 402-413) and wiki.py:380 (doc_clusters_from_graph → skeleton communities); THEN delete
concept_graph.py + scripts/build_concept_graph.py + tests/unit/test_concept_graph.py (add these three
deletions to affects at build time); THEN flip the config default. wiki.py edit is the cluster seam
ONLY — do not disturb its LLM synthesis path (wiki.py:495). Deleting concept_graph.py also removes
scripts/build_concept_graph.py's import — retire that script in the same sprint.
-->
