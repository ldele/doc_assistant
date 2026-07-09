<!-- status: archived · updated: 2026-07-09 · class: disposable -->
<!-- KI-15 (.claude/KNOWN_ISSUES.md): epistemics.concepts_in_text matches skeleton node *ids*
     literally against chunk text. That was correct for the retired open-vocabulary
     concept_graph.py, whose node id WAS canonical_key(label) — a real lowercase string. The
     curated concept_skeleton.py (Node A, PR-A) that replaced it uses the `Concept.id` UUID
     primary key as the node id instead, so the match can never fire on the real corpus: a UUID
     never appears in chunk text. Node-level weights (coverage/direction, G3/G6) are unaffected —
     this bug is entirely in the projection step that turns those weights into chunk_epistemics
     rows, which is what the live answer-time contested/superseded_trend chips (PR-M1) read. Found
     during G6's real-corpus validation (build_epistemics reported 0 chunks with a claim against a
     skeleton with 226 contested / 9 superseded_trend nodes). -->

# SPRINT-007 — fix-epistemics-label-attribution

- **base:** main
- **depends-on:** none structurally, but validation is far more meaningful with a populated
  skeleton — SPRINT-006 (G6) already left `data/skeleton/skeleton.json` on this box with fresh
  `doc_years` + Node-B stance data (46 LLM calls, 1254/1534 edges annotated) from its own host
  apply. No new Node A/B run is needed for this sprint's validation; `compute_epistemics --apply`
  is free (no LLM) and re-reads that same skeleton.
- **DoD:** `epistemics.concepts_in_text` attributes a concept to a chunk by matching its
  **label** (word-boundary, casefolded), not its node id. The boundary-matching regex is **shared**
  with `concept_skeleton`'s existing R2 presence matcher (alnum lookarounds, not `\b` — `\b`
  mishandles non-word edge chars like `gpt-4`/`c++`, the exact bug R2 already fixed once for Node
  A; this sprint must not reintroduce it in `epistemics.py`) via one extracted public function,
  not two independent regex implementations. `project_chunk_weights` passes a `{node_id: label}`
  map instead of a bare id list. Every existing `concepts_in_text`/`project_chunk_weights` test
  behavior is preserved (same word-boundary semantics, same substring-false-positive avoidance,
  same de-dup, same short-form skip) — only the thing being matched changes, from id to label.
  Guard tests must cover the exact gap the old fixtures didn't: a UUID-shaped id whose *label*
  appears in the text gets attributed (the old tests used ids that were themselves valid labels,
  e.g. `"bm25"`, which is why this bug shipped unnoticed). **Real-corpus validation (the point of
  this sprint):** re-run `compute_epistemics --apply` against the existing on-disk skeleton (no
  rebuild) and confirm `chunks_with_a_claim` and `n_chunks_marked` are materially non-zero — record
  the counts + a handful of sample marked rows in a new baseline. `concept_skeleton.py`'s Node-A
  presence matching (`match_presence`) must stay behaviorally byte-identical — the shared function
  is an extraction of existing logic, not a rewrite; guarded by the existing Node-A test suite
  passing unchanged. `_aggregate_direction`/G6, `_graph_version`, and the skeleton write path are
  untouched — this sprint is entirely on the read/projection side. Full gate green (ruff / ruff
  format / mypy --strict / bandit 0 HIGH / pytest); nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). -->
- .claude/KNOWN_ISSUES.md
- docs/specs/pr-m1-epistemics-markers.md
- src/doc_assistant/concept_skeleton.py
- src/doc_assistant/epistemics.py
- tests/unit/test_epistemics.py

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/concept_skeleton.py
- src/doc_assistant/epistemics.py
- tests/unit/test_epistemics.py
- tests/integration/test_compute_epistemics.py
- tests/eval/baselines/epistemics_label_attribution_2026-07.md

## contracts
<!-- type: target [ | when: <glob> ]  —  test = verify-loop reminder -->
- test: tests/unit/test_epistemics.py::test_concepts_in_text_matches_label_not_uuid_id
- test: tests/unit/test_epistemics.py::test_concepts_in_text_boundary_handles_nonword_edge_chars
- test: tests/unit/test_concept_skeleton.py::test_compile_boundary_pattern_matches_presence_matchers_output
- test: tests/integration/test_compute_epistemics.py::test_uuid_id_node_is_attributed_via_label_end_to_end

## docs
<!-- must be touched when this lands (the docs gate enforces these appear in the diff) -->
- docs/DEVLOG.md
- docs/ROADMAP.md
- .claude/KNOWN_ISSUES.md

<!--
Scope boundary (NOT machine-read):
- The behavioral change is confined to `epistemics.py`'s attribution step
  (`concepts_in_text` + its one caller `project_chunk_weights`) plus a small, non-behavioral
  extraction in `concept_skeleton.py`: pull the existing inline boundary-regex expression out of
  `_presence_matchers` into a public `compile_boundary_pattern(form: str) -> re.Pattern[str]`,
  and have `_presence_matchers` call it. `_presence_matchers`'s own behavior must not change —
  this is a refactor-extraction so `epistemics.py` can reuse the *same* regex definition instead
  of hand-rolling a second (and, per the module's current code, worse — `\b`-based) one.
- `concepts_in_text`'s new signature is `(text: str, labels_by_id: dict[str, str]) -> list[str]`
  (was `(text: str, node_ids: list[str]) -> list[str]`). This is a breaking signature change to a
  function with exactly one production caller (`project_chunk_weights`, in this same module) — no
  back-compat shim needed, per the project's stated preference for changing code over
  compatibility shims.
- Do NOT touch `_aggregate_direction`, `MIN_DATED_DOCS_PER_SIDE`, `_graph_version`, or any
  skeleton *write* path (`build_concept_skeleton`, `write_skeleton`, `concept_skeleton_enrich.py`)
  — G3/G6 already validated those; this sprint is entirely downstream, on the projection that
  turns already-correct node weights into `chunk_epistemics` rows.
- Do NOT wire the parent-child (PC) chunk store. `epistemics.py`'s own module docstring already
  flags PC re-projection as a documented, separate follow-up (`docs/specs/
  pr-m1-epistemics-markers.md` ADR-1 option 2) — stay confined to the baseline flat-chunk
  projection that already exists (`load_doc_chunks`).
- Do NOT touch `ChatController`/the frontend chunk-key join (PR-M1's read side,
  `markers_for_chunk_keys`/`markers_for_parent`) — once `chunk_epistemics` rows exist again, that
  join is already correct (it was never the broken part); verifying the live UI picks them up is a
  natural follow-up smoke test, not this sprint's DoD.
- Do NOT re-tune `_MIN_CONCEPT_LEN` (still 3) or change which polarities count as supporting/
  opposing — unrelated axes, out of scope.
-->
