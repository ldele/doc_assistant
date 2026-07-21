<!-- status: design-locked · created: 2026-07-21 · owner: Code · plan: docs/PLAN_2026-07-21_exploration-epistemics.md (E1) -->

# Feature spec — E1.1: marker-join trustworthiness (KI-8 re-projection)

Build contract for ROADMAP row **E1** — the *correctness core* (the `_handle_rag` extraction is
deferred to **E1.2**, a separate refactor). Fixes KI-8: the 7d epistemics marker chip, the flagship
integrity surface and the join ADR-027's always-on strip (E2) will render, silently under-reports by
**~40%** in the default parent-child retrieval mode.

## Why now
ADR-027 D3 makes the source-evaluation strip **always-on** (E2). E1 is its honesty prerequisite: an
always-on strip built on a join that drops ~40% of markers is a silently-lying UI. (E0 already made
the *sidecar data* trustworthy; E1 makes the *join* trustworthy.)

## The defect (verified in code)
`chunk_epistemics` rows are keyed on the **baseline** segmentation (`{document_id}:{chunk_index}`).
In default PC mode retrieval returns **parents** (`pipeline.py:314` — a parent carries `parent_index`,
never `chunk_index`), so `chat_controller._chunk_key` returns `None` and `_attach_markers` falls back
to `epistemics.markers_for_parent` — a **strict substring-containment** test. A `BASELINE_CHUNK_SIZE=1000`
baseline chunk cannot be a substring of a parent it only partially overlaps (`PARENT_CHUNK_OVERLAP=200`),
so a marked chunk straddling a parent boundary is contained in **neither** parent and its markers
vanish (review WE-7; systematic false *negatives*, not fail-safe over-attribution).

## The fix (KI-8 option 2 — re-projection)
Project the node weights **directly onto the PC parent segmentation** using the same structural
attribution the baseline projection already uses, keyed `{document_id}:p{parent_index}` (the ADR-4
composite `concept_skeleton.load_presence_inputs` already builds). The PC marker join then becomes a
**direct key lookup** — no containment — and the coarse path is retired.

## Items

### E1.1a — `chunk_key` column (additive migration)
- Add a nullable `chunk_key` VARCHAR to the `chunk_epistemics` model + `_ADDITIVE_COLUMNS`
  (`("chunk_epistemics","chunk_key","VARCHAR","ix_chunk_epistemics_chunk_key")`). It is the
  **authoritative, segmentation-agnostic** join key: `{doc}:{chunk_index}` (baseline) or
  `{doc}:p{parent_index}` (parent). `chunk_index` stays (holds `parent_index` for a parent row).
- The table is regenerable (dropped-rows + rewritten each `compute_epistemics --apply`), so no hard
  backfill: the migration makes the column exist; the next recompute fills it. `load_epistemics_index`
  falls back to `{doc}:{chunk_index}` when `chunk_key IS NULL`, so a migrated-but-not-recomputed DB
  still joins flat rows and simply has no parent rows yet (honest — same as today).

### E1.1b — project onto both segmentations
- `ChunkEpistemics.chunk_key` becomes a **stored field** (was a derived property); `project_chunk`
  and `project_chunk_weights` carry an explicit `chunk_key`. `doc_chunks` tuples become
  `(chunk_key, document_id, chunk_index, text)`.
- `load_doc_chunks()` yields baseline keys `{doc}:{chunk_index}`. **New** `load_pc_parent_chunks()`
  reads the PC store, dedups children to parents by `parent_index`, yields `{doc}:p{parent_index}`
  (mirrors `concept_skeleton.load_presence_inputs`; returns `[]` if the PC collection is absent).
- `build_epistemics` projects over `load_doc_chunks() + load_pc_parent_chunks()` — both segmentations,
  same weights, same `concepts_in_text` attribution. `_write_rows` persists `chunk_key`.
- **Retire** `markers_for_parent`, `load_marked_chunks`, `MarkedChunk` (no remaining consumer once the
  controller joins by key).

### E1.1c — controller: PC key + direct join + WARNING log
- `_chunk_key(meta)`: baseline → `{doc}:{chunk_index}`; PC parent (`parent_index`, no `chunk_index`)
  → `{doc}:p{parent_index}`; missing `document_id` → `None`.
- `_attach_markers`: **both** flat and PC sources join `sv.chunk_key` against a single
  `load_epistemics_index()` (loaded once). Delete the containment branch + `marked_by_doc`.
- The blanket `except Exception: return` gains a **WARNING log** (`log.warning("attach_markers_failed", …)`)
  — advisory markers still never break a turn, but a silent failure under an always-on strip is a
  silently-lying UI, so it must be observable.

## DoD / guard tests (each fails against today's code)
1. **Re-projection (the KI-8 fix), integration:** a marked concept present in a PC parent's text whose
   marked baseline chunk is **not a substring** of the parent → after `build_epistemics --apply`,
   `load_epistemics_index()` contains the parent key `{doc}:p{idx}` with the marker. Fails today
   (no parent rows exist; the containment path would miss it).
2. **`_chunk_key` PC:** a PC parent meta → `{doc}:p{parent_index}` (was `None`). Inverts the existing
   `test_chunk_key_parent_child_chunk_is_none`.
3. **`_attach_markers` PC direct join:** a PC source whose key is in the index → marked, with no
   `load_marked_chunks` call.
4. **WARNING log:** a forced load failure logs a WARNING and leaves the turn unmarked (not raised).
5. **Flat unchanged / parity:** baseline join + a no-marker turn stay byte-identical.

## Build order & gate
E1.1a → E1.1b → E1.1c → tests. Per item: a non-vacuous guard test, then `ruff`/`ruff format`/
`mypy --strict src`/`bandit`/full `pytest`; a live $0 probe on a **copy** of the real `data/library.db`
(inject one contested stance so a marker exists → show the parent key appears in the index and the
join attaches a marker the containment path missed). Update the E0.4 empty-input test to also stub
`load_pc_parent_chunks`. One `docs/DEVLOG.md` entry; KI-8 → Resolved; ROADMAP E1 row.

## Out of scope (deferred)
- **E1.2** — `_handle_rag` extraction (~287 lines) into staged steps, before E2/E3 wire into it.
- Marker *quality* (RG-019 `contested` denominator; Node-B stance regen on the RTX box) — unchanged.
