<!-- status: archived · updated: 2026-07-11 · class: disposable -->
<!-- BUILT 2026-07-11 — committed `09afd0c` (U1/U1b/U1c). status:archived so sprint_check sees
     U1c (once it has a build spec + SPRINT contract) as the sole active contract. Pre-existing
     [lifecycle] warn applies (tolerated across all SPRINT-*.md).

     U1b (ROADMAP), 4th in the locked Phase-8 UI build order (U2 → U3 → U1 → U1b → U1c).
     Design: docs/specs/feature-phase8-ui-upgrade.md §U1b + the 2026-07-10 amendment section of
     docs/decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md. Small: adds the two ADR-010
     "must revisit" query-time knobs to the sandbox surface U1 built. HARD-DEPENDS on U1
     (SPRINT-010) — it extends the same RagOverrides dataclass, the same wire model, and the same
     Settings sandbox section. Building it before U1 exists means threading override fields through
     code U1 hasn't written yet.

     Scope boundary (NOT machine-read — kept here, above ## uses, so the contract parser does not
     slurp its bullets into the docs section):
     - ONLY these two knobs are added — do NOT widen the sandbox knob set further (anything past
       this is a new ADR-010 amendment, not this sprint; spec Out-of-scope).
     - Same non-persistence wall as U1: never written to config/.env/app_settings; restart clears.
     - The isolation guard test must now assert NO leak across all five fields — extend U1's test,
       don't fork a parallel one.
     - EPISTEMICS_MARKERS_ENABLED here is the request-scoped OVERRIDE only; it does not change the
       config default (which G1 already flipped on) and does not touch epistemics projection logic
       (KI-15 is resolved; that's the read side, untouched here).
     - Depends structurally on U1's dataclass/model/Settings-section existing — if U1 is split per
       its ESCAPE HATCH, this depends on the backend half of that split. -->

# SPRINT-011 — settings-niche-knobs

- **base:** main
- **depends-on:** SPRINT-010 (U1) landed — extends U1's `RagOverrides` (3 fields → 5), U1's wire
  model, and U1's Settings "RAG sandbox" section. Not startable before U1 is on main.
- **DoD:** Two more query-time knobs join the sandbox, threaded with the **same** non-persistent,
  request-scoped mechanics ADR-010 Decision 4 mandates (no module-global mutation), the same
  effective-value provenance (Decision 5), and covered by the **same isolation guard test extended
  to all five fields** (not just U1's original three):
  - `EPISTEMICS_MARKERS_ENABLED` → a 4th `RagOverrides` field (`bool | None = None`); Settings gets
    an on/off switch labeled "Show contested/superseded chips" in U1's sandbox section.
  - `REVIEWER_EVIDENCE_CHARS` → a 5th field (`int | None = None`); Settings gets an integer input.
    **Bounds are NOT guessed here** — the build session must first read the reviewer prompt's actual
    current usage of `REVIEWER_EVIDENCE_CHARS` (grep it in `src/doc_assistant/`) to pick a sane
    min/max before exposing a raw number field; out-of-range → 422 like `top_k`, no silent clamp.
  Both render under U1's existing "Session only — resets on restart" banner. Backward compat holds:
  with none of the five fields set, the turn is byte-identical to today. Full gate green (ruff /
  ruff format / mypy --strict / bandit 0 HIGH / pytest); no paid LLM call in tests (cpc §13);
  `svelte-check` clean; preview-harness-verified. Nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). REVIEWER_EVIDENCE_CHARS bounds: grep its usage in
     src/doc_assistant/ at build time (targeted read, not a full-file preload). -->
- docs/specs/feature-phase8-ui-upgrade.md
- docs/decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md
- src/doc_assistant/chat_controller.py
- apps/api/models.py
- apps/desktop/src/lib/Settings.svelte
- src/doc_assistant/epistemics.py

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/chat_controller.py
- apps/api/models.py
- apps/api/main.py
- apps/desktop/src/lib/Settings.svelte
- apps/desktop/src/lib/types.ts
- tests/unit/test_chat_controller.py

## contracts
<!-- pytest guard tests — run in the verify loop. Frontend surface preview-harness-verified. -->
- test: tests/unit/test_chat_controller.py::test_overrides_isolation_covers_all_five_fields
- test: tests/unit/test_chat_controller.py::test_epistemics_markers_override_per_turn
- test: tests/unit/test_chat_controller.py::test_reviewer_evidence_chars_override_per_turn

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
