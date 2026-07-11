<!-- status: archived · updated: 2026-07-11 · class: disposable -->
<!-- BUILT 2026-07-11 — committed `09afd0c` (U1/U1b/U1c). status:archived so sprint_check sees
     SPRINT-011 (U1b, unblocked now that this landed) as the sole active contract once it's flipped.
     Pre-existing [lifecycle] warn applies (tolerated across all SPRINT-*.md).

     U1 (ROADMAP), 3rd in the locked Phase-8 UI build order (U2 → U3 → U1 → U1b → U1c).
     Design: docs/specs/feature-phase8-ui-upgrade.md §U1 (theme toggle + settings disclosure)
     ADOPTS docs/specs/feature-rag-sandbox.md AS-IS (ADR-010, accepted) for the sandbox half.
     This is the HEAVY track: it spans the RAG-sandbox backend (chat_controller/pipeline/api) AND
     the Settings surface AND the theme system.

     READ-SET IS DELIBERATELY SPEC-LED. The cpc uses budget is 12 files / 2500 lines; U1's full
     write-set (backend + all of apps/desktop/src) does not fit under it. The read-set below is
     "the design contracts + the two backend threading seams you must grok first" — the frontend
     and test files are small and fully line-specified inside the two specs (which ARE in the
     read-set), so they are opened on demand, not preloaded. This produces expected uses⊇affects
     WARNs on the frontend/test files (advisory, superset_affects="warn").

     ESCAPE HATCH (if a build session prefers clean sprints or hits --strict on those WARNs): split
     U1 at the backend/frontend seam feature-rag-sandbox.md already draws — a "U1a: RAG-sandbox
     backend" sprint (chat_controller/pipeline/apps/api + pytest) and a "U1-ui: Settings surface +
     theme + disclosure" sprint (apps/desktop/**). U1 was design-locked as ONE track (roadmap
     concept); splitting it into two PRs for EXECUTION does not reopen ADR-010 and is consistent
     with "one PR per session, never bundle." Flagged to the user 2026-07-10; not taken here so the
     contract matches the locked single-track plan.

     Scope boundary (NOT machine-read — kept here, above ## uses, so the contract parser does not
     slurp its bullets into the docs section):
     - The sandbox half is feature-rag-sandbox.md verbatim — do NOT re-decide any ADR-010 knob; the
       live set is exactly {TOP_K, SYNTHESIS_MODE, USE_MULTI_QUERY}. EPISTEMICS_MARKERS_ENABLED /
       REVIEWER_EVIDENCE_CHARS are U1b's (SPRINT-011), NOT this sprint.
     - The load-bearing test is the isolation guard (a turn with overrides set, then a turn with
       overrides=None, uses locked defaults on the second) — no monkeypatch anywhere in the path.
     - Theme is client-only: localStorage + [data-theme], never a backend setting. Do NOT route it
       through POST /api/settings or RagOverrides (both explicitly rejected in the spec).
     - Disclosure (deliverable 2) is display-only: render the already-fetched fields with reasons;
       do NOT make any read-only knob live (needs a pipeline rebuild / re-ingest — out of scope).
     - POST /api/settings still writes ONLY source_dir — overrides are never persisted.
     - If the read-set budget / --strict WARNs bite when this is activated, split at the backend/
       frontend seam per the ESCAPE HATCH above rather than trimming the DoD. -->

# SPRINT-010 — settings-sandbox-theme

- **base:** main
- **depends-on:** SPRINT-009 (U3) landed (build-order-locked; U1 and U3 both edit
  `Settings.svelte`/`App.svelte` regions, so U1 rebases on U3's Turn/App changes). ADR-010
  (accepted) + feature-rag-sandbox.md are the sandbox contract; PR-M0/M2/M3 seams all present.
- **DoD (three bundled deliverables — all live in or around `Settings.svelte`):**

  **(1) RAG sandbox — build feature-rag-sandbox.md exactly (its own DoD applies unchanged).**
  `RagOverrides` (frozen dataclass: `top_k`, `synthesis_mode`, `use_multi_query`, all
  `X | None = None`) added to `chat_controller.py`; `ChatController.handle(..., *, overrides=None)`
  threads it into `_handle_rag` — `overrides=None` is **byte-identical to today** (backward compat).
  `pipeline.retrieve_with_scores(..., use_multi_query=None)` branches request-scoped;
  `apps/api/models.py` gains a `RagOverrides` wire model + optional `ChatRequest.overrides`
  (`top_k` validated `[1, CANDIDATE_K]` → 422 on out-of-range, no silent clamp);
  `apps/api/main.py` maps body→`chat_controller.RagOverrides` and sources `_settings_view`'s
  `retrieval_weights` from `config.BM25_WEIGHT` (not the hardcoded literal). **The one correctness
  obligation (ADR-010 Decision 4 / Confidence ⚠): overrides thread as explicit request-scoped
  params — NO module-global (`pipeline.USE_MULTI_QUERY`, `chat_controller.SYNTHESIS_MODE`) is ever
  assigned; concurrent turns on the shared singleton must not interfere.** Provenance reflects the
  **effective** values and flags any that differ from the locked default (Decision 5). Settings
  gets a "RAG sandbox" section (TOP_K slider `1..candidate_k`, SYNTHESIS_MODE ai/human toggle,
  USE_MULTI_QUERY switch), a persistent muted "Session only — resets on restart; run the eval
  harness to change a default" banner, and a "Reset to locked defaults" button. Overrides live in
  `App.svelte` `$state` (the session) → cleared on restart (non-persistent; never written to
  config/.env/app_settings).

  **(2) Full read-only disclosure (closes the blind-spot gap).** The "Engine (read-only)" section
  renders **every** field in the `Settings` TS type — today `retrieval_weights`, `use_parent_child`,
  `parent_chunk`, `child_chunk` are fetched and silently dropped. Each locked knob shows **the
  reason it can't be live** (construction-time / ingest-time / measured-inert); the BM25/vector
  weight is labeled *"inert on the shipped top-K by construction (measured)"* — a fact, not a slider.

  **(3) Manual theme — a real tri-state setting (System / Light / Dark).** New
  `apps/desktop/src/lib/theme.ts` (~20 lines, pure DOM+storage, no framework dep):
  `type Theme = 'system'|'light'|'dark'`, `getTheme/setTheme` over `localStorage['theme']`,
  `applyTheme` sets/clears `document.documentElement.dataset.theme`. `main.ts` calls
  `applyTheme(getTheme())` as its **first line, before `mount()`** (no flash-of-wrong-theme).
  `app.css` re-keys the existing dark palette off `[data-theme='dark'|'light']` **in addition to**
  the `prefers-color-scheme` media query (no new colors) — `:root[data-theme]` wins by
  attribute-selector specificity + source order; `@media` guarded to `:root:not([data-theme])`.
  Settings gains a "Display" section (a 3-way segmented control, above "Your documents") that calls
  `setTheme`+`applyTheme` on change — no `busy`, no backend round-trip, **theme never touches the
  backend** (Decision: it's a pure rendering preference, correctly NOT routed through the governed
  settings endpoint / RagOverrides non-persistence wall).

  Full gate: ruff / ruff format / mypy --strict / bandit 0 HIGH / pytest all green; **no paid LLM
  call in tests** (cpc §13); `svelte-check` clean; preview-harness-verified (theme flip actually
  changes `<body>` computed background in all 3 states; the isolation guard test passes). Nothing
  committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). SPEC-LED — see header note; frontend/test files are
     line-specified inside the two specs and opened on demand (expected uses⊇affects WARNs). -->
- docs/specs/feature-rag-sandbox.md
- docs/decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md
- docs/specs/feature-phase8-ui-upgrade.md
- src/doc_assistant/chat_controller.py
- src/doc_assistant/pipeline.py
- apps/api/main.py
- apps/api/models.py

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/chat_controller.py
- src/doc_assistant/pipeline.py
- apps/api/models.py
- apps/api/main.py
- apps/desktop/src/lib/theme.ts
- apps/desktop/src/main.ts
- apps/desktop/src/app.css
- apps/desktop/src/lib/Settings.svelte
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/types.ts
- apps/desktop/src/lib/api.ts
- tests/unit/test_chat_controller.py
- tests/unit/test_pipeline_retrieval.py
- tests/unit/test_api_models.py
- tests/unit/test_settings_view.py

## contracts
<!-- pytest guard tests (feature-rag-sandbox.md §Guard tests) — run in the verify loop.
     Frontend (theme.ts, sandbox surface) is preview-harness-verified, no vitest (grill ledger #5). -->
- test: tests/unit/test_chat_controller.py::test_overrides_isolation_no_state_leak_between_turns
- test: tests/unit/test_chat_controller.py::test_top_k_override_changes_retrieve_arg
- test: tests/unit/test_chat_controller.py::test_overrides_none_reproduces_default_effective_values
- test: tests/unit/test_pipeline_retrieval.py::test_use_multi_query_override_ignores_global
- test: tests/unit/test_api_models.py::test_chat_request_top_k_out_of_range_422
- test: tests/unit/test_settings_view.py::test_retrieval_weights_sourced_from_config

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
