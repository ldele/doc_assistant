<!-- status: archived · updated: 2026-07-11 · class: disposable -->
<!-- BUILT 2026-07-11 — staged for review, not yet committed. status:archived so sprint_check sees
     no active contract (Phase 8's five-track UI build order — U2/U3/U1/U1b/U1c — is now fully
     shipped; the next SPRINT contract, whatever it is, starts fresh). Pre-existing [lifecycle]
     warn applies (tolerated across all SPRINT-*.md).

     U1c (ROADMAP), 5th and last in the locked Phase-8 UI build order (U2 → U3 → U1 → U1b → U1c).
     Design: docs/decisions/ADR-011-desktop-provider-apikey-management.md (accepted, grilled
     2026-07-10) + the v1 build spec docs/specs/feature-provider-switch.md (already committed as
     8217454 "U1c - Spec"; its line-number citations were refreshed 2026-07-11 against the
     post-U1/U1b code, no design change).

     Scope boundary (NOT machine-read — kept here, above ## uses, so the contract parser does not
     slurp its bullets into the docs section):
     - v1 ONLY: switch provider + model among already-configured providers (key stays in .env,
       live swap between turns). Keyring / in-app key entry is ADR-011's v2 north-star — NOT this
       sprint.
     - No module-global mutation: config.LLM_PROVIDER/REVIEWER_PROVIDER/etc. are never assigned.
       The effective provider/model is resolved from app_settings' persisted selection at
       construction/turn time (RAGPipeline.set_chat_model, a direct method call).
     - The reviewer follows an unpinned switch (fork C); an explicit .env REVIEWER_PROVIDER pin
       still wins — detected via config.REVIEWER_PROVIDER_PINNED (was the env var actually set),
       not by comparing resolved values (REVIEWER_PROVIDER's own default already equals
       LLM_PROVIDER, so a value-equality check can't tell "explicitly pinned to the same thing"
       from "never pinned").
     - The eval judge (JUDGE_PROVIDER/JUDGE_MODEL) and the CLI --apply / assert_provider_intent
       path are UNTOUCHED — out of scope (ADR-011 Consequences).
     - A provider whose key is absent must be rejected server-side (ValueError → 400) even if the
       frontend also disables it — inform-don't-corrupt, never trust the client alone.
     - Independent of U1's RagOverrides — provider is a persisted global (settings.json), not a
       per-request override; only Settings.svelte is a shared file (this sprint adds a section,
       does not touch U1/U1b's sandbox section). -->

# SPRINT-012 — provider-switch

- **base:** main
- **depends-on:** ADR-011 (accepted) + `docs/specs/feature-provider-switch.md` (design-locked v1
  build spec). Independent of SPRINT-010/011's `RagOverrides` work (different surface); both are
  currently staged-uncommitted on the same working tree, so this sprint's diff lands on top of
  them.
- **DoD (from the spec's own Definition of done):**
  - Switching provider **+ model** from Settings takes effect on the **next turn**, no restart, no
    embedder/reranker reload; a mid-stream turn finishes on the old provider (proven by a test that
    builds a chain, swaps, and asserts the pre-built chain is unaffected).
  - The per-answer reviewer **follows** the switch when `REVIEWER_PROVIDER` was never explicitly
    set in the environment; an explicit pin still wins. A switch to Ollama makes the turn truly
    free (no metered token).
  - A provider with no key renders **unavailable with the reason** in the settings view and is
    **rejected server-side** (400) if selected anyway.
  - The **effective** (not boot-time) provider/model appears in `_settings_view()`/`/api/health`/
    provenance; the per-turn cost chip is unchanged; **no confirm dialog**.
  - The selection **persists across restart** via `settings.json` (the `source_dir` precedent); no
    `config` module attribute is ever assigned a new value at runtime.
  - **Backward compat:** no `llm_provider`/`llm_model` sent → `/api/settings` and every chat turn
    are byte-identical to today; `source_dir`-only posts keep working.
  - Full gate green (ruff / ruff format / `mypy --strict src` / bandit 0 HIGH/MED / pytest); no
    paid LLM call in any test (cpc §13); `svelte-check` clean; preview-harness-verified (construction
    only, no chat turn — $0); DEVLOG entry.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). -->
- docs/decisions/ADR-011-desktop-provider-apikey-management.md
- docs/specs/feature-provider-switch.md
- src/doc_assistant/config.py
- src/doc_assistant/llm.py
- src/doc_assistant/pipeline.py
- src/doc_assistant/chat_controller.py
- src/doc_assistant/app_settings.py
- apps/api/main.py
- apps/api/models.py

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/config.py
- src/doc_assistant/llm.py
- src/doc_assistant/pipeline.py
- src/doc_assistant/chat_controller.py
- src/doc_assistant/app_settings.py
- apps/api/main.py
- apps/api/models.py
- apps/desktop/src/lib/types.ts
- apps/desktop/src/lib/api.ts
- apps/desktop/src/lib/Settings.svelte
- tests/unit/test_pipeline_retrieval.py
- tests/unit/test_app_settings.py
- tests/unit/test_llm.py
- tests/unit/test_chat_controller.py
- tests/unit/test_api_models.py
- tests/integration/test_api_chat_sse.py
- tests/integration/test_api_settings_ingest.py

## contracts
<!-- pytest guard tests — run in the verify loop. Frontend surface preview-harness-verified. -->
- test: tests/unit/test_pipeline_retrieval.py::test_set_chat_model_swaps_llm
- test: tests/unit/test_pipeline_retrieval.py::test_in_flight_chain_survives_a_swap
- test: tests/unit/test_app_settings.py::test_llm_selection_round_trips
- test: tests/unit/test_app_settings.py::test_set_llm_selection_rejects_keyless_provider
- test: tests/unit/test_llm.py::test_provider_available
- test: tests/unit/test_llm.py::test_get_reviewer_client_follows_unpinned_switch
- test: tests/unit/test_llm.py::test_get_reviewer_client_respects_explicit_pin
- test: tests/unit/test_chat_controller.py::test_reconfigure_persists_and_swaps_no_global_mutation
- test: tests/unit/test_chat_controller.py::test_persisted_selection_applied_at_construction
- test: tests/integration/test_api_settings_ingest.py::test_post_settings_provider_switch_reconfigures_controller
- test: tests/integration/test_api_settings_ingest.py::test_post_settings_keyless_provider_is_400

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
