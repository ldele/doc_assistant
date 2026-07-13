<!-- status: active · updated: 2026-07-13 · class: disposable -->
<!-- BUILDING 2026-07-13. Sole active contract (SPRINT-014 archived on commit). Design lock:
     docs/specs/feature-ab-compare-sandbox.md (grilled 2026-07-13). Feature: A/B-compare sandbox v1 —
     a per-turn "Compare" action runs the query under the locked defaults (A) and the current session
     RagOverrides (B) and shows the two retrieved source sets side by side with a computed diff.
     RETRIEVAL DIFF ONLY — $0 (no LLM call), the retrieval-only, live-verifiable first slice of
     ADR-010 option-4's north-star. Full-answer 2x compare is deferred (cost-gated, unverifiable
     without a model). Request-scoped, no module-global mutation (U1 isolation discipline). Flip to
     status:archived after commit. Pre-existing [lifecycle] warn applies (tolerated across SPRINT-*.md). -->

# SPRINT-015 — ab-compare

- **base:** main
- **DoD:** A per-turn "Compare" action posts `POST /api/compare {text, overrides}`; the backend runs
  `retrieve_with_scores` twice (A = locked defaults, B = the session `RagOverrides`) and returns both
  ranked source lists + a diff (each source classified `in_both` with rank-delta / `only_in_a` /
  `only_in_b`) + a `{top_k, use_multi_query}` effective header per side + an honest `note` (no
  retrieval-affecting override → "doesn't change retrieval"; `top_k`-only → depth note; membership move
  → ""). **No LLM call** on the compare path (guard-tested); **no module-global mutation** (isolation
  guard). Desktop: a `Compare` button beside Send renders a two-column `CompareCard` (A | B) with diff
  badges + the "indicative, not a verdict" framing; columns stack on mobile; both themes; no horizontal
  overflow. `svelte-check` 0; full gate green (ruff / ruff format / mypy src / bandit / pytest);
  preview-harness-verified live on the real corpus ($0/offline — a `use_multi_query` flip shows a real
  membership diff).

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
- docs/specs/feature-ab-compare-sandbox.md
- docs/decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md
- src/doc_assistant/chat_controller.py
- src/doc_assistant/pipeline.py
- apps/api/main.py
- apps/api/models.py
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/api.ts
- apps/desktop/src/lib/types.ts

## affects
- src/doc_assistant/chat_controller.py
- apps/api/main.py
- apps/api/models.py
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/CompareCard.svelte
- apps/desktop/src/lib/api.ts
- apps/desktop/src/lib/types.ts
- tests/unit/test_compare.py
- tests/integration/test_api_compare.py

## contracts
- test: tests/unit/test_compare.py::test_diff_sources_classifies_and_ranks
- test: tests/integration/test_api_compare.py::test_compare_endpoint
- map: apps/desktop/src/lib/types.ts | when: apps/api/models.py

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
- docs/ui-checklist.md
