<!-- status: archived · updated: 2026-07-10 · class: disposable -->
<!-- BUILT 2026-07-10 — staged for review, not yet committed. status:archived so sprint_check sees
     SPRINT-009 (U3) as the sole active contract.
     U2 (ROADMAP), 1st in the locked Phase-8 UI build order (U2 → U3 → U1 → U1b → U1c).
     Design-locked spec: docs/specs/feature-phase8-ui-upgrade.md §U2 (grilled 2026-07-10).
     Frontend-only, CSS + one template-structure tweak inside a single component. No backend,
     no wire type, no locked-setting touch. Verified via svelte-check + the preview harness
     (no vitest — grill ledger #5; screenshots flaky on this box → snapshots + synchronous
     evals per .claude/KNOWN_ISSUES.md).
     ORDERING NOTE: U3 (SPRINT-009) also edits Turn.svelte (removes the always-on sources grid,
     adds onCitationClick). U2 lands first and only touches the .you/.turn CSS + the "You"
     label placement, so U3 rebases cleanly on top.

     Scope boundary (NOT machine-read — kept here, above ## uses, so sprint_check's contract
     parser does not slurp its bullets into the docs section):
     - CSS + template-structure only, confined to apps/desktop/src/lib/Turn.svelte. The one
       template change permitted is moving/wrapping the "You" label relative to the .you div so
       the bubble reads correctly; no logic, no new prop, no new state.
     - Do NOT give the assistant answer a matching bubble — the requirement is the asymmetry
       (bounded user bubble vs. unbounded assistant block). Leave .assistant and its children alone.
     - Do NOT touch App.svelte / SourceCard / Provenance / Markdown or any type — that surface is
       U3's (SPRINT-009). If a change here seems to need one of those files, it is out of scope.
     - Guard tests (preview harness, this box's known-good path — snapshots + synchronous evals):
       preview_inspect .you computed max-width + align-self/margins to confirm it is actually
       right-bounded (not eyeballed); confirm .assistant still spans full .turn width beside a
       narrow bubble; preview_resize to mobile to confirm no overflow. -->

# SPRINT-008 — chat-bubble-layout

- **base:** main
- **depends-on:** none (shipped Tauri/Svelte shell, PR-M0–M5). 1st in the Phase-8 UI order.
- **DoD:** The **user** turn renders as a bounded, right-aligned bubble; the **assistant** (RAG
  answer) turn renders **unchanged** — full-width, no wrapper, no max-width, no background — so
  sources/claims/provenance/figures keep the room they have today. Concretely in `Turn.svelte`:
  `.turn` becomes a flex column (`display: flex; flex-direction: column`), replacing today's
  implicit block flow; `.you` gains `align-self: flex-end`, `max-width: min(72%, 640px)`,
  `background: var(--surface-2)` (neutral — **not** `var(--accent)`; grill ledger #4, keeps accent
  as the app's one CTA signal), `border-radius: 14px` with a squared bottom-right corner
  (`border-bottom-right-radius: 4px`, the standard tail cue), `padding: 0.55rem 0.85rem`. The "You"
  label stays legible (inside the bubble, small+muted, or above it right-aligned — decide by look).
  `.assistant` and every child (`Markdown`, the sources grid, `Provenance`, the usage chip) is
  byte-untouched. No change to `App.svelte`, `SourceCard.svelte`, `Provenance.svelte`,
  `Markdown.svelte`, or any wire type. The bubble reads with sufficient contrast against `--bg` in
  **both** themes (light + OS-dark today; the manual toggle arrives in U1 — this must not assume it).
  At `mobile` viewport the `min(72%, 640px)` cap must not overflow. `svelte-check` clean (0 errors);
  preview-harness-verified per the guard tests noted above; nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). -->
- docs/specs/feature-phase8-ui-upgrade.md
- apps/desktop/src/lib/Turn.svelte
- apps/desktop/src/app.css

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- apps/desktop/src/lib/Turn.svelte

## contracts
<!-- Frontend-only: no vitest (grill ledger #5). Verification is svelte-check + preview harness;
     there is no machine-read test/snap/map target for a CSS-only change. -->

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
