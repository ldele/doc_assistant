<!-- status: active · updated: 2026-07-10 · class: disposable -->
<!-- ACTIVE 2026-07-10 — SPRINT-008 (U2) landed (staged for review), no sibling to archive since
     U2 was the only other active contract. This is now the sole active contract.

     U3 (ROADMAP), 2nd in the locked Phase-8 UI build order (U2 → U3 → U1 → U1b → U1c).
     Design-locked spec: docs/specs/feature-phase8-ui-upgrade.md §U3 (grilled 2026-07-10).
     Frontend-only. Reuses Settings.svelte's shipped drawer pattern verbatim (fly/scrim, focus
     trap, Esc) rather than inventing a second modal. No backend, no wire type change (the
     SourceView + citation_note_md it needs are already in memory / already computed server-side).

     Scope boundary (NOT machine-read — kept here, above ## uses, so the contract parser does not
     slurp its bullets into the docs section):
     - SourcePanel.svelte is a copy-adaptation of Settings.svelte's scrim+panel with the form body
       replaced by one <SourceCard source={...} /> (unchanged) + a small "Source [n]" header and a
       close button. Do NOT re-implement SourceCard or the modal a11y from scratch.
     - Citation linkifier lives in Markdown.svelte behind a new optional onCitationClick prop, run
       in an $effect after {@html html}, with ONE delegated click listener on the container root
       (event.target.closest('.citation')), not one-per-button. It must never mutate HTML
       structure/attributes — only text nodes it already holds. Do NOT switch the file to raw-HTML
       or raw-markdown regex replacement (both rejected in the spec: fragile / code-span false hits).
     - Turn.svelte: remove the unconditional <div class="sources"> grid; add ONLY the
       malformed-citation fallback (gated on result.citation_note_md !== '', the value the backend
       already computed via audit_citations — do NOT re-derive citation coverage client-side). Pass
       onCitationClick through to Markdown.
     - Backend untouched: synthesis.py's _CITATION_RE and citation_note_md are read-only references,
       not edit targets. No new API field, no SSE change.
     - Out of scope: multi-panel stacking, a "pin panel open" mode, the rich marker hover-UI, the
       parent-child re-projection — all in the spec's Related-backlog / Out-of-scope tables. -->

# SPRINT-009 — citation-side-panel

- **base:** main
- **depends-on:** SPRINT-008 (U2) landed — both edit `Turn.svelte`; U2 lands first (CSS on
  `.you`/`.turn`), U3 rebases on top and removes the always-on sources grid. Not a hard code
  dependency, but build-order-locked so the Turn.svelte edits don't collide.
- **DoD:** Inline `[n]` markers in the answer become clickable, accent-colored controls; clicking
  one opens a right-hand slide-over panel showing that source's `SourceCard` detail, and the
  clicked `[n]` visibly highlights (`.citation.active`) while its panel is open. Source cards **no
  longer render inline by default** — a source is reachable only via its `[n]`, with **one
  fallback**: when `result.sources.length > 0` and the answer has zero recognized `[n]` markers
  (the malformed/no-citation case, keyed off the backend-computed `result.citation_note_md !== ''`),
  the inline source list is shown so the reader can still reach them (grill ledger #3 — hiding
  sources is worst exactly when the model cited badly). `[n]` linkification is done by a **DOM
  text-node walk** after `marked.parse()` renders (`createTreeWalker`, `NodeFilter.SHOW_TEXT`),
  **skipping any node with a `<code>`/`<pre>` ancestor** so a bracketed number inside a code span
  is never linkified (technical corpus — real case), replacing `/\[(\d+)\]/g` with
  `<button class="citation" data-n="…">`. The panel **reuses Settings.svelte's a11y mechanics**
  (`role="dialog"`, `aria-modal`, focus trap on Tab, Esc-to-close, `fly`+`fade`,
  `prefers-reduced-motion` → instant) — not a second divergent modal. **One panel app-wide**:
  clicking a different `[n]` (even in another turn) swaps its content, never stacks (matches the
  Settings gear's single-drawer behavior). `App.svelte` owns
  `activeCitation: {turnId, n} | null` (same ownership shape as `showSettings`) and resolves the
  `SourceView` by turn+n. No network call — the panel is handed a `SourceView` already in memory.
  `svelte-check` clean; preview-harness-verified per the guard tests; nothing committed — staged
  for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set (cap 12 files / 2500 lines). -->
- docs/specs/feature-phase8-ui-upgrade.md
- apps/desktop/src/lib/Settings.svelte
- apps/desktop/src/lib/Turn.svelte
- apps/desktop/src/lib/Markdown.svelte
- apps/desktop/src/lib/SourceCard.svelte
- apps/desktop/src/App.svelte

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- apps/desktop/src/lib/SourcePanel.svelte
- apps/desktop/src/lib/Markdown.svelte
- apps/desktop/src/lib/Turn.svelte
- apps/desktop/src/App.svelte

## contracts
<!-- Frontend-only: no vitest (grill ledger #5). The two behavioral guarantees are verified via the
     preview harness, not a machine-read test file:
       - [1]/[12] inside a <code> block are left as text; outside code they become .citation buttons
         with the right data-n.
       - clicking [n] opens the panel with the right source; the button gets .active; Esc closes; a
         second [n] swaps content without a second panel; a zero-source turn renders no grid; a
         malformed-citation turn (citation_note_md != '') still shows its sources inline. -->

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
