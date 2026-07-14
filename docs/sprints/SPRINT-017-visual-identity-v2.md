<!-- status: active · updated: 2026-07-14 · class: disposable -->
<!-- BUILDING 2026-07-14. Sole active contract (SPRINT-016 archived on V1 commit 35b8627).
     Design lock: docs/specs/feature-visual-identity.md §V2 (grilled 2026-07-13, fork #9 locks the
     scope; V2 design specifics chosen 2026-07-14 with the user — wordmark option A serif + book
     mark; empty state with sample-question chips). Feature: Visual-identity pass V2 — layout rhythm.
     Header/wordmark treatment (serif "doc_assistant" + indigo book mark), a coherent spacing +
     type scale (new --space-*/--text-* tokens + --measure), restyled empty + first-run states with
     clickable sample-question chips, and a ~70ch reading measure on answer/excerpt prose. SHELL
     TOPOLOGY STAYS OUT (fork #9 — sidebar│main│drawer unchanged, verified behavior not re-risked).
     FRONTEND-ONLY — CSS + Svelte templates + one new Icon glyph set; no backend, no src/ change, no
     wire type, no locked setting, no behavior change (the sample chips only prefill the existing
     composer input). $0/offline. Gate for a frontend-only sprint = svelte-check 0 (no pytest surface). -->

# SPRINT-017 — visual-identity-v2

- **base:** main
- **DoD:** **Header/wordmark** — `App.svelte` header renders a wordmark: an indigo book mark
  (new `Icon` glyph) + `doc_assistant` in `--font-serif` (`doc` ink, `_assistant` muted), the
  engine meta line kept as a quieter subtitle beneath it. **Spacing + type scale** — `app.css`
  gains a small, *used* set of layout tokens (`--space-1..6`, `--text-meta/-sm/-title/-display`,
  `--measure`), applied to the header, footer, conversation padding, and empty/first-run states for
  coherent rhythm (no dead tokens — every token added is referenced). **Empty states** — the
  no-turns empty state and the no-corpus first-run banner are restyled (icon mark + serif headline +
  tightened copy); the empty state carries 2–3 clickable sample-question chips that **prefill the
  existing composer input** (no new turn/behavior — sets `input`, focuses the textarea). **Reading
  measure** — `Markdown.svelte` `.md` prose caps at `--measure` (~68ch), left-aligned, so answer and
  excerpt prose reads at a comfortable line length; source/provenance cards keep the fuller width.
  **Shell topology unchanged** (fork #9). `svelte-check` **0 errors**; both themes styled + legible
  (AA body); manual theme still overrides the OS media query; no horizontal overflow at mobile width;
  new icon carries correct `aria-hidden` semantics; chips are real `<button>`s with accessible labels;
  reduced-motion respected. **Byte-level backend / wire types unchanged** (a look pass changes no
  logic). Preview-harness-verified live, $0/offline: wordmark + restyled empty state + first-run
  banner on the app shell (both themes + mobile), a chip click prefilling the composer, and the ~68ch
  measure on a real reading surface (a live answer or the `$0` Library browser). **Copy de-tell
  (user request):** em dashes removed from all user-facing UI copy (an "AI-written" tell), swapped for
  the app's own idioms (period / comma / colon / parentheses / `·`); code comments untouched (not
  rendered); live-confirmed 0 em/en dashes in the rendered page.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set is SPEC-LED (SPRINT-016/010 precedent): a look pass legitimately touches the shell +
     the reading-surface component + the token source-of-truth. Expect tolerated uses⊇affects WARNs
     on the affects-only files — advisory (superseset_affects="warn"), not a gate failure. -->
- docs/specs/feature-visual-identity.md
- docs/ui-checklist.md
- apps/desktop/src/app.css
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/Icon.svelte
- apps/desktop/src/lib/Markdown.svelte
<!-- The 6 copy de-tell components (Settings/CompareCard/SourceCard/LibraryBrowser/Sidebar/Turn) are
     affects-only: adding them to `uses` blows the 2500-line read budget (Settings.svelte alone is
     ~480 lines). Per the SPRINT-016/010 precedent, they stay affects-only and produce tolerated
     `uses⊇affects` advisory WARNs under --strict (superseset_affects="warn") — not a gate failure;
     the change to each is a one-line text swap that needs no full preloading. -->

## affects
<!-- write-set: CSS tokens + the shell template + the reading-surface measure + the new icon glyph. -->
- apps/desktop/src/app.css
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/Icon.svelte
- apps/desktop/src/lib/Markdown.svelte
- docs/specs/feature-visual-identity.md
<!-- em-dash copy de-tell (user request, same session): removed em dashes from user-facing UI copy
     across these components, swapped for the app's own idioms (period/comma/colon/paren/middot).
     Text-only, no logic change. -->
- apps/desktop/src/lib/Settings.svelte
- apps/desktop/src/lib/CompareCard.svelte
- apps/desktop/src/lib/SourceCard.svelte
- apps/desktop/src/lib/LibraryBrowser.svelte
- apps/desktop/src/lib/Sidebar.svelte
- apps/desktop/src/lib/Turn.svelte
<!-- V1's committed contract, archived this session (git mv) now that V1 is committed (35b8627) —
     the skipped V1 post-commit hygiene, folded in here. -->
- docs/archive/sprints/SPRINT-016-visual-identity-v1.md

## contracts
<!-- No pytest surface (frontend-only). The gate is svelte-check; run in the verify loop. -->
- test: apps/desktop (npm run check) :: svelte-check 0 errors

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
- docs/ui-checklist.md
