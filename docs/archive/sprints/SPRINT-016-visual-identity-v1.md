<!-- status: archived · updated: 2026-07-14 · class: disposable · built: V1 committed 35b8627 -->
<!-- BUILDING 2026-07-13. Sole active contract. Design lock:
     docs/specs/feature-visual-identity.md (grilled 2026-07-13, 11 forks, design-locked).
     Feature: Visual-identity pass V1 — "paper & ink". Re-key the app.css token set to a warm
     ivory/charcoal palette with a deep-indigo accent + font-stack + 2 shadow tokens; add Icon.svelte
     (Lucide inline SVGs) and replace every chrome emoji glyph; apply a scholarly serif (Spectral) to
     reading surfaces, chrome stays sans (Inter). FRONTEND-ONLY — CSS + Svelte templates; no backend,
     no src/ change, no wire type, no locked setting, no behavior change. $0/offline. Fonts resolved
     vendored+committed (user pick) — 4 woff2 in assets/fonts/ + lib/fonts.css. Light palette pulled to
     white/ivory on user feedback (the first warm-ivory cut read too beige). Pre-existing [lifecycle] warn applies
     (tolerated across SPRINT-*.md). Gate for a frontend-only sprint = svelte-check 0 (no pytest surface). -->

# SPRINT-016 — visual-identity-v1

- **base:** main
- **DoD:** `apps/desktop/src/app.css` re-keyed to the paper & ink palette across **all four** theme
  blocks (`:root` light default · `[data-theme='dark']` · `[data-theme='light']` ·
  `@media (prefers-color-scheme: dark) :root:not([data-theme])`) — warm ivory/charcoal neutrals,
  **deep-indigo** `--accent` (no warn/ok collision), warmed warn/ok pair — plus new `--font-sans` /
  `--font-serif` stack tokens and two shadow tokens (`--shadow-1` resting, `--shadow-2` raised,
  per-theme opacity). New `apps/desktop/src/lib/Icon.svelte` renders **Lucide** inline SVGs
  (`currentColor`, `aria-hidden` default, `size` prop); **every chrome emoji glyph replaced**
  (`☰`→menu, `⬇`→download, `⚙`→settings, `←`→arrow-left, `↻`→rotate-ccw, `✕`→x, `✓`→check,
  `⚠`→triangle-alert; `●` → CSS-drawn dot). `--font-serif` applied to reading surfaces (answers via
  `Markdown.svelte` `.md`; library chunk text; source excerpts; content `h1/h2/h3`); chrome stays
  `--font-sans`. Spectral + Inter loaded locally via `@font-face` from **vendored + committed** woff2
  (`apps/desktop/src/assets/fonts/` + `lib/fonts.css`, imported in `main.ts`; no CDN — the user's
  mechanism pick). `svelte-check` **0 errors**; both themes styled + legible (AA body);
  manual theme still overrides the OS media query; no horizontal overflow at mobile width; icon buttons
  keep correct `aria-hidden`/label semantics; reduced-motion respected. **Byte-level behavior, wire
  types, and backend unchanged** (a look pass changes no logic). Preview-harness-verified live,
  $0/offline: palette + icons on the app shell (both themes + mobile) and serif on a real reading
  surface (the Library browser is $0).

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set is SPEC-LED (U1/SPRINT-010 precedent): a look pass legitimately touches ALL of
     apps/desktop/src, which cannot fit the 12-file/2500-line uses budget. So `uses` lists the design
     lock + the token/icon/font source-of-truth anchors; the emoji-glyph swaps across the other 9
     components are mechanical (import Icon, replace one glyph, add flex alignment) and don't need full
     preloading. Expect tolerated `uses⊇affects` WARNs for the affects-only components — advisory
     (superseset_affects="warn"), not a gate failure, same as SPRINT-010. -->
- docs/specs/feature-visual-identity.md
- docs/ui-checklist.md
- apps/desktop/src/app.css
- apps/desktop/src/main.ts
- apps/desktop/src/lib/theme.ts
- apps/desktop/src/lib/Icon.svelte
- apps/desktop/src/lib/Markdown.svelte

## affects
<!-- write-set: CSS + Svelte templates + new Icon component + local font assets. Frontend-only. -->
- apps/desktop/src/app.css
- apps/desktop/src/main.ts
- apps/desktop/src/lib/Icon.svelte
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/Markdown.svelte
- apps/desktop/src/lib/SourceCard.svelte
- apps/desktop/src/lib/SourcePanel.svelte
- apps/desktop/src/lib/LibraryBrowser.svelte
- apps/desktop/src/lib/Sidebar.svelte
- apps/desktop/src/lib/Settings.svelte
- apps/desktop/src/lib/CompareCard.svelte
- apps/desktop/src/lib/ClaimReview.svelte
- apps/desktop/src/lib/Turn.svelte
- apps/desktop/src/lib/fonts.css
- apps/desktop/src/assets/fonts/*
- docs/specs/feature-visual-identity.md

## contracts
<!-- No pytest surface (frontend-only). The gate is svelte-check; run in the verify loop. -->
- test: apps/desktop (npm run check) :: svelte-check 0 errors

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
- docs/ui-checklist.md
