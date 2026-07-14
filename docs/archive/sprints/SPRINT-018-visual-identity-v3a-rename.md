<!-- status: archived · updated: 2026-07-14 · class: disposable · built: V3a committed 181046c -->
<!-- BUILDING 2026-07-14. Sole active contract (SPRINT-017/V2 committed 4fd772c + archived).
     Design lock: docs/specs/feature-visual-identity.md §V3 / §V3a (name locked 2026-07-14 = Provenote;
     scope decisions with the user: rename + polish audit in V3a, app icon/branding carved to V3b;
     bundle id -> com.provenote.desktop). Feature: Visual-identity V3a — rename doc_assistant -> Provenote
     across the PRODUCT identity + a cross-screen polish audit. FRONTEND + one Tauri-config + docs; no
     src/ change, no wire type, no locked setting, no behavior change. $0/offline. App icon + branding
     assets are OUT (V3b follow-up — image tooling, not browser-verifiable). Gate = svelte-check 0. -->

# SPRINT-018 — visual-identity-v3a-rename

- **base:** main
- **DoD:** The **product** identity is renamed `doc_assistant` -> `Provenote`; the internal Python
  package, the npm package name (`doc-assistant-desktop`), the API sidecar binary (`doc-assistant-api`),
  the `python -m doc_assistant.*` commands, and the dev/architecture docs **keep `doc_assistant`**
  (module != product). Specifically:
  - **Wordmark** — `App.svelte` renders `Provenote` in `--font-serif`, **treatment B**: lowercase,
    `proven` in `--fg` ink + `ote` in `--accent` indigo (rename the V2 `.wm-dim` scoped class in
    `App.svelte` to `.wm-accent` + repoint its color to `var(--accent)` — no dead selector), beside the
    existing `book-open` mark.
  - **Titles** — `index.html` `<title>` and `tauri.conf.json` `app.windows[0].title` -> `Provenote`.
  - **Tauri bundle** — `productName` -> `Provenote`; `identifier` `com.doc-assistant.desktop` ->
    `com.provenote.desktop`. `externalBin` (`binaries/doc-assistant-api`) **unchanged** (internal
    sidecar; renaming risks the M4 freeze pipeline).
  - **Docs** — `README.md` H1 (`Document Assistant` -> `Provenote`) + the one prose product mention;
    `package.json` `description` product reference -> `Provenote` (internal package/module/command
    references to `doc_assistant` stay).
  - **Cross-screen polish audit** — walk every shipped surface (chat + streaming, empty/first-run
    states, Library browser, Settings drawer, A/B Compare card, citation side panel, provenance/claim
    cards) in **both themes + mobile**; fix any spacing/type/measure blemish the V2 rhythm surfaced;
    record the audit findings (fixed vs. deferred) in the DEVLOG entry.
  - `svelte-check` **0 errors**; the Provenote wordmark + `<title>` verified live ($0/offline, preview
    harness) in both themes + mobile, no horizontal overflow; **no behavior / wire-type / logic /
    locked-setting change**; sidecar binary + internal package names confirmed unchanged.
- **Out of scope (this sprint):** the **app icon + branding assets** (`src-tauri/icons/*` regeneration
  from a designed Provenote source) — carved to **V3b** (needs the Tauri CLI + SVG rasterization, and is
  not browser-verifiable). Any `src/` / API / packaging-pipeline change beyond the two config strings.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check. -->

## uses
- docs/specs/feature-visual-identity.md
- docs/ui-checklist.md
- apps/desktop/src/App.svelte
- apps/desktop/src/app.css
- apps/desktop/index.html
- apps/desktop/src-tauri/tauri.conf.json
- apps/desktop/package.json

## affects
- apps/desktop/src/App.svelte
- apps/desktop/index.html
- apps/desktop/src-tauri/tauri.conf.json
- apps/desktop/package.json
- README.md
- docs/specs/feature-visual-identity.md
<!-- Plus any component the polish audit lands a blemish fix in (affects-only, tolerated uses⊇affects
     WARN under --strict — same precedent as SPRINT-016/017); the audit's write-set isn't knowable
     until the walk runs. -->

## contracts
- test: apps/desktop (npm run check) :: svelte-check 0 errors

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
- docs/ui-checklist.md
