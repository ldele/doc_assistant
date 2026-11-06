# apps/desktop/ — Svelte 5 / Tauri frontend (thin renderer)

**Owns:** the desktop UI only. No business logic — renders what `apps/api` serves; never imports
from `src/` (non-negotiable #3, `.claude/CONTEXT.md`).

**Key files**
- `src/App.svelte` — the shell: cross-domain orchestration only (nav history, readiness gate,
  `selectMode`, chat-scope guard, conversation/chat lifecycle) + every overlay's mount.
- `src/screens/<screen>/` — **components, by what the app shows** (ADR-064, GLOSSARY U-004): an
  entry file named for the folder, its parts, its dialogs (named for their headings). `src/shell/`:
  the frame, its windows, `Icon` + `ConfirmDialog`. Every door: `docs/architecture.md` § the screens.
  First run (`GettingStarted`, `ProviderSetup`) renders `/api/setup`'s verdicts, never recomputes.
- `src/lib/<domain>/` — **logic and state, no components**, mirroring `apps/api/routers/` (`chat` ·
  `library` · `graph` · `settings` · `shell` · `core`): `docs/architecture.md` § the domain spine.
- `<domain>/*.svelte.ts` — **rune modules** (13): export ONE `$state` *object* (an import can't be reassigned).
- `lib/core/api|types/<domain>.ts` — one per `apps/api/models/` domain; tokens: `app.css` + `core/fonts.css`.

**Rules that bite here**
- **Imports point one way** (`src/layout.test.ts`): no `.ts` in `screens/`/`shell/`, no component in
  `lib/`, nothing in `lib/`/`shell/` imports `screens/`. History before 2026-11-05 needs `--follow`.
- Wire-type drift: change `apps/api/models/<domain>.py` ⇒ update `core/types/<domain>.ts` in the
  same change. Same word both sides — that is the point of the split.
- **`.svelte.ts` can't run under `node:test`** (needs the compiler). Keep pure, tested logic in plain
  `.ts` beside it — the extension is the marker (`taxonomy.ts` tested, `taxonomy.svelte.ts` state);
  `$effect` needs a script, so export a `useXxx()` hook rather than running it at module top level.
- **Renaming a variable breaks Svelte shorthand** — `{x}`, `x,` and `bind:x` each need hand-editing;
  a rename regex must exclude `<`. A `const` prop needs `$bindable()`.
- **No optional params in `<script lang="ts">` functions.** The TS-strip drops the type but leaves the
  `?`, emitting `function f(x?)` → `SyntaxError: Unexpected token '?'` that blanks the whole app mount —
  and `svelte-check` passes it (it checks the source). Use a defaulted param: `x: T | null = null`.
- **The type gate is not the run gate:** asset paths and mount failures are invisible to `svelte-check`
  *and* `node:test`; verify in the live preview (light/dark/375px, 0 console errors, $0). A hidden pane
  sets `innerWidth` to 0, defers Svelte's flush and freezes transitions (a panel "never closes"): when
  DOM and state disagree, read the rune module from the page (DEVLOG 2026-07-26, -07-28, -08-31 (3)).
- Product name: ADR-012 (wordmark only — never rename the package). Dev: `just app`.

**Tests:** `npm test` (`node:test`, zero deps; pure `.ts` only) + `svelte-check` + live preview are the gate. A tested `.ts` module importing a sibling needs the explicit `.ts` extension.
A button takes `use:tooltip={…}` (`lib/shell/tooltip.ts`), never `title`; `src/ui-rules.test.ts` fails one without it, a local `.primary`/`.ghost`/`.danger`, a literal colour (ADR-063).

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->
