# apps/desktop/ — Svelte 5 / Tauri frontend (thin renderer)

**Owns:** the desktop UI only. No business logic — renders what `apps/api` serves; never imports
from `src/` (non-negotiable #3, `.claude/CONTEXT.md`).

**Key files**
- `src/App.svelte` — the shell: cross-domain orchestration only (nav history, readiness gate,
  `selectMode`, chat-scope guard, conversation/chat lifecycle) + overlay wiring.
- `src/lib/<domain>/` — one folder per domain, mirroring `apps/api/routers/`: `chat` · `library` ·
  `graph` · `settings`, plus `shell/` (Topbar, StatusBar, Sidebar, search, dialogs, `Icon`) and
  `core/` (wire boundary + theme/fonts), which have no API counterpart. End-to-end table +
  naming traps: `docs/architecture.md` § `apps/` — the domain spine.
- `<domain>/*.svelte.ts` — **rune modules** holding that domain's `$state` (7). Export ONE `$state`
  *object*: an imported binding can't be reassigned across a module boundary.
- `core/api|types/<domain>.ts` — one module per `apps/api/models/` domain (barrels re-export, so
  `../core/api` resolves). `app.css`+`core/fonts.css` = "paper & ink" tokens; `core/theme.ts` =
  theme toggle (`localStorage`, never a backend setting).

**Rules that bite here**
- Wire-type drift: change `apps/api/models/<domain>.py` ⇒ update `core/types/<domain>.ts` in the
  same change. Same word both sides — that is the point of the split.
- **`.svelte.ts` can't run under `node:test`** (needs the compiler). Keep pure, tested logic in
  plain `.ts` beside it — the extension is the marker (`taxonomy.ts` tested, `taxonomy.svelte.ts`
  state). `$effect` can't run at module top level: export a `useXxx()` hook, call it from a script.
- **Renaming a variable breaks Svelte shorthand** — `{x}`, `x,` and `bind:x` each need hand-editing,
  and a rename regex must exclude `<` or `<input` becomes `<chat.input` (bit 3× on 2026-07-26).
  A `const` prop can't be bound: use `$bindable()`.
- **No optional params in `<script lang="ts">` functions.** The TS-strip drops the type but leaves the
  `?`, emitting `function f(x?)` → `SyntaxError: Unexpected token '?'` that blanks the whole app mount —
  and `svelte-check` passes it (it checks the source). Use a defaulted param: `x: T | null = null`.
- **The type gate is not the run gate:** asset paths and mount failures are invisible to
  `svelte-check` *and* `node:test`. Verify with `npm run check` + live preview (light/dark/375px,
  0 console errors, $0). Geometry looks broken? Check `innerWidth` — a hidden pane collapses it to
  0 and freezes transitions (DEVLOG 2026-07-26).
- Product name: ADR-012 (wordmark only — never rename the package). Dev: `just app`.

**Tests:** `npm test` (`node:test`, zero deps; pure `lib/**/*.ts` only) + `svelte-check` + the live
preview are the gate. A tested `.ts` module must stay free of runtime *value* imports from siblings
(node strips type-only imports but can't resolve extensionless value ones).

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->
