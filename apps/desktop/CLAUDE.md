# apps/desktop/ — Svelte 5 / Tauri frontend (thin renderer)

**Owns:** the desktop UI only. No business logic — renders what `apps/api` serves; never imports
from `src/` (non-negotiable #3, `.claude/CONTEXT.md`).

**Key files**
- `src/App.svelte` — shell (`sidebar│main│drawer`), mode switch (chat/library/graph), theme.
- `src/lib/<domain>/` — one folder per domain, matching `apps/api/routers/`: `chat/` · `library/` ·
  `graph/` · `settings/`, plus two with no API counterpart — `shell/` (sidebar, search, dialogs,
  `Icon.svelte`) and `core/` (`api.ts`, `types.ts`, `theme.ts`, `fonts.css`). End-to-end table:
  `docs/architecture.md` § `apps/` — the domain spine.
- `src/lib/core/api.ts` — the one API client; `src/lib/core/types.ts` — wire types, mirror
  `apps/api/models/`. `src/app.css` + `src/lib/core/fonts.css` — "paper & ink" tokens, vendored fonts.
- `src/lib/core/theme.ts` — System/Light/Dark toggle, `localStorage`, never a backend setting.
- Naming trap: `settings/Sources.svelte` is **ingestion** (files on disk); the *citation* sources of
  an answer are `chat/Source*.svelte`.

**Rules that bite here**
- Wire-type drift: change `apps/api/models/<domain>.py` ⇒ update `core/types.ts` in the same change.
- Verify with `npm run check` (`svelte-check` 0 errors) + live preview: light + dark + mobile
  (375px, no overflow), 0 console errors, $0/offline where possible.
- Product name is **Provenote** (ADR-012): wordmark/window title only — package/binary names stay
  `doc_assistant`; do not "finish" the rename.
- **No optional params in `<script lang="ts">` functions.** The TS-strip drops the type but leaves the
  `?`, emitting `function f(x?)` → `SyntaxError: Unexpected token '?'` that blanks the whole app mount —
  and `svelte-check` passes it (it checks the source). Use a defaulted param: `x: T | null = null`.
- Dev run: `npm run dev` (Vite :1420) against `just api` (:8001), or `just app` for both.

**Tests:** `npm test` (Node's built-in `node:test`, zero deps; pure `lib/*.ts` only — since PR-2.5)
+ `svelte-check` + the live preview harness are the gate. A tested `.ts` module must stay free of
runtime *value* imports from sibling modules (node strips type-only imports but can't resolve
extensionless value ones) — keep pure helpers self-contained.

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->
