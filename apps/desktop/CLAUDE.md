# apps/desktop/ — Svelte 5 / Tauri frontend (thin renderer)

**Owns:** the desktop UI only. No business logic — renders what `apps/api` serves; never imports
from `src/` (non-negotiable #3, `.claude/CONTEXT.md`).

**Key files**
- `src/App.svelte` — shell (`sidebar│main│drawer`), mode switch (chat/library/graph), theme.
- `src/lib/api.ts` — the one API client; `src/lib/types.ts` — wire types, mirror `apps/api/models.py`.
- `src/lib/*.svelte` — components (Turn, SourcePanel, Settings, Library*, ConceptGraph, Sources…).
- `src/app.css` + `src/lib/fonts.css` — "paper & ink" tokens (4 theme blocks), vendored fonts.
- `src/lib/theme.ts` — System/Light/Dark toggle, `localStorage`, never a backend setting.

**Rules that bite here**
- Wire-type drift: change `apps/api/models.py` ⇒ update `types.ts` in the same change.
- Verify with `npm run check` (`svelte-check` 0 errors) + live preview: light + dark + mobile
  (375px, no overflow), 0 console errors, $0/offline where possible.
- Product name is **Provenote** (ADR-012): wordmark/window title only — package/binary names stay
  `doc_assistant`; do not "finish" the rename.
- Dev run: `npm run dev` (Vite :1420) against `just api` (:8001), or `just app` for both.

**Tests:** none (no JS test runner) — `svelte-check` + the live preview harness are the gate.

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->
