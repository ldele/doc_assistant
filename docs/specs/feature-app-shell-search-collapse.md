<!-- status: design-locked · created: 2026-07-21 · owner: Code -->

# Feature spec — App-shell polish: global search overlay + collapsible sidebar

Code-level contract for sub-items **(a)** and **(b)** of the `docs/ui-checklist.md` row *"App shell →
chat-first layout"* (2026-07-20, user closing note). Frontend-only, no backend, no new API.

> **Out of scope, deliberately.** The same backlog row's third clause — *demote Graph out of the
> top-level nav* — is **NOT** built here. Where Graph goes is an open design fork (empty-Graph page →
> per-folder-concepts panel, ADR-025 fork 5, parked; the user is explicitly undecided) and the baton
> says *"do not move it without settling that"* → it needs `grill-me` first. This spec touches the
> Chat/Library/Graph switch not at all.

Reference framing (user): the conventional chat-app sidebar — *brand · search · collapse* in the
header, then *New chat / Library / …* below.

---

## What is being built

**(a) Global search overlay.** A scrim + centred dialog (Esc / scrim to close) that searches across
**conversations** and **library documents** and jumps to the chosen one. Opened from a header button
and from **Cmd/Ctrl-K**. Reuses the `LibraryKeywordFilter` modal shell (same `.scrim`/`.modal`
tokens, `svelte:window` Esc, autofocused input).

**(b) Collapsible sidebar.** A header toggle that hides the left rail on desktop and brings it back,
persisted like the other client-only view prefs (theme, sidebar width).

---

## (a) Global search — decisions

| # | Decision | Deciding reason |
|---|----------|-----------------|
| **A1** | **It is a *navigation* search, not a corpus search.** It matches conversation **titles** and document **title / filename / authors / keywords** — never message bodies or chunk text. Placeholder + empty state say so ("Search chats and documents"). | The composer **is** the corpus-search surface; a second box that looked like corpus search but wasn't would be the integrity lie the whole product avoids. And message bodies / chunk text aren't client-side — searching them needs a backend, which this row explicitly excludes. Honest scope: this jumps you *to* a chat or a document, it does not answer questions. |
| **A2** | **Trigger lives in the header** (a search button beside Settings) **+ Cmd/Ctrl-K**. Not in the sidebar. | The sidebar can be collapsed (b); a search entry point inside it would vanish exactly when you collapse. The header is always visible. Cmd/Ctrl-K is the standard command-palette affordance. |
| **A3** | **Results are two groups — Chats, then Documents** — each capped at **8**, with a muted "+N more — keep typing" line when a group overflows. | No silent truncation (cpc rigor): a cap that hides matches without saying so reads as "nothing more matched". Two groups because the two destinations are navigated differently (A6). |
| **A4** | **Empty query shows up to 6 recent (non-archived) chats** under a "Recent" heading; no documents. | Makes the overlay useful the instant it opens (the familiar quick-switcher behaviour) without inventing a document-recency notion. Recency = `last_at` desc, already the sidebar's default. |
| **A5** | **Keyboard-first:** input autofocused; ↑/↓ move a highlight across the *flat* result order; Enter opens the highlight; Esc closes. Mouse hover also sets the highlight. | A command palette that needs the mouse isn't one. The highlight indexes the flattened `[...chats, ...docs]` list so ↓ crosses the group boundary naturally. |
| **A6** | **Navigation:** a chat → `selectMode('chat')` then `openConversation(sid)`; a document → `selectMode('library')` then `openDocument(id)`. Overlay closes on select. | Reuse the exact existing entry points (no new nav logic). `selectMode` already lazy-loads what each mode needs; opening a doc while in chat mode would render nothing. |
| **A7** | **On open, refresh conversations + documents** (`refreshDocuments()` if `!documentsLoaded`), inform-don't-block. | Documents lazy-load only on entering Library today; a user who never opened the Library must still find their papers. Both are sidecar reads — a failure keeps the prior list, never blocks the overlay. |
| **A8** | **The match logic is a pure module** `lib/search.ts` (`searchEverything`), unit-tested via `node:test`. | House pattern since PR-2.5/2.7: rules live in pure helpers behind `npm test`, not eyeballed in a component. The overlay component stays a dumb renderer; App owns the data + navigation. |

**Match semantics (A1, in `searchEverything`):** case-insensitive substring, query trimmed. A chat
matches on `title`. A document matches on `docLabel(d)` **or** `filename` **or** `authors` **or** any
`keywords[]` entry (a superset of the Library rail's `filterDocs`, which omits keywords — a keyword is
exactly the kind of term you'd search a paper by). Order preserved from the inputs (chats by the
caller's recency order, docs by the caller's order); dedupe is not needed (ids are unique per group).

## (b) Collapsible sidebar — decisions

| # | Decision | Deciding reason |
|---|----------|-----------------|
| **B1** | **Desktop-only affordance.** The toggle button shows at ≥721 px; under 720 px the sidebar is already an off-canvas drawer driven by the hamburger, which is untouched. | Two controls for one slot, split by the existing 720 px breakpoint: hamburger (mobile) ↔ collapse (desktop). `sidebarCollapsed` never affects the mobile drawer (`sidebarOpen`). |
| **B2** | **`sidebarCollapsed` is a client-only pref in `localStorage('sidebarCollapsed')`.** | Same class as `theme` and `sidebarWidth` — a view preference, never a backend setting (apps/desktop rule). |
| **B3** | **Collapse hides `.sidebar` *and* `.resizer`** via `.app.collapsed` under a `min-width:721px` guard; expanding restores the persisted `sidebarWidth` untouched. | The drag handle is meaningless with nothing to drag. Width is preserved because collapse ≠ resize — you get your rail back exactly as wide as you left it. |
| **B4** | **One toggle button, `panel-left` glyph, label flips** ("Collapse sidebar" ↔ "Expand sidebar"). New Lucide `panel-left` path added to `Icon.svelte`. | The icon set is a closed union; one glyph + an `aria-label`/`title` that reflects state is less chrome than two icons. |

---

## Files

**New**
- `apps/desktop/src/lib/search.ts` — `SearchResults` type + `searchEverything(query, chats, docs, opts)` (pure).
- `apps/desktop/src/lib/search.test.ts` — `node:test` unit tests for `searchEverything`.
- `apps/desktop/src/lib/GlobalSearch.svelte` — the overlay (dumb renderer; props in, `onSelectChat`/`onSelectDoc`/`onClose` out).

**Edited**
- `apps/desktop/src/lib/Icon.svelte` — add `panel-left`.
- `apps/desktop/src/App.svelte` — `searchOpen` + `sidebarCollapsed` state (+ load/persist), a global Cmd/Ctrl-K handler, two header buttons (search + collapse), `class:collapsed` on `.app`, `<GlobalSearch>` render, collapse CSS.
- `docs/DEVLOG.md`, `docs/ui-checklist.md`.

No wire-type change (`types.ts`/`models.py` untouched) — nothing crosses the API boundary.

---

## Definition of Done

1. **Search opens** from the header button and from Cmd/Ctrl-K (and Cmd/Ctrl-K does **not** fire while
   typing into it — it toggles closed or is a no-op, never inserts a "k").
2. **Typing a query** shows matching chats then documents, each capped at 8 with a "+N more" line when
   overflowing; **empty query** shows up to 6 recent chats.
3. **Selecting** a chat opens it in Chat mode; selecting a document opens it in Library mode; the
   overlay closes in both cases. ↑/↓/Enter/Esc and mouse all work.
4. **Honest-empty:** a query that matches nothing shows a "No chats or documents match …" line, not a
   blank box. Zero documents / zero chats never throws.
5. **Collapse** hides the rail on desktop and the button re-expands it to the same width; state
   survives a reload; **no effect** on the mobile drawer (< 720 px still uses the hamburger).
6. `svelte-check` **0/0**; `npm test` green (existing + new `search.test.ts`).
7. **Live preview, $0/offline:** overlay open → type → keyboard-navigate → open a chat and a document;
   collapse/expand round-trip; light + dark; 375 px no horizontal overflow; 0 console errors. Verified
   with `read_page` + computed styles per the box's screenshot-flakiness rule.

## Guard against regressions

- The Chat/Library/Graph mode switch and the sidebar's own per-mode search inputs are **unchanged** —
  the new overlay is additive. A grep for `onSelectMode`/`selectMode` call sites must show no behaviour
  change beyond the two new callers in the overlay's navigation handlers.
- `searchEverything('', …)` returns recents, never the whole document list (an empty query must not
  dump 76 papers into the overlay).
