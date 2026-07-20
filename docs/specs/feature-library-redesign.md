# Spec — Library redesign: folder rail + inventory grid + drill-down

**Status:** ✅ **Phase A SHIPPED 2026-07-15 (commit `9f597df`; library follow-ups `8f31fe3` metadata
enrichment · `e549254` metadata editing/ADR-013 · `95817fc` safe-delete/ADR-014). Phase B (folders)
NOT built.** Was DESIGN-LOCKED (grilled 2026-07-14, `grill-me`; ledger at foot). Direction chosen in-session by
the user from a clickable prototype
([artifact](https://claude.ai/code/artifact/54804676-82fc-4d93-9f54-c4a70b074dfd)). The next Library
increment after `feature-library-browser.md` (L1, shipped), which explicitly parked "search / filter / sort
within the library" and any folder/type navigation. Call it **L4** on the roadmap.

**Owner:** Claude Code (frontend rebuild + a thin backend wiring pass). **Two phases, two PRs** — never bundle.
**Build Phase A first** (frontend, real-data, $0); Phase B (folders) follows.

---

## The decision (user, 2026-07-14)

Keep the existing left rail, but make it a **navigation tree** (folders / types / date), with **expandable
folders**. The main pane becomes an **inventory grid** of the selected collection's documents (video-game
-inventory feel — a flat list laid out as a 2-D tile grid), and opening a document **drills down** in place
to the existing parent→child chunk view. Navigation model = **drill-down with Back**: a breadcrumb
`Library › Folder › Document` plus a Back control walks back up. The persistent rail means changing folder
never needs "back" — only the document→chunks step drills.

Rejected in the prototype: two-pane (chunks always in a right pane) and detail-drawer. The user preferred
the drill-down's file-browser feel; the drawer variant is noted as a possible later toggle (it would reuse
`SourcePanel`).

## Grounding (read from the live corpus + code 2026-07-14, not assumed)

- **The folder/tag model already exists.** `db/models.py` has `Folder` (**hierarchical** —
  `parent_folder_id`, self-referential `children`/`parent`, `name`, `description`), `Tag`, `Keyword`, and
  the `document_folders` / `document_tags` / `document_keywords` M2M join tables. `Document.folders/tags/
  keywords` relationships are live. **No new schema is needed for navigation.**
- **`library.list_documents(health, format, tag, folder)` already filters** by format / tag / folder
  (joins the M2M). `LibraryDocumentPayload` already ships `folders`, `tags`, `keywords`, `page_count`,
  `added_at` to the client (`types.ts` mirrors them). The wire contract is already rich enough.
- **Populated on the real 76-doc corpus today:** `format` (all `pdf` here, but the axis is real),
  `page_count` (64/76), `added_at` (76/76), `chunk_count`, `health`, and **`keywords`** (auto-extracted,
  partial). → **Type, Date, and keyword grouping ship on real data with zero backend change.**
- **Empty on the corpus:** `folders` (0), `tags` (0), `title`/`authors` (0/76). Nothing assigns folders at
  ingest; the corpus was ingested from a flat directory. → Folders need a **population path** (below); tiles
  show **filenames** until metadata extraction improves (unchanged from L1's honest-empty rule).

## Phasing

**Phase A — frontend, ships now, $0/offline, no backend change.** The whole navigation model on real data:
inventory grid + grid/list toggle + drill-down/back + a rail with **Types** (by `format`) and **Added** (by
date bucket) sections + a **keywords** quick-filter. The **Collections (folders)** rail section renders but
is empty-stated ("No folders yet — see Phase B") until populated. This delivers everything the user can see
and click; it is provable on the preview harness like L1.

**Phase B — backend, lights up folders.** A population path for `Folder` rows + the rail's Collections tree
+ server-side filtering. Gated on the folder-population decision below.

---

## Decisions

| # | Decision | Reason |
|---|----------|--------|
| 1 | **Navigation = drill-down + Back**, persistent rail. Main pane swaps collection-grid ⇄ doc-chunks; breadcrumb `Library › Folder › Doc` + Back. Rail never needs Back (click another node). | User's choice; the one deep step is doc→chunks |
| 2 | **Inventory grid is the default** collection view; a **grid ⇄ list toggle** (list = today's `.row` idiom) persists in `localStorage` (`libraryView`), like the other client-only view prefs. | The user's "2-D inventory"; keep the familiar list as an option |
| 3 | **Rail = a nav tree**, not the doc list. Section order: **All documents** → **Collections** (folders, hierarchical, expandable — expanding reveals child docs inline for quick-pick) → **Types** (`format`) → **Added** (date buckets from `added_at`). | Matches the prototype + the user's "expand folder on the left"; organization → format → time |
| 3a | **Adaptive sections** (grill Q3): Types / Added render only when they'd have **≥2 entries** — a one-format or one-bucket section is a dead filter, so it's hidden until the corpus earns it (all-PDF, all-early-July today → both hidden). | A single-option filter is noise; appears automatically when the corpus diversifies |
| 3b | **Date buckets** (grill Q2): `Today` / `This week` / `This month` / `Earlier`, relative to now, adaptive-hidden per 3a. | Familiar relative bucketing; no config |
| 4 | **The doc list moves into the main pane** (as the grid). The chunk view is the **existing `LibraryBrowser.svelte`** rendered in the drill-down slot — reused verbatim, not rebuilt. | Maximal reuse; the deep view already exists and is design-locked |
| 4a | **Rail ↔ main sync** (grill Q5): clicking a document **anywhere** (rail tree or main grid) drills the main pane into it; selecting/expanding a folder sets it the active collection so the grid shows its docs. One behavior, no divergence. | Predictable; the rail tree and grid are two views of one selection |
| 5 | **Reuse `list_documents(...)` filters + the `Folder`/`Tag` model.** No new table. Phase A filters **client-side** (payload already carries every field); Phase B wires the existing server-side filters + a folder-tree endpoint. | The infrastructure is already there; don't duplicate it |
| 5a | **Search scope** (grill Q4): the search bar filters the **active collection**; a 0-match empty-state offers a one-click **"Search all N documents"** escape. | "Filter what's in front of me" without dead-ends |
| 6 | **Folder population = mirror source-dir subfolders at ingest + a one-off backfill** (grill Q6, **locked** as the Phase B direction). Creates/reuses hierarchical `Folder` rows keyed by the relative sub-path, links the document. **Reopens if** the user's `source_dir` is intentionally flat → manual assignment fits better. **Projects** = folders with no filesystem parent (`ChatConcept.folder_id` note) — deferred. | Local-first, zero manual tagging, fits the hierarchical model |
| 7 | **Read-only still.** Phase A/B write nothing at browse time; folders are written *at ingest / by the backfill*, not by the browser. Manual folder/tag **editing** in the UI = a later increment (first browse-time write path — same ADR consideration L3 flagged). | Enrichment-Layer discipline; no scope creep |
| 8 | **Honest empties throughout** (L1 rule): NULL title→filename; empty Collections→a "why" empty-state, never a fake tree; a keyword/type with 0 docs is hidden. | inform-don't-block |

## Contract — Phase A (frontend only)

- `apps/desktop/src/lib/LibraryGrid.svelte` (new) — given a `documents` array (a collection), render the
  **inventory grid** (tiles: format chip, filename/title, `page_count`·`chunk_count`·`added_at`, up to 3
  keyword chips) or the **list** rows per `libraryView`; emits `onOpenDocument(id)`. Both themes; no body
  overflow; `grid-template-columns: repeat(auto-fill, minmax(150px, 1fr))`.
- `apps/desktop/src/lib/Sidebar.svelte` — in `library` mode, replace the flat doc list with the **nav
  tree** in order: "All documents" → **Collections** (Phase A: empty-state) → **Types** (group by
  `format`) → **Added** (buckets `Today`/`This week`/`This month`/`Earlier` from `added_at`). Types/Added
  render only with **≥2 entries** (Decision 3a). Selecting a node sets the active collection; expanding a
  folder (Phase B) reveals its docs inline, and clicking a doc there drills the main pane (Decision 4a).
  The **search bar** (just shipped) filters the active collection, with a "Search all" escape on 0 matches
  (Decision 5a). Chat mode is untouched.
- `apps/desktop/src/App.svelte` — library drill-down state: `libraryCollection` (`{kind:'all'|'type'|
  'date'|'folder'|'keyword', value}`), `libraryDocId | null`, derived breadcrumb. Main pane renders
  `LibraryGrid` (collection) or `LibraryBrowser` (a doc is open) with a breadcrumb `Library › Collection ›
  Doc` + Back bar (Back: doc→grid, then collection→all). Client-side `docsFor(collection)` filter over the
  cached document list. `libraryView` (`grid`|`list`) in `localStorage`.
- No `api.ts` / `types.ts` / backend change in Phase A (payload already sufficient).

## Contract — Phase B (backend wiring, gated on the decision)

- **Ingest** (`ingest/…`) — on ingest, derive the document's folder path from its location under
  `source_dir` (each sub-path segment → a nested `Folder`, `get-or-create` by `(name, parent)`); link the
  `Document`. Idempotent; re-ingest reconciles. No-op when the file sits at `source_dir` root.
- **Backfill** — a CLI runner (Enrichment-Layer style, `python -m doc_assistant.<runner>`) that walks the
  current `source_dir` and assigns folders to already-ingested docs, so the existing flat corpus can be
  organised without a full re-ingest.
- **API** — `GET /api/library/folders` → the folder tree (id, name, parent_id, doc_count) [payload
  `LibraryFolderPayload`]; add `folder` / `format` / `tag` query params to `GET /api/library/documents`
  (wire the existing `list_documents` filters). `api.ts` + `types.ts` mirror.
- **Frontend** — the rail's Collections section renders the tree (expand/collapse, counts); selecting a
  folder filters the grid (server or client).

## Tests

**Phase A** — `svelte-check` 0/0; live preview-harness proof on the real corpus ($0/offline): grid renders
76 docs, Type/Added sections group correctly, a tile opens the chunk view, Back returns, grid/list toggle
persists, both themes.

**Phase B** — unit: folder-path→nested-`Folder` derivation (root file → no folder; `a/b/x.pdf` → `a`→`b`);
get-or-create idempotency; backfill assignment. Integration: `GET /api/library/folders` tree + `doc_count`;
`?folder=/?format=/?tag=` filter the doc list; a write-trap guard that **browsing** still writes nothing.
Full gate green (`ruff`/`ruff format`/`mypy --strict src`/`bandit`/`pytest`).

## Definition of done

- **Phase A:** Library rail is a nav tree (Types/Added real, Collections empty-stated); main pane is the
  inventory grid; opening a doc drills to its chunks with breadcrumb + Back; grid/list toggle persists;
  both themes; mobile off-canvas still works; one DEVLOG entry; ROADMAP L4 row.
- **Phase B:** folders populate at ingest + a backfill for the current corpus; the rail's Collections tree
  is live and filters the grid; `GET /api/library/folders` + filter params + tests; one DEVLOG entry.

## Folder population — REOPENED then SHELVED (user, 2026-07-15)

Was locked to **A: mirror source-dir subfolders at ingest + a one-off backfill** (Decision 6), with an
explicit **reopens-if the `source_dir` is intentionally flat**. **That condition is now confirmed
true** (user, 2026-07-15): the source is a single flat folder by design (0 folders on the corpus is
not a metadata gap — there are no subfolders to mirror). So **path-derived folder-mirroring is
shelved** — Phase B as specified (ingest-time subfolder mirror + backfill) is **not built**. The
Phase-A **Collections** rail section stays honest-empty (as shipped). If folder/Collection grouping is
ever wanted on a flat corpus, the path is **manual assignment** (option B — its own ADR, the first
browse-time write path), never a fake path-derived tree. Deferring folders entirely (C) is effectively
where we land for now. **Note for the ingestion work:** with no folders, **`doc_type`** (paper/book/
web/note) is the primary selection axis in `feature-selective-ingestion.md`, not folder scoping.

## Phase B superseded by ADR-025 F1 (2026-07-20)

The shelving above held: subfolder mirroring was **not** built. The manual-assignment path this
section named as the alternative ("option B — its own ADR, the first browse-time write path") was
unblocked by `docs/decisions/ADR-025-corpus-folders-retrieval-scope.md` and **shipped as F1** —
contract and the full reconciliation of these Phase-B locks in
`docs/specs/feature-corpus-folders.md`. What carried over from Phase B: the `GET
/api/library/folders` endpoint (extended to full CRUD) and the rail's folder section filtering the
grid. What did not: path-derived mirroring, its backfill runner, and the hierarchical tree (F1 is
flat — spec D1). Decision 7's "browsing writes nothing" is narrowed, not dropped (spec D7).

## Decision ledger (grill-me 2026-07-14)

| Branch | Resolution | Deciding reason / reopens-if |
|---|---|---|
| Phasing | **Phase A (frontend, real-data) ships first**; folders = Phase B | ~90% of value now, provable $0; reopens if an empty Collections section is unacceptable |
| Single-type sections | **Adaptive** — Types/Added hidden below 2 entries | a one-option filter is noise; auto-appears when the corpus diversifies |
| Search scope | **Active collection** + "Search all N" escape on 0 matches | "filter what's in front of me" without dead-ends |
| Folder source (Phase B) | **Mirror source subfolders at ingest + backfill** | local-first, zero tagging, model already hierarchical; reopens if `source_dir` is intentionally flat |
| Date buckets | `Today`/`This week`/`This month`/`Earlier`, adaptive-hidden | familiar relative bucketing |
| Rail ↔ main sync | click a doc anywhere → drills main pane; folder select → active collection | one behavior, no divergence |
| Grid defaults | grid default, list toggle, `libraryView` in `localStorage` | matches the other client-only view prefs |
| Projects / manual edit / drawer / title backfill / virtualization | **parked** (see Out of scope) | scope discipline; virtualization re-check if a collection >~500 tiles |

**Routing:** resolutions live in this spec (the design-lock). No ADR needed yet — the first browse-time
write path (manual folder/tag editing, Phase-B+) is the ADR trigger and is parked. Handoff recorded in
`.claude/SESSION.md`.

## Out of scope (deferred)

- ~~Manual metadata editing~~ **DONE** — the first browse-time write path shipped 2026-07-16 as
  title/authors/year override editing + reveal-in-explorer (`ADR-013`, `DocumentMeta` sidecar). Manual
  **folder/tag** editing remains deferred (same write path, layered on the same model).
- **Projects** as top-level folders (the `ChatConcept.folder_id` scoping) — a later increment on the same model.
- Title/author **metadata backfill** — DONE (deterministic `enrich_metadata`, 2026-07-16); tiles show real
  titles + an author line. In-app PDF source viewer; the detail-**drawer** navigation variant (reuses
  `SourcePanel`); virtualization of very large collections; user-selectable library columns.
