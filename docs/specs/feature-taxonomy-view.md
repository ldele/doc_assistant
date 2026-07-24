<!-- status: design-locked · created: 2026-07-24 · owner: Code · plan: docs/decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md (increment 2b) -->

# Feature spec — Taxonomy increment 2b: the Svelte taxonomy view

Build contract for **increment 2b** of [ADR-028](../decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md).
Increment 2a shipped the **serve + edit backend** (`knowledge/taxonomy_view.py` read model + the
`apps/api/routers/taxonomy.py` read/write routes + the `types.ts` wire mirror). This increment is the
**renderer**: a Svelte view that renders the field forest and **places** concepts/documents onto it —
driving the 2a endpoints. Frontend only, **$0, zero-LLM**. The backend already
exists and is verified; nothing in `src/` changes.
**Narrowed by the 2026-07-24 code-grounded review (ledger #6–7):** 2b ships *placement* — concept→field
attach/detach + document→field attach. **No field→field re-parenting UI** (stays API-only in 2b).

## Placement — resolved by the 2026-07-24 grill (see `## Open questions`)
The view is a **separate, dedicated modal overlay** (like Manage-keywords), opened from a **Library-space
entry point** plus the **existing graph deep-link** (ADR-019 D11). It is **fully decoupled** from the
parked "demote Graph from top-level nav" direction (memory `ui-direction-app-shell`) — a modal never
touches the top nav, so it neither forces nor blocks that decision. Scope is the **taxonomy modal only**:
no Library Collections-rail wiring and no per-folder-concepts (both stay parked).

## Why this slice
The graph shipped its read model (PR-G1) before any renderer; the same order applied here — 2a served
the model, 2b renders it. This is the surface where ADR-019 D11's "a dedicated taxonomy view owns all
tree edits" starts to be realised — placement edits now, structure edits (field re-parenting) when a
real curation need appears (ledger #6). It needs the running app + live-app verification, which is why it was split
from 2a (verifiable on this box only via `read_page` + `javascript_tool` — screenshots time out here,
memory `preview-env-screenshot-eval-flaky`).

## Scope
In: `apps/desktop/src/lib/api.ts` (taxonomy client fns), a new `lib/taxonomy.ts` (pure tree-shaping
helper) + `lib/taxonomy.test.ts`, a new `lib/LibraryTaxonomy.svelte` modal, and the `App.svelte` wiring
(state + open/close + load + mutation handlers + the Library trigger + graph deep-link).
Out: **field→field re-parenting UI** (ledger #6 — the structure edit stays API-only in 2b), the
Collections-rail population (later increment), per-folder-concepts (ADR-025 fork 5, parked),
auto-propose (increment 3, $0/Ollama, KI-4), coverage-based gap detectors (RG-015), the CC-BY
attribution UI (T4). No backend change — `src/` and `apps/api/` are untouched.

## Design decisions
- **Separate modal, reusing the existing overlay shell** (scrim + centred dialog + Esc), the same shell
  `GlobalSearch`/`LibraryKeywordFilter`/`LibraryManageKeywords` use — not a tab inside Manage-keywords
  (grill Q5: distinct object + verbs; Manage-keywords is already heavy at F1–F4 scale).
- **App owns the data; the component is a dumb renderer** (the repo pattern — App derives, the `.svelte`
  renders). `App.svelte` holds `taxonomyView`/`fieldDetail` state, calls the api client, and re-fetches
  after every mutation (write-then-refetch — the server is the authority on counts + acyclicity, never
  patch the local tree by hand; mirrors the folder/family mutation handlers).
- **Polyhierarchy-safe rendering.** A field can have multiple parents (ADR-028 D1). The tree is rendered
  from `roots` + each field's `child_ids`; a field reachable under two parents renders under **both**
  (the DAG is displayed as a tree with repeats), so no node is orphaned or silently deduped. Cycle-free
  is the backend's invariant (409 on write) — the renderer only needs a **visited-guard** so a shared
  descendant can't recurse forever. This guard is the pure, tested helper (`lib/taxonomy.ts`).
- **Honest zero-state (robustness contract; inform-don't-block).** Every rollup is currently 0 and all 26
  concepts are unassigned. The view shows the real forest with real zeros + an explicit "26 concepts not
  yet placed" affordance — never hidden, never faked. It must read as *unpopulated*, not *broken* (this
  is the exact failure mode the empty-Graph panel has; do not repeat it).
- **Read-only vocabulary boundary holds (ADR-017 A1 / ADR-019 D11).** The modal edits *placement*
  (concept→field `in_field` edges + document→field), never concept create/rename/delete — and, in 2b,
  never field→field structure (next bullet).
- **Placement-only editing (ledger #6).** No field re-parenting control in 2b: the ANZSRC trunk is
  seeded structure that rarely moves, and field→field edges are the *only* edit that can trip the 409
  cycle guard — a concept attach cannot form a cycle because concepts are never edge targets. The
  guard stays exercised at the API level (DoD 5); a re-parent control is ledger #6's reopener.
- **Concept picker source = the graph vocabulary (ledger #7).** 2a deliberately serves no concept
  list (`TaxonomyView` carries only counts; `FieldDetail` only already-attached members), so the
  attach picker is fed from `getConceptGraph().nodes` mapped to `{ id, label }` — today all 26
  concepts. **Accepted limitation:** that is the `graph_include` opt-in vocabulary, so a
  curation-demoted concept (E0.1, `graph_include=false`) is not offered for placement — acceptable
  while demoted ≈ noise; reopener in ledger #7.
- **Wire types already exist** (2a shipped `TaxonomyField`/`TaxonomyView`/`FieldDetail`/
  `HierarchyEdgeRequest` in `types.ts`); 2b adds no new wire type — only the `api.ts` fns that call them.

## Items

### T1 — `lib/api.ts` taxonomy client fns
- `getTaxonomy(): Promise<TaxonomyView>` → `GET /api/taxonomy`.
- `getFieldDetail(fieldId): Promise<FieldDetail>` → `GET /api/taxonomy/fields/{id}` (404 → throw).
- `addHierarchyEdge(body: HierarchyEdgeRequest): Promise<void>` → `POST /api/taxonomy/hierarchy`.
- `removeHierarchyEdge(body: HierarchyEdgeRequest): Promise<{ removed: number }>` → `DELETE …/hierarchy`.
- `attachDocumentField(docId, fieldId): Promise<void>` → `POST …/documents/{doc}/fields/{field}`.
- Mirror the existing client style — **self-contained fns** (`fetch` + `if (!r.ok) throw` + typed
  cast; truth-fix: `api.ts` has no shared `request` helper — don't introduce one here).
  `removeHierarchyEdge` is the client's first DELETE with a JSON body (fine with `fetch`; set
  `Content-Type` like the POSTs do). No new wire type.

### T2 — `lib/taxonomy.ts` (pure, tested) + `lib/taxonomy.test.ts`
- `buildForest(view: TaxonomyView) -> TaxonomyRow[]` — flattens the DAG into ordered display rows
  `{ field, depth, hasChildren }` from `roots` + `child_ids`, guarded by the **ancestor set of the
  current path** — NOT a global visited set, which would stop a poly-parented subtree expanding under
  its second parent and contradict the render-under-both rule. In a true DAG the path guard never
  fires (a diamond terminates guard-free); it exists so a corrupt DB containing a cycle renders
  truncated instead of hanging (the backend's 409 is the real invariant).
  Deterministic order (roots in `view.roots` order, children in `child_ids` order).
- Self-contained — **no runtime value import** from sibling modules (node:test strips type-only imports
  but can't resolve extensionless value ones; the `search.ts`/`library.ts` rule, apps/desktop/CLAUDE.md).
- Tests (node:test): forest order + depth; a poly-parented field appears under both parents **with
  its subtree expanded both times** (kills a global-visited implementation); the path guard terminates
  on a hand-built **cycle** (not a diamond — a diamond terminates without any guard); empty
  `fields` → `[]`.

### T3 — `lib/LibraryTaxonomy.svelte` (the modal)
- Props in: `view: TaxonomyView | null`, `fieldDetail: FieldDetail | null`, `loading`, `error`,
  `documents: LibraryDocument[]` (for the attach-document picker + id→title),
  `concepts: FieldMember[]` (the attachable vocabulary, ledger #7), `focusConceptId: string | null`
  (graph deep-link: preselect this concept for placement), plus callbacks: `onSelectField(id)`,
  `onAddEdge(body)`, `onRemoveEdge(body)`, `onAttachDocument(docId, fieldId)`, `onClose`.
  Dumb renderer — no fetch, no business logic.
- Left: the field forest (indented rows from `buildForest`), each row showing label + rollup counts
  (`n_concepts_rollup`/`n_documents_rollup`) and a direct-count hint. Selecting a field loads its detail.
- Right: the selected field's detail — its directly-attached concepts + documents, with a remove control
  per attachment (`onRemoveEdge` for a concept `in_field` edge; there is no doc-detach route in 2a, so
  document rows are read-only in 2b — note this, don't invent an endpoint). Below the lists, the two
  attach controls: **"Attach concept…"** over `concepts` minus the already-attached
  (→ `onAddEdge({ source_id: conceptId, target_id: fieldId, type: 'in_field' })`) and
  **"Attach document…"** over `documents` (→ `onAttachDocument`). When `focusConceptId` is set, the
  header names it ("Placing: <label>") and the attach-concept control arrives preselected — placing
  it is one field click + one confirm.
- Header: corpus totals + the **"N concepts not yet placed"** zero-state affordance.
- Reuse the scrim + centred dialog + Esc shell; theme-token styling; light + dark; 375px no overflow.

### T4 — `App.svelte` wiring + entry points
- State: `taxonomyOpen`, `taxonomyView`, `taxonomyFieldDetail`, `taxonomyConcepts`,
  `taxonomyFocusConceptId`, `taxonomyLoading`, `taxonomyError`.
- `openTaxonomy(focusConceptId?)` — set open, lazy-load `getTaxonomy()` (+ `refreshDocuments()` if not
  loaded, for the doc picker; + the concept vocabulary for the attach picker — reuse the already-loaded
  graph state when present, else `getConceptGraph()`, nodes mapped to `{ id, label }`, ledger #7),
  inform-don't-block on failure. `closeTaxonomy()` clears the focus concept.
- Mutation handlers (write-then-refetch `getTaxonomy()`, and re-load the open field detail):
  `taxonomyAddEdge`/`taxonomyRemoveEdge`/`taxonomyAttachDocument`. A 409/400/404 surfaces in
  `taxonomyError`, never blocks.
- **Library entry point** — a trigger in the Library space beside the existing curation entries
  (Manage-keywords / Manage-folders). **Graph deep-link** — add a **sibling** to `manageConcept`
  (`App.svelte:242`, which today ignores its args and just opens Manage-keywords) that calls
  `openTaxonomy(conceptId)`, so a graph node opens the taxonomy modal with that concept preselected
  for placement (D11) while Manage-keywords stays reachable. The modal opens
  from any mode (it's a global overlay like Settings/Search), so demoting Graph later can't strand it.
- `<LibraryTaxonomy … />` rendered under `{#if taxonomyOpen}` beside the other overlays; Esc/`onClose`.

## DoD / guard tests (each fails against today's code)
1. `lib/taxonomy.test.ts` — `buildForest` returns rows in root-then-child order with correct `depth`;
   a poly-parented field appears under both parents with its subtree expanded both times; the
   ancestor-path guard terminates on a cycle; empty → `[]`.
2. `svelte-check` 0 errors/0 warnings with the new modal + wiring.
3. `npm test` green (existing + the new taxonomy tests).
4. Live ($0, no LLM): open the modal from the Library entry → the 236-field forest renders, 23 roots,
   header shows **26 concepts / 76 documents / 26 not yet placed**, all rollups 0 (honest zero-state).
5. Live: attach a concept to a group via the attach picker (`POST /hierarchy` `in_field`) → that
   group's direct count ticks to 1 and its **division's rollup** ticks to 1 (crosses the
   group→division edge); remove it → back to 0. The 409 cycle guard is exercised at the **API level**
   (`javascript_tool` → the api client: a field→field edge that closes a loop; assert 409 + message +
   no partial write via a re-`GET`) — there is deliberately no UI path that can form a cycle
   (ledger #6; concepts are never edge targets).
6. Live: the graph deep-link opens the taxonomy modal with that concept preselected for placement
   (D11); the modal opens/closes in Chat, Library,
   and Graph modes (proves it's mode-independent — decoupled from the nav).
7. Dark theme + 375px: 0 console errors, 0 horizontal overflow.

## Gate
`svelte-check` 0/0 · `npm test` green · `docs_check`/`integrity_check` 0/0 · live verify per the DoD via
`read_page` + `javascript_tool` (this box; screenshots time out). No backend gate — `src/` untouched, so
`ruff`/`mypy`/`bandit`/`pytest` are unaffected but run once to confirm no accidental drift.

## Open questions
_Scoped-grill ledger (2026-07-24) + the same-day code-grounded review (#6–7). Every row resolved — no
`open` status (cpc docs_check rule 15)._

| # | Branch | Status | Resolution | Reopens if |
|---|--------|--------|-----------|-----------|
| 1 | Surface type | resolved | **Modal overlay**, like Manage-keywords (honors ADR-019 D11; decouples from parked nav forks) | Tree+coverage browse proves too cramped in a modal → graduate to a Library-integrated view |
| 2 | Entry points | resolved | **Library-space trigger** (beside Manage-keywords/Manage-folders) **+ the D11 graph deep-link** | Graph removed *and* the Library entry proves undiscoverable |
| 3 | Graph coupling | resolved | **Fully decoupled** — 2b ships; demote-Graph stays parked | User decides to restructure the nav before/with 2b |
| 4 | Scope boundary | resolved | **Taxonomy modal only** — no Collections-rail, no per-folder-concepts | Collections-rail wiring is pulled forward |
| 5 | Modal identity | resolved | **Separate, dedicated modal** (not a tab in Manage-keywords) | Modal proliferation becomes a real problem → future unified "Manage" tabbed shell |
| 6 | Hierarchy-edit scope | resolved | **Placement only** — concept attach/detach + doc attach; no field→field re-parenting UI (2026-07-24 review) | A curator actually needs to move a group/division → add a re-parent control (giving the 409 guard its UI test) |
| 7 | Concept-picker source | resolved | **Graph vocabulary** — `getConceptGraph().nodes` → `{ id, label }`; 2a serves no concept list by design | Demoted (`graph_include=false`) concepts need placement too → add a vocabulary-list endpoint |

## Out of scope → later increments
- **Collections-rail population** from the field taxonomy (ADR-019's stated payoff) — a dedicated
  follow-up once the modal exists and concepts are actually attached.
- **Increment 3 — auto-propose** `in_field` parents for the 26 unassigned concepts ($0/Ollama, KI-4,
  **RTX box** — not this box).
- **Demote-Graph / per-folder-concepts / empty-Graph-panel** — the parked nav fork (`ui-direction-app-shell`);
  grill separately before building.
- **Field→field re-parenting UI** (ledger #6) — the structure edit stays API-only; adding the control
  later also gives the 409 cycle guard a UI surface.
- Document-detach route (2a shipped attach-only) — add server-side if 2b's read-only doc rows prove
  limiting.
