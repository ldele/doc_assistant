# Spec — Keyword filtering as a two-pane overlay

**Status:** ✅ **SHIPPED — commit `ce8b112` (2026-07-16).** Grilled 2026-07-16 (`grill-me`; ledger at foot).
Redesigned the faceted-keyword-filtering v1 (the inline `LibraryFacetBar`) into an on-demand overlay; v1 was
uncommitted, so both folded into `ce8b112`. The pure filter logic (`facetFilter` / `keywordFacets` in
`lib/library.ts`) is **unchanged** — a presentation swap, not a logic change. Frontend-only, `$0`/offline,
preview-harness-verified (search, live toggle 76→22, 26 greyed, Esc-persist, Clear, dark, mobile panes stack,
0 console errors).

**Owner:** Claude Code. One PR. Since v1 is uncommitted, **fold** v1 + this redesign into a single commit
(no throwaway inline-bar commit).

---

## The decision (user, 2026-07-16)

The always-visible inline keyword bar does not scale — a corpus can hold hundreds of keywords. Move keyword
picking into an **on-demand two-pane overlay** (the modal idiom already used by `LibraryMetaEditor`), and keep
only the **selected** keywords visible inline as a compact strip on the main view.

Grounded in **Zotero's tag selector** (researched in-session): AND semantics, a filter/search box, a
self-pruning tag list, and colored/pinned "favorite" tags. We adopt Zotero's *mechanics* but not its
*container* — Zotero docks a permanent tag panel because its left rail is a wide dedicated column; Provenote's
rail already carries the Chat/Library switch + the nav tree (All / Collections / Types / Added), so a docked
panel would crowd Collections (the user's own concern). An overlay sidesteps that and scales better.

## Grounding (read from the code + live corpus, 2026-07-16, not assumed)

- **The filter logic already exists and is verified.** `lib/library.ts::facetFilter` (AND) + `keywordFacets`
  (per-chip live count, `available`/greyed, selected-first ordering) were built + verified live on the real
  76-doc corpus (76→22 narrow, dynamic recount, 26 chips greyed, AND intersection, Clear). This spec **reuses
  them unchanged**.
- **`LibraryDocument.keywords` already ships client-side** — the overlay's doc-preview pane needs no backend
  call; it filters the in-memory list, same as the grid.
- **A modal idiom exists to reuse.** `LibraryMetaEditor.svelte` is a `role="dialog"` + `aria-modal="true"`
  overlay with a backdrop, `<svelte:window onkeydown>` Esc-to-close, and a `use:autofocus` action. **No
  command-palette / global keyboard-shortcut infra exists** — a Cmd-K trigger would be net-new (deferred).
- **The live keyword pool proves the scale premise and the families overlap:** 60 keywords on this corpus;
  `llms`(13)/`llm`(12) and `connectome`(18)/`connectomics`(12) appear as separate chips — the tag-families
  case, which will later collapse into single overlay entries.

---

## The two-pane contract

A centered modal overlay (reusing the `LibraryMetaEditor` shell: backdrop, `role="dialog"`, `aria-modal`,
Esc-to-close, focus the search box on open). On narrow widths the two panes **stack vertically** (keyword
picker on top, results below).

**Left pane — keyword picker:**
- A **search box** at the top filters the keyword list by substring (autofocused on open).
- The keyword list: **selected-first, then by doc count desc** (ties alphabetical) — the `keywordFacets`
  order. Most-used naturally lands on top (the v1 "favorites" stand-in; user-pinned favorites are deferred,
  see Parked).
- Each row: the keyword, its **live co-occurrence count**, **selected** (highlighted + removable) or
  **greyed + disabled** when adding it would empty the result (the `available` flag), exactly as v1.

**Right pane — live document preview:**
- A header count: **"N documents"**.
- A scrollable list of the matching documents (`title · author · year`; filename fallback), updating **live**
  as keywords toggle. Read-only preview (clicking a doc is out of scope for v1 — the grid behind is where you
  open one).

**Apply:** **live commit, no Apply button.** Toggling a keyword instantly updates the right-pane preview *and*
the real selection (grid + inline strip). Closing the overlay only dismisses it. Removal is via toggling off,
the inline strip's ✕, or a "Clear" in the overlay.

**Inline strip (main view, above the grid):** replaces the v1 facet bar. When no keyword is selected: a single
**"Filter by keyword"** button (opens the overlay). When keywords are selected: the selected-keyword chips
(each with ✕) + the button (now an edit/add affordance) + a "Clear" — this *is* the "selected keywords on
top" idea. It is the overlay's trigger and the always-visible active-filter state.

---

## Decisions (grill ledger, 2026-07-16)

| # | Decision | Reason / reopens if |
|---|---|---|
| A | **Overlay + inline selected-keywords strip** (Zotero mechanics, not its container) | Rail already carries Chat/Library + nav tree; overlay avoids crowding Collections. Reopens if the rail is redesigned to free space |
| C | **Two-pane** overlay: searchable keyword list (left) + live matching-docs preview `title · author · year` + "N documents" (right) | Delivers "preview of selected documents" self-contained |
| D | **Live commit, no Apply button** | The right-pane preview is the feedback; matches the shipped facet behavior + GitHub's live labels |
| E | **Search box** in the overlay (filter the keyword list) | Zotero/GitHub idiom; mechanical |
| J | **Most-used-on-top now**; user-pinned favorites **deferred to tag-families** | A pinned keyword ≈ a promoted `Concept`; building a separate favorite flag now duplicates that curation |
| G | List order **selected-first, then count desc** (ties alpha) | Falls out of J; already what `keywordFacets` returns |
| F | **Keyword-only overlay ("Keywords")**; families slot into the same list | v1 is keywords; a general filter hub is speculative |
| B | Trigger = the **inline-strip button**; no Cmd-K | No palette infra; tile chips stay display-only (nested-button constraint) |
| H | **No keyword descriptions** in v1 | Raw `Keyword` has no description field |

## Parked (reopen triggers named)

- **General "Filters" hub** (this overlay swallowing tags / article-type / year / journal too) — reopen when
  the **extended-metadata** feature lands those as real filter dimensions. Until then, keyword-only.
- **User-pinned favorites** (Zotero colored tags, pinned regardless of count, number shortcuts) — rides the
  **tag-families** work as *promote-to-`Concept`* (a promoted concept = a de facto favorite). A lightweight
  `localStorage` pin set was offered and declined (would be throwaway once Concepts subsume it).
- **Cmd-K / global shortcut** — deferred until (if) a command-palette surface exists.

## Definition of Done

- New overlay component (e.g. `LibraryKeywordFilter.svelte`) reusing the `LibraryMetaEditor` modal shell;
  two panes (stack on mobile); reuses `facetFilter`/`keywordFacets` unchanged.
- The inline `LibraryFacetBar` is replaced by the compact selected-strip + trigger (or the component is
  reshaped to it).
- `svelte-check` 0 errors; both themes (reuse existing CSS vars, no new palette); mobile no horizontal
  overflow (panes stack); overlay a11y (dialog role, aria-modal, Esc, focus the search box, `aria-pressed`
  on chips, disabled greyed rows).
- Preview-harness verified `$0`/offline on the real corpus: open overlay → search filters the list → toggle
  → right-pane preview + "N documents" update live → grey-out + AND narrowing match v1 numbers → Clear resets
  → inline strip reflects the selection → 0 console errors.
- DEVLOG entry; ui-checklist §3 row updated; this spec's status flipped to SHIPPED with the commit SHA.

---

## Grill ledger

Grilled 2026-07-16 (`grill-me`), 9 branches, all resolved, 3 parked (above). Interview walked container →
layout → apply → search → favorites → order → scope → trigger → descriptions in dependency order. Zotero's
tag selector researched live ([Zotero — Collections and Tags](https://www.zotero.org/support/collections_and_tags),
[Working with Tags — Mastering Zotero](https://pressbooks.library.yorku.ca/masteringzotero/chapter/working-with-tags/))
and used to reconcile the user's "selected-on-top" and "favorites/search on the side" instincts into the
locked design. Full ledger table above.
