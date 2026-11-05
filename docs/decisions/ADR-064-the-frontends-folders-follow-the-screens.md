<!-- status: active · updated: 2026-11-05 (accepted; pass 1 built: every component moved; pass 2 open) · class: append-only -->

# ADR-064 — The frontend's folders follow the screens, and the domain spine stays underneath

- **Status:** accepted (user, 2026-11-05: the seven choices as recommended). **Pass 1 built** the
  same evening, after `c2b2fef` left `apps/desktop/` clear: all 54 components moved, 28 renamed,
  `src/layout.test.ts` added. Pass 2 (the splits, choice 5 and 6) is open and starts after pass 1 is
  committed, so no file is renamed and split in one commit. Proposed by Claude Code the same day.
  The request is the
  user's, 2026-11-05, during the UI batch of ROADMAP rows 110–112: could `apps/desktop/` be
  organised to mirror the interface, *"something like one file per screen"*, so that a person who
  sees a screen or a dialog finds its code in one obvious place.
- **Date:** 2026-11-05
- **Deciders:** user + Claude Code
- **Revisits** the desktop half of the RV1 reviewability pass (2026-07-26, `docs/architecture.md`
  § `apps/` — the domain spine): component files leave `lib/<domain>/` for screen folders. The spine
  itself stays: rune modules, pure `.ts` logic and `core/api|types` keep their domain folders.
- **In one sentence:** every screen the app shows gets one folder named as the app names it, holding
  one entry file, the screen's parts and the dialogs it opens; the frame around the screens gets its
  own folder; and `lib/` keeps the domain spine as it is.

## To decide — seven choices, each with a recommendation

1. **Where the screen folders go.** *Recommended:* three folders under `src/`, each described in
   one line:
   - `screens/`: one folder per screen, components only;
   - `shell/`: the frame around every screen (top bar, sidebar frame, status bar) and the windows
     the frame opens (search, about, shortcuts, indexing), plus the two components every screen
     uses, `Icon` and `ConfirmDialog`;
   - `lib/`: the domain spine, unchanged (rune modules, pure tested `.ts`, `core/`).

   The alternative is screen folders inside the domain folders (`lib/library/document-page/`). It is
   rejected because `lib/library/` would then be a domain and a screen at once, and *which* is the
   question this ADR exists to answer.
2. **What counts as a screen, and where a dialog lives.** *Recommended:* a **screen** is a place you
   go: a tab (Chat, Library, Graph), the document page, or a view that covers what is under it and
   has parts of its own (Settings, Manage keywords, Taxonomy). A **dialog** asks one thing and
   closes. It lives with the screen whose thing it acts on, even when another screen also has a
   door to it, and the table below lists every door. Re-run ingestion acts on documents, so it lives
   with the document page although the Library's select bar and Settings also open it.
3. **"One file per screen", literally?** *Recommended:* no. One **entry file** per screen, named
   for the folder (`screens/graph/Graph.svelte`), with the parts beside it. Taken literally the
   request makes files larger, not easier to find. Concatenating each screen's components today
   gives the document page **2,260** lines, the Graph tab **2,370**, the Chat tab **2,097** before
   its sidebar list, and Settings **3,515**; cpc CONVENTIONS §16's soft cap for a component file is
   about 300.
4. **Names.** *Recommended:* a folder or file is named for what the app shows, and is renamed during
   the move, so each file moves once. A dialog is named for its heading (`ReingestDialog` becomes
   `RerunIngestion`, its heading being *Re-run ingestion*). The folder's own word is dropped from
   the file names inside it (`screens/library/ManageFolders.svelte`, not `LibraryManageFolders`).
   A name that already says what the app shows is kept. Two follow from this:
   - `settings/Sources.svelte` becomes `ManageFiles.svelte` (its heading is *Manage files*). That
     retires the first naming trap in `docs/architecture.md` (*`sources` is ingestion, not
     citations*), because no component is called *Sources* any more.
   - The Graph tab's right-hand side has two names today: *concept panel* (ADR-053) and *concept
     view* (ROADMAP 111). U-001 says one. *Recommended:* **concept panel**, the older and
     ADR-recorded name; a new GLOSSARY entry U-004 lists every screen's name.
5. **What leaves `App.svelte`.** *Recommended:* App keeps the navigation history, the readiness
   gate, mode switching, global keys, overlay mounting, and the **glue**: a function that reads or
   writes two domains (the keyword-family remap of the Library's facets, a chat turn that also
   refreshes the conversation list). RV1 left coupled domains in the shell so coupling stays
   visible, and this keeps that rule. What leaves is state and actions that belong to **one**
   domain: documents, folders, the selection, keyword families, each dialog's open flag and its
   arguments. They go to that domain's rune module, the pattern `taxonomy.svelte.ts`
   (`taxonomy.open`) and `shell.svelte.ts` (`shell.showSettings`) already follow. A dialog's door
   then calls the module directly instead of handing a callback up through two components. Read
   from App's function list, about 830 of its 1,206 script lines are one-domain. App would land
   near **600–800** lines, still past the cap, and the ADR records that as a deliberate exception:
   App mounts every overlay.
6. **Which files split, and where.** *Recommended:* the nine files of 690 lines or more (table below), cut
   along the parts the screen visibly has: its headings, sections and rails. A cut is made only
   where the part's styles go with it. Svelte scopes styles per component, so two parts that share
   most of their CSS would each carry a copy, and they stay one part. The seventeen components
   between 300 and 690 lines are left as they are, with the reason recorded: most of their bulk is
   scoped CSS, and the part that repeats across files is per-component button classes (16 files,
   removed by ROADMAP 110) and a per-dialog backdrop and frame (20 files define their own
   `.scrim`, 14 of them a `.modal`).
7. **Pure `.ts` logic used by one screen** (`forceLayout.ts`, `sourceviewer.ts`, `labels.ts`).
   *Recommended:* it stays in `lib/`. Then `screens/` and `shell/` hold only `.svelte` files, every
   module `npm test` runs stays in the spine, and the explicit-`.ts`-extension rule for tested
   imports never meets a screen folder.

## In practice — the document page and its Re-run dialog

**Today.** You open a document from the Library and press *Re-run…*. A dialog headed *Re-run
ingestion* appears. To find the code behind what you just saw:

1. The page is `lib/library/LibraryBrowser.svelte` (748 lines). Its name says *browser*, and it
   shares a flat folder of 47 files with the Library grid, the Add documents sheets and the Manage
   keywords view.
2. *Re-run…* (`LibraryBrowser.svelte:256`) calls a prop, `onReingest`. `LibraryPane.svelte:280`
   passes it through as `onReingestDoc`. `App.svelte:1355` turns it into `openReingest(…)`, which
   sets three `$state` variables at `App.svelte:1059`. `App.svelte:1551` mounts the dialog from
   `lib/library/ReingestDialog.svelte`, and `afterReingest` at `App.svelte:1073` refreshes the list
   when it finishes. That is four files and five hops, two of them in a 1,769-line file.
3. The same dialog has three more doors: *Re-run text now…* in the page's letter-spacing block
   (`DocSpacing.svelte:108`), *Re-run…* on the Library's select bar (`LibraryPane.svelte:266`), and
   *Re-run text now…* in Settings › Documents (`SpacingIgnoreList.svelte:170`, through
   `Settings.svelte:659` to `App.svelte:1447`). Nothing in the tree says these four are one dialog.

**After both passes.**

```
src/screens/document-page/
  DocumentPage.svelte       the page (was lib/library/LibraryBrowser.svelte)
  DocumentHeader.svelte     title, metadata, the index, Source, Re-run…   (split out, pass 2)
  Chunks.svelte             the Chunks block                              (split out, pass 2)
  Connections.svelte  Figures.svelte  FigureViewer.svelte  References.svelte
  LetterSpacing.svelte      "Letter-spacing on this document" (was DocSpacing, U-002)
  EditSpacing.svelte        "Edit spacing" (was SpacingEditor, U-002)
  SourceViewer.svelte
  RerunIngestion.svelte     "Re-run ingestion" (was lib/library/ReingestDialog.svelte)
src/lib/library/
  reingest.ts               unchanged, tested
  reingest.svelte.ts        new: the dialog's targets, label and parts, and openRerun()
```

Each of the four doors calls `openRerun(…)` from `lib/library/reingest.svelte.ts`. App keeps one
line, `{#if rerun.targets}<RerunIngestion />{/if}`, because it mounts every overlay. A person who
sees *Re-run ingestion* finds `RerunIngestion.svelte` by its heading, in the folder of the page it
was built for.

**A second case, in one line.** The *Taxonomy* view's code is `lib/library/LibraryTaxonomy.svelte`,
but both of its doors are on the Graph tab: *Taxonomy* in the concept index's footer
(`Sidebar.svelte:623`) and *Place* on a concept (`ConceptGraph.svelte:449`). After the move it is
`screens/taxonomy/Taxonomy.svelte`, a screen of its own, because it covers what is under it and has
two parts (the field forest and a field's detail).

## Every screen and dialog, and its folder

Doors are the controls that open a screen or a dialog. File names are after pass 1; a part marked
*(pass 2)* is cut out of a larger file in the second pass.

| What the app shows | Its doors | Today (`src/lib/…`) | After (`src/…`) |
|---|---|---|---|
| **Frame** | | | `shell/` |
| Top bar | always | `shell/Topbar.svelte` | `shell/Topbar.svelte` |
| Sidebar (collapse, resize, the tab's rail) | always | `shell/Sidebar.svelte` | `shell/Sidebar.svelte`, frame only after pass 2 |
| Status bar, its indexing chip | always | `shell/StatusBar.svelte`, `IngestChip.svelte` | `shell/` (same names) |
| *Search chats and documents* | top bar, Ctrl/⌘ K | `shell/GlobalSearch.svelte` | `shell/GlobalSearch.svelte` |
| *Keyboard shortcuts* | app menu | `shell/ShortcutsDialog.svelte` | `shell/KeyboardShortcuts.svelte` |
| *About Provenote* | app menu | `shell/AboutDialog.svelte` | `shell/About.svelte` |
| Indexing (a run's progress) | the status bar's chip, *View indexing* in Settings | `shell/IndexingScreen.svelte` | `shell/Indexing.svelte` |
| The library switch failed | start-up | `shell/LibrarySwitchFailed.svelte` | `shell/LibrarySwitchFailed.svelte` |
| A confirm (delete chats, stop indexing, switch library) | several | `shell/ConfirmDialog.svelte` | `shell/ConfirmDialog.svelte` |
| Icons | every screen | `shell/Icon.svelte` | `shell/Icon.svelte` |
| **Chat tab** | top bar | | `screens/chat/` |
| the tab | *Chat* | `chat/ChatPane.svelte` | `Chat.svelte` |
| Chat history (the sidebar rail) | sidebar, Chat | inside `shell/Sidebar.svelte` | `ChatHistory.svelte` (pass 2) |
| an answer and its parts | | `chat/Turn`, `ReadonlyTurn`, `Markdown`, `Provenance`, `ClaimReview`, `SourceEvaluation` | same names |
| the comparison | *Test override* | `chat/CompareCard.svelte` | `CompareCard.svelte` |
| a citation's source | an answer's [n] | `chat/SourcePanel`, `SourceCard`, `ChunkContextView` | same names |
| **Library tab** | top bar | | `screens/library/` |
| the tab | *Library* | `library/LibraryPane.svelte` | `Library.svelte` |
| the grid or list of documents | | `library/LibraryGrid.svelte` | `DocumentGrid.svelte` |
| the keyword filter strip | | `library/LibraryFilterStrip.svelte` | `FilterStrip.svelte` |
| the library summary (ROADMAP 112) | the layout toggle's *Summary* | `library/LibrarySummary.svelte` | `Summary.svelte` |
| Library navigation (the sidebar rail) | sidebar, Library | inside `shell/Sidebar.svelte` | `LibraryRail.svelte` (pass 2) |
| *Filter by keyword* | the filter strip | `library/LibraryKeywordFilter.svelte` | `FilterByKeyword.svelte` |
| *Manage folders* | the rail's *Manage…*, a tile's *Add to folder*, the select bar's *Add to folder…* | `library/LibraryManageFolders.svelte` | `ManageFolders.svelte` |
| *Edit metadata* | a tile's ⋯ menu | `library/LibraryMetaEditor.svelte` | `EditMetadata.svelte` |
| *Delete this document?* | a tile's ⋯ menu | `library/LibraryDeleteConfirm.svelte` | `DeleteDocument.svelte` |
| *Add documents*, first step | app menu, the empty Library, a drop | `library/AddDocumentsChooser.svelte` | `AddDocumentsChooser.svelte` |
| *Add documents*, the review sheet | the staged-files bar's *Review* | `library/AddDocuments.svelte` | `AddDocumentsReview.svelte` |
| **Document page** | a document in the Library, Search, a graph concept | | `screens/document-page/` |
| the page | | `library/LibraryBrowser.svelte` | `DocumentPage.svelte` |
| its header, its Chunks block | | inside `LibraryBrowser.svelte` | `DocumentHeader.svelte`, `Chunks.svelte` (pass 2) |
| Connections, Figures, References | the page's index | `library/DocConnections`, `DocFigures`, `DocReferences` | `Connections`, `Figures`, `References` |
| the figure viewer | a figure | `library/FigureViewer.svelte` | `FigureViewer.svelte` |
| *Letter-spacing on this document* | | `library/DocSpacing.svelte` | `LetterSpacing.svelte` |
| *Edit spacing* | a run's *Edit spacing* | `library/SpacingEditor.svelte` | `EditSpacing.svelte` |
| the source viewer | *Source* | `library/SourceViewer.svelte` | `SourceViewer.svelte` |
| *Re-run ingestion* | *Re-run…*, *Re-run text now…* (page and Settings), the Library's select bar | `library/ReingestDialog.svelte` | `RerunIngestion.svelte` |
| **Graph tab** | top bar | | `screens/graph/` |
| the tab, its empty states, the graph bar | *Graph* | `graph/ConceptGraph.svelte` | `Graph.svelte` |
| the concept panel | a concept | inside `ConceptGraph.svelte` | `ConceptPanel.svelte` (pass 2) |
| *Definition* | the concept panel | `graph/ConceptDefinition.svelte` | `Definition.svelte` |
| the concept index (the sidebar rail) | sidebar, Graph | `graph/GraphIndex.svelte` | `ConceptIndex.svelte` |
| the gap list | the rail's *Gaps* | `graph/GapList.svelte` | `GapList.svelte` |
| ***Manage keywords*** | *Manage keywords…* in *Filter by keyword*; on the Graph tab, *Manage keywords* (the graph bar and the empty state) and a concept's *Edit* | `library/LibraryManageKeywords.svelte` | `screens/manage-keywords/ManageKeywords.svelte`, parts (pass 2) |
| ***Taxonomy*** | the concept index's footer, *Place* on a concept | `library/LibraryTaxonomy.svelte` | `screens/taxonomy/Taxonomy.svelte`, parts (pass 2) |
| ***Settings*** | top bar, app menu, *Finish setup…* on the Chat tab, the letter-spacing block's link | | `screens/settings/` |
| the window and its rail | | `settings/Settings.svelte` | `Settings.svelte` |
| its five sections | the rail | inside `Settings.svelte`, plus `GettingStarted.svelte` | `GettingStarted`, `Documents`, `ProviderAndModel`, `Retrieval`, `General` (pass 2) |
| the provider list and its forms | *Provider & model* | `settings/ProviderSetup.svelte` | `ProviderSetup.svelte`, parts (pass 2) |
| *Manage files* | *Documents* | `settings/Sources.svelte` | `ManageFiles.svelte` |
| *Letter-spacing ignore list* | *Documents* | `settings/SpacingIgnoreList.svelte` | `SpacingIgnoreList.svelte` (U-002) |
| *Re-index this folder?* | *Re-index* in *Documents* | `settings/ReindexConfirm.svelte` | `ReindexFolder.svelte` |
| *Open this library?* | *Open another library…* in *Documents* | `settings/OpenLibraryDialog.svelte` | `OpenLibrary.svelte` |

All 54 `.svelte` files under `lib/` appear above, the library summary's included (added by
ROADMAP 112 the same day, uncommitted when this was written). `lib/` keeps its 124 `.ts` files
where they are, rune modules and tests included.

## The nine files that split (choice 6)

Lines are script / markup / style, measured 2026-11-05.

| File | Lines | Cut along what the screen shows |
|---|---|---|
| `App.svelte` | 1,769 (1,206 / 408 / 155) | choice 5: one-domain state leaves for rune modules; each tab's rail is handed to the sidebar as a snippet, the way the concept index already is |
| `Settings.svelte` | 1,315 (449 / 510 / 356) | its five rail sections: *Getting started*, *Documents*, *Provider & model*, *Retrieval*, *General* |
| `LibraryManageKeywords.svelte` | 1,230 (296 / 397 / 537) | *Detect proposals* · *New family* · a family row (rename, delete, its forms) · *Concepts* · *Terms* |
| `Sidebar.svelte` | 1,150 (311 / 343 / 496) | the frame stays in `shell/`; Chat history and Library navigation go to their tabs |
| `ProviderSetup.svelte` | 978 (297 / 343 / 338) | the provider list · *OpenAI-compatible server* form · a provider's form |
| `ConceptGraph.svelte` | 959 (293 / 316 / 350) | the graph bar and empty states stay; the concept panel goes |
| `LibraryTaxonomy.svelte` | 840 (150 / 255 / 435) | the field forest · a field's detail |
| `LibraryBrowser.svelte` | 748 (227 / 200 / 321) | the header · the Chunks block |
| `LibraryPane.svelte` | 744 (115 / 309 / 320), after ROADMAP 112 | the header bar (back, location, sort, layout) · the select bar · the empty Library |

## Context

RV1 (2026-07-26) gave both shells one axis, the domain, with the same word on both sides of the
wire: `apps/api/routers/library/` → `lib/core/api/library.ts` → `lib/library/`. That made a
feature reviewable end to end, and it is why the spine stays. What it did not give is a way in
from the screen. The domain folders hold components, rune modules and tested logic side by side
(`lib/library/` has 47 files, 20 of them components), and a component's name is its history, not
what the app shows: the document page is a *browser*, the Taxonomy view sits under *library* with
both doors on the Graph tab, and *Sources* is the Manage files list.

`App.svelte` is the other half. RV1 took it from 2,725 to 1,245 lines with seven rune modules and
four pane components. It has been changed in 28 commits since and is 1,769 lines again (+42%),
mostly Library state that grew with folders, select mode and keyword families. Each dialog added
one more flag, one more callback prop and one more mount to it.

## Options

- **A. Screen folders for components, the spine kept in `lib/`** (recommended). A screen is found
  by its name, a feature by its domain word. Costs one move of every component and one pass of
  import edits.
- **B. Keep `lib/<domain>/`, rename files to what the app shows, split the large ones in place.**
  Cheaper and keeps the history in place. The Taxonomy view would still sit under *library*, the
  document page among 47 files, and App would keep every dialog's state.
- **C. Screen folders nested in domain folders.** Mixes the two axes in one path, so a folder's
  name stops saying which axis it is on.
- **D. One file per screen, literally.** Measured above: files of 2,000 to 3,500 lines.

## Decision

Option A, with the seven choices as recommended, accepted by the user on 2026-11-05.

**Order of work**, after acceptance:

1. **Wait.** No file moves while another session has uncommitted work under `apps/desktop/` or is
   editing it. On 2026-11-05 the UI pass has staged `ConceptGraph.svelte` (ROADMAP 111), ROADMAP 110
   will touch 16 components, and the library summary (ROADMAP 112) is being built. Check
   `git status` and the linked sessions before starting.
2. **Pass 1, moves only.** `git mv` every component to its row above, rename as the table says,
   and fix imports, asset paths and the paths pinned in tests and living docs. No behaviour change.
3. **Pass 2, splits.** The nine files, one at a time, each verified before the next.
4. **Docs.** `apps/desktop/CLAUDE.md`, `docs/architecture.md` § `apps/` (the spine table gains a
   *screen* column, and its first naming trap goes), GLOSSARY U-001/U-002 *Authoritative in* paths,
   and the new U-004.

Each pass is staged for the user to commit, and passes `npm test`, `npm run check`, the security
config test, and a live preview in light and dark at 375 px with 0 console errors, opening every
screen and overlay in the table.

## Consequences

- A screen or dialog is found by its name: the folder for a screen, the file name for a dialog. The
  table above, once copied into `docs/architecture.md`, answers the rest, including every door.
- The spine is unchanged for a reviewer reading a feature end to end: `lib/<domain>/` and
  `core/api|types/<domain>.ts` keep their names and contents, and gain a few rune modules from App.
- Import direction becomes a rule a test can hold: `App` imports anything; `screens/` imports
  `lib/` and, from `shell/`, only `Icon` and `ConfirmDialog`; `shell/` never imports `screens/`;
  `lib/` never imports a component (true today).
- **Every component moves once**, so `git blame` needs `--follow` for history before the move. Git
  records a rename when content is at least 50% similar, which a move with import edits keeps.
- **What a move breaks, known in advance.** `tests/unit/test_desktop_security_config.py:150` pins
  the one `{@html}` sink at `lib/chat/Markdown.svelte`, and fails loudly when it moves (its scan
  covers all of `src/`, so no sink can hide). Two asset imports of `app-mark.png` change depth
  (`AboutDialog`, `Topbar`), and RV1 found that neither `svelte-check` nor `npm test` sees a wrong
  asset path: the live preview is the gate for it. Tracked docs name a component 180 times in 34
  files outside the archives and the DEVLOG. The six living ones are updated (`docs/architecture.md`,
  `apps/desktop/CLAUDE.md`, `GLOSSARY.md`, `docs/security.md`, `docs/knowledge-layer.md`,
  `docs/ROADMAP.md`); ADRs and dated specs keep their old paths as the record of their time.
- App stays past the component cap at 600–800 lines, recorded here as deliberate. The seventeen
  components between 300 and 690 lines stay past it too, until ROADMAP 110 removes their button
  classes. A shared dialog frame (20 copies of `.scrim`) would be the next lever. It is not part of
  this ADR, because it changes how every dialog looks and behaves.
- Work in flight moves too. A session that starts from a branch cut before pass 1 meets renamed
  files when it merges; git follows a rename, but a file both renamed and split in pass 2 needs a
  hand merge. That is why step 1 waits.

## Enforced by

- Once built, a `node:test` module (`src/layout.test.ts`, pure `.ts`, reading the tree) that
  fails when `screens/` or `shell/` holds a `.ts` file, when `shell/` or `lib/` imports from
  `screens/`, or when `lib/` imports a `.svelte` component.
- `tests/unit/test_desktop_security_config.py` (the sink's path, updated in pass 1).
- `GLOSSARY.md` U-004 (the screens' names) and `docs/architecture.md` § `apps/` (the table).

## Confidence

High on where things go. The table is the app's own headings, and every door was read in the
source on 2026-11-05. Medium on pass 2's line counts. App's 600–800 comes from reading its function
list, not from a split, and a part whose styles do not separate stays whole, so some files may end
above the cap with that reason written beside them.
