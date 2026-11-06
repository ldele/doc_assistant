<script lang="ts">
  // The Library tab's header bar (feature-library-redesign.md L4 Phase A) and, in select mode,
  // the select bar under it. The header bar: Back, the location (Library › a collection › a
  // document) and, over the documents, sort, the layout toggle (Grid / List / Summary), *Select*
  // and *Add documents*. The select bar (ADR-063): the count, *Select all* / *Select none*, *Add to
  // folder…*, *Re-run…* and *Done*.
  //
  // The two bars are one file because they open the same dropdown menu (sort; *Add to folder…*):
  // its style is about 40 lines of scoped CSS that two files would each carry (ADR-064 choice 6).
  // Library hands in where the user is and the select mode's state, and gets every action back
  // as a callback; the only state written here is `libPrefs` (the sort menu, the layout).
  import Icon from '../../shell/Icon.svelte'
  import { selectAllLabel, selectAllOrNone } from '../../lib/shell/selection'
  import { LIB_SORTS, libPrefs, setLibrarySort, setLibraryView } from '../../lib/library/prefs.svelte'
  import { accept, canAccept, openChooser, unavailableReason } from '../../lib/library/accept.svelte'
  import { collectionLabel, docLabel, type LibraryCollection } from '../../lib/library/library'
  import type { LibraryDocument, LibraryFolder } from '../../lib/core/types'

  interface Props {
    /** The whole list: *Select* shows only when it has documents. */
    documents: LibraryDocument[]
    /** What the grid shows: *Select all* picks these. */
    visibleDocs: LibraryDocument[]
    openDoc: LibraryDocument | null
    /** id → display name, for the breadcrumb label. */
    folderNames: Map<string, string>
    // --- where the user is
    libraryCollection: LibraryCollection
    libraryDocId: string | null
    folders: LibraryFolder[]
    /** The layout toggle's *Summary*: where it is offered, and whether it is the layout shown. */
    summaryScope: { folder: string | null } | null
    showSummary: boolean
    // --- batch add-to-folder select mode
    libSelectMode: boolean
    libSelected: string[]
    libAddMenuOpen: boolean
    // --- navigation
    onLibraryBack: () => void
    onSelectCollection: (c: LibraryCollection) => void
    onSetDocId: (id: string | null) => void
    // --- select mode
    onEnterSelectMode: () => void
    onExitSelectMode: () => void
    onSetAddMenuOpen: (open: boolean) => void
    onAddSelectionToFolder: (folderId: string) => void
    /** Replace the selection: what *Select all* / *Select none* sets. */
    onSetSelection: (ids: string[]) => void
    onOpenManageFolders: () => void
    /** Open the re-run picker for the selection (ROADMAP 21). */
    onReingestSelection: () => void
  }
  const {
    documents, visibleDocs, openDoc, folderNames,
    libraryCollection, libraryDocId, folders,
    summaryScope, showSummary,
    libSelectMode, libSelected, libAddMenuOpen,
    onLibraryBack, onSelectCollection, onSetDocId,
    onEnterSelectMode, onExitSelectMode, onSetAddMenuOpen,
    onAddSelectionToFolder, onSetSelection, onOpenManageFolders, onReingestSelection,
  }: Props = $props()
</script>

<div class="libnav">
  {#if libraryDocId !== null || libraryCollection.kind !== 'all'}
    <button class="libback" onclick={onLibraryBack} aria-label="Back" title="Back">
      <Icon name="arrow-left" size={15} />
    </button>
  {/if}
  <nav class="crumbs" aria-label="Library location">
    <button
      class="crumb"
      onclick={() => onSelectCollection({ kind: 'all' })}
      disabled={libraryDocId === null && libraryCollection.kind === 'all'}
      type="button">Library</button
    >
    {#if libraryCollection.kind !== 'all'}
      <span class="crumbsep"><Icon name="chevron-right" size={13} /></span>
      <button
        class="crumb"
        onclick={() => onSetDocId(null)}
        disabled={libraryDocId === null}
        type="button">{collectionLabel(libraryCollection, folderNames)}</button
      >
    {/if}
    {#if openDoc}
      <span class="crumbsep"><Icon name="chevron-right" size={13} /></span>
      <span class="crumb current" title={openDoc.filename}>{docLabel(openDoc)}</span>
    {/if}
  </nav>
  {#if libraryDocId === null}
    {#if !showSummary}
    <div class="libsort">
      <button
        class="sortbtn"
        onclick={() => (libPrefs.sortOpen = !libPrefs.sortOpen)}
        aria-haspopup="menu"
        aria-expanded={libPrefs.sortOpen}
        title="Sort documents"
        type="button"><Icon name="arrow-up-down" size={15} /></button
      >
      {#if libPrefs.sortOpen}
        <div
          class="sort-backdrop"
          onclick={() => (libPrefs.sortOpen = false)}
          role="presentation"
        ></div>
        <div class="sortmenu" role="menu">
          {#each LIB_SORTS as s}
            <button
              class="sortitem"
              class:on={libPrefs.sort === s.key}
              role="menuitemradio"
              aria-checked={libPrefs.sort === s.key}
              onclick={() => setLibrarySort(s.key)}
              type="button"
            >
              <span class="tick"
                >{#if libPrefs.sort === s.key}<Icon name="check" size={13} />{/if}</span
              >
              {s.label}
            </button>
          {/each}
        </div>
      {/if}
    </div>
    {/if}
    <div class="viewtoggle" role="group" aria-label="Layout">
      <button
        class:active={!showSummary && libPrefs.view === 'grid'}
        onclick={() => setLibraryView('grid')}
        aria-label="Grid view"
        aria-pressed={!showSummary && libPrefs.view === 'grid'}
        title="Grid view"
        type="button"><Icon name="layout-grid" size={15} /></button
      >
      <button
        class:active={!showSummary && libPrefs.view === 'list'}
        onclick={() => setLibraryView('list')}
        aria-label="List view"
        aria-pressed={!showSummary && libPrefs.view === 'list'}
        title="List view"
        type="button"><Icon name="list" size={15} /></button
      >
      <!-- ROADMAP 112: a third layout, for the library or the open folder. Navigation, not an
           action (ADR-063), so it keeps the toggle's look. -->
      {#if summaryScope !== null}
        <button
          class="summarytoggle"
          class:active={showSummary}
          onclick={() => {
            if (libSelectMode) onExitSelectMode()
            libPrefs.summaryOpen = true
          }}
          aria-pressed={showSummary}
          title={summaryScope.folder === null
            ? 'What went wrong at indexing, and what is known about your documents'
            : 'What went wrong at indexing, and what is known about this folder'}
          type="button">Summary</button
        >
      {/if}
    </div>
    {#if documents.length > 0 && !showSummary}
      <button
        class="selecttoggle"
        class:active={libSelectMode}
        aria-pressed={libSelectMode}
        onclick={() => (libSelectMode ? onExitSelectMode() : onEnterSelectMode())}
        title="Select documents to add to a folder"
        type="button"
      >
        <Icon name="square-check-big" size={14} /> Select
      </button>
    {/if}
    <!-- The primary action for this pane, so it lives with the other document actions rather
         than in the app toolbar — whose right cluster is identity + config by design ("Brand =
         identity anchor only, parked on the right beside Settings", Topbar.svelte). Moved here
         2026-08-24 on the user's call. Outside the Tauri window there is no picker and no
         drag-drop, so it stays visible, disabled, and SAYS WHY: hiding it would make a browser
         look like a missing feature. Reachable from Chat via the app menu. -->
    <button
      class="addbtn"
      onclick={openChooser}
      disabled={!canAccept() || accept.picking}
      title={unavailableReason() ?? 'Add documents'}
      type="button"
    >
      <Icon name="plus" size={14} /> Add documents
    </button>
  {/if}
</div>

{#if libSelectMode && libraryDocId === null}
  <!-- Action roles (ADR-063): the selection helper is quiet, the actions are plain, and Done
       ends the mode, so it sits apart at the trailing end in the primary style. "Clear" read
       as a possible delete (the user, 2026-11-05); it only ever emptied the selection, which
       is what *Select none* now says. -->
  <div class="selectbar" role="toolbar" aria-label="Selection actions">
    <span class="selcount">{libSelected.length} selected</span>
    <button
      class="btn btn-quiet"
      disabled={visibleDocs.length === 0}
      onclick={() => onSetSelection(selectAllOrNone(visibleDocs.map((d) => d.id), libSelected))}
      type="button"
    >
      {selectAllLabel(libSelected)}
    </button>
    <span class="selsep" aria-hidden="true"></span>
    <div class="addwrap">
      <button
        class="btn"
        disabled={libSelected.length === 0}
        aria-haspopup="menu"
        aria-expanded={libAddMenuOpen}
        onclick={() => onSetAddMenuOpen(!libAddMenuOpen)}
        type="button"
      >
        <Icon name="folder" size={13} /> Add to folder…
      </button>
      {#if libAddMenuOpen}
        <div class="sort-backdrop" onclick={() => onSetAddMenuOpen(false)} role="presentation"></div>
        <div class="sortmenu" role="menu">
          {#each folders as f (f.id)}
            <button class="sortitem" role="menuitem" onclick={() => onAddSelectionToFolder(f.id)} type="button">
              <span class="tick"><Icon name="folder" size={13} /></span>
              {f.name}
            </button>
          {:else}
            <button
              class="sortitem"
              role="menuitem"
              onclick={() => {
                onSetAddMenuOpen(false)
                onOpenManageFolders()
              }}
              type="button"
            >
              <span class="tick"><Icon name="plus" size={13} /></span>
              No folders yet — create one…
            </button>
          {/each}
        </div>
      {/if}
    </div>
    <!-- ROADMAP 21: the same operation as the document panel's, over the selection. It opens
         the SAME dialog with a list instead of one id, so the parts and the cost statement
         cannot drift between the two entry points. -->
    <button
      class="btn"
      disabled={libSelected.length === 0}
      onclick={() => onReingestSelection()}
      type="button"
    >
      <Icon name="rotate-ccw" size={13} /> Re-run…
    </button>
    <button class="btn btn-primary seldone" onclick={onExitSelectMode} type="button">Done</button>
  </div>
{/if}

<style>
  /* The header bar: Back, the location, then sort / layout / Select / Add documents. */
  .libnav {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) 0;
    border-bottom: 1px solid var(--border);
    min-height: 2.4rem;
  }
  .libback {
    flex: none;
    display: inline-flex;
    align-items: center;
    padding: 0.28rem 0.5rem;
    color: var(--fg-2);
  }
  .libback:hover {
    color: var(--fg);
  }
  .crumbs {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 0.2rem;
    overflow: hidden;
  }
  .crumb {
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    border: none;
    background: none;
    color: var(--accent);
    padding: 0.15rem 0.25rem;
    border-radius: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .crumb:hover:not(:disabled) {
    background: var(--surface);
  }
  /* The current location isn't a link — a disabled tail crumb reads as "you are here". */
  .crumb:disabled,
  .crumb.current {
    color: var(--fg);
    cursor: default;
    opacity: 1;
    font-weight: 600;
  }
  .crumbsep {
    color: var(--fg-2);
    display: inline-flex;
    flex: none;
  }
  /* The pane's primary action: filled accent, unlike the neutral view/sort toggles beside it.
     A single class, and nothing else in this file targets `.addbtn` — but if a sibling rule is
     ever added, give this one two classes rather than relying on source order (the 2026-08-19
     tick-row accident, repeated in the toolbar on 2026-08-21). */
  .addbtn {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.28rem 0.7rem;
    border: 1px solid var(--accent);
    border-radius: 8px;
    background: var(--accent);
    color: var(--accent-fg);
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .addbtn:disabled {
    background: var(--surface);
    border-color: var(--border);
    color: var(--fg-2);
    opacity: 0.7;
    cursor: default;
  }
  .libsort {
    position: relative;
    flex: none;
  }
  .sortbtn {
    display: inline-flex;
    align-items: center;
    padding: 0.28rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--fg-2);
    cursor: pointer;
  }
  .sortbtn:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  /* The dropdown menu both bars open: the sort menu here, *Add to folder…* on the select bar. */
  .sort-backdrop {
    position: fixed;
    inset: 0;
    z-index: 20;
  }
  .sortmenu {
    position: absolute;
    z-index: 21;
    top: calc(100% + 4px);
    right: 0;
    min-width: 200px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: var(--shadow-2);
    padding: 0.25rem;
  }
  .sortitem {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    width: 100%;
    padding: 0.4rem 0.5rem;
    border: none;
    background: none;
    color: var(--fg);
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 0.82rem;
    text-align: left;
    white-space: nowrap;
  }
  .sortitem:hover {
    background: var(--surface-2);
  }
  .sortitem.on {
    color: var(--accent);
  }
  .sortitem .tick {
    display: inline-flex;
    width: 13px;
    flex: none;
  }
  .viewtoggle {
    flex: none;
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .viewtoggle button {
    border: none;
    border-radius: 0;
    background: var(--surface);
    color: var(--fg-2);
    padding: 0.28rem 0.55rem;
    display: inline-flex;
    align-items: center;
  }
  .viewtoggle button.active {
    background: var(--surface-2);
    color: var(--accent);
  }
  .viewtoggle .summarytoggle {
    font: inherit;
    font-size: 0.78rem;
    border-left: 1px solid var(--border);
  }
  /* Select mode (batch add-to-folder) — toggle beside the view switch + a slim action bar. */
  .selecttoggle {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font: inherit;
    font-size: 0.78rem;
    padding: 0.28rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--fg-2);
    cursor: pointer;
  }
  .selecttoggle:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  .selecttoggle.active {
    background: var(--surface-2);
    color: var(--accent);
    border-color: var(--accent);
  }
  .selectbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--border);
  }
  .selcount {
    font-size: 0.78rem;
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    min-width: 6.5em;
  }
  .addwrap {
    position: relative;
    flex: none;
  }
  .addwrap .sortmenu {
    left: 0;
    right: auto;
  }
  /* A thin rule between what changes the selection and what acts on it. */
  .selsep {
    width: 1px;
    align-self: stretch;
    margin: 0.15rem 0.2rem;
    background: var(--border);
  }
  /* Done ends the mode: the trailing end, apart from the actions (ADR-063). */
  .seldone {
    margin-left: auto;
  }
</style>
