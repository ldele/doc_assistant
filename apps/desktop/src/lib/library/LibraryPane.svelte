<script lang="ts">
  // The Library workspace (feature-library-browser.md L1, feature-library-redesign.md L4 Phase A):
  // breadcrumb + Back + sort/view toggles over either the inventory grid or a drilled-in document.
  //
  // Presentational by design. It reads its *preferences* from `library/prefs.svelte.ts` and its
  // pure helpers from `library.ts`, but everything else arrives as an **explicit prop contract**.
  // That is deliberate: the derived facet pipeline (`facetList`, `keywordsOf`) depends on the
  // keyword-family state, which App owns because a family write also re-points the live facet
  // selection across three domains (PR-2.5 D5). Rather than break that boundary to shorten this
  // list, the dependency surface is written out here where a reviewer can see all of it.
  import Icon from '../shell/Icon.svelte'
  import LibraryGrid from './LibraryGrid.svelte'
  import LibraryBrowser from './LibraryBrowser.svelte'
  import LibraryFilterStrip from './LibraryFilterStrip.svelte'
  import { LIB_SORTS, libPrefs, setLibrarySort, setLibraryView } from './prefs.svelte'
  import { accept, canAccept, openChooser, unavailableReason } from './accept.svelte'
  import {
    collectionLabel,
    docLabel,
    type KeywordFacet,
    type LibraryCollection,
  } from './library'
  import type { LibraryDocument, LibraryFolder } from '../core/types'

  interface Props {
    // --- the document pipeline (App owns the state; these are its derived outputs)
    documents: LibraryDocument[]
    visibleDocs: LibraryDocument[]
    facetList: KeywordFacet[]
    keywordsOf: (d: LibraryDocument) => string[]
    openDoc: LibraryDocument | null
    /** id → display name, for the breadcrumb label. */
    folderNames: Map<string, string>
    // --- where the user is
    libraryCollection: LibraryCollection
    libraryDocId: string | null
    libraryQuery: string
    libraryKeywords: string[]
    folders: LibraryFolder[]
    // --- batch add-to-folder select mode
    libSelectMode: boolean
    libSelected: string[]
    libAddMenuOpen: boolean
    // --- navigation
    onLibraryBack: () => void
    onOpenDocument: (id: string) => void
    onSearchAll: () => void
    onSelectCollection: (c: LibraryCollection) => void
    onSetDocId: (id: string | null) => void
    // --- facets
    onToggleKeywordFacet: (v: string) => void
    onClearKeywordFacets: () => void
    onOpenKeywordFilter: () => void
    // --- select mode
    onEnterSelectMode: () => void
    onExitSelectMode: () => void
    onToggleSelected: (id: string) => void
    onSetAddMenuOpen: (open: boolean) => void
    onAddSelectionToFolder: (folderId: string) => void
    onClearSelection: () => void
    onOpenManageFolders: () => void
    // --- per-document actions
    onEditDoc: (id: string) => void
    onDeleteDoc: (id: string) => void
    onRevealDoc: (id: string) => void
    onManageFoldersForDoc: (id: string) => void
  }
  const {
    documents, visibleDocs, facetList, keywordsOf, openDoc,
    libraryCollection, libraryDocId, libraryQuery, libraryKeywords, folders,
    libSelectMode, libSelected, libAddMenuOpen,
    folderNames,
    onLibraryBack, onOpenDocument, onSearchAll,
    onSelectCollection, onSetDocId,
    onToggleKeywordFacet, onClearKeywordFacets, onOpenKeywordFilter,
    onEnterSelectMode, onExitSelectMode, onToggleSelected, onSetAddMenuOpen,
    onAddSelectionToFolder, onClearSelection, onOpenManageFolders,
    onEditDoc, onDeleteDoc, onRevealDoc, onManageFoldersForDoc,
  }: Props = $props()
</script>

  <div class="library">
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
        <div class="viewtoggle" role="group" aria-label="Layout">
          <button
            class:active={libPrefs.view === 'grid'}
            onclick={() => setLibraryView('grid')}
            aria-label="Grid view"
            aria-pressed={libPrefs.view === 'grid'}
            title="Grid view"
            type="button"><Icon name="layout-grid" size={15} /></button
          >
          <button
            class:active={libPrefs.view === 'list'}
            onclick={() => setLibraryView('list')}
            aria-label="List view"
            aria-pressed={libPrefs.view === 'list'}
            title="List view"
            type="button"><Icon name="list" size={15} /></button
          >
        </div>
        {#if documents.length > 0}
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
      <div class="selectbar" role="toolbar" aria-label="Selection actions">
        <span class="selcount">{libSelected.length} selected</span>
        <div class="addwrap">
          <button
            class="selact primaryish"
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
        <button class="selact" disabled={libSelected.length === 0} onclick={() => onClearSelection()} type="button">
          Clear
        </button>
        <button class="selact" onclick={onExitSelectMode} type="button">Done</button>
      </div>
    {/if}

    {#if libraryDocId !== null}
      <LibraryBrowser docId={libraryDocId} doc={openDoc} onOpenDocument={onOpenDocument} />
    {:else}
      <section class="libmain">
        {#if documents.length === 0}
          <div class="libempty">
            <span class="state-mark"><Icon name="library" size={26} /></span>
            <strong>No documents indexed yet</strong>
            <p>Point doc_assistant at a folder of your documents to fill the library.</p>
          </div>
        {:else}
          <LibraryFilterStrip
            selected={libraryKeywords}
            resultCount={visibleDocs.length}
            hasKeywords={facetList.length > 0}
            onOpen={onOpenKeywordFilter}
            onRemove={onToggleKeywordFacet}
            onClear={onClearKeywordFacets}
          />
          {#if visibleDocs.length === 0}
            <div class="libempty">
              <span class="state-mark"><Icon name="search" size={26} /></span>
              {#if libraryQuery.trim() !== '' || libraryKeywords.length > 0}
                <strong>No documents match your filters</strong>
                <p>
                  Nothing in {collectionLabel(libraryCollection, folderNames)} matches{#if libraryQuery.trim() !== ''}
                    “{libraryQuery.trim()}”{/if}.
                </p>
                {#if libraryKeywords.length > 0}
                  <button class="widen" onclick={onClearKeywordFacets} type="button">
                    Clear keyword filters
                  </button>
                {/if}
                {#if libraryCollection.kind !== 'all'}
                  <button class="widen" onclick={onSearchAll} type="button">
                    Search all {documents.length} documents
                  </button>
                {/if}
              {:else}
                <strong>Nothing in {collectionLabel(libraryCollection, folderNames)}</strong>
                <p>This collection is empty right now.</p>
              {/if}
            </div>
          {:else}
            <LibraryGrid
              documents={visibleDocs}
              view={libPrefs.view}
              activeKeywords={libraryKeywords}
              {keywordsOf}
              selectMode={libSelectMode}
              selectedIds={libSelected}
              onToggleSelect={onToggleSelected}
              onOpenDocument={onOpenDocument}
              onEditMetadata={onEditDoc}
              onReveal={onRevealDoc}
              onAddToFolder={onManageFoldersForDoc}
              onDelete={onDeleteDoc}
            />
          {/if}
        {/if}
      </section>
    {/if}
  </div>

<style>
  /* Library pane (L4): breadcrumb/Back/view-toggle bar over the grid or the drilled chunk view. */
  .library {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
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
  .selact {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font: inherit;
    font-size: 0.78rem;
    padding: 0.26rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--fg);
    cursor: pointer;
  }
  .selact:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .selact:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .selact.primaryish {
    background: var(--surface-2);
    font-weight: 600;
  }
  .libmain {
    flex: 1;
    overflow-y: auto;
    min-width: 0;
  }
  .libempty {
    max-width: 540px;
    margin: var(--space-6) auto 0;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .libempty strong {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    font-weight: 600;
    color: var(--fg);
  }
  .libempty p {
    color: var(--fg-2);
    font-size: var(--text-sm);
    line-height: 1.6;
    max-width: 46ch;
    margin: var(--space-2) 0 var(--space-4);
  }
  .widen {
    font-size: var(--text-sm);
    color: var(--accent);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: var(--space-2) var(--space-3);
  }
  .widen:hover {
    border-color: var(--accent);
  }
</style>
