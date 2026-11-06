<script lang="ts">
  // Left-rail frame (feature-conversation-history.md, Decisions 7, 8; library nav tree by
  // feature-library-redesign.md, L4 Phase A). Persistent column on desktop; an off-canvas drawer
  // under 720px. Depending on `mode` it holds the chat rail (Chat history, a snippet App composes
  // from `screens/chat/ChatHistory.svelte`), the library navigation tree (All documents → Folders
  // → Types → Added; the doc list itself is the main-pane grid), or the graph rail (the concept
  // index, App-composed too, with the Taxonomy footer). The library tree stays here, not in
  // `screens/library/`: its styles are this frame's tree rows and filter box (ADR-064 choice 6).
  import type { Snippet } from 'svelte'
  import type { LibraryDocument, LibraryFolder } from '../lib/core/types'
  import {
    type LibraryCollection,
    DATE_BUCKET_LABELS,
    collectionLabel,
    dateGroups,
    folderNameMap,
    sameCollection,
    typeGroups,
  } from '../lib/library/library'
  import Icon from './Icon.svelte'

  let {
    mode,
    documents,
    folders,
    libraryCollection,
    libraryQuery = $bindable(''),
    open = false,
    chatRail = null,
    graphRail = null,
    onSelectCollection,
    onManageFolders,
    onOpenTaxonomy,
    onClose,
  }: {
    mode: 'chat' | 'library' | 'graph'
    documents: LibraryDocument[]
    folders: LibraryFolder[]
    libraryCollection: LibraryCollection
    libraryQuery?: string
    open?: boolean
    chatRail?: Snippet | null
    graphRail?: Snippet | null
    onSelectCollection: (c: LibraryCollection) => void
    onManageFolders: () => void
    onOpenTaxonomy: () => void
    onClose?: () => void
  } = $props()

  // Library nav-tree groups (L4 Decision 3), computed client-side from the payload. Types/Added
  // render only with ≥2 entries (Decision 3a — a one-option filter is noise). Keywords are no
  // longer a nav group — they moved to the main-pane facet bar as a multi-select filter.
  // Folders are the exception: they come from the API as a prop, because a folder derived from
  // the document payload could never appear while empty (ADR-025 F1, spec D3).
  const types = $derived(typeGroups(documents))
  const dates = $derived(dateGroups(documents, new Date()))
  const folderNames = $derived(folderNameMap(folders))
</script>

<!-- The mode switch + collapse + search moved to the top toolbar (App). The sidebar is now purely
     the contextual list for the active mode. -->
<aside class="sidebar" class:open>
  {#if mode === 'library' && documents.length > 0}
    <div class="toolbar">
      <div class="search">
        <Icon name="search" size={13} />
        <!-- Library filter narrows the active collection shown in the grid (Decision 5a).
             "Filter", not "Search" — the Ctrl/⌘-K overlay is the one global search. -->
        <input
          type="text"
          bind:value={libraryQuery}
          placeholder={libraryCollection.kind === 'all'
            ? 'Filter library…'
            : `Filter ${collectionLabel(libraryCollection, folderNames)}…`}
          aria-label="Filter library"
        />
        {#if libraryQuery}
          <button
            class="clear"
            onclick={() => (libraryQuery = '')}
            aria-label="Clear filter"
            type="button"
          >
            <Icon name="x" size={13} />
          </button>
        {/if}
      </div>
    </div>
  {/if}

  {#if mode === 'chat'}
    <!-- Chat history is App-composed (`screens/chat/ChatHistory.svelte`): the Chat tab's rail,
         with its own filter, select mode and sort. -->
    {@render chatRail?.()}
  {:else if mode === 'library'}
    <nav class="list" aria-label="Library navigation">
      {#if documents.length === 0}
        <p class="empty">No documents indexed yet.</p>
      {:else}
        <button
          class="treerow"
          class:active={libraryCollection.kind === 'all'}
          aria-current={libraryCollection.kind === 'all' ? 'true' : undefined}
          onclick={() => onSelectCollection({ kind: 'all' })}
          type="button"
        >
          <span class="treeicon"><Icon name="library" size={14} /></span>
          <span class="treelabel">All documents</span>
          <span class="count">{documents.length}</span>
        </button>

        <div class="section-head-row">
          <p class="section-header">Folders</p>
          <button class="managelink" onclick={onManageFolders} type="button">Manage…</button>
        </div>
        {#if folders.length === 0}
          <p class="tree-empty">No folders yet — create one in Manage…</p>
        {:else}
          {#each folders as f (f.id)}
            <button
              class="treerow"
              class:active={sameCollection(libraryCollection, { kind: 'folder', value: f.id })}
              aria-current={sameCollection(libraryCollection, { kind: 'folder', value: f.id })
                ? 'true'
                : undefined}
              onclick={() => onSelectCollection({ kind: 'folder', value: f.id })}
              type="button"
            >
              <span class="treeicon"><Icon name="folder" size={14} /></span>
              <span class="treelabel">{f.name}</span>
              <span class="count">{f.doc_count}</span>
            </button>
          {/each}
        {/if}

        {#if types.length >= 2}
          <p class="section-header">Types</p>
          {#each types as g (g.value)}
            <button
              class="treerow"
              class:active={sameCollection(libraryCollection, { kind: 'type', value: g.value })}
              aria-current={sameCollection(libraryCollection, { kind: 'type', value: g.value })
                ? 'true'
                : undefined}
              onclick={() => onSelectCollection({ kind: 'type', value: g.value })}
              type="button"
            >
              <span class="treeicon"><Icon name="file-text" size={14} /></span>
              <span class="treelabel">{g.value.toUpperCase()}</span>
              <span class="count">{g.count}</span>
            </button>
          {/each}
        {/if}

        {#if dates.length >= 2}
          <p class="section-header">Added</p>
          {#each dates as g (g.value)}
            <button
              class="treerow"
              class:active={sameCollection(libraryCollection, { kind: 'date', value: g.value })}
              aria-current={sameCollection(libraryCollection, { kind: 'date', value: g.value })
                ? 'true'
                : undefined}
              onclick={() => onSelectCollection({ kind: 'date', value: g.value })}
              type="button"
            >
              <span class="treeicon"><Icon name="calendar" size={14} /></span>
              <span class="treelabel">{DATE_BUCKET_LABELS[g.value]}</span>
              <span class="count">{g.count}</span>
            </button>
          {/each}
        {/if}

      {/if}
    </nav>
  {:else if mode === 'graph'}
    <nav class="list graphrail" aria-label="Concept index">
      <!-- The concept index itself is App-composed (ConceptIndex) — the graph data + selection live
           in App, and threading them through this component's props would double its surface. -->
      {@render graphRail?.()}
    </nav>
    <!-- Taxonomy (ADR-028 2b) — a dedicated field-forest modal; an action, not a filter, so no
         active/count. Parked in a pinned footer *outside* the scrolling rail: it is a destination
         rather than an index entry, so it should not drift away as the concept list scrolls. -->
    <div class="railfoot">
      <button class="treerow" onclick={onOpenTaxonomy} type="button" title="Organise concepts + documents into research fields">
        <span class="treeicon"><Icon name="tag" size={14} /></span>
        <span class="treelabel">Taxonomy</span>
      </button>
    </div>
  {/if}
</aside>

{#if open}
  <div class="scrim" onclick={onClose} role="presentation"></div>
{/if}

<style>
  .sidebar {
    width: var(--sidebar-width, 260px);
    flex-shrink: 0;
    /* 100% of the .below row (between the toolbar and the status bar) — not 100vh, which would
       overshoot by the toolbar's height now that the sidebar no longer spans the full window. */
    height: 100%;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    background: var(--bg);
    overflow: hidden;
  }
  /* Search + sort toolbar — fixed header strip above the scrolling list. */
  .toolbar {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border);
  }
  /* A *filter* of the visible list, styled lighter than the header's global search (Ctrl/⌘-K) so
     the two read as different tools: transparent fill, smaller type, border only. */
  .search {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
    color: var(--fg-2);
  }
  .search:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 22%, transparent);
  }
  .search input {
    flex: 1;
    min-width: 0;
    font: inherit;
    font-size: 0.78rem;
    border: none;
    background: none;
    color: var(--fg);
    padding: 0;
  }
  .search input:focus {
    outline: none;
  }
  .search input::placeholder {
    color: var(--fg-2);
  }
  .clear {
    font: inherit;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg-2);
    padding: 0;
    display: inline-flex;
    flex: none;
  }
  .clear:hover {
    color: var(--fg);
  }
  /* Pinned rail footer — sits below the scrolling `.list`, so its action stays reachable no
     matter how long the concept index gets. `flex: none` keeps it out of the scroll region. */
  .railfoot {
    flex: none;
    padding: 0.4rem;
    border-top: 1px solid var(--border);
  }
  .list {
    flex: 1;
    overflow-y: auto;
    padding: 0.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  /* The graph rail hosts its own scrolling regions (concept list / gap list) — the outer nav must
     not double-scroll. */
  .list.graphrail {
    overflow-y: hidden;
  }
  .empty {
    color: var(--fg-2);
    font-size: 0.82rem;
    padding: 0.6rem;
    line-height: 1.4;
  }
  /* The Folders header carries its own "Manage…" entry point — the only rail section with a
     write path (ADR-025 F1). Baseline-aligned with the label, not a full-width bar. */
  .section-head-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
  }
  .managelink {
    font: inherit;
    /* A control, not a label: the walkthrough named it as too small to read and hit. */
    font-size: var(--text-sm);
    cursor: pointer;
    color: var(--fg-2);
    background: none;
    border: none;
    padding: 0.2rem 0.4rem;
    border-radius: 6px;
    flex: none;
  }
  .managelink:hover {
    color: var(--accent);
    background: var(--surface-2);
  }
  /* Lavender "tab" labels for Pinned / Recent — grouped + legible (user request). */
  .section-header {
    align-self: flex-start;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--lavender);
    background: color-mix(in srgb, var(--lavender) 14%, transparent);
    border-radius: 6px;
    margin: 0.55rem 0.1rem 0.3rem;
    padding: 0.16rem 0.5rem;
  }
  /* Library nav tree (L4): single-line rows — icon · label · doc count. */
  .treerow {
    font: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    text-align: left;
    border: 1px solid transparent;
    background: none;
    color: var(--fg);
    border-radius: 8px;
    padding: 0.4rem 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    width: 100%;
    min-width: 0;
  }
  .treerow:hover {
    background: var(--surface);
  }
  .treerow.active {
    background: var(--surface-2);
    border-color: var(--border);
  }
  .treeicon {
    color: var(--fg-2);
    display: inline-flex;
    flex: none;
  }
  .treerow.active .treeicon {
    color: var(--accent);
  }
  .treelabel {
    flex: 1;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .count {
    font-size: 0.68rem;
    color: var(--fg-2);
    flex: none;
  }
  .tree-empty {
    color: var(--fg-2);
    font-size: 0.74rem;
    line-height: 1.4;
    padding: 0 0.55rem;
    margin: 0 0 0.2rem;
  }
  .scrim {
    display: none;
  }

  @media (max-width: 720px) {
    /* Off-canvas drawer, anchored to the .below wrapper (App sets position: relative) so it slides
       in *under* the top toolbar, which stays visible and usable. */
    .sidebar {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 0;
      width: min(85vw, 320px);
      z-index: 20;
      transform: translateX(-100%);
      transition: transform 0.2s ease;
      box-shadow: 4px 0 20px rgba(0, 0, 0, 0.2);
    }
    .sidebar.open {
      transform: translateX(0);
    }
    .scrim {
      display: block;
      position: absolute;
      inset: 0;
      background: rgba(0, 0, 0, 0.35);
      z-index: 15;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .sidebar {
      transition: none;
    }
  }
</style>
