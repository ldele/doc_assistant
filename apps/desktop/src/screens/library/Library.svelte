<script lang="ts">
  // The Library workspace (feature-library-browser.md L1, feature-library-redesign.md L4 Phase A):
  // the header bar (and, in select mode, the select bar) over either the library's body or a
  // drilled-in document. The bars are `HeaderBar`; the body here is the summary, the empty
  // Library (the drop target), the filter strip with the grid or list, and the no-match state.
  //
  // Presentational by design. It reads its *preferences* from `library/prefs.svelte.ts`, the
  // select mode from `library/documents.svelte.ts` (ADR-064 choice 5) and its pure helpers from
  // `library.ts`, but everything else arrives as an **explicit prop contract**.
  // That is deliberate: the derived facet pipeline (`facetList`, `keywordsOf`) depends on the
  // keyword-family state, which App owns because a family write also re-points the live facet
  // selection across three domains (PR-2.5 D5). Rather than break that boundary to shorten this
  // list, the dependency surface is written out here where a reviewer can see all of it.
  import Icon from '../../shell/Icon.svelte'
  import DocumentGrid from './DocumentGrid.svelte'
  import DocumentPage from '../document-page/DocumentPage.svelte'
  import FilterStrip from './FilterStrip.svelte'
  import HeaderBar from './HeaderBar.svelte'
  import Summary from './Summary.svelte'
  import { shell } from '../../lib/shell/shell.svelte'
  import { libPrefs } from '../../lib/library/prefs.svelte'
  import { library, pickDocument } from '../../lib/library/documents.svelte'
  import { accept, canAccept, openChooser, unavailableReason } from '../../lib/library/accept.svelte'
  import { formatLine, limitPhrase } from '../../lib/library/formats'
  import {
    collectionLabel,
    type KeywordFacet,
    type LibraryCollection,
  } from '../../lib/library/library'
  import type { LibraryDocument, LibraryFolder } from '../../lib/core/types'

  interface Props {
    /** A chat citation asking for the source pane at a page (ADR-050 D2). Passed straight
     *  through to the browser, which owns the pane. */
    sourceJump?: { docId: string; page: number | null; nonce: number } | null
    // --- the document pipeline (App owns the state; these are its derived outputs)
    documents: LibraryDocument[]
    /** Whether the list above is an answer yet — an empty list before the first fetch is
     *  not one, and the empty state below is a claim about the user's library. */
    documentsLoaded: boolean
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
  }
  const {
    sourceJump = null,
    documents, documentsLoaded, visibleDocs, facetList, keywordsOf, openDoc,
    libraryCollection, libraryDocId, libraryQuery, libraryKeywords, folders,
    folderNames,
    onLibraryBack, onOpenDocument, onSearchAll,
    onSelectCollection, onSetDocId,
    onToggleKeywordFacet, onClearKeywordFacets, onOpenKeywordFilter,
  }: Props = $props()

  // The Summary layout (ROADMAP 112) covers the whole library or one folder: the collections the
  // rail opens as places. A type, a date bucket or a summary's own list is a cut through them, so
  // there the toggle offers Grid and List only and the documents show.
  const summaryScope = $derived<{ folder: string | null } | null>(
    libraryCollection.kind === 'all'
      ? { folder: null }
      : libraryCollection.kind === 'folder'
        ? { folder: libraryCollection.value }
        : null,
  )
  const showSummary = $derived(libPrefs.summaryOpen && summaryScope !== null)

  function openFilesInSettings(): void {
    // The files that are not in the library are listed, with their state, in Settings > Documents.
    shell.settingsSection = 'documents'
    shell.showSettings = true
  }
</script>

  <div class="library">
    <HeaderBar
      {documents}
      {visibleDocs}
      {openDoc}
      {folderNames}
      {libraryCollection}
      {libraryDocId}
      {folders}
      {summaryScope}
      {showSummary}
      {onLibraryBack}
      {onSelectCollection}
      {onSetDocId}
    />

    {#if libraryDocId !== null}
      <DocumentPage
        docId={libraryDocId}
        doc={openDoc}
        onOpenDocument={onOpenDocument}
        {sourceJump}
      />
    {:else}
      <section class="libmain">
        {#if showSummary && summaryScope !== null}
          <Summary
            folderId={summaryScope.folder}
            onOpenCollection={(c) => {
              libPrefs.summaryOpen = false
              onSelectCollection(c)
            }}
            onOpenLibrary={() => onSelectCollection({ kind: 'all' })}
            onShowFiles={openFilesInSettings}
          />
        {:else if documents.length === 0 && !documentsLoaded}
          <!-- "Empty" and "not yet known" are different answers, and `documents` starts empty
               either way. Without this the one screen whose job is to say "there is nothing here"
               said it to a user with 98 documents for as long as the fetch took. -->
          <p class="libloading">Loading your library…</p>
        {:else if documents.length === 0}
          <!-- AD4 — the first-run empty state IS the drop target.
               The window has always accepted a drop; on the one screen that exists to say "there
               is nothing here yet", saying nothing about how to change that was the gap. It
               highlights from `accept.dragging`, the same window-level Tauri signal the chooser
               and the full-window veil use — dropping anywhere works, this only says so.
               The old copy said "Point doc_assistant at a folder", which named the code identity
               rather than the product (ADR-012) and described the pre-AD3b model where the library
               was a folder you aimed at rather than somewhere you add documents. -->
          <div class="libempty">
            <div class="emptydrop" class:over={accept.dragging}>
              <span class="state-mark"><Icon name="library" size={26} /></span>
              <strong>{accept.dragging ? 'Drop them now' : 'Your library is empty'}</strong>
              {#if canAccept()}
                <p>Drag files or folders anywhere in this window, or add them from your computer.</p>
                {#if accept.accepts}
                  <p class="formats">
                    {formatLine(accept.accepts.extensions)} · {limitPhrase(accept.accepts)}
                  </p>
                {/if}
                <button class="emptyadd" onclick={openChooser} type="button">
                  <Icon name="plus" size={15} />
                  Add documents
                </button>
              {:else}
                <!-- Honest degradation rather than a dead button: in a browser the accept surface
                     does not exist at all, and saying so beats a control that cannot work. -->
                <p>{unavailableReason()}</p>
              {/if}
            </div>
          </div>
        {:else}
          <FilterStrip
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
            <DocumentGrid
              documents={visibleDocs}
              view={libPrefs.view}
              activeKeywords={libraryKeywords}
              {keywordsOf}
              selectMode={library.selectMode}
              selectedIds={library.selected}
              onPick={(id, shift) => pickDocument(visibleDocs.map((d) => d.id), id, shift)}
              onOpenDocument={onOpenDocument}
            />
          {/if}
        {/if}
      </section>
    {/if}
  </div>

<style>
  /* Library pane (L4): the bars (HeaderBar) over the grid or the drilled chunk view. */
  .library {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .libmain {
    flex: 1;
    overflow-y: auto;
    min-width: 0;
  }
  /* AD4 — the empty state's drop target. Dashed, so it reads as somewhere things go rather than
     as a card; the highlight is driven by the window-level drag signal, not by a DOM dragover
     (Tauri intercepts the OS drag before the DOM sees it). */
  .emptydrop {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    padding: var(--space-6) var(--space-4);
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    transition: border-color 120ms ease, background 120ms ease;
  }
  .emptydrop.over {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
  }
  .formats {
    font-size: var(--text-meta);
    margin-top: calc(-1 * var(--space-2));
  }
  .emptyadd {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--fg);
  }
  .emptyadd:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  @media (prefers-reduced-motion: reduce) {
    .emptydrop {
      transition: none;
    }
  }

  .libloading {
    margin: 2rem auto;
    text-align: center;
    color: var(--fg-2);
    font-size: 0.85rem;
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
