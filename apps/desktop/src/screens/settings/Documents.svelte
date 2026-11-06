<script lang="ts">
  // Settings > Documents: *Your documents* (the library folder, Index / Re-index and the run's
  // line), *Manage files*, *Corpus* (counts, disk, the keyword index and its Rebuild, the data home,
  // Open another library…) and the letter-spacing ignore list.
  //
  // Settings holds what this section shows between visits, since the rail unmounts it: the folder
  // typed, the last run's result, a Rebuild in flight and their error lines. Settings also starts
  // the run and opens the library dialogs, because the questions they ask are mounted there,
  // outside the panel. This file owns the folder picker, the Rebuild and View indexing.
  import type { IngestStatus, Settings } from '../../lib/core/types'
  import { reindexKeywords } from '../../lib/core/api'
  import { canPickFiles, pickPaths, type LibraryState } from '../../lib/core/tauri'
  import { ingestRun, openIndexing } from '../../lib/core/ingest.svelte'
  import Icon from '../../shell/Icon.svelte'
  import ManageFiles from './ManageFiles.svelte'
  import SpacingIgnoreList from './SpacingIgnoreList.svelte'
  import { describeIndex, formatBytes, perDocument } from '../../lib/settings/corpus'

  let {
    settings = $bindable(),
    dir = $bindable(),
    inputEl = $bindable(),
    reindexing = $bindable(),
    reindexError = $bindable(),
    busy,
    ingest,
    actionError,
    libState,
    libraryError,
    focusSpacing,
    onIndex,
    onFolderEdit,
    onOpenAnotherLibrary,
    onUseOwnLibrary,
    onClose,
  }: {
    settings: Settings
    /** The folder field, seeded from the saved folder. */
    dir: string
    /** The folder field's input, which Settings focuses when the window opens. */
    inputEl: HTMLInputElement | null
    /** The keyword index's Rebuild is out, and how it failed. */
    reindexing: boolean
    reindexError: string | null
    /** A run is being asked for or is running, so the folder and its actions are locked. */
    busy: boolean
    /** How the last run this window watched ended. */
    ingest: IngestStatus | null
    actionError: string | null
    /** The library the shell opened; `bundled` is false where the shell cannot restart one. */
    libState: LibraryState | null
    libraryError: string | null
    /** Scroll the ignore list into view (opened from a document's letter-spacing block). */
    focusSpacing: boolean
    /** Index the folder, or ask first when it would re-index. */
    onIndex: () => Promise<void>
    /** The folder changed: drop the result and error written for the old one. */
    onFolderEdit: () => void
    onOpenAnotherLibrary: () => Promise<void>
    /** Ask before switching to the app's own library. */
    onUseOwnLibrary: () => void
    onClose: () => void
  } = $props()

  // ADR-037 — the Corpus panel. `indexInfo` derives the label + the honest sentence from the
  // payload's `keyword_index.mode`; the states say materially different things, and since ADR-038
  // one of them (`unavailable`) means retrieval is running on one arm.
  const indexInfo = $derived(describeIndex(settings.corpus))

  // Non-destructive and bounded (the index is derived data the next launch would rebuild anyway),
  // so no confirmation step — inform, don't block. The response carries refreshed settings, so the
  // panel updates without a second request. Every write lands in Settings, which outlives this
  // section, so a rebuild that ends after a switch still shows; once the window has closed the
  // writes reach nothing.
  async function rebuildIndex(): Promise<void> {
    if (reindexing) return
    reindexing = true
    reindexError = null
    try {
      settings = await reindexKeywords()
    } catch (e) {
      reindexError = String(e)
    } finally {
      reindexing = false
    }
  }

  function viewIndexing(): void {
    onClose()
    openIndexing()
  }

  let picking = $state(false) // the OS folder picker is open, so Browse can't be double-fired

  /**
   * CS1 — choose the library folder with the OS picker instead of pasting a path.
   *
   * Fills the field rather than saving directly: the text input stays the override, and the user
   * still confirms with the same button they would have used after typing. `pickPaths` resolves to
   * `null` for a cancel *and* for "no picker at all", which is why the button is only rendered when
   * `canPickFiles()` — the two cases are indistinguishable here by design.
   */
  async function browseForFolder(): Promise<void> {
    if (picking) return
    picking = true
    try {
      const chosen = await pickPaths({
        directory: true,
        multiple: false,
        title: 'Choose the folder Provenote keeps your documents in',
      })
      if (chosen && chosen.length > 0) {
        dir = chosen[0]
        onFolderEdit()
      }
    } finally {
      picking = false
    }
  }

  function onInputKey(e: KeyboardEvent): void {
    if (e.key === 'Enter') {
      e.preventDefault()
      void onIndex() // self-guards on empty / busy; a Re-index asks first
    }
  }
</script>

<section>
  <h3>Your documents</h3>
  <!-- CS2 — this folder is *where Provenote keeps the documents you add*, not "the folder
       holding your documents". One sentence used to describe both models, which stopped being
       true at AD3b: documents are added to the library (copied here, or referenced where they
       live), and this folder is the copy destination. Pointing at a folder and indexing it
       wholesale still works and is still offered below — it is just no longer what the folder
       *is*. -->
  <label for="src">Library folder</label>
  <!-- CS1 — the picker is the primary route; the text field stays as an override, because a
       path typed or pasted from elsewhere is a real workflow and a picker cannot express one
       that does not exist yet (which the warning below is about). In a browser there is no
       picker, so the button is not rendered at all rather than rendered dead. -->
  <div class="srcrow">
    <input
      id="src"
      type="text"
      bind:value={dir}
      bind:this={inputEl}
      oninput={onFolderEdit}
      onkeydown={onInputKey}
      spellcheck="false"
      placeholder="C:\path\to\your\documents"
      disabled={busy}
    />
    {#if canPickFiles()}
      <button class="ghost browse" onclick={browseForFolder} disabled={busy || picking} type="button">
        <Icon name="folder" size={14} />
        {picking ? 'Choosing…' : 'Browse…'}
      </button>
    {/if}
  </div>
  <p class="hint">
    Where Provenote keeps the documents you add — anything already in it can be indexed here
    too. Supported: {settings.supported_formats}.
  </p>
  {#if settings.source_dir && !settings.source_dir_exists}
    <p class="warn"><Icon name="triangle-alert" size={14} /> The saved folder doesn't exist yet: <code>{settings.source_dir}</code></p>
  {/if}

  <button class="primary" onclick={onIndex} disabled={busy || dir.trim() === ''}>
    {#if busy}
      Indexing…
    {:else if settings.chunk_count > 0}
      Re-index
    {:else}
      Index folder
    {/if}
  </button>

  <!-- aria-live so a screen-reader user hears the index progress / completion / failure
       without re-navigating; the failure cases also assert role="alert". -->
  <div aria-live="polite">
    {#if ingestRun.status?.state === 'running'}
      <p class="muted">Indexing <code>{ingestRun.status.source_dir}</code>. This can take a
        while for large folders. You can close Settings: indexing keeps running.</p>
    {:else if ingest?.state === 'done'}
      <p class="ok"><Icon name="check" size={14} /> {ingest.message}</p>
    {:else if ingest?.state === 'stopped'}
      <p class="muted">Stopped: {ingest.message}. What finished is in your library.</p>
    {:else if ingest?.state === 'error'}
      <p class="err" role="alert">Indexing failed: {ingest.message}</p>
    {/if}
    {#if ingestRun.lostContact}
      <p class="err" role="alert">Lost contact with the indexer. It may still be running.</p>
    {/if}
    {#if actionError}
      <p class="err" role="alert">{actionError}</p>
    {/if}
    <!-- ROADMAP 31: every file's status, Pause and Stop. Closes Settings first, so one
         overlay is open at a time and Escape closes the one in front. -->
    {#if ingestRun.status?.counts}
      <button class="ghost view-run" type="button" onclick={viewIndexing}>
        View indexing
      </button>
    {/if}
  </div>
</section>

<section>
  <h3>Manage files <span class="muted">(selective indexing)</span></h3>
  <p class="hint">
    See each file's status, exclude ones you don't want, or index just a selection. Excluded
    files are skipped by <strong>Index folder</strong> above; an explicit selection here still
    indexes them.
  </p>
  <ManageFiles />
</section>

<section>
  <h3>Corpus</h3>
  <dl>
    <dt>Library</dt>
    <dd>
      {settings.corpus.documents.toLocaleString()} documents ·
      {settings.corpus.chunks.toLocaleString()} chunks
    </dd>
    <dt>Disk</dt>
    <dd>
      {formatBytes(settings.corpus.disk.total_bytes)}
      {#if perDocument(settings.corpus.disk.total_bytes, settings.corpus.documents)}
        <span class="muted"
          >· {perDocument(settings.corpus.disk.total_bytes, settings.corpus.documents)}</span
        >
      {/if}
    </dd>
    <dt>Keyword index</dt>
    <dd class="index-row">
      <span class:degraded={indexInfo.degraded}>{indexInfo.label}</span>
      {#if indexInfo.rebuildable}
        <button class="ghost" onclick={rebuildIndex} disabled={reindexing || busy}>
          {reindexing ? 'Rebuilding…' : 'Rebuild'}
        </button>
      {/if}
    </dd>
    <dt>Data home</dt>
    <dd class="path">{settings.data_home}</dd>
  </dl>
  {#if libState?.bundled}
    <div class="library-row">
      <button class="ghost" type="button" onclick={() => void onOpenAnotherLibrary()} disabled={busy}>
        <Icon name="folder" size={14} /> Open another library…
      </button>
      {#if libState.data_home}
        <button class="ghost" type="button" onclick={onUseOwnLibrary} disabled={busy}>
          Use the app's own library
        </button>
      {/if}
    </div>
    <p class="hint">
      A library indexed elsewhere on this computer opens where it is; nothing is copied.
    </p>
    {#if libraryError}
      <p class="err" role="alert">{libraryError}</p>
    {/if}
  {/if}
  <p class="banner" class:warn={indexInfo.degraded}>{indexInfo.memory}</p>
  {#if reindexError}
    <p class="banner err">{reindexError}</p>
  {/if}
</section>
<section>
  <SpacingIgnoreList focus={focusSpacing} />
</section>

<style>
  section {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  section:last-child {
    border-bottom: none;
  }
  h3 {
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
  }
  label {
    display: block;
    font-size: 0.8rem;
    color: var(--fg-2);
    margin-bottom: 0.3rem;
  }
  input {
    width: 100%;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.45rem 0.55rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
  }
  input:disabled {
    opacity: 0.6;
  }
  .srcrow {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .srcrow input {
    flex: 1;
    min-width: 0;
  }
  .browse {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    white-space: nowrap;
  }
  .hint {
    font-size: 0.76rem;
    color: var(--fg-2);
    margin: 0.4rem 0 0;
  }
  .primary {
    margin-top: 0.7rem;
    font: inherit;
    font-weight: 600;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: var(--accent-fg);
    padding: 0.4rem 1rem;
  }
  .primary:disabled {
    opacity: 0.5;
    cursor: default;
  }
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.25rem 0.8rem;
    margin: 0;
    font-size: 0.82rem;
  }
  dt {
    color: var(--fg-2);
  }
  dd {
    margin: 0;
  }
  dd.path,
  .warn code {
    font-family: ui-monospace, monospace;
    font-size: 0.76rem;
    word-break: break-all;
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
  }
  .ok {
    color: var(--ok-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
  .warn {
    color: var(--warn-fg);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 8px;
    padding: 0.4rem 0.55rem;
    font-size: 0.8rem;
    margin: 0.5rem 0 0;
  }
  .banner {
    color: var(--fg-2);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    font-size: 0.78rem;
    margin: 0 0 0.7rem;
  }
  /* Two classes on purpose: `.banner` is declared after `.warn`, so at equal specificity it would
     win and quietly neutralise the warning colours. ADR-038's degraded state must look degraded. */
  .banner.warn {
    color: var(--warn-fg);
    background: var(--warn-bg);
    border-color: var(--warn-border);
  }
  .degraded {
    color: var(--warn-fg);
    font-weight: 600;
  }
  section label {
    margin-top: 0.7rem;
  }
  section label:first-of-type {
    margin-top: 0;
  }
  /* ADR-037 — the Corpus panel's index row: label left, the bounded rebuild action right. */
  .index-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .index-row .ghost {
    margin-top: 0;
    padding: 0.2rem 0.55rem;
    font-size: 0.76rem;
    flex: none;
  }
  .library-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.6rem;
  }
  .view-run {
    margin-top: 0.5rem;
  }
  .ghost {
    margin-top: 0.8rem;
    font: inherit;
    font-size: 0.82rem;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    padding: 0.35rem 0.8rem;
  }
</style>
