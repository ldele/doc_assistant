<script lang="ts">
  // Settings: the drawer, its rail and the section it shows. Each of the five sections is a file
  // beside this one (GettingStarted, Documents, ProviderAndModel, Retrieval, General); this one
  // keeps what spans them: the window (scrim, Tab trap, Escape), `/api/settings` and the update
  // status it loads, the run Documents starts and the dialogs it opens (mounted here, outside the
  // panel), and the state a section shows between visits.
  import type { Settings, IngestStatus, RagOverrides, UpdateStatus } from '../../lib/core/types'
  import {
    getSettings,
    setSourceDir,
    startIngest,
    startIngestNewOnly,
    getUpdateStatus,
  } from '../../lib/core/api'
  import { onDestroy, tick } from 'svelte'
  import Icon from '../../shell/Icon.svelte'
  import {
    libraryState,
    markLibrarySwitch,
    openLibrary,
    pickPaths,
    type LibraryState,
  } from '../../lib/core/tauri'
  import ConfirmDialog from '../../shell/ConfirmDialog.svelte'
  import Dialog from '../../shell/Dialog.svelte'
  import { indexLocked, ingestRun, noteIngestStarted, useRunEnd } from '../../lib/core/ingest.svelte'
  import GettingStarted from './GettingStarted.svelte'
  import Documents from './Documents.svelte'
  import ProviderAndModel from './ProviderAndModel.svelte'
  import Retrieval from './Retrieval.svelte'
  import General from './General.svelte'
  import ReindexFolder from './ReindexFolder.svelte'
  import OpenLibrary from './OpenLibrary.svelte'
  import { SETTINGS_SECTIONS, initialSection, sectionBadge, type SettingsSectionId } from '../../lib/settings/sections'
  import { outstandingSteps } from '../../lib/settings/setup'
  import { shell } from '../../lib/shell/shell.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  // `onCorpusChanged` re-reads /api/health after a provider switch. The end of an ingest reaches App
  // through `ingestRun.completedRuns`, which also holds when this panel has been closed.
  // `overrides` is bindable — the RAG sandbox (Retrieval) mutates it directly; the session-scoped
  // state itself lives in App.svelte (ADR-010: in-memory only, cleared on restart).
  let {
    onClose,
    onCorpusChanged,
    overrides = $bindable(),
  }: {
    onClose: () => void
    onCorpusChanged: () => void
    overrides: RagOverrides
  } = $props()

  // Which category the drawer shows (2026-08-30). The panel is a rail + one pane rather than one
  // flat scroll: see `./sections.ts` for why, and for the list itself.
  //
  // Seeded from `shell.setup`, which App has *already* loaded for the chat pane's setup banner —
  // so a fresh install opens on the checklist instead of opening on Documents and jumping there a
  // moment later when a fetch of our own lands. Reading it also costs no second request.
  const setupStepsLeft = $derived(outstandingSteps(shell.setup).length)
  let activeSection = $state<SettingsSectionId>(
    SETTINGS_SECTIONS.some((s) => s.id === shell.settingsSection)
      ? (shell.settingsSection as SettingsSectionId)
      : initialSection(outstandingSteps(shell.setup).length, ingestRun.status?.state === 'running'),
  )
  shell.settingsSection = null
  const focusSpacing = shell.settingsFocus === 'spacing'
  shell.settingsFocus = null
  const activeBlurb = $derived(
    SETTINGS_SECTIONS.find((s) => s.id === activeSection)?.blurb ?? '',
  )

  // Switching category: bring the new rail item into view, and start its pane at the top.
  //
  // Both only bite once the rail outgrows its box — vertically when the category list is longer
  // than the panel, horizontally in the narrow layout where the rail is a scrolling strip. That is
  // the "more settings later" case this restructure exists for, so it is wired now rather than
  // discovered later. Without the second half, a pane scrolled deep in Documents opens General
  // halfway down a page the user has never seen.
  let railEl = $state<HTMLElement | null>(null)
  let paneEl = $state<HTMLElement | null>(null)
  $effect(() => {
    const current = activeSection // the dependency: re-run on every category change
    if (!current) return
    railEl?.querySelector('[aria-current="page"]')?.scrollIntoView({
      block: 'nearest',
      inline: 'nearest',
    })
    if (paneEl) paneEl.scrollTop = 0
  })

  // A Getting started row's link goes away with the page it sits on, and focus would leave the
  // dialog with it.
  async function openSection(id: SettingsSectionId): Promise<void> {
    activeSection = id
    await tick()
    if (!panelEl?.contains(document.activeElement)) panelEl?.focus()
  }

  let settings = $state<Settings | null>(null)
  let loadError = $state<string | null>(null)

  // What a section shows between visits. The rail mounts one section at a time and unmounts the
  // one it leaves, so anything that must still be there when the person comes back (a request in
  // flight, its error line) is held here for as long as the window is open, and the section that
  // shows it writes it back through `bind:`.
  //
  // Documents › Corpus: the keyword index's Rebuild (ADR-037).
  let reindexing = $state(false)
  let reindexError = $state<string | null>(null)
  // Provider & model › Answer epistemics, the saved default (ADR-027 D2).
  let markersBusy = $state(false)
  let markersError = $state<string | null>(null)
  // General › Chat history.
  let exporting = $state(false)
  let exportError = $state<string | null>(null)
  // General › Updates (ADR-044). `update` is null until the first read; the section renders a
  // neutral line rather than guessing a state, because guessing "up to date" is exactly the
  // failure this feature must not have.
  let update = $state<UpdateStatus | null>(null)
  let updateBusy = $state(false)
  let updateError = $state<string | null>(null)

  // Documents › Your documents: the folder field and the run it starts.
  let dir = $state('')
  let requesting = $state(false) // the folder is being saved and the run asked for
  // The run lives on the server and `ingestRun` follows it, so this panel holds none of it: closing
  // loses nothing, and reopening mid-run shows the same run. A run can take hours; nothing here may
  // keep the panel open for it.
  const busy = $derived(indexLocked(requesting, ingestRun.status))
  let actionError = $state<string | null>(null)
  let ingest = $state<IngestStatus | null>(null) // how the last run this panel watched ended
  let inputEl = $state<HTMLInputElement | null>(null)
  let panelEl = $state<HTMLDivElement | null>(null)

  let cancelled = false
  onDestroy(() => {
    cancelled = true
  })

  void load()

  useRunEnd((s) => {
    ingest = s
    void load(true) // refresh this panel's chunk_count; a blip here stays non-fatal
  })

  // Move focus into the dialog when it opens (the input, in Documents, mounts once settings load)
  // so a keyboard / screen-reader user lands in the panel, not on the gear button behind the scrim.
  // Opened mid-run the input is disabled and refuses focus, so the panel takes it.
  $effect(() => {
    if (inputEl?.disabled) panelEl?.focus()
    else inputEl?.focus()
  })

  // `silent` post-ingest refreshes don't raise the fatal panel-level loadError: a transient blip
  // on the refresh must not collapse the panel + erase the "✓ indexed" confirmation (the header
  // was already updated via onCorpusChanged). Default-valued param, not optional — Svelte 5's TS
  // strip rejects `?:` here.
  async function load(silent = false): Promise<void> {
    try {
      const s = await getSettings()
      if (cancelled) return
      settings = s
      if (!dir) dir = s.source_dir
    } catch (e) {
      if (!silent) loadError = String(e)
    }
    // ADR-044: free in the default configuration — this reads the cached answer and only
    // touches the network if the user opted in and a check is due. Deliberately after the
    // settings load and independently caught: the panel must open even if it fails.
    try {
      const u = await getUpdateStatus()
      if (!cancelled) update = u
    } catch {
      // Leave `update` null — the section says "version unknown until the first check"
      // rather than inventing a state. No red error for a background read nobody asked for.
    }
  }

  // Re-index asks first (the user, 2026-11-04): how many files it reads, roughly how long, and
  // whether to take only the new ones. A first index has nothing indexed to skip and no measured
  // speed, so it starts at once, as before.
  let reindexOpen = $state(false)

  async function onIndexClick(): Promise<void> {
    const target = dir.trim()
    if (!target || busy || !settings) return
    if (settings.chunk_count === 0) {
      void indexFolder()
      return
    }
    // The plan counts the saved folder, so a folder typed but not saved is saved first.
    if (target !== settings.source_dir) {
      try {
        settings = await setSourceDir(target)
      } catch (e) {
        actionError = String(e)
        return
      }
    }
    reindexOpen = true
  }

  function confirmReindex(onlyNew: boolean): void {
    reindexOpen = false
    void indexFolder(onlyNew)
  }

  // One low-friction action: validate+persist the folder, then re-index it. Saving first
  // means a bad path surfaces the backend's 400 before any ingest starts (inform-don't-corrupt).
  async function indexFolder(onlyNew = false): Promise<void> {
    const target = dir.trim()
    if (!target || busy) return
    requesting = true
    actionError = null
    ingest = null
    try {
      settings = await setSourceDir(target)
      const started = onlyNew ? await startIngestNewOnly() : await startIngest()
      // Without this a short run can end between two idle polls: the watcher never sees it running
      // and no completion fires, which nothing else covers once this panel has closed.
      noteIngestStarted(started.source_dir)
    } catch (e) {
      actionError = String(e)
    } finally {
      requesting = false
    }
  }

  // Feedback always refers to the path it was produced for — drop a stale ✓ / error once the
  // user edits the folder, so it never sits next to a path it no longer describes.
  function clearFeedback(): void {
    ingest = null
    actionError = null
  }

  // ADR-061: another library folder opens where it is. The shell restarts the backend on it, so
  // the control exists only where there is a shell with a backend to restart (choice 9).
  let libState = $state<LibraryState | null>(null)
  void libraryState().then((s) => (libState = s))
  let libraryToOpen = $state<string | null>(null)
  let confirmOwnLibrary = $state(false)
  let libraryError = $state<string | null>(null)

  async function chooseLibrary(): Promise<void> {
    libraryError = null
    const picked = await pickPaths({
      directory: true,
      multiple: false,
      title: 'Choose a library folder (the one holding library.db)',
    })
    if (picked && picked[0]) libraryToOpen = picked[0]
  }

  async function useOwnLibrary(): Promise<void> {
    confirmOwnLibrary = false
    try {
      markLibrarySwitch(null)
      await openLibrary(null)
      window.location.reload()
    } catch (e) {
      libraryError = String(e)
    }
  }
</script>

<!-- A side panel on the shared frame (shell/Dialog.svelte): it slides in (instantly when the system
     asks for reduced motion), Esc closes it, and Tab stays inside. The questions it opens below
     (Re-index, Open library, the own-library confirm) are dialogs of their own, so Esc answers
     them first. Wider than the 420px it was: the rail costs a column, and the controls it navigates
     to were already cramped in one. Still a drawer, not a page: `96vw` keeps the backdrop reachable,
     which is how most people close this. -->
<Dialog shape="side" label="Settings" width="min(52rem, 96vw)" slide={760} bind:frame={panelEl} {onClose}>
  <header>
    <strong>Settings</strong>
    <button class="x" onclick={onClose} aria-label="Close" use:tooltip={'Close Settings'}>
      <Icon name="x" />
    </button>
  </header>

  {#if loadError}
    <p class="err">Couldn't load settings: {loadError}</p>
  {:else if !settings}
    <p class="muted">Loading…</p>
  {:else}
    <div class="body">
      <!-- Navigation, deliberately NOT `role="tablist"`. Tabs come with a keyboard contract —
           roving tabindex, arrow-key movement — and a rail that claims the role without honouring
           it is worse for a screen-reader user than one that never claimed it. As plain nav
           buttons every category is reachable with Tab, which is what actually happens here. The
           rail is also the panel's heading: the current item names the page, so no category
           repeats its own title inside the pane. -->
      <nav class="rail" aria-label="Settings categories" bind:this={railEl}>
        {#each SETTINGS_SECTIONS as s (s.id)}
          {@const badge = sectionBadge(s.id, setupStepsLeft)}
          <button
            type="button"
            aria-current={activeSection === s.id ? 'page' : undefined}
            class:active={activeSection === s.id}
            onclick={() => (activeSection = s.id)}
            use:tooltip={s.blurb}
          >
            <Icon name={s.icon} size={15} />
            <span class="rl">{s.label}</span>
            {#if badge}
              <span class="badge" aria-label="{badge} setup steps left">{badge}</span>
            {/if}
          </button>
        {/each}
      </nav>

      <!-- aria-live on the blurb, not the pane: announcing a whole category on every rail click
           would talk over the user, while the lead line is exactly "which page am I on now". -->
      <div class="pane" bind:this={paneEl}>
        <p class="blurb" aria-live="polite">{activeBlurb}</p>

        {#if activeSection === 'setup'}
          <!-- First (ADR-034): on a fresh install this is the only section that matters, and it says
               so — a model and documents, each one link from the tab that sets it up. -->
          <GettingStarted onOpen={openSection} />
        {:else if activeSection === 'documents'}
          <Documents
            bind:settings
            bind:dir
            bind:inputEl
            bind:reindexing
            bind:reindexError
            {busy}
            {ingest}
            {actionError}
            {libState}
            {libraryError}
            {focusSpacing}
            onIndex={onIndexClick}
            onFolderEdit={clearFeedback}
            onOpenAnotherLibrary={chooseLibrary}
            onUseOwnLibrary={() => (confirmOwnLibrary = true)}
            {onClose}
          />
        {:else if activeSection === 'models'}
          <!-- `load(true)` keeps a switch made in there from leaving the Retrieval tab's Engine
               list stale. A switch changes the model the status bar names, so its health is re-read
               too (onCorpusChanged → App's refreshHealth); it used to keep the old model. -->
          <ProviderAndModel
            bind:settings
            bind:markersBusy
            bind:markersError
            onProviderChanged={() => {
              void load(true)
              onCorpusChanged()
            }}
          />
        {:else if activeSection === 'answers'}
          <Retrieval {settings} bind:overrides />
        {:else}
          <General bind:update bind:updateBusy bind:updateError bind:exporting bind:exportError />
        {/if}
      </div>
    </div>
  {/if}
</Dialog>

{#if libraryToOpen}
  <OpenLibrary folder={libraryToOpen} onClose={() => (libraryToOpen = null)} />
{/if}

{#if confirmOwnLibrary}
  <ConfirmDialog
    title="Switch to the app's own library?"
    body="The app restarts its engine on its own library. This library stays as it is, and Open another library… brings it back."
    confirmLabel="Switch"
    tone="default"
    onConfirm={() => void useOwnLibrary()}
    onClose={() => (confirmOwnLibrary = false)}
  />
{/if}

{#if reindexOpen && settings}
  <ReindexFolder
    folder={settings.source_dir}
    onConfirm={confirmReindex}
    onClose={() => (reindexOpen = false)}
  />
{/if}

<style>
  header {
    flex: none;
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.4rem;
  }
  .body {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 13rem 1fr;
    gap: 1.2rem;
  }
  .rail {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: 0.5rem 0.6rem 0.5rem 0;
    border-right: 1px solid var(--border);
    overflow-y: auto;
  }
  .rail button {
    font: inherit;
    font-size: 0.85rem;
    text-align: left;
    cursor: pointer;
    border: 1px solid transparent;
    background: none;
    color: var(--fg-2);
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .rail button:hover {
    color: var(--fg);
    background: var(--surface);
  }
  .rail button.active {
    color: var(--fg);
    background: var(--surface);
    border-color: var(--border);
    font-weight: 600;
  }
  .rail .rl {
    flex: 1;
    min-width: 0;
  }
  .badge {
    flex: none;
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1;
    padding: 0.2rem 0.4rem;
    border-radius: 999px;
    background: var(--accent);
    color: var(--bg);
  }
  .pane {
    min-width: 0;
    overflow-y: auto;
    padding-bottom: 1rem;
  }
  .blurb {
    margin: 0.5rem 0 0.2rem;
    font-size: 0.8rem;
    color: var(--fg-2);
    max-width: 46rem;
  }
  /* Narrow: the rail becomes a scrollable strip above the pane. Same buttons, same order — it
     stops being a column, not a navigation. */
  @media (max-width: 700px) {
    .body {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr;
      gap: 0.4rem;
    }
    .rail {
      flex-direction: row;
      overflow-x: auto;
      padding: 0.3rem 0;
      border-right: none;
      border-bottom: 1px solid var(--border);
    }
    .rail button {
      flex: none;
    }
  }
  .x {
    font: inherit;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg-2);
    padding: 0.2rem 0.4rem;
    display: inline-flex;
    align-items: center;
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
</style>
