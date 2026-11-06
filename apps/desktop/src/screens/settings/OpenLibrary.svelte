<script lang="ts">
  // The check before the app opens another library folder (ADR-061 choices 3, 5, 6): what the
  // folder holds, read by the backend without writing, and *Open* as the person's call. Opening
  // asks the shell to restart the backend on that folder, then reloads the window; the start-up
  // gate waits for the new backend, and offers *Go back* if it never answers (choice 7).
  import { inspectLibrary } from '../../lib/core/api/libraries'
  import Dialog from '../../shell/Dialog.svelte'
  import { markLibrarySwitch, openLibrary } from '../../lib/core/tauri'
  import type { LibraryInspection } from '../../lib/core/types'
  import { inspectionSummary, sourcesLine } from '../../lib/settings/library'
  import { tooltip } from '../../lib/shell/tooltip'

  let { folder, onClose }: { folder: string; onClose: () => void } = $props()

  let found = $state<LibraryInspection | null>(null)
  let loadError = $state<string | null>(null)
  let opening = $state(false)
  let openError = $state<string | null>(null)

  $effect(() => {
    inspectLibrary(folder)
      .then((f) => (found = f))
      .catch((e: unknown) => (loadError = String(e)))
  })

  async function open(): Promise<void> {
    if (!found?.can_open || opening) return
    opening = true
    openError = null
    try {
      markLibrarySwitch(found.path)
      await openLibrary(found.path)
      window.location.reload()
    } catch (e) {
      openError = String(e)
      opening = false
    }
  }
</script>

<Dialog labelledby="open-library-title" width="34rem" locked={opening} {onClose}>
  <h2 class="dialog-title" id="open-library-title">Open this library?</h2>
  <p class="folder mono" title={folder}>{folder}</p>

  {#if loadError}
    <p class="err" role="alert">Couldn't read the folder: {loadError}</p>
  {:else if !found}
    <p class="muted">Reading what it holds…</p>
  {:else}
    {#if inspectionSummary(found)}
      <p class="summary">{inspectionSummary(found)}</p>
    {/if}
    {#if sourcesLine(found)}
      <p class="muted">{sourcesLine(found)}</p>
    {/if}
    {#each found.problems as problem (problem)}
      <p class="err" role="alert">{problem}</p>
    {/each}
    {#each found.notes as note (note)}
      <p class="note">{note}</p>
    {/each}
    {#if found.can_open}
      <p class="muted small">
        From now on the app works in this folder: documents you add, its settings and an update of
        its database are written there. Use it from one copy of the app at a time. Settings ›
        Documents switches back whenever you like.
      </p>
    {/if}
  {/if}
  {#if openError}
    <p class="err" role="alert">{openError}</p>
  {/if}

  <div class="dialog-actions">
    <button class="btn" onclick={onClose} type="button" disabled={opening} use:tooltip={'Stay in this library'}>Cancel</button>
    <button
      class="btn btn-primary"
      type="button"
      disabled={!found?.can_open || opening}
      onclick={() => void open()}
      use:tooltip={!found?.can_open ? 'Choose a folder that holds a library first' : 'Switch to this library; the app reloads'}
    >
      {opening ? 'Opening…' : 'Open this library'}
    </button>
  </div>
</Dialog>

<style>
  p {
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.55;
  }
  .folder {
    font-size: var(--text-meta);
    color: var(--fg-2);
    overflow-wrap: anywhere;
  }
  .summary {
    font-weight: 600;
  }
  .note {
    padding: 0.45rem 0.6rem;
    border-radius: 8px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-fg);
  }
  .small {
    font-size: var(--text-meta);
  }
  .err {
    color: var(--danger);
  }
  .mono {
    font-family: var(--font-mono, ui-monospace, monospace);
  }
</style>
