<script lang="ts">
  // AD1 entry — the step between "Add documents" and files being staged.
  //
  // **Why it exists.** The button used to call the OS picker directly, so the app jumped straight
  // to Explorer. That hid the other half of AD1: the window has always accepted a dropped file,
  // but nothing on screen ever said so unless you were already mid-drag. This dialog names both
  // routes in one place, which is also the only way a first-time user learns the drop exists.
  //
  // **The drop target is still the whole window, not this box.** Tauri intercepts the OS drag
  // before the DOM sees it (`tauri://drag-drop`), so a DOM drop handler here would never fire.
  // The zone is therefore an *aiming point and a label*, and it highlights from `accept.dragging`
  // — the same window-level signal that drives App's full-window veil. Dropping anywhere works;
  // this simply tells you that.
  //
  // Once anything is staged, `stagePaths` closes this dialog and the review sheet (AD2) takes over.

  import Icon from '../shell/Icon.svelte'
  import { scanZoteroLibrary } from '../core/api'
  import { pickPaths } from '../core/tauri'
  import { accept, closeChooser, pickDocuments, stagePaths } from './accept.svelte'
  import { nothingToAdd } from './catalogue'

  /** Mirrors the server-side format list in words; the verdicts themselves come from AD2. */
  const FORMATS = 'PDF · EPUB · HTML · DOCX · MD · ODT · RTF'

  // --- Import from Zotero (ROADMAP 17) --------------------------------------------------------
  //
  // A third *route to paths*, not a third add path. The scan hands back absolute paths and this
  // stages them, so the review sheet, the duplicate check and the copy-or-reference choice are
  // the ones the user already knows — and re-importing a library they have already imported is
  // simply an add where everything reads as a duplicate.
  //
  // The default folder is tried first because it is right for almost everyone; the folder picker
  // only appears once that has actually failed, so the common case is one click.
  let zoteroBusy = $state(false)
  let zoteroError = $state<string | null>(null)
  let zoteroNote = $state<string | null>(null)
  let askedForFolder = $state(false)

  async function importFromZotero(dataDir: string | null = null): Promise<void> {
    if (zoteroBusy) return
    zoteroBusy = true
    zoteroError = null
    zoteroNote = null
    try {
      const scan = dataDir === null ? await scanZoteroLibrary() : await scanZoteroLibrary(dataDir)
      if (scan.paths.length > 0) {
        // The catalogue's own storage folder, so a reference-add registers one root rather than
        // one per attachment. The adapter reports it precisely because it knows and we do not.
        stagePaths(scan.paths, scan.root)
        return
      }
      // Found the library, staged nothing. Say what was in it rather than "0 documents", which
      // reads as a broken import when the reasons are all perfectly ordinary.
      zoteroNote = nothingToAdd(scan.skipped)
    } catch (e) {
      zoteroError = e instanceof Error ? e.message : String(e)
      askedForFolder = true
    } finally {
      zoteroBusy = false
    }
  }

  async function chooseZoteroFolder(): Promise<void> {
    const chosen = await pickPaths({
      directory: true,
      multiple: false,
      title: 'Choose your Zotero data folder',
    })
    if (chosen && chosen.length > 0) await importFromZotero(chosen[0])
  }

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') closeChooser()
  }

  // Move focus into the dialog on open, so Escape and Tab act on it rather than on whatever was
  // focused behind the scrim. `tabindex="-1"` makes the container focusable without putting it in
  // the tab order itself.
  let dialogEl: HTMLDivElement | null = $state(null)
  $effect(() => {
    dialogEl?.focus()
  })
</script>

<svelte:window on:keydown={onKeydown} />

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
<div class="scrim" role="presentation" onclick={closeChooser}>
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
  <div
    class="dialog"
    role="dialog"
    aria-modal="true"
    aria-labelledby="addchooser-title"
    tabindex="-1"
    bind:this={dialogEl}
    onclick={(e) => e.stopPropagation()}
  >
    <div class="head">
      <h2 id="addchooser-title">Add documents</h2>
      <button class="x" onclick={closeChooser} aria-label="Close" type="button">
        <Icon name="x" size={16} />
      </button>
    </div>

    <div class="zone" class:over={accept.dragging}>
      <Icon name="file-text" size={26} />
      <strong>{accept.dragging ? 'Drop them now' : 'Drag files or folders here'}</strong>
      <span class="formats">{FORMATS}</span>
    </div>

    <p class="or"><span>or</span></p>

    <div class="routes">
      <button
        class="route"
        onclick={() => void pickDocuments()}
        disabled={accept.picking}
        type="button"
      >
        <Icon name="file-text" size={15} />
        Choose files…
      </button>
      <button
        class="route"
        onclick={() => void pickDocuments({ directory: true })}
        disabled={accept.picking}
        type="button"
      >
        <Icon name="folder" size={15} />
        Choose a folder…
      </button>
      <button
        class="route"
        onclick={() => void importFromZotero()}
        disabled={zoteroBusy || accept.picking}
        type="button"
      >
        <Icon name="library" size={15} />
        {zoteroBusy ? 'Reading your library…' : 'Import from Zotero…'}
      </button>
    </div>

    {#if zoteroError}
      <!-- Not having Zotero, or keeping it elsewhere, is an ordinary state of the world. The
           server's message is already a sentence written for a person, so it is shown verbatim
           and followed by the one action that can fix it. -->
      <p class="zotero err" role="alert">
        {zoteroError}
        {#if askedForFolder}
          <button class="inline" onclick={() => void chooseZoteroFolder()} type="button">
            Choose the folder…
          </button>
        {/if}
      </p>
    {:else if zoteroNote}
      <p class="zotero">{zoteroNote}</p>
    {/if}

    <!-- Said before anything is chosen, because it is the answer to "what is about to happen to my
         files?" — the placement choice itself lives in the review sheet, where it can be changed. -->
    <p class="note">
      Nothing is copied or indexed yet. You will see what would happen to each file, and choose
      whether to copy them or leave them where they are, before anything is applied.
    </p>
  </div>
</div>

<style>
  .zotero {
    margin: 0.55rem 0 0;
    font-size: 0.76rem;
    line-height: 1.45;
    color: var(--fg-2);
  }
  .zotero.err {
    color: var(--danger, #c0392b);
  }
  .inline {
    font: inherit;
    color: var(--accent);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    text-decoration: underline;
  }
  .scrim {
    position: fixed;
    inset: 0;
    z-index: 60;
    display: grid;
    place-items: center;
    background: rgb(0 0 0 / 0.45);
    padding: 1rem;
  }
  .dialog {
    width: min(30rem, 100%);
    outline: none; /* focused programmatically for Escape/Tab, not as a visible affordance */
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius, 10px);
    box-shadow: var(--shadow-lg, 0 18px 50px rgb(0 0 0 / 0.4));
    padding: 0.9rem 1rem 1rem;
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.7rem;
  }
  h2 {
    margin: 0;
    font-size: 1rem;
    color: var(--fg);
  }
  .x {
    background: none;
    border: none;
    color: var(--fg-2);
    cursor: pointer;
    padding: 0.2rem;
    border-radius: var(--radius-sm, 4px);
  }
  .x:hover {
    background: var(--bg-2);
    color: var(--fg);
  }

  .zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    padding: 1.5rem 1rem;
    border: 1.5px dashed var(--border);
    border-radius: var(--radius, 10px);
    color: var(--fg-2);
    text-align: center;
    transition: border-color 120ms ease, background 120ms ease;
  }
  .zone.over {
    border-color: var(--accent, #7c6cf0);
    background: color-mix(in srgb, var(--accent, #7c6cf0) 10%, transparent);
    color: var(--fg);
  }
  .zone strong {
    color: var(--fg);
    font-size: 0.95rem;
  }
  .formats {
    font-size: var(--text-meta);
  }

  /* A rule with the word sitting on it — the two routes are alternatives, not a sequence. */
  .or {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.8rem 0;
    color: var(--fg-2);
    font-size: var(--text-meta);
  }
  .or::before,
  .or::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .routes {
    display: flex;
    gap: 0.5rem;
  }
  .route {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    padding: 0.5rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm, 6px);
    background: var(--bg-2);
    color: var(--fg);
    font-size: 0.9rem;
    cursor: pointer;
  }
  .route:hover:not(:disabled) {
    border-color: var(--accent, #7c6cf0);
  }
  .route:disabled {
    opacity: 0.55;
    cursor: default;
  }

  .note {
    margin: 0.8rem 0 0;
    font-size: var(--text-meta);
    color: var(--fg-2);
    line-height: 1.45;
  }

  @media (prefers-reduced-motion: reduce) {
    .zone {
      transition: none;
    }
  }
</style>
