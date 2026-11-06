<script lang="ts">
  // The indexing screen (ROADMAP 31; wireframe `docs/wireframes/indexing.md`, approved at rev 858):
  // every file of the current or last run with its status and what happened, Pause and Stop while it
  // runs, and the follow-up runs once it has ended.
  //
  // A renderer of `ingestRun` and of GET /api/ingest/files, nothing more. The run lives on the server:
  // closing this screen loses nothing, and the status bar keeps following the run.
  //
  // **The list is paged.** At 10,000 documents the status poll carries counts only (the filters'
  // numbers); the rows come from the files route, a page at a time, refreshed on each poll while
  // the screen is open.
  import { fade } from 'svelte/transition'

  import { getIngestFiles } from '../lib/core/api/sources'
  import {
    controlRun,
    fraction,
    groupChips,
    ingestRun,
    rerunLabel,
    rerunLast,
    runBanner,
    runFooter,
    runTitle,
  } from '../lib/core/ingest.svelte'
  import { FILE_STATE_LABEL, READING_DETAIL } from '../lib/core/ingest'
  import type { IngestFile, IngestGroup } from '../lib/core/types'
  import ConfirmDialog from './ConfirmDialog.svelte'
  import Icon from './Icon.svelte'
  import { tooltip } from '../lib/shell/tooltip'

  const animate =
    typeof window === 'undefined' || !window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const DUR = animate ? 150 : 0
  /** Rows per page. A screen holds a few dozen; this many keeps scrolling smooth without a pager. */
  const PAGE = 200

  const status = $derived(ingestRun.status)
  const running = $derived(status?.state === 'running')
  const control = $derived(status?.control ?? 'none')
  const chips = $derived(groupChips(status))
  const pct = $derived(fraction(status))
  const banner = $derived(runBanner(status))
  const remaining = $derived(rerunLabel('remaining', status))
  const retry = $derived(rerunLabel('failed', status))

  let group = $state<IngestGroup>('all')
  let limit = $state(PAGE)
  let rows = $state<IngestFile[]>([])
  let inGroup = $state(0)
  let listError = $state<string | null>(null)
  let actionError = $state<string | null>(null)
  let acting = $state(false)
  let confirmingStop = $state(false)
  let dialogEl = $state<HTMLDivElement | null>(null)

  /** What a stop keeps and what it leaves, in numbers, for the confirmation (the user, 2026-11-04). */
  const stopBody = $derived.by(() => {
    const c = status?.counts
    if (!c) return 'Indexing ends after the document in progress. What finished stays in your library.'
    const kept = c.added + c.unchanged
    const left = c.waiting + c.reading
    return (
      `Indexing ends after the document in progress. The ${kept.toLocaleString()} finished ` +
      `stay in your library and the next run skips them. The ${left.toLocaleString()} not yet ` +
      `indexed can be indexed later with "Index the remaining".`
    )
  })

  // A filter the run no longer offers (In progress, once it has stopped) falls back to All.
  $effect(() => {
    if (chips.length > 0 && !chips.some((c) => c.group === group)) group = 'all'
  })

  // Refetch on every poll while the screen is open, and on a filter or page change. A sequence
  // number drops an answer that arrives after a newer request, so a slow page never overwrites a
  // fresh one.
  let seq = 0
  $effect(() => {
    void status // re-run on each poll
    const g = group
    const n = limit
    const mine = ++seq
    getIngestFiles(g, 0, n)
      .then((page) => {
        if (mine !== seq) return
        rows = page.files
        inGroup = page.total
        listError = null
      })
      .catch((e: unknown) => {
        if (mine === seq) listError = String(e)
      })
  })

  $effect(() => {
    dialogEl?.focus()
  })

  function close(): void {
    ingestRun.panelOpen = false
  }

  async function act(run: () => Promise<void>): Promise<void> {
    if (acting) return
    acting = true
    actionError = null
    try {
      await run()
    } catch (e) {
      actionError = String(e)
    } finally {
      acting = false
    }
  }

  function chooseGroup(g: IngestGroup): void {
    group = g
    limit = PAGE
  }

  function onWindowKey(e: KeyboardEvent): void {
    // The stop confirmation handles its own Escape: it closes the question, not this screen.
    if (e.key === 'Escape' && !confirmingStop) close()
  }

  function confirmStop(): void {
    confirmingStop = false
    void act(() => controlRun('stop'))
  }

  // Honour aria-modal: keep Tab inside the dialog (the same rule Settings keeps).
  function onDialogKey(e: KeyboardEvent): void {
    if (e.key !== 'Tab' || !dialogEl) return
    const focusable = Array.from(
      dialogEl.querySelectorAll('button:not([disabled]), [tabindex]:not([tabindex="-1"])'),
    ) as HTMLElement[]
    if (focusable.length === 0) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault()
      first.focus()
    }
  }
</script>

<svelte:window onkeydown={onWindowKey} />

<div class="scrim" onclick={close} role="presentation" transition:fade={{ duration: DUR }}></div>

<div
  class="screen"
  role="dialog"
  aria-modal="true"
  aria-labelledby="indexing-title"
  tabindex="-1"
  bind:this={dialogEl}
  onkeydown={onDialogKey}
  transition:fade={{ duration: DUR }}
>
  <header>
    <div class="titles">
      <h2 id="indexing-title" aria-live="polite">{runTitle(status)}</h2>
      {#if status?.source_dir}
        <p class="sub mono" title={status.source_dir}>From {status.source_dir}</p>
      {/if}
    </div>
    <div class="actions">
      {#if running}
        <!-- Action roles (ADR-063): Resume carries the run on, so it is the primary; Pause holds it
             and removes nothing, so it is ordinary (it was amber before the roles). -->
        {#if control === 'paused'}
          <button class="btn btn-primary" onclick={() => act(() => controlRun('resume'))} disabled={acting} use:tooltip={'Carry the run on from where it paused'}>
            <Icon name="play" size={14} /> Resume
          </button>
        {:else}
          <button
            class="btn"
            onclick={() => act(() => controlRun('pause'))}
            disabled={acting || control !== 'none'}
            use:tooltip={'Hold the run after the file in progress; Resume carries it on'}
          >
            <Icon name="pause" size={14} /> {control === 'pausing' ? 'Pausing…' : 'Pause'}
          </button>
        {/if}
        <!-- Red, and confirmed (the user, 2026-11-04, over the wireframe's "no confirmation"): a
             stop loses nothing finished, but it does end the run. The "…" says a confirm follows. -->
        <button
          class="btn btn-danger"
          onclick={() => (confirmingStop = true)}
          disabled={acting || control === 'stopping' || control === 'finishing'}
          use:tooltip={'End the run, after you confirm; what finished stays in your library'}
        >
          <Icon name="square" size={14} /> {control === 'stopping' ? 'Stopping…' : 'Stop…'}
        </button>
      {:else}
        {#if remaining}
          <button class="btn btn-primary" onclick={() => act(() => rerunLast('remaining'))} disabled={acting} use:tooltip={'Index the files this run did not reach'}>
            {remaining}
          </button>
        {/if}
        {#if retry}
          <button
            class="btn"
            class:btn-primary={!remaining}
            onclick={() => act(() => rerunLast('failed'))}
            disabled={acting}
            use:tooltip={'Try the files that failed again'}
          >
            {retry}
          </button>
        {/if}
      {/if}
      <button class="btn" onclick={close} use:tooltip={running ? 'Close this screen; the run carries on' : 'Close this screen'}>Close</button>
    </div>
  </header>

  {#if status && status.total > 0 && (running || status.state === 'stopped')}
    <div
      class="bar"
      role="progressbar"
      aria-label="Documents done"
      aria-valuemin="0"
      aria-valuemax={status.total}
      aria-valuenow={status.done}
    >
      <span class="fill" class:paused={control === 'paused'} style={`width: ${Math.round((pct ?? 0) * 100)}%`}></span>
    </div>
  {/if}

  {#if banner}
    <p class="banner" class:warn={control === 'paused' || control === 'pausing'}>{banner}</p>
  {/if}
  {#if status?.state === 'error'}
    <p class="banner err" role="alert">The indexer stopped with an error: {status.message ?? 'no reason given.'}</p>
  {/if}
  {#if ingestRun.lostContact}
    <p class="banner err" role="alert">Lost contact with the indexer. It may still be running; reopen the app to check.</p>
  {/if}
  {#if actionError}
    <p class="banner err" role="alert">{actionError}</p>
  {/if}

  {#if chips.length > 0}
    <div class="filters">
      <div class="chips" role="group" aria-label="Show files">
        {#each chips as chip (chip.group)}
          <button
            class="chip"
            class:on={group === chip.group}
            aria-pressed={group === chip.group}
            onclick={() => chooseGroup(chip.group)}
            use:tooltip={'Show only these files in the list'}
          >
            {chip.label} <span class="n">{chip.count.toLocaleString()}</span>
          </button>
        {/each}
      </div>
      {#if running && control === 'paused'}
        <p class="when">Quitting the app ends the run. What is finished stays.</p>
      {:else if running}
        <p class="when">Pause and Stop take effect when the document in progress finishes.</p>
      {/if}
    </div>

    <div class="list" role="table" aria-label="Files in this run">
      <div class="row head" role="row">
        <span role="columnheader">File</span>
        <span role="columnheader">Status</span>
        <span role="columnheader">What happened</span>
      </div>
      {#each rows as f (f.path)}
        <div class="row" role="row">
          <span class="name" role="cell" title={f.path}>{f.name}</span>
          <span role="cell"><span class="pill {f.state}">{FILE_STATE_LABEL[f.state]}</span></span>
          <span class="detail" class:err={f.state === 'failed'} role="cell">
            {f.state === 'reading' ? READING_DETAIL : (f.detail ?? '')}
          </span>
        </div>
      {:else}
        <p class="empty">{listError ? `Couldn't load the files: ${listError}` : 'No files here.'}</p>
      {/each}
      {#if rows.length < inGroup}
        <button class="btn more" onclick={() => (limit += PAGE)} use:tooltip={'List more files'}>
          Show {Math.min(PAGE, inGroup - rows.length).toLocaleString()} more of {inGroup.toLocaleString()}
        </button>
      {/if}
    </div>
  {:else if !running}
    <p class="empty">
      No indexing run since the app started. Settings › Documents indexes a folder; Library › Add
      documents indexes what you add.
    </p>
  {:else}
    <p class="empty">Counting the files…</p>
  {/if}

  <footer>{runFooter(status)}</footer>
</div>

{#if confirmingStop}
  <ConfirmDialog
    title="Stop indexing?"
    body={stopBody}
    confirmLabel="Stop indexing"
    cancelLabel="Keep indexing"
    confirmIcon="square"
    onConfirm={confirmStop}
    onClose={() => (confirmingStop = false)}
  />
{/if}

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    z-index: 30;
  }
  .screen {
    position: fixed;
    z-index: 31;
    inset: 0;
    margin: auto;
    width: min(72rem, calc(100vw - 2rem));
    height: min(52rem, calc(100vh - 2rem));
    display: flex;
    flex-direction: column;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow-2);
    overflow: hidden;
    outline: none;
  }
  header {
    flex: none;
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 1rem 1.25rem 0.8rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .titles {
    min-width: 0;
    flex: 1 1 16rem;
  }
  h2 {
    margin: 0;
    font-size: var(--text-title);
  }
  .sub {
    margin: 0.2rem 0 0;
    font-size: var(--text-meta);
    color: var(--fg-2);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .bar {
    flex: none;
    height: 8px;
    margin: 0.9rem 1.25rem 0;
    border-radius: 4px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .fill {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 220ms ease;
  }
  .fill.paused {
    background: var(--warn-border);
  }
  .banner {
    flex: none;
    margin: 0.8rem 1.25rem 0;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    font-size: var(--text-sm);
    background: var(--surface);
    border: 1px solid var(--border);
  }
  .banner.warn {
    background: var(--warn-bg);
    border-color: var(--warn-border);
    color: var(--warn-fg);
  }
  .banner.err {
    color: var(--danger);
  }
  .filters {
    flex: none;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem 1rem;
    padding: 0.8rem 1.25rem 0.4rem;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }
  /* A filter, not an action (ADR-063: view toggles keep their own look). */
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    background: var(--bg);
    color: var(--fg-2);
    border: 1px solid var(--border);
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
  }
  .chip.on {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--accent-fg);
  }
  .chip .n {
    font-variant-numeric: tabular-nums;
    opacity: 0.85;
  }
  .when {
    margin: 0;
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .list {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 0 1.25rem 0.8rem;
  }
  .row {
    display: grid;
    grid-template-columns: minmax(0, 2fr) 7.5rem minmax(0, 2.2fr);
    gap: 0.75rem;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
    font-size: var(--text-sm);
  }
  .row.head {
    position: sticky;
    top: 0;
    background: var(--bg);
    color: var(--fg-2);
    font-size: var(--text-meta);
    border-bottom-width: 2px;
  }
  .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .detail {
    color: var(--fg-2);
    overflow-wrap: anywhere;
  }
  .detail.err {
    color: var(--danger);
  }
  .pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    font-size: var(--text-meta);
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg-2);
    white-space: nowrap;
  }
  .pill.reading {
    border-color: var(--accent);
    color: var(--accent);
  }
  .pill.added {
    border-color: var(--ok-border);
    color: var(--ok-fg);
  }
  .pill.failed {
    border-color: var(--danger);
    color: var(--danger);
  }
  .pill.unchanged {
    background: var(--surface-2);
  }
  .empty {
    margin: 1.2rem 1.25rem;
    color: var(--fg-2);
    font-size: var(--text-sm);
  }
  .list .empty {
    margin: 1rem 0;
  }
  .more {
    margin-top: 0.8rem;
  }
  footer {
    flex: none;
    padding: 0.7rem 1.25rem;
    font-size: var(--text-meta);
    color: var(--fg-2);
    background: var(--surface);
    border-top: 1px solid var(--border);
  }
  .mono {
    font-family: var(--font-mono, ui-monospace, monospace);
  }
  /* Narrow: the status moves under the name, so the reason keeps a readable width. */
  @media (max-width: 640px) {
    .row {
      grid-template-columns: minmax(0, 1fr) auto;
    }
    .row .detail {
      grid-column: 1 / -1;
    }
    .row.head span:last-child {
      display: none;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .fill {
      transition: none;
    }
  }
</style>
