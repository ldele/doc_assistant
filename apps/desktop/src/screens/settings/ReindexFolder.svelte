<script lang="ts">
  // The Re-index confirmation (the user, 2026-11-04): before a run, what it would read, roughly how
  // long that takes on this computer, and *Only index new files*. Renders GET /api/ingest/plan; the
  // wording is in `reindex.ts`, where it is tested.
  //
  // Same house style as ConfirmDialog (scrim + centred card, Escape and Cancel back out), but its
  // own component: it loads a plan and carries a checkbox, which the generic dialog does not.
  import { getIngestPlan } from '../../lib/core/api/sources'
  import type { IngestPlan } from '../../lib/core/types'
  import {
    confirmDisabled,
    confirmLabel,
    estimateLine,
    measuredLine,
    planLines,
  } from '../../lib/settings/reindex'

  let {
    folder,
    onConfirm,
    onClose,
  }: {
    folder: string
    onConfirm: (onlyNew: boolean) => void
    onClose: () => void
  } = $props()

  let plan = $state<IngestPlan | null>(null)
  let loadError = $state<string | null>(null)
  let onlyNew = $state(false)
  let cardEl = $state<HTMLDivElement | null>(null)

  getIngestPlan()
    .then((p) => (plan = p))
    .catch((e: unknown) => (loadError = String(e)))

  $effect(() => {
    cardEl?.focus()
  })

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div
  class="modal"
  role="dialog"
  aria-modal="true"
  aria-labelledby="reindex-title"
  tabindex="-1"
  bind:this={cardEl}
>
  <h2 id="reindex-title">Re-index this folder?</h2>
  <p class="folder mono" title={folder}>{folder}</p>

  {#if loadError}
    <p class="err" role="alert">Couldn't count the files: {loadError}</p>
  {:else if !plan}
    <p class="muted">Counting the files…</p>
  {:else}
    <ul class="lines">
      {#each planLines(plan, onlyNew) as line (line)}
        <li>{line}</li>
      {:else}
        <li>The folder has no files Provenote can read.</li>
      {/each}
    </ul>
    <p class="estimate" aria-live="polite">{estimateLine(plan, onlyNew)}</p>
    {#if measuredLine(plan)}
      <p class="muted small">{measuredLine(plan)}</p>
    {/if}
    <label class="check">
      <input type="checkbox" bind:checked={onlyNew} />
      Only index new files
      <span class="muted small">Changed files keep the version already indexed.</span>
    </label>
    <p class="muted small">You can pause or stop the run at any time; what finishes stays.</p>
  {/if}

  <div class="mactions">
    <button class="btn" onclick={onClose} type="button">Cancel</button>
    <button
      class="btn btn-primary"
      type="button"
      disabled={!plan || confirmDisabled(plan, onlyNew)}
      onclick={() => onConfirm(onlyNew)}
    >
      {plan ? confirmLabel(plan, onlyNew) : 'Re-index'}
    </button>
  </div>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--fg) 32%, transparent);
    z-index: 40;
  }
  .modal {
    position: fixed;
    z-index: 41;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(92vw, 32rem);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    outline: none;
  }
  h2 {
    margin: 0;
    font-size: var(--text-title);
    font-family: var(--font-serif);
  }
  .folder {
    margin: 0;
    font-size: var(--text-meta);
    color: var(--fg-2);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .lines {
    margin: 0.2rem 0;
    padding-left: 1.1rem;
    font-size: 0.85rem;
    line-height: 1.6;
  }
  .estimate {
    margin: 0.2rem 0 0;
    font-weight: 600;
    font-size: 0.9rem;
  }
  .check {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.4rem;
    margin-top: 0.4rem;
    font-size: 0.85rem;
    cursor: pointer;
  }
  .check .small {
    flex-basis: 100%;
    margin-left: 1.5rem;
  }
  .muted {
    margin: 0;
    color: var(--fg-2);
  }
  .small {
    font-size: var(--text-meta);
  }
  .err {
    margin: 0;
    color: var(--danger);
    font-size: 0.85rem;
  }
  .mono {
    font-family: var(--font-mono, ui-monospace, monospace);
  }
  .mactions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
</style>
