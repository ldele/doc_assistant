<script lang="ts">
  // The ingest indicator that lives on the right of the status bar: a thin determinate bar, a
  // count, and a click target that opens the detail.
  //
  // Split out of StatusBar so the bar keeps its "zero props, ambient" shape — this component owns
  // one concern (a run in flight) and reads it from the shared `ingestRun` rune, exactly as
  // StatusBar reads `shell`.
  //
  // **It renders position, never outcome, while running.** `added`/`skipped`/`errors` are 0 for
  // the whole duration by design, so showing them mid-run would report a result that has not
  // happened. They appear only once the run reaches `done`.
  import Icon from './Icon.svelte'
  import {
    currentLabel,
    dismissIngest,
    fraction,
    ingestLabel,
    ingestRun,
  } from '../core/ingest.svelte'

  const status = $derived(ingestRun.status)
  const pct = $derived(fraction(status))
  const label = $derived(ingestLabel(status))
  const running = $derived(status?.state === 'running')
  const failed = $derived(status?.state === 'error')
</script>

{#if ingestRun.visible && status}
  <div class="ingest" class:failed>
    <button
      class="ingest-btn"
      onclick={() => (ingestRun.panelOpen = !ingestRun.panelOpen)}
      aria-expanded={ingestRun.panelOpen}
      title={ingestRun.panelOpen ? 'Hide indexing detail' : 'Show indexing detail'}
      type="button"
    >
      <!-- Determinate whenever the backend has counted the batch; indeterminate only in the brief
           window before it has, and it says so in words rather than animating a lie. -->
      <span class="track" class:indeterminate={running && pct === null} aria-hidden="true">
        <span class="fill" style={pct === null ? undefined : `width: ${Math.round(pct * 100)}%`}
        ></span>
      </span>
      <span class="ingest-label">{label}</span>
      <Icon name={ingestRun.panelOpen ? 'chevron-down' : 'chevron-up'} size={12} />
    </button>

    {#if !running}
      <button class="ingest-x" onclick={dismissIngest} title="Dismiss" type="button">
        <Icon name="x" size={12} />
      </button>
    {/if}
  </div>

  {#if ingestRun.panelOpen}
    <!-- Anchored above the bar; the bar is the last row of the app, so the panel opens upward. -->
    <div class="panel" role="dialog" aria-label="Indexing detail">
      <p class="panel-title">{label}</p>

      {#if running}
        <!-- No "Progress" row: the title above already says "Indexing 4 of 12", and repeating the
             same pair two lines apart is noise. This row answers the question the title cannot. -->
        <p class="panel-line">
          <span class="k">Now</span>
          <span class="v mono">{currentLabel(status)}</span>
        </p>
      {:else if failed}
        <p class="panel-line err">{status.message ?? 'The indexer stopped with an error.'}</p>
      {:else}
        <p class="panel-line">
          <span class="k">Added</span><span class="v">{status.added}</span>
        </p>
        <p class="panel-line">
          <span class="k">Unchanged</span><span class="v">{status.skipped}</span>
        </p>
        <p class="panel-line" class:err={status.errors > 0}>
          <span class="k">Errors</span><span class="v">{status.errors}</span>
        </p>
      {/if}

      {#if ingestRun.lostContact}
        <p class="panel-line err">
          Lost contact with the indexer. It may still be running — reopen the app to check.
        </p>
      {/if}

      {#if status.source_dir}
        <p class="panel-foot mono" title={status.source_dir}>{status.source_dir}</p>
      {/if}
    </div>
  {/if}
{/if}

<style>
  .ingest {
    flex: none;
    display: flex;
    align-items: center;
    gap: 0.15rem;
    margin-left: auto; /* pushes the whole chip to the right of the status bar */
  }
  .ingest-btn {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: none;
    border: none;
    padding: 0.1rem 0.25rem;
    border-radius: var(--radius-sm, 4px);
    color: var(--fg-2);
    font-size: var(--text-meta);
    cursor: pointer;
  }
  .ingest-btn:hover {
    background: var(--bg-2);
    color: var(--fg);
  }
  .ingest-label {
    white-space: nowrap;
  }
  .track {
    position: relative;
    display: inline-block;
    width: 68px;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    inset: 0 auto 0 0;
    width: 0;
    background: var(--accent, #7c6cf0);
    border-radius: 2px;
    transition: width 220ms ease;
  }
  .failed .fill {
    background: var(--danger);
  }
  /* Only used before the batch has been counted — a real fraction replaces it as soon as one
     exists, so this never stands in for progress the app actually knows. */
  .track.indeterminate .fill {
    width: 40%;
    animation: slide 1.1s ease-in-out infinite;
  }
  @keyframes slide {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(250%);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .track.indeterminate .fill {
      animation: none;
      width: 100%;
      opacity: 0.4;
    }
    .fill {
      transition: none;
    }
  }
  .ingest-x {
    background: none;
    border: none;
    padding: 0.15rem;
    color: var(--fg-2);
    cursor: pointer;
    border-radius: var(--radius-sm, 4px);
  }
  .ingest-x:hover {
    background: var(--bg-2);
    color: var(--fg);
  }

  .panel {
    position: absolute;
    right: 0.6rem;
    bottom: 1.9rem;
    z-index: 40;
    min-width: 15rem;
    max-width: min(26rem, 90vw);
    padding: 0.6rem 0.7rem;
    border: 1px solid var(--border);
    border-radius: var(--radius, 8px);
    background: var(--bg);
    box-shadow: var(--shadow-md, 0 6px 24px rgb(0 0 0 / 0.28));
  }
  .panel-title {
    margin: 0 0 0.4rem;
    font-size: var(--text-meta);
    font-weight: 600;
    color: var(--fg);
  }
  .panel-line {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0.15rem 0;
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .panel-line .v {
    color: var(--fg);
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .panel-line.err {
    color: var(--danger);
  }
  .panel-foot {
    margin: 0.45rem 0 0;
    padding-top: 0.35rem;
    border-top: 1px solid var(--border);
    font-size: var(--text-meta);
    color: var(--fg-2);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .mono {
    font-family: var(--font-mono, ui-monospace, monospace);
  }
</style>
