<script lang="ts">
  // The ingest indicator that lives on the right of the status bar: a thin determinate bar, a
  // count, and a click target that opens the indexing screen (ROADMAP 31, `Indexing`).
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
    dismissIngest,
    failedCount,
    fraction,
    ingestLabel,
    ingestRun,
  } from '../lib/core/ingest.svelte'
  import { tooltip } from '../lib/shell/tooltip'

  const status = $derived(ingestRun.status)
  const pct = $derived(fraction(status))
  const label = $derived(ingestLabel(status))
  const running = $derived(status?.state === 'running')
  // Red when the run failed, and when it ended with files that failed: a finished bar in the
  // accent colour beside "2 failed" would still read as all well.
  const failed = $derived(
    status?.state === 'error' || (!running && status !== null && failedCount(status) > 0),
  )
  const paused = $derived(running && status?.control === 'paused')
</script>

{#if ingestRun.visible && status}
  <div class="ingest" class:failed class:paused>
    <button
      class="ingest-btn"
      onclick={() => (ingestRun.panelOpen = !ingestRun.panelOpen)}
      aria-haspopup="dialog"
      aria-expanded={ingestRun.panelOpen}
      use:tooltip={ingestRun.panelOpen ? 'Close the indexing screen' : 'Open the indexing screen'}
      type="button"
    >
      <!-- Determinate whenever the backend has counted the batch; indeterminate only in the brief
           window before it has, and it says so in words rather than animating a lie. -->
      <span class="track" class:indeterminate={running && pct === null} aria-hidden="true">
        <span class="fill" style={pct === null ? undefined : `width: ${Math.round(pct * 100)}%`}
        ></span>
      </span>
      <span class="ingest-label">{label}</span>
    </button>

    {#if !running}
      <button class="ingest-x" onclick={dismissIngest} aria-label="Dismiss" use:tooltip={'Hide this note from the status bar'} type="button">
        <Icon name="x" size={12} />
      </button>
    {/if}
  </div>
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
    width: 96px;
    height: 5px;
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
  .paused .fill {
    background: var(--warn-border);
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
</style>
