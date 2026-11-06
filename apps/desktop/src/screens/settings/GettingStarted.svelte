<script lang="ts">
  // Settings > Getting started: what this install still needs, and the tab that does each job
  // (ADR-034, amended 2026-11-01). Two rows and no form: everything about models lives in Provider
  // & model, so a first run has one place to look.
  //
  // The rows are `/api/setup`'s steps as the backend worded them, so this renders a verdict rather
  // than forming one. It reads `shell.setup`, the one the rail badge and the chat banner read, so
  // the three cannot disagree.
  import { getSetup } from '../../lib/core/api'
  import Icon from '../../shell/Icon.svelte'
  import { shell } from '../../lib/shell/shell.svelte'
  import { stepLink, type SettingsSectionId } from '../../lib/settings/sections'
  import { setupSummary, stepTone } from '../../lib/settings/setup'
  import { tooltip } from '../../lib/shell/tooltip'

  let { onOpen }: { onOpen: (section: SettingsSectionId) => void } = $props()

  let loadError = $state<string | null>(null)

  // What the shell last read can be old: Ollama may have stopped since. The rows keep that reading
  // until this one lands, which with nothing listening takes the probe's whole budget.
  void refresh()

  async function refresh(): Promise<void> {
    try {
      shell.setup = await getSetup()
      loadError = null
    } catch (e) {
      loadError = e instanceof Error ? e.message : String(e)
    }
  }
</script>

<section class="start">
  {#if shell.setup}
    <h3>{setupSummary(shell.setup)}</h3>
    <p class="sub">
      {shell.setup.ready
        ? 'It can answer from your documents.'
        : 'Searching your documents runs on this computer. Writing the answer needs a model.'}
    </p>

    <ul class="rows">
      {#each shell.setup.steps as step (step.id)}
        {@const tone = stepTone(step)}
        {@const link = stepLink(step.id)}
        <li class="row">
          <span class="mark {tone}" aria-hidden="true">
            {#if tone === 'ok'}
              <Icon name="check" size={14} />
            {:else if tone === 'down'}
              <Icon name="x" size={14} />
            {/if}
          </span>
          <div class="what">
            <strong>{step.title}</strong>
            <p>{step.detail}</p>
          </div>
          <div class="ends">
            <span class="pill {tone}">{step.status}</span>
            {#if link}
              <button type="button" class="go" onclick={() => onOpen(link.section)} use:tooltip={'Open the tab where this step is done'}>{link.label}</button>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
  {:else if loadError}
    <p class="err" role="alert">Couldn't read the setup state: {loadError}</p>
  {:else}
    <p class="muted">Checking…</p>
  {/if}
</section>

<style>
  .start {
    padding: 0.8rem 0;
  }
  h3 {
    margin: 0;
    font-size: 1.05rem;
  }
  .sub {
    margin: 0.25rem 0 0.8rem;
    font-size: 0.8rem;
    color: var(--fg-2);
  }
  .muted {
    color: var(--fg-2);
    font-size: 0.82rem;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.6rem;
  }
  /* Wraps rather than squeezes: at phone width the status and the link take their own line. */
  .row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem 0.7rem;
    padding: 0.75rem 0.8rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
  }
  .mark {
    flex: none;
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 50%;
    border: 1.5px solid var(--border);
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .mark.ok {
    color: var(--ok-fg);
    border-color: var(--ok-fg);
  }
  .mark.down {
    color: var(--danger);
    border-color: var(--danger);
  }
  .what {
    flex: 1 1 11rem;
    min-width: 0;
  }
  .what strong {
    font-size: 0.92rem;
  }
  .what p {
    margin: 0.1rem 0 0;
    font-size: 0.8rem;
    color: var(--fg-2);
    overflow-wrap: anywhere;
  }
  .ends {
    flex: none;
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .pill {
    font-size: 0.72rem;
    white-space: nowrap;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    border: 1px solid var(--border);
    color: var(--fg-2);
  }
  .pill.ok {
    color: var(--ok-fg);
    border-color: var(--ok-fg);
  }
  .pill.down {
    color: var(--danger);
    border-color: var(--danger);
  }
  .go {
    font: inherit;
    font-size: 0.8rem;
    white-space: nowrap;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--fg);
    padding: 0.3rem 0.7rem;
  }
  .go:hover {
    border-color: var(--accent);
  }
</style>
