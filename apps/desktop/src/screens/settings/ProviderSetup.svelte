<script lang="ts">
  // Settings > Provider & model: which model answers, where its text goes and who bills, then the
  // providers grouped by where the text goes, and one provider's form at a time. ADR-034, amended
  // 2026-11-01: Getting started holds no form, so the keys and the Ollama probe live here. ADR-060:
  // every provider but Claude and Ollama is a preset or a server of your own, on one form.
  //
  // Everything shown here is computed by the backend (`GET /api/setup` -> doc_assistant.readiness),
  // so this component renders a verdict rather than forming one — the same wording the docs use.
  // The only local logic is presentation (`./setup.ts`, unit-tested under `npm test`).
  //
  // This file reads the state and lays out its three parts, each a file beside it: `AnsweringWith`
  // (the card on top), `ProviderList` (the providers) and `ProviderForm` (the open one's form).
  import type { SetupState } from '../../lib/core/types/setup'
  import { getSetup } from '../../lib/core/api'
  import { shell } from '../../lib/shell/shell.svelte'
  import { answeringWith, modelForProvider, providerGroups } from '../../lib/settings/setup'
  import AnsweringWith from './AnsweringWith.svelte'
  import ProviderForm from './ProviderForm.svelte'
  import ProviderList from './ProviderList.svelte'

  // `onProviderChanged` lets the parent re-pull /api/settings so the status bar and the Retrieval
  // tab agree with the switch made here (one source of truth per surface, refreshed together).
  let { onProviderChanged }: { onProviderChanged: () => void } = $props()

  let setup = $state<SetupState | null>(null)
  let loadError = $state<string | null>(null)
  // True while the probe is out. The panel draws from local state first (`probe=false`) and fills
  // in the probe's verdict after: with nothing listening, it takes seconds.
  let probing = $state(false)
  // The provider whose form is open: the one in use, until the user picks another from the list.
  let picked = $state<string | null>(null)
  // The model box of each form, by provider id. Free text over the server's list. Held here, not
  // in the form, because every reading of the state seeds it (`show`).
  let models = $state<Record<string, string>>({})

  const open = $derived(picked ?? setup?.active_provider ?? null)
  const now = $derived(setup ? answeringWith(setup, probing) : null)
  const groups = $derived(setup ? providerGroups(setup.providers) : [])

  void refresh()

  async function refresh(probe = true): Promise<void> {
    try {
      if (probe && setup === null) {
        // Local state first, in well under a second; the probe's verdict replaces it below.
        show(await getSetup(false))
      }
      probing = probe
      const state = await getSetup(probe)
      show(state)
      // A Re-check that finds a server back has to reach the rail badge and the chat banner too.
      if (probe) shell.setup = state
    } catch (e) {
      loadError = message(e)
    } finally {
      probing = false
    }
  }

  function show(state: SetupState): void {
    setup = state
    loadError = null
    // Seed each model box with a model its provider can serve; only an empty box is filled, so a
    // refresh never undoes typing. Ollama's is re-seeded when its pick is not installed: offering
    // a model the server does not have would fail on the next question.
    for (const p of state.providers) {
      const mine = state.active_provider === p.id ? state.active_model : ''
      if (p.id === 'ollama') {
        models[p.id] = modelForProvider('ollama', models[p.id] || mine, [...p.models])
      } else if (!models[p.id]) {
        models[p.id] = modelForProvider(p.id, mine, [...p.models])
      }
    }
  }

  const message = (e: unknown): string => (e instanceof Error ? e.message : String(e))
</script>

<section class="setup">
  {#if loadError}
    <p class="err" role="alert">Couldn't read the setup state: {loadError}</p>
  {:else if !setup || !now}
    <p class="muted">Checking…</p>
  {:else}
    <AnsweringWith {now} />

    <div class="split">
      <ProviderList {groups} {open} {probing} bind:picked />
      <ProviderForm
        {setup}
        {open}
        bind:probing
        bind:picked
        bind:models
        onSetup={show}
        onRecheck={refresh}
        {onProviderChanged}
      />
    </div>
  {/if}
</section>

<style>
  .setup {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
  }
  /* Two columns while both fit, one when they do not: the list keeps its width and the form takes
     what is left, so the drawer's own width decides and no breakpoint has to guess it. */
  .split {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 0.8rem;
    margin-top: 0.8rem;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.8rem;
    margin: 0.5rem 0 0;
  }
</style>
