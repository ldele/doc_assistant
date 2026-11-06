<script module lang="ts">
  /** What the *Add an OpenAI-compatible server* row picks: the form for a server not saved yet. */
  export const NEW_SERVER = 'new-server'
</script>

<script lang="ts">
  // Settings > Provider & model > the provider list, left of the open form: the providers grouped
  // by where the text goes (*On this computer*, *Cloud, with your own key*, *Your own server*), each
  // with its status, and the row that adds an OpenAI-compatible server. Owns nothing but the pick,
  // which it writes back to ProviderSetup; the form it opens is `ProviderForm.svelte`.
  import type { ProviderReadiness } from '../../lib/core/types/setup'
  import { providerStatusLabel } from '../../lib/settings/setup'

  let {
    groups,
    open,
    probing,
    picked = $bindable(),
  }: {
    groups: { id: string; label: string; providers: ProviderReadiness[] }[]
    /** The provider whose form is open: the one picked, or the one in use. */
    open: string | null
    /** The probe is out, so a status may still be a guess. */
    probing: boolean
    picked: string | null
  } = $props()

  function pillClass(p: ProviderReadiness): string {
    if (p.key_status === 'refused') return 'down'
    return p.ready ? 'ok' : 'todo'
  }
</script>

<!-- Grouped by where the text goes. A new provider takes its row from the payload, its group
     included, so this list does not change when one is added. -->
<div class="plist" role="group" aria-label="Providers">
  {#each groups as group (group.id)}
    <h4>{group.label}</h4>
    {#each group.providers as p (p.id)}
      <button
        type="button"
        class="prow"
        class:on={open === p.id}
        aria-current={open === p.id ? 'true' : undefined}
        onclick={() => (picked = p.id)}
      >
        <span class="pname">{p.name}</span>
        <span class="pill {pillClass(p)}">{providerStatusLabel(p, probing)}</span>
      </button>
    {/each}
  {/each}
  {#if !groups.some((g) => g.id === 'server')}<h4>Your own server</h4>{/if}
  <button
    type="button"
    class="prow add"
    class:on={open === NEW_SERVER}
    aria-current={open === NEW_SERVER ? 'true' : undefined}
    onclick={() => (picked = NEW_SERVER)}
  >
    <span class="pname">Add an OpenAI-compatible server</span>
  </button>
</div>

<style>
  .plist {
    flex: 1 1 10.5rem;
    min-width: 0;
  }
  h4 {
    margin: 0.7rem 0 0.3rem;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--fg-2);
  }
  h4:first-child {
    margin-top: 0;
  }
  .prow {
    width: 100%;
    font: inherit;
    font-size: 0.84rem;
    cursor: pointer;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.25rem 0.5rem;
    text-align: left;
    padding: 0.4rem 0.55rem;
    margin-bottom: 0.3rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--fg);
  }
  .prow:hover {
    border-color: var(--accent);
  }
  /* The smaller, muted text of the picked row came from the *Answering with* card's `.on`, which
     matched this class too while both lived in one file; written out here so it looks the same. */
  .prow.on {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 14%, var(--bg));
    font-weight: 600;
    font-size: 0.8rem;
    color: var(--fg-2);
  }
  .pname {
    min-width: 0;
  }
  .pill {
    flex: none;
    margin-left: auto;
    font-size: 0.72rem;
    font-weight: 400;
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
  .prow.add {
    justify-content: flex-start;
    color: var(--fg-2);
    border-style: dashed;
  }
</style>
