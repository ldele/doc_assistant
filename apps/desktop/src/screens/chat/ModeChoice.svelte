<script lang="ts">
  // *Answer as*: one radio per chat mode, built in first, then a person's own, each with what it
  // is (A8a, the *New chat defaults* window and *Continue in a new chat*). A person's own mode is
  // marked *not evaluated*, as its answers are (A7).
  import type { ChatMode } from '../../lib/core/types'
  import { CRITICAL_READING, RESEARCH_ASSISTANT } from '../../lib/chat/setup'

  let {
    modes,
    modeId = $bindable(),
    name,
  }: { modes: ChatMode[]; modeId: string; name: string } = $props()

  function about(m: ChatMode): string {
    if (m.id === RESEARCH_ASSISTANT) return 'Built in · the default'
    if (m.id === CRITICAL_READING) return 'Built in · where sources agree, disagree, and their limits'
    return 'Yours · its answers are marked not evaluated'
  }
</script>

<fieldset class="modes">
  <legend class="muted">Answer as</legend>
  {#each modes as m (m.id)}
    <label class="moderow">
      <input type="radio" {name} value={m.id} bind:group={modeId} />
      <span class="modetext">
        <span class="modename">{m.name}</span>
        <span class="modeabout muted">{about(m)}</span>
      </span>
    </label>
  {/each}
</fieldset>

<style>
  .modes {
    border: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  legend {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0;
    margin-bottom: var(--space-1);
  }
  .moderow {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    cursor: pointer;
  }
  .moderow input {
    accent-color: var(--accent);
    margin: 0.2rem 0 0;
    flex: none;
  }
  .modetext {
    display: flex;
    flex-direction: column;
  }
  .modename {
    font-size: var(--text-sm);
    color: var(--fg);
  }
  .modeabout {
    font-size: 0.74rem;
  }
</style>
