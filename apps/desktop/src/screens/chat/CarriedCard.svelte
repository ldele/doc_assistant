<script lang="ts">
  // *What the answer carried* (A8b, `[s8]`), inside *How this answer was made*: the earlier
  // messages kept of the total, the carried exchanges kept (A8a), the passages kept, shortened and
  // left out, the window and where its size came from, and the mode with the prompt version. Read
  // from the turn's record (A5's `context_json`, A7's mode); the fitting note stays in the text.
  import type { TurnContext } from '../../lib/core/types'
  import { anythingLeftOut, carriedRows } from '../../lib/chat/turnDetails'

  let {
    context,
    chatMode = null,
    promptVersion = null,
  }: { context: TurnContext | null; chatMode?: string | null; promptVersion?: string | null } = $props()

  const rows = $derived(carriedRows(context, chatMode, promptVersion))
  const gaveWay = $derived(anythingLeftOut(context))
</script>

{#if rows.length > 0}
  <section class="carried" aria-label="What the answer carried">
    <h4>What the answer carried</h4>
    {#if gaveWay}
      <p class="warn">Fitted to the window: something was left out or shortened, as said below.</p>
    {/if}
    <dl>
      {#each rows as r (r.key)}
        <dt class="muted">{r.key}</dt>
        <dd>{r.value}</dd>
      {/each}
    </dl>
  </section>
{/if}

<style>
  .carried {
    margin: 0.3rem 0 0.5rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 0.55rem 0.75rem;
    max-width: 40rem;
  }
  h4 {
    margin: 0 0 0.35rem;
    font-size: 0.86rem;
    color: var(--fg);
  }
  .warn {
    margin: 0 0 0.4rem;
    font-size: 0.78rem;
    color: var(--warn-fg);
    background: var(--warn-bg);
    border-radius: 6px;
    padding: 0.25rem 0.45rem;
  }
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.15rem 0.8rem;
    margin: 0;
    font-size: 0.8rem;
  }
  dd {
    margin: 0;
    color: var(--fg);
  }
</style>
