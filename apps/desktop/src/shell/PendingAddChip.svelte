<script lang="ts">
  // T-007 (A8″): papers handed over while the engine is not ready. They stay staged and the review
  // sheet opens once the engine answers (App.svelte); this chip says so meanwhile, and says it
  // plainly if the engine stalled, with *Try again* and *Dismiss*.
  import { accept, clearPending } from '../lib/library/accept.svelte'
  import { pendingAddText } from '../lib/library/accept'
  import { shell } from '../lib/shell/shell.svelte'
  import { tooltip } from '../lib/shell/tooltip'

  const waiting = $derived(accept.pending.length > 0 && shell.status !== 'ready')
  const text = $derived(
    waiting ? pendingAddText(accept.pending.length, accept.engineGaveUp ? 'failed' : 'waiting') : null,
  )

  function tryAgain(): void {
    accept.engineGaveUp = false
  }

  function dismiss(): void {
    accept.engineGaveUp = false
    clearPending()
  }
</script>

{#if text}
  <div class="pending" class:failed={accept.engineGaveUp} role="status">
    <span class="what">{text}</span>
    {#if accept.engineGaveUp}
      <button class="btn small" type="button" onclick={tryAgain} use:tooltip={'Keep these papers and wait for the engine again'}>Try again</button>
      <button class="btn small" type="button" onclick={dismiss} use:tooltip={'Drop these papers; nothing was added'}>Dismiss</button>
    {/if}
  </div>
{/if}

<style>
  .pending {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-left: auto;
    font-size: var(--text-xs);
    color: var(--fg-2);
  }
  .pending.failed .what {
    color: var(--danger);
  }
  .small {
    padding: 0.05rem 0.45rem;
    font-size: var(--text-xs);
  }
</style>
