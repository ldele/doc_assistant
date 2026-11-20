<script lang="ts">
  // ADR-025 F2, non-negotiable: a scoped answer states its scope, ABOVE the answer and outside the
  // collapsed Provenance panel. An answer drawn from part of the corpus that doesn't say so is
  // indistinguishable from one drawn from all of it. A7b: every folder is named, a folder that
  // added nothing is named as left out, and a concept scope names the concept. The words are
  // `lib/chat/scope.ts`'s (tested), the same as the backend's `retrieval_scope.scope_summary`.
  import type { TurnScope } from '../../lib/core/types'
  import Icon from '../../shell/Icon.svelte'
  import { scopeIcon, scopeSentences } from '../../lib/chat/scope'

  let { scope }: { scope: TurnScope } = $props()
  const sentences = $derived(scopeSentences(scope))
</script>

<div class="scopechip">
  <Icon name={scopeIcon(scope)} size={13} />
  <span>
    {#each sentences as sentence, i (i)}
      {#if i > 0}{' '}{/if}{#each sentence as part, j (j)}{#if part.strong}<strong>{part.text}</strong
          >{:else}{part.text}{/if}{/each}
    {/each}
  </span>
</div>

<style>
  .scopechip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.6rem;
    padding: 0.3rem 0.55rem;
    border: 1px solid var(--accent);
    border-radius: 8px;
    background: color-mix(in srgb, var(--accent) 10%, transparent);
    color: var(--fg);
    font-size: 0.76rem;
    line-height: 1.35;
  }
</style>
