<script lang="ts">
  import type { TurnResult } from './types'
  import Markdown from './Markdown.svelte'
  import SourceCard from './SourceCard.svelte'
  import ClaimReview from './ClaimReview.svelte'
  import Provenance from './Provenance.svelte'

  let {
    question,
    answer,
    result,
    streaming = false,
    error = null,
  }: {
    question: string
    answer: string
    result: TurnResult | null
    streaming?: boolean
    error?: string | null
  } = $props()
</script>

<div class="turn">
  <div class="you">
    <span class="who">You</span>
    <p>{question}</p>
  </div>

  <div class="assistant">
    {#if error}
      <p class="error">⚠ {error}</p>
    {:else if result}
      <Markdown source={result.answer} />
      {#if result.sources.length}
        <div class="sources">
          {#each result.sources as s (s.n)}
            <SourceCard source={s} />
          {/each}
        </div>
      {/if}
      <ClaimReview claims={result.flagged_claims} />
      <Provenance {result} />
    {:else}
      <Markdown source={answer} />
      {#if streaming}<span class="cursor">▍</span>{/if}
    {/if}
  </div>
</div>

<style>
  .turn {
    border-bottom: 1px solid var(--border);
    padding: 1rem 0;
  }
  .you {
    margin-bottom: 0.6rem;
  }
  .who {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--fg-2);
  }
  .you p {
    margin: 0.1rem 0 0;
    font-weight: 600;
  }
  .sources {
    display: grid;
    gap: 0.5rem;
    margin-top: 0.7rem;
  }
  .cursor {
    animation: blink 1s steps(2) infinite;
  }
  .error {
    color: var(--warn-fg);
  }
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }
</style>
