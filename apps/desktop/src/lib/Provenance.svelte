<script lang="ts">
  import type { TurnResult } from './types'
  import Markdown from './Markdown.svelte'

  let { result }: { result: TurnResult } = $props()
  const id8 = $derived(result.record_id ? result.record_id.slice(0, 8) : null)
</script>

{#if result.citation_note_md}
  <div class="citation"><Markdown source={result.citation_note_md} /></div>
{/if}

<details class="prov">
  <summary>Provenance &amp; usage{id8 ? ` · ${id8}` : ''}</summary>
  <div class="body">
    <Markdown source={result.provenance_card_md} />
    <Markdown source={result.usage_md} />
  </div>
</details>

<style>
  .citation :global(.md) {
    color: var(--warn-fg);
    font-size: 0.82rem;
  }
  .prov {
    margin-top: 0.6rem;
    font-size: 0.82rem;
    color: var(--fg-2);
  }
  summary {
    cursor: pointer;
    user-select: none;
  }
  .body {
    padding: 0.3rem 0 0 0.5rem;
  }
</style>
