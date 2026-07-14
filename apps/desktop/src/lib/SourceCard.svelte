<script lang="ts">
  import type { SourceView } from './types'
  import { figureUrl } from './api'
  import Icon from './Icon.svelte'

  let { source }: { source: SourceView } = $props()

  function markerLabel(m: string): string {
    if (m === 'contested') return 'contested in corpus'
    if (m === 'superseded_trend') return 'trend superseded'
    return m
  }
</script>

<article class="source">
  <header>
    <span class="cite">{source.citation}</span>
    {#each source.markers as m (m)}
      <span class="chip" title="From your corpus's concept graph (advisory, not a gate)">
        <Icon name="triangle-alert" size={12} /> {markerLabel(m)}
      </span>
    {/each}
  </header>
  {#if source.figure_id}
    <img class="figure" src={figureUrl(source.figure_id)} alt={`figure for source ${source.n}`} />
  {/if}
  <p class="excerpt">{source.excerpt}</p>
</article>

<style>
  .source {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    background: var(--surface);
  }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
  }
  .cite {
    font-weight: 600;
    font-size: 0.85rem;
  }
  .chip {
    font-size: 0.72rem;
    color: var(--warn-fg);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 999px;
    padding: 0.05rem 0.5rem;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
  }
  .figure {
    max-width: 100%;
    margin-top: 0.5rem;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .excerpt {
    margin: 0.4rem 0 0;
    font-size: 0.86rem;
    color: var(--fg-2);
    white-space: pre-wrap;
    font-family: var(--font-serif);
    line-height: 1.55;
  }
</style>
