<script lang="ts">
  import type { SourceView } from '../core/types'
  import { figureUrl } from '../core/api'
  import Icon from '../shell/Icon.svelte'
  import ChunkContextView from './ChunkContextView.svelte'

  let {
    source,
    onOpenInDocument = null,
  }: {
    source: SourceView
    /** Open this citation's page in the Library's source pane. Absent when the shell cannot
     *  navigate — the control is then not rendered rather than rendered dead. */
    onOpenInDocument?: ((chunkKey: string) => void) | null
  } = $props()

  function markerLabel(m: string): string {
    if (m === 'contested') return 'contested in corpus'
    if (m === 'superseded_trend') return 'trend superseded'
    return m
  }

  // These chips are OFF by default and are labelled experimental wherever they appear
  // (REVIEW 2026-08-12 §2b R3). KI-33: the stance pass they rest on judges a concept pair with
  // **no document text in the prompt**, has no neutral label, and its verdict moves with the
  // pair's position in a generated list. So a chip is a prompt to go look, never a finding —
  // and the tooltip has to say which, because a warning triangle beside a citation does not
  // read as "exploratory" on its own. The rebuild is ADR-041.
  const MARKER_TOOLTIP =
    'Experimental. A hint from your corpus’s concept graph that this topic may be disputed — ' +
    'it is not a measurement, and it can be wrong. Treat it as a prompt to check the sources ' +
    'yourself, never as a verdict on them.'
</script>

<article class="source">
  <header>
    <span class="cite">{source.citation}</span>
    {#each source.markers as m (m)}
      <span class="chip" title={MARKER_TOOLTIP}>
        <Icon name="triangle-alert" size={12} /> {markerLabel(m)}
        <span class="chip-exp">experimental</span>
      </span>
    {/each}
  </header>
  {#if source.figure_id}
    <img class="figure" src={figureUrl(source.figure_id)} alt={`figure for source ${source.n}`} />
  {/if}
  <p class="excerpt">{source.excerpt}</p>
  <!-- ROADMAP 19. Only offered when the citation carries a `chunk_key`: without one there is
       nothing to locate, and a dead control is worse than none. A figure source has no position
       in the text either — its place is the page image, which is ROADMAP 18. -->
  {#if source.chunk_key && !source.figure_id}
    <ChunkContextView chunkKey={source.chunk_key} />
  {/if}
  <!-- ROADMAP 18. Offered for a **figure** too, which the comment above anticipated: a figure
       has no position in the text, so the page image is the only place it can be shown. -->
  {#if source.chunk_key && onOpenInDocument}
    {@const key = source.chunk_key}
    <button class="inpage" type="button" onclick={() => onOpenInDocument(key)}>
      <Icon name="book-open" size={12} />
      Show the page
    </button>
  {/if}
</article>

<style>
  .inpage {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    margin-top: 0.45rem;
    background: none;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.1rem 0.55rem;
    font: inherit;
    font-size: 0.72rem;
    color: var(--fg-2);
    cursor: pointer;
  }
  .inpage:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
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
  /* Rides inside the chip rather than beside it, so the qualifier cannot be separated from the
     claim it qualifies by a line wrap (REVIEW 2026-08-12 §2b R3). */
  .chip-exp {
    font-size: 0.62rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    opacity: 0.75;
    border-left: 1px solid currentColor;
    padding-left: 0.3rem;
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
