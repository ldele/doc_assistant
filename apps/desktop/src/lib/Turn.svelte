<script lang="ts">
  import type { TurnResult } from './types'
  import Markdown from './Markdown.svelte'
  import ClaimReview from './ClaimReview.svelte'
  import Provenance from './Provenance.svelte'
  import Icon from './Icon.svelte'

  let {
    question,
    answer,
    result,
    streaming = false,
    error = null,
    onCitationClick,
    activeCitationN = null,
  }: {
    question: string
    answer: string
    result: TurnResult | null
    streaming?: boolean
    error?: string | null
    onCitationClick?: (n: number) => void
    activeCitationN?: number | null
  } = $props()

  // Glanceable turn cost — the same numbers the Provenance panel spells out, surfaced up front
  // so the reader sees the spend without expanding anything (inform, don't gate).
  const usage = $derived(result?.usage ?? null)
  const tokens = $derived(usage ? (usage.turn_input + usage.turn_output).toLocaleString() : '')
  const spend = $derived(
    !usage
      ? ''
      : usage.is_local
        ? 'local'
        : usage.cost_usd != null
          ? `$${usage.cost_usd.toFixed(4)}`
          : 'n/a',
  )
</script>

<div class="turn">
  <div class="you">
    <span class="who">You</span>
    <p>{question}</p>
  </div>

  <div class="assistant">
    {#if error}
      <p class="error"><Icon name="triangle-alert" size={15} /> {error}</p>
    {:else if result}
      {#if result.scope}
        <!-- ADR-025 F2, non-negotiable: a scoped answer states its scope, ABOVE the answer and
             outside the collapsed Provenance panel. An answer drawn from part of the corpus that
             doesn't say so is indistinguishable from one drawn from all of it. -->
        <div class="scopechip">
          <Icon name="folder" size={13} />
          {#if result.scope.folder_name}
            <span>
              Searched <strong>{result.scope.folder_name}</strong> only —
              {result.scope.doc_count} document{result.scope.doc_count === 1 ? '' : 's'}, not the
              whole library.
            </span>
          {:else}
            <span>
              Scoped to a folder that no longer exists — <strong>no documents were searched</strong>.
            </span>
          {/if}
        </div>
      {/if}
      <Markdown source={result.answer} {onCitationClick} {activeCitationN} />
      {#if result.sources.length}
        <details class="sources">
          <summary
            >{result.sources.length} source{result.sources.length === 1 ? '' : 's'}</summary
          >
          <ul class="citelist">
            {#each result.sources as s (s.n)}
              <li>
                <button
                  type="button"
                  class="citelink"
                  class:active={activeCitationN === s.n}
                  onclick={() => onCitationClick?.(s.n)}
                  title="Open this source in the side panel"
                >
                  {s.citation}
                </button>
                {#if s.markers.length}
                  <span class="cite-marker" title={s.markers.join(', ')}>
                    <Icon name="triangle-alert" size={11} />
                  </span>
                {/if}
              </li>
            {/each}
          </ul>
        </details>
      {/if}
      <ClaimReview claims={result.flagged_claims} />
      <Provenance {result} />
      {#if usage}
        <div
          class="usage"
          title={`in ${usage.turn_input.toLocaleString()} · out ${usage.turn_output.toLocaleString()} · session ${usage.session_total.toLocaleString()} tokens`}
        >
          {tokens} tokens · {spend}
        </div>
      {/if}
    {:else}
      <Markdown source={answer} />
      {#if streaming}<span class="cursor">▍</span>{/if}
    {/if}
  </div>
</div>

<style>
  .turn {
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid var(--border);
    padding: 1rem 0;
  }
  .you {
    align-self: flex-end;
    max-width: min(72%, 640px);
    margin-bottom: 0.6rem;
    background: var(--surface-2);
    border-radius: 14px;
    border-bottom-right-radius: 4px;
    padding: 0.55rem 0.85rem;
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
  /* Sources render as a compact, collapsed-by-default bib-style list (user feedback 2026-07-14:
     the old full-card grid forced a long scroll). Each row opens the citation side panel; the
     inline [n] links in the answer prose remain the primary path. */
  .sources {
    margin-top: 0.7rem;
  }
  .sources > summary {
    cursor: pointer;
    color: var(--fg-2);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.15rem 0;
    user-select: none;
  }
  .sources > summary:hover {
    color: var(--fg);
  }
  .citelist {
    list-style: none;
    margin: 0.35rem 0 0;
    padding: 0;
    display: grid;
    gap: 0.1rem;
  }
  .citelist li {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    min-width: 0;
  }
  .citelink {
    font: inherit;
    font-size: 0.82rem;
    text-align: left;
    background: none;
    border: none;
    padding: 0.08rem 0;
    color: var(--fg);
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
  }
  .citelink:hover,
  .citelink:focus-visible {
    color: var(--accent);
    text-decoration: underline;
  }
  .citelink.active {
    color: var(--accent);
    font-weight: 600;
    text-decoration: underline;
  }
  .cite-marker {
    color: var(--warn-fg);
    display: inline-flex;
    flex: none;
  }
  .usage {
    margin-top: 0.6rem;
    font-size: 0.72rem;
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
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
  .cursor {
    animation: blink 1s steps(2) infinite;
  }
  .error {
    color: var(--warn-fg);
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }
</style>
