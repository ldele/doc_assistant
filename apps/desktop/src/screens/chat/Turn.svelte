<script lang="ts">
  import type { TurnResult } from '../../lib/core/types'
  import Markdown from './Markdown.svelte'
  import ClaimReview from './ClaimReview.svelte'
  import Provenance from './Provenance.svelte'
  import SourceEvaluation from './SourceEvaluation.svelte'
  import Icon from '../../shell/Icon.svelte'
  import ScopeChip from './ScopeChip.svelte'
  import RecordsCard from './RecordsCard.svelte'
  import { usageLabel } from '../../lib/chat/usage'
  import { CITED_TOOLTIP, WEAK_TOOLTIP, claimKind } from '../../lib/chat/claims'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    question,
    answer,
    result,
    streaming = false,
    error = null,
    onCitationClick,
    activeCitationN = null,
    onClaimClick,
    activeClaimN = null,
    onOpenRecord,
  }: {
    question: string
    answer: string
    result: TurnResult | null
    streaming?: boolean
    error?: string | null
    onCitationClick?: (n: number) => void
    activeCitationN?: number | null
    /** A8b: a coloured claim was clicked: open every passage it cites. */
    onClaimClick?: (claimN: number, sourceNs: number[]) => void
    activeClaimN?: number | null
    onOpenRecord?: (kind: 'document' | 'folder' | 'concept', id: string) => void
  } = $props()

  // A8b, G4: the claims coloured by their marker only; an uncited one stays plain.
  const claimMarks = $derived(
    (result?.claims ?? []).flatMap((c) => {
      const kind = claimKind(c)
      return kind ? [{ key: String(c.n), kind, text: c.text, tip: kind === 'weak' ? WEAK_TOOLTIP : CITED_TOOLTIP }] : []
    }),
  )
  function clickClaim(key: string): void {
    const c = result?.claims?.find((x) => String(x.n) === key)
    if (c && c.sources.length > 0) onClaimClick?.(c.n, c.sources)
  }

  // Glanceable turn cost — the same numbers the Provenance panel spells out, surfaced up front
  // so the reader sees the spend without expanding anything (inform, don't gate).
  const usage = $derived(result?.usage ?? null)
  const usageText = $derived(usageLabel(usage))
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
        <ScopeChip scope={result.scope} />
      {/if}
      <Markdown
        source={result.answer}
        {onCitationClick}
        {activeCitationN}
        allowedLinks={result.source_links ?? []}
        claims={claimMarks}
        activeClaimKey={activeClaimN === null ? null : String(activeClaimN)}
        onClaimClick={clickClaim}
      />
      {#if result.records}
        <RecordsCard records={result.records} {onOpenRecord} />
      {/if}
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
                  use:tooltip={'Open this source in the side panel'}
                >
                  {s.citation}
                </button>
                {#if s.markers.length || s.addressed_to_ai?.length}
                  <span class="cite-marker" title={[...s.markers, ...(s.addressed_to_ai?.length ? ['contains text addressed to an AI'] : [])].join(', ')}>
                    <Icon name="triangle-alert" size={11} />
                  </span>
                {/if}
              </li>
            {/each}
          </ul>
        </details>
      {/if}
      {#if result.source_eval && result.sources.length}
        <!-- ADR-027 D3: always-on per-source epistemic evaluation, below the answer. Advisory —
             it never gates the answer and is independent of the epistemics-influence toggle. -->
        <SourceEvaluation sources={result.sources} summary={result.source_eval} />
      {/if}
      <ClaimReview claims={result.flagged_claims} />
      <Provenance {result} />
      {#if usage}
        <div
          class="usage"
          title={`in ${usage.turn_input.toLocaleString()} · out ${usage.turn_output.toLocaleString()} · session ${usage.session_total.toLocaleString()} tokens`}
        >
          {usageText}
        </div>
      {/if}
    {:else}
      <Markdown source={answer} allowedLinks={[]} />
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
