<script lang="ts">
  // The *Detect proposals* section of Manage keywords (feature-tag-families.md, PR-2): the Detect
  // button, its error, and the near-duplicate families it found, each with Accept and Dismiss.
  // Owns nothing — the proposals and the run are App's, and every click is handed straight up.
  import type { KeywordFamilyProposal } from '../../lib/core/types'
  import Icon from '../../shell/Icon.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    proposals,
    detecting,
    detectError,
    onDetect,
    onAcceptProposal,
    onDismissProposal,
  }: {
    proposals: KeywordFamilyProposal[] // zero-LLM detection results (PR-2); [] until Detect runs
    detecting: boolean
    detectError: string | null
    onDetect: () => void
    onAcceptProposal: (p: KeywordFamilyProposal) => void
    onDismissProposal: (canonical: string) => void
  } = $props()
</script>

<section class="block">
  <h3>Detect proposals</h3>
  <p class="hint">
    Zero-LLM, on request — checks keyword spelling (morphology) and meaning (bge embedding)
    for near-duplicates. Nothing is written until you accept a proposal below.
  </p>
  <button class="secondary" onclick={onDetect} disabled={detecting} type="button" use:tooltip={'Look for keywords spelled alike or meaning the same; nothing is written'}>
    {detecting ? 'Detecting…' : 'Detect'}
  </button>
  {#if detectError}
    <p class="error">{detectError}</p>
  {/if}
  {#if proposals.length > 0}
    <div class="proplist" role="group" aria-label="Detected family proposals">
      {#each proposals as p (p.canonical)}
        <div class="proprow">
          <span class="proptier" class:embedding={p.tier === 'embedding'}>
            {p.tier === 'morphological' ? 'spelling' : 'meaning'}
          </span>
          <span class="proptext">
            <strong>{p.canonical}</strong>
            <span class="propmembers">+ {p.members.join(', ')}</span>
          </span>
          <span class="propconf">{Math.round(p.confidence * 100)}%</span>
          <button class="propaccept" onclick={() => onAcceptProposal(p)} type="button" use:tooltip={'Make these keywords one family under the first name'}>
            Accept
          </button>
          <button
            class="iconbtn"
            onclick={() => onDismissProposal(p.canonical)}
            aria-label="Dismiss proposal for {p.canonical}"
            use:tooltip={'Hide this proposal; nothing is merged'}
            type="button"
          >
            <Icon name="x" size={14} />
          </button>
        </div>
      {/each}
    </div>
  {:else if !detecting && !detectError}
    <p class="hint">No proposals yet — run Detect to check for near-duplicate keywords.</p>
  {/if}
</section>

<style>
  .iconbtn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.2rem;
    border: none;
    background: none;
    color: var(--fg-2);
    border-radius: 6px;
    cursor: pointer;
    flex: none;
  }
  .iconbtn:hover {
    color: var(--fg);
    background: var(--surface-2);
  }
  .block h3 {
    margin: 0 0 0.3rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--fg-2);
  }
  .hint {
    margin: 0 0 0.5rem;
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .secondary {
    font: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--fg);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
    align-self: flex-start;
  }
  .secondary:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .secondary:disabled {
    opacity: 0.6;
    cursor: wait;
  }
  .error {
    margin: 0.4rem 0 0;
    font-size: 0.78rem;
    color: var(--danger);
  }
  .proplist {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin-top: 0.5rem;
  }
  .proprow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.55rem;
  }
  .proptier {
    flex: none;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--fg-2);
    background: var(--surface-2);
    border-radius: 999px;
    padding: 0.1rem 0.45rem;
  }
  .proptier.embedding {
    color: var(--accent-fg);
    background: var(--accent);
  }
  .proptext {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    font-size: 0.82rem;
    overflow: hidden;
  }
  .propmembers {
    color: var(--fg-2);
    font-size: 0.75rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .propconf {
    flex: none;
    font-size: 0.7rem;
    font-variant-numeric: tabular-nums;
    color: var(--fg-2);
  }
  .propaccept {
    flex: none;
    font: inherit;
    font-size: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--accent);
    background: none;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
  }
  .propaccept:hover {
    color: var(--accent-fg);
    background: var(--accent);
  }
</style>
