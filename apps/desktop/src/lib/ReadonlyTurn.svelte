<script lang="ts">
  // A reopened past turn (feature-conversation-history.md, Decision 4). Read-only: question +
  // answer with clickable citations, but no claim-review / provenance / composer — those aren't
  // reconstructed from AnswerRecord. Mirrors Turn.svelte's bubble so a reopened chat reads the
  // same as a live one.
  import type { TurnScope } from './types'
  import Markdown from './Markdown.svelte'
  import Icon from './Icon.svelte'

  let {
    question,
    answer,
    scope = null,
    onCitationClick,
    activeCitationN = null,
  }: {
    question: string
    answer: string
    // ADR-025 F2: replayed from the provenance record. A reopened scoped answer must carry the
    // same chip the live one did — otherwise the lie is merely deferred to the next session.
    scope?: TurnScope | null
    onCitationClick?: (n: number) => void
    activeCitationN?: number | null
  } = $props()
</script>

<div class="turn">
  <div class="you">
    <span class="who">You</span>
    <p>{question}</p>
  </div>
  <div class="assistant">
    {#if scope}
      <div class="scopechip">
        <Icon name="folder" size={13} />
        {#if scope.folder_name}
          <span>
            Searched <strong>{scope.folder_name}</strong> only — {scope.doc_count} document{scope.doc_count ===
            1
              ? ''
              : 's'}, not the whole library.
          </span>
        {:else}
          <span>Scoped to a folder that no longer exists — <strong>nothing was searched</strong>.</span>
        {/if}
      </div>
    {/if}
    <Markdown source={answer} {onCitationClick} {activeCitationN} />
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
