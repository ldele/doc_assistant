<script lang="ts">
  // A reopened past turn (feature-conversation-history.md, Decision 4). Read-only: question +
  // answer with clickable citations, but no claim-review / provenance / composer — those aren't
  // reconstructed from AnswerRecord. Mirrors Turn.svelte's bubble so a reopened chat reads the
  // same as a live one.
  import Markdown from './Markdown.svelte'

  let {
    question,
    answer,
    onCitationClick,
    activeCitationN = null,
  }: {
    question: string
    answer: string
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
</style>
