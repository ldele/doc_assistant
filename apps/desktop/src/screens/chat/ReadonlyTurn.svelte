<script lang="ts">
  // A reopened past turn (feature-conversation-history.md, Decision 4). Read-only: question +
  // answer with clickable citations, but no claim-review / provenance / composer — those aren't
  // reconstructed from AnswerRecord. Mirrors Turn.svelte's bubble so a reopened chat reads the
  // same as a live one.
  //
  // A8b (row 44, `[s10]`): with `turn`, it shows what the record holds besides the text: the
  // claims coloured as a live turn's are (G4), the decisions a person recorded on them, the
  // reviewer's recorded verdict, a records answer's lookup, and *How this answer was made*.
  import type { ConversationTurn, TurnScope } from '../../lib/core/types'
  import Markdown from './Markdown.svelte'
  import ScopeChip from './ScopeChip.svelte'
  import RecordsCard from './RecordsCard.svelte'
  import CarriedCard from './CarriedCard.svelte'
  import { CITED_TOOLTIP, WEAK_TOOLTIP, claimKind } from '../../lib/chat/claims'
  import { decisionLines, reviewLine } from '../../lib/chat/turnDetails'

  let {
    question,
    answer,
    scope = null,
    onCitationClick,
    activeCitationN = null,
    turn = null,
    onClaimClick,
    activeClaimN = null,
    onOpenRecord,
  }: {
    question: string
    answer: string
    // ADR-025 F2: replayed from the provenance record. A reopened scoped answer must carry the
    // same chip the live one did — otherwise the lie is merely deferred to the next session.
    scope?: TurnScope | null
    onCitationClick?: (n: number) => void
    activeCitationN?: number | null
    /** The whole recorded turn (A8b), for its claims, decisions, review, lookup and context. */
    turn?: ConversationTurn | null
    onClaimClick?: (claimN: number, sourceNs: number[]) => void
    activeClaimN?: number | null
    onOpenRecord?: (kind: 'document' | 'folder' | 'concept', id: string) => void
  } = $props()

  const claims = $derived(turn?.claims ?? [])
  const claimMarks = $derived(
    claims.flatMap((c) => {
      const kind = claimKind(c)
      return kind ? [{ key: String(c.n), kind, text: c.text, tip: kind === 'weak' ? WEAK_TOOLTIP : CITED_TOOLTIP }] : []
    }),
  )
  const decisions = $derived(decisionLines(claims))
  const showMade = $derived(Boolean(turn && (turn.context || turn.chat_mode)))
  function clickClaim(key: string): void {
    const c = claims.find((x) => String(x.n) === key)
    if (c && c.sources.length > 0) onClaimClick?.(c.n, c.sources)
  }
</script>

<div class="turn">
  <div class="you">
    <span class="who">You</span>
    <p>{question}</p>
  </div>
  <div class="assistant">
    {#if scope}
      <ScopeChip {scope} />
    {/if}
    <Markdown
      source={answer}
      {onCitationClick}
      {activeCitationN}
      allowedLinks={[]}
      claims={claimMarks}
      activeClaimKey={activeClaimN === null ? null : String(activeClaimN)}
      onClaimClick={clickClaim}
    />
    {#if turn?.records}
      <RecordsCard records={turn.records} {onOpenRecord} />
    {/if}
    {#if decisions.length > 0 || turn?.review}
      <section class="recorded" aria-label="As recorded">
        {#if decisions.length > 0}
          <p><span class="muted">Your decisions on its claims:</span> {decisions.join(' · ')}</p>
        {/if}
        {#if turn?.review}
          <p>{reviewLine(turn.review)}</p>
        {/if}
      </section>
    {/if}
    {#if showMade && turn}
      <details class="made">
        <summary>How this answer was made</summary>
        <CarriedCard context={turn.context ?? null} chatMode={turn.chat_mode} promptVersion={turn.prompt_version ?? null} />
      </details>
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
  .recorded {
    margin-top: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 0.45rem 0.75rem;
    font-size: 0.8rem;
    max-width: 40rem;
  }
  .recorded p {
    margin: 0.15rem 0;
  }
  .made {
    margin-top: 0.5rem;
    font-size: 0.82rem;
    color: var(--fg-2);
  }
  .made summary {
    cursor: pointer;
    user-select: none;
  }
</style>
