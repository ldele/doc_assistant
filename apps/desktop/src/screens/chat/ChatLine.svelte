<script lang="ts">
  // The chat line (A8a, `[s1-chatline]`, `[s6-from]`, `[s6-carried]`): the chat's mode and its
  // folders with their document count, always shown, and *Continue in a new chat…* beside it. It
  // is the safeguard ADR-025's addendum puts in place of S9's unstored scope: a scope stored on
  // the chat cannot be forgotten while the line names it. A folder that holds nothing is named
  // (F2). A chat continued from another says so, links to it, and shows the exchanges it
  // carried, folded, each opening as it was answered.
  import Icon from '../../shell/Icon.svelte'
  import ReadonlyTurn from './ReadonlyTurn.svelte'
  import type { ChatSetup, ConversationTurn } from '../../lib/core/types'
  import { carriedText, chatLineText, continuedFromText, emptyFolders } from '../../lib/chat/setup'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    setup,
    carried = [],
    activeCitation = null,
    onContinueNew,
    onOpenChat,
    onCitationClick,
  }: {
    setup: ChatSetup
    /** The exchanges this chat carried, as answered (read from the old chat's records). */
    carried?: ConversationTurn[]
    activeCitation?: { turnKey: string; n: number } | null
    onContinueNew: () => void
    onOpenChat: (sessionId: string) => void
    onCitationClick: (turnKey: string, n: number) => void
  } = $props()

  let open = $state(false)
  const empty = $derived(emptyFolders(setup.scope))
  const from = $derived(setup.continued_from)
  const shownCount = $derived(carried.length || setup.carried_exchanges)
</script>

<div class="chatline">
  <div class="line">
    <span class="what">
      <Icon name={setup.scope ? 'folder' : 'library'} size={13} />
      <span>{chatLineText(setup.mode.name, setup.scope)}</span>
      {#if !setup.mode.evaluated}<span class="muted">· not evaluated</span>{/if}
    </span>
    <button class="btn continue" onclick={onContinueNew} type="button" use:tooltip={'Change the mode or the folders: open a new chat that can carry this one\'s exchanges; this chat stays as it is'}>
      Continue in a new chat…
    </button>
  </div>
  {#if empty.length > 0}
    <p class="note warn">
      {empty.join(', ')} {empty.length === 1 ? 'holds' : 'hold'} no documents, so {empty.length === 1 ? 'it adds' : 'they add'} nothing to the search.
    </p>
  {/if}
  {#if from}
    <p class="note muted">
      <button class="linkish" onclick={() => onOpenChat(from.session_id)} type="button" use:tooltip={'Open the chat this one was continued from; it is unchanged'}>
        {continuedFromText(from.title, from.mode_name, from.scope)}
      </button>
    </p>
  {/if}
  {#if shownCount > 0}
    <button class="carried" aria-expanded={open} onclick={() => (open = !open)} type="button" use:tooltip={open ? 'Fold the carried exchanges' : 'Show the exchanges this chat carried, as they were answered'}>
      <Icon name={open ? 'chevron-down' : 'chevron-right'} size={13} />
      {carriedText(shownCount)}
    </button>
    {#if open}
      <div class="carriedlist">
        {#if carried.length === 0}
          <p class="note muted">The carried exchanges are in the chat it was continued from.</p>
        {/if}
        {#each carried as t (t.record_id)}
          <ReadonlyTurn
            question={t.question}
            answer={t.answer}
            scope={t.scope}
            onCitationClick={(n) => onCitationClick(t.record_id, n)}
            activeCitationN={activeCitation?.turnKey === t.record_id ? activeCitation.n : null}
            turn={t}
            onClaimClick={(_cn, n) => onCitationClick(t.record_id, n)}
          />
        {/each}
      </div>
    {/if}
  {/if}
</div>

<style>
  .chatline {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    margin: 0 0 var(--space-3);
  }
  .line {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .what {
    flex: 1;
    min-width: 12rem;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    color: var(--fg);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
  }
  .continue {
    flex: none;
  }
  .note {
    margin: 0;
    font-size: 0.76rem;
  }
  .warn {
    color: var(--warn-fg);
  }
  .linkish {
    font: inherit;
    background: none;
    border: none;
    color: var(--accent);
    cursor: pointer;
    padding: 0;
    text-align: left;
    text-decoration: underline;
  }
  .carried {
    align-self: flex-start;
    font: inherit;
    font-size: 0.8rem;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    cursor: pointer;
    color: var(--fg);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
  }
  .carriedlist {
    border-left: 2px solid var(--border);
    padding-left: var(--space-3);
    margin: var(--space-1) 0 var(--space-2);
  }
</style>
