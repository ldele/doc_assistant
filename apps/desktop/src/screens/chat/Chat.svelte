<script lang="ts">
  // The Chat workspace: the transcript (read-only replay, resumed history, or live turns) plus the
  // composer footer.
  //
  // Reads live turn state straight from `chat.svelte.ts` — including the two DOM refs it binds
  // (`chat.convoEl`, `chat.taEl`), which live in the module precisely because App's own handlers
  // (`newConversation`, `resumeConversation`) still focus the composer after this pane mounted it.
  //
  // Everything conversation-*view* related (`viewing`, `viewedConvo`, `resumedHistory`) and every
  // action that spans domains (send, compare, export, resume) arrives as a prop: those read or
  // write state App owns, so that coupling stays visible there rather than being reached for here.
  //
  // A8a (docs/specs/chat-interface-pass.md): a chat's mode and folders are chosen on the start
  // screen, then fixed and named by the chat line at the top; the composer holds the text box and
  // *Send* only (plus *Test override* while an override is set).
  import Icon from '../../shell/Icon.svelte'
  import Turn from './Turn.svelte'
  import ReadonlyTurn from './ReadonlyTurn.svelte'
  import CompareCard from './CompareCard.svelte'
  import StartScreen from './StartScreen.svelte'
  import ChatLine from './ChatLine.svelte'
  import { autogrow, chat, onConvoScroll } from '../../lib/chat/chat.svelte'
  import { shell } from '../../lib/shell/shell.svelte'
  import { outstandingSteps } from '../../lib/settings/setup'
  import type {
    ChatMode,
    ChatSetup,
    ConversationDetail,
    ConversationTurn,
    LibraryFolder,
  } from '../../lib/core/types'
  import type { FolderChoice } from '../../lib/chat/setup'
  import { tooltip } from '../../lib/shell/tooltip'

  interface Props {
    viewing: string | null
    viewedConvo: ConversationDetail | null
    resumedHistory: ConversationDetail | null
    folders: LibraryFolder[]
    libraryCount: number
    modes: ChatMode[]
    /** The start screen's choice, bindable: the chat's first turn stores it. */
    startModeId: string
    startFolders: FolderChoice
    startNotice: string
    savedNote: string
    /** The live chat's setup once it has a turn or was continued from another; else null. */
    liveSetup: ChatSetup | null
    /** The exchanges the live chat carried in, as they were answered. */
    carried: ConversationTurn[]
    /** Only true while a retrieval-affecting override is set — with none, both sides of a compare
     *  retrieve identically and the button is dead weight. */
    hasRetrievalOverride: boolean
    onSend: () => void
    onKey: (e: KeyboardEvent) => void
    onCompare: () => void
    onUseSample: (q: string) => void
    onResume: () => void
    onBackToCurrent: () => void
    onMakeDefault: () => void
    onContinueNew: (sessionId: string, setup: ChatSetup | null) => void
    onOpenChat: (sessionId: string) => void
    /** A8b: a records answer's row: open its document, folder or concept. */
    onOpenRecord: (kind: 'document' | 'folder' | 'concept', id: string) => void
  }
  let {
    viewing, viewedConvo, resumedHistory, folders, libraryCount, modes,
    startModeId = $bindable(), startFolders = $bindable(), startNotice, savedNote,
    liveSetup, carried,
    hasRetrievalOverride,
    onSend, onKey, onCompare, onUseSample, onResume, onBackToCurrent, onMakeDefault,
    onContinueNew, onOpenChat, onOpenRecord,
  }: Props = $props()

  // The live chat's line: its stored setup, else (a reopened chat continued, before its first
  // new turn has been read back) the setup its detail came with.
  const liveLine = $derived(liveSetup ?? resumedHistory?.setup ?? null)
  const showStart = $derived(
    chat.turns.length === 0 && !resumedHistory && !liveSetup?.continued_from,
  )
  function citation(turnKey: string, n: number): void {
    chat.activeCitation = { turnKey, n }
    chat.activeClaim = null
  }
  // A8b: a coloured claim opens its first cited passage, and is drawn darker while it is open.
  function claimOpened(turnKey: string, claimN: number, sourceNs: number[]): void {
    chat.activeCitation = { turnKey, n: sourceNs[0], ns: sourceNs }
    chat.activeClaim = { turnKey, n: claimN }
  }

  // The outstanding first-run steps (ADR-034), shaped by the same tested helper the Settings
  // panel uses so the two surfaces can never disagree about what is left.
  const setupSteps = $derived(outstandingSteps(shell.setup))
</script>

<section class="conversation" bind:this={chat.convoEl} onscroll={onConvoScroll}>
  {#if viewing && viewedConvo}
    {#if viewedConvo.setup}
      <ChatLine
        setup={viewedConvo.setup}
        activeCitation={chat.activeCitation}
        onContinueNew={() => onContinueNew(viewedConvo.session_id, viewedConvo.setup)}
        {onOpenChat}
        onCitationClick={citation}
        onClaimClick={claimOpened}
      />
    {/if}
    <p class="readonly-note">
      Viewing a past conversation (read-only).
      <button class="linkish" onclick={onResume} use:tooltip={'Carry on this conversation from where it ended'}>Continue this chat</button>
      ·
      <button class="linkish" onclick={onBackToCurrent} use:tooltip={'Leave this past conversation and return to the one you were in'}>Back to current chat</button>
    </p>
    {#each viewedConvo.turns as t (t.record_id)}
      <ReadonlyTurn
        question={t.question}
        answer={t.answer}
        scope={t.scope}
        onCitationClick={(n) => citation(t.record_id, n)}
        activeCitationN={chat.activeCitation?.turnKey === t.record_id ? chat.activeCitation.n : null}
        turn={t}
        onClaimClick={(cn, ns) => claimOpened(t.record_id, cn, ns)}
        activeClaimN={chat.activeClaim?.turnKey === t.record_id ? chat.activeClaim.n : null}
        {onOpenRecord}
      />
    {/each}
  {:else}
    {#if liveLine && !showStart}
      <ChatLine
        setup={liveLine}
        {carried}
        activeCitation={chat.activeCitation}
        onContinueNew={() => onContinueNew(chat.sessionId, liveLine)}
        {onOpenChat}
        onCitationClick={citation}
        onClaimClick={claimOpened}
      />
    {/if}
    {#if resumedHistory}
      <!-- A8a, open question 1: a reopened chat continued remembers its own exchanges (fitted to
           the model's window), superseding the resume spec's empty memory. -->
      <p class="readonly-note resumed">
        Continuing <strong>{resumedHistory.title}</strong>. Its earlier turns are shown for
        reference, and new questions remember them, fitted to the model's window.
      </p>
      {#each resumedHistory.turns as t (t.record_id)}
        <ReadonlyTurn
          question={t.question}
          answer={t.answer}
          scope={t.scope}
          onCitationClick={(n) => citation(t.record_id, n)}
          activeCitationN={chat.activeCitation?.turnKey === t.record_id ? chat.activeCitation.n : null}
          turn={t}
          onClaimClick={(cn, ns) => claimOpened(t.record_id, cn, ns)}
          activeClaimN={chat.activeClaim?.turnKey === t.record_id ? chat.activeClaim.n : null}
          {onOpenRecord}
        />
      {/each}
      <div class="resume-divider"><span>continuing below</span></div>
    {/if}
    {#if shell.status === 'ready' && shell.setup && !shell.setup.ready}
      <!-- ADR-034: the setup banner replaced a documents-only one. A first-run install needs an
           answer engine *and* documents, and the backend already knows which are missing — so the
           card lists exactly what is left instead of naming only the half it used to check. -->
      <div class="banner">
        <span class="state-mark"><Icon name="library" size={26} /></span>
        <strong>{setupSteps.length === 1 ? 'One step to go' : 'Two steps to get started'}</strong>
        <ul class="todo">
          {#each setupSteps as step (step.id)}
            <li><strong>{step.title}</strong> — {step.detail}</li>
          {/each}
        </ul>
        <button class="btn btn-primary" onclick={() => (shell.showSettings = true)} use:tooltip={'Open Settings at the steps that are left'}>Finish setup…</button>
      </div>
    {/if}
    {#if showStart}
      <!-- A8a: the start screen shows at 0 documents too (folders empty, *All documents · 0*),
           under the first-run steps when they are left; the first question then gets today's
           empty-library answer. -->
      <StartScreen
        {modes}
        {folders}
        {libraryCount}
        bind:modeId={startModeId}
        bind:choice={startFolders}
        notice={startNotice}
        {savedNote}
        {onUseSample}
        {onMakeDefault}
      />
    {/if}
    {#each chat.turns as t (t.id)}
      <Turn
        question={t.question}
        answer={t.answer}
        result={t.result}
        streaming={t.streaming}
        error={t.error}
        onCitationClick={(n) => citation(String(t.id), n)}
        activeCitationN={chat.activeCitation?.turnKey === String(t.id) ? chat.activeCitation.n : null}
        onClaimClick={(cn, ns) => claimOpened(String(t.id), cn, ns)}
        activeClaimN={chat.activeClaim?.turnKey === String(t.id) ? chat.activeClaim.n : null}
        {onOpenRecord}
      />
    {/each}
    {#if chat.compareResult}
      <CompareCard result={chat.compareResult} onClose={() => (chat.compareResult = null)} />
    {/if}
  {/if}
</section>

<footer>
  {#if viewing}
    <div class="viewing-bar">
      <button class="btn grow" onclick={onBackToCurrent}
        use:tooltip={'Leave this past conversation and return to the one you were in'}
        ><Icon name="arrow-left" size={15} /> Back to current chat</button
      >
      <button class="btn btn-primary grow" onclick={onResume}
        use:tooltip={'Carry on this conversation from where it ended'}
        ><Icon name="rotate-ccw" size={15} /> Continue this chat</button
      >
    </div>
  {:else}
    <textarea
      bind:this={chat.taEl}
      bind:value={chat.input}
      onkeydown={onKey}
      oninput={autogrow}
      placeholder="Ask your documents…  (Enter to send, Shift+Enter for newline)"
      rows="2"
      disabled={chat.sending}
    ></textarea>
    {#if hasRetrievalOverride}
      <button
        class="btn"
        onclick={onCompare}
        disabled={chat.sending || chat.comparing || chat.input.trim() === ''}
        use:tooltip={'See how your override changes retrieval for this question: locked defaults vs override, sources only, no answer ($0)'}
        type="button"
      >
        {chat.comparing ? 'Comparing…' : 'Test override'}
      </button>
    {/if}
    <button class="btn btn-primary send" onclick={onSend} disabled={chat.sending || chat.input.trim() === ''} aria-busy={chat.sending} use:tooltip={chat.sending ? 'Waiting for the answer' : chat.input.trim() === '' ? 'Type a question first' : 'Ask your documents this question (Enter)'}>
      {#if chat.sending}<span class="spinner" aria-hidden="true"></span>{:else}Send{/if}
    </button>
  {/if}
</footer>

<style>
  .conversation {
    flex: 1;
    overflow-y: auto;
    /* Horizontal padding is load-bearing, not decoration. This pane scrolls, so its vertical
       scrollbar eats ~15px off the *content* box (width 790 → clientWidth 775). With `padding: … 0`
       every right-anchored child — the `.usage` token line, the source-evaluation score column,
       both `margin-left: auto` — ended exactly ON that boundary and rendered underneath the bar:
       "0 tokens · local" read as "0 tokens · loca". Keep a gutter wider than any scrollbar.
       `scrollbar-gutter: stable` reserves the space even when no bar is showing, so a turn that
       grows past the fold no longer shifts the whole column left by 15px as it appears. */
    padding: var(--space-2) var(--space-3);
    scrollbar-gutter: stable;
  }
  /* The first-run state: one centered, mark-led layout (V2); the start screen draws its own. */
  .banner {
    max-width: 540px;
    margin: var(--space-6) auto 0;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .state-mark {
    flex: none;
    width: 46px;
    height: 46px;
    border-radius: 12px;
    background: var(--surface-2);
    color: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: var(--space-4);
  }
  .banner strong {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    font-weight: 600;
    color: var(--fg);
    margin: 0;
  }
  /* The outstanding-steps list (ADR-034): left-aligned inside the centered card, because these
     are instructions to read in order, not a headline. */
  .banner .todo {
    list-style: none;
    margin: var(--space-2) 0 var(--space-4);
    padding: 0;
    text-align: left;
    max-width: 46ch;
    display: grid;
    gap: var(--space-1);
    color: var(--fg-2);
    font-size: var(--text-sm);
    line-height: 1.55;
  }
  .banner .todo strong {
    font-family: inherit;
    font-size: inherit;
    color: var(--fg);
  }
  .readonly-note {
    font-size: 0.78rem;
    color: var(--fg-2);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    margin: 0 0 0.5rem;
  }
  .linkish {
    font: inherit;
    font-size: inherit;
    background: none;
    border: none;
    color: var(--accent);
    cursor: pointer;
    padding: 0;
    text-decoration: underline;
  }
  /* Resume banner: tinted with the accent so "continuing" reads distinct from "viewing". */
  .readonly-note.resumed {
    background: color-mix(in srgb, var(--accent) 8%, var(--surface));
    border-color: color-mix(in srgb, var(--accent) 25%, var(--border));
    color: var(--fg);
  }
  .readonly-note.resumed strong {
    font-weight: 600;
  }
  .resume-divider {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.4rem 0 0.8rem;
    color: var(--fg-2);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .resume-divider::before,
  .resume-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }
  .banner {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    box-shadow: var(--shadow-1);
    padding: var(--space-6) var(--space-5);
  }
  footer {
    display: flex;
    gap: var(--space-2);
    padding: var(--space-3) 0;
    border-top: 1px solid var(--border);
  }
  textarea {
    flex: 1;
    resize: none;
    font: inherit;
    padding: 0.5rem 0.6rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    min-height: 3.4rem;
    max-height: 160px;
    overflow-y: auto;
  }
  .viewing-bar {
    display: flex;
    gap: var(--space-2);
    width: 100%;
  }
  /* Layout only: the roles' look is app.css's (ADR-063). */
  .grow {
    flex: 1;
    justify-content: center;
    padding-block: var(--space-2);
  }
  .send {
    min-width: 4.4rem;
    justify-content: center;
  }
  .spinner {
    width: 0.95em;
    height: 0.95em;
    border: 2px solid var(--accent-fg);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .spinner {
      animation: none;
    }
  }
</style>
