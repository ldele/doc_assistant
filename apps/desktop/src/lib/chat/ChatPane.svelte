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
  import Icon from '../shell/Icon.svelte'
  import Turn from './Turn.svelte'
  import ReadonlyTurn from './ReadonlyTurn.svelte'
  import CompareCard from './CompareCard.svelte'
  import { autogrow, chat, onConvoScroll } from './chat.svelte'
  import { shell } from '../shell/shell.svelte'
  import { outstandingSteps } from '../settings/setup'
  import type { ConversationDetail, LibraryFolder } from '../core/types'

  interface Props {
    viewing: string | null
    viewedConvo: ConversationDetail | null
    resumedHistory: ConversationDetail | null
    folders: LibraryFolder[]
    /** Bindable: the `<select>` writes it directly, exactly as before the extraction. */
    chatScopeFolderId: string | null
    /** Only true while a retrieval-affecting override is set — with none, both sides of a compare
     *  retrieve identically and the button is dead weight. */
    hasRetrievalOverride: boolean
    sampleQuestions: string[]
    onSend: () => void
    onKey: (e: KeyboardEvent) => void
    onCompare: () => void
    onUseSample: (q: string) => void
    onResume: () => void
    onBackToCurrent: () => void
  }
  let {
    viewing, viewedConvo, resumedHistory, folders,
    chatScopeFolderId = $bindable(),
    hasRetrievalOverride, sampleQuestions,
    onSend, onKey, onCompare, onUseSample, onResume, onBackToCurrent,
  }: Props = $props()

  // The outstanding first-run steps (ADR-034), shaped by the same tested helper the Settings
  // panel uses so the two surfaces can never disagree about what is left.
  const setupSteps = $derived(outstandingSteps(shell.setup))
</script>

<section class="conversation" bind:this={chat.convoEl} onscroll={onConvoScroll}>
  {#if viewing && viewedConvo}
    <p class="readonly-note">
      Viewing a past conversation (read-only).
      <button class="linkish" onclick={onResume}>Continue this chat</button>
      ·
      <button class="linkish" onclick={onBackToCurrent}>Back to current chat</button>
    </p>
    {#each viewedConvo.turns as t (t.record_id)}
      <ReadonlyTurn
        question={t.question}
        answer={t.answer}
        scope={t.scope}
        onCitationClick={(n) => (chat.activeCitation = { turnKey: t.record_id, n })}
        activeCitationN={chat.activeCitation?.turnKey === t.record_id ? chat.activeCitation.n : null}
      />
    {/each}
  {:else}
    {#if resumedHistory}
      <p class="readonly-note resumed">
        Continuing <strong>{resumedHistory.title}</strong> · earlier chat.turns are shown for
        reference. New questions start fresh — grounded in your corpus, not the old chat.
      </p>
      {#each resumedHistory.turns as t (t.record_id)}
        <ReadonlyTurn
          question={t.question}
          answer={t.answer}
          scope={t.scope}
          onCitationClick={(n) => (chat.activeCitation = { turnKey: t.record_id, n })}
          activeCitationN={chat.activeCitation?.turnKey === t.record_id ? chat.activeCitation.n : null}
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
        <button class="primary" onclick={() => (shell.showSettings = true)}>Finish setup…</button>
      </div>
    {:else if chat.turns.length === 0 && !resumedHistory}
      <div class="empty">
        <span class="state-mark"><Icon name="book-open-text" size={26} /></span>
        <h2>Ask your library a question</h2>
        <p>
          Every answer is grounded in your own documents, with inline citations, provenance,
          and per-claim review.
        </p>
        <div class="chips">
          {#each sampleQuestions as q}
            <button class="chip" onclick={() => onUseSample(q)}>{q}</button>
          {/each}
        </div>
      </div>
    {/if}
    {#each chat.turns as t (t.id)}
      <Turn
        question={t.question}
        answer={t.answer}
        result={t.result}
        streaming={t.streaming}
        error={t.error}
        onCitationClick={(n) => (chat.activeCitation = { turnKey: String(t.id), n })}
        activeCitationN={chat.activeCitation?.turnKey === String(t.id) ? chat.activeCitation.n : null}
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
      <button class="back" onclick={onBackToCurrent}
        ><Icon name="arrow-left" size={15} /> Back to current chat</button
      >
      <button class="resume" onclick={onResume}
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
        class="compare"
        onclick={onCompare}
        disabled={chat.sending || chat.comparing || chat.input.trim() === ''}
        title="See how your override changes retrieval for this question: locked defaults vs override, sources only, no answer ($0)"
        type="button"
      >
        {chat.comparing ? 'Comparing…' : 'Test override'}
      </button>
    {/if}
    {#if folders.length > 0}
      <!-- ADR-025 F2 scope selector. Session-sticky, never persisted (see chatScopeFolderId).
           "All documents" is always the first option, so returning to the whole library is
           one click and never a hidden state. -->
      <label class="scopepick" class:scoped={chatScopeFolderId !== null}>
        <Icon name="folder" size={13} />
        <select
          bind:value={chatScopeFolderId}
          disabled={chat.sending}
          aria-label="Search scope"
          title="Which documents this question searches"
        >
          <option value={null}>All documents</option>
          {#each folders as f (f.id)}
            <option value={f.id}>{f.name} ({f.doc_count})</option>
          {/each}
        </select>
      </label>
    {/if}
    <button class="send" onclick={onSend} disabled={chat.sending || chat.input.trim() === ''} aria-busy={chat.sending}>
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
  /* Empty + first-run states share one centered, mark-led layout (V2). */
  .empty,
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
  .empty h2,
  .banner strong {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    font-weight: 600;
    color: var(--fg);
    margin: 0;
  }
  .empty p {
    color: var(--fg-2);
    font-size: var(--text-sm);
    line-height: 1.6;
    max-width: 46ch;
    margin: var(--space-2) 0 var(--space-4);
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
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    justify-content: center;
  }
  .chip {
    font-size: var(--text-sm);
    color: var(--accent);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: var(--space-2) var(--space-3);
  }
  .chip:hover {
    border-color: var(--accent);
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
  .banner .primary {
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
    font-weight: 600;
    padding: 0.45rem 1.1rem;
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
  button {
    font: inherit;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg);
    padding: 0 1rem;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .viewing-bar {
    display: flex;
    gap: 0.5rem;
    width: 100%;
  }
  .back {
    flex: 1;
    padding: 0.6rem;
    color: var(--fg-2);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
  }
  .resume {
    flex: 1;
    padding: 0.6rem;
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
  }
  .send {
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
    font-weight: 600;
    min-width: 4.4rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  /* ADR-025 F2 scope selector — reads as a quiet control until a scope is set, then it is
     tinted so a narrowed conversation is visible without opening anything. */
  .scopepick {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    flex: none;
    padding: 0.25rem 0.4rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--fg-2);
    background: var(--surface);
  }
  .scopepick.scoped {
    color: var(--accent);
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
  }
  .scopepick select {
    font: inherit;
    font-size: 0.78rem;
    max-width: 11rem;
    border: none;
    background: none;
    color: inherit;
    cursor: pointer;
  }
  .scopepick select:focus {
    outline: none;
  }
  .scopepick select:disabled {
    cursor: not-allowed;
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
  .compare {
    font-size: 0.82rem;
    white-space: nowrap;
    color: var(--fg-2);
  }
</style>
