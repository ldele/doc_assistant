<script lang="ts">
  import type {
    CompareResult,
    ConversationDetail,
    ConversationSummary,
    Health,
    LibraryDocument,
    RagOverrides,
    TurnResult,
  } from './lib/types'
  import {
    compareRetrieval,
    getConversation,
    getHealth,
    exportConversation,
    listConversations,
    listLibraryDocuments,
    streamChat,
    updateConversationMeta,
  } from './lib/api'
  import Turn from './lib/Turn.svelte'
  import ReadonlyTurn from './lib/ReadonlyTurn.svelte'
  import Settings from './lib/Settings.svelte'
  import SourcePanel from './lib/SourcePanel.svelte'
  import Sidebar from './lib/Sidebar.svelte'
  import LibraryBrowser from './lib/LibraryBrowser.svelte'
  import CompareCard from './lib/CompareCard.svelte'
  import Icon from './lib/Icon.svelte'
  import appMark from './assets/brand/app-mark.png'

  interface TurnState {
    id: number
    question: string
    answer: string
    result: TurnResult | null
    streaming: boolean
    error: string | null
  }

  function freshSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  }
  // $state: the sidebar's "current" marker + the citation-source derivation read this, so a fresh
  // id from ↻ New must trigger updates.
  let sessionId = $state(freshSessionId())

  let health = $state<Health | null>(null)
  let status = $state<'connecting' | 'ready' | 'down'>('connecting')
  let turns = $state<TurnState[]>([])
  let input = $state('')
  let sending = $state(false)
  let showSettings = $state(false)
  // ADR-010: the RAG-sandbox overrides for this app session. In-memory only — a fresh
  // launch always starts from {} (locked defaults), never persisted to disk.
  let overrides = $state<RagOverrides>({})
  let nextId = 0

  // A/B-compare (U6): a per-turn retrieval diff (locked defaults vs the session override). $0 —
  // retrieval only, no answer generation. The result is an ephemeral card, not a chat turn.
  let compareResult = $state<CompareResult | null>(null)
  let comparing = $state(false)
  // The Test-override button only exists while a retrieval-affecting override is set — with none,
  // both sides retrieve identically and the button is dead weight (2026-07-13 UX review). Settings
  // writes these fields only when touched; Reset returns overrides to {}.
  const hasRetrievalOverride = $derived(
    overrides.top_k != null || overrides.use_multi_query != null,
  )

  // Conversation history (feature-conversation-history.md). `viewing` is the session_id shown as a
  // read-only transcript; `null` means the live chat (composer + claims bound to `sessionId`).
  let conversations = $state<ConversationSummary[]>([])
  let viewing = $state<string | null>(null)
  let viewedConvo = $state<ConversationDetail | null>(null)
  // Resume (fresh-context): a reopened past chat the user chose to *continue*. Its turns render
  // read-only above the composer for reference; `sessionId` is switched to it so new turns thread
  // to the same conversation and persist. The in-memory backend session starts fresh (empty
  // history), so new questions are standalone corpus queries — no replay of the old turns.
  let resumedHistory = $state<ConversationDetail | null>(null)
  let sidebarOpen = $state(false) // mobile drawer

  // Library space (feature-library-browser.md, L1). `mode` swaps the sidebar list + main pane
  // between Chat and Library; the chat state (turns/viewing/sessionId) is untouched by the switch.
  let mode = $state<'chat' | 'library'>('chat')
  let documents = $state<LibraryDocument[]>([])
  let libraryDocId = $state<string | null>(null)
  let documentsLoaded = false

  // Which citation panel is open — keyed by a turn *key* (a live turn's id as string, or a past
  // turn's record_id) so a click resolves against the right turn in either mode.
  let activeCitation = $state<{ turnKey: string; n: number } | null>(null)
  const activeSource = $derived.by(() => {
    if (!activeCitation) return null
    // Read-only transcripts (a viewed chat, or a resumed chat's history) key by record_id;
    // a resumed chat also has live turns below, so fall through to those if not found here.
    const detail = viewedConvo ?? resumedHistory
    if (detail) {
      const t = detail.turns.find((t) => t.record_id === activeCitation!.turnKey)
      const s = t?.sources.find((s) => s.n === activeCitation!.n)
      // A rehydrated source is degraded — no markers/figures (not persisted). Shape it as a
      // SourceView so SourcePanel/SourceCard render it unchanged.
      if (s) {
        return { n: s.n, citation: s.citation, excerpt: s.excerpt, figure_id: null, chunk_key: null, markers: [] }
      }
    }
    const t = turns.find((t) => String(t.id) === activeCitation!.turnKey)
    return t?.result?.sources.find((s) => s.n === activeCitation!.n) ?? null
  })

  let convoEl = $state<HTMLElement | null>(null)
  let taEl = $state<HTMLTextAreaElement | null>(null)
  // Follow a streaming answer only while the reader is already at the bottom, so a long
  // response never yanks them away from something they scrolled up to re-read.
  let pinned = true

  // Re-pull /api/health after an ingest so the header chunk count + the empty-corpus banner
  // reflect the new corpus (the backend rebuilds the controller before reporting "done").
  async function refreshHealth(): Promise<void> {
    try {
      health = await getHealth()
      status = 'ready'
    } catch {
      // leave the prior health/status; a transient blip shouldn't blank the header
    }
  }

  // History is a sidecar read — a failure must never break the chat (inform, don't block).
  async function refreshConversations(): Promise<void> {
    try {
      conversations = await listConversations()
    } catch {
      // keep the prior list
    }
  }

  // Conversation management (pin / archive / soft-delete): PATCH the flag, then refresh the list.
  async function pinConversation(sid: string, pinned: boolean): Promise<void> {
    try {
      await updateConversationMeta(sid, { pinned })
      await refreshConversations()
    } catch (e) {
      console.error('pin failed', e)
    }
  }
  async function archiveConversation(sid: string, archived: boolean): Promise<void> {
    try {
      await updateConversationMeta(sid, { archived })
      await refreshConversations()
    } catch (e) {
      console.error('archive failed', e)
    }
  }
  async function renameConversation(sid: string, title: string): Promise<void> {
    try {
      await updateConversationMeta(sid, { title })
      await refreshConversations()
    } catch (e) {
      console.error('rename failed', e)
    }
  }

  // Resizable left sidebar. Width is a client-only view preference (like the theme toggle),
  // persisted in localStorage; clamped so it can't be dragged uselessly narrow or wide.
  const SIDEBAR_MIN = 200
  const SIDEBAR_MAX = 480
  function loadSidebarWidth(): number {
    try {
      const v = Number(localStorage.getItem('sidebarWidth'))
      return v >= SIDEBAR_MIN && v <= SIDEBAR_MAX ? v : 260
    } catch {
      return 260
    }
  }
  let sidebarWidth = $state(loadSidebarWidth())
  function startResize(e: PointerEvent): void {
    e.preventDefault()
    const onMove = (ev: PointerEvent) => {
      sidebarWidth = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, ev.clientX))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      try {
        localStorage.setItem('sidebarWidth', String(Math.round(sidebarWidth)))
      } catch {
        /* ignore — width just won't persist */
      }
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }
  async function deleteConversation(sid: string): Promise<void> {
    // Soft-delete is reversible, but there's no restore UI yet — confirm to avoid a mis-click.
    if (
      !window.confirm(
        'Delete this conversation? It is removed from your history; the underlying records are kept.',
      )
    )
      return
    try {
      await updateConversationMeta(sid, { deleted: true })
      // If the deleted chat is the one on screen (viewed, resumed, or live), start fresh.
      if (viewing === sid || resumedHistory?.session_id === sid || sessionId === sid) {
        newConversation()
      }
      await refreshConversations()
    } catch (e) {
      console.error('delete failed', e)
    }
  }

  // Readiness gate (PR-M4): the frozen sidecar takes a few seconds to load models before
  // it accepts requests. Poll /api/health until it answers (or give up after ~60s), then
  // load the conversation history.
  $effect(() => {
    let cancelled = false
    void (async () => {
      for (let i = 0; i < 60 && !cancelled; i++) {
        try {
          const h = await getHealth()
          if (!cancelled) {
            health = h
            status = 'ready'
            void refreshConversations()
          }
          return
        } catch {
          await new Promise((r) => setTimeout(r, 1000))
        }
      }
      if (!cancelled) status = 'down'
    })()
    return () => {
      cancelled = true
    }
  })

  // Keep the newest content in view as tokens stream in / a turn is added / a chat is opened —
  // but only when the reader is pinned to the bottom (see `pinned`).
  $effect(() => {
    const last = turns[turns.length - 1]
    void last?.answer
    void turns.length
    void viewing
    if (pinned && convoEl) convoEl.scrollTop = convoEl.scrollHeight
  })

  // A reader who has scrolled up to re-read is no longer pinned; snap back on once they
  // return to the bottom (small slack so it engages just before the exact edge).
  function onConvoScroll(): void {
    if (!convoEl) return
    pinned = convoEl.scrollHeight - convoEl.scrollTop - convoEl.clientHeight < 80
  }

  // Grow the composer with its content up to a cap, then let it scroll. Reset to the base
  // height after a send (measuring `scrollHeight` on empty content would keep the tall size).
  function autogrow(): void {
    if (!taEl) return
    taEl.style.height = 'auto'
    taEl.style.height = `${Math.min(taEl.scrollHeight, 160)}px`
  }
  function resetComposer(): void {
    if (taEl) taEl.style.height = 'auto'
  }

  // Sample questions for the empty state — corpus-agnostic openers that run one-click on any
  // library. Picking one only prefills the existing composer (no turn sent, no new behavior); the
  // reader still presses Send. Kept generic because the corpus topics aren't known at this layer.
  const sampleQuestions = [
    'What are the main themes across my documents?',
    'Where do my sources agree and disagree?',
    'What are the key findings, with citations?',
  ]
  function useSample(q: string): void {
    input = q
    taEl?.focus()
  }

  async function send(): Promise<void> {
    const text = input.trim()
    if (!text || sending) return
    input = ''
    resetComposer()
    pinned = true // sending jumps the reader to their own new turn
    sending = true
    const idx =
      turns.push({
        id: nextId++,
        question: text,
        answer: '',
        result: null,
        streaming: true,
        error: null,
      }) - 1
    try {
      for await (const ev of streamChat(text, sessionId, overrides)) {
        if (ev.event === 'token') turns[idx].answer += ev.data
        else if (ev.event === 'result') turns[idx].result = JSON.parse(ev.data) as TurnResult
        // `step` events are advisory; ignored for now.
      }
    } catch (e) {
      turns[idx].error = String(e)
    } finally {
      turns[idx].streaming = false
      sending = false
      // The finished turn is now persisted — refresh the sidebar so this chat appears/updates.
      void refreshConversations()
    }
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send()
    }
  }

  // A/B-compare (U6): retrieve the current question under the locked defaults and the session
  // override, and show the source-set diff. $0 (no LLM); the composer text is left intact.
  async function doCompare(): Promise<void> {
    const text = input.trim()
    if (!text || sending || comparing) return
    comparing = true
    try {
      compareResult = await compareRetrieval(text, overrides)
      pinned = true // bring the fresh compare card into view
    } catch (e) {
      console.error('compare failed', e)
    } finally {
      comparing = false
    }
  }

  async function doExport(): Promise<void> {
    try {
      // Export the conversation on screen: the viewed past chat, else the live/resumed session.
      // The backend sources the transcript from the durable records by id, so both work.
      await exportConversation(viewing ?? sessionId, false)
    } catch (e) {
      console.error('export failed', e)
    }
  }

  // Clear the conversation and start a fresh question (U4). Resets the on-screen turns, any open
  // citation panel, the read-only view, and the composer — and mints a new sessionId so the
  // backend doesn't thread the previous conversation's context into the next question. Session
  // overrides (ADR-010) are left as-is: a deliberate sandbox setting, not conversation state.
  function newConversation(): void {
    if (sending) return
    turns = []
    activeCitation = null
    compareResult = null
    viewing = null
    viewedConvo = null
    resumedHistory = null
    input = ''
    resetComposer()
    nextId = 0
    sessionId = freshSessionId()
    pinned = true
    sidebarOpen = false
    taEl?.focus()
  }

  // Continue a viewed past chat (fresh-context resume). Switch the live session to it: its turns
  // become read-only reference above the composer, and new turns thread to the same session_id
  // (so they append + persist). The backend session for this id starts empty — new questions are
  // standalone corpus queries, not a replay of the old conversation (memory is a later increment).
  function resumeConversation(): void {
    if (!viewedConvo || !viewing) return
    resumedHistory = viewedConvo
    sessionId = viewing
    viewing = null
    viewedConvo = null
    turns = []
    nextId = 0
    activeCitation = null
    compareResult = null
    input = ''
    resetComposer()
    pinned = true
    sidebarOpen = false
    taEl?.focus()
  }

  // Open a past conversation read-only (H2). Selecting the live chat returns to it; the live
  // chat's in-memory state is never destroyed by viewing an old one.
  async function openConversation(sid: string): Promise<void> {
    sidebarOpen = false
    activeCitation = null
    if (sid === sessionId) {
      viewing = null
      viewedConvo = null
      return
    }
    try {
      viewedConvo = await getConversation(sid)
      viewing = sid
      pinned = true
    } catch (e) {
      console.error('open conversation failed', e)
    }
  }

  function backToCurrent(): void {
    viewing = null
    viewedConvo = null
    activeCitation = null
  }

  // Library documents are a sidecar read — a failure must never break the app (inform, don't block).
  async function refreshDocuments(): Promise<void> {
    try {
      documents = await listLibraryDocuments()
      documentsLoaded = true
    } catch {
      // keep the prior list
    }
  }

  // Switch between Chat and Library. Entering Library closes any open citation panel and lazy-loads
  // the document list once; the live chat's in-memory state is preserved across the switch.
  function selectMode(m: 'chat' | 'library'): void {
    mode = m
    sidebarOpen = false
    activeCitation = null
    if (m === 'library' && !documentsLoaded) void refreshDocuments()
  }

  function selectDocument(id: string): void {
    libraryDocId = id
    sidebarOpen = false
  }
</script>

<div class="app" style="--sidebar-width: {sidebarWidth}px">
  <Sidebar
    {mode}
    {conversations}
    {documents}
    liveSessionId={sessionId}
    viewingSessionId={viewing}
    selectedDocId={libraryDocId}
    open={sidebarOpen}
    onNew={newConversation}
    onSelect={openConversation}
    onSelectMode={selectMode}
    onSelectDocument={selectDocument}
    onClose={() => (sidebarOpen = false)}
    onPin={pinConversation}
    onArchive={archiveConversation}
    onDelete={deleteConversation}
    onRename={renameConversation}
  />
  <div
    class="resizer"
    role="separator"
    aria-orientation="vertical"
    aria-label="Resize sidebar"
    onpointerdown={startResize}
  ></div>

  <div class="content">
    <main>
      <header>
        <button class="hamburger" onclick={() => (sidebarOpen = true)} aria-label="Open conversations">
          <Icon name="menu" />
        </button>
        <div class="brand">
          <span class="mark"><img src={appMark} alt="" width="32" height="32" /></span>
          <div class="brandtext">
            <span class="wordmark">proven<span class="wm-accent">ote</span></span>
            {#if status === 'ready' && health}
              <span class="meta">
                {health.chunk_count.toLocaleString()} chunks · {health.model} · {health.embedding_model}
              </span>
            {:else if status === 'connecting'}
              <span class="meta">starting the engine…</span>
            {:else}
              <span class="meta err">backend unreachable. Run <code>just api</code></span>
            {/if}
          </div>
        </div>
        <div class="actions">
          <button
            class="ghost"
            onclick={doExport}
            disabled={mode === 'library' ||
              (viewing === null && resumedHistory === null && turns.length === 0)}
            ><Icon name="download" size={15} /> Export</button
          >
          <button class="ghost" onclick={() => (showSettings = true)} aria-label="Settings">
            <Icon name="settings" />
          </button>
        </div>
      </header>

      {#if mode === 'library'}
        <LibraryBrowser docId={libraryDocId} />
      {:else}
      <section class="conversation" bind:this={convoEl} onscroll={onConvoScroll}>
        {#if viewing && viewedConvo}
          <p class="readonly-note">
            Viewing a past conversation (read-only).
            <button class="linkish" onclick={resumeConversation}>Continue this chat</button>
            ·
            <button class="linkish" onclick={backToCurrent}>Back to current chat</button>
          </p>
          {#each viewedConvo.turns as t (t.record_id)}
            <ReadonlyTurn
              question={t.question}
              answer={t.answer}
              onCitationClick={(n) => (activeCitation = { turnKey: t.record_id, n })}
              activeCitationN={activeCitation?.turnKey === t.record_id ? activeCitation.n : null}
            />
          {/each}
        {:else}
          {#if resumedHistory}
            <p class="readonly-note resumed">
              Continuing <strong>{resumedHistory.title}</strong> · earlier turns are shown for
              reference. New questions start fresh — grounded in your corpus, not the old chat.
            </p>
            {#each resumedHistory.turns as t (t.record_id)}
              <ReadonlyTurn
                question={t.question}
                answer={t.answer}
                onCitationClick={(n) => (activeCitation = { turnKey: t.record_id, n })}
                activeCitationN={activeCitation?.turnKey === t.record_id ? activeCitation.n : null}
              />
            {/each}
            <div class="resume-divider"><span>continuing below</span></div>
          {/if}
          {#if status === 'ready' && health && health.chunk_count === 0}
            <div class="banner">
              <span class="state-mark"><Icon name="library" size={26} /></span>
              <strong>No documents indexed yet</strong>
              <p>
                Point doc_assistant at a folder of your documents to get started. It'll index them
                locally, then you can ask questions grounded in them.
              </p>
              <button class="primary" onclick={() => (showSettings = true)}>Choose a folder…</button>
            </div>
          {:else if turns.length === 0 && !resumedHistory}
            <div class="empty">
              <span class="state-mark"><Icon name="book-open-text" size={26} /></span>
              <h2>Ask your library a question</h2>
              <p>
                Every answer is grounded in your own documents, with inline citations, provenance,
                and per-claim review.
              </p>
              <div class="chips">
                {#each sampleQuestions as q}
                  <button class="chip" onclick={() => useSample(q)}>{q}</button>
                {/each}
              </div>
            </div>
          {/if}
          {#each turns as t (t.id)}
            <Turn
              question={t.question}
              answer={t.answer}
              result={t.result}
              streaming={t.streaming}
              error={t.error}
              onCitationClick={(n) => (activeCitation = { turnKey: String(t.id), n })}
              activeCitationN={activeCitation?.turnKey === String(t.id) ? activeCitation.n : null}
            />
          {/each}
          {#if compareResult}
            <CompareCard result={compareResult} onClose={() => (compareResult = null)} />
          {/if}
        {/if}
      </section>

      <footer>
        {#if viewing}
          <div class="viewing-bar">
            <button class="back" onclick={backToCurrent}
              ><Icon name="arrow-left" size={15} /> Back to current chat</button
            >
            <button class="resume" onclick={resumeConversation}
              ><Icon name="rotate-ccw" size={15} /> Continue this chat</button
            >
          </div>
        {:else}
          <textarea
            bind:this={taEl}
            bind:value={input}
            onkeydown={onKey}
            oninput={autogrow}
            placeholder="Ask your documents…  (Enter to send, Shift+Enter for newline)"
            rows="2"
            disabled={sending}
          ></textarea>
          {#if hasRetrievalOverride}
            <button
              class="compare"
              onclick={doCompare}
              disabled={sending || comparing || input.trim() === ''}
              title="See how your override changes retrieval for this question: locked defaults vs override, sources only, no answer ($0)"
              type="button"
            >
              {comparing ? 'Comparing…' : 'Test override'}
            </button>
          {/if}
          <button class="send" onclick={send} disabled={sending || input.trim() === ''} aria-busy={sending}>
            {#if sending}<span class="spinner" aria-hidden="true"></span>{:else}Send{/if}
          </button>
        {/if}
      </footer>
      {/if}
    </main>
  </div>
</div>

{#if showSettings}
  <Settings onClose={() => (showSettings = false)} onCorpusChanged={refreshHealth} bind:overrides />
{/if}

{#if activeCitation && activeSource}
  <SourcePanel source={activeSource} onClose={() => (activeCitation = null)} />
{/if}

<style>
  .app {
    display: flex;
    height: 100vh;
  }
  /* Drag handle between the sidebar and the content — a thin hit area with a hover cue. */
  .resizer {
    flex: none;
    width: 5px;
    margin-left: -3px;
    cursor: col-resize;
    background: transparent;
    z-index: 5;
    transition: background 0.15s ease;
  }
  .resizer:hover,
  .resizer:active {
    background: var(--accent);
    opacity: 0.5;
  }
  @media (max-width: 720px) {
    .resizer {
      display: none;
    }
  }
  .content {
    flex: 1;
    min-width: 0;
    display: flex;
    justify-content: center;
    overflow: hidden;
  }
  main {
    width: 100%;
    max-width: 820px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 0 1rem;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    padding: var(--space-3) 0;
    border-bottom: 1px solid var(--border);
  }
  .hamburger {
    display: none;
    font: inherit;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg);
    border-radius: 8px;
    padding: 0.2rem 0.55rem;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex: 1;
    min-width: 0;
  }
  .mark {
    flex: none;
    width: 32px;
    height: 32px;
    border-radius: 9px;
    overflow: hidden;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--shadow-1);
  }
  .mark img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .brandtext {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .wordmark {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    line-height: 1.15;
    color: var(--fg);
  }
  .wm-accent {
    color: var(--accent-wordmark);
  }
  .meta {
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .meta.err {
    color: var(--warn-fg);
  }
  .conversation {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-2) 0;
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
  .empty p,
  .banner p {
    color: var(--fg-2);
    font-size: var(--text-sm);
    line-height: 1.6;
    max-width: 46ch;
    margin: var(--space-2) 0 var(--space-4);
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
  .actions {
    display: flex;
    gap: 0.4rem;
    align-items: center;
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
  .ghost {
    font-size: 0.82rem;
    padding: 0.3rem 0.7rem;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }
  .compare {
    font-size: 0.82rem;
    white-space: nowrap;
    color: var(--fg-2);
  }
  @media (max-width: 720px) {
    .hamburger {
      display: inline-flex;
    }
  }
</style>
