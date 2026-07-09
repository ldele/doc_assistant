<script lang="ts">
  import type { Health, TurnResult } from './lib/types'
  import { getHealth, streamChat, exportConversation } from './lib/api'
  import Turn from './lib/Turn.svelte'
  import Settings from './lib/Settings.svelte'

  interface TurnState {
    id: number
    question: string
    answer: string
    result: TurnResult | null
    streaming: boolean
    error: string | null
  }

  const sessionId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

  let health = $state<Health | null>(null)
  let status = $state<'connecting' | 'ready' | 'down'>('connecting')
  let turns = $state<TurnState[]>([])
  let input = $state('')
  let sending = $state(false)
  let showSettings = $state(false)
  let nextId = 0

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

  // Readiness gate (PR-M4): the frozen sidecar takes a few seconds to load models before
  // it accepts requests. Poll /api/health until it answers (or give up after ~60s).
  $effect(() => {
    let cancelled = false
    void (async () => {
      for (let i = 0; i < 60 && !cancelled; i++) {
        try {
          const h = await getHealth()
          if (!cancelled) {
            health = h
            status = 'ready'
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

  // Keep the newest content in view as tokens stream in / a turn is added — but only when the
  // reader is pinned to the bottom (see `pinned`). Reading the last turn's answer + the turn
  // count is what makes this re-run on each streamed token.
  $effect(() => {
    const last = turns[turns.length - 1]
    void last?.answer
    void turns.length
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
      for await (const ev of streamChat(text, sessionId)) {
        if (ev.event === 'token') turns[idx].answer += ev.data
        else if (ev.event === 'result') turns[idx].result = JSON.parse(ev.data) as TurnResult
        // `step` events are advisory; ignored for now.
      }
    } catch (e) {
      turns[idx].error = String(e)
    } finally {
      turns[idx].streaming = false
      sending = false
    }
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send()
    }
  }

  async function doExport(): Promise<void> {
    try {
      await exportConversation(sessionId, false)
    } catch (e) {
      console.error('export failed', e)
    }
  }
</script>

<main>
  <header>
    <div class="brand">
      <strong>doc_assistant</strong>
      {#if status === 'ready' && health}
        <span class="meta">
          {health.chunk_count.toLocaleString()} chunks · {health.model} · {health.embedding_model}
        </span>
      {:else if status === 'connecting'}
        <span class="meta">starting the engine…</span>
      {:else}
        <span class="meta err">backend unreachable — run <code>just api</code></span>
      {/if}
    </div>
    <div class="actions">
      <button class="ghost" onclick={doExport} disabled={turns.length === 0}>⬇ Export</button>
      <button class="ghost" onclick={() => (showSettings = true)} aria-label="Settings">⚙</button>
    </div>
  </header>

  <section class="conversation" bind:this={convoEl} onscroll={onConvoScroll}>
    {#if status === 'ready' && health && health.chunk_count === 0}
      <div class="banner">
        <strong>No documents indexed yet.</strong>
        <p>Point doc_assistant at a folder of your documents to get started — it'll index them
          locally, then you can ask questions grounded in them.</p>
        <button class="primary" onclick={() => (showSettings = true)}>Choose a folder…</button>
      </div>
    {:else if turns.length === 0}
      <p class="empty">Ask a question grounded in your documents. Answers carry inline citations,
        provenance, and per-claim review.</p>
    {/if}
    {#each turns as t (t.id)}
      <Turn
        question={t.question}
        answer={t.answer}
        result={t.result}
        streaming={t.streaming}
        error={t.error}
      />
    {/each}
  </section>

  <footer>
    <textarea
      bind:this={taEl}
      bind:value={input}
      onkeydown={onKey}
      oninput={autogrow}
      placeholder="Ask your documents…  (Enter to send, Shift+Enter for newline)"
      rows="2"
      disabled={sending}
    ></textarea>
    <button
      class="send"
      onclick={send}
      disabled={sending || input.trim() === ''}
      aria-busy={sending}
    >
      {#if sending}<span class="spinner" aria-hidden="true"></span>{:else}Send{/if}
    </button>
  </footer>
</main>

{#if showSettings}
  <Settings onClose={() => (showSettings = false)} onCorpusChanged={refreshHealth} />
{/if}

<style>
  main {
    max-width: 820px;
    margin: 0 auto;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 0 1rem;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  .brand {
    display: flex;
    flex-direction: column;
  }
  .meta {
    font-size: 0.76rem;
    color: var(--fg-2);
  }
  .meta.err {
    color: var(--warn-fg);
  }
  .conversation {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem 0;
  }
  .empty {
    color: var(--fg-2);
    margin-top: 2rem;
    text-align: center;
  }
  .actions {
    display: flex;
    gap: 0.4rem;
    align-items: center;
  }
  .banner {
    margin: 2rem auto 0;
    max-width: 520px;
    text-align: center;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    padding: 1.4rem 1.6rem;
  }
  .banner p {
    color: var(--fg-2);
    font-size: 0.9rem;
    margin: 0.5rem 0 1rem;
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
    gap: 0.5rem;
    padding: 0.8rem 0;
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
  }
</style>
