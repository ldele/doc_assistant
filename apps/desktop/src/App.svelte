<script lang="ts">
  import type { Health, TurnResult } from './lib/types'
  import { getHealth, streamChat, exportConversation } from './lib/api'
  import Turn from './lib/Turn.svelte'

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
  let nextId = 0

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

  async function send(): Promise<void> {
    const text = input.trim()
    if (!text || sending) return
    input = ''
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
    <button class="ghost" onclick={doExport} disabled={turns.length === 0}>⬇ Export</button>
  </header>

  <section class="conversation">
    {#if turns.length === 0}
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
      bind:value={input}
      onkeydown={onKey}
      placeholder="Ask your documents…  (Enter to send, Shift+Enter for newline)"
      rows="2"
      disabled={sending}
    ></textarea>
    <button class="send" onclick={send} disabled={sending || input.trim() === ''}>
      {sending ? '…' : 'Send'}
    </button>
  </footer>
</main>

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
  }
  .ghost {
    font-size: 0.82rem;
    padding: 0.3rem 0.7rem;
  }
</style>
