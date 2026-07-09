<script lang="ts">
  import type { Settings, IngestStatus } from './types'
  import { getSettings, setSourceDir, startIngest, getIngestStatus } from './api'
  import { onDestroy } from 'svelte'
  import { fade, fly } from 'svelte/transition'

  // Slide the drawer in/out — but collapse to an instant swap when the OS asks for reduced motion.
  const animate =
    typeof window !== 'undefined' && window.matchMedia
      ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : true
  const DUR = animate ? 180 : 0

  // The parent refreshes /api/health after a successful ingest so the chunk count in the
  // header goes live (the backend rebuilds the controller on the new corpus before "done").
  let { onClose, onCorpusChanged }: { onClose: () => void; onCorpusChanged: () => void } = $props()

  let settings = $state<Settings | null>(null)
  let loadError = $state<string | null>(null)
  let dir = $state('')
  let busy = $state(false) // a save+ingest cycle is in flight
  let actionError = $state<string | null>(null)
  let ingest = $state<IngestStatus | null>(null)
  let inputEl = $state<HTMLInputElement | null>(null)
  let panelEl = $state<HTMLDivElement | null>(null)

  let cancelled = false
  onDestroy(() => {
    cancelled = true
  })

  void load()

  // Move focus into the dialog when it opens (the input mounts once settings load) so a
  // keyboard / screen-reader user lands in the panel, not on the gear button behind the scrim.
  $effect(() => {
    inputEl?.focus()
  })

  // `silent` post-ingest refreshes don't raise the fatal panel-level loadError: a transient blip
  // on the refresh must not collapse the panel + erase the "✓ indexed" confirmation (the header
  // was already updated via onCorpusChanged). Default-valued param, not optional — Svelte 5's TS
  // strip rejects `?:` here.
  async function load(silent = false): Promise<void> {
    try {
      const s = await getSettings()
      if (cancelled) return
      settings = s
      if (!dir) dir = s.source_dir
    } catch (e) {
      if (!silent) loadError = String(e)
    }
  }

  // One low-friction action: validate+persist the folder, then re-index it. Saving first
  // means a bad path surfaces the backend's 400 before any ingest starts (inform-don't-corrupt).
  async function indexFolder(): Promise<void> {
    const target = dir.trim()
    if (!target || busy) return
    busy = true
    actionError = null
    ingest = null
    try {
      settings = await setSourceDir(target)
      ingest = await startIngest()
      await pollUntilDone()
    } catch (e) {
      actionError = String(e)
    } finally {
      busy = false
    }
  }

  // Poll the background ingest. A large folder indexes for minutes, so a single transient status
  // blip on the local sidecar must not tear the cycle down and falsely report failure (same
  // posture as App.svelte's readiness gate + refreshHealth) — tolerate a few in a row, give up
  // only if contact is genuinely lost.
  async function pollUntilDone(): Promise<void> {
    let misses = 0
    for (;;) {
      if (cancelled) return
      await new Promise((r) => setTimeout(r, 1500))
      if (cancelled) return
      try {
        const st = await getIngestStatus()
        ingest = st
        misses = 0
        if (st.state === 'done' || st.state === 'error') break
      } catch {
        if (++misses >= 5) throw new Error('lost contact with the indexer — it may still be running')
      }
    }
    if (ingest?.state === 'done') {
      onCorpusChanged() // header chunk count is now stale
      await load(true) // refresh this panel's chunk_count; a blip here stays non-fatal
    }
  }

  // Feedback always refers to the path it was produced for — drop a stale ✓ / error once the
  // user edits the folder, so it never sits next to a path it no longer describes.
  function clearFeedback(): void {
    ingest = null
    actionError = null
  }

  function onInputKey(e: KeyboardEvent): void {
    if (e.key === 'Enter') {
      e.preventDefault()
      void indexFolder() // self-guards on empty / busy
    }
  }

  function onWindowKey(e: KeyboardEvent): void {
    if (e.key === 'Escape' && !busy) onClose()
  }

  // Honour aria-modal: keep Tab inside the dialog instead of walking out into the UI the modal
  // has told assistive tech is inert.
  function onPanelKey(e: KeyboardEvent): void {
    if (e.key !== 'Tab' || !panelEl) return
    const focusable = Array.from(
      panelEl.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    ) as HTMLElement[]
    if (focusable.length === 0) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault()
      first.focus()
    }
  }
</script>

<svelte:window onkeydown={onWindowKey} />

<div
  class="scrim"
  onclick={() => !busy && onClose()}
  role="presentation"
  transition:fade={{ duration: DUR }}
></div>

<div
  class="panel"
  role="dialog"
  aria-modal="true"
  aria-label="Settings"
  tabindex="-1"
  bind:this={panelEl}
  onkeydown={onPanelKey}
  transition:fly={{ x: 420, opacity: 1, duration: DUR }}
>
  <header>
    <strong>Settings</strong>
    <button class="x" onclick={onClose} disabled={busy} aria-label="Close">✕</button>
  </header>

  {#if loadError}
    <p class="err">Couldn't load settings: {loadError}</p>
  {:else if !settings}
    <p class="muted">Loading…</p>
  {:else}
    <section>
      <h3>Your documents</h3>
      <label for="src">Source folder</label>
      <input
        id="src"
        type="text"
        bind:value={dir}
        bind:this={inputEl}
        oninput={clearFeedback}
        onkeydown={onInputKey}
        spellcheck="false"
        placeholder="C:\path\to\your\documents"
        disabled={busy}
      />
      <p class="hint">
        Paste the full path to the folder holding your documents, then index it. Supported:
        {settings.supported_formats}.
      </p>
      {#if settings.source_dir && !settings.source_dir_exists}
        <p class="warn">⚠ The saved folder doesn't exist yet: <code>{settings.source_dir}</code></p>
      {/if}

      <button class="primary" onclick={indexFolder} disabled={busy || dir.trim() === ''}>
        {#if busy}
          Indexing…
        {:else if settings.chunk_count > 0}
          Re-index
        {:else}
          Index folder
        {/if}
      </button>

      <!-- aria-live so a screen-reader user hears the index progress / completion / failure
           without re-navigating; the failure cases also assert role="alert". -->
      <div aria-live="polite">
        {#if busy && ingest?.state === 'running'}
          <p class="muted">Indexing <code>{ingest.source_dir}</code> — this can take a while for
            large folders. You can keep this open.</p>
        {/if}
        {#if ingest?.state === 'done'}
          <p class="ok">✓ {ingest.message}</p>
        {/if}
        {#if ingest?.state === 'error'}
          <p class="err" role="alert">Indexing failed: {ingest.message}</p>
        {/if}
        {#if actionError}
          <p class="err" role="alert">{actionError}</p>
        {/if}
      </div>
    </section>

    <section>
      <h3>Corpus</h3>
      <dl>
        <dt>Indexed chunks</dt>
        <dd>{settings.chunk_count.toLocaleString()}</dd>
        <dt>Data home</dt>
        <dd class="path">{settings.data_home}</dd>
      </dl>
    </section>

    <section>
      <h3>Engine <span class="muted">(read-only)</span></h3>
      <dl>
        <dt>LLM</dt>
        <dd>{settings.provider} / {settings.model}</dd>
        <dt>Embeddings</dt>
        <dd>{settings.embedding_model}</dd>
        <dt>Reranker cut</dt>
        <dd>top {settings.top_k} of {settings.candidate_k}</dd>
        <dt>Synthesis</dt>
        <dd>{settings.synthesis_mode}</dd>
      </dl>
      <p class="hint">These are locked defaults (changed only via the eval harness).</p>
    </section>
  {/if}
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    z-index: 10;
  }
  .panel {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(420px, 92vw);
    z-index: 11;
    background: var(--bg);
    border-left: 1px solid var(--border);
    padding: 0 1.2rem 1.2rem;
    overflow-y: auto;
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.18);
  }
  header {
    position: sticky;
    top: 0;
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.4rem;
  }
  .x {
    font: inherit;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg-2);
    font-size: 1rem;
    padding: 0.2rem 0.4rem;
  }
  .x:disabled {
    opacity: 0.4;
    cursor: default;
  }
  section {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  section:last-child {
    border-bottom: none;
  }
  h3 {
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
  }
  label {
    display: block;
    font-size: 0.8rem;
    color: var(--fg-2);
    margin-bottom: 0.3rem;
  }
  input {
    width: 100%;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.45rem 0.55rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
  }
  input:disabled {
    opacity: 0.6;
  }
  .hint {
    font-size: 0.76rem;
    color: var(--fg-2);
    margin: 0.4rem 0 0;
  }
  .primary {
    margin-top: 0.7rem;
    font: inherit;
    font-weight: 600;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: var(--accent-fg);
    padding: 0.4rem 1rem;
  }
  .primary:disabled {
    opacity: 0.5;
    cursor: default;
  }
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.25rem 0.8rem;
    margin: 0;
    font-size: 0.82rem;
  }
  dt {
    color: var(--fg-2);
  }
  dd {
    margin: 0;
  }
  dd.path,
  .warn code {
    font-family: ui-monospace, monospace;
    font-size: 0.76rem;
    word-break: break-all;
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
  }
  .ok {
    color: var(--ok-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
  .warn {
    color: var(--warn-fg);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 8px;
    padding: 0.4rem 0.55rem;
    font-size: 0.8rem;
    margin: 0.5rem 0 0;
  }
</style>
