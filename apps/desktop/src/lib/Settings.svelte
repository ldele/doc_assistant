<script lang="ts">
  import type { Settings, IngestStatus, RagOverrides } from './types'
  import { getSettings, setSourceDir, setLlmProvider, startIngest, getIngestStatus } from './api'
  import { onDestroy } from 'svelte'
  import { fade, fly } from 'svelte/transition'
  import { getTheme, setTheme, applyTheme, type Theme } from './theme'
  import Icon from './Icon.svelte'
  import Sources from './Sources.svelte'

  // Slide the drawer in/out — but collapse to an instant swap when the OS asks for reduced motion.
  const animate =
    typeof window !== 'undefined' && window.matchMedia
      ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : true
  const DUR = animate ? 180 : 0

  // The parent refreshes /api/health after a successful ingest so the chunk count in the
  // header goes live (the backend rebuilds the controller on the new corpus before "done").
  // `overrides` is bindable — the RAG-sandbox section mutates it directly; the session-scoped
  // state itself lives in App.svelte (ADR-010: in-memory only, cleared on restart).
  let {
    onClose,
    onCorpusChanged,
    overrides = $bindable(),
  }: { onClose: () => void; onCorpusChanged: () => void; overrides: RagOverrides } = $props()

  let theme = $state<Theme>(getTheme())

  function onThemeChange(t: Theme): void {
    theme = t
    setTheme(t)
    applyTheme(t)
  }

  function resetSandbox(): void {
    overrides = {}
  }

  let settings = $state<Settings | null>(null)

  // Effective (session-override-or-locked-default) values the sandbox controls display.
  const effTopK = $derived(overrides.top_k ?? settings?.top_k ?? 1)
  const effSynthesisMode = $derived(overrides.synthesis_mode ?? settings?.synthesis_mode ?? 'ai')
  const effMultiQuery = $derived(overrides.use_multi_query ?? settings?.use_multi_query ?? false)
  const effMarkersEnabled = $derived(
    overrides.epistemics_markers_enabled ?? settings?.epistemics_markers_enabled ?? true,
  )
  const effReviewerEvidenceChars = $derived(
    overrides.reviewer_evidence_chars ?? settings?.reviewer_evidence_chars ?? 1500,
  )
  // ADR-011 (U1c) — the provider/model switch. Local inputs are seeded once from the loaded
  // settings (in load(), mirroring `dir`'s own seed-once pattern below) so an in-progress edit
  // survives a background settings refresh.
  let llmProvider = $state('')
  let llmModel = $state('')
  let llmBusy = $state(false)
  let llmError = $state<string | null>(null)

  async function applyProvider(): Promise<void> {
    if (llmBusy || !llmProvider || !llmModel.trim()) return
    llmBusy = true
    llmError = null
    try {
      settings = await setLlmProvider(llmProvider, llmModel.trim())
    } catch (e) {
      llmError = String(e)
    } finally {
      llmBusy = false
    }
  }

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
      if (!llmProvider) {
        llmProvider = s.provider
        llmModel = s.model
      }
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
        if (++misses >= 5) throw new Error('lost contact with the indexer (it may still be running)')
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
    <button class="x" onclick={onClose} disabled={busy} aria-label="Close">
      <Icon name="x" />
    </button>
  </header>

  {#if loadError}
    <p class="err">Couldn't load settings: {loadError}</p>
  {:else if !settings}
    <p class="muted">Loading…</p>
  {:else}
    <section>
      <h3>Display</h3>
      <div class="segmented" role="radiogroup" aria-label="Theme">
        <button
          type="button"
          role="radio"
          aria-checked={theme === 'system'}
          class:active={theme === 'system'}
          onclick={() => onThemeChange('system')}
        >
          System
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={theme === 'light'}
          class:active={theme === 'light'}
          onclick={() => onThemeChange('light')}
        >
          Light
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={theme === 'dark'}
          class:active={theme === 'dark'}
          onclick={() => onThemeChange('dark')}
        >
          Dark
        </button>
      </div>
    </section>

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
        <p class="warn"><Icon name="triangle-alert" size={14} /> The saved folder doesn't exist yet: <code>{settings.source_dir}</code></p>
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
          <p class="muted">Indexing <code>{ingest.source_dir}</code>. This can take a while for
            large folders. You can keep this open.</p>
        {/if}
        {#if ingest?.state === 'done'}
          <p class="ok"><Icon name="check" size={14} /> {ingest.message}</p>
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
      <h3>Manage files <span class="muted">(selective indexing)</span></h3>
      <p class="hint">
        See each file's status, exclude ones you don't want, or index just a selection. Excluded
        files are skipped by <strong>Index folder</strong> above; an explicit selection here still
        indexes them.
      </p>
      <Sources {onCorpusChanged} />
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
      <h3>Provider &amp; model</h3>
      <p class="hint">
        Switch between already-configured providers. Takes effect on your next question, no
        restart. The API key stays in <code>.env</code>.
      </p>

      <label for="llm-provider">Provider</label>
      <select id="llm-provider" bind:value={llmProvider} disabled={llmBusy}>
        {#each settings.providers as p (p.id)}
          <option value={p.id} disabled={!p.available}>
            {p.id} ({p.paid ? 'metered' : 'local'}){p.available ? '' : ' · add its key to .env'}
          </option>
        {/each}
      </select>

      <label for="llm-model">Model</label>
      <input id="llm-model" type="text" bind:value={llmModel} disabled={llmBusy} spellcheck="false" />

      <button
        class="primary"
        onclick={applyProvider}
        disabled={llmBusy || !llmProvider || !llmModel.trim()}
      >
        {llmBusy ? 'Applying…' : 'Apply'}
      </button>

      <p class="hint">
        Active: <code>{settings.provider}/{settings.model}</code>
      </p>
      <div aria-live="polite">
        {#if llmError}
          <p class="err" role="alert">{llmError}</p>
        {/if}
      </div>
    </section>

    <section>
      <h3>RAG sandbox</h3>
      <p class="banner">
        Session only. Resets when you restart. To change a default, run the eval harness.
      </p>

      <label for="topk">Top-K <span class="muted">({effTopK} of {settings.candidate_k})</span></label>
      <input
        id="topk"
        type="range"
        min="1"
        max={settings.candidate_k}
        value={effTopK}
        oninput={(e) => (overrides.top_k = Number((e.target as HTMLInputElement).value))}
      />

      <label for="mode-group">Synthesis mode</label>
      <div id="mode-group" class="segmented" role="radiogroup" aria-label="Synthesis mode">
        <button
          type="button"
          role="radio"
          aria-checked={effSynthesisMode === 'ai'}
          class:active={effSynthesisMode === 'ai'}
          onclick={() => (overrides.synthesis_mode = 'ai')}
        >
          AI
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={effSynthesisMode === 'human'}
          class:active={effSynthesisMode === 'human'}
          onclick={() => (overrides.synthesis_mode = 'human')}
        >
          Human
        </button>
      </div>

      <label class="switch-row">
        <input
          type="checkbox"
          checked={effMultiQuery}
          onchange={(e) => (overrides.use_multi_query = (e.target as HTMLInputElement).checked)}
        />
        Multi-query expansion <span class="muted">(costs one extra LLM call)</span>
      </label>

      <label class="switch-row">
        <input
          type="checkbox"
          checked={effMarkersEnabled}
          onchange={(e) =>
            (overrides.epistemics_markers_enabled = (e.target as HTMLInputElement).checked)}
        />
        Show contested/superseded chips
      </label>

      <label for="reviewer-chars"
        >Reviewer evidence
        <span class="muted">({effReviewerEvidenceChars.toLocaleString()} chars)</span></label
      >
      <!-- Commit on change (blur/Enter/spinner), not per keystroke: a partial value ("15" en
           route to 1500, or a cleared field) must never become the override — the API rejects
           out-of-range with a 422 and every later question would fail on it. Out-of-range
           clamps to the API bounds [200, 6000]; an emptied field drops the override entirely
           (back to the locked default). -->
      <input
        id="reviewer-chars"
        type="number"
        min="200"
        max="6000"
        step="100"
        value={effReviewerEvidenceChars}
        onchange={(e) => {
          const el = e.target as HTMLInputElement
          if (el.value.trim() === '') {
            overrides.reviewer_evidence_chars = null
            el.value = String(effReviewerEvidenceChars)
            return
          }
          const n = Math.round(Number(el.value))
          const clamped = Number.isFinite(n)
            ? Math.min(6000, Math.max(200, n))
            : effReviewerEvidenceChars
          overrides.reviewer_evidence_chars = clamped
          el.value = String(clamped)
        }}
      />

      <button class="ghost" onclick={resetSandbox}>Reset to locked defaults</button>
    </section>

    <section>
      <h3>Engine <span class="muted">(read-only)</span></h3>
      <dl>
        <dt>LLM</dt>
        <dd>{settings.provider} / {settings.model}</dd>
        <dt>Embeddings</dt>
        <dd>{settings.embedding_model}</dd>
        <dt>Candidate pool (pre-rerank)</dt>
        <dd>
          {settings.candidate_k}
          <span class="muted">(fixed at construction; Top-K above cuts it after rerank)</span>
        </dd>
        <dt>Retrieval weights</dt>
        <dd>
          bm25 {settings.retrieval_weights.bm25} / vector {settings.retrieval_weights.vector}
          <span class="muted">(inert on the shipped top-K by construction, measured)</span>
        </dd>
        <dt>Parent-child retrieval</dt>
        <dd>
          {settings.use_parent_child ? 'on' : 'off'}
          <span class="muted">(needs a re-ingest to change)</span>
        </dd>
        <dt>Parent chunk size / overlap</dt>
        <dd>
          {settings.parent_chunk[0]} / {settings.parent_chunk[1]}
          <span class="muted">(needs a re-ingest to change)</span>
        </dd>
        <dt>Child chunk size / overlap</dt>
        <dd>
          {settings.child_chunk[0]} / {settings.child_chunk[1]}
          <span class="muted">(needs a re-ingest to change)</span>
        </dd>
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
    padding: 0.2rem 0.4rem;
    display: inline-flex;
    align-items: center;
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
  input,
  select {
    width: 100%;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.45rem 0.55rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
  }
  input:disabled,
  select:disabled {
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
  .banner {
    color: var(--fg-2);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    font-size: 0.78rem;
    margin: 0 0 0.7rem;
  }
  .segmented {
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .segmented button {
    font: inherit;
    font-size: 0.82rem;
    cursor: pointer;
    border: none;
    background: var(--surface);
    color: var(--fg);
    padding: 0.35rem 0.8rem;
  }
  .segmented button + button {
    border-left: 1px solid var(--border);
  }
  .segmented button.active {
    background: var(--accent);
    color: var(--accent-fg);
  }
  section label {
    margin-top: 0.7rem;
  }
  section label:first-of-type {
    margin-top: 0;
  }
  input[type='range'] {
    width: 100%;
    accent-color: var(--accent);
  }
  .switch-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--fg);
    cursor: pointer;
  }
  .switch-row input {
    width: auto;
    accent-color: var(--accent);
  }
  .ghost {
    margin-top: 0.8rem;
    font: inherit;
    font-size: 0.82rem;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    padding: 0.35rem 0.8rem;
  }
</style>
