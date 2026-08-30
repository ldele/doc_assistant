<script lang="ts">
  import type { Settings, IngestStatus, RagOverrides, UpdateStatus } from '../core/types'
  import {
    getSettings,
    setSourceDir,
    setLlmProvider,
    setMarkersEnabled,
    startIngest,
    getIngestStatus,
    reindexKeywords,
    exportAllConversations,
    getUpdateStatus,
    checkForUpdate,
    setAutoUpdateCheck,
  } from '../core/api'
  import { describeUpdate } from './updates'
  import { onDestroy } from 'svelte'
  import { fade, fly } from 'svelte/transition'
  import { getTheme, setTheme, applyTheme, type Theme } from '../core/theme'
  import Icon from '../shell/Icon.svelte'
  import { canPickFiles, pickPaths } from '../core/tauri'
  import ProviderSetup from './ProviderSetup.svelte'
  import Sources from './Sources.svelte'
  import { describeIndex, formatBytes, perDocument } from './corpus'
  import { SETTINGS_SECTIONS, initialSection, sectionBadge, type SettingsSectionId } from './sections'
  import { outstandingSteps } from './setup'
  import { shell } from '../shell/shell.svelte'

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

  // Chat-history export. A failure is shown, not swallowed: the whole point of the button is to
  // be trusted before a delete, so a silent no-op would be the worst outcome it could have.
  let exporting = $state(false)
  let exportError = $state<string | null>(null)
  async function exportHistory(): Promise<void> {
    exporting = true
    exportError = null
    try {
      await exportAllConversations()
    } catch (e) {
      exportError = `Export failed: ${e instanceof Error ? e.message : String(e)}`
    } finally {
      exporting = false
    }
  }

  // Which category the drawer shows (2026-08-29). The panel is a rail + one pane rather than one
  // flat scroll: see `./sections.ts` for why, and for the list itself.
  //
  // Seeded from `shell.setup`, which App has *already* loaded for the chat pane's setup banner —
  // so a fresh install opens on the checklist instead of opening on Documents and jumping there a
  // moment later when a fetch of our own lands. Reading it also costs no second request.
  const setupStepsLeft = $derived(outstandingSteps(shell.setup).length)
  let activeSection = $state<SettingsSectionId>(initialSection(outstandingSteps(shell.setup).length))
  const activeBlurb = $derived(
    SETTINGS_SECTIONS.find((s) => s.id === activeSection)?.blurb ?? '',
  )

  // Switching category: bring the new rail item into view, and start its pane at the top.
  //
  // Both only bite once the rail outgrows its box — vertically when the category list is longer
  // than the panel, horizontally in the narrow layout where the rail is a scrolling strip. That is
  // the "more settings later" case this restructure exists for, so it is wired now rather than
  // discovered later. Without the second half, a pane scrolled deep in Documents opens General
  // halfway down a page the user has never seen.
  let railEl = $state<HTMLElement | null>(null)
  let paneEl = $state<HTMLElement | null>(null)
  $effect(() => {
    const current = activeSection // the dependency: re-run on every category change
    if (!current) return
    railEl?.querySelector('[aria-current="page"]')?.scrollIntoView({
      block: 'nearest',
      inline: 'nearest',
    })
    if (paneEl) paneEl.scrollTop = 0
  })

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

  // ADR-027 D2 (E3) — the persisted answer-layer epistemics default. Distinct from the RAG
  // sandbox's session-only override below: this one survives a restart and becomes the
  // sandbox's baseline. On failure the checkbox snaps back to the server state (no lying UI).
  let markersBusy = $state(false)
  let markersError = $state<string | null>(null)

  async function applyMarkersDefault(enabled: boolean): Promise<void> {
    if (markersBusy) return
    markersBusy = true
    markersError = null
    try {
      settings = await setMarkersEnabled(enabled)
    } catch (e) {
      markersError = String(e)
    } finally {
      markersBusy = false
    }
  }

  // ADR-044 — the update check. Notification only; nothing here can install anything.
  // `update` is null until the first read; the section renders a neutral line rather than
  // guessing a state, because guessing "up to date" is exactly the failure this feature must
  // not have. A failed *check* is not an error to show in red — it comes back as state
  // 'unknown' with a reason, and only a transport failure lands in `updateError`.
  let update = $state<UpdateStatus | null>(null)
  let updateBusy = $state(false)
  let updateError = $state<string | null>(null)

  async function runUpdateCheck(): Promise<void> {
    if (updateBusy) return
    updateBusy = true
    updateError = null
    try {
      update = await checkForUpdate()
    } catch (e) {
      updateError = `Update check failed: ${e instanceof Error ? e.message : String(e)}`
    } finally {
      updateBusy = false
    }
  }

  async function applyAutoCheck(enabled: boolean): Promise<void> {
    if (updateBusy) return
    updateBusy = true
    updateError = null
    try {
      update = await setAutoUpdateCheck(enabled)
      // Turning it on should answer the question it implies, not wait a day to do it.
      if (enabled) update = await checkForUpdate()
    } catch (e) {
      updateError = `Couldn't save that: ${e instanceof Error ? e.message : String(e)}`
    } finally {
      updateBusy = false
    }
  }

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

  // ADR-037 — the Corpus panel. `indexInfo` derives the label + the honest sentence from the
  // payload's `keyword_index.mode`; the states say materially different things, and since ADR-038
  // one of them (`unavailable`) means retrieval is running on one arm.
  let reindexing = $state(false)
  let reindexError = $state<string | null>(null)
  const indexInfo = $derived(
    settings
      ? describeIndex(settings.corpus)
      : { label: '', memory: '', rebuildable: false, degraded: false }
  )

  // Non-destructive and bounded (the index is derived data the next launch would rebuild anyway),
  // so no confirmation step — inform, don't block. The response carries refreshed settings, so the
  // panel updates without a second request.
  async function rebuildIndex(): Promise<void> {
    if (reindexing) return
    reindexing = true
    reindexError = null
    try {
      const fresh = await reindexKeywords()
      if (!cancelled) settings = fresh
    } catch (e) {
      if (!cancelled) reindexError = String(e)
    } finally {
      if (!cancelled) reindexing = false
    }
  }

  let loadError = $state<string | null>(null)
  let dir = $state('')
  let busy = $state(false) // a save+ingest cycle is in flight
  let picking = $state(false) // the OS folder picker is open, so Browse can't be double-fired
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
    // ADR-044: free in the default configuration — this reads the cached answer and only
    // touches the network if the user opted in and a check is due. Deliberately after the
    // settings load and independently caught: the panel must open even if it fails.
    try {
      const u = await getUpdateStatus()
      if (!cancelled) update = u
    } catch {
      // Leave `update` null — the section says "version unknown until the first check"
      // rather than inventing a state. No red error for a background read nobody asked for.
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
  /**
   * CS1 — choose the library folder with the OS picker instead of pasting a path.
   *
   * Fills the field rather than saving directly: the text input stays the override, and the user
   * still confirms with the same button they would have used after typing. `pickPaths` resolves to
   * `null` for a cancel *and* for "no picker at all", which is why the button is only rendered when
   * `canPickFiles()` — the two cases are indistinguishable here by design.
   */
  async function browseForFolder(): Promise<void> {
    if (picking) return
    picking = true
    try {
      const chosen = await pickPaths({
        directory: true,
        multiple: false,
        title: 'Choose the folder Provenote keeps your documents in',
      })
      if (chosen && chosen.length > 0) {
        dir = chosen[0]
        clearFeedback()
      }
    } finally {
      picking = false
    }
  }

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
  transition:fly={{ x: 760, opacity: 1, duration: DUR }}
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
    <div class="body">
      <!-- Navigation, deliberately NOT `role="tablist"`. Tabs come with a keyboard contract —
           roving tabindex, arrow-key movement — and a rail that claims the role without honouring
           it is worse for a screen-reader user than one that never claimed it. As plain nav
           buttons every category is reachable with Tab, which is what actually happens here. The
           rail is also the panel's heading: the current item names the page, so no category
           repeats its own title inside the pane. -->
      <nav class="rail" aria-label="Settings categories" bind:this={railEl}>
        {#each SETTINGS_SECTIONS as s (s.id)}
          {@const badge = sectionBadge(s.id, setupStepsLeft)}
          <button
            type="button"
            aria-current={activeSection === s.id ? 'page' : undefined}
            class:active={activeSection === s.id}
            onclick={() => (activeSection = s.id)}
          >
            <Icon name={s.icon} size={15} />
            <span class="rl">{s.label}</span>
            {#if badge}
              <span class="badge" aria-label="{badge} setup steps left">{badge}</span>
            {/if}
          </button>
        {/each}
      </nav>

      <!-- aria-live on the blurb, not the pane: announcing a whole category on every rail click
           would talk over the user, while the lead line is exactly "which page am I on now". -->
      <div class="pane" bind:this={paneEl}>
        <p class="blurb" aria-live="polite">{activeBlurb}</p>

        {#if activeSection === 'setup'}
          <!-- First (ADR-034): on a fresh install this is the only section that matters, and it says
               so — a provider and a folder. `load(true)` keeps a provider switch made in there from
               leaving the "Active:" line below stale. -->
          <ProviderSetup onProviderChanged={() => void load(true)} />
        {:else if activeSection === 'documents'}
          <section>
            <h3>Your documents</h3>
            <!-- CS2 — this folder is *where Provenote keeps the documents you add*, not "the folder
                 holding your documents". One sentence used to describe both models, which stopped being
                 true at AD3b: documents are added to the library (copied here, or referenced where they
                 live), and this folder is the copy destination. Pointing at a folder and indexing it
                 wholesale still works and is still offered below — it is just no longer what the folder
                 *is*. -->
            <label for="src">Library folder</label>
            <!-- CS1 — the picker is the primary route; the text field stays as an override, because a
                 path typed or pasted from elsewhere is a real workflow and a picker cannot express one
                 that does not exist yet (which the warning below is about). In a browser there is no
                 picker, so the button is not rendered at all rather than rendered dead. -->
            <div class="srcrow">
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
              {#if canPickFiles()}
                <button class="ghost browse" onclick={browseForFolder} disabled={busy || picking} type="button">
                  <Icon name="folder" size={14} />
                  {picking ? 'Choosing…' : 'Browse…'}
                </button>
              {/if}
            </div>
            <p class="hint">
              Where Provenote keeps the documents you add — anything already in it can be indexed here
              too. Supported: {settings.supported_formats}.
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
              <dt>Library</dt>
              <dd>
                {settings.corpus.documents.toLocaleString()} documents ·
                {settings.corpus.chunks.toLocaleString()} chunks
              </dd>
              <dt>Disk</dt>
              <dd>
                {formatBytes(settings.corpus.disk.total_bytes)}
                {#if perDocument(settings.corpus.disk.total_bytes, settings.corpus.documents)}
                  <span class="muted"
                    >· {perDocument(settings.corpus.disk.total_bytes, settings.corpus.documents)}</span
                  >
                {/if}
              </dd>
              <dt>Keyword index</dt>
              <dd class="index-row">
                <span class:degraded={indexInfo.degraded}>{indexInfo.label}</span>
                {#if indexInfo.rebuildable}
                  <button class="ghost" onclick={rebuildIndex} disabled={reindexing || busy}>
                    {reindexing ? 'Rebuilding…' : 'Rebuild'}
                  </button>
                {/if}
              </dd>
              <dt>Data home</dt>
              <dd class="path">{settings.data_home}</dd>
            </dl>
            <p class="banner" class:warn={indexInfo.degraded}>{indexInfo.memory}</p>
            {#if reindexError}
              <p class="banner err">{reindexError}</p>
            {/if}
          </section>
        {:else if activeSection === 'models'}
          <section>
            <h3>Provider &amp; model</h3>
            <p class="hint">
              The advanced form of the switch in <strong>Getting started</strong> above: any provider,
              any model name. Takes effect on your next question, no restart.
            </p>

            <label for="llm-provider">Provider</label>
            <select id="llm-provider" bind:value={llmProvider} disabled={llmBusy}>
              {#each settings.providers as p (p.id)}
                <option value={p.id} disabled={!p.available}>
                  {p.id} ({p.paid ? 'metered' : 'local'}){p.available ? '' : ' · needs a key'}
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
            <h3>Answer epistemics <span class="muted">(experimental)</span></h3>
            <p class="hint">
              Whether corpus epistemics (contested / superseded chips) may appear on an answer's
              sources. Saved as your default; the per-source evaluation strip below answers is
              always shown either way.
            </p>
            <!-- Off by default, and said plainly rather than left as a silent default
                 (REVIEW 2026-08-12 §2b R3 · KI-33 · ADR-041). A user turning this on deserves to know
                 what they are turning on; a user leaving it off deserves to know they are not missing
                 a measurement. -->
            <p class="hint">
              <strong>Known limitation:</strong> these chips come from a stance pass that judges a topic
              <em>without reading the document</em>, has no “neutral” verdict, and can change its answer
              when the same pair appears in a different order. They are a prompt to go and look, not a
              finding. Off by default until that is rebuilt on evidence.
            </p>
            <label class="switch-row">
              <input
                type="checkbox"
                checked={settings.epistemics_markers_enabled}
                disabled={markersBusy}
                onchange={(e) => void applyMarkersDefault((e.target as HTMLInputElement).checked)}
              />
              Epistemics chips in answers <span class="muted">(saved default)</span>
            </label>
            <div aria-live="polite">
              {#if markersError}
                <p class="err" role="alert">{markersError}</p>
              {/if}
            </div>
          </section>
        {:else if activeSection === 'answers'}
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

          <!-- Chat history (user request 2026-08-10). Export lives here rather than next to the
               sidebar's delete because it is a whole-history action, and because the order matters:
               the file is the copy you can act on, the soft delete is not. -->
          <section>
            <h3>Chat history</h3>
            <p class="hint">
              Every conversation in one markdown file — uncapped, unlike the sidebar list. Delete chats
              from the sidebar’s select mode (✓).
            </p>
            <button class="ghost" onclick={exportHistory} disabled={exporting} type="button">
              {exporting ? 'Exporting…' : 'Export all conversations'}
            </button>
            {#if exportError}<p class="err">{exportError}</p>{/if}
          </section>

          <!-- Updates (ADR-044). Notification only: this app never downloads or installs anything.
               The toggle governs the *automatic* daily check; "Check now" always runs, because an
               explicit press is its own consent and gating it would leave a user who declined
               background traffic with no way to find out whether they're current. -->
          <section>
            <h3>Updates</h3>
            <p class="hint">
              Provenote can check whether a newer version has been published, and link you to it.
              It never downloads or installs anything — you stay in control of what runs on your
              machine.
            </p>
            <label class="switch-row">
              <input
                type="checkbox"
                checked={update?.auto_check_enabled ?? false}
                disabled={updateBusy}
                onchange={(e) => void applyAutoCheck((e.target as HTMLInputElement).checked)}
              />
              Check for updates automatically <span class="muted">(once a day)</span>
            </label>
            <p class="hint">
              When on, Provenote asks GitHub for the latest release version once a day. Nothing about
              you or your documents is sent — no queries, no titles, no identifier.
            </p>

            <div class="update-status" aria-live="polite">
              {#if update}
                {@const line = describeUpdate(update)}
                <p class:ok={line.tone === 'ok'} class:muted={line.tone === 'muted'}>
                  <strong>{line.headline}</strong>
                </p>
                <p class="hint">{line.detail}</p>
                {#if line.showLink}
                  <a class="primary link-button" href={update.release_url} target="_blank" rel="noreferrer noopener">
                    Open the release page
                  </a>
                {/if}
              {:else}
                <p class="muted">Provenote {'—'} version unknown until the first check.</p>
              {/if}
              {#if updateError}
                <p class="err" role="alert">{updateError}</p>
              {/if}
            </div>

            <button class="ghost" onclick={runUpdateCheck} disabled={updateBusy} type="button">
              {updateBusy ? 'Checking…' : 'Check now'}
            </button>
          </section>
        {/if}
      </div>
    </div>
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
    /* Wider than the 420px it was: the rail costs a column, and the controls it navigates to were
       already cramped in one. Still a drawer, not a page — `96vw` keeps the scrim reachable, which
       is how most people close this. */
    width: min(760px, 96vw);
    z-index: 11;
    background: var(--bg);
    border-left: 1px solid var(--border);
    padding: 0 1.2rem 1.2rem;
    /* The PANE scrolls, not the panel: the rail has to stay put or it stops being navigation. */
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.18);
  }
  header {
    flex: none;
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.4rem;
  }
  .body {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 13rem 1fr;
    gap: 1.2rem;
  }
  .rail {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: 0.5rem 0.6rem 0.5rem 0;
    border-right: 1px solid var(--border);
    overflow-y: auto;
  }
  .rail button {
    font: inherit;
    font-size: 0.85rem;
    text-align: left;
    cursor: pointer;
    border: 1px solid transparent;
    background: none;
    color: var(--fg-2);
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .rail button:hover {
    color: var(--fg);
    background: var(--surface);
  }
  .rail button.active {
    color: var(--fg);
    background: var(--surface);
    border-color: var(--border);
    font-weight: 600;
  }
  .rail .rl {
    flex: 1;
    min-width: 0;
  }
  .badge {
    flex: none;
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1;
    padding: 0.2rem 0.4rem;
    border-radius: 999px;
    background: var(--accent);
    color: var(--bg);
  }
  .pane {
    min-width: 0;
    overflow-y: auto;
    padding-bottom: 1rem;
  }
  .blurb {
    margin: 0.5rem 0 0.2rem;
    font-size: 0.8rem;
    color: var(--fg-2);
    max-width: 46rem;
  }
  /* Narrow: the rail becomes a scrollable strip above the pane. Same buttons, same order — it
     stops being a column, not a navigation. */
  @media (max-width: 700px) {
    .body {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr;
      gap: 0.4rem;
    }
    .rail {
      flex-direction: row;
      overflow-x: auto;
      padding: 0.3rem 0;
      border-right: none;
      border-bottom: 1px solid var(--border);
    }
    .rail button {
      flex: none;
    }
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
  .srcrow {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .srcrow input {
    flex: 1;
    min-width: 0;
  }
  .browse {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    white-space: nowrap;
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
  /* Two classes on purpose: `.banner` is declared after `.warn`, so at equal specificity it would
     win and quietly neutralise the warning colours. ADR-038's degraded state must look degraded. */
  .banner.warn {
    color: var(--warn-fg);
    background: var(--warn-bg);
    border-color: var(--warn-border);
  }
  .degraded {
    color: var(--warn-fg);
    font-weight: 600;
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
  /* ADR-044 — the update verdict. Boxed so the state reads as a standing fact about this
     install rather than as transient feedback from the button below it. */
  .update-status {
    margin-top: 0.7rem;
    padding: 0.6rem 0.7rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-2, transparent);
  }
  .update-status p {
    margin: 0;
  }
  .update-status p + p {
    margin-top: 0.25rem;
  }
  /* An <a> styled as the primary button: this is a link on purpose — it opens the release page
     in the browser, and must never look like an in-app install action (ADR-044). */
  .link-button {
    display: inline-block;
    text-decoration: none;
  }
  /* ADR-037 — the Corpus panel's index row: label left, the bounded rebuild action right. */
  .index-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .index-row .ghost {
    margin-top: 0;
    padding: 0.2rem 0.55rem;
    font-size: 0.76rem;
    flex: none;
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
