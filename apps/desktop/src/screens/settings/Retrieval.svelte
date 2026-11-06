<script lang="ts">
  // Settings > Retrieval: the *RAG sandbox* (session-only overrides of how an answer is assembled:
  // Top-K, synthesis mode, multi-query, the experimental chips, reviewer evidence) and the *Engine*,
  // read-only: the locked defaults the sandbox starts from. The overrides are App's (ADR-010: in
  // memory only, cleared on restart); this section writes them through `bind:overrides`.
  import type { RagOverrides, Settings } from '../../lib/core/types'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    settings,
    overrides = $bindable(),
  }: {
    settings: Settings
    overrides: RagOverrides
  } = $props()

  // Effective (session-override-or-locked-default) values the sandbox controls display.
  const effTopK = $derived(overrides.top_k ?? settings?.top_k ?? 1)
  const effSynthesisMode = $derived(overrides.synthesis_mode ?? settings?.synthesis_mode ?? 'ai')
  const effMultiQuery = $derived(overrides.use_multi_query ?? settings?.use_multi_query ?? false)
  const effMarkersEnabled = $derived(
    overrides.epistemics_markers_enabled ?? settings?.epistemics_markers_enabled ?? false,
  )
  const effReviewerEvidenceChars = $derived(
    overrides.reviewer_evidence_chars ?? settings?.reviewer_evidence_chars ?? 1500,
  )

  function resetSandbox(): void {
    overrides = {}
  }
</script>

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
      use:tooltip={'Write an answer from the passages found, with citations'}
    >
      AI
    </button>
    <button
      type="button"
      role="radio"
      aria-checked={effSynthesisMode === 'human'}
      class:active={effSynthesisMode === 'human'}
      onclick={() => (overrides.synthesis_mode = 'human')}
      use:tooltip={'Show the passages found, with no written answer'}
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
    Show experimental contested/superseded chips — not a corpus measurement
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

  <button class="btn below" onclick={resetSandbox} use:tooltip={'Drop your sandbox changes and use the locked settings again'}>Reset to locked defaults</button>
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

<style>
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
  .hint {
    font-size: 0.76rem;
    color: var(--fg-2);
    margin: 0.4rem 0 0;
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
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
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
  /* Layout only: the roles' look is app.css's (ADR-063). */
  .below {
    margin-top: 0.8rem;
  }
</style>
