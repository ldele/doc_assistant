<script lang="ts">
  // Re-run picker (ADR-048, ROADMAP 20/21) — one dialog, two entry points.
  //
  // The document panel opens it for one document; the grid's Select mode opens it for a selection.
  // Nothing here knows which: it takes a list of ids and a label, so row 21 needed no second
  // component and cannot drift from row 20's wording or its cost statement.
  //
  // **The parts and their costs are SERVED, never hardcoded here.** They differ by four orders of
  // magnitude (metadata is milliseconds, text can be minutes on a scan), which is the entire reason
  // the parts are separate — and a copy of those numbers in the client would drift from
  // `docs/performance.md` silently. `/api/library/reingest/options` is the one source.
  //
  // It stays open through the run rather than handing off to the status bar: the useful output is
  // the per-part report ("no reference list found", "the source file is not reachable"), and that
  // has nowhere else to be read.
  import type { ReingestOptions, ReingestStatus } from '../core/types'
  import { getReingestOptions, getReingestStatus, startReingest } from '../core/api'
  import Icon from '../shell/Icon.svelte'
  import { costSummary, fraction, needsConfirmation, orderedParts, outcomesByDocument } from './reingest'

  let {
    documentIds,
    label,
    onClose,
    onFinished,
  }: {
    documentIds: string[]
    /** What is being re-run, in the user's terms: a filename, or "12 documents". */
    label: string
    onClose: () => void
    /** Fired once, after a run finishes, so the caller can refetch what changed. */
    onFinished: () => void
  } = $props()

  let options = $state<ReingestOptions | null>(null)
  let loadError = $state<string | null>(null)
  let selected = $state(new Set<string>())
  let confirmed = $state(false)
  let status = $state<ReingestStatus | null>(null)
  let startError = $state<string | null>(null)
  let polling = $state(false)

  const chosen = $derived(orderedParts(options, selected))
  const needsConfirm = $derived(needsConfirmation(chosen))
  const summary = $derived(costSummary(chosen, documentIds.length))
  const progress = $derived(fraction(status))
  const grouped = $derived(outcomesByDocument(status?.outcomes ?? []))
  const running = $derived(status?.state === 'running' || polling)
  const finished = $derived(status?.state === 'done' || status?.state === 'error')

  void load()

  async function load(): Promise<void> {
    try {
      options = await getReingestOptions()
    } catch (e) {
      loadError = e instanceof Error ? e.message : String(e)
    }
  }

  function toggle(id: string): void {
    // A new Set, not a mutation: `$state` tracks the reference, and mutating in place would leave
    // every derived value above stale.
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    selected = next
    confirmed = false // re-tick the confirmation whenever the selection changes
  }

  async function start(): Promise<void> {
    if (chosen.length === 0 || running) return
    startError = null
    polling = true
    try {
      await startReingest(documentIds, chosen.map((p) => p.id))
    } catch (e) {
      polling = false
      startError = e instanceof Error ? e.message : String(e)
      return
    }
    void poll()
  }

  async function poll(): Promise<void> {
    // Poll rather than stream: this is the same 202 + poll shape every background job here uses,
    // and the run is short enough that a socket would be ceremony.
    for (;;) {
      try {
        status = await getReingestStatus()
      } catch {
        // A blip must not end the run's reporting — try again on the next tick.
      }
      if (status && status.state !== 'running') break
      await new Promise((r) => setTimeout(r, 400))
    }
    polling = false
    onFinished()
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape' && !running) onClose()
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="scrim" onclick={() => !running && onClose()} role="presentation"></div>

<div class="modal" role="dialog" aria-modal="true" aria-labelledby="reingest-title">
  <h2 id="reingest-title">Re-run ingestion</h2>
  <p class="target" title={label}>{label}</p>

  {#if loadError}
    <p class="err" role="alert">Couldn’t load the options: {loadError}</p>
  {:else if !options}
    <p class="muted">Loading…</p>
  {:else if finished}
    <!-- The report is the point of keeping the dialog open: a skip has a reason, and this is the
         only place it is ever said. -->
    <p class="summary" role="status">{status?.message ?? 'Done.'}</p>
    <div class="report">
      {#each grouped as g (g.documentId)}
        <div class="repdoc">
          <p class="repname" title={g.filename}>{g.filename}</p>
          {#each g.parts as p (p.part)}
            <p class="repline" class:warn={p.status === 'skipped'} class:err={p.status === 'error'}>
              <span class="reppart">{p.part}</span>
              <span>{p.detail}</span>
            </p>
          {/each}
        </div>
      {:else}
        <p class="muted">Nothing ran.</p>
      {/each}
    </div>
    <div class="actions">
      <button class="primary" onclick={onClose} type="button">Close</button>
    </div>
  {:else if running}
    <p class="summary" role="status">
      {#if status?.current}Re-running {status.current}{:else}Starting…{/if}
    </p>
    <!-- `null` draws no bar at all. A bar pinned at zero claims "no progress"; not knowing the
         denominator yet is a different statement, and the ingest bar makes the same distinction. -->
    {#if progress !== null}
      <div class="bar" role="progressbar" aria-valuemin={0} aria-valuemax={status?.total ?? 0} aria-valuenow={status?.done ?? 0}>
        <div class="fill" style="width: {Math.round(progress * 100)}%"></div>
      </div>
      <p class="hint">{status?.done ?? 0} of {status?.total ?? 0}</p>
    {/if}
  {:else}
    <fieldset>
      <legend class="hint">What should be re-derived?</legend>
      {#each options.parts as p (p.id)}
        <label class="part">
          <input type="checkbox" checked={selected.has(p.id)} onchange={() => toggle(p.id)} />
          <span class="pbody">
            <span class="phead">
              {p.label}
              <span class="pcost">{p.cost}</span>
            </span>
            <span class="pblurb">{p.blurb}</span>
          </span>
        </label>
      {/each}
    </fieldset>

    <p class="summary">{summary}</p>

    <!-- Named, not hidden (ADR-048): a user who cannot find a button deserves to know there is no
         button, rather than concluding the feature is broken. -->
    <p class="hint">
      {options.corpus_wide.join(', ')} are computed across your whole library, so they can’t be
      re-run for one document. Re-index from Settings for those.
    </p>

    {#if needsConfirm}
      <label class="confirm">
        <input type="checkbox" bind:checked={confirmed} />
        <span>
          <strong>Text &amp; chunks re-extracts the file from scratch.</strong> It replaces this
          document’s indexed text, and it is the slowest thing the app does.
        </span>
      </label>
    {/if}

    {#if startError}<p class="err" role="alert">{startError}</p>{/if}

    <div class="actions">
      <button class="ghost" onclick={onClose} type="button">Cancel</button>
      <button
        class="primary"
        disabled={chosen.length === 0 || (needsConfirm && !confirmed)}
        onclick={start}
        type="button"
      >
        <Icon name="rotate-ccw" size={14} /> Re-run
      </button>
    </div>
  {/if}
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--fg) 32%, transparent);
    z-index: 40;
  }
  .modal {
    position: fixed;
    z-index: 41;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(92vw, 520px);
    max-height: 86vh;
    overflow-y: auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  h2 {
    margin: 0;
    font-size: var(--text-title);
    font-family: var(--font-serif);
  }
  .target {
    margin: 0;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  fieldset {
    border: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  legend {
    padding: 0;
  }
  .part {
    display: flex;
    gap: 0.55rem;
    align-items: flex-start;
    padding: 0.45rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
  }
  .part:hover {
    background: var(--surface-2);
  }
  .pbody {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    min-width: 0;
  }
  .phead {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    font-weight: 600;
    font-size: 0.85rem;
  }
  .pcost {
    font-weight: 400;
    font-size: 0.72rem;
    color: var(--fg-2);
  }
  .pblurb {
    font-size: 0.76rem;
    color: var(--fg-2);
  }
  .confirm {
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
    font-size: 0.78rem;
    padding: 0.5rem;
    border: 1px solid var(--warn-fg, var(--border));
    border-radius: 8px;
  }
  .summary {
    margin: 0.2rem 0 0;
    font-size: 0.8rem;
  }
  .hint {
    margin: 0;
    font-size: 0.74rem;
    color: var(--fg-2);
  }
  .muted {
    color: var(--fg-2);
    font-size: 0.8rem;
    margin: 0;
  }
  .err {
    color: var(--danger, #c0392b);
    font-size: 0.78rem;
    margin: 0.2rem 0 0;
  }
  .bar {
    height: 6px;
    border-radius: 999px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: var(--accent);
    transition: width 120ms linear;
  }
  .report {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-height: 40vh;
    overflow-y: auto;
  }
  .repdoc {
    border-top: 1px solid var(--border);
    padding-top: 0.4rem;
  }
  .repname {
    margin: 0 0 0.2rem;
    font-size: 0.8rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .repline {
    margin: 0;
    font-size: 0.76rem;
    color: var(--fg-2);
    display: flex;
    gap: 0.4rem;
  }
  .repline.warn {
    color: var(--warn-fg, var(--fg));
  }
  .repline.err {
    color: var(--danger, #c0392b);
  }
  .reppart {
    flex: none;
    min-width: 5.5rem;
    font-variant: small-caps;
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 0.3rem;
  }
  .primary,
  .ghost {
    font: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .primary {
    font-weight: 600;
    border: 1px solid transparent;
    background: var(--accent);
    color: var(--bg);
  }
  .primary:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .ghost {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
  }
</style>
