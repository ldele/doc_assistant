<script lang="ts">
  // AD2 + AD3 — the review sheet: what would happen to each file, then make it happen.
  //
  // ADR-046 chose both placement modes for v1 and AD3b built the second, so the radio is a real
  // control: `copy` puts the files in the Provenote folder, `reference` registers them where they
  // already live. The wording says which one touches the user's own folders, because that is the
  // consequence they cannot undo by guessing.
  //
  // Every verdict comes from the server (`POST /api/documents/inspect`) and arrives already
  // sorted, exceptions first. Nothing is re-sorted or re-classified here: the format list lives
  // in `extractors.get_format_status`, and a second copy in TypeScript would drift from it.

  import Icon from '../shell/Icon.svelte'
  import { addDocuments, indexPaths, inspectDocuments, undoAddDocuments } from '../core/api/documents'
  import type { AddMode, AddResult } from '../core/api/documents'
  import type { FileVerdict, InspectResponse } from '../core/types/documents'
  import { sourceKeyName } from './accept'

  interface Props {
    /** Absolute paths staged by the accept surface (AD1). */
    paths: string[]
    onClose: () => void
    /** Called after a successful add so the caller can clear the staged paths + refresh. */
    onAdded?: () => void
  }
  const { paths, onClose, onAdded }: Props = $props()

  let result = $state<InspectResponse | null>(null)
  let error = $state<string | null>(null)
  let loading = $state(true)
  /** Rows rendered at once. Bounds the DOM, never the batch — the batch is uncapped by decision. */
  let shown = $state(50)

  // AD3 state.
  // ADR-046 placement. Defaults to `copy`: it is the mode that cannot surprise anyone, since
  // everything it touches is Provenote's own folder.
  let mode = $state<AddMode>('copy')
  let indexNow = $state(true) // ticked by default (grill branch 8): one gesture, drop to searchable
  let applying = $state(false)
  let applied = $state<AddResult | null>(null)
  let applyError = $state<string | null>(null)
  let undoing = $state(false)

  $effect(() => {
    const wanted = paths
    loading = true
    error = null
    inspectDocuments(wanted)
      .then((r) => {
        result = r
        shown = 50
      })
      .catch((e: unknown) => (error = e instanceof Error ? e.message : String(e)))
      .finally(() => (loading = false))
  })

  const files = $derived(result?.files ?? [])
  const counts = $derived(result?.counts ?? {})
  const visible = $derived(files.slice(0, shown))
  const hidden = $derived(Math.max(0, files.length - shown))

  /** The header line. Never invents a number: every part comes from the server's counts. */
  const headline = $derived.by(() => {
    const total = counts.total ?? 0
    if (total === 0) return 'Nothing to add'
    const addable = counts.add ?? 0
    const noun = total === 1 ? 'file' : 'files'
    return `${total} ${noun} · ${addable} would be added`
  })

  // Literal union rather than Icon's own `IconName`: that type is declared in the component's
  // instance script and is not importable. Narrower anyway — these are the only three used.
  function icon(v: FileVerdict['verdict']): 'check' | 'rotate-ccw' | 'triangle-alert' {
    if (v === 'add') return 'check'
    if (v === 'duplicate') return 'rotate-ccw'
    return 'triangle-alert'
  }

  /**
   * The one-line reason. `advisory` is the server's own sentence and is never rewritten — but a
   * duplicate's advisory ("Already in your library.") always arrives set, which used to make the
   * `duplicate_of` branch below unreachable and left the user told *that* a file is a duplicate
   * and never *which* one. The duplicate case is answered first for that reason.
   */
  function reason(f: FileVerdict): string {
    if (f.verdict === 'duplicate') {
      const of = f.duplicate_of ? ` — matches ${sourceKeyName(f.duplicate_of)}` : ''
      return `${f.advisory ?? 'Already in your library.'}${of}`
    }
    if (f.advisory) return f.advisory
    return ''
  }

  const addable = $derived(files.filter((f) => f.selected_by_default))

  /** Rough, and labelled as such. A wrong-looking precise number is worse than an honest range. */
  const estimate = $derived.by(() => {
    const n = addable.length
    if (n === 0) return ''
    const mins = Math.max(1, Math.round((n * 25) / 60))
    return `about ${mins} min to index`
  })

  async function apply(): Promise<void> {
    applying = true
    applyError = null
    try {
      const result = await addDocuments(addable.map((f) => f.path), mode)
      applied = result
      // Index only what actually landed — never what the run did not reach.
      if (indexNow && !result.stopped_early && result.added.length > 0) {
        await indexPaths(result.added.map((o) => o.key).filter((k): k is string => !!k))
      }
      if (!result.stopped_early) onAdded?.()
    } catch (e: unknown) {
      applyError = e instanceof Error ? e.message : String(e)
    } finally {
      applying = false
    }
  }

  async function undoAll(): Promise<void> {
    if (!applied) return
    undoing = true
    try {
      await undoAddDocuments(applied.added.map((o) => o.key).filter((k): k is string => !!k))
      applied = null
      onAdded?.()
      onClose()
    } catch (e: unknown) {
      applyError = e instanceof Error ? e.message : String(e)
    } finally {
      undoing = false
    }
  }

  function keepAndClose(): void {
    onAdded?.()
    onClose()
  }

  function kb(size: number | null): string {
    if (size === null) return ''
    if (size < 1024) return `${size} B`
    if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`
    return `${(size / (1024 * 1024)).toFixed(1)} MB`
  }
</script>

<div class="backdrop" role="presentation" onclick={onClose}></div>
<div class="sheet" role="dialog" aria-modal="true" aria-label="Add documents">
  <header>
    <h2>Add documents</h2>
    <button class="close" onclick={onClose} aria-label="Close" type="button">
      <Icon name="x" size={16} />
    </button>
  </header>

  {#if loading}
    <p class="muted">Checking {paths.length} {paths.length === 1 ? 'item' : 'items'}…</p>
  {:else if error}
    <p class="err" role="alert"><Icon name="triangle-alert" size={14} /> {error}</p>
  {:else}
    <p class="headline">{headline}</p>

    <ul class="rows">
      {#each visible as f (f.path)}
        <li class="row {f.verdict}">
          <span class="ricon"><Icon name={icon(f.verdict)} size={14} /></span>
          <span class="rname" title={f.path}>{f.name}</span>
          <span class="rsize">{kb(f.size)}</span>
          {#if reason(f)}<span class="rwhy">{reason(f)}</span>{/if}
        </li>
      {/each}
    </ul>

    {#if hidden > 0}
      <!-- The server sorted every exception above every clean file, so what is hidden here is
           only ever ordinary additions — the count says so rather than leaving it implied. -->
      <p class="more">
        and {hidden} more, all ready to add
        <button class="linkish" onclick={() => (shown += 200)} type="button">Show more</button>
      </p>
    {/if}

    {#if applied}
      <!-- Grill branch 6: a run that stopped part-way must say what landed, what failed and what
           it never reached, then let the user keep or undo. Silence on the untouched files would
           read as "skipped", which they were not. -->
      <div class="outcome" role="status">
        {#if applied.stopped_early}
          <p class="warn">
            <Icon name="triangle-alert" size={14} />
            Stopped at <strong>{applied.failed?.name}</strong>: {applied.failed?.error}
          </p>
          <p class="muted">
            {applied.added.length} added · {applied.not_attempted.length} not attempted
          </p>
        {:else}
          <p class="ok"><Icon name="check" size={14} /> Added {applied.added.length}</p>
        {/if}
      </div>
    {:else}
      <fieldset class="where">
        <legend>Where they go</legend>
        <label class="opt">
          <input
            type="radio"
            name="mode"
            value="copy"
            checked={mode === 'copy'}
            onchange={() => (mode = 'copy')}
            disabled={applying}
          />
          <span>Copy into my Provenote folder</span>
        </label>
        <label class="opt">
          <input
            type="radio"
            name="mode"
            value="reference"
            checked={mode === 'reference'}
            onchange={() => (mode = 'reference')}
            disabled={applying}
          />
          <span>Leave them where they are <em>— Provenote will not move or copy them</em></span>
        </label>
      </fieldset>

      <label class="opt">
        <input type="checkbox" bind:checked={indexNow} disabled={applying} />
        <span>Index them now {estimate ? `— ${estimate}` : ''}</span>
      </label>
    {/if}

    {#if applyError}
      <p class="err" role="alert"><Icon name="triangle-alert" size={14} /> {applyError}</p>
    {/if}
  {/if}

  <footer>
    {#if applied}
      {#if applied.added.length > 0}
        <button class="secondary" onclick={undoAll} disabled={undoing} type="button">
          {undoing ? 'Undoing…' : 'Undo all'}
        </button>
      {/if}
      <button class="primary" onclick={keepAndClose} type="button">
        {applied.stopped_early ? `Keep the ${applied.added.length}` : 'Done'}
      </button>
    {:else}
      <button class="secondary" onclick={onClose} type="button">Cancel</button>
      <button
        class="primary"
        onclick={apply}
        disabled={applying || loading || addable.length === 0}
        type="button"
      >
        {applying ? 'Adding…' : `Add ${addable.length} document${addable.length === 1 ? '' : 's'}`}
      </button>
    {/if}
  </footer>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(20, 19, 15, 0.35);
    z-index: 70;
  }
  .sheet {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 71;
    width: min(680px, calc(100vw - 2rem));
    max-height: min(78vh, 900px);
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 1rem 1.1rem 0.9rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
  }
  header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  h2 {
    margin: 0;
    font-size: 1.05rem;
  }
  .close {
    margin-left: auto;
    background: none;
    border: none;
    color: var(--fg-2);
    cursor: pointer;
    padding: 0.2rem;
  }
  .headline {
    margin: 0;
    font-size: var(--text-sm);
    color: var(--fg);
  }
  .muted,
  .next {
    margin: 0;
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .err {
    margin: 0;
    font-size: var(--text-sm);
    color: var(--danger);
  }
  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .row {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: var(--text-sm);
  }
  .row.duplicate,
  .row.unsupported,
  .row.unreadable {
    background: var(--warn-bg);
    border-color: var(--warn-border);
    color: var(--warn-fg);
  }
  .rname {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .rsize {
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .rwhy {
    grid-column: 2 / -1;
    font-size: var(--text-meta);
    opacity: 0.9;
  }
  .ricon {
    display: inline-flex;
  }
  .more {
    margin: 0;
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .linkish {
    background: none;
    border: none;
    padding: 0;
    margin-left: 0.4rem;
    font: inherit;
    color: var(--accent);
    cursor: pointer;
    text-decoration: underline;
  }
  footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    padding-top: 0.2rem;
  }
  .where {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.7rem 0.6rem;
    margin: 0;
  }
  legend {
    font-size: var(--text-meta);
    color: var(--fg-2);
    padding: 0 0.3rem;
  }
  .opt {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: var(--text-sm);
    padding: 0.15rem 0;
  }
  .opt em {
    font-style: normal;
    color: var(--fg-2);
    font-size: var(--text-meta);
  }
  .outcome p {
    margin: 0;
    font-size: var(--text-sm);
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  .outcome .warn {
    color: var(--warn-fg);
  }
  .outcome .ok {
    color: var(--ok-fg);
  }
  .primary {
    background: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font: inherit;
    font-size: var(--text-sm);
    color: var(--accent-fg);
    cursor: pointer;
  }
  .primary:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .secondary {
    background: none;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    color: var(--fg);
  }
</style>
