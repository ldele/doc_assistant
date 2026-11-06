<script lang="ts">
  // Settings > Documents > Letter-spacing ignore list (ADR-056 point 5): every document the list
  // names, with its runs, and the way to empty it. Removing writes the list only; the documents
  // it named re-extract when their text is re-run, so Undo is safe until then and replaces the
  // confirmation a destructive button would otherwise need (inform, don't block).
  import type { IgnoreListEntry, IgnoreListRemoved } from '../../lib/core/types'
  import { getIgnoreList, removeIgnoreEntries, restoreIgnoreList } from '../../lib/core/api'
  import { removedNote } from '../../lib/library/spacing'
  import { openRerun } from '../../lib/library/reingest.svelte'
  import { shell } from '../../lib/shell/shell.svelte'

  let {
    focus = false,
  }: {
    /** Scroll the list into view once it has loaded (opened from a document's block). */
    focus?: boolean
  } = $props()

  // *Re-run text now…*: the list takes effect through the text. Settings closes first, so the
  // dialog opens over the window the person came from.
  function rerunText(ids: string[], label: string): void {
    shell.showSettings = false
    openRerun(ids, label, ['text'])
  }

  let entries = $state<IgnoreListEntry[] | null>(null)
  let error = $state<string | null>(null)
  let busy = $state(false)
  let open = $state<Set<string>>(new Set())
  let removed = $state<{ result: IgnoreListRemoved; ids: string[] } | null>(null)
  let sectionEl = $state<HTMLElement | null>(null)

  void load().then(() => {
    if (focus && sectionEl) holdInView(sectionEl)
  })

  /** Bring the list into view and keep it there for two seconds: the sections above it load
   *  after it (the file list grows by hundreds of pixels), and at scroll position 0 the browser's
   *  scroll anchoring does not hold it. The reader's first scroll, click or key hands it back.
   *  Sets the pane's `scrollTop` rather than `scrollIntoView`, which moves every scrolling
   *  ancestor (see ChunkContextView). */
  function holdInView(el: HTMLElement): void {
    let pane = el.parentElement
    while (pane && !/(auto|scroll)/.test(getComputedStyle(pane).overflowY)) pane = pane.parentElement
    if (!pane) return
    const scroller = pane
    const stop = new AbortController()
    for (const type of ['wheel', 'pointerdown', 'keydown', 'touchstart']) {
      window.addEventListener(type, () => stop.abort(), { capture: true, once: true, signal: stop.signal })
    }
    const until = performance.now() + 2000
    const step = (): void => {
      if (stop.signal.aborted) return
      if (performance.now() > until) {
        stop.abort()
        return
      }
      scroller.scrollTop += el.getBoundingClientRect().top - scroller.getBoundingClientRect().top
      requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }

  async function load(): Promise<void> {
    try {
      entries = (await getIgnoreList()).entries
      error = null
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    }
  }

  function summary(e: IgnoreListEntry): string {
    if (e.keep_all) return 'every run kept'
    const parts: string[] = []
    if (e.kept.length) parts.push(`${e.kept.length} ${e.kept.length === 1 ? 'run' : 'runs'} kept`)
    const yours = Object.keys(e.spacing).length
    if (yours) parts.push(`${yours} ${yours === 1 ? 'run' : 'runs'} spaced your way`)
    return parts.join(' · ')
  }

  async function remove(sha256: string | null): Promise<void> {
    busy = true
    error = null
    const ids = (entries ?? [])
      .filter((e) => sha256 === null || e.sha256 === sha256)
      .map((e) => e.document_id)
      .filter((id): id is string => id !== null)
    try {
      const result = await removeIgnoreEntries(sha256)
      removed = { result, ids }
      await load()
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  async function undo(): Promise<void> {
    if (!removed) return
    busy = true
    try {
      entries = (await restoreIgnoreList(removed.result.previous)).entries
      removed = null
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  function toggle(sha: string): void {
    const next = new Set(open)
    if (next.has(sha)) next.delete(sha)
    else next.add(sha)
    open = next
  }
</script>

<section class="ignore" bind:this={sectionEl}>
  <h3>Letter-spacing ignore list</h3>
  <p class="hint">
    Runs you keep as extracted or spaced your own way, by document. A change takes effect when that
    document's text is re-run.
  </p>
  {#if error}
    <p class="err" role="alert">{error}</p>
  {/if}

  {#if entries === null && !error}
    <p class="hint">Loading…</p>
  {:else if entries && entries.length === 0 && !removed}
    <p class="hint empty">The list is empty: every letter-spaced run follows the rule.</p>
  {:else if entries}
    <ul class="entries">
      {#each entries as e (e.sha256)}
        <li>
          <div class="row">
            <span class="name" title={e.file}>{e.file}</span>
            <span class="what">{summary(e)}{#if e.document_id === null}{' · not in this library'}{/if}</span>
            <span class="spacer"></span>
            {#if !e.keep_all}
              <button class="ghost" type="button" aria-expanded={open.has(e.sha256)} onclick={() => toggle(e.sha256)}>
                {open.has(e.sha256) ? 'Hide runs' : 'Show runs'}
              </button>
            {/if}
            <button class="ghost" type="button" disabled={busy} onclick={() => remove(e.sha256)}>Remove</button>
          </div>
          {#if open.has(e.sha256)}
            <ul class="runs">
              {#each e.kept as run (run)}<li><code>{run}</code> <span class="what">kept as extracted</span></li>{/each}
              {#each Object.entries(e.spacing) as [run, chosen] (run)}
                <li><code>{run}</code> <span class="what">as</span> <code>{chosen}</code></li>
              {/each}
            </ul>
          {/if}
        </li>
      {/each}
    </ul>
    {#if entries.length > 0}
      <button class="ghost" type="button" disabled={busy} onclick={() => remove(null)}>Remove all</button>
    {/if}
  {/if}

  {#if removed}
    <div class="undo" role="status">
      <p>{removedNote(removed.result.removed, removed.ids.length)}</p>
      <div class="acts">
        <button class="primary" type="button" disabled={busy} onclick={undo}>Undo</button>
        {#if removed.ids.length > 0}
          <button
            class="ghost"
            type="button"
            onclick={() =>
              rerunText(
                removed?.ids ?? [],
                removed && removed.ids.length === 1 ? (removed.result.documents[0] ?? '1 document') : `${removed?.ids.length ?? 0} documents`,
              )}
          >
            Re-run text now…
          </button>
        {/if}
      </div>
    </div>
  {/if}
</section>

<style>
  h3 {
    margin: 0 0 0.3rem;
  }
  .hint {
    margin: 0;
    font-size: 0.82rem;
    color: var(--fg-2);
    line-height: 1.5;
  }
  .err {
    margin: 0.4rem 0 0;
    font-size: 0.82rem;
    color: var(--danger);
  }
  .entries {
    list-style: none;
    margin: 0.6rem 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .name {
    font-weight: 600;
    font-size: 0.85rem;
    overflow-wrap: anywhere;
    min-width: 0;
  }
  .what {
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .spacer {
    flex: 1;
  }
  .runs {
    margin: 0.3rem 0 0 0.8rem;
    padding: 0;
    list-style: none;
    font-size: 0.8rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  code {
    font-size: 0.78rem;
    overflow-wrap: anywhere;
  }
  .undo {
    margin-top: 0.6rem;
    padding: 0.6rem 0.75rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface-2);
  }
  .undo p {
    margin: 0 0 0.5rem;
    font-size: 0.85rem;
  }
  .acts {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .ghost,
  .primary {
    font: inherit;
    font-size: 0.82rem;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    padding: 0.35rem 0.8rem;
  }
  .primary {
    font-weight: 600;
    border-color: var(--accent);
    background: var(--accent);
    color: var(--accent-fg);
  }
  .ghost:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .ghost:disabled,
  .primary:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .ghost:focus-visible,
  .primary:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
</style>
