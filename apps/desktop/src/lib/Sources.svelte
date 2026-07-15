<script lang="ts">
  // Selective ingestion panel (feature-selective-ingestion.md, S2). Lives inside the Settings
  // drawer, under "Your documents". Scans the source folder (stat-only, $0), shows each file with
  // a derived status, lets the user exclude files and ingest a chosen subset. All rules are
  // server-side — this is a thin renderer over GET/PATCH /api/sources + POST /api/ingest {paths}.
  import type { IngestStatus, SourceFile } from './types'
  import { getSources, patchSource, startIngest, getIngestStatus } from './api'
  import { onDestroy } from 'svelte'
  import Icon from './Icon.svelte'

  let { onCorpusChanged }: { onCorpusChanged: () => void } = $props()

  let rows = $state<SourceFile[]>([])
  let loading = $state(false)
  let scanError = $state<string | null>(null)
  let selected = $state<Set<string>>(new Set())
  let busy = $state(false) // a selective-ingest cycle is in flight
  let ingest = $state<IngestStatus | null>(null)
  let ingestError = $state<string | null>(null)

  let cancelled = false
  onDestroy(() => {
    cancelled = true
  })

  // Only files that are on disk can be selected/ingested — a `missing` row (file gone) can't.
  const selectable = $derived(rows.filter((r) => r.status !== 'missing'))
  const counts = $derived({
    total: rows.length,
    new: rows.filter((r) => r.status === 'new').length,
    changed: rows.filter((r) => r.status === 'changed').length,
    excluded: rows.filter((r) => r.excluded).length,
  })
  const freshCount = $derived(
    selectable.filter((r) => (r.status === 'new' || r.status === 'changed') && !r.excluded).length,
  )

  void scan()

  async function scan(): Promise<void> {
    loading = true
    scanError = null
    try {
      const s = await getSources()
      if (cancelled) return
      rows = s
      // Drop selections for files that no longer exist / vanished from the scan.
      const present = new Set(s.filter((r) => r.status !== 'missing').map((r) => r.rel_path))
      selected = new Set([...selected].filter((p) => present.has(p)))
    } catch (e) {
      if (!cancelled) scanError = String(e)
    } finally {
      if (!cancelled) loading = false
    }
  }

  function toggleSelect(rel: string): void {
    const next = new Set(selected)
    if (next.has(rel)) next.delete(rel)
    else next.add(rel)
    selected = next
  }

  function selectNewChanged(): void {
    selected = new Set(
      selectable
        .filter((r) => (r.status === 'new' || r.status === 'changed') && !r.excluded)
        .map((r) => r.rel_path),
    )
  }

  function clearSelection(): void {
    selected = new Set()
  }

  // Persist the exclude flag; update the one row in place. Excluding also drops it from the
  // current selection (you excluded it — don't leave it queued in the batch).
  async function toggleExclude(row: SourceFile): Promise<void> {
    try {
      const updated = await patchSource(row.rel_path, !row.excluded)
      rows = rows.map((r) => (r.rel_path === updated.rel_path ? updated : r))
      if (updated.excluded && selected.has(updated.rel_path)) {
        const next = new Set(selected)
        next.delete(updated.rel_path)
        selected = next
      }
    } catch (e) {
      ingestError = String(e)
    }
  }

  async function ingestSelected(): Promise<void> {
    if (busy || selected.size === 0) return
    busy = true
    ingestError = null
    ingest = null
    try {
      ingest = await startIngest([...selected])
      await pollUntilDone()
    } catch (e) {
      ingestError = String(e)
    } finally {
      busy = false
    }
  }

  // Same tolerant poll as the whole-folder index in Settings: a transient blip on the local
  // sidecar shouldn't tear the cycle down; give up only if contact is genuinely lost.
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
      onCorpusChanged() // the header chunk count is now stale
      clearSelection()
      await scan() // re-derive statuses (ingested files flip new/changed → ingested)
    }
  }

  const STATUS_LABEL: Record<SourceFile['status'], string> = {
    new: 'new',
    changed: 'changed',
    ingested: 'indexed',
    missing: 'missing',
  }
</script>

<div class="sources">
  <div class="head">
    <p class="summary">
      {#if loading && rows.length === 0}
        Scanning…
      {:else}
        {counts.total} file{counts.total === 1 ? '' : 's'}
        {#if counts.new > 0}· <span class="c-new">{counts.new} new</span>{/if}
        {#if counts.changed > 0}· <span class="c-changed">{counts.changed} changed</span>{/if}
        {#if counts.excluded > 0}· {counts.excluded} excluded{/if}
      {/if}
    </p>
    <button class="rescan" onclick={scan} disabled={loading || busy} title="Rescan the folder">
      <Icon name="rotate-ccw" size={14} /> Rescan
    </button>
  </div>

  {#if scanError}
    <p class="err" role="alert">Couldn't scan: {scanError}</p>
  {:else if rows.length === 0 && !loading}
    <p class="hint">No files in the source folder. Choose a folder above, then rescan.</p>
  {:else if rows.length > 0}
    <div class="actions">
      <button class="ghost" onclick={selectNewChanged} disabled={busy || freshCount === 0}>
        Select new + changed{freshCount > 0 ? ` (${freshCount})` : ''}
      </button>
      {#if selected.size > 0}
        <button class="ghost" onclick={clearSelection} disabled={busy}>Clear</button>
      {/if}
      <button class="primary" onclick={ingestSelected} disabled={busy || selected.size === 0}>
        {busy ? 'Indexing…' : `Ingest selected (${selected.size})`}
      </button>
    </div>

    <ul class="list">
      {#each rows as r (r.rel_path)}
        <li class="row" class:excluded={r.excluded} class:missing={r.status === 'missing'}>
          <label class="pick">
            <input
              type="checkbox"
              checked={selected.has(r.rel_path)}
              disabled={busy || r.status === 'missing'}
              onchange={() => toggleSelect(r.rel_path)}
            />
            <span class="name" title={r.rel_path}>{r.rel_path}</span>
          </label>
          <span class="chip c-{r.status}">{STATUS_LABEL[r.status]}</span>
          <button
            class="excl"
            onclick={() => toggleExclude(r)}
            disabled={busy}
            title={r.excluded ? 'Include in whole-folder indexing' : 'Exclude from whole-folder indexing'}
          >
            {r.excluded ? 'Excluded' : 'Exclude'}
          </button>
        </li>
      {/each}
    </ul>

    <div aria-live="polite">
      {#if busy && ingest?.state === 'running'}
        <p class="muted">Indexing your selection. This can take a moment. You can keep this open.</p>
      {/if}
      {#if ingest?.state === 'done'}
        <p class="ok"><Icon name="check" size={14} /> {ingest.message}</p>
      {/if}
      {#if ingest?.state === 'error'}
        <p class="err" role="alert">Indexing failed: {ingest.message}</p>
      {/if}
      {#if ingestError}
        <p class="err" role="alert">{ingestError}</p>
      {/if}
    </div>
  {/if}
</div>

<style>
  .sources {
    margin-top: 0.4rem;
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .summary {
    margin: 0;
    font-size: 0.8rem;
    color: var(--fg-2);
  }
  .c-new {
    color: var(--accent);
  }
  .c-changed {
    color: var(--warn-fg);
  }
  .rescan {
    font: inherit;
    font-size: 0.78rem;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg-2);
    border-radius: 8px;
    padding: 0.25rem 0.55rem;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    flex: none;
  }
  .rescan:hover:not(:disabled) {
    color: var(--fg);
  }
  .rescan:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
    margin: 0.6rem 0;
  }
  .list {
    list-style: none;
    margin: 0;
    padding: 0;
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.5rem;
    border-bottom: 1px solid var(--border);
  }
  .row:last-child {
    border-bottom: none;
  }
  .row.excluded {
    opacity: 0.55;
  }
  .pick {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex: 1;
    min-width: 0;
    cursor: pointer;
    margin: 0;
  }
  .pick input {
    width: auto;
    accent-color: var(--accent);
    flex: none;
  }
  .name {
    font-size: 0.8rem;
    color: var(--fg);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .chip {
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border-radius: 5px;
    padding: 0.05rem 0.35rem;
    flex: none;
    border: 1px solid var(--border);
    color: var(--fg-2);
    background: var(--surface);
  }
  .chip.c-new {
    color: var(--accent);
    border-color: color-mix(in srgb, var(--accent) 40%, var(--border));
  }
  .chip.c-changed {
    color: var(--warn-fg);
    border-color: color-mix(in srgb, var(--warn-fg) 40%, var(--border));
  }
  .chip.c-missing {
    color: var(--danger);
    border-color: color-mix(in srgb, var(--danger) 40%, var(--border));
  }
  .excl {
    font: inherit;
    font-size: 0.72rem;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg-2);
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
    flex: none;
  }
  .excl:hover:not(:disabled) {
    color: var(--fg);
  }
  .excl:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .primary {
    font: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: var(--accent-fg);
    padding: 0.32rem 0.8rem;
    margin-left: auto;
  }
  .primary:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .ghost {
    font: inherit;
    font-size: 0.8rem;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    padding: 0.3rem 0.7rem;
  }
  .ghost:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .hint {
    font-size: 0.76rem;
    color: var(--fg-2);
    margin: 0.4rem 0 0;
  }
  .muted {
    color: var(--fg-2);
    font-size: 0.8rem;
    margin: 0.5rem 0 0;
  }
  .ok {
    color: var(--ok-fg);
    font-size: 0.8rem;
    margin: 0.5rem 0 0;
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.8rem;
    margin: 0.5rem 0 0;
  }
</style>
