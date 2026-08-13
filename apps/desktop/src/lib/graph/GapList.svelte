<script lang="ts">
  // First-class gap list (ROADMAP E5, ADR-004 / ADR-017 C1). A triageable list is the right
  // renderer for the gap payload — RG-014 found the strong signals (`single_source`,
  // `unsourced_claim`) are LIST-shaped, the weak ones (`under_connected`) graph-shaped. So the gap
  // list is not "the graph, again"; it is the corroboration/coverage worklist.
  //
  // Self-contained on purpose (the recorded Graph-destination iteration gate): it fetches its own
  // effective-status data and owns its triage writes, so it can later move out of the Graph view
  // with zero coupling. `onSelectConcept` is the only host hook — in the graph it jumps the ego view
  // to that concept.
  import type { GapListItem, GapKind, GapStatus } from '../core/types'
  import { getGapList, triageGap } from '../core/api'
  import Icon from '../shell/Icon.svelte'
  import { GAP_META, filterGapRows, gapVisible, orderGaps } from './gaps'

  let { onSelectConcept }: { onSelectConcept?: (conceptId: string) => void } = $props()

  let items = $state<GapListItem[]>([])
  let loading = $state(true)
  let error = $state<string | null>(null)
  let showUnderConnected = $state(false)
  let showDismissed = $state(false)
  // Filter box (ui-checklist §2). Matches the concept label *and* the gap kind, so "single"
  // finds every Single-source row — the list's own vocabulary is what people type.
  let query = $state('')
  // Guards a triage round-trip per (concept_id, kind) so double-clicks can't race.
  let busy = $state<Set<string>>(new Set())

  const key = (it: { concept_id: string; kind: GapKind }): string => `${it.concept_id}\0${it.kind}`

  async function load(): Promise<void> {
    loading = true
    error = null
    try {
      items = await getGapList()
    } catch (e) {
      error = String(e)
    } finally {
      loading = false
    }
  }
  void load()

  // RG-014 presentation: strong list-shaped kinds first; `under_connected` opt-in; dismissed hidden
  // unless the user asks (so a dismissal actually clears the worklist, but stays recoverable).
  const gapKey = (it: GapListItem): { kind: GapKind; label: string } => ({
    kind: it.kind,
    label: it.label,
  })

  const visible = $derived.by(() => {
    const rows = items
      .filter((it) => gapVisible(it.kind, showUnderConnected))
      .filter((it) => showDismissed || it.status !== 'dismissed')
    return orderGaps(filterGapRows(rows, gapKey, query), gapKey)
  })

  const openCount = $derived(items.filter((it) => it.status === 'surfaced').length)
  const dismissedCount = $derived(items.filter((it) => it.status === 'dismissed').length)

  async function setStatus(it: GapListItem, status: GapStatus): Promise<void> {
    const k = key(it)
    if (busy.has(k)) return
    busy = new Set(busy).add(k)
    try {
      await triageGap(it.concept_id, it.kind, status)
      // Reflect the effective status locally (the server resolves override ?? row on next load).
      items = items.map((x) => (key(x) === k ? { ...x, status } : x))
    } catch (e) {
      error = String(e)
    } finally {
      const next = new Set(busy)
      next.delete(k)
      busy = next
    }
  }
</script>

<div class="gaplist">
  {#if loading}
    <p class="muted">Loading gaps…</p>
  {:else if error}
    <p class="err" role="alert">Couldn’t load gaps: {error}</p>
  {:else if items.length === 0}
    <p class="muted">
      No gaps detected. This is empty before the concept graph is built, or when the corpus has no
      under-supported concepts.
    </p>
  {:else}
    <div class="searchrow">
      <Icon name="search" size={13} />
      <input
        bind:value={query}
        placeholder="Filter gaps…"
        aria-label="Filter gaps by concept or kind"
      />
      {#if query}
        <button class="clearq" onclick={() => (query = '')} aria-label="Clear filter" type="button">
          <Icon name="x" size={13} />
        </button>
      {/if}
    </div>

    <!-- One control style for the two lenses (ui-checklist §2): both are on/off filters over the
         same list, so both are pressed-state buttons. A checkbox beside a button read as two
         different kinds of thing doing the same job. -->
    <div class="lenses">
      <span class="tally" title="Surfaced (untriaged) gaps">{openCount} open</span>
      <button
        class="lens"
        class:on={showUnderConnected}
        aria-pressed={showUnderConnected}
        onclick={() => (showUnderConnected = !showUnderConnected)}
        type="button"
        title="Under-connected measures graph degree. It is noisy at this vocabulary size (RG-014), so it is hidden until you ask for it."
      >
        Include under-connected
      </button>
      {#if dismissedCount > 0}
        <button
          class="lens"
          class:on={showDismissed}
          aria-pressed={showDismissed}
          onclick={() => (showDismissed = !showDismissed)}
          type="button"
          title="Dismissed gaps stay recoverable — this shows them again without un-dismissing them."
        >
          Show dismissed <span class="count">{dismissedCount}</span>
        </button>
      {/if}
    </div>

    {#if visible.length === 0}
      <p class="muted">No gap matches “{query}”.</p>
    {/if}

    <ul>
      {#each visible as it (key(it))}
        {@const meta = GAP_META[it.kind]}
        <li class="gaprow" class:dismissed={it.status === 'dismissed'}>
          <div class="head">
            <span class="badge {meta.tone}" title={meta.blurb}>{meta.label}</span>
            <button
              class="conceptlink"
              type="button"
              onclick={() => onSelectConcept?.(it.concept_id)}
              title="Show this concept in the graph"
            >
              {it.label}
            </button>
            {#if it.status === 'promoted'}<span class="statustag ok">promoted</span>{/if}
            {#if it.status === 'dismissed'}<span class="statustag muted">dismissed</span>{/if}
          </div>
          <p class="blurb">{meta.blurb}</p>
          {#if it.fact_ids.length > 0}
            <p class="evidence muted" title="Supporting graph facts / claims">
              {it.fact_ids.length} item{it.fact_ids.length === 1 ? '' : 's'} of evidence
            </p>
          {/if}
          <div class="actions">
            {#if it.status !== 'promoted'}
              <button type="button" onclick={() => setStatus(it, 'promoted')} disabled={busy.has(key(it))}>
                Promote
              </button>
            {/if}
            {#if it.status !== 'dismissed'}
              <button type="button" onclick={() => setStatus(it, 'dismissed')} disabled={busy.has(key(it))}>
                Dismiss
              </button>
            {/if}
            {#if it.status !== 'surfaced'}
              <button type="button" class="ghost" onclick={() => setStatus(it, 'surfaced')} disabled={busy.has(key(it))}>
                Reset
              </button>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .gaplist {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-height: 0;
    overflow-y: auto;
  }
  .muted {
    color: var(--fg-2);
    font-size: var(--text-sm);
  }
  .err {
    color: var(--warn-fg);
    font-size: var(--text-sm);
  }
  .lenses {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-3);
    font-size: var(--text-sm);
    color: var(--fg-2);
  }
  .tally {
    font-weight: 600;
    color: var(--fg);
  }
  /* Same shapes as the concept rail's controls (GraphIndex.svelte) — the two panels are the
     same rail, so a filter box and a lens must not look like different inventions per tab. */
  .searchrow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.5rem;
    color: var(--fg-2);
    background: transparent;
  }
  .searchrow:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 22%, transparent);
  }
  .searchrow input {
    flex: 1;
    min-width: 0;
    border: none;
    background: none;
    font: inherit;
    font-size: 0.78rem;
    color: var(--fg);
    outline: none;
  }
  .clearq {
    border: none;
    background: none;
    color: var(--fg-2);
    cursor: pointer;
    display: inline-flex;
    padding: 0;
  }
  .clearq:hover {
    color: var(--fg);
  }
  .lens {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 2px var(--space-3);
    background: var(--surface);
    color: var(--fg);
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .lens.on {
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
  }
  .count {
    font-size: var(--text-meta);
    opacity: 0.8;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .gaprow {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    padding: var(--space-2) var(--space-3);
  }
  .gaprow.dismissed {
    opacity: 0.6;
  }
  .head {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .badge {
    border-radius: 999px;
    padding: 1px var(--space-2);
    font-size: var(--text-meta);
    font-weight: 600;
    white-space: nowrap;
  }
  .badge.danger {
    color: var(--danger);
    border: 1px solid var(--danger);
  }
  .badge.warn {
    color: var(--warn-fg);
    border: 1px solid var(--warn-fg);
  }
  .conceptlink {
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    font-weight: 600;
    color: var(--accent);
    cursor: pointer;
    text-align: left;
    overflow-wrap: anywhere;
    min-width: 0;
  }
  .conceptlink:hover {
    text-decoration: underline;
  }
  .statustag {
    font-size: var(--text-meta);
    border-radius: 6px;
    padding: 0 var(--space-2);
    border: 1px solid var(--border);
  }
  .statustag.ok {
    color: var(--ok-fg);
  }
  .statustag.muted {
    color: var(--fg-2);
  }
  .blurb {
    margin: var(--space-1) 0 0;
    font-size: var(--text-sm);
    color: var(--fg);
    line-height: 1.4;
  }
  .evidence {
    margin: var(--space-1) 0 0;
    font-size: var(--text-meta);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
  .actions button {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px var(--space-3);
    background: var(--bg);
    color: var(--fg);
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .actions button:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .actions button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .actions button.ghost {
    background: none;
    color: var(--fg-2);
  }
</style>
