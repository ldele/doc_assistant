<script lang="ts">
  // The concept index rail — the Graph view's navigation, hosted in the shared Sidebar (App
  // composes this and injects it via Sidebar's `graphRail` snippet prop, so the graph data and
  // selection stay App-owned). Extracted from ConceptGraph.svelte so the view has ONE left rail:
  // this index in the sidebar, the ego graph filling the main pane. Rail-only presentation state
  // (tab, filter query, gaps-only lens) is local and ephemeral, like the chat-history filter; the
  // under-connected lens is bindable because the ego panel's gap notes must agree with the badges.
  import type { ConceptGraphNode, Gap } from './types'
  import { GAP_META, conceptIndexRows, visibleConceptGaps } from './gaps'
  import GapList from './GapList.svelte'
  import Icon from './Icon.svelte'

  let {
    nodes,
    gaps,
    selectedId,
    showUnderConnected = $bindable(false),
    loading,
    built,
    graphError,
    onSelectConcept,
  }: {
    nodes: ConceptGraphNode[]
    gaps: Gap[]
    selectedId: string | null
    showUnderConnected?: boolean
    loading: boolean
    built: boolean
    graphError: string | null
    onSelectConcept: (id: string) => void
  } = $props()

  // Rail mode (E5): the concept index, or the first-class triageable gap list. The Gaps tab stays
  // reachable even while the graph is loading/unbuilt — GapList fetches its own data.
  let railMode = $state<'concepts' | 'gaps'>('concepts')
  let query = $state('')
  let gapsOnly = $state(false)

  // concept_id → its gaps (a concept can carry more than one kind).
  const gapsByConcept = $derived.by(() => {
    const m = new Map<string, Gap[]>()
    for (const g of gaps) {
      const list = m.get(g.concept_id) ?? []
      list.push(g)
      m.set(g.concept_id, list)
    }
    return m
  })
  const indexRows = $derived(conceptIndexRows(nodes, gapsByConcept, query, gapsOnly, showUnderConnected))
  const gapConceptCount = $derived(
    nodes.filter((n) => visibleConceptGaps(gapsByConcept.get(n.id) ?? [], showUnderConnected).length > 0)
      .length,
  )

  function commColor(n: ConceptGraphNode): string {
    return `var(--comm-${((n.community % 12) + 12) % 12})`
  }
</script>

<div class="graphindex">
  <div class="railmode segmented" role="tablist" aria-label="Rail mode">
    <button
      type="button"
      role="tab"
      aria-selected={railMode === 'concepts'}
      class:active={railMode === 'concepts'}
      onclick={() => (railMode = 'concepts')}
    >
      Concepts
    </button>
    <button
      type="button"
      role="tab"
      aria-selected={railMode === 'gaps'}
      class:active={railMode === 'gaps'}
      onclick={() => (railMode = 'gaps')}
    >
      Gaps
    </button>
  </div>

  {#if railMode === 'gaps'}
    <GapList {onSelectConcept} />
  {:else if loading}
    <p class="railstate muted">Loading concept graph…</p>
  {:else if graphError}
    <p class="railstate muted">Couldn’t load the concept graph.</p>
  {:else if !built}
    <p class="railstate muted">No concept graph yet — build it from the main panel.</p>
  {:else}
    <div class="searchrow">
      <Icon name="search" size={13} />
      <input bind:value={query} placeholder="Filter concepts…" aria-label="Filter concepts" />
      {#if query}
        <button class="clearq" onclick={() => (query = '')} aria-label="Clear filter" type="button">
          <Icon name="x" size={13} />
        </button>
      {/if}
    </div>
    <div class="lenses">
      <button
        class="lens"
        class:on={gapsOnly}
        aria-pressed={gapsOnly}
        onclick={() => (gapsOnly = !gapsOnly)}
        type="button"
        title="Show only concepts with a detected gap"
      >
        Gaps only <span class="count">{gapConceptCount}</span>
      </button>
      <label class="toggle" title="Under-connected is noisy at this vocabulary size (RG-014)">
        <input type="checkbox" bind:checked={showUnderConnected} />
        Include under-connected
      </label>
    </div>
    <div class="clist" role="listbox" aria-label="Concepts">
      {#each indexRows as row (row.node.id)}
        {@const g = row.gaps[0]}
        <button
          class="crow"
          class:sel={row.node.id === selectedId}
          role="option"
          aria-selected={row.node.id === selectedId}
          onclick={() => onSelectConcept(row.node.id)}
          type="button"
        >
          <span class="dot" style="background:{commColor(row.node)}" aria-hidden="true"></span>
          <span class="clabel">{row.node.label}</span>
          {#if g}
            <span class="badge {GAP_META[g.kind].tone}" title={GAP_META[g.kind].blurb}>
              {GAP_META[g.kind].label}{row.gaps.length > 1 ? ` +${row.gaps.length - 1}` : ''}
            </span>
          {/if}
          <span class="dcount" title="{row.node.doc_ids.length} documents">{row.node.doc_ids.length}</span>
        </button>
      {:else}
        <p class="empty-list muted">No concepts match.</p>
      {/each}
    </div>
  {/if}
</div>

<style>
  .graphindex {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding-top: var(--space-2);
  }
  .muted {
    color: var(--fg-2);
  }
  .railstate {
    font-size: var(--text-sm);
    padding: var(--space-2) var(--space-2);
    line-height: 1.4;
    margin: 0;
  }
  .railmode {
    display: flex;
    gap: 2px;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 2px;
    background: var(--surface);
  }
  .railmode button {
    flex: 1;
    border: none;
    border-radius: 6px;
    padding: 3px var(--space-3);
    background: none;
    color: var(--fg-2);
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .railmode button.active {
    background: var(--bg);
    color: var(--fg);
    font-weight: 600;
  }
  /* Filter styling matches the sidebar's other list filters (transparent, small) — the header's
     Ctrl/⌘-K overlay is the one global search. */
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
  .lenses {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
    font-size: var(--text-sm);
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
  .lens .count {
    font-variant-numeric: tabular-nums;
    opacity: 0.8;
  }
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    color: var(--fg-2);
    cursor: pointer;
  }
  .clist {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding-right: 2px;
  }
  .crow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    text-align: left;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: var(--space-2) var(--space-2);
    background: none;
    color: var(--fg);
    font: inherit;
    cursor: pointer;
  }
  .crow:hover {
    background: var(--surface);
  }
  .crow.sel {
    background: var(--surface-2);
    border-color: var(--border);
  }
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex: none;
  }
  .clabel {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .badge {
    font-size: var(--text-meta);
    padding: 1px 7px;
    border-radius: 999px;
    white-space: nowrap;
  }
  .badge.danger {
    color: var(--danger);
    border: 1px solid var(--danger);
  }
  .badge.warn {
    color: var(--warn-fg);
    border: 1px solid var(--warn-border);
    background: var(--warn-bg);
  }
  .dcount {
    font-size: var(--text-meta);
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    min-width: 1.5em;
    text-align: right;
  }
  .empty-list {
    padding: var(--space-3);
    font-size: var(--text-sm);
  }
</style>
