<script lang="ts">
  // The field forest, the left side of the Taxonomy view: the *Proposed placements* row (shown
  // while anything awaits review), the field search, and the research fields as an indented tree
  // with their concept · document counts. Owns the search. Which field is selected and which pane
  // the right side shows are the view's: a field click is handed up, and the proposals row sets
  // the pane (bound).
  import type { ProposedEdge, TaxonomyView } from '../../lib/core/types'
  import { buildForest } from '../../lib/library/taxonomy'
  import Icon from '../../shell/Icon.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    view,
    proposals,
    selectedId,
    pane = $bindable(),
    onSelectField,
  }: {
    view: TaxonomyView
    proposals: ProposedEdge[] // counted on the row here; the right side lists them
    selectedId: string | null
    pane: 'field' | 'proposals'
    onSelectField: (fieldId: string) => void
  } = $props()

  // The forest as ordered display rows (root→child, depth for indent).
  const rows = $derived(buildForest(view))

  // Field-label search. When active, matched fields render flat (depth 0) — the same "flatten the
  // pool on search" pattern Manage-keywords uses; the full indented forest returns when it clears.
  let query = $state('')
  const shownRows = $derived.by(() => {
    const q = query.trim().toLowerCase()
    if (q === '') return rows
    return rows
      .filter((r) => r.field.label.toLowerCase().includes(q))
      .map((r) => ({ ...r, depth: 0 }))
  })
</script>

<section class="forest" aria-label="Field forest">
  {#if proposals.length > 0}
    <button
      class="proposalsrow"
      class:sel={pane === 'proposals'}
      onclick={() => (pane = 'proposals')}
      type="button"
      aria-pressed={pane === 'proposals'}
      use:tooltip={'List the placements proposed for you to accept or reject'}
    >
      <Icon name="tag" size={13} />
      <span>Proposed placements</span>
      <span class="pcount">{proposals.length}</span>
    </button>
  {/if}
  <div class="searchrow small">
    <Icon name="search" size={13} />
    <input bind:value={query} placeholder="Search fields" aria-label="Search fields" />
  </div>
  <div class="treescroll">
    {#each shownRows as row (row.field.id + ':' + row.depth)}
      <button
        class="fieldrow"
        class:sel={selectedId === row.field.id}
        style="padding-left: {0.5 + row.depth * 0.95}rem"
        onclick={() => onSelectField(row.field.id)}
        type="button"
        aria-pressed={selectedId === row.field.id}
        use:tooltip={'Show this field’s concepts and documents'}
      >
        <span class="flabel">{row.field.label}</span>
        <span
          class="fcount"
          class:zero={row.field.n_concepts_rollup === 0 && row.field.n_documents_rollup === 0}
          title="Concepts · documents under this field (including narrower fields)"
        >
          {row.field.n_concepts_rollup} · {row.field.n_documents_rollup}
        </span>
      </button>
    {:else}
      <p class="hint pad">No fields match “{query.trim()}”.</p>
    {/each}
  </div>
</section>

<style>
  .hint {
    font-size: 0.78rem;
    color: var(--fg-2);
    margin: 0;
  }
  .hint.pad {
    padding: 0.6rem 0.2rem;
  }
  .forest {
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem;
    overflow: hidden;
  }
  .searchrow.small {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex: none;
    border: 1px solid var(--line, var(--border));
    border-radius: var(--radius-sm, 6px);
    padding: 0.2rem 0.45rem;
    color: var(--fg-2);
  }
  .searchrow.small input {
    flex: 1;
    min-width: 0;
    border: none;
    background: none;
    color: var(--fg);
    font: inherit;
    font-size: 0.8rem;
    outline: none;
  }
  .treescroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  .fieldrow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    text-align: left;
    font: inherit;
    font-size: 0.82rem;
    color: var(--fg);
    background: none;
    border: none;
    border-radius: 6px;
    padding: 0.28rem 0.4rem;
    cursor: pointer;
  }
  .fieldrow:hover {
    background: var(--surface-2);
  }
  .fieldrow.sel {
    background: var(--accent);
    color: var(--accent-fg);
  }
  .flabel {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .fcount {
    flex: none;
    font-size: 0.7rem;
    font-variant-numeric: tabular-nums;
    color: var(--fg-2);
  }
  .fieldrow.sel .fcount {
    color: var(--accent-fg);
  }
  .fcount.zero {
    opacity: 0.45;
  }
  .proposalsrow {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    width: 100%;
    font: inherit;
    font-size: 0.8rem;
    color: var(--fg);
    background: color-mix(in srgb, var(--warn-fg, var(--accent)) 10%, transparent);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.45rem;
    cursor: pointer;
    text-align: left;
  }
  .proposalsrow.sel {
    background: var(--accent);
    color: var(--accent-fg);
  }
  .pcount {
    margin-left: auto;
    font-variant-numeric: tabular-nums;
    font-size: 0.76rem;
  }
  @media (max-width: 640px) {
    /* Stacked on mobile: cap the forest height so the detail stays reachable below it. */
    .forest {
      max-height: 40vh;
    }
  }
</style>
