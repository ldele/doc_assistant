<script lang="ts">
  // Inline keyword-filter strip above the Library grid (docs/specs/feature-keyword-filter-overlay.md).
  // The always-visible residue of the keyword filter: a "Filter by keyword" trigger that opens the
  // two-pane overlay, plus the currently-*selected* keywords as removable chips (the "selected on top"
  // idea). Dumb by design — App owns the selection; this renders + emits open/remove/clear.
  import Icon from './Icon.svelte'

  let {
    selected,
    resultCount,
    hasKeywords,
    onOpen,
    onRemove,
    onClear,
  }: {
    selected: string[]
    resultCount: number
    hasKeywords: boolean // the corpus has keywords to filter by (else the strip hides entirely)
    onOpen: () => void
    onRemove: (value: string) => void
    onClear: () => void
  } = $props()
</script>

{#if hasKeywords}
  <div class="strip">
    <button class="trigger" class:on={selected.length > 0} onclick={onOpen} type="button">
      <Icon name="tag" size={13} />
      {selected.length === 0 ? 'Filter by keyword' : `Keywords · ${selected.length}`}
    </button>
    {#if selected.length > 0}
      <div class="chips" role="group" aria-label="Selected keyword filters">
        {#each selected as k (k)}
          <button
            class="chip"
            type="button"
            title="Remove “{k}” filter"
            aria-label="Remove {k} filter"
            onclick={() => onRemove(k)}
          >
            <span class="chiptext">{k}</span>
            <Icon name="x" size={11} />
          </button>
        {/each}
      </div>
      <span class="count">{resultCount.toLocaleString()} doc{resultCount === 1 ? '' : 's'}</span>
      <button class="clear" onclick={onClear} type="button">Clear</button>
    {/if}
  </div>
{/if}

<style>
  .strip {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem 0.55rem;
    padding: var(--space-2) 0;
    border-bottom: 1px solid var(--border);
  }
  .trigger {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font: inherit;
    font-size: 0.75rem;
    cursor: pointer;
    color: var(--fg-2);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.2rem 0.6rem;
    flex: none;
  }
  .trigger:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  .trigger.on {
    color: var(--accent);
    border-color: var(--accent);
    font-weight: 600;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    min-width: 0;
  }
  /* Selected keyword — filled accent + ✕, clicking removes it (the overlay is where you add more). */
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font: inherit;
    font-size: 0.7rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--accent-fg);
    background: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 0.12rem 0.42rem;
    max-width: 100%;
    min-width: 0;
  }
  .chip:hover {
    filter: brightness(1.05);
  }
  .chiptext {
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .count {
    font-size: 0.72rem;
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    flex: none;
  }
  .clear {
    font: inherit;
    font-size: 0.72rem;
    cursor: pointer;
    background: none;
    border: none;
    color: var(--fg-2);
    padding: 0.1rem 0.2rem;
    border-radius: 6px;
    flex: none;
  }
  .clear:hover {
    color: var(--fg);
  }
</style>
