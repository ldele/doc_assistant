<script lang="ts">
  // Inventory grid for the library main pane (feature-library-redesign.md, L4 Phase A,
  // Decision 2). Given the active collection's documents, renders them as a 2-D tile grid
  // (the "video-game inventory") or as list rows (today's .row idiom) per the persisted
  // view preference. Dumb by design: no filtering here — App owns collection + search.
  import type { LibraryDocument } from './types'
  import { docLabel } from './library'
  import Icon from './Icon.svelte'

  let {
    documents,
    view,
    onOpenDocument,
  }: {
    documents: LibraryDocument[]
    view: 'grid' | 'list'
    onOpenDocument: (id: string) => void
  } = $props()

  // NULL metadata is omitted, never shown blank (L1's honest-empty rule).
  function addedShort(iso: string | null): string | null {
    if (!iso) return null
    const t = new Date(iso)
    return Number.isNaN(t.getTime()) ? null : t.toLocaleDateString()
  }
</script>

{#if view === 'grid'}
  <div class="grid">
    {#each documents as d (d.id)}
      <button
        class="tile"
        onclick={() => onOpenDocument(d.id)}
        type="button"
        title={d.filename}
        aria-label={docLabel(d)}
      >
        <span class="tilehead">
          <span class="docmark"><Icon name="file-text" size={16} /></span>
          <span class="fmt">{d.format}</span>
        </span>
        <span class="name">{docLabel(d)}</span>
        <span class="meta">
          {#if d.page_count != null}<span>{d.page_count} pages</span>{/if}
          {#if d.chunk_count != null}<span>{d.chunk_count.toLocaleString()} chunks</span>{/if}
          {#if addedShort(d.added_at)}<span>{addedShort(d.added_at)}</span>{/if}
        </span>
        {#if d.keywords.length > 0}
          <!-- Unkeyed: a doc's raw keyword array may repeat a string, and this static sublist
               never reorders — keying by value would crash on a duplicate. Cap at 3 + a
               "+N" overflow chip so tags never wrap unpredictably (fixed card footprint). -->
          <span class="kws">
            {#each d.keywords.slice(0, 3) as k}<span class="kw">{k}</span>{/each}
            {#if d.keywords.length > 3}
              <span class="kw more" title={d.keywords.slice(3).join(', ')}
                >+{d.keywords.length - 3}</span
              >
            {/if}
          </span>
        {/if}
      </button>
    {/each}
  </div>
{:else}
  <div class="rows">
    {#each documents as d (d.id)}
      <button
        class="row"
        onclick={() => onOpenDocument(d.id)}
        type="button"
        title={d.filename}
        aria-label={docLabel(d)}
      >
        <span class="name">{docLabel(d)}</span>
        <span class="rowmeta">
          <span>{d.format}</span>
          {#if d.page_count != null}<span>· {d.page_count} pages</span>{/if}
          {#if d.chunk_count != null}<span>· {d.chunk_count.toLocaleString()} chunks</span>{/if}
          {#if addedShort(d.added_at)}<span>· {addedShort(d.added_at)}</span>{/if}
        </span>
      </button>
    {/each}
  </div>
{/if}

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: var(--space-3);
    padding: var(--space-2) 0;
  }
  .tile {
    font: inherit;
    cursor: pointer;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    min-width: 0;
    /* Fixed footprint: a long filename clamps (below) rather than reflowing the row,
       and the floor keeps keyword-less tiles level with their neighbours. */
    min-height: 128px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--fg);
    padding: 0.6rem 0.65rem;
  }
  .tile:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow-1);
  }
  .tilehead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
  }
  .docmark {
    color: var(--accent);
    display: inline-flex;
  }
  .fmt {
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--fg-2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 0.06rem 0.32rem;
    background: var(--surface-2);
    flex: none;
  }
  .name {
    font-size: var(--text-sm);
    line-height: 1.35;
    overflow-wrap: anywhere;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    /* Reserve two lines so single- and double-line titles align the metadata below. */
    min-height: 2.7em;
  }
  .meta {
    font-size: 0.68rem;
    color: var(--fg-2);
    display: flex;
    flex-wrap: wrap;
    gap: 0.15rem 0.45rem;
    margin-top: auto;
  }
  .kws {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
  }
  .kw {
    font-size: 0.64rem;
    color: var(--lavender);
    background: color-mix(in srgb, var(--lavender) 12%, transparent);
    border-radius: 5px;
    padding: 0.05rem 0.35rem;
    max-width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* Overflow count (+N) — muted, never truncates, sits at the end of the tag row. */
  .kw.more {
    color: var(--fg-2);
    background: var(--surface-2);
    flex: none;
  }
  /* List view — the stacked title+meta row idiom the old rail used. */
  .rows {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: var(--space-2) 0;
  }
  .row {
    font: inherit;
    cursor: pointer;
    text-align: left;
    border: 1px solid transparent;
    background: none;
    color: var(--fg);
    border-radius: 8px;
    padding: 0.45rem 0.55rem;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    width: 100%;
    min-width: 0;
  }
  .row:hover {
    background: var(--surface);
  }
  .row .name {
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.85rem;
  }
  .rowmeta {
    font-size: 0.7rem;
    color: var(--fg-2);
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
</style>
