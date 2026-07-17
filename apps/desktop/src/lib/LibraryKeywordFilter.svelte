<script lang="ts">
  // Two-pane keyword-filter overlay (docs/specs/feature-keyword-filter-overlay.md). Left: a
  // searchable keyword list (Zotero mechanics — AND, grey-out unavailable, most-used-on-top); right:
  // a live preview of the matching documents. Live commit (no Apply): toggling a keyword updates the
  // real selection immediately, so this shares App's `facetList` (from `keywordFacets`) + `previewDocs`
  // (the faceted+sorted grid docs). Reuses the LibraryMetaEditor modal shell (scrim + centered dialog,
  // Esc-to-close, autofocused control). Dumb by design — App owns the selection + the filter math.
  import type { KeywordFacet } from './library'
  import type { KeywordFamily, LibraryDocument } from './types'
  import { authorLabel, familyByCanonical } from './library'
  import Icon from './Icon.svelte'

  let {
    facets,
    previewDocs,
    selectedCount,
    families,
    onToggle,
    onClear,
    onClose,
    onManage,
  }: {
    facets: KeywordFacet[]
    previewDocs: LibraryDocument[]
    selectedCount: number
    families: KeywordFamily[]
    onToggle: (value: string) => void
    onClear: () => void
    onClose: () => void
    onManage: () => void
  } = $props()

  let query = $state('')
  // A facet's value that matches a family's canonical -> "N forms" subtitle + hover (feature-
  // tag-families.md, PR-1 DoD: "the overlay renders a family as an atomic entry").
  const familyMap = $derived(familyByCanonical(families))

  // The overlay's own search box filters the keyword list by substring. Selected keywords always
  // stay visible (so a search can't hide a chip you still need to unselect), matching Zotero.
  const shown = $derived.by(() => {
    const q = query.trim().toLowerCase()
    if (q === '') return facets
    return facets.filter((f) => f.selected || f.value.toLowerCase().includes(q))
  })

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
  function autofocus(node: HTMLInputElement): void {
    node.focus()
  }

  function byline(d: LibraryDocument): string {
    const a = authorLabel(d)
    const y = d.year != null ? String(d.year) : ''
    return a && y ? `${a} · ${y}` : a || y
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="Filter documents by keyword">
  <div class="mhead">
    <h2>Filter by keyword</h2>
    <button class="iconbtn" onclick={onClose} aria-label="Close" type="button">
      <Icon name="x" size={16} />
    </button>
  </div>

  <div class="panes">
    <!-- Left: searchable keyword picker -->
    <section class="pane" aria-label="Keywords">
      <div class="searchrow">
        <Icon name="search" size={14} />
        <input use:autofocus bind:value={query} placeholder="Search keywords" aria-label="Search keywords" />
        {#if query}
          <button class="clearq" onclick={() => (query = '')} aria-label="Clear search" type="button">
            <Icon name="x" size={13} />
          </button>
        {/if}
      </div>
      <div class="kwlist" role="group" aria-label="Keyword filters">
        {#each shown as f (f.value)}
          {@const fam = familyMap.get(f.value)}
          <button
            class="kwrow"
            class:selected={f.selected}
            class:unavailable={!f.available}
            type="button"
            aria-pressed={f.selected}
            disabled={!f.available && !f.selected}
            title={fam && fam.aliases.length > 0
              ? `Also matches: ${fam.aliases.join(', ')}`
              : !f.available && !f.selected
                ? `No documents match “${f.value}” with the current filters`
                : undefined}
            onclick={() => onToggle(f.value)}
          >
            <span class="tick">{#if f.selected}<Icon name="check" size={13} />{/if}</span>
            <span class="kwtext">
              <span class="kwname">{f.value}</span>
              {#if fam && fam.aliases.length > 0}
                <span class="kwforms">{fam.aliases.length + 1} forms</span>
              {/if}
            </span>
            {#if !f.selected}<span class="kwcount">{f.count}</span>{/if}
          </button>
        {:else}
          <p class="nomatch">No keywords match “{query.trim()}”.</p>
        {/each}
      </div>
      <div class="kwlistfoot">
        {#if selectedCount > 0}
          <button class="clearall" onclick={onClear} type="button">
            <Icon name="x" size={12} /> Clear {selectedCount} selected
          </button>
        {/if}
        <button class="managebtn" onclick={onManage} type="button">Manage keywords…</button>
      </div>
    </section>

    <!-- Right: live preview of the matching documents -->
    <section class="pane preview" aria-label="Matching documents">
      <p class="previewhead">
        {previewDocs.length.toLocaleString()} document{previewDocs.length === 1 ? '' : 's'}
      </p>
      <div class="doclist">
        {#each previewDocs as d (d.id)}
          <div class="docrow" title={d.filename}>
            <span class="docmark"><Icon name="file-text" size={14} /></span>
            <span class="doctext">
              <span class="doctitle">{d.title ?? d.filename}</span>
              {#if byline(d)}<span class="docby">{byline(d)}</span>{/if}
            </span>
          </div>
        {:else}
          <p class="nomatch">No documents match the selected keywords.</p>
        {/each}
      </div>
    </section>
  </div>
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
    width: min(94vw, 760px);
    max-height: min(84vh, 620px);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .mhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .mhead h2 {
    margin: 0;
    font-size: var(--text-title);
    font-family: var(--font-serif);
  }
  .iconbtn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.2rem;
    border: none;
    background: none;
    color: var(--fg-2);
    border-radius: 6px;
    cursor: pointer;
  }
  .iconbtn:hover {
    color: var(--fg);
    background: var(--surface-2);
  }
  /* Two equal panes, each scrolling on its own; the modal height is fixed so both stay reachable. */
  .panes {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
    overflow: hidden;
  }
  .pane {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    gap: var(--space-2);
  }
  .pane.preview {
    border-left: 1px solid var(--border);
    padding-left: var(--space-3);
  }
  .searchrow {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--fg-2);
    flex: none;
  }
  .searchrow:focus-within {
    border-color: var(--accent);
  }
  .searchrow input {
    font: inherit;
    font-size: 0.85rem;
    border: none;
    background: none;
    color: var(--fg);
    width: 100%;
    min-width: 0;
    outline: none;
  }
  .clearq {
    display: inline-flex;
    border: none;
    background: none;
    color: var(--fg-2);
    cursor: pointer;
    padding: 0;
  }
  .clearq:hover {
    color: var(--fg);
  }
  .kwlist {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    padding-right: 0.2rem;
  }
  .kwrow {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    width: 100%;
    text-align: left;
    font: inherit;
    font-size: 0.82rem;
    cursor: pointer;
    border: 1px solid transparent;
    background: none;
    color: var(--fg);
    border-radius: 7px;
    padding: 0.32rem 0.45rem;
  }
  .kwrow:hover {
    background: var(--surface-2);
  }
  .kwrow.selected {
    background: color-mix(in srgb, var(--accent) 14%, transparent);
    color: var(--fg);
    font-weight: 600;
  }
  .kwrow.unavailable {
    color: var(--fg-2);
    opacity: 0.45;
    cursor: not-allowed;
  }
  .tick {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    flex: none;
    color: var(--accent);
  }
  .kwtext {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
  }
  .kwname {
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .kwforms {
    font-size: 0.68rem;
    color: var(--fg-2);
    flex: none;
  }
  .kwcount {
    font-size: 0.7rem;
    font-variant-numeric: tabular-nums;
    color: var(--fg-2);
    flex: none;
  }
  .kwlistfoot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    flex: none;
  }
  .clearall {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font: inherit;
    font-size: 0.75rem;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg-2);
    padding: 0.2rem 0;
    flex: none;
  }
  .clearall:hover {
    color: var(--fg);
  }
  .managebtn {
    font: inherit;
    font-size: 0.75rem;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--accent);
    padding: 0.2rem 0;
    flex: none;
    margin-left: auto;
  }
  .managebtn:hover {
    text-decoration: underline;
  }
  .previewhead {
    margin: 0;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--fg-2);
    flex: none;
  }
  .doclist {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .docrow {
    display: flex;
    align-items: flex-start;
    gap: 0.45rem;
    padding: 0.32rem 0.2rem;
    border-radius: 7px;
    min-width: 0;
  }
  .docrow:hover {
    background: var(--surface-2);
  }
  .docmark {
    color: var(--accent);
    display: inline-flex;
    padding-top: 0.1rem;
    flex: none;
  }
  .doctext {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .doctitle {
    font-size: 0.82rem;
    line-height: 1.3;
    color: var(--fg);
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
  }
  .docby {
    font-size: 0.7rem;
    color: var(--fg-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .nomatch {
    margin: 0.4rem 0.2rem;
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  /* Mobile: stack the panes vertically, splitting the modal height between them. */
  @media (max-width: 640px) {
    .panes {
      grid-template-columns: 1fr;
      grid-template-rows: 1fr 1fr;
    }
    .pane.preview {
      border-left: none;
      padding-left: 0;
      border-top: 1px solid var(--border);
      padding-top: var(--space-2);
    }
  }
</style>
