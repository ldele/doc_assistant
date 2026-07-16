<script lang="ts">
  // Inventory grid for the library main pane (feature-library-redesign.md, L4 Phase A,
  // Decision 2). Given the active collection's documents, renders them as a 2-D tile grid
  // (the "video-game inventory") or as list rows per the persisted view preference. Dumb by
  // design: no filtering here — App owns collection + search. Per-tile ⋯ menu (ADR-013):
  // Edit metadata / Reveal in file explorer, mirroring the conversation ⋯ menu in Sidebar.
  import type { LibraryDocument } from './types'
  import { authorLabel, docLabel } from './library'
  import Icon from './Icon.svelte'

  let {
    documents,
    view,
    onOpenDocument,
    onEditMetadata,
    onReveal,
  }: {
    documents: LibraryDocument[]
    view: 'grid' | 'list'
    onOpenDocument: (id: string) => void
    onEditMetadata: (id: string) => void
    onReveal: (id: string) => void
  } = $props()

  // NULL metadata is omitted, never shown blank (L1's honest-empty rule).
  function addedShort(iso: string | null): string | null {
    if (!iso) return null
    const t = new Date(iso)
    return Number.isNaN(t.getTime()) ? null : t.toLocaleDateString()
  }

  // A single floating ⋯ menu (position:fixed, so the scrolling main pane can't clip it),
  // flipped up when near the viewport bottom. Mirrors Sidebar's conversation menu.
  let openMenuFor = $state<string | null>(null)
  let menuPos = $state<{ x: number; y: number }>({ x: 0, y: 0 })
  const menuDoc = $derived(
    openMenuFor ? (documents.find((d) => d.id === openMenuFor) ?? null) : null,
  )
  function openMenu(id: string, ev: MouseEvent): void {
    ev.stopPropagation()
    if (openMenuFor === id) {
      closeMenu()
      return
    }
    const r = (ev.currentTarget as HTMLElement).getBoundingClientRect()
    const flip = r.bottom + 96 > window.innerHeight
    menuPos = { x: Math.max(8, r.right - 196), y: flip ? Math.max(8, r.top - 88) : r.bottom + 4 }
    openMenuFor = id
  }
  function closeMenu(): void {
    openMenuFor = null
  }
  function menuEdit(): void {
    if (menuDoc) onEditMetadata(menuDoc.id)
    closeMenu()
  }
  function menuReveal(): void {
    if (menuDoc) onReveal(menuDoc.id)
    closeMenu()
  }
  // Close on Esc or any scroll/wheel (the fixed menu would otherwise detach from its tile).
  $effect(() => {
    if (openMenuFor === null) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeMenu()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('wheel', closeMenu, { passive: true })
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('wheel', closeMenu)
    }
  })
</script>

{#if view === 'grid'}
  <div class="grid">
    {#each documents as d (d.id)}
      <div class="tile" class:menuopen={openMenuFor === d.id}>
        <button
          class="tilebody"
          onclick={() => onOpenDocument(d.id)}
          type="button"
          title={d.filename}
          aria-label={docLabel(d)}
        >
          <span class="tilehead">
            <span class="docmark"><Icon name="file-text" size={16} /></span>
            <span class="fmt">{d.format}</span>
            {#if d.customized}<span class="editmark" title="Edited metadata"></span>{/if}
          </span>
          <span class="name">{d.title ?? d.filename}</span>
          {#if authorLabel(d)}<span class="authors">{authorLabel(d)}</span>{/if}
          <span class="meta">
            {#if d.year != null}<span>{d.year}</span>{/if}
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
        <button
          class="tileact"
          title="More options"
          aria-label="Document options"
          aria-haspopup="menu"
          onclick={(e) => openMenu(d.id, e)}
          type="button"><Icon name="ellipsis" size={15} /></button
        >
      </div>
    {/each}
  </div>
{:else}
  <div class="rows">
    {#each documents as d (d.id)}
      <div class="row" class:menuopen={openMenuFor === d.id}>
        <button
          class="rowbody"
          onclick={() => onOpenDocument(d.id)}
          type="button"
          title={d.filename}
          aria-label={docLabel(d)}
        >
          <span class="name"
            >{d.title ?? d.filename}{#if d.customized}<span
                class="editmark inline"
                title="Edited metadata"
              ></span>{/if}</span
          >
          <span class="rowmeta">
            {#if authorLabel(d)}<span class="rowauthor">{authorLabel(d)}</span>{/if}
            <span
              >{d.format}{#if d.year != null} · {d.year}{/if}{#if d.page_count != null}
                · {d.page_count} pages{/if}{#if d.chunk_count != null}
                · {d.chunk_count.toLocaleString()} chunks{/if}{#if addedShort(d.added_at)}
                · {addedShort(d.added_at)}{/if}</span
            >
          </span>
        </button>
        <button
          class="rowact"
          title="More options"
          aria-label="Document options"
          aria-haspopup="menu"
          onclick={(e) => openMenu(d.id, e)}
          type="button"><Icon name="ellipsis" size={14} /></button
        >
      </div>
    {/each}
  </div>
{/if}

{#if openMenuFor && menuDoc}
  <div class="menu-backdrop" onclick={closeMenu} role="presentation"></div>
  <div class="menu" style="left: {menuPos.x}px; top: {menuPos.y}px" role="menu">
    <button class="menuitem" role="menuitem" onclick={menuEdit} type="button">
      <Icon name="pencil" size={14} /> Edit metadata
    </button>
    <button class="menuitem" role="menuitem" onclick={menuReveal} type="button">
      <Icon name="folder" size={14} /> Reveal in file explorer
    </button>
  </div>
{/if}

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: var(--space-3);
    padding: var(--space-2) 0;
  }
  /* The tile is a container (a <button> can't nest the ⋯ <button>): a full-card body button
     plus an absolutely-positioned ⋯ action revealed on hover. */
  .tile {
    position: relative;
    min-width: 0;
  }
  .tilebody {
    font: inherit;
    cursor: pointer;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    width: 100%;
    min-width: 0;
    /* Fixed footprint: a long title clamps (below) rather than reflowing the row, and the
       floor keeps author-/keyword-less tiles level with their neighbours. */
    min-height: 150px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--fg);
    padding: 0.6rem 0.65rem;
  }
  .tilebody:hover,
  .tile.menuopen .tilebody {
    border-color: var(--accent);
    box-shadow: var(--shadow-1);
  }
  .tilehead {
    display: flex;
    align-items: center;
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
  /* "Edited" indicator — a small accent dot when a user override is in force (d.customized). */
  .editmark {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    flex: none;
  }
  .editmark.inline {
    display: inline-block;
    margin-left: 0.35rem;
    vertical-align: middle;
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
    /* Reserve two lines so single- and double-line titles align what follows. */
    min-height: 2.7em;
  }
  .authors {
    font-size: 0.72rem;
    color: var(--fg-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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
  .kw.more {
    color: var(--fg-2);
    background: var(--surface-2);
    flex: none;
  }
  /* The ⋯ action — hidden until the tile is hovered or its menu is open (mirrors Sidebar). */
  .tileact,
  .rowact {
    position: absolute;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 0.15rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--fg-2);
    cursor: pointer;
  }
  .tileact {
    top: 6px;
    right: 6px;
  }
  .tileact:hover,
  .rowact:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  .tile:hover .tileact,
  .tile.menuopen .tileact,
  .row:hover .rowact,
  .row.menuopen .rowact {
    display: inline-flex;
  }
  /* List view — the stacked title+meta row idiom the old rail used. */
  .rows {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: var(--space-2) 0;
  }
  .row {
    position: relative;
    min-width: 0;
  }
  .rowbody {
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
  .rowbody:hover,
  .row.menuopen .rowbody {
    background: var(--surface);
  }
  .row .name {
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.85rem;
    min-height: 0;
  }
  .rowmeta {
    font-size: 0.7rem;
    color: var(--fg-2);
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .rowauthor {
    color: var(--fg);
  }
  .rowact {
    top: 50%;
    right: 6px;
    transform: translateY(-50%);
  }
  /* The floating ⋯ menu — mirrors Sidebar's conversation menu. */
  .menu-backdrop {
    position: fixed;
    inset: 0;
    z-index: 30;
  }
  .menu {
    position: fixed;
    z-index: 31;
    min-width: 190px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: var(--shadow-2);
    padding: 0.25rem;
  }
  .menuitem {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.4rem 0.55rem;
    border: none;
    background: none;
    color: var(--fg);
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 0.82rem;
    text-align: left;
  }
  .menuitem:hover {
    background: var(--surface-2);
  }
</style>
