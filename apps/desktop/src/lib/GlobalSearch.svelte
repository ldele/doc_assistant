<script lang="ts">
  // Global-search overlay (docs/specs/feature-app-shell-search-collapse.md, sub-item a). A
  // *navigation* search across conversation titles + document title/filename/authors/keywords —
  // it jumps you to a chat or a paper, it does NOT answer questions (that is the composer). Dumb
  // by design: App owns `query` (bound) + derives `results` via `searchEverything`; this component
  // renders and emits selections. Reuses the LibraryKeywordFilter modal shell (scrim + centred
  // dialog, Esc-to-close, autofocused input), single-column.
  import type { ConversationSummary, LibraryDocument } from './types'
  import type { SearchResults } from './search'
  import { authorLabel } from './library'
  import Icon from './Icon.svelte'

  let {
    query = $bindable(''),
    results,
    onSelectChat,
    onSelectDoc,
    onClose,
  }: {
    query?: string
    results: SearchResults
    onSelectChat: (sessionId: string) => void
    onSelectDoc: (documentId: string) => void
    onClose: () => void
  } = $props()

  // A flat, keyboard-navigable list over both groups so ↓ crosses the Chats→Documents boundary
  // (A5). Each entry carries just what a click needs.
  type Row =
    | { kind: 'chat'; id: string; convo: ConversationSummary }
    | { kind: 'doc'; id: string; doc: LibraryDocument }
  const flat = $derived<Row[]>([
    ...results.chats.map((c): Row => ({ kind: 'chat', id: c.session_id, convo: c })),
    ...results.docs.map((d): Row => ({ kind: 'doc', id: d.id, doc: d })),
  ])

  let active = $state(0)
  // A new result set (query changed) re-highlights the first row. Reading `results` subscribes.
  $effect(() => {
    void results
    active = 0
  })
  // Keep the highlighted row in view when driven by the keyboard (short list, but honest).
  let rowEls = $state<(HTMLButtonElement | null)[]>([])
  $effect(() => {
    rowEls[active]?.scrollIntoView({ block: 'nearest' })
  })

  function choose(row: Row): void {
    if (row.kind === 'chat') onSelectChat(row.id)
    else onSelectDoc(row.id)
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      onClose()
    } else if (e.key === 'ArrowDown') {
      if (flat.length === 0) return
      e.preventDefault()
      active = (active + 1) % flat.length
    } else if (e.key === 'ArrowUp') {
      if (flat.length === 0) return
      e.preventDefault()
      active = (active - 1 + flat.length) % flat.length
    } else if (e.key === 'Enter') {
      const row = flat[active]
      if (row) {
        e.preventDefault()
        choose(row)
      }
    }
  }

  function autofocus(node: HTMLInputElement): void {
    node.focus()
  }

  // ISO → "3d ago" etc., NaN-safe. Local to the overlay (the sidebar has its own copy — a 10-line
  // formatter isn't worth a shared module).
  function relTime(iso: string): string {
    const then = new Date(iso).getTime()
    if (Number.isNaN(then)) return ''
    const secs = Math.max(0, (Date.now() - then) / 1000)
    if (secs < 60) return 'just now'
    const mins = Math.floor(secs / 60)
    if (mins < 60) return `${mins}m ago`
    const hrs = Math.floor(mins / 60)
    if (hrs < 24) return `${hrs}h ago`
    const days = Math.floor(hrs / 24)
    if (days < 7) return `${days}d ago`
    return new Date(iso).toLocaleDateString()
  }

  function byline(d: LibraryDocument): string {
    const a = authorLabel(d)
    const y = d.year != null ? String(d.year) : ''
    return a && y ? `${a} · ${y}` : a || y
  }

  // Row index within the flat list, for the highlight + Enter target.
  const chatBase = 0
  const docBase = $derived(results.chats.length)
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="Search chats and documents">
  <div class="searchrow">
    <Icon name="search" size={16} />
    <input
      use:autofocus
      bind:value={query}
      placeholder="Search chats and documents"
      aria-label="Search chats and documents"
    />
    {#if query}
      <button class="clearq" onclick={() => (query = '')} aria-label="Clear search" type="button">
        <Icon name="x" size={14} />
      </button>
    {/if}
    <button class="closebtn" onclick={onClose} aria-label="Close search" type="button">Esc</button>
  </div>

  <div class="results" role="listbox" aria-label="Search results">
    {#if flat.length === 0}
      <p class="empty">
        {#if query.trim() === ''}
          Type to search your chats and documents.
        {:else}
          No chats or documents match “{query.trim()}”.
        {/if}
      </p>
    {:else}
      {#if results.chats.length > 0}
        <p class="grouphead">{results.mode === 'recent' ? 'Recent' : 'Chats'}</p>
        {#each results.chats as c, i (c.session_id)}
          <button
            bind:this={rowEls[chatBase + i]}
            class="row"
            class:active={active === chatBase + i}
            role="option"
            aria-selected={active === chatBase + i}
            onmouseenter={() => (active = chatBase + i)}
            onclick={() => onSelectChat(c.session_id)}
            type="button"
          >
            <span class="rowicon"><Icon name="message-square" size={15} /></span>
            <span class="rowtext">
              <span class="rowtitle">{c.title}</span>
              <span class="rowmeta"
                >{relTime(c.last_at)} · {c.turn_count} turn{c.turn_count === 1 ? '' : 's'}</span
              >
            </span>
          </button>
        {/each}
        {#if results.mode === 'query' && results.chatsTotal > results.chats.length}
          <p class="more">+{results.chatsTotal - results.chats.length} more — keep typing</p>
        {/if}
      {/if}

      {#if results.docs.length > 0}
        <p class="grouphead">Documents</p>
        {#each results.docs as d, i (d.id)}
          <button
            bind:this={rowEls[docBase + i]}
            class="row"
            class:active={active === docBase + i}
            role="option"
            aria-selected={active === docBase + i}
            onmouseenter={() => (active = docBase + i)}
            onclick={() => onSelectDoc(d.id)}
            type="button"
          >
            <span class="rowicon"><Icon name="file-text" size={15} /></span>
            <span class="rowtext">
              <span class="rowtitle">{d.title ?? d.filename}</span>
              {#if byline(d)}<span class="rowmeta">{byline(d)}</span>{/if}
            </span>
          </button>
        {/each}
        {#if results.docsTotal > results.docs.length}
          <p class="more">+{results.docsTotal - results.docs.length} more — keep typing</p>
        {/if}
      {/if}
    {/if}
  </div>
</div>

<style>
  /* Modal shell — same tokens as LibraryKeywordFilter (scrim + centred dialog). */
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--fg) 32%, transparent);
    z-index: 40;
  }
  .modal {
    position: fixed;
    z-index: 41;
    top: 12vh;
    left: 50%;
    transform: translateX(-50%);
    width: min(94vw, 640px);
    max-height: min(76vh, 560px);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .searchrow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--bg);
    color: var(--fg-2);
    flex: none;
  }
  .searchrow:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 22%, transparent);
  }
  .searchrow input {
    font: inherit;
    font-size: 0.95rem;
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
    flex: none;
  }
  .clearq:hover {
    color: var(--fg);
  }
  .closebtn {
    font: inherit;
    font-size: 0.68rem;
    letter-spacing: 0.03em;
    color: var(--fg-2);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.15rem 0.4rem;
    cursor: pointer;
    flex: none;
  }
  .closebtn:hover {
    color: var(--fg);
  }
  .results {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    padding-right: 0.15rem;
  }
  .grouphead {
    align-self: flex-start;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--fg-2);
    margin: 0.5rem 0.2rem 0.15rem;
  }
  .results .grouphead:first-child {
    margin-top: 0.15rem;
  }
  .row {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    width: 100%;
    text-align: left;
    font: inherit;
    cursor: pointer;
    border: 1px solid transparent;
    background: none;
    color: var(--fg);
    border-radius: 8px;
    padding: 0.4rem 0.5rem;
  }
  /* Highlight is keyboard/hover-driven (no :hover rule — the shared `active` state owns it, so the
     mouse and arrow keys can't disagree about which row is selected). */
  .row.active {
    background: var(--surface-2);
    border-color: var(--border);
  }
  .rowicon {
    color: var(--accent);
    display: inline-flex;
    padding-top: 0.1rem;
    flex: none;
  }
  .rowtext {
    display: flex;
    flex-direction: column;
    min-width: 0;
    gap: 0.1rem;
  }
  .rowtitle {
    font-size: 0.88rem;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .rowmeta {
    font-size: 0.72rem;
    color: var(--fg-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .more {
    font-size: 0.72rem;
    color: var(--fg-2);
    margin: 0.15rem 0.2rem 0.2rem 2.1rem;
  }
  .empty {
    color: var(--fg-2);
    font-size: 0.85rem;
    padding: 1rem 0.5rem;
    text-align: center;
    line-height: 1.4;
  }
</style>
