<script lang="ts">
  // Left-rail app shell (feature-conversation-history.md, Decisions 7, 8; Library tab enabled by
  // feature-library-browser.md, L1; management actions by feature-conversation-management.md;
  // library nav tree by feature-library-redesign.md, L4 Phase A). Hosts the Chat/Library switch,
  // "New chat", and — depending on `mode` — the conversation history (with per-row pin + ⋯ menu,
  // pinned grouped into their own section) or the library navigation tree (All documents →
  // Collections → Types → Added → Keywords; the doc list itself moved to the main-pane grid).
  // Persistent column on desktop; an off-canvas drawer under 720px.
  import type { ConversationSummary, LibraryDocument } from './types'
  import {
    type LibraryCollection,
    DATE_BUCKET_LABELS,
    collectionLabel,
    dateGroups,
    folderGroups,
    sameCollection,
    typeGroups,
  } from './library'
  import Icon from './Icon.svelte'

  let {
    mode,
    conversations,
    documents,
    liveSessionId,
    viewingSessionId,
    libraryCollection,
    libraryQuery = $bindable(''),
    open = false,
    onNew,
    onSelect,
    onSelectMode,
    onSelectCollection,
    onClose,
    onPin,
    onArchive,
    onDelete,
    onRename,
  }: {
    mode: 'chat' | 'library' | 'graph'
    conversations: ConversationSummary[]
    documents: LibraryDocument[]
    liveSessionId: string
    viewingSessionId: string | null
    libraryCollection: LibraryCollection
    libraryQuery?: string
    open?: boolean
    onNew: () => void
    onSelect: (sessionId: string) => void
    onSelectMode: (mode: 'chat' | 'library' | 'graph') => void
    onSelectCollection: (c: LibraryCollection) => void
    onClose?: () => void
    onPin: (sessionId: string, pinned: boolean) => void
    onArchive: (sessionId: string, archived: boolean) => void
    onDelete: (sessionId: string) => void
    onRename: (sessionId: string, title: string) => void
  } = $props()

  // Inline rename: "Rename" in the ⋯ menu turns the row's title into an editable input.
  let editingId = $state<string | null>(null)
  let editValue = $state('')
  function startRename(c: ConversationSummary): void {
    editingId = c.session_id
    editValue = c.title
    closeMenu()
  }
  function saveRename(sid: string): void {
    if (editingId !== sid) return // guard the blur-after-Enter double fire
    editingId = null
    if (editValue.trim()) onRename(sid, editValue.trim())
  }
  function cancelRename(): void {
    editingId = null
  }
  function focusSelect(node: HTMLInputElement): void {
    node.focus()
    node.select()
  }

  // Search (V1: a plain case-insensitive title filter, ephemeral, ✕-to-clear). Chat search stays
  // local to this rail; the library search moved to a bindable prop (`libraryQuery`) because it now
  // filters the *active collection* shown in the main-pane grid (L4 Decision 5a) — App owns that.
  let query = $state('')
  const q = $derived(query.trim().toLowerCase())

  // Sort order for the chat history. Persisted like the other client-only view prefs (theme, panel
  // widths). Default: newest-first. Pinned still float to their own section regardless of order.
  type SortKey = 'recent' | 'oldest' | 'az' | 'za'
  const SORT_KEYS: SortKey[] = ['recent', 'oldest', 'az', 'za']
  const SORT_LABELS: Record<SortKey, string> = {
    recent: 'Newest first',
    oldest: 'Oldest first',
    az: 'Name A–Z',
    za: 'Name Z–A',
  }
  function loadSort(): SortKey {
    try {
      const v = localStorage.getItem('convSort')
      if (v === 'recent' || v === 'oldest' || v === 'az' || v === 'za') return v
    } catch {
      /* ignore — fall back to default */
    }
    return 'recent'
  }
  let sortKey = $state<SortKey>(loadSort())
  let sortOpen = $state(false)
  let sortwrapEl = $state<HTMLElement | null>(null)
  function setSort(k: SortKey): void {
    sortKey = k
    sortOpen = false
    try {
      localStorage.setItem('convSort', k)
    } catch {
      /* ignore — order just won't persist */
    }
  }
  function ts(iso: string): number {
    const t = new Date(iso).getTime()
    return Number.isNaN(t) ? 0 : t
  }
  function sortConvos(list: ConversationSummary[], key: SortKey): ConversationSummary[] {
    const arr = [...list]
    if (key === 'oldest') arr.sort((a, b) => ts(a.last_at) - ts(b.last_at))
    else if (key === 'az') arr.sort((a, b) => a.title.localeCompare(b.title))
    else if (key === 'za') arr.sort((a, b) => b.title.localeCompare(a.title))
    else arr.sort((a, b) => ts(b.last_at) - ts(a.last_at))
    return arr
  }

  // Archived conversations are hidden behind a toggle; pinned ones get their own section on top.
  let showArchived = $state(false)
  const archivedCount = $derived(conversations.filter((c) => c.archived).length)
  const matchedConvos = $derived(
    (showArchived ? conversations : conversations.filter((c) => !c.archived)).filter(
      (c) => q === '' || c.title.toLowerCase().includes(q),
    ),
  )
  const sortedConvos = $derived(sortConvos(matchedConvos, sortKey))
  const pinnedConvos = $derived(sortedConvos.filter((c) => c.pinned))
  const otherConvos = $derived(sortedConvos.filter((c) => !c.pinned))
  // Library nav-tree groups (L4 Decision 3), computed client-side from the payload. Types/Added
  // render only with ≥2 entries (Decision 3a — a one-option filter is noise); Collections shows a
  // "why" empty-state until folders exist (Phase B populates them). Keywords are no longer a nav
  // group — they moved to the main-pane facet bar as a multi-select filter (pure-facet model).
  const types = $derived(typeGroups(documents))
  const dates = $derived(dateGroups(documents, new Date()))
  const folders = $derived(folderGroups(documents))

  // The per-row ⋯ menu: a single floating menu, positioned at the clicked button (fixed, so the
  // sidebar's overflow can't clip it). Closes on outside-click, Esc, or a list scroll.
  let openMenuFor = $state<string | null>(null)
  let menuPos = $state<{ x: number; y: number }>({ x: 0, y: 0 })
  const menuConvo = $derived(
    openMenuFor ? (conversations.find((c) => c.session_id === openMenuFor) ?? null) : null,
  )

  function openMenu(sid: string, ev: MouseEvent): void {
    ev.stopPropagation()
    if (openMenuFor === sid) {
      closeMenu()
      return
    }
    const r = (ev.currentTarget as HTMLElement).getBoundingClientRect()
    const flip = r.bottom + 130 > window.innerHeight
    menuPos = { x: Math.max(8, r.right - 168), y: flip ? Math.max(8, r.top - 130) : r.bottom + 4 }
    openMenuFor = sid
  }
  function closeMenu(): void {
    openMenuFor = null
  }
  function menuPin(): void {
    if (menuConvo) onPin(menuConvo.session_id, !menuConvo.pinned)
    closeMenu()
  }
  function menuRename(): void {
    if (menuConvo) startRename(menuConvo)
  }
  function menuArchive(): void {
    if (menuConvo) onArchive(menuConvo.session_id, !menuConvo.archived)
    closeMenu()
  }
  function menuDelete(): void {
    if (menuConvo) onDelete(menuConvo.session_id)
    closeMenu()
  }

  $effect(() => {
    function onKey(e: KeyboardEvent): void {
      if (e.key === 'Escape') {
        closeMenu()
        sortOpen = false
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // Close the sort dropdown on any click outside its wrapper (only wired while it's open).
  $effect(() => {
    if (!sortOpen) return
    function onDown(e: PointerEvent): void {
      if (sortwrapEl && !sortwrapEl.contains(e.target as Node)) sortOpen = false
    }
    window.addEventListener('pointerdown', onDown)
    return () => window.removeEventListener('pointerdown', onDown)
  })

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

  // The row shown in the main pane: the viewed chat, or the live chat when nothing is being viewed.
  function isActive(sid: string): boolean {
    return viewingSessionId === null ? sid === liveSessionId : sid === viewingSessionId
  }
</script>

<aside class="sidebar" class:open>
  <div class="top">
    <div class="modes" role="tablist" aria-label="Workspace">
      <button
        class="mode"
        class:active={mode === 'chat'}
        role="tab"
        aria-selected={mode === 'chat'}
        type="button"
        onclick={() => onSelectMode('chat')}
      >
        <Icon name="message-square" size={14} /> Chat
      </button>
      <button
        class="mode"
        class:active={mode === 'library'}
        role="tab"
        aria-selected={mode === 'library'}
        type="button"
        onclick={() => onSelectMode('library')}
      >
        <Icon name="library" size={14} /> Library
      </button>
      <button
        class="mode"
        class:active={mode === 'graph'}
        role="tab"
        aria-selected={mode === 'graph'}
        type="button"
        onclick={() => onSelectMode('graph')}
      >
        <Icon name="waypoints" size={14} /> Graph
      </button>
    </div>
    {#if mode === 'chat'}
      <button class="new" onclick={onNew} type="button"><Icon name="rotate-ccw" size={15} /> New chat</button>
    {/if}
  </div>

  {#if (mode === 'chat' && conversations.length > 0) || (mode === 'library' && documents.length > 0)}
    <div class="toolbar">
      <div class="search">
        <Icon name="search" size={14} />
        {#if mode === 'chat'}
          <input type="text" bind:value={query} placeholder="Search chats" aria-label="Search chats" />
          {#if query}
            <button class="clear" onclick={() => (query = '')} aria-label="Clear search" type="button">
              <Icon name="x" size={13} />
            </button>
          {/if}
        {:else}
          <!-- Library search filters the active collection shown in the grid (Decision 5a). -->
          <input
            type="text"
            bind:value={libraryQuery}
            placeholder={libraryCollection.kind === 'all'
              ? 'Search library'
              : `Search ${collectionLabel(libraryCollection)}`}
            aria-label="Search library"
          />
          {#if libraryQuery}
            <button
              class="clear"
              onclick={() => (libraryQuery = '')}
              aria-label="Clear search"
              type="button"
            >
              <Icon name="x" size={13} />
            </button>
          {/if}
        {/if}
      </div>
      {#if mode === 'chat'}
        <div class="sortwrap" bind:this={sortwrapEl}>
          <button
            class="sortbtn"
            class:on={sortOpen}
            onclick={() => (sortOpen = !sortOpen)}
            aria-haspopup="menu"
            aria-expanded={sortOpen}
            title="Sort: {SORT_LABELS[sortKey]}"
            aria-label="Sort conversations"
            type="button"><Icon name="arrow-up-down" size={15} /></button
          >
          {#if sortOpen}
            <div class="sortmenu" role="menu">
              {#each SORT_KEYS as k}
                <button
                  class="sortitem"
                  class:active={sortKey === k}
                  role="menuitemradio"
                  aria-checked={sortKey === k}
                  onclick={() => setSort(k)}
                  type="button"
                >
                  <span class="tick">{#if sortKey === k}<Icon name="check" size={14} />{/if}</span>
                  {SORT_LABELS[k]}
                </button>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}

  {#snippet convRow(c: ConversationSummary)}
    <div
      class="convrow"
      class:active={isActive(c.session_id)}
      class:menuopen={openMenuFor === c.session_id}
    >
      {#if editingId === c.session_id}
        <input
          class="rename-input"
          bind:value={editValue}
          use:focusSelect
          onkeydown={(e) => {
            if (e.key === 'Enter') saveRename(c.session_id)
            else if (e.key === 'Escape') cancelRename()
          }}
          onblur={() => saveRename(c.session_id)}
        />
      {:else}
        <button
          class="rowmain"
          aria-current={isActive(c.session_id) ? 'true' : undefined}
          onclick={() => onSelect(c.session_id)}
          type="button"
        >
          <span class="title">{c.title}</span>
          <span class="rowmeta">
            {#if c.session_id === liveSessionId}<span class="dot" title="Current chat" aria-hidden="true"></span>{/if}
            <span>{relTime(c.last_at)} · {c.turn_count} turn{c.turn_count === 1 ? '' : 's'}</span>
          </span>
        </button>
        <div class="rowactions">
          <button
            class="act"
            class:on={c.pinned}
            title={c.pinned ? 'Unpin' : 'Pin'}
            aria-label={c.pinned ? 'Unpin conversation' : 'Pin conversation'}
            onclick={() => onPin(c.session_id, !c.pinned)}
            type="button"><Icon name="pin" size={14} /></button
          >
          <button
            class="act"
            title="More"
            aria-label="Conversation options"
            aria-haspopup="menu"
            onclick={(e) => openMenu(c.session_id, e)}
            type="button"><Icon name="ellipsis" size={14} /></button
          >
        </div>
      {/if}
    </div>
  {/snippet}

  {#if mode === 'chat'}
    <nav class="list" aria-label="Conversation history" onscroll={closeMenu}>
      {#if conversations.length === 0}
        <p class="empty">No conversations yet. Ask a question to start one.</p>
      {:else}
        {#if pinnedConvos.length > 0}
          <p class="section-header">Pinned</p>
          {#each pinnedConvos as c (c.session_id)}{@render convRow(c)}{/each}
        {/if}
        {#if otherConvos.length > 0}
          <p class="section-header">Recent</p>
          {#each otherConvos as c (c.session_id)}{@render convRow(c)}{/each}
        {/if}
        {#if q !== '' && pinnedConvos.length === 0 && otherConvos.length === 0}
          <p class="empty">No chats match “{query.trim()}”.</p>
        {/if}
        {#if archivedCount > 0}
          <button class="archived-toggle" onclick={() => (showArchived = !showArchived)} type="button">
            {showArchived ? 'Hide' : 'Show'} archived ({archivedCount})
          </button>
        {/if}
      {/if}
    </nav>
  {:else if mode === 'library'}
    <nav class="list" aria-label="Library navigation">
      {#if documents.length === 0}
        <p class="empty">No documents indexed yet.</p>
      {:else}
        <button
          class="treerow"
          class:active={libraryCollection.kind === 'all'}
          aria-current={libraryCollection.kind === 'all' ? 'true' : undefined}
          onclick={() => onSelectCollection({ kind: 'all' })}
          type="button"
        >
          <span class="treeicon"><Icon name="library" size={14} /></span>
          <span class="treelabel">All documents</span>
          <span class="count">{documents.length}</span>
        </button>

        <p class="section-header">Collections</p>
        {#if folders.length === 0}
          <p class="tree-empty">No folders yet — folders arrive with source-dir mirroring (Phase B).</p>
        {:else}
          {#each folders as g (g.value)}
            <button
              class="treerow"
              class:active={sameCollection(libraryCollection, { kind: 'folder', value: g.value })}
              aria-current={sameCollection(libraryCollection, { kind: 'folder', value: g.value })
                ? 'true'
                : undefined}
              onclick={() => onSelectCollection({ kind: 'folder', value: g.value })}
              type="button"
            >
              <span class="treeicon"><Icon name="folder" size={14} /></span>
              <span class="treelabel">{g.value}</span>
              <span class="count">{g.count}</span>
            </button>
          {/each}
        {/if}

        {#if types.length >= 2}
          <p class="section-header">Types</p>
          {#each types as g (g.value)}
            <button
              class="treerow"
              class:active={sameCollection(libraryCollection, { kind: 'type', value: g.value })}
              aria-current={sameCollection(libraryCollection, { kind: 'type', value: g.value })
                ? 'true'
                : undefined}
              onclick={() => onSelectCollection({ kind: 'type', value: g.value })}
              type="button"
            >
              <span class="treeicon"><Icon name="file-text" size={14} /></span>
              <span class="treelabel">{g.value.toUpperCase()}</span>
              <span class="count">{g.count}</span>
            </button>
          {/each}
        {/if}

        {#if dates.length >= 2}
          <p class="section-header">Added</p>
          {#each dates as g (g.value)}
            <button
              class="treerow"
              class:active={sameCollection(libraryCollection, { kind: 'date', value: g.value })}
              aria-current={sameCollection(libraryCollection, { kind: 'date', value: g.value })
                ? 'true'
                : undefined}
              onclick={() => onSelectCollection({ kind: 'date', value: g.value })}
              type="button"
            >
              <span class="treeicon"><Icon name="calendar" size={14} /></span>
              <span class="treelabel">{DATE_BUCKET_LABELS[g.value]}</span>
              <span class="count">{g.count}</span>
            </button>
          {/each}
        {/if}

      {/if}
    </nav>
  {/if}
</aside>

{#if openMenuFor && menuConvo}
  <div class="menu-backdrop" onclick={closeMenu} role="presentation"></div>
  <div class="menu" style="left: {menuPos.x}px; top: {menuPos.y}px" role="menu">
    <button class="menuitem" role="menuitem" onclick={menuPin} type="button">
      <Icon name="pin" size={14} /> {menuConvo.pinned ? 'Unpin' : 'Pin'}
    </button>
    <button class="menuitem" role="menuitem" onclick={menuRename} type="button">
      <Icon name="pencil" size={14} /> Rename
    </button>
    <button class="menuitem muted" role="menuitem" onclick={menuArchive} type="button">
      <Icon name="archive" size={14} /> {menuConvo.archived ? 'Unarchive' : 'Archive'}
    </button>
    <div class="menusep"></div>
    <button class="menuitem danger" role="menuitem" onclick={menuDelete} type="button">
      <Icon name="trash-2" size={14} /> Delete
    </button>
  </div>
{/if}

{#if open}
  <div class="scrim" onclick={onClose} role="presentation"></div>
{/if}

<style>
  .sidebar {
    width: var(--sidebar-width, 260px);
    flex-shrink: 0;
    height: 100vh;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    background: var(--bg);
    overflow: hidden;
  }
  .top {
    padding: 0.8rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .modes {
    display: flex;
    gap: 0.3rem;
  }
  .mode {
    flex: 1;
    font: inherit;
    cursor: pointer;
    padding: 0.35rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg-2);
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
  }
  .mode.active {
    background: var(--surface-2);
    color: var(--fg);
    font-weight: 600;
  }
  .new {
    font: inherit;
    cursor: pointer;
    padding: 0.45rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg);
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
  }
  /* Search + sort toolbar — fixed header strip above the scrolling list. */
  .toolbar {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border);
  }
  .search {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--fg-2);
  }
  .search:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 22%, transparent);
  }
  .search input {
    flex: 1;
    min-width: 0;
    font: inherit;
    font-size: 0.82rem;
    border: none;
    background: none;
    color: var(--fg);
    padding: 0;
  }
  .search input:focus {
    outline: none;
  }
  .search input::placeholder {
    color: var(--fg-2);
  }
  .clear {
    font: inherit;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg-2);
    padding: 0;
    display: inline-flex;
    flex: none;
  }
  .clear:hover {
    color: var(--fg);
  }
  .sortwrap {
    position: relative;
    flex: none;
  }
  .sortbtn {
    font: inherit;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg-2);
    padding: 0.36rem;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
  }
  .sortbtn:hover,
  .sortbtn.on {
    background: var(--surface-2);
    color: var(--fg);
  }
  .sortmenu {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    z-index: 31;
    min-width: 168px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow-2);
    padding: 0.3rem;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .sortitem {
    font: inherit;
    font-size: 0.82rem;
    text-align: left;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg);
    padding: 0.4rem 0.5rem;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  .sortitem:hover {
    background: var(--surface-2);
  }
  .sortitem.active {
    color: var(--accent);
    font-weight: 600;
  }
  .tick {
    width: 14px;
    display: inline-flex;
    flex: none;
  }
  .list {
    flex: 1;
    overflow-y: auto;
    padding: 0.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .empty {
    color: var(--fg-2);
    font-size: 0.82rem;
    padding: 0.6rem;
    line-height: 1.4;
  }
  /* Lavender "tab" labels for Pinned / Recent — grouped + legible (user request). */
  .section-header {
    align-self: flex-start;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--lavender);
    background: color-mix(in srgb, var(--lavender) 14%, transparent);
    border-radius: 6px;
    margin: 0.55rem 0.1rem 0.3rem;
    padding: 0.16rem 0.5rem;
  }
  /* Chat row: a container (main button + hover-revealed pin + ⋯). */
  .convrow {
    display: flex;
    align-items: center;
    border: 1px solid transparent;
    border-radius: 8px;
    width: 100%;
  }
  .convrow:hover {
    background: var(--surface);
  }
  .convrow.active {
    background: var(--surface-2);
    border-color: var(--border);
  }
  .rowmain {
    text-align: left;
    font: inherit;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg);
    padding: 0.45rem 0.55rem;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    flex: 1;
    min-width: 0;
  }
  .rename-input {
    flex: 1;
    min-width: 0;
    margin: 0.3rem 0.4rem;
    font: inherit;
    font-size: 0.85rem;
    color: var(--fg);
    background: var(--bg);
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 0.3rem 0.4rem;
  }
  .rename-input:focus {
    outline: none;
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 30%, transparent);
  }
  /* Actions show only on mouse hover, or while this row's menu is open (item 6: no focus-within). */
  .rowactions {
    display: none;
    align-items: center;
    gap: 0.05rem;
    padding-right: 0.25rem;
    flex: none;
  }
  .convrow:hover .rowactions,
  .convrow.menuopen .rowactions {
    display: flex;
  }
  .act {
    font: inherit;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg-2);
    padding: 0.22rem;
    border-radius: 6px;
    display: inline-flex;
  }
  .act:hover {
    background: var(--surface-2);
    color: var(--fg);
  }
  .act.on {
    color: var(--accent);
  }
  .archived-toggle {
    font: inherit;
    font-size: 0.75rem;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg-2);
    padding: 0.4rem 0.55rem;
    margin-top: 0.2rem;
    text-align: left;
    text-decoration: underline;
  }
  /* Library nav tree (L4): single-line rows — icon · label · doc count. */
  .treerow {
    font: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    text-align: left;
    border: 1px solid transparent;
    background: none;
    color: var(--fg);
    border-radius: 8px;
    padding: 0.4rem 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    width: 100%;
    min-width: 0;
  }
  .treerow:hover {
    background: var(--surface);
  }
  .treerow.active {
    background: var(--surface-2);
    border-color: var(--border);
  }
  .treeicon {
    color: var(--fg-2);
    display: inline-flex;
    flex: none;
  }
  .treerow.active .treeicon {
    color: var(--accent);
  }
  .treelabel {
    flex: 1;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .count {
    font-size: 0.68rem;
    color: var(--fg-2);
    flex: none;
  }
  .tree-empty {
    color: var(--fg-2);
    font-size: 0.74rem;
    line-height: 1.4;
    padding: 0 0.55rem;
    margin: 0 0 0.2rem;
  }
  .title {
    font-size: 0.85rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .rowmeta {
    font-size: 0.7rem;
    color: var(--fg-2);
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }
  .dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: var(--accent);
    flex: none;
  }
  /* ⋯ dropdown menu (fixed — never clipped by the sidebar). */
  .menu-backdrop {
    position: fixed;
    inset: 0;
    z-index: 30;
  }
  .menu {
    position: fixed;
    z-index: 31;
    min-width: 160px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow-2);
    padding: 0.3rem;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .menuitem {
    font: inherit;
    font-size: 0.82rem;
    text-align: left;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg);
    padding: 0.4rem 0.5rem;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .menuitem:hover {
    background: var(--surface-2);
  }
  .menuitem.muted {
    color: var(--fg-2);
  }
  .menuitem.danger {
    color: var(--danger);
  }
  .menuitem.danger:hover {
    background: color-mix(in srgb, var(--danger) 12%, transparent);
  }
  .menusep {
    height: 1px;
    background: var(--border);
    margin: 0.2rem 0.1rem;
  }
  .scrim {
    display: none;
  }

  @media (max-width: 720px) {
    .sidebar {
      position: fixed;
      top: 0;
      left: 0;
      width: min(85vw, 320px);
      z-index: 20;
      transform: translateX(-100%);
      transition: transform 0.2s ease;
      box-shadow: 4px 0 20px rgba(0, 0, 0, 0.2);
    }
    .sidebar.open {
      transform: translateX(0);
    }
    .scrim {
      display: block;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.35);
      z-index: 15;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .sidebar {
      transition: none;
    }
  }
</style>
