<script lang="ts">
  // Chat history: the Chat tab's sidebar rail (feature-conversation-history.md, Decisions 7, 8;
  // management actions by feature-conversation-management.md). The filter, select mode and sort,
  // pinned grouped into their own section, per-row pin + ⋯ menu, inline rename, the archived
  // toggle. App hands it to the sidebar frame (`shell/Sidebar.svelte`) as the chat rail.
  import type { ConversationSummary } from '../../lib/core/types'
  import Icon from '../../shell/Icon.svelte'
  import { isSelectClick, pick, selectAllLabel, selectAllOrNone } from '../../lib/shell/selection'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    conversations,
    conversationsLoaded,
    liveSessionId,
    viewingSessionId,
    onNew,
    onSelect,
    onPin,
    onArchive,
    onDelete,
    onRename,
    onDeleteMany,
    onArchiveMany,
  }: {
    conversations: ConversationSummary[]
    /** Whether the list above is an answer yet. An empty list before the first fetch is not one. */
    conversationsLoaded: boolean
    liveSessionId: string
    viewingSessionId: string | null
    onNew: () => void
    onSelect: (sessionId: string) => void
    onPin: (sessionId: string, pinned: boolean) => void
    onArchive: (sessionId: string, archived: boolean) => void
    onDelete: (sessionId: string) => void
    onRename: (sessionId: string, title: string) => void
    /** Bulk soft-delete of the ticked conversations. The *selection* stays local to this
     *  component — nothing else in the app needs to know which rows are ticked, unlike the
     *  Library's select mode, where the main pane's add-to-folder menu participates too. */
    onDeleteMany: (sessionIds: string[]) => void
    /** Bulk archive/unarchive of the ticked conversations. Archiving is reversible and hides
     *  nothing the user cannot get back from the "Show archived" toggle, so it applies straight
     *  away — unlike delete, which routes through a confirm in the parent. */
    onArchiveMany: (sessionIds: string[], archived: boolean) => void
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

  // --- Select mode: tick several chats, delete them in one action (user request 2026-08-10).
  // Shift-click ranges and Ctrl/Cmd-click toggles (ROADMAP 78; logic in `selection.ts`), and
  // either one on a chat outside select mode starts it, so the button is not the only way in.
  let selectMode = $state(false)
  let selected = $state<string[]>([])
  let anchor = $state<string | null>(null)
  // In the order shown: pinned first, then the rest. A Shift-click ranges over what is on screen,
  // so the filter and the archived toggle are part of the range, and of Select all.
  const visibleIds = $derived([...pinnedConvos, ...otherConvos].map((c) => c.session_id))

  function enterSelect(): void {
    selectMode = true
    selected = []
    anchor = null
  }
  function exitSelect(): void {
    selectMode = false
    selected = []
    anchor = null
  }
  function pickRow(sid: string, shift: boolean): void {
    const r = pick(visibleIds, selected, anchor, sid, shift)
    selected = r.selected
    anchor = r.anchor
  }
  // A Shift- or Ctrl-click on a chat while none is ticked: select mode starts with that chat, and
  // a Shift-click ranges from the chat that is open, the row the user was last on.
  function startSelectFrom(sid: string, shift: boolean): void {
    const open = visibleIds.find((id) => isActive(id)) ?? null
    enterSelect()
    anchor = shift ? open : null
    pickRow(sid, shift)
  }
  // Shift-click would otherwise also drag a text selection across the rows it ranges over.
  function noTextSelect(e: MouseEvent): void {
    if (e.shiftKey) e.preventDefault()
  }
  // Delete asks first, in the parent (ADR-063). The selection stays until the answer: a Cancel
  // keeps it, and a confirmed delete empties it below, as the deleted chats leave the list.
  function deleteSelected(): void {
    if (selected.length === 0) return
    onDeleteMany([...selected])
  }
  $effect(() => {
    const listed = new Set(conversations.map((c) => c.session_id))
    const kept = selected.filter((id) => listed.has(id))
    if (kept.length === selected.length) return
    if (kept.length === 0) exitSelect()
    else selected = kept
  })
  // Archive is a toggle, decided by what is ticked: all-archived means the user is looking at the
  // "Show archived" list and wants them back, anything else means archive. One button, no mode.
  const selectedAllArchived = $derived(
    selected.length > 0 &&
      selected.every((sid) => conversations.find((c) => c.session_id === sid)?.archived === true),
  )
  function archiveSelected(): void {
    if (selected.length === 0) return
    onArchiveMany([...selected], !selectedAllArchived)
    exitSelect()
  }

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

{#if conversations.length > 0}
  <div class="toolbar">
    <div class="search">
      <Icon name="search" size={13} />
      <input type="text" bind:value={query} placeholder="Filter chats…" aria-label="Filter chats" />
      {#if query}
        <button class="clear" onclick={() => (query = '')} aria-label="Clear filter" type="button" use:tooltip={'Empty the filter and show every chat'}>
          <Icon name="x" size={13} />
        </button>
      {/if}
    </div>
    <button
      class="btn btn-icon"
      onclick={() => (selectMode ? exitSelect() : enterSelect())}
      aria-label="Select chats"
      use:tooltip={selectMode ? 'Leave select mode' : 'Select chats to archive or delete them together'}
      aria-pressed={selectMode}
      type="button"><Icon name="check" size={15} /></button
    >
    <div class="sortwrap" bind:this={sortwrapEl}>
      <button
        class="btn btn-icon"
        onclick={() => (sortOpen = !sortOpen)}
        aria-haspopup="menu"
        aria-expanded={sortOpen}
        use:tooltip={`Change the order of the list (now: ${SORT_LABELS[sortKey]})`}
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
              use:tooltip={'Order the chats this way; the choice is remembered'}
            >
              <span class="tick">{#if sortKey === k}<Icon name="check" size={14} />{/if}</span>
              {SORT_LABELS[k]}
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}

{#snippet convRow(c: ConversationSummary)}
  <div
    class="convrow"
    class:active={isActive(c.session_id)}
    class:menuopen={openMenuFor === c.session_id}
    class:picked={selectMode && selected.includes(c.session_id)}
  >
    {#if selectMode}
      <!-- The whole row toggles the tick: in select mode there is nothing else a click on a
           chat could sensibly mean, and a 13px checkbox is a poor target. -->
      <button
        class="rowmain pickrow"
        aria-pressed={selected.includes(c.session_id)}
        onmousedown={noTextSelect}
        onclick={(e) => pickRow(c.session_id, e.shiftKey)}
        type="button"
        use:tooltip={selected.includes(c.session_id) ? 'Take this chat out of the selection (Shift-click selects a range)' : 'Add this chat to the selection (Shift-click selects a range)'}
      >
        <span class="tickbox" aria-hidden="true">
          {#if selected.includes(c.session_id)}<Icon name="check" size={12} />{/if}
        </span>
        <span class="pickbody">
          <span class="title">{c.title}</span>
          <span class="rowmeta">
            <span>{relTime(c.last_at)} · {c.turn_count} turn{c.turn_count === 1 ? '' : 's'}</span>
          </span>
        </span>
      </button>
    {:else if editingId === c.session_id}
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
        onmousedown={noTextSelect}
        onclick={(e) =>
          isSelectClick(e) ? startSelectFrom(c.session_id, e.shiftKey) : onSelect(c.session_id)}
        type="button"
        use:tooltip={isActive(c.session_id) ? 'The chat you are in' : 'Open this chat (Ctrl-click or Shift-click starts selecting)'}
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
          use:tooltip={c.pinned ? 'Move this chat back among the others' : 'Keep this chat in the pinned section at the top'}
          aria-label={c.pinned ? 'Unpin conversation' : 'Pin conversation'}
          onclick={() => onPin(c.session_id, !c.pinned)}
          type="button"><Icon name="pin" size={14} /></button
        >
        <button
          class="act"
          use:tooltip={'Pin, rename, archive or delete this chat'}
          aria-label="Conversation options"
          aria-haspopup="menu"
          onclick={(e) => openMenu(c.session_id, e)}
          type="button"><Icon name="ellipsis" size={14} /></button
        >
      </div>
    {/if}
  </div>
{/snippet}

{#if selectMode}
  <!-- Action roles (ADR-063): the selection helper is quiet, Done ends the mode and so sits
       at the trailing end in the primary style, and Delete is red and confirms in the parent,
       naming how many. Two lines because the rail is too narrow for one: the count sits with
       the actions it counts for. -->
  <div class="selbar" role="toolbar" aria-label="Selection actions">
    <div class="selline">
      <button
        class="btn btn-quiet"
        disabled={visibleIds.length === 0}
        onclick={() => (selected = selectAllOrNone(visibleIds, selected))}
        type="button" use:tooltip={selected.length > 0 ? 'Empty the selection' : 'Select every chat in the list'}>{selectAllLabel(selected)}</button
      >
      <button class="btn btn-primary seldone" onclick={exitSelect} type="button" use:tooltip={'Leave select mode'}>Done</button>
    </div>
    <div class="selline">
      <span class="selcount">{selected.length} selected</span>
      <button
        class="btn"
        disabled={selected.length === 0}
        onclick={archiveSelected}
        type="button" use:tooltip={selected.length === 0 ? 'Select chats first' : selectedAllArchived ? `Bring the ${selected.length} selected back from the archive` : `Move the ${selected.length} selected to the archive; Show archived lists them`}>{selectedAllArchived ? 'Unarchive' : 'Archive'}</button
      >
      <button
        class="btn btn-danger"
        disabled={selected.length === 0}
        onclick={deleteSelected}
        type="button" use:tooltip={selected.length === 0 ? 'Select chats first' : `Delete the ${selected.length} selected conversation${selected.length === 1 ? '' : 's'}, after you confirm`}>Delete…</button
      >
    </div>
  </div>
{/if}
<nav class="list" aria-label="Conversation history" onscroll={closeMenu}>
  <!-- New chat as a tree-style row (like Library's "All documents" and Graph's "Taxonomy") —
       each rail opens with the same slim row idiom instead of a chunky filled button. -->
  {#if !selectMode}
    <button class="treerow" onclick={onNew} type="button" use:tooltip={'Start an empty conversation'}>
      <span class="treeicon"><Icon name="plus" size={14} /></span>
      <span class="treelabel">New chat</span>
    </button>
  {/if}
  {#if conversations.length === 0 && !conversationsLoaded}
    <p class="empty">Loading…</p>
  {:else if conversations.length === 0}
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
      <button class="archived-toggle" onclick={() => (showArchived = !showArchived)} type="button" use:tooltip={showArchived ? 'Take the archived chats out of the list' : 'List the chats you archived'}>
        {showArchived ? 'Hide' : 'Show'} archived ({archivedCount})
      </button>
    {/if}
  {/if}
</nav>

{#if openMenuFor && menuConvo}
  <div class="menu-backdrop" onclick={closeMenu} role="presentation"></div>
  <div class="menu" style="left: {menuPos.x}px; top: {menuPos.y}px" role="menu">
    <button class="menuitem" role="menuitem" onclick={menuPin} type="button" use:tooltip={menuConvo.pinned ? 'Move this chat back among the others' : 'Keep this chat in the pinned section at the top'}>
      <Icon name="pin" size={14} /> {menuConvo.pinned ? 'Unpin' : 'Pin'}
    </button>
    <button class="menuitem" role="menuitem" onclick={menuRename} type="button" use:tooltip={'Give this chat a new title'}>
      <Icon name="pencil" size={14} /> Rename
    </button>
    <button class="menuitem muted" role="menuitem" onclick={menuArchive} type="button" use:tooltip={menuConvo.archived ? 'Bring this chat back into the list' : 'Move this chat out of the list; Show archived finds it'}>
      <Icon name="archive" size={14} /> {menuConvo.archived ? 'Unarchive' : 'Archive'}
    </button>
    <div class="menusep"></div>
    <button class="menuitem danger" role="menuitem" onclick={menuDelete} type="button" use:tooltip={'Delete this chat for good, after you confirm'}>
      <Icon name="trash-2" size={14} /> Delete…
    </button>
  </div>
{/if}

<style>
  /* Search + sort toolbar — fixed header strip above the scrolling list. */
  .toolbar {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border);
  }
  /* A *filter* of the visible list, styled lighter than the header's global search (Ctrl/⌘-K) so
     the two read as different tools: transparent fill, smaller type, border only. */
  .search {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
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
    font-size: 0.78rem;
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
  /* --- select mode ------------------------------------------------------------------ */
  .convrow.picked {
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    border-color: var(--accent);
  }
  /* Two classes on purpose. This element is also `.rowmain`, which sets `flex-direction: column`
     for the normal row; at equal specificity the later rule won, so the tick stacked ON TOP of the
     title and centred, clipping long names (reported 2026-08-19). Raising specificity here makes
     the row layout independent of where either rule sits in the file. */
  .rowmain.pickrow {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 0.55rem;
    /* The rows a Shift-click ranges over must not also catch a text selection. */
    user-select: none;
  }
  .pickbody {
    display: flex;
    flex-direction: column;
    min-width: 0;
    flex: 1;
  }
  .tickbox {
    flex: none;
    width: 15px;
    height: 15px;
    border: 1px solid var(--border);
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--accent);
  }
  .convrow.picked .tickbox {
    border-color: var(--accent);
  }
  /* Sized to be read and hit: this bar is the only way out of a destructive mode, and at
     0.72rem/0.05rem padding it read as a caption rather than a toolbar (reported 2026-08-19). */
  .selbar {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding: 0.5rem 0.6rem;
    border-bottom: 1px solid var(--border);
  }
  .selline {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .selcount {
    margin-right: auto;
    font-size: 0.8rem;
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
  }
  .seldone {
    margin-left: auto;
  }
  /* The rail is ~240px inside: a little less padding is what fits three buttons and a count. */
  .selbar .btn {
    padding-inline: 0.5rem;
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
  /* The New chat row: the same slim tree-row idiom as the other rails' first entries. */
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
  .treeicon {
    color: var(--fg-2);
    display: inline-flex;
    flex: none;
  }
  .treelabel {
    flex: 1;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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
  /* Allow-listed (ui-rules.allow.ts): a menu row, not a button bar (the WAI-ARIA menu pattern keeps
     its rows' look). Red names the act; the confirm follows in App. */
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
</style>
