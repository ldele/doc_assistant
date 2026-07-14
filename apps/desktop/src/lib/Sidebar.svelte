<script lang="ts">
  // Left-rail app shell (feature-conversation-history.md, Decisions 7, 8; Library tab enabled by
  // feature-library-browser.md, L1). Hosts the Chat/Library switch, the "↻ New chat" action, and
  // — depending on `mode` — the conversation history list (chat) or the document list (library).
  // Persistent column on desktop; an off-canvas drawer under 720px.
  import type { ConversationSummary, LibraryDocument } from './types'
  import Icon from './Icon.svelte'

  let {
    mode,
    conversations,
    documents,
    liveSessionId,
    viewingSessionId,
    selectedDocId,
    open = false,
    onNew,
    onSelect,
    onSelectMode,
    onSelectDocument,
    onClose,
    onPin,
    onArchive,
    onDelete,
  }: {
    mode: 'chat' | 'library'
    conversations: ConversationSummary[]
    documents: LibraryDocument[]
    liveSessionId: string
    viewingSessionId: string | null
    selectedDocId: string | null
    open?: boolean
    onNew: () => void
    onSelect: (sessionId: string) => void
    onSelectMode: (mode: 'chat' | 'library') => void
    onSelectDocument: (docId: string) => void
    onClose?: () => void
    onPin: (sessionId: string, pinned: boolean) => void
    onArchive: (sessionId: string, archived: boolean) => void
    onDelete: (sessionId: string) => void
  } = $props()

  // Archived conversations are hidden behind a toggle so they don't clutter the list.
  let showArchived = $state(false)
  const archivedCount = $derived(conversations.filter((c) => c.archived).length)
  const visibleConvos = $derived(
    showArchived ? conversations : conversations.filter((c) => !c.archived),
  )

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

  // Library row label: prefer "Title — First Author" over the raw filename (which stays as the
  // row tooltip). The registry stores `authors` as one string with no locked format yet — split on
  // the common separators and take the first name; "et al." when more are listed. Tighten once the
  // metadata-enrichment sidecar defines the canonical format.
  function docLabel(d: LibraryDocument): string {
    if (!d.title) return d.filename
    if (!d.authors) return d.title
    const names = d.authors.split(/\s*(?:;|,| and )\s*/).filter(Boolean)
    const first = names[0] ?? d.authors
    return `${d.title} · ${names.length > 1 ? `${first} et al.` : first}`
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
        Chat
      </button>
      <button
        class="mode"
        class:active={mode === 'library'}
        role="tab"
        aria-selected={mode === 'library'}
        type="button"
        onclick={() => onSelectMode('library')}
      >
        Library
      </button>
    </div>
    {#if mode === 'chat'}
      <button class="new" onclick={onNew} type="button"><Icon name="rotate-ccw" size={15} /> New chat</button>
    {/if}
  </div>

  {#if mode === 'chat'}
    <nav class="list" aria-label="Conversation history">
      {#if conversations.length === 0}
        <p class="empty">No conversations yet. Ask a question to start one.</p>
      {:else}
        {#each visibleConvos as c (c.session_id)}
          <div class="row" class:active={isActive(c.session_id)} class:archived={c.archived}>
            <button
              class="rowmain"
              aria-current={isActive(c.session_id) ? 'true' : undefined}
              onclick={() => onSelect(c.session_id)}
              type="button"
            >
              <span class="title">
                {#if c.pinned}<span class="pinmark" aria-hidden="true"><Icon name="pin" size={11} /></span>{/if}{c.title}
              </span>
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
                class:on={c.archived}
                title={c.archived ? 'Unarchive' : 'Archive'}
                aria-label={c.archived ? 'Unarchive conversation' : 'Archive conversation'}
                onclick={() => onArchive(c.session_id, !c.archived)}
                type="button"><Icon name="archive" size={14} /></button
              >
              <button
                class="act danger"
                title="Delete"
                aria-label="Delete conversation"
                onclick={() => onDelete(c.session_id)}
                type="button"><Icon name="trash-2" size={14} /></button
              >
            </div>
          </div>
        {/each}
        {#if archivedCount > 0}
          <button class="archived-toggle" onclick={() => (showArchived = !showArchived)} type="button">
            {showArchived ? 'Hide' : 'Show'} archived ({archivedCount})
          </button>
        {/if}
      {/if}
    </nav>
  {:else}
    <nav class="list" aria-label="Documents">
      {#if documents.length === 0}
        <p class="empty">No documents indexed yet.</p>
      {:else}
        {#each documents as d (d.id)}
          <button
            class="row"
            class:active={d.id === selectedDocId}
            aria-current={d.id === selectedDocId ? 'true' : undefined}
            onclick={() => onSelectDocument(d.id)}
            type="button"
            title={d.title ? d.filename : undefined}
          >
            <span class="title">{docLabel(d)}</span>
            <span class="rowmeta">
              <span>{d.format}{#if d.chunk_count != null} · {d.chunk_count.toLocaleString()} chunks{/if}</span>
            </span>
          </button>
        {/each}
      {/if}
    </nav>
  {/if}
</aside>

{#if open}
  <div class="scrim" onclick={onClose} role="presentation"></div>
{/if}

<style>
  .sidebar {
    width: 260px;
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
  }
  .mode.active {
    background: var(--surface-2);
    color: var(--fg);
    font-weight: 600;
  }
  .mode:disabled {
    opacity: 0.5;
    cursor: default;
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
    gap: 0.4rem;
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
  .row {
    display: flex;
    align-items: center;
    border: 1px solid transparent;
    border-radius: 8px;
    width: 100%;
  }
  .row:hover {
    background: var(--surface);
  }
  .row.active {
    background: var(--surface-2);
    border-color: var(--border);
  }
  .row.archived {
    opacity: 0.55;
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
  .pinmark {
    color: var(--accent);
    margin-right: 0.2rem;
  }
  .rowactions {
    display: none;
    align-items: center;
    gap: 0.05rem;
    padding-right: 0.25rem;
    flex: none;
  }
  .row:hover .rowactions,
  .row:focus-within .rowactions {
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
  .act.danger:hover {
    color: var(--warn-fg);
    background: var(--warn-bg);
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
  .scrim {
    display: none;
  }

  @media (max-width: 720px) {
    .sidebar {
      position: fixed;
      top: 0;
      left: 0;
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
