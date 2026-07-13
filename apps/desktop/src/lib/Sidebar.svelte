<script lang="ts">
  // Left-rail app shell (feature-conversation-history.md, Decisions 7, 8). Hosts the Chat/Library
  // switch (Library reserved but disabled), the "↻ New chat" action, and the conversation history
  // list. Persistent column on desktop; an off-canvas drawer under 720px.
  import type { ConversationSummary } from './types'

  let {
    conversations,
    liveSessionId,
    viewingSessionId,
    open = false,
    onNew,
    onSelect,
    onClose,
  }: {
    conversations: ConversationSummary[]
    liveSessionId: string
    viewingSessionId: string | null
    open?: boolean
    onNew: () => void
    onSelect: (sessionId: string) => void
    onClose?: () => void
  } = $props()

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
      <button class="mode active" role="tab" aria-selected="true" type="button">Chat</button>
      <button
        class="mode"
        role="tab"
        aria-selected="false"
        disabled
        type="button"
        title="Library — coming soon"
      >
        Library
      </button>
    </div>
    <button class="new" onclick={onNew} type="button">↻ New chat</button>
  </div>

  <nav class="list" aria-label="Conversation history">
    {#if conversations.length === 0}
      <p class="empty">No conversations yet. Ask a question to start one.</p>
    {:else}
      {#each conversations as c (c.session_id)}
        <button
          class="row"
          class:active={isActive(c.session_id)}
          aria-current={isActive(c.session_id) ? 'true' : undefined}
          onclick={() => onSelect(c.session_id)}
          type="button"
        >
          <span class="title">{c.title}</span>
          <span class="rowmeta">
            {#if c.session_id === liveSessionId}<span class="dot" title="Current chat">●</span>{/if}
            <span>{relTime(c.last_at)} · {c.turn_count} turn{c.turn_count === 1 ? '' : 's'}</span>
          </span>
        </button>
      {/each}
    {/if}
  </nav>
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
    text-align: left;
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
    text-align: left;
    font: inherit;
    cursor: pointer;
    border: 1px solid transparent;
    background: none;
    color: var(--fg);
    border-radius: 8px;
    padding: 0.45rem 0.55rem;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    width: 100%;
  }
  .row:hover {
    background: var(--surface);
  }
  .row.active {
    background: var(--surface-2);
    border-color: var(--border);
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
    color: var(--accent);
    font-size: 0.7em;
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
