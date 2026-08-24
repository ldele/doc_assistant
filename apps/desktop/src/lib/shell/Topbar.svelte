<script lang="ts">
  // Unified top toolbar (browser-chrome shell): one bar across the whole window carrying the app
  // menu, sidebar toggle, back/forward, brand, the mode tabs, and search/settings — the pattern
  // replaces the old split of mode-pills-in-sidebar + actions-in-header.
  //
  // Reads shell chrome state (`shell`, `sidebarPrefs`) straight from its leaf rune modules rather
  // than through props — that is the point of step 5 phase 1. Props are only the things App
  // genuinely owns: the nav-history cursor and the callbacks that orchestrate other domains.
  import Icon from './Icon.svelte'
  import { shell, type Mode } from './shell.svelte'
  import { sidebarPrefs, toggleSidebarCollapsed } from './prefs.svelte'
  import { GRAPH_TAB_ENABLED } from '../core/features'
  import { canAccept, pickDocuments, unavailableReason } from '../library/accept.svelte'
  import appMark from '../../assets/brand/app-mark.png'

  interface Props {
    canBack: boolean
    canForward: boolean
    onNavBack: () => void
    onNavForward: () => void
    /** App owns mode switching — it lazy-loads each mode's data (four domains). */
    onSelectMode: (m: Mode) => void
    onOpenSearch: () => void
    /** Export is chat-only and needs live turn state, so App decides when it is available. */
    exportDisabled: boolean
    onExport: () => void
  }
  const {
    canBack,
    canForward,
    onNavBack,
    onNavForward,
    onSelectMode,
    onOpenSearch,
    exportDisabled,
    onExport,
  }: Props = $props()
</script>

<div class="topbar">
  <div class="tb-cluster">
    <div class="menuwrap">
      <button
        class="tb-btn"
        class:on={shell.appMenuOpen}
        onclick={() => (shell.appMenuOpen = !shell.appMenuOpen)}
        aria-label="Menu"
        aria-haspopup="menu"
        aria-expanded={shell.appMenuOpen}
        title="Menu"
        type="button"><Icon name="menu" size={17} /></button
      >
      {#if shell.appMenuOpen}
        <div class="menu-backdrop" onclick={() => (shell.appMenuOpen = false)} role="presentation"></div>
        <div class="appmenu" role="menu">
          <!-- The add-documents action lives in the Library header row (2026-08-24), where it sits
               with the other document actions. This entry is what keeps it reachable from Chat —
               without it, moving the button out of the toolbar would cost that reach entirely. -->
          <button
            class="appmenuitem"
            role="menuitem"
            onclick={() => { shell.appMenuOpen = false; void pickDocuments() }}
            disabled={!canAccept()}
            title={unavailableReason() ?? 'Add documents'}
            type="button"
          >
            <Icon name="plus" size={15} /> Add documents…
          </button>
          <button class="appmenuitem" role="menuitem" onclick={() => { shell.appMenuOpen = false; shell.showSettings = true }} type="button">
            <Icon name="settings" size={15} /> Settings
          </button>
          <button class="appmenuitem" role="menuitem" onclick={() => { shell.appMenuOpen = false; shell.showShortcuts = true }} type="button">
            <Icon name="keyboard" size={15} /> Keyboard shortcuts
          </button>
          {#if shell.mode === 'chat'}
            <button
              class="appmenuitem"
              role="menuitem"
              disabled={exportDisabled}
              onclick={() => { shell.appMenuOpen = false; onExport() }}
              type="button"
            >
              <Icon name="download" size={15} /> Export transcript
            </button>
          {/if}
          <div class="appmenusep"></div>
          <button class="appmenuitem" role="menuitem" onclick={() => { shell.appMenuOpen = false; shell.showAbout = true }} type="button">
            <Icon name="info" size={15} /> About Provenote
          </button>
        </div>
      {/if}
    </div>
    <!-- Sidebar toggle: desktop collapses inline, mobile opens the off-canvas drawer. -->
    <button
      class="tb-btn hide-mobile"
      onclick={toggleSidebarCollapsed}
      aria-label={sidebarPrefs.collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      aria-pressed={sidebarPrefs.collapsed}
      title={sidebarPrefs.collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      type="button"><Icon name="panel-left" size={16} /></button
    >
    <button
      class="tb-btn only-mobile"
      onclick={() => (shell.sidebarOpen = true)}
      aria-label="Open sidebar"
      title="Open sidebar"
      type="button"><Icon name="panel-left" size={16} /></button
    >
    <!-- Search sits with the sidebar toggle, ahead of back/forward: it opens a navigation overlay,
         so it belongs with the other navigation affordances rather than with Settings. -->
    <button class="tb-btn" onclick={onOpenSearch} aria-label="Search chats and documents" title="Search  (Ctrl/⌘ K)" type="button">
      <Icon name="search" size={16} />
    </button>
    <button
      class="tb-btn"
      onclick={onNavBack}
      disabled={!canBack}
      aria-label="Back"
      title="Back"
      type="button"><Icon name="arrow-left" size={16} /></button
    >
    <button
      class="tb-btn"
      onclick={onNavForward}
      disabled={!canForward}
      aria-label="Forward"
      title="Forward"
      type="button"><Icon name="arrow-right" size={16} /></button
    >
  </div>

  <!-- Mode tabs (Chat/Library/Graph) — moved out of the sidebar into the toolbar. -->
  <div class="tb-modes" role="tablist" aria-label="Workspace">
    <button
      class="tb-mode"
      class:active={shell.mode === 'chat'}
      role="tab"
      aria-selected={shell.mode === 'chat'}
      onclick={() => onSelectMode('chat')}
      type="button"><Icon name="message-square" size={15} /><span class="tb-modelabel">Chat</span></button
    >
    <button
      class="tb-mode"
      class:active={shell.mode === 'library'}
      role="tab"
      aria-selected={shell.mode === 'library'}
      onclick={() => onSelectMode('library')}
      type="button"><Icon name="library" size={15} /><span class="tb-modelabel">Library</span></button
    >
    <!-- Hidden for 0.6 — see core/features.ts GRAPH_TAB_ENABLED for why, and to bring it back. -->
    {#if GRAPH_TAB_ENABLED}
      <button
        class="tb-mode"
        class:active={shell.mode === 'graph'}
        role="tab"
        aria-selected={shell.mode === 'graph'}
        onclick={() => onSelectMode('graph')}
        type="button"><Icon name="waypoints" size={15} /><span class="tb-modelabel">Graph</span></button
      >
    {/if}
  </div>

  <div class="tb-spacer"></div>

  <div class="tb-cluster">
    <!-- Brand = identity anchor only (small mark + wordmark), parked on the right beside Settings.
         The corpus/model status lives in the bottom status bar — ambient, not navigation. -->
    <div class="brand">
      <span class="mark"><img src={appMark} alt="" width="26" height="26" /></span>
      <div class="brandtext">
        <span class="wordmark">proven<span class="wm-accent">ote</span></span>
      </div>
    </div>
    <button class="tb-btn" onclick={() => (shell.showSettings = true)} aria-label="Settings" title="Settings" type="button">
      <Icon name="settings" size={17} />
    </button>
  </div>
</div>

<style>
  /* ---- top toolbar (browser-chrome shell) ---- */
  .topbar {
    flex: none;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: 0.45rem 0.7rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .tb-cluster {
    display: flex;
    align-items: center;
    gap: 0.12rem;
    flex: none;
  }
  .tb-spacer {
    flex: 1;
    min-width: var(--space-2);
  }
  .tb-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    cursor: pointer;
    border: 1px solid transparent;
    background: none;
    color: var(--fg-2);
    border-radius: 8px;
    padding: 0.32rem;
  }
  .tb-btn:hover:not(:disabled),
  .tb-btn.on {
    color: var(--fg);
    background: var(--surface-2);
  }
  .tb-btn:disabled {
    opacity: 0.32;
    cursor: default;
  }
  .menuwrap {
    position: relative;
    display: inline-flex;
  }
  .only-mobile {
    display: none;
  }
  @media (max-width: 720px) {
    .hide-mobile {
      display: none;
    }
    .only-mobile {
      display: inline-flex;
    }
  }
  /* App menu (☰) dropdown — mirrors the library sort menu idiom. */
  .menu-backdrop {
    position: fixed;
    inset: 0;
    z-index: 30;
  }
  .appmenu {
    position: absolute;
    z-index: 31;
    top: calc(100% + 6px);
    left: 0;
    min-width: 212px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow-2);
    padding: 0.3rem;
    display: flex;
    flex-direction: column;
    gap: 0.05rem;
  }
  .appmenuitem {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.45rem 0.55rem;
    border: none;
    background: none;
    color: var(--fg);
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 0.85rem;
    text-align: left;
  }
  .appmenuitem:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .appmenuitem:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .appmenusep {
    height: 1px;
    background: var(--border);
    margin: 0.25rem 0.3rem;
  }
  /* Mode tabs — segmented control in the toolbar. */
  .tb-modes {
    display: flex;
    align-items: center;
    gap: 0.2rem;
    flex: none;
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 2px;
    background: var(--surface);
  }
  .tb-mode {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font: inherit;
    font-size: 0.82rem;
    cursor: pointer;
    border: none;
    border-radius: 7px;
    padding: 0.28rem 0.6rem;
    background: none;
    color: var(--fg-2);
  }
  .tb-mode:hover {
    color: var(--fg);
  }
  .tb-mode.active {
    background: var(--bg);
    color: var(--fg);
    font-weight: 600;
    box-shadow: var(--shadow-1);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex: none;
    min-width: 0;
  }
  .mark {
    flex: none;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    overflow: hidden;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--shadow-1);
  }
  .mark img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .brandtext {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .wordmark {
    font-family: var(--font-serif);
    font-size: 1.1rem;
    line-height: 1.15;
    color: var(--fg);
    white-space: nowrap;
  }
  .wm-accent {
    color: var(--accent-wordmark);
  }
  /* Toolbar crowding: drop the tab labels (icon-only) then the wordmark, keeping the mark. */
  @media (max-width: 780px) {
    .tb-modelabel {
      display: none;
    }
    .tb-mode {
      padding: 0.28rem 0.45rem;
    }
    .brandtext {
      display: none;
    }
  }
</style>
