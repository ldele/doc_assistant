<script lang="ts">
  // Keyboard-shortcuts help, opened from the toolbar app menu (☰). Read-only reference; the scrim +
  // centered card + Esc-to-close shell matches ConfirmDialog / the other hand-rolled modals.
  import Icon from './Icon.svelte'

  let { onClose }: { onClose: () => void } = $props()

  // The modifier label reads ⌘ on macOS, Ctrl elsewhere — cosmetic only (both bindings are live).
  const mod =
    typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl'

  const rows: { keys: string[]; label: string }[] = [
    { keys: [mod, 'K'], label: 'Open search (chats + documents)' },
    { keys: ['Enter'], label: 'Send message (in the chat composer)' },
    { keys: ['Shift', 'Enter'], label: 'New line in the composer' },
    { keys: ['Esc'], label: 'Close a dialog, menu, or the search overlay' },
  ]

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
  <div class="mhead">
    <h2><Icon name="keyboard" size={18} /> Keyboard shortcuts</h2>
    <button class="iconbtn" onclick={onClose} aria-label="Close" type="button">
      <Icon name="x" size={16} />
    </button>
  </div>
  <ul class="rows">
    {#each rows as r (r.label)}
      <li>
        <span class="keys">
          {#each r.keys as k (k)}<kbd>{k}</kbd>{/each}
        </span>
        <span class="label">{r.label}</span>
      </li>
    {/each}
  </ul>
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
    width: min(92vw, 440px);
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
  h2 {
    margin: 0;
    font-size: var(--text-title);
    font-family: var(--font-serif);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .iconbtn {
    display: inline-flex;
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
  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .rows li {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: 0.4rem 0.2rem;
  }
  .keys {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    min-width: 6.5rem;
  }
  kbd {
    font: inherit;
    font-size: 0.72rem;
    line-height: 1;
    padding: 0.2rem 0.4rem;
    border: 1px solid var(--border);
    border-bottom-width: 2px;
    border-radius: 5px;
    background: var(--surface-2);
    color: var(--fg);
  }
  .label {
    font-size: 0.85rem;
    color: var(--fg-2);
  }
</style>
