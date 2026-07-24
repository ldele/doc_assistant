<script lang="ts">
  // Generic in-app confirmation dialog — the standard replacement for the native window.confirm()
  // (which renders an OS "localhost:1420 says" chrome that breaks the app's look). Modeled on
  // LibraryDeleteConfirm's house style: scrim + centered card, Esc / scrim-click / Cancel back out,
  // the confirm action carries the danger (red) styling by default since these gate destructive acts.
  import Icon from './Icon.svelte'

  let {
    title,
    body,
    confirmLabel = 'Delete',
    cancelLabel = 'Cancel',
    tone = 'danger',
    busy = false,
    onConfirm,
    onClose,
  }: {
    title: string
    body: string
    confirmLabel?: string
    cancelLabel?: string
    tone?: 'danger' | 'default'
    busy?: boolean
    onConfirm: () => void
    onClose: () => void
  } = $props()

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label={title}>
  <h2>{title}</h2>
  <p class="body">{body}</p>
  <div class="mactions">
    <button class="ghost" onclick={onClose} type="button" disabled={busy}>{cancelLabel}</button>
    <button class={tone} onclick={onConfirm} type="button" disabled={busy}>
      {#if tone === 'danger'}<Icon name="trash-2" size={14} />{/if}
      {busy ? 'Working…' : confirmLabel}
    </button>
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
    width: min(92vw, 420px);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  h2 {
    margin: 0;
    font-size: var(--text-title);
    font-family: var(--font-serif);
  }
  .body {
    margin: 0;
    font-size: 0.85rem;
    color: var(--fg-2);
    line-height: 1.6;
  }
  .mactions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: var(--space-2);
  }
  .mactions button {
    font: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    border: 1px solid var(--border);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .mactions button:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .ghost {
    background: none;
    color: var(--fg);
  }
  .ghost:hover {
    background: var(--surface-2);
  }
  .default {
    background: var(--surface-2);
    color: var(--fg);
    font-weight: 600;
  }
  .default:hover:not(:disabled) {
    background: color-mix(in srgb, var(--fg) 8%, var(--surface-2));
  }
  .danger {
    background: color-mix(in srgb, var(--danger) 14%, transparent);
    color: var(--danger);
    border-color: var(--danger);
    font-weight: 600;
  }
  .danger:hover:not(:disabled) {
    background: color-mix(in srgb, var(--danger) 22%, transparent);
  }
</style>
