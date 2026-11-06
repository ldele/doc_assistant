<script lang="ts">
  // Generic in-app confirmation dialog — the standard replacement for the native window.confirm()
  // (which renders an OS "localhost:1420 says" chrome that breaks the app's look). Modeled on
  // DeleteDocument's house style: scrim + centered card, Esc / scrim-click / Cancel back out,
  // the confirm action carries the danger (red) styling by default since these gate destructive acts.
  import Icon from './Icon.svelte'
  import { tooltip } from '../lib/shell/tooltip'

  let {
    title,
    body,
    confirmLabel = 'Delete',
    cancelLabel = 'Cancel',
    tone = 'danger',
    confirmIcon = 'trash-2',
    busy = false,
    onConfirm,
    onClose,
  }: {
    title: string
    body: string
    confirmLabel?: string
    cancelLabel?: string
    tone?: 'danger' | 'default'
    /** The danger button's icon: a bin for a delete, a square for a stop; null for none. */
    confirmIcon?: 'trash-2' | 'square' | null
    busy?: boolean
    onConfirm: () => void
    onClose: () => void
  } = $props()

  const uid = $props.id()
  let cancelEl: HTMLButtonElement | undefined = $state()
  $effect(() => {
    cancelEl?.focus()
  })

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
</script>

<!-- The WAI-ARIA alert dialog pattern (ADR-063 choice 7): it interrupts to confirm an act, its title
     names it and its body describes it, and focus starts on Cancel, the act that changes nothing.
     The confirm is the dialog's one action (`btn-primary`); when it removes something it is also
     `btn-danger`, which app.css fills red. -->
<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div
  class="modal"
  role={tone === 'danger' ? 'alertdialog' : 'dialog'}
  aria-modal="true"
  aria-labelledby="{uid}-title"
  aria-describedby="{uid}-body"
>
  <h2 id="{uid}-title">{title}</h2>
  <p class="body" id="{uid}-body">{body}</p>
  <div class="mactions">
    <button class="btn" bind:this={cancelEl} onclick={onClose} type="button" disabled={busy} use:tooltip={'Close and change nothing'}>{cancelLabel}</button>
    <button
      class="btn btn-primary"
      class:btn-danger={tone === 'danger'}
      onclick={onConfirm}
      type="button"
      disabled={busy}
      use:tooltip={tone === 'danger' ? 'Go ahead with the removal described above' : 'Go ahead as described above'}
    >
      {#if tone === 'danger' && confirmIcon}<Icon name={confirmIcon} size={14} />{/if}
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
    width: min(92vw, 28rem);
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
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
</style>
