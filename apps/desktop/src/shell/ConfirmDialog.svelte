<script lang="ts">
  // Generic in-app confirmation dialog — the standard replacement for the native window.confirm()
  // (which renders an OS "localhost:1420 says" chrome that breaks the app's look). A card on the
  // shared frame (Dialog: Esc, a click on the backdrop and Cancel back out); the confirm action
  // carries the danger (red) styling by default since these gate destructive acts.
  import Dialog from './Dialog.svelte'
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
</script>

<!-- The WAI-ARIA alert dialog pattern (ADR-063 choice 7): it interrupts to confirm an act, its title
     names it and its body describes it, and focus starts on Cancel, the act that changes nothing.
     The confirm is the dialog's one action (`btn-primary`); when it removes something it is also
     `btn-danger`, which app.css fills red. -->
<Dialog
  role={tone === 'danger' ? 'alertdialog' : 'dialog'}
  labelledby="{uid}-title"
  describedby="{uid}-body"
  width="28rem"
  {onClose}
>
  <h2 class="dialog-title" id="{uid}-title">{title}</h2>
  <p class="body" id="{uid}-body">{body}</p>
  <div class="dialog-actions">
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
</Dialog>

<style>
  .body {
    margin: 0;
    font-size: 0.85rem;
    color: var(--fg-2);
    line-height: 1.6;
  }
</style>
