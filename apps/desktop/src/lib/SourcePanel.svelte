<script lang="ts">
  import type { SourceView } from './types'
  import SourceCard from './SourceCard.svelte'
  import { fade, fly } from 'svelte/transition'

  // Same drawer mechanics as Settings.svelte (fly/scrim, reduced-motion collapse, focus trap,
  // Esc-to-close) — one interaction vocabulary for every slide-over in the app, not a second one.
  const animate =
    typeof window !== 'undefined' && window.matchMedia
      ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : true
  const DUR = animate ? 180 : 0

  let { source, onClose }: { source: SourceView; onClose: () => void } = $props()

  let panelEl = $state<HTMLDivElement | null>(null)
  let closeEl = $state<HTMLButtonElement | null>(null)

  // No input field here (unlike Settings), so land focus on the close control.
  $effect(() => {
    closeEl?.focus()
  })

  function onWindowKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }

  // Honour aria-modal: keep Tab inside the dialog instead of walking out into the UI the modal
  // has told assistive tech is inert. (Settings.svelte:117-136, copy-adapted.)
  function onPanelKey(e: KeyboardEvent): void {
    if (e.key !== 'Tab' || !panelEl) return
    const focusable = Array.from(
      panelEl.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    ) as HTMLElement[]
    if (focusable.length === 0) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault()
      first.focus()
    }
  }
</script>

<svelte:window onkeydown={onWindowKey} />

<div class="scrim" onclick={onClose} role="presentation" transition:fade={{ duration: DUR }}></div>

<div
  class="panel"
  role="dialog"
  aria-modal="true"
  aria-label={`Source ${source.n}`}
  tabindex="-1"
  bind:this={panelEl}
  onkeydown={onPanelKey}
  transition:fly={{ x: 420, opacity: 1, duration: DUR }}
>
  <header>
    <strong>Source [{source.n}]</strong>
    <button class="x" bind:this={closeEl} onclick={onClose} aria-label="Close">✕</button>
  </header>
  <SourceCard {source} />
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    z-index: 10;
  }
  .panel {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(420px, 92vw);
    z-index: 11;
    background: var(--bg);
    border-left: 1px solid var(--border);
    padding: 0 1.2rem 1.2rem;
    overflow-y: auto;
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.18);
  }
  header {
    position: sticky;
    top: 0;
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.8rem;
  }
  .x {
    font: inherit;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--fg-2);
    font-size: 1rem;
    padding: 0.2rem 0.4rem;
  }
</style>
