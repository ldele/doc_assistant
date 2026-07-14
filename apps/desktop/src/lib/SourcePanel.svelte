<script lang="ts">
  import type { SourceView } from './types'
  import SourceCard from './SourceCard.svelte'
  import Icon from './Icon.svelte'
  import { fade, fly } from 'svelte/transition'

  // Same drawer mechanics as Settings.svelte (fly/scrim, reduced-motion collapse, focus trap,
  // Esc-to-close) — one interaction vocabulary for every slide-over in the app, not a second one.
  const animate =
    typeof window !== 'undefined' && window.matchMedia
      ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : true
  const DUR = animate ? 180 : 0

  let { source, onClose }: { source: SourceView; onClose: () => void } = $props()

  // Resizable width (drag the left edge). Client-only view preference in localStorage, like the
  // sidebar width and the theme toggle. Clamped, and capped at 92vw so it can't cover the screen.
  const SOURCE_MIN = 320
  const SOURCE_MAX = 720
  function loadWidth(): number {
    try {
      const v = Number(localStorage.getItem('sourcePanelWidth'))
      return v >= SOURCE_MIN && v <= SOURCE_MAX ? v : 420
    } catch {
      return 420
    }
  }
  let width = $state(loadWidth())
  function startResize(e: PointerEvent): void {
    e.preventDefault()
    const cap = Math.min(SOURCE_MAX, window.innerWidth * 0.92)
    const onMove = (ev: PointerEvent) => {
      width = Math.min(cap, Math.max(SOURCE_MIN, window.innerWidth - ev.clientX))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      try {
        localStorage.setItem('sourcePanelWidth', String(Math.round(width)))
      } catch {
        /* ignore — width just won't persist */
      }
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

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
  class="resize-handle"
  style="right: {width}px"
  role="separator"
  aria-orientation="vertical"
  aria-label="Resize source panel"
  onpointerdown={startResize}
></div>

<div
  class="panel"
  role="dialog"
  aria-modal="true"
  aria-label={`Source ${source.n}`}
  tabindex="-1"
  style="width: {width}px"
  bind:this={panelEl}
  onkeydown={onPanelKey}
  transition:fly={{ x: width, opacity: 1, duration: DUR }}
>
  <header>
    <strong>Source [{source.n}]</strong>
    <button class="x" bind:this={closeEl} onclick={onClose} aria-label="Close">
      <Icon name="x" />
    </button>
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
    max-width: 92vw;
    z-index: 11;
    background: var(--bg);
    border-left: 1px solid var(--border);
    padding: 0 1.2rem 1.2rem;
    overflow-y: auto;
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.18);
  }
  /* Drag the left edge to resize; sits just outside the panel's overflow so scrolling won't clip it. */
  .resize-handle {
    position: fixed;
    top: 0;
    bottom: 0;
    width: 6px;
    margin-right: -3px;
    cursor: col-resize;
    z-index: 12;
    background: transparent;
    transition: background 0.15s ease;
  }
  .resize-handle:hover,
  .resize-handle:active {
    background: var(--accent);
    opacity: 0.5;
  }
  @media (max-width: 720px) {
    .resize-handle {
      display: none;
    }
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
    padding: 0.2rem 0.4rem;
    display: inline-flex;
    align-items: center;
  }
</style>
