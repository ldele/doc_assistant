<script lang="ts">
  import type { SourceView } from '../../lib/core/types'
  import SourceCard from './SourceCard.svelte'
  import Icon from '../../shell/Icon.svelte'
  import Dialog from '../../shell/Dialog.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  // A side panel on the shared frame (shell/Dialog.svelte), like Settings: the slide, the backdrop,
  // Esc and the Tab trap are the frame's; the resize handle is this panel's own.

  let {
    source,
    onClose,
    onOpenInDocument = null,
  }: {
    source: SourceView
    onClose: () => void
    /** Open this citation's page in the Library (ROADMAP 18). Threaded from App, which owns
     *  cross-space navigation; null when the shell cannot offer it. */
    onOpenInDocument?: ((chunkKey: string) => void) | null
  } = $props()

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

  let closeEl = $state<HTMLButtonElement | null>(null)

  // No input field here (unlike Settings), so land focus on the close control.
  $effect(() => {
    closeEl?.focus()
  })
</script>

<Dialog shape="side" label={`Source ${source.n}`} width="min({width}px, 92vw)" slide={width} scroll {onClose}>
  <header>
    <strong>Source [{source.n}]</strong>
    <button class="x" bind:this={closeEl} onclick={onClose} aria-label="Close" use:tooltip={'Close the source panel'}>
      <Icon name="x" />
    </button>
  </header>
  <SourceCard {source} {onOpenInDocument} />
  {#snippet edge()}
    <div
      class="resize-handle"
      style="right: {width}px"
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize source panel"
      onpointerdown={startResize}
    ></div>
  {/snippet}
</Dialog>

<style>
  /* Drag the left edge to resize; sits just outside the panel's overflow so scrolling won't clip it. */
  .resize-handle {
    position: fixed;
    top: 0;
    bottom: 0;
    width: 6px;
    margin-right: -3px;
    cursor: col-resize;
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
