<script lang="ts">
  // Full-size figure viewer (user request 2026-08-10). The panel's cards cap images at 180 px,
  // which is enough to recognise a figure and not enough to *read* one — a plotted axis or a
  // circuit diagram needs the pixels.
  //
  // Two levels, because "fit the window" and "zoom in" are different needs: the image opens
  // contained in the viewport, and one click switches to its natural size inside a scrollable
  // frame, which is the zoom. No pan/zoom library — the browser's own scrolling does it.
  //
  // Scrim + centered card + Esc-to-close, matching the other hand-rolled modals (AboutDialog).
  // ← / → step through the document's figures so a reader can walk the plates without closing
  // and re-opening the viewer each time.
  import type { LibraryFigure } from '../core/types'
  import { API_BASE } from '../core/api/_base'
  import Icon from '../shell/Icon.svelte'

  let {
    figures,
    index,
    onIndex,
    onClose,
  }: {
    /** Only figures that actually have a rendered image — the caller filters. */
    figures: LibraryFigure[]
    index: number
    onIndex: (i: number) => void
    onClose: () => void
  } = $props()

  let actualSize = $state(false)

  const fig = $derived(figures[index])

  // A new figure starts fitted: carrying the previous one's zoom over would drop the reader
  // into the middle of an image they have not seen whole.
  function step(delta: number): void {
    const next = index + delta
    if (next < 0 || next >= figures.length) return
    actualSize = false
    onIndex(next)
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
    else if (e.key === 'ArrowLeft') step(-1)
    else if (e.key === 'ArrowRight') step(1)
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="viewer" role="dialog" aria-modal="true" aria-label="Figure viewer">
  {#if fig}
    <div class="vhead">
      <span class="pos">{index + 1} of {figures.length}</span>
      <span class="meta">
        page {fig.page}
        {#if fig.kind}· {fig.kind}{/if}
        {#if fig.retrievable}
          <span class="badge ok" title="This figure is searchable — the assistant can find and cite it">searchable</span>
        {:else}
          <span class="badge off" title={fig.not_retrievable_reason ?? ''}>not searchable</span>
        {/if}
      </span>
      <button
        class="zoombtn"
        type="button"
        onclick={() => (actualSize = !actualSize)}
        aria-pressed={actualSize}
      >
        {actualSize ? 'Fit to window' : 'Actual size'}
      </button>
      <button class="iconbtn" onclick={onClose} aria-label="Close" type="button">
        <Icon name="x" size={16} />
      </button>
    </div>

    <div class="stage" class:zoomed={actualSize}>
      <!-- A button, not a click handler on the image: clicking to zoom has to be reachable
           from the keyboard too, and Space/Enter come free. -->
      <button
        class="imgbtn"
        type="button"
        aria-label={actualSize ? 'Fit the figure to the window' : 'Zoom the figure to actual size'}
        aria-pressed={actualSize}
        onclick={() => (actualSize = !actualSize)}
      >
        <img
          src={`${API_BASE}/api/figures/${encodeURIComponent(fig.id)}`}
          alt={fig.caption ?? `Figure on page ${fig.page}`}
          class:zoomed={actualSize}
        />
      </button>
    </div>

    {#if fig.caption}<p class="caption">{fig.caption}</p>{/if}

    {#if figures.length > 1}
      <div class="nav">
        <button type="button" onclick={() => step(-1)} disabled={index === 0}>‹ Previous</button>
        <button type="button" onclick={() => step(1)} disabled={index === figures.length - 1}>
          Next ›
        </button>
      </div>
    {/if}
  {/if}
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--fg) 55%, transparent);
    z-index: 40;
  }
  .viewer {
    position: fixed;
    z-index: 41;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(94vw, 1100px);
    max-height: 92vh;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-height: 0;
  }
  .vhead {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.75rem;
    color: var(--fg-2);
  }
  .pos {
    font-weight: 600;
  }
  .meta {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-right: auto;
  }
  .badge {
    padding: 0.05rem 0.35rem;
    border-radius: 3px;
    font-size: 0.68rem;
    white-space: nowrap;
  }
  .badge.ok {
    background: color-mix(in srgb, green 25%, transparent);
  }
  .badge.off {
    background: color-mix(in srgb, gray 25%, transparent);
  }
  .zoombtn {
    background: none;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    font: inherit;
    font-size: 0.72rem;
    color: var(--fg-2);
    cursor: pointer;
  }
  .zoombtn:hover {
    color: var(--fg);
    border-color: var(--accent);
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
  }
  /* Fitted: the image is bounded by the frame. Zoomed: the frame scrolls and the image is
     whatever size it actually is — that is the zoom, done by the browser. */
  .stage {
    flex: 1;
    min-height: 0;
    display: grid;
    place-items: center;
    background: #fff;
    border-radius: 6px;
    overflow: hidden;
  }
  .stage.zoomed {
    place-items: start;
    overflow: auto;
  }
  .imgbtn {
    display: block;
    padding: 0;
    border: none;
    background: none;
    cursor: zoom-in;
    max-width: 100%;
  }
  .stage.zoomed .imgbtn {
    cursor: zoom-out;
    max-width: none;
  }
  img {
    max-width: 100%;
    max-height: 72vh;
    object-fit: contain;
    display: block;
  }
  img.zoomed {
    max-width: none;
    max-height: none;
  }
  .caption {
    margin: 0;
    font-size: 0.8rem;
    line-height: 1.45;
    color: var(--fg);
    max-height: 6.5rem;
    overflow-y: auto;
    flex: none;
  }
  .nav {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    flex: none;
  }
  .nav button {
    background: none;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.2rem 0.7rem;
    font: inherit;
    font-size: 0.78rem;
    color: var(--fg);
    cursor: pointer;
  }
  .nav button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .nav button:not(:disabled):hover {
    border-color: var(--accent);
  }
</style>
