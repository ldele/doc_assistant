<script lang="ts">
  // The one dialog frame (ADR-063 choice 8): the backdrop, the frame, *Esc*, where focus starts,
  // *Tab* kept inside, and focus handed back on close. Every dialog in the app is one of these;
  // what it shows is its own (`children`). Until 2026-11-06 twenty components drew all of this
  // for themselves, with four looks of backdrop and an *Esc* guard per question they could open.
  //
  // Three shapes: a `card` centred on the screen (most dialogs), a `side` panel along the right
  // edge (Settings, a source), and a `screen` that takes most of the window (Indexing). The list
  // of open dialogs (lib/shell/dialogs.ts) decides which one *Esc* closes and puts a dialog
  // opened from another above it.
  import type { Snippet } from 'svelte'
  import { onDestroy, tick } from 'svelte'
  import { fade, fly } from 'svelte/transition'
  import {
    FOCUSABLE,
    closeDialog,
    frontFrame,
    isFront,
    layerOf,
    mayRestoreTo,
    openDialog,
    setFrame,
    wrapTab,
  } from '../lib/shell/dialogs'

  let {
    shape = 'card',
    role = 'dialog',
    label,
    labelledby,
    describedby,
    width = '31rem',
    height,
    top = false,
    padding,
    gap,
    scroll = false,
    slide = 760,
    locked = false,
    onClose,
    onEscape,
    frame = $bindable(null),
    children,
    edge,
  }: {
    shape?: 'card' | 'side' | 'screen'
    /** `alertdialog` for a question that interrupts before an act (WAI-ARIA alert dialog). */
    role?: 'dialog' | 'alertdialog'
    label?: string
    labelledby?: string
    describedby?: string
    /** A card's widest, a side panel's width, a screen's widest (any CSS length). */
    width?: string
    /** A card's tallest (default 90vh); a screen's height. */
    height?: string
    /** A card near the top of the window rather than centred (a search). */
    top?: boolean
    padding?: string
    gap?: string
    /** A side panel that scrolls as a whole, rather than a part of it. */
    scroll?: boolean
    /** How far a side panel slides in from, in pixels. */
    slide?: number
    /** While true (a run in progress), *Esc* and a click on the backdrop do nothing. */
    locked?: boolean
    onClose: () => void
    /** What *Esc* does when it is not just closing: answering an open question first. */
    onEscape?: () => void
    frame?: HTMLDivElement | null
    children: Snippet
    /** Drawn above the frame, outside its scrolling: a side panel's resize handle. */
    edge?: Snippet
  } = $props()

  const id = openDialog()
  const layer = layerOf(id)
  const opener = typeof document === 'undefined' ? null : (document.activeElement as HTMLElement | null)

  const motion =
    typeof window === 'undefined' ||
    !window.matchMedia ||
    !window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const duration = $derived(!motion || shape === 'card' ? 0 : shape === 'side' ? 180 : 150)

  function enter(node: Element): ReturnType<typeof fly> {
    if (shape === 'side') return fly(node, { x: slide, opacity: 1, duration })
    return fade(node, { duration })
  }

  $effect(() => {
    setFrame(id, frame)
  })

  // Focus starts inside: on what the dialog's own content focused (a field, Cancel), else on the
  // frame, so *Tab* and *Esc* act on the dialog and not on what was focused behind it.
  $effect(() => {
    void tick().then(() => {
      if (frame && !frame.contains(document.activeElement)) frame.focus()
    })
  })

  onDestroy(() => {
    closeDialog(id)
    const front = frontFrame()
    if (mayRestoreTo(opener, front ? (o) => front.contains(o as Node) : null)) opener?.focus()
  })

  function onWindowKey(e: KeyboardEvent): void {
    // A field that uses Esc itself (a rename) calls preventDefault, and the dialog stays.
    if (e.key !== 'Escape' || e.defaultPrevented || !isFront(id) || locked) return
    e.preventDefault()
    if (onEscape) onEscape()
    else onClose()
  }

  function onFrameKey(e: KeyboardEvent): void {
    if (e.key !== 'Tab' || !frame) return
    const focusable = Array.from(frame.querySelectorAll<HTMLElement>(FOCUSABLE))
    const to = wrapTab(focusable, document.activeElement, e.shiftKey, frame)
    if (to) {
      e.preventDefault()
      to.focus()
    }
  }
</script>

<svelte:window onkeydown={onWindowKey} />

<div
  class="scrim"
  style:z-index={layer}
  onclick={() => !locked && onClose()}
  role="presentation"
  transition:fade={{ duration }}
></div>

<div
  class="frame {shape}"
  class:top
  class:scroll
  style:z-index={layer + 1}
  style:--w={width}
  style:--h={height}
  style:--pad={padding}
  style:--gap={gap}
  {role}
  aria-modal="true"
  aria-label={label}
  aria-labelledby={labelledby}
  aria-describedby={describedby}
  tabindex="-1"
  bind:this={frame}
  onkeydown={onFrameKey}
  transition:enter
>
  {@render children()}
</div>

{#if edge}
  <div class="edge" style:z-index={layer + 2}>{@render edge()}</div>
{/if}

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: var(--scrim);
  }
  .frame {
    position: fixed;
    display: flex;
    flex-direction: column;
    gap: var(--gap, 0);
    color: var(--fg);
    outline: none; /* focused when nothing inside asks for it, so Tab and Esc act here */
  }
  /* Centred without a transform: a transform would make every `position: fixed` inside the
     frame (a menu, a tooltip anchor) measure from the frame instead of the window. */
  .card {
    inset: 0;
    margin: auto;
    width: min(94vw, var(--w));
    height: fit-content;
    max-height: var(--h, 90vh);
    overflow: auto;
    gap: var(--gap, var(--space-2));
    padding: var(--pad, var(--space-4));
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
  }
  .card.top {
    top: 12vh;
    bottom: auto;
  }
  .side {
    top: 0;
    right: 0;
    bottom: 0;
    width: var(--w);
    max-width: 96vw;
    padding: var(--pad, 0 1.2rem 1.2rem);
    overflow: hidden;
    background: var(--bg);
    border-left: 1px solid var(--border);
    box-shadow: var(--shadow-side);
  }
  .side.scroll {
    overflow-y: auto;
  }
  .screen {
    inset: 0;
    margin: auto;
    width: min(var(--w), calc(100vw - 2rem));
    height: min(var(--h, 52rem), calc(100vh - 2rem));
    padding: var(--pad, 0);
    overflow: hidden;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow-2);
  }
  /* The layer above the frame takes no clicks itself; what it draws does. */
  .edge {
    position: fixed;
    inset: 0;
    pointer-events: none;
  }
  .edge > :global(*) {
    pointer-events: auto;
  }
  @media (max-width: 640px) {
    .card {
      width: 94vw;
      max-height: 90vh;
    }
  }
</style>
