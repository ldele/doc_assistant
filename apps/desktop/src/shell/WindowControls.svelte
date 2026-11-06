<script lang="ts">
  // The window's own minimise, maximise/restore and close, at the trailing end of the top bar
  // (ADR-065). On Windows the window has no native title bar (`src-tauri/tauri.windows.conf.json`),
  // so the top bar is the title bar: these three buttons, and its empty space drags the window.
  // Draws nothing in a browser (the Vite preview) or in a window that keeps its native frame.
  //
  // A written exception to ADR-063 (the four action roles): these follow the platform's caption
  // buttons, not the app's button look. Full height and 46px wide, flush with the window's top
  // and right edges, so a maximised window's corner pixel is Close; 10px hairline glyphs, not the
  // app's Lucide set; the arrow cursor; Close turns red on hover (the platform's red, in both
  // themes, which here means "this closes the window", not ADR-063's danger role); and Close asks
  // nothing first, as the native close never did. Focus shows the app's ring, inset, because the
  // buttons touch the window's edge.
  // Hover text: each button takes the shared tooltip (ADR-063 choice 6), never the native `title`.
  import { onMount } from 'svelte'
  import { tooltip } from '../lib/shell/tooltip'
  import {
    closeWindow,
    drawsOwnWindowControls,
    isWindowMaximized,
    minimizeWindow,
    onWindowFocusChanged,
    onWindowResized,
    toggleMaximizeWindow,
  } from '../lib/core/tauri'

  let shown = $state(false)
  let maximized = $state(false)
  let active = $state(true)

  onMount(() => {
    let cancelled = false
    const stops: Array<() => void> = []
    const sync = async (): Promise<void> => {
      maximized = await isWindowMaximized()
    }
    void drawsOwnWindowControls().then((own) => {
      if (cancelled || !own) return
      shown = true
      void sync()
      stops.push(onWindowResized(() => void sync()))
      stops.push(onWindowFocusChanged((focused) => (active = focused)))
    })
    return () => {
      cancelled = true
      for (const stop of stops) stop()
    }
  })
</script>

{#if shown}
  <div class="wc" class:inactive={!active} role="group" aria-label="Window">
    <button class="wc-btn" type="button" aria-label="Minimise" onclick={minimizeWindow} use:tooltip={'Hide the window in the taskbar'}>
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true" focusable="false">
        <path d="M0 5.5H10" />
      </svg>
    </button>
    <button
      class="wc-btn"
      type="button"
      aria-label={maximized ? 'Restore' : 'Maximise'}
      onclick={toggleMaximizeWindow}
      use:tooltip={maximized ? 'Return the window to its earlier size' : 'Make the window fill the screen'}
    >
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true" focusable="false">
        {#if maximized}
          <rect x="0.5" y="2.5" width="7" height="7" rx="1" />
          <path d="M2.5 2.5V1.5a1 1 0 0 1 1-1h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-1" />
        {:else}
          <rect x="0.5" y="0.5" width="9" height="9" rx="1" />
        {/if}
      </svg>
    </button>
    <button class="wc-btn wc-close" type="button" aria-label="Close" onclick={closeWindow} use:tooltip={'Quit Provenote'}>
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true" focusable="false">
        <path d="M0.5 0.5L9.5 9.5M9.5 0.5L0.5 9.5" />
      </svg>
    </button>
  </div>
{/if}

<style>
  .wc {
    display: flex;
    flex: none;
    align-self: stretch;
    /* Flush with the window's top, right and the bar's bottom edge: the top bar's own padding is
       cancelled with the variables it sets (Topbar.svelte), 0 anywhere else. */
    margin: calc(-1 * var(--tb-pad-y, 0px)) calc(-1 * var(--tb-pad-x, 0px))
      calc(-1 * var(--tb-pad-y, 0px)) 0;
    user-select: none;
  }
  .wc-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 46px;
    margin: 0;
    padding: 0;
    border: none;
    border-radius: 0;
    background: none;
    color: var(--fg);
    cursor: default;
  }
  .wc-btn svg {
    fill: none;
    stroke: currentColor;
    stroke-width: 1;
    overflow: visible;
  }
  .wc-btn:hover {
    background: var(--surface-2);
  }
  .wc-btn:active {
    background: color-mix(in srgb, var(--fg) 14%, transparent);
  }
  .wc-close:hover {
    background: var(--caption-close);
    color: var(--caption-close-fg);
  }
  .wc-close:active {
    background: color-mix(in srgb, var(--caption-close) 85%, var(--caption-close-fg));
    color: var(--caption-close-fg);
  }
  .wc-btn:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
  }
  /* An inactive window greys its caption glyphs, as the platform does. */
  .inactive .wc-btn:not(:hover):not(:focus-visible) {
    color: color-mix(in srgb, var(--fg) 45%, transparent);
  }
</style>
