<script lang="ts">
  // The source pane (ROADMAP 18, ADR-050) — the document itself, beside its library entry.
  //
  // One page image at a time, fetched by the browser as a plain `<img src>`: the server renders
  // on demand (19-31 ms, 140-261 KB a page measured), so there is nothing to cache here and no
  // PDF renderer in the client. The decisions live in `sourceviewer.ts` and are tested there;
  // this file is the pane.
  //
  // The state that is easy to get wrong is `located`. A citation that could not be placed opens
  // at page 1 — which looks *exactly* like a citation that really is on page 1. The flag is why
  // the header says "page 4 of this document" in one case and nothing in the other.
  import { untrack } from 'svelte'
  import type { SourceDocumentView } from '../core/types'
  import { getSourceDocumentView, sourcePageUrl } from '../core/api'
  import {
    BASE_DPI,
    canStep,
    clampPage,
    clampZoom,
    fitPageZoom,
    openingPage,
    pageLabel,
    pageWidthPt,
    renderDpi,
    sourceMode,
    unavailableMessage,
    zoomLabel,
    zoomStep,
  } from './sourceviewer'
  import { libPrefs, setSourceFit } from './prefs.svelte'
  import Icon from '../shell/Icon.svelte'

  let {
    docId,
    citedPage = null,
    onClose,
  }: {
    docId: string
    /** The page a citation resolved to, or null when it could not be placed (ADR-050 D2). */
    citedPage: number | null
    onClose: () => void
  } = $props()

  // `undefined` = not asked yet, `null` = asked and there is no such document. `sourceMode`
  // keeps those apart because they are different sentences.
  let view = $state<SourceDocumentView | null | undefined>(undefined)
  let page = $state(1)
  let located = $state(false)
  let loadFailed = $state(false)

  let mode = $derived(sourceMode(view))
  let pageCount = $derived(view?.page_count ?? null)

  // --- zoom -------------------------------------------------------------------------------
  //
  // `zoom` is a multiple of the pane's own width (see `sourceviewer.ts`). The two fit presets
  // are not a separate mode: they *set* zoom, so a reader who then presses + is zooming from
  // where they were rather than from somewhere invisible.
  let zoom = $state(1)
  /** The pane's **content** box, tracked so a fit preset and a resize both stay correct.
   *  Content, not border: the body carries vertical padding, and fitting to the padded height
   *  left a genuinely-fitted page needing 16px of scroll. */
  let boxW = $state(0)
  let boxH = $state(0)

  /** The page image's own 1px border, on each axis. Structural, not a tunable — it is the
   *  difference between the box the picture is fitted to and the box it actually occupies. */
  const PAGE_CHROME_PX = 2
  /** Page shape and true size, learned from whatever render arrived — the API never states it. */
  let aspect = $state(0)
  let widthPt = $state(0)
  let bodyEl = $state<HTMLDivElement | null>(null)

  let cssWidth = $derived(boxW > 0 ? Math.round((boxW - PAGE_CHROME_PX) * zoom) : 0)
  /** Sharper renders as the page is drawn bigger; snapped to a ladder so a drag is not a
   *  request per frame, and `dpr` because a dense display needs more pixels for the same box. */
  let dpi = $derived(
    widthPt > 0 && cssWidth > 0
      ? renderDpi(cssWidth, widthPt, typeof devicePixelRatio === 'number' ? devicePixelRatio : 1)
      : BASE_DPI,
  )

  /** True once the reader has zoomed by hand, which is what stops the pane snapping back.
   *
   * An earlier version inferred this by comparing `zoom` against the fit — that raced with image
   * load (a fit computed before the box was measured is 1, and 1 is also a legitimate zoom), and
   * the pane opened at 100% instead of fitted. A flag cannot race. */
  let userZoomed = $state(false)

  function applyFit(fit: 'page' | 'width'): void {
    setSourceFit(fit)
    userZoomed = false
  }

  function nudgeZoom(direction: 1 | -1): void {
    zoom = zoomStep(zoom, direction)
    userZoomed = true
  }

  /** Ctrl/Cmd + wheel zooms, which is what every document viewer does; a bare wheel scrolls. */
  function onWheel(e: WheelEvent): void {
    if (!(e.ctrlKey || e.metaKey)) return
    e.preventDefault()
    zoom = clampZoom(zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12))
    userZoomed = true
  }

  // The fit is *derived* while the reader has not taken over: box, page shape and the chosen
  // preset all feed it, so a resized pane or a differently-shaped page re-fits on its own.
  $effect(() => {
    if (userZoomed) return
    const fitted =
      libPrefs.sourceFit === 'width'
        ? 1
        : fitPageZoom(boxW - PAGE_CHROME_PX, boxH - PAGE_CHROME_PX, aspect)
    if (Math.abs(fitted - untrack(() => zoom)) > 0.001) zoom = fitted
  })

  /** Learn the page's shape and size from the render that just arrived, then honour the
   *  preferred fit. Runs per page, because pages within a document can differ in size. */
  function onPageLoad(e: Event): void {
    const img = e.currentTarget as HTMLImageElement
    if (!img.naturalWidth || !img.naturalHeight) return
    aspect = img.naturalHeight / img.naturalWidth
    // `dpi` is what we asked for, so this is exact rather than an estimate.
    widthPt = pageWidthPt(img.naturalWidth, dpi)
  }

  // Track the pane's box so a fitted page stays fitted when the split or the window is dragged.
  $effect(() => {
    const el = bodyEl
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver((entries) => {
      // `contentRect` is the content box — padding and border already removed, which is exactly
      // the space a page has to fit into.
      const r = entries[0]?.contentRect
      if (!r) return
      boxW = r.width
      boxH = r.height
    })
    ro.observe(el)
    return () => ro.disconnect()
  })

  // Fetching is keyed on the document; the opening page comes from the citation the pane was
  // opened with. `untrack` on the writes, so setting `page` here does not re-run the fetch.
  let token = 0
  $effect(() => {
    const id = docId
    const cited = citedPage
    const mine = ++token
    untrack(() => {
      view = undefined
      loadFailed = false
    })
    void getSourceDocumentView(id)
      .then((v) => {
        if (mine !== token) return
        view = v
        const opening = openingPage(cited, v?.page_count ?? null)
        page = opening.page
        located = opening.located
        // A new document is a new page shape; re-learn it rather than carry the old one.
        aspect = 0
        widthPt = 0
        zoom = 1
        userZoomed = false
      })
      .catch(() => {
        if (mine !== token) return
        // A failed fetch is not a missing document — say so rather than claiming it is unknown.
        view = null
        loadFailed = true
      })
  })

  function step(delta: number): void {
    if (!canStep(page, pageCount, delta)) return
    page = clampPage(page + delta, pageCount)
    // Stepping away from the cited page means the header should stop claiming it.
    located = false
    loadFailed = false
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault()
      step(-1)
    } else if (e.key === 'ArrowRight' || e.key === 'PageDown') {
      e.preventDefault()
      step(1)
    }
  }

  // The next page, requested invisibly so a forward step is instant. Only ever one page ahead —
  // this is a convenience, not a prefetcher, and the corpus is 2,973 pages.
  let nextUrl = $derived(
    mode === 'pages' && canStep(page, pageCount, 1) ? sourcePageUrl(docId, page + 1, dpi) : null,
  )
</script>

<aside class="sourcepane" aria-label="Source document" style={`--source-split:${(libPrefs.sourceSplit * 100).toFixed(2)}%`}>
  <header class="shead">
    <span class="stitle" title={view?.filename ?? ''}>
      <Icon name="book-open" size={14} />
      {view?.filename ?? 'Source'}
    </span>
    <!-- Fit control. `page` is the default because the pane answers *where did this come from*;
         at fit-width no page in the corpus fits its box (measured 2026-09-01 — 94% visible on US
         Letter, 67% on the tallest), so the whole-page view was unreachable without scrolling. -->
    <!-- Only where there is a page to size. Over an unreachable file or a format without pages
         these controls do nothing, and a dead control is worse than none — found by driving it,
         not by a test, because the corpus has no unreachable or non-PDF document to hit. -->
    {#if mode === 'pages'}
    <div class="sfit" role="group" aria-label="Page size">
      <button
        class:active={libPrefs.sourceFit === 'page'}
        onclick={() => applyFit('page')}
        aria-pressed={libPrefs.sourceFit === 'page'}
        title="Fit the whole page"
        type="button">Fit page</button
      >
      <button
        class:active={libPrefs.sourceFit === 'width'}
        onclick={() => applyFit('width')}
        aria-pressed={libPrefs.sourceFit === 'width'}
        title="Fill the width and scroll"
        type="button">Width</button
      >
    </div>
    <div class="szoom" role="group" aria-label="Zoom">
      <button onclick={() => nudgeZoom(-1)} type="button" aria-label="Zoom out" title="Zoom out">&minus;</button>
      <!-- The reading is the button: pressing it returns to the preferred fit, which is the
           only exit from a deep zoom that does not involve hunting for the right level. -->
      <button
        class="szoomlabel"
        onclick={() => applyFit(libPrefs.sourceFit)}
        type="button"
        title="Back to {libPrefs.sourceFit === 'page' ? 'fit page' : 'width'}"
      >{zoomLabel(zoom)}</button>
      <button onclick={() => nudgeZoom(1)} type="button" aria-label="Zoom in" title="Zoom in">+</button>
    </div>
    {/if}
    <button class="sclose" onclick={onClose} type="button" aria-label="Close the source view" title="Close">
      <Icon name="x" size={15} />
    </button>
  </header>

  {#if mode === 'loading'}
    <p class="snote">Opening the document…</p>
  {:else if mode === 'pages'}
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="sbody"
      bind:this={bodyEl}
      role="document"
      tabindex="-1"
      onkeydown={onKey}
      onwheel={onWheel}
    >
      {#if loadFailed}
        <p class="snote">
          <Icon name="triangle-alert" size={14} />
          This page could not be rendered.
        </p>
      {:else}
        <!-- Keyed on the URL so a page change swaps the element rather than mutating `src`,
             which would leave the previous page on screen while the next one decodes. -->
        <!-- Keyed on page AND dpi: a sharper render is a different image, and swapping the
             element rather than mutating `src` avoids showing the old one stretched while the
             new one decodes. Width is set explicitly so the picture's own pixel size — which
             changes with dpi — never drives the layout. -->
        {#key `${page}:${dpi}`}
          <img
            class="spage"
            src={sourcePageUrl(docId, page, dpi)}
            alt={`Page ${page} of ${view?.filename ?? 'the document'}`}
            style={cssWidth > 0 ? `width:${cssWidth}px` : ''}
            onload={onPageLoad}
            onerror={() => (loadFailed = true)}
          />
        {/key}
      {/if}
    </div>

    <footer class="sfoot">
      <button
        class="snav"
        type="button"
        onclick={() => step(-1)}
        disabled={!canStep(page, pageCount, -1)}
        aria-label="Previous page"
      >
        <Icon name="arrow-left" size={13} />
      </button>
      <span class="spagelabel">
        {pageLabel(page, pageCount)}
        <!-- Only said when the citation actually resolved. An unplaceable one opens here too. -->
        {#if located}<span class="scited">cited here</span>{/if}
      </span>
      <button
        class="snav"
        type="button"
        onclick={() => step(1)}
        disabled={!canStep(page, pageCount, 1)}
        aria-label="Next page"
      >
        <Icon name="arrow-right" size={13} />
      </button>
    </footer>

    {#if nextUrl}
      <img class="sprefetch" src={nextUrl} alt="" aria-hidden="true" />
    {/if}
  {:else}
    <div class="sbody">
      <p class="snote">
        <Icon name={mode === 'text-only' ? 'file-text' : 'triangle-alert'} size={14} />
        {unavailableMessage(view)}
      </p>
      {#if mode === 'text-only'}
        <!-- Not a dead end: the reader is already on the page that holds the extracted text. -->
        <p class="shint">Open <strong>Chunks</strong> to read its extracted text.</p>
      {/if}
    </div>
  {/if}
</aside>

<style>
  .sourcepane {
    /* Width is the reader's, dragged on the splitter and persisted; the fallback is the old
       fixed share, for the moment before the pref is read. */
    flex: 0 0 var(--source-split, min(46%, 620px));
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    border-left: 1px solid var(--border);
    padding-left: 0.7rem;
  }
  .shead {
    flex: none;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }
  .stitle {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.78rem;
    color: var(--fg-2);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* The header controls were too small to find (user, 2026-09-01) — 0.15rem of padding on a
     0.7rem label, in the muted `--fg-2`, against a pane border that reads as part of the frame.
     Three changes, none of them a redesign: a real hit target (28px, the size a pointer expects
     rather than the size the text needs), the resting colour moved off the muted token onto
     `--fg`, and a hover that fills rather than only tinting the glyph. */
  .sfit,
  .szoom {
    flex: none;
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .sfit {
    margin-left: auto;
  }
  .sfit button,
  .szoom button {
    border: none;
    border-radius: 0;
    background: var(--surface);
    color: var(--fg);
    font: inherit;
    font-size: 0.76rem;
    line-height: 1;
    min-height: 1.75rem;
    padding: 0 0.6rem;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
  }
  .sfit button:hover,
  .szoom button:hover {
    background: var(--surface-2);
  }
  .sfit button.active {
    background: var(--accent);
    color: var(--accent-fg);
  }
  /* The steppers carry a single glyph, so padding alone left them 23px wide against a 26px
     row — square is the shape a pointer aims at. */
  .szoom button {
    padding: 0 0.4rem;
    min-width: 1.75rem;
    justify-content: center;
  }
  .szoomlabel {
    min-width: 3.4rem;
    justify-content: center;
    font-variant-numeric: tabular-nums;
    border-left: 1px solid var(--border) !important;
    border-right: 1px solid var(--border) !important;
  }
  /* Pushed right whether or not the size controls are there — they carry the auto margin when
     present, and this takes over when they are not. A square target rather than a pill: at this
     size a circle reads as smaller than it is, which was half the findability problem. */
  .sclose {
    margin-left: auto;
    flex: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    min-width: 1.75rem;
    min-height: 1.75rem;
    color: var(--fg);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .sclose:hover {
    background: var(--surface-2);
    border-color: var(--accent);
    color: var(--accent);
  }
  .sfit button:focus-visible,
  .szoom button:focus-visible,
  .sclose:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .sbody {
    flex: 1;
    overflow: auto;
    min-height: 0;
    padding: 0.6rem 0;
    display: flex;
    /* `safe center` keeps a zoomed page reachable: plain centring pushes the overflowing left
       edge out of the scroll range, so the start of a line can never be scrolled back to. */
    justify-content: safe center;
    align-items: safe center;
  }
  .sbody:focus-visible {
    outline: none;
  }
  /* Two ways to size a page, because the two things a reader does with it are different.
     `width` fills the pane and scrolls — for reading. `page` fits the whole thing, so the
     position of a passage on the page is visible at a glance, which is what the pane is for.
     Height is the binding constraint in both: measured, no corpus page fits at fit-width. */
  /* The width comes from `zoom` in the script, not from CSS: the picture's own pixel size
     changes every time a sharper render arrives, and letting that drive layout would make the
     page jump on every zoom step. `flex: none` stops the flex parent shrinking it back to fit. */
  .spage {
    flex: none;
    height: auto;
    border: 1px solid var(--border);
    background: #fff;
  }
  .sfoot {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    padding-top: 0.4rem;
    border-top: 1px solid var(--border);
  }
  .snav {
    background: none;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.1rem 0.45rem;
    color: var(--fg-2);
    cursor: pointer;
    display: inline-flex;
  }
  .snav:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .spagelabel {
    font-size: 0.76rem;
    color: var(--fg-2);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .scited {
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0 0.4rem;
    font-size: 0.68rem;
    color: var(--accent);
  }
  .snote {
    display: inline-flex;
    align-items: flex-start;
    gap: 0.4rem;
    margin: 1.2rem 0 0;
    font-size: 0.82rem;
    color: var(--fg-2);
    text-align: left;
  }
  .shint {
    margin: 0.4rem 0 0;
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  /* Stacked under the document rather than beside it (the `.split` breakpoint in
     LibraryBrowser). The divider has to move with the layout: a left border on a pane that is
     no longer to the right of anything reads as a stray vertical rule. */
  @media (max-width: 900px) {
    .sourcepane {
      /* Stacked: the split fraction is horizontal and means nothing here. */
      flex: 0 0 auto;
      border-left: none;
      border-top: 1px solid var(--border);
      padding-left: 0;
      padding-top: 0.6rem;
      margin-top: 0.6rem;
    }
    .sbody {
      max-height: 60vh;
    }
  }
  /* Requested so the browser has it decoded before the reader steps forward; never shown. */
  .sprefetch {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }
</style>
