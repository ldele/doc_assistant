<script lang="ts">
  // The document page (feature-library-browser.md L1 + the 2026-08-09 user feedback).
  //
  // Five blocks in a fixed order — Metadata → Connections → Chunks → Figures → References —
  // each with a heading and an anchor the page's index jumps to. The organisation was already
  // liked; what was missing was that it read as one undifferentiated scroll.
  //
  // This file keeps the block order and its anchors, and the split that puts the source pane
  // beside the page. Its parts, each in its own file: `PageIndex` (the index, *Source*,
  // *Re-run…*), `DocumentHeader` (title and metadata: the Metadata block), the letter-spacing
  // block, `Connections`, `Chunks` (which owns its fetch: collapsed means not fetched),
  // `Figures`, `References`, and `SourceViewer`.
  import type { LibraryDocument } from '../../lib/core/types'
  import Chunks from './Chunks.svelte'
  import Connections from './Connections.svelte'
  import DocumentHeader from './DocumentHeader.svelte'
  import Figures from './Figures.svelte'
  import PageIndex from './PageIndex.svelte'
  import References from './References.svelte'
  import LetterSpacing from './LetterSpacing.svelte'
  import { shell } from '../../lib/shell/shell.svelte'
  import SourceViewer from './SourceViewer.svelte'
  import { clampSplit } from '../../lib/library/sourceviewer'
  import { libPrefs, setSourceSplit } from '../../lib/library/prefs.svelte'

  let {
    docId,
    doc,
    onOpenDocument,
    onReingest,
    sourceJump = null,
  }: {
    docId: string | null
    /** The open document's list summary — the metadata block's source. */
    doc: LibraryDocument | null
    onOpenDocument?: (id: string) => void
    /** A request from elsewhere in the app — a chat citation — to open the source pane at a
     *  page. `nonce` is what lets the same document+page be asked for twice in a row. */
    sourceJump?: { docId: string; page: number | null; nonce: number } | null
    /** Open the re-run picker for this document (ADR-048). App owns the dialog, as it owns every
     *  other overlay — so the grid's Select mode reuses the same one rather than a second copy.
     *  `parts` are ticked on open. */
    onReingest: (parts: string[]) => void
  } = $props()

  // The source pane (ROADMAP 18). Closed by default: it is a second read of the same document,
  // and opening it unasked would halve the width of the one the reader came for.
  let viewerOpen = $state(false)
  // Which page the pane should open on, and it is `null` far more often than not — every block
  // opened without asking for its place in the original. `SourceViewer` treats null as "page 1,
  // claiming nothing", which is the honest reading (ADR-050 D2).
  let viewerPage = $state<number | null>(null)

  // --- the splitter ------------------------------------------------------------------------
  //
  // Pointer events rather than mouse, so a trackpad, a pen and a touch drag all work from one
  // path; capture, so a fast drag that outruns the 6px handle keeps its grip.
  let splitEl = $state<HTMLDivElement | null>(null)

  function fractionAt(clientX: number): number {
    const box = splitEl?.getBoundingClientRect()
    if (!box || box.width <= 0) return libPrefs.sourceSplit
    return clampSplit((box.right - clientX) / box.width)
  }

  function onSplitDown(e: PointerEvent): void {
    e.preventDefault()
    const handle = e.currentTarget as HTMLElement
    // Capture is an optimisation, not a requirement: it throws for a pointer the browser no
    // longer considers active, and an exception here would abort before the listeners below are
    // attached — leaving a handle that looks grabbed and does nothing. Drag still works without.
    try {
      handle.setPointerCapture(e.pointerId)
    } catch {
      /* ignore — the drag falls back to plain listeners */
    }
    const move = (ev: PointerEvent) => setSourceSplit(fractionAt(ev.clientX))
    const up = (ev: PointerEvent) => {
      try {
        handle.releasePointerCapture(ev.pointerId)
      } catch {
        /* ignore — nothing was captured */
      }
      handle.removeEventListener('pointermove', move)
      handle.removeEventListener('pointerup', up)
      handle.removeEventListener('pointercancel', up)
    }
    handle.addEventListener('pointermove', move)
    handle.addEventListener('pointerup', up)
    handle.addEventListener('pointercancel', up)
  }

  /** Keyboard: a separator that can only be dragged is a separator some people cannot move. */
  function onSplitKey(e: KeyboardEvent): void {
    const step = e.shiftKey ? 0.1 : 0.02
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      setSourceSplit(libPrefs.sourceSplit + step)
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      setSourceSplit(libPrefs.sourceSplit - step)
    } else if (e.key === 'Home') {
      e.preventDefault()
      setSourceSplit(0.46)
    }
  }

  // A jump from outside opens the pane where it was asked to. Keyed on the nonce so a second
  // request for the same page still fires, and guarded on the id so a jump does not open the
  // pane over a *different* document the reader has since navigated to.
  let servedJump = -1
  $effect(() => {
    const jump = sourceJump
    if (!jump || jump.nonce === servedJump || jump.docId !== docId) return
    servedJump = jump.nonce
    viewerOpen = true
    viewerPage = jump.page
  })

  // The index and the block order are the same list — a block cannot be added to one and
  // forgotten in the other. Each id is an anchor below (`doc-metadata` is on DocumentHeader).
  const BLOCKS = [
    { id: 'doc-metadata', label: 'Metadata' },
    { id: 'doc-connections', label: 'Connections' },
    { id: 'doc-chunks', label: 'Chunks' },
    { id: 'doc-figures', label: 'Figures' },
    { id: 'doc-references', label: 'References' },
  ]
</script>

<section class="browser">
  {#if !docId}
    <p class="hint">Select a document from the sidebar to read it.</p>
  {:else}
    <PageIndex blocks={BLOCKS} bind:viewerOpen bind:viewerPage {onReingest} />

    <div class="split" bind:this={splitEl}>
    <div class="scroller">
    <DocumentHeader {doc} />
    <!-- ADR-056: what extraction did to this document's letter-spaced runs. Renders nothing for
         a document that has none, which is nearly every document. -->
    <LetterSpacing
      {docId}
      listed={doc}
      onShowPage={(page) => {
        viewerOpen = true
        viewerPage = page
      }}
      onReingest={() => onReingest(['text'])}
      onOpenSettings={() => {
        shell.settingsSection = 'documents'
        shell.settingsFocus = 'spacing'
        shell.showSettings = true
      }}
    />

    <!-- E4 (ADR-027 D1): what this document is *near* + who cites it. What it cites is the
         References block at the foot. Advisory; degrades to one quiet line on failure. -->
    <div class="block" id="doc-connections">
      <Connections {docId} {onOpenDocument} />
    </div>

    <div class="block" id="doc-chunks">
      <Chunks {docId} {doc} bind:viewerOpen bind:viewerPage />
    </div>

    <!-- L1b: figures, addressed separately from the text chunks — a figure is a different
         kind of object, and the panel states which of them are searchable. -->
    <div class="block" id="doc-figures">
      <Figures {docId} />
    </div>

    <div class="block" id="doc-references">
      <References {docId} {onOpenDocument} />
    </div>
    </div>

    {#if viewerOpen}
      <!-- A focusable `separator` IS the ARIA window-splitter pattern: the role is
           non-interactive until it takes a `tabindex`, which is exactly what makes the split
           movable without a pointer. The two rules below fire on the shape, not the mistake. -->
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <div
        class="splitter"
        role="separator"
        aria-label="Resize the source pane"
        aria-orientation="vertical"
        aria-valuenow={Math.round(libPrefs.sourceSplit * 100)}
        aria-valuemin={25}
        aria-valuemax={75}
        tabindex="0"
        onpointerdown={onSplitDown}
        onkeydown={onSplitKey}
        ondblclick={() => setSourceSplit(0.46)}
      ></div>
      <SourceViewer
        {docId}
        citedPage={viewerPage}
        onClose={() => {
          viewerOpen = false
          viewerPage = null
        }}
      />
    {/if}
    </div>
  {/if}
</section>

<style>
  /* The pane is a column: a fixed index strip, then everything that scrolls. `min-height: 0`
     is load-bearing — without it a flex child refuses to shrink and the inner scroller never
     gets a scrollbar. */
  .browser {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
  }
  /* The split (ROADMAP 18). A row, so the pane sits *beside* the document rather than over it —
     the reader is comparing the two, which an overlay makes impossible. */
  .split {
    flex: 1;
    display: flex;
    gap: 0.7rem;
    min-height: 0;
    min-width: 0;
  }
  .scroller {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
    padding: 0.8rem 0;
  }
  /* A 3px rule with a 9px grab area: thin enough to read as a divider, wide enough to catch a
     pointer. The cursor is the affordance — nothing else advertises it, and nothing needs to. */
  .splitter {
    flex: none;
    width: 9px;
    margin: 0 -3px;
    cursor: col-resize;
    background: transparent;
    border: none;
    position: relative;
    touch-action: none;
  }
  .splitter::after {
    content: '';
    position: absolute;
    inset: 0 3px;
    border-radius: 2px;
    background: transparent;
  }
  .splitter:hover::after,
  .splitter:focus-visible::after {
    background: var(--accent);
  }
  .splitter:focus-visible {
    outline: none;
  }
  @media (max-width: 900px) {
    /* Stacked: there is no horizontal split to drag. */
    .splitter {
      display: none;
    }
  }
  /* Below this the split would give each half too little to be readable, so the pane stacks
     under the document instead of beside it. */
  @media (max-width: 900px) {
    .split {
      flex-direction: column;
      overflow-y: auto;
    }
  }
  .hint {
    color: var(--fg-2);
    margin-top: 2rem;
    text-align: center;
  }
  .block {
    /* No sticky strip to clear any more — just enough that a jumped-to heading is not flush
       against the top edge. */
    scroll-margin-top: 0.5rem;
  }
</style>
