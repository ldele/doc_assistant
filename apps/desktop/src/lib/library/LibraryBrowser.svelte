<script lang="ts">
  // Library document view (feature-library-browser.md L1 + the 2026-08-09 user feedback).
  //
  // Five blocks in a fixed order — Metadata → Connections → Chunks → Figures → References —
  // each with a heading and an anchor the nav strip jumps to. The organisation was already
  // liked; what was missing was that it read as one undifferentiated scroll.
  //
  // **Chunks is collapsed, and collapsed means not fetched.** That is a measured cost, not a
  // tidiness preference: the detail payload is a median 170 KB and up to 1.85 MB per document
  // on this corpus (663 parent blocks + 2,608 children for the largest), and the old view
  // rendered every parent's text plus every child node eagerly on open. Opening a document now
  // costs zero chunk bytes; the text arrives when the reader asks for it — and once they have
  // asked, `chunkmemory` keeps that document open for the session, so the top ← → arrows come
  // back to it as they left it.
  //
  // The header therefore reads the document summary the Library list already holds (`doc`)
  // rather than the chunk payload — the same fields, without the text.
  import { untrack } from 'svelte'
  import type { LibraryDocument, LibraryDocumentChunks } from '../core/types'
  import { getLibraryDocument, locateChunk } from '../core/api'
  import { blockPreview } from './library'
  import { rememberChunksOpen, wereChunksOpen } from './chunkmemory'
  import Icon from '../shell/Icon.svelte'
  import DocConnections from './DocConnections.svelte'
  import DocFigures from './DocFigures.svelte'
  import DocReferences from './DocReferences.svelte'
  import SourceViewer from './SourceViewer.svelte'
  import { clampSplit } from './sourceviewer'
  import { libPrefs, setSourceSplit } from './prefs.svelte'

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
     *  other overlay — so the grid's Select mode reuses the same one rather than a second copy. */
    onReingest: () => void
  } = $props()

  // The chunk payload: null until the reader opens the Chunks block.
  let detail = $state<LibraryDocumentChunks | null>(null)
  let loading = $state(false)
  let error = $state<string | null>(null)
  let chunksOpen = $state(false)

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

  /** Open the pane at the page a parent block sits on.
   *
   * The chunk key is the same `{document_id}:p{parent_index}` shape a chat citation carries, so
   * this asks the same endpoint a citation would. A block whose page cannot be resolved still
   * opens the pane — at page 1, saying nothing about position. */
  async function showBlockInSource(parentIndex: number): Promise<void> {
    if (!docId) return
    viewerOpen = true
    viewerPage = null
    const key = `${docId}:p${parentIndex}`
    const found = await locateChunk(key)
    // Guard the late answer: the reader may have opened another block, or closed the pane.
    if (viewerOpen) viewerPage = found?.page ?? null
  }

  // Changing document always drops the previous payload — holding 1.85 MB of text for a
  // document the reader has left is the cost this block exists to avoid. The *open state*,
  // though, is remembered per document for the session (`chunkmemory`), so coming back via the
  // top ← → arrows lands where they left off. Remembering the state costs one re-fetch;
  // remembering the payload would cost the whole point.
  //
  // Wrapped in `untrack`: `startChunkLoad` reads `detail`/`loading`, and a synchronous read
  // inside an effect makes them dependencies — the completing fetch would then re-run this
  // effect, null the payload, and fetch again, forever.
  let token = 0
  $effect(() => {
    const id = docId
    untrack(() => {
      detail = null
      error = null
      loading = false
      openParents = {}
      token++
      chunksOpen = wereChunksOpen(id)
      if (chunksOpen && id) startChunkLoad(id)
    })
  })

  function startChunkLoad(id: string): void {
    if (detail !== null || loading) return
    loading = true
    const mine = ++token
    void (async () => {
      try {
        const d = await getLibraryDocument(id)
        if (mine === token) detail = d
      } catch (e) {
        if (mine === token) error = String(e)
      } finally {
        if (mine === token) loading = false
      }
    })()
  }

  function toggleChunks(): void {
    chunksOpen = !chunksOpen
    const id = docId
    if (!id) return
    rememberChunksOpen(id, chunksOpen)
    if (chunksOpen) startChunkLoad(id)
  }

  // Which parent blocks the reader has opened. Collapsed is the default here too, and for the
  // same reason one level up: 82 blocks of prose rendered eagerly is what made the old view
  // expensive, and a reader scanning for one passage wants the *list*, not all of it at once.
  // Keyed by parent_index; reset when the document changes.
  let openParents = $state<Record<number, boolean>>({})

  function toggleParent(index: number): void {
    openParents[index] = !openParents[index]
  }

  function setAllParents(open: boolean): void {
    if (!detail) return
    openParents = open
      ? Object.fromEntries(detail.parents.map((p) => [p.parent_index, true]))
      : {}
  }

  // The nav strip and the block order are the same list — a block cannot be added to one and
  // forgotten in the other.
  const BLOCKS = [
    { id: 'doc-metadata', label: 'Metadata' },
    { id: 'doc-connections', label: 'Connections' },
    { id: 'doc-chunks', label: 'Chunks' },
    { id: 'doc-figures', label: 'Figures' },
    { id: 'doc-references', label: 'References' },
  ]

  // Instant, not smooth: with Chunks expanded the document view can be tens of thousands of
  // pixels tall, and smooth-scrolling that distance is a multi-second animation, not a jump.
  function jumpTo(id: string): void {
    document.getElementById(id)?.scrollIntoView({ block: 'start' })
  }
</script>

<section class="browser">
  {#if !docId}
    <p class="hint">Select a document from the sidebar to read it.</p>
  {:else}
    <!-- The index sits above the document and *outside* the scrolling area: as a sticky strip
         inside it, it passed over the text as the reader scrolled. Here it stays put under the
         breadcrumb band and never covers anything. -->
    <nav class="blocknav" aria-label="Jump to a section of this document">
      {#each BLOCKS as b (b.id)}
        <button type="button" onclick={() => jumpTo(b.id)}>{b.label}</button>
      {/each}
      <!-- ROADMAP 20 asked for this beside the block list, and that is the right place for a
           different reason than symmetry: the blocks are what a re-run re-derives, so the action
           belongs where the reader is already looking at the thing that came out wrong. -->
      <button
        class="rerun srcbtn"
        class:on={viewerOpen}
        type="button"
        aria-pressed={viewerOpen}
        onclick={() => {
          viewerOpen = !viewerOpen
          if (!viewerOpen) viewerPage = null
        }}
        title="Show the document itself beside this page"
      >
        <Icon name="book-open" size={13} /> Source
      </button>
      <button class="rerun" type="button" onclick={onReingest} title="Re-run part of ingestion for this document">
        <Icon name="rotate-ccw" size={13} /> Re-run…
      </button>
    </nav>

    <div class="split" bind:this={splitEl}>
    <div class="scroller">
    <header class="dochead" id="doc-metadata">
      <h2>{doc?.title ?? doc?.filename ?? 'Document'}</h2>
      {#if doc}
        <p class="metaline">
          {#if doc.title}<span>{doc.filename}</span><span>·</span>{/if}
          <span>{doc.format}</span>
          {#if doc.page_count != null}<span>· {doc.page_count} pages</span>{/if}
          {#if doc.health}<span>· {doc.health}</span>{/if}
        </p>
        {#if doc.authors}<p class="metaextra"><strong>Authors</strong> {doc.authors}</p>{/if}
        {#if doc.year != null}<p class="metaextra"><strong>Year</strong> {doc.year}</p>{/if}
        {#if doc.keywords.length > 0}
          <p class="metaextra">
            <strong>Keywords</strong>
            {doc.keywords.join(' · ')}
          </p>
        {/if}
      {:else}
        <!-- Inform, don't block: the blocks below key off the id and still render. -->
        <p class="metaline">This document isn’t in the loaded library list.</p>
      {/if}
    </header>

    <!-- E4 (ADR-027 D1): what this document is *near* + who cites it. What it cites is the
         References block at the foot. Advisory; degrades to one quiet line on failure. -->
    <div class="block" id="doc-connections">
      <DocConnections {docId} {onOpenDocument} />
    </div>

    <div class="block" id="doc-chunks">
      <h3>
        <button
          class="blocktoggle"
          type="button"
          aria-expanded={chunksOpen}
          onclick={toggleChunks}
        >
          <span class="chev" class:open={chunksOpen} aria-hidden="true">›</span>
          Chunks
        </button>
        <span class="count">
          {#if detail}
            {detail.parents.length} block{detail.parents.length === 1 ? '' : 's'} ·
            {detail.child_count} indexed chunk{detail.child_count === 1 ? '' : 's'}
          {:else if doc?.chunk_count != null}
            <!-- The registry's ingest-time count, labelled for what it is: the live index
                 counts below are a different number, and one screen used to show both
                 unlabelled (L1 review finding, 2026-07-13). -->
            {doc.chunk_count.toLocaleString()} recorded at ingest
          {/if}
        </span>
      </h3>
      {#if chunksOpen}
        {#if loading}
          <p class="quiet">Loading chunks…</p>
        {:else if error}
          <p class="quiet err">Couldn’t load this document’s chunks: {error}</p>
        {:else if detail && detail.parents.length === 0}
          <p class="quiet">No chunks stored for this document.</p>
        {:else if detail}
          <div class="listtools">
            <button type="button" onclick={() => setAllParents(true)}>Expand all</button>
            <button type="button" onclick={() => setAllParents(false)}>Collapse all</button>
          </div>
          <!-- A list of blocks, each opening to its own text. Every row carries a preview
               because "Block 0 / Block 1 / Block 2" is not something a reader can scan. -->
          {#each detail.parents as p (p.parent_index)}
            <article class="parent" class:open={openParents[p.parent_index]}>
              <button
                class="phead"
                type="button"
                aria-expanded={!!openParents[p.parent_index]}
                onclick={() => toggleParent(p.parent_index)}
              >
                <span class="chev" class:open={openParents[p.parent_index]} aria-hidden="true">›</span>
                <span class="plabel">Block {p.parent_index}</span>
                <span class="ppreview">{blockPreview(p.parent_text)}</span>
                <span class="pcount">{p.children.length}</span>
              </button>
              {#if openParents[p.parent_index]}
                <p class="blocktext">{p.parent_text}</p>
                <!-- ROADMAP 18/19 meet here: 19 shows the passage in the extracted *text*, this
                     shows the page of the original it came off. Same chunk key either way. -->
                <button class="insource" type="button" onclick={() => showBlockInSource(p.parent_index)}>
                  <Icon name="book-open" size={12} /> Show this page in the document
                </button>
                <details class="children">
                  <summary>{p.children.length} child chunk{p.children.length === 1 ? '' : 's'}</summary>
                  {#each p.children as c (c.child_index)}
                    <div class="child">
                      <div class="chead">
                        #{c.child_index}
                        {#if !c.retrievable}<span class="flag" title="Excluded from retrieval">not retrievable</span>{/if}
                      </div>
                      <p class="childtext">{c.text}</p>
                    </div>
                  {/each}
                </details>
              {/if}
            </article>
          {/each}
        {/if}
      {/if}
    </div>

    <!-- L1b: figures, addressed separately from the text chunks — a figure is a different
         kind of object, and the panel states which of them are searchable. -->
    <div class="block" id="doc-figures">
      <DocFigures {docId} />
    </div>

    <div class="block" id="doc-references">
      <DocReferences {docId} {onOpenDocument} />
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
  .blocknav .srcbtn {
    margin-left: auto;
  }
  /* With the source button taking `margin-left: auto`, Re-run must not also claim it — the two
     would then split the gap and neither would sit at the edge. */
  .blocknav .srcbtn ~ .rerun {
    margin-left: 0;
  }
  .blocknav .srcbtn.on {
    color: var(--fg);
    border-color: var(--accent);
  }
  .insource {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: none;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.1rem 0.55rem;
    margin: 0 0 0.5rem;
    font: inherit;
    font-size: 0.72rem;
    color: var(--fg-2);
    cursor: pointer;
  }
  .insource:hover {
    color: var(--fg);
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
  .dochead {
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin-bottom: 0.5rem;
  }
  .dochead h2 {
    margin: 0;
    font-size: 1.2rem;
    word-break: break-word;
    font-family: var(--font-serif);
  }
  .metaline {
    margin: 0.3rem 0 0;
    font-size: 0.78rem;
    color: var(--fg-2);
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .metaextra {
    margin: 0.25rem 0 0;
    font-size: 0.82rem;
    color: var(--fg);
  }
  .metaextra strong {
    color: var(--fg-2);
    font-weight: 600;
    margin-right: 0.35rem;
  }
  /* The strip is what makes the fixed block order navigable rather than merely present. It sits
     outside `.scroller`, so it is always there and never passes over the text. */
  .blocknav {
    flex: none;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    padding: 0 0 0.5rem;
    border-bottom: 1px solid var(--border);
  }
  .blocknav .rerun {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }
  .blocknav button {
    background: none;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    font: inherit;
    font-size: 0.74rem;
    color: var(--fg-2);
    cursor: pointer;
  }
  .blocknav button:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  .block {
    /* No sticky strip to clear any more — just enough that a jumped-to heading is not flush
       against the top edge. */
    scroll-margin-top: 0.5rem;
  }
  .block h3 {
    font-size: 0.9rem;
    margin: 1.25rem 0 0.5rem;
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
  }
  .blocktoggle {
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    color: var(--fg);
    cursor: pointer;
    display: inline-flex;
    align-items: baseline;
    gap: 0.35rem;
  }
  .chev {
    display: inline-block;
    transition: transform 0.12s ease;
    color: var(--fg-2);
  }
  .chev.open {
    transform: rotate(90deg);
  }
  .count {
    font-weight: 400;
    opacity: 0.65;
    font-size: 0.8rem;
  }
  .quiet {
    opacity: 0.6;
    font-size: 0.85rem;
  }
  .quiet.err {
    color: var(--warn-fg);
    opacity: 1;
  }
  .listtools {
    display: flex;
    gap: 0.4rem;
    margin-bottom: 0.5rem;
  }
  .listtools button {
    background: none;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.05rem 0.55rem;
    font: inherit;
    font-size: 0.72rem;
    color: var(--fg-2);
    cursor: pointer;
  }
  .listtools button:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  .parent {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    margin-bottom: 0.35rem;
    overflow: hidden;
  }
  /* Collapsed, a block is one scannable row: marker, label, preview, child count. */
  .phead {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    width: 100%;
    background: none;
    border: none;
    padding: 0.45rem 0.7rem;
    font: inherit;
    font-size: 0.78rem;
    color: var(--fg-2);
    cursor: pointer;
    text-align: left;
  }
  .phead:hover {
    background: var(--surface-2, var(--bg));
  }
  .plabel {
    font-weight: 600;
    white-space: nowrap;
  }
  .ppreview {
    flex: 1;
    min-width: 0;
    color: var(--fg);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .pcount {
    flex: none;
    font-size: 0.68rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0 0.3rem;
  }
  .parent.open .ppreview {
    /* Once the block is open its own text is right below — the preview would repeat it. */
    visibility: hidden;
  }
  .blocktext {
    margin: 0;
    padding: 0 0.75rem 0.6rem;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: 0.95rem;
    line-height: 1.6;
    font-family: var(--font-serif);
  }
  .children {
    margin: 0 0.75rem 0.6rem;
    border-top: 1px dashed var(--border);
    padding-top: 0.4rem;
  }
  .children summary {
    cursor: pointer;
    font-size: 0.76rem;
    color: var(--accent);
    user-select: none;
  }
  .child {
    margin-top: 0.5rem;
    padding-left: 0.6rem;
    border-left: 2px solid var(--border);
  }
  .chead {
    font-size: 0.7rem;
    color: var(--fg-2);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.2rem;
  }
  .flag {
    color: var(--warn-fg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0 0.3rem;
    font-size: 0.66rem;
  }
  .childtext {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: 0.86rem;
    line-height: 1.55;
    color: var(--fg);
    font-family: var(--font-serif);
  }
</style>
