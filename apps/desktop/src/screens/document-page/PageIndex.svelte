<script lang="ts">
  // The document page's index (ADR-063 choice 5): a pill per block — Metadata, Connections,
  // Chunks, Figures, References — that jumps to it, then *Source* (the source pane beside the
  // page) and *Re-run…* (the Re-run ingestion dialog) at the trailing end. It sits above the
  // document, outside its scrolling area. The list of blocks and the anchors it jumps to are
  // DocumentPage's (the block order); *Source* writes DocumentPage's split state, bound here.
  import type { LibraryDocument } from '../../lib/core/types'
  import Icon from '../../shell/Icon.svelte'
  import { rerunLabel } from '../../lib/library/reingest'
  import { openRerun } from '../../lib/library/reingest.svelte'

  let {
    blocks,
    docId,
    doc,
    viewerOpen = $bindable(),
    viewerPage = $bindable(),
  }: {
    /** The blocks in page order: each `id` is an element id on the page. */
    blocks: { id: string; label: string }[]
    /** The open document, and its row in the list: what *Re-run…* re-runs, and its name. */
    docId: string
    doc: LibraryDocument | null
    /** Whether the source pane is open; closing it forgets the page it was asked for. */
    viewerOpen: boolean
    viewerPage: number | null
  } = $props()

  // Instant, not smooth: with Chunks expanded the document view can be tens of thousands of
  // pixels tall, and smooth-scrolling that distance is a multi-second animation, not a jump.
  function jumpTo(id: string): void {
    document.getElementById(id)?.scrollIntoView({ block: 'start' })
  }
</script>

<!-- The index sits above the document and *outside* the scrolling area: as a sticky strip
     inside it, it passed over the text as the reader scrolled. Here it stays put under the
     breadcrumb band and never covers anything. -->
<nav class="blocknav" aria-label="Jump to a section of this document">
  {#each blocks as b (b.id)}
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
  <button class="rerun" type="button" onclick={() => openRerun([docId], rerunLabel(doc))} title="Re-run part of ingestion for this document">
    <Icon name="rotate-ccw" size={13} /> Re-run…
  </button>
</nav>

<style>
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
  /* The strip is what makes the fixed block order navigable rather than merely present. It sits
     outside `.scroller`, so it is always there and never passes over the text. */
  .blocknav {
    flex: none;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
    /* Equal room above and below: with none above, the pills sat on the header's rule. */
    padding: 0.45rem 0;
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
</style>
