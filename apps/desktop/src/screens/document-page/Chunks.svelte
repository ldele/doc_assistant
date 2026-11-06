<script lang="ts">
  // The document page's Chunks block (feature-library-browser.md L1): its heading and counts,
  // then the document's parent blocks, each opening to its text, *Show this page in the document*
  // and its child chunks. Owns the chunk payload, the block's open/closed memory and which parent
  // blocks are open. Its wrapper and `doc-chunks` anchor stay with the block order in
  // DocumentPage; the source pane it opens is DocumentPage's split, bound here.
  //
  // **Chunks is collapsed, and collapsed means not fetched.** That is a measured cost, not a
  // tidiness preference: the detail payload is a median 170 KB and up to 1.85 MB per document
  // on this corpus (663 parent blocks + 2,608 children for the largest), and the old view
  // rendered every parent's text plus every child node eagerly on open. Opening a document now
  // costs zero chunk bytes; the text arrives when the reader asks for it — and once they have
  // asked, `chunkmemory` keeps that document open for the session, so the top ← → arrows come
  // back to it as they left it.
  import { untrack } from 'svelte'
  import type { LibraryDocument, LibraryDocumentChunks } from '../../lib/core/types'
  import { getLibraryDocument, locateChunk } from '../../lib/core/api'
  import { blockPreview } from '../../lib/library/library'
  import { rememberChunksOpen, wereChunksOpen } from '../../lib/library/chunkmemory'
  import Icon from '../../shell/Icon.svelte'

  let {
    docId,
    doc,
    viewerOpen = $bindable(),
    viewerPage = $bindable(),
  }: {
    docId: string
    /** The open document's list summary: its chunk count recorded at ingest, shown until the
     *  payload is loaded. */
    doc: LibraryDocument | null
    /** DocumentPage's source pane, which *Show this page in the document* opens at a page. */
    viewerOpen: boolean
    viewerPage: number | null
  } = $props()

  // The chunk payload: null until the reader opens the Chunks block.
  let detail = $state<LibraryDocumentChunks | null>(null)
  let loading = $state(false)
  let error = $state<string | null>(null)
  let chunksOpen = $state(false)

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
</script>

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

<style>
  /* The block's heading. It was `.block h3` in DocumentPage, whose `.block` wrapper still holds
     this block; no other rule targets this heading, so the shorter selector looks the same. */
  h3 {
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
