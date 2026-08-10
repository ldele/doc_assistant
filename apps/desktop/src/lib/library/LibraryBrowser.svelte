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
  import { getLibraryDocument } from '../core/api'
  import { rememberChunksOpen, wereChunksOpen } from './chunkmemory'
  import DocConnections from './DocConnections.svelte'
  import DocFigures from './DocFigures.svelte'
  import DocReferences from './DocReferences.svelte'

  let {
    docId,
    doc,
    onOpenDocument,
  }: {
    docId: string | null
    /** The open document's list summary — the metadata block's source. */
    doc: LibraryDocument | null
    onOpenDocument?: (id: string) => void
  } = $props()

  // The chunk payload: null until the reader opens the Chunks block.
  let detail = $state<LibraryDocumentChunks | null>(null)
  let loading = $state(false)
  let error = $state<string | null>(null)
  let chunksOpen = $state(false)

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

  // The nav strip and the block order are the same list — a block cannot be added to one and
  // forgotten in the other.
  const BLOCKS = [
    { id: 'doc-metadata', label: 'Metadata' },
    { id: 'doc-connections', label: 'Connections' },
    { id: 'doc-chunks', label: 'Chunks' },
    { id: 'doc-figures', label: 'Figures' },
    { id: 'doc-references', label: 'References' },
  ]

  // Instant, not smooth: with Chunks expanded the document view is ~76,000 px tall (measured
  // on a 142-block paper), and smooth-scrolling that distance is a multi-second animation
  // rather than a jump.
  function jumpTo(id: string): void {
    document.getElementById(id)?.scrollIntoView({ block: 'start' })
  }
</script>

<section class="browser">
  {#if !docId}
    <p class="hint">Select a document from the sidebar to read it.</p>
  {:else}
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

    <nav class="blocknav" aria-label="Jump to a section of this document">
      {#each BLOCKS as b (b.id)}
        <button type="button" onclick={() => jumpTo(b.id)}>{b.label}</button>
      {/each}
    </nav>

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
          {#each detail.parents as p (p.parent_index)}
            <article class="parent">
              <div class="phead">Block {p.parent_index}</div>
              <p class="blocktext">{p.parent_text}</p>
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
  {/if}
</section>

<style>
  .browser {
    flex: 1;
    overflow-y: auto;
    padding: 0.8rem 0;
    min-width: 0;
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
  /* The strip is what makes the fixed block order navigable rather than merely present. */
  .blocknav {
    position: sticky;
    top: 0;
    z-index: 2;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    padding: 0.35rem 0;
    margin-bottom: 0.2rem;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
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
    /* Clears the sticky nav when jumped to, so a block's heading is never hidden under it.
       Measured, not guessed: at 2.4rem the heading landed ~9 px *under* the 33 px strip. */
    scroll-margin-top: 3.6rem;
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
  .parent {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 0.6rem 0.75rem;
    margin-bottom: 0.6rem;
  }
  .phead {
    font-size: 0.72rem;
    color: var(--fg-2);
    font-weight: 600;
    margin-bottom: 0.35rem;
  }
  .blocktext {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: 0.95rem;
    line-height: 1.6;
    font-family: var(--font-serif);
  }
  .children {
    margin-top: 0.5rem;
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
