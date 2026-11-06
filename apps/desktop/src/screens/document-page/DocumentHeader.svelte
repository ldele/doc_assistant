<script lang="ts">
  // The document page's header (feature-library-browser.md L1): the document's title and its
  // metadata (file, format, pages, health, authors, year, keywords). It is the block the page's
  // index calls *Metadata*, and carries that block's `doc-metadata` anchor; the other anchors stay
  // with the block order in DocumentPage.
  //
  // It reads the document summary the Library list already holds (`doc`) rather than the chunk
  // payload — the same fields, without the text, so opening a document fetches no chunk bytes.
  import type { LibraryDocument } from '../../lib/core/types'

  let {
    doc,
  }: {
    /** The open document's list summary; null when it is not in the loaded library list. */
    doc: LibraryDocument | null
  } = $props()
</script>

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

<style>
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
</style>
