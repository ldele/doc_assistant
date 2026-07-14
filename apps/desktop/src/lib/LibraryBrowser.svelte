<script lang="ts">
  // Library browser main pane (feature-library-browser.md, L1). Given a selected document id,
  // fetches its chunks and renders them the way the two-tier retriever stores them: parent blocks
  // (the parent_text the LLM reads), each inline-expandable to its embedded child chunks. Read-only,
  // no model. NULL metadata (title/authors/year, health) is omitted, never shown blank.
  import type { LibraryDocumentChunks } from './types'
  import { getLibraryDocument } from './api'

  let { docId }: { docId: string | null } = $props()

  let detail = $state<LibraryDocumentChunks | null>(null)
  let loading = $state(false)
  let error = $state<string | null>(null)

  // Load whenever the selected doc changes. A token guards against a slow response for a doc the
  // user has already navigated away from (last-write-wins).
  let token = 0
  $effect(() => {
    const id = docId
    detail = null
    error = null
    if (!id) return
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
  })
</script>

<section class="browser">
  {#if !docId}
    <p class="hint">Select a document from the sidebar to read its chunks.</p>
  {:else if loading}
    <p class="hint">Loading…</p>
  {:else if error}
    <p class="hint err">Couldn’t load this document: {error}</p>
  {:else if detail}
    <header class="dochead">
      <h2>{detail.title ?? detail.filename}</h2>
      <p class="metaline">
        {#if detail.title}<span>{detail.filename}</span><span>·</span>{/if}
        <span>{detail.format}</span>
        {#if detail.chunk_count != null}<span>· {detail.chunk_count.toLocaleString()} chunks</span>{/if}
        {#if detail.health}<span>· {detail.health}</span>{/if}
      </p>
      {#if detail.authors}<p class="metaextra"><strong>Authors</strong> {detail.authors}</p>{/if}
      {#if detail.year != null}<p class="metaextra"><strong>Year</strong> {detail.year}</p>{/if}
    </header>

    {#if detail.parents.length === 0}
      <p class="hint">No chunks stored for this document.</p>
    {:else}
      <p class="count">
        {detail.parents.length} parent block{detail.parents.length === 1 ? '' : 's'} ·
        {detail.child_count} child chunk{detail.child_count === 1 ? '' : 's'}
      </p>
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
  .hint.err {
    color: var(--warn-fg);
  }
  .dochead {
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin-bottom: 0.6rem;
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
  .count {
    font-size: 0.76rem;
    color: var(--fg-2);
    margin: 0 0 0.6rem;
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
