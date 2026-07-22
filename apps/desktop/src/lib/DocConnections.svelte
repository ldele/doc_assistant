<script lang="ts">
  // Document connections panel (ADR-027 D1 — E4 exploration surface). Renders one document's
  // pre-computed neighbourhood: related papers (doc_similarities), resolved in-corpus citation
  // edges both directions, and the extracted-but-unresolved external references (collapsed).
  // Advisory + read-only: a load failure degrades to one quiet line, never blocking the doc
  // view. List-shaped v1 by design — the graph/navigation treatment is a recorded open gate
  // (E4 DEVLOG); a later iteration reads the same bundle.
  import type { DocConnections } from './types'
  import { getDocConnections } from './api'

  let {
    docId,
    onOpenDocument,
  }: { docId: string; onOpenDocument?: (id: string) => void } = $props()

  let conn = $state<DocConnections | null>(null)
  let error = $state<string | null>(null)

  // Last-write-wins token, mirroring LibraryBrowser's own load guard.
  let token = 0
  $effect(() => {
    const id = docId
    conn = null
    error = null
    if (!id) return
    const mine = ++token
    void (async () => {
      try {
        const c = await getDocConnections(id)
        if (mine === token) conn = c
      } catch (e) {
        if (mine === token) error = String(e)
      }
    })()
  })

  const empty = $derived(
    conn !== null &&
      conn.related.length === 0 &&
      conn.cites.length === 0 &&
      conn.cited_by.length === 0 &&
      conn.external_total === 0,
  )

  function open(id: string): void {
    onOpenDocument?.(id)
  }
</script>

{#if error}
  <p class="connerr">Couldn’t load connections: {error}</p>
{:else if conn}
  <section class="connections" aria-label="Document connections">
    <h3>Connections</h3>
    {#if empty}
      <p class="muted">No connections computed for this document yet.</p>
    {:else}
      {#if conn.related.length > 0}
        <h4>Related papers <span class="muted">(semantic similarity)</span></h4>
        <ul>
          {#each conn.related as r (r.document_id)}
            <li>
              <button class="doclink" onclick={() => open(r.document_id)}>
                {r.title ?? r.filename}
              </button>
              <span class="score" title="cosine similarity">{r.score.toFixed(2)}</span>
            </li>
          {/each}
        </ul>
      {/if}

      {#if conn.cites.length > 0}
        <h4>Cites <span class="muted">(in your library)</span></h4>
        <ul>
          {#each conn.cites as c, i (c.document_id + '-' + i)}
            <li>
              <button class="doclink" onclick={() => open(c.document_id)}>
                {c.title ?? c.filename ?? c.document_id.slice(0, 8)}
              </button>
              {#if c.year != null}<span class="muted">{c.year}</span>{/if}
            </li>
          {/each}
        </ul>
      {/if}

      {#if conn.cited_by.length > 0}
        <h4>Cited by <span class="muted">(in your library)</span></h4>
        <ul>
          {#each conn.cited_by as d (d.document_id)}
            <li>
              <button class="doclink" onclick={() => open(d.document_id)}>{d.filename}</button>
              {#if d.n_citations > 1}<span class="score">×{d.n_citations}</span>{/if}
            </li>
          {/each}
        </ul>
      {/if}

      {#if conn.external_total > 0}
        <details class="external">
          <summary>References ({conn.external_total} extracted, not in your library)</summary>
          <ul>
            {#each conn.external_refs as e, i (i)}
              <li class="ref">
                <span class="reftitle">{e.title}</span>
                {#if e.authors}<span class="muted"> — {e.authors}</span>{/if}
                {#if e.year != null}<span class="muted"> ({e.year})</span>{/if}
              </li>
            {/each}
          </ul>
          {#if conn.external_refs.length < conn.external_total}
            <p class="muted capnote">
              Showing the first {conn.external_refs.length} of {conn.external_total}.
            </p>
          {/if}
        </details>
      {/if}
    {/if}
  </section>
{/if}

<style>
  .connections {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 0.6rem 0.75rem;
    margin-bottom: 0.6rem;
  }
  h3 {
    margin: 0 0 0.2rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--fg-2);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  h4 {
    margin: 0.55rem 0 0.2rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--fg);
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.76rem;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  li {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    padding: 0.12rem 0;
    min-width: 0;
  }
  .doclink {
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    font-size: 0.85rem;
    color: var(--accent);
    cursor: pointer;
    text-align: left;
    overflow-wrap: anywhere;
    min-width: 0;
  }
  .doclink:hover {
    text-decoration: underline;
  }
  .score {
    font-size: 0.72rem;
    color: var(--fg-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0 0.3rem;
    white-space: nowrap;
  }
  .external {
    margin-top: 0.55rem;
    border-top: 1px dashed var(--border);
    padding-top: 0.4rem;
  }
  .external summary {
    cursor: pointer;
    font-size: 0.78rem;
    color: var(--accent);
    user-select: none;
  }
  .ref {
    display: block;
    font-size: 0.8rem;
    line-height: 1.45;
    overflow-wrap: anywhere;
  }
  .reftitle {
    color: var(--fg);
  }
  .capnote {
    margin: 0.3rem 0 0;
  }
  .connerr {
    color: var(--warn-fg);
    font-size: 0.8rem;
    margin: 0 0 0.6rem;
  }
</style>
