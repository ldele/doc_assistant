<script lang="ts">
  // Document connections panel (ADR-027 D1 — E4 exploration surface). Renders one document's
  // pre-computed **neighbourhood**: related papers (doc_similarities) and the library documents
  // that cite it. What this paper itself cites is the References block at the foot of the view —
  // moved out 2026-08-10, because a reader asking "what does this cite" was being shown the
  // resolved half here and the unresolved half there.
  // Advisory + read-only: a load failure degrades to one quiet line, never blocking the doc
  // view. List-shaped v1 by design — the graph/navigation treatment is a recorded open gate
  // (E4 DEVLOG); a later iteration reads the same bundle.
  import type { DocConnections } from '../core/types'
  import { getDocConnections } from '../core/api'

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
    conn !== null && conn.related.length === 0 && conn.cited_by.length === 0,
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

    {/if}
  </section>
{/if}

<style>
  /* One block among five (Metadata → Connections → Chunks → Figures → References): same
     heading treatment and spacing as its siblings, so the order reads as structure. */
  .connections {
    margin-top: 1.25rem;
  }
  h3 {
    margin: 0 0 0.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--fg);
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
  .connerr {
    color: var(--warn-fg);
    font-size: 0.8rem;
    margin: 0 0 0.6rem;
  }
</style>
