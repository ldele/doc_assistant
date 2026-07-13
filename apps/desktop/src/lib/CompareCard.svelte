<script lang="ts">
  // A/B-compare result card (feature-ab-compare-sandbox.md, U6). Renders the two retrieved source
  // sets side by side — A = locked defaults, B = the session override — with per-source diff badges
  // (in both / only-A / only-B, + rank delta) and the honest note. Retrieval-only, $0. Read-only.
  import type { CompareResult, CompareRow, CompareSource } from './types'

  let { result, onClose }: { result: CompareResult; onClose: () => void } = $props()

  // identity -> its diff row, for per-source badges.
  const rowById = $derived(new Map<string, CompareRow>(result.rows.map((r) => [r.identity, r])))

  function badge(s: CompareSource, side: 'a' | 'b'): { label: string; kind: string } {
    const row = rowById.get(s.identity)
    if (!row || row.status === 'in_both') {
      // in both sides; show the rank delta when this source moved (rank_delta = a_rank - b_rank).
      const d = row?.rank_delta ?? 0
      if (d === 0) return { label: 'both', kind: 'both' }
      // Positive delta => B ranks it higher (smaller rank). Describe the move from this side's view.
      const up = side === 'b' ? d > 0 : d < 0
      return { label: `both ${up ? '↑' : '↓'}${Math.abs(d)}`, kind: 'both' }
    }
    return row.status === 'only_in_a'
      ? { label: 'only A', kind: side === 'a' ? 'only' : 'gone' }
      : { label: 'only B', kind: side === 'b' ? 'only' : 'gone' }
  }
</script>

<div class="card">
  <div class="head">
    <div>
      <strong>A/B compare — retrieval</strong>
      <span class="q">“{result.query}”</span>
    </div>
    <button class="x" onclick={onClose} aria-label="Close comparison" type="button">✕</button>
  </div>

  {#if result.note}
    <p class="note">{result.note}</p>
  {/if}

  <div class="cols">
    {#each [{ side: 'a', title: 'Locked defaults', eff: result.eff_a, sources: result.sources_a }, { side: 'b', title: 'Session override', eff: result.eff_b, sources: result.sources_b }] as col (col.side)}
      <section class="col">
        <header class="colhead">
          <span class="coltitle">{col.title}</span>
          <span class="eff">top_k {col.eff.top_k} · multi-query {col.eff.use_multi_query ? 'on' : 'off'}</span>
        </header>
        {#if col.sources.length === 0}
          <p class="empty">No sources.</p>
        {:else}
          {#each col.sources as s (s.identity)}
            {@const b = badge(s, col.side as 'a' | 'b')}
            <div class="src">
              <div class="srctop">
                <span class="cite">{s.citation}</span>
                <span class="badge {b.kind}">{b.label}</span>
              </div>
              <div class="srcmeta">score {s.score.toFixed(3)}</div>
              <p class="excerpt">{s.excerpt}</p>
            </div>
          {/each}
        {/if}
      </section>
    {/each}
  </div>

  <p class="foot">
    Indicative on one query — <strong>not a verdict</strong>. Only <code>top_k</code> /
    <code>multi-query</code> change retrieval; the eval harness is the only path to a new default.
  </p>
</div>

<style>
  .card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    padding: 0.8rem 0.9rem;
    margin: 0.6rem 0;
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
  }
  .q {
    color: var(--fg-2);
    font-size: 0.85rem;
    margin-left: 0.4rem;
  }
  .x {
    font: inherit;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg-2);
    border-radius: 6px;
    padding: 0.1rem 0.4rem;
  }
  .note {
    margin: 0.5rem 0 0;
    font-size: 0.82rem;
    color: var(--fg-2);
    background: var(--surface-2);
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
  }
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-top: 0.6rem;
  }
  .col {
    min-width: 0;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.55rem;
    background: var(--bg);
  }
  .colhead {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.35rem;
    margin-bottom: 0.4rem;
  }
  .coltitle {
    font-weight: 600;
    font-size: 0.85rem;
  }
  .eff {
    font-size: 0.7rem;
    color: var(--fg-2);
  }
  .empty {
    color: var(--fg-2);
    font-size: 0.8rem;
  }
  .src {
    padding: 0.35rem 0;
    border-bottom: 1px dashed var(--border);
  }
  .src:last-child {
    border-bottom: none;
  }
  .srctop {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.4rem;
  }
  .cite {
    font-size: 0.78rem;
    overflow-wrap: anywhere;
  }
  .srcmeta {
    font-size: 0.68rem;
    color: var(--fg-2);
    margin-top: 0.1rem;
  }
  .excerpt {
    margin: 0.2rem 0 0;
    font-size: 0.74rem;
    color: var(--fg-2);
    line-height: 1.4;
    overflow-wrap: anywhere;
    max-height: 3.6em;
    overflow: hidden;
  }
  .badge {
    font-size: 0.66rem;
    border-radius: 6px;
    padding: 0 0.32rem;
    border: 1px solid var(--border);
    white-space: nowrap;
    flex-shrink: 0;
  }
  .badge.both {
    color: var(--fg-2);
  }
  .badge.only {
    color: var(--accent);
    border-color: var(--accent);
  }
  .badge.gone {
    color: var(--fg-2);
    opacity: 0.6;
    text-decoration: line-through;
  }
  .foot {
    margin: 0.6rem 0 0;
    font-size: 0.72rem;
    color: var(--fg-2);
  }
  .foot code {
    font-size: 0.9em;
  }
  @media (max-width: 640px) {
    .cols {
      grid-template-columns: 1fr;
    }
  }
</style>
