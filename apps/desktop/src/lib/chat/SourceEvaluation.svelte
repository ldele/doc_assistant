<script lang="ts">
  // ADR-027 D3 — the always-on source-evaluation strip below a chat answer. Advisory assessment
  // only (never gates the answer; independent of the D2/E3 influence toggle). One compact row per
  // retrieved source: its epistemic coverage, a superseded flag, the doc year, and the rerank
  // score — plus a footer stating which concept-graph build it was computed from (and whether that
  // build is stale). Renders nothing when `summary` is null (no epistemics sidecar / 0 docs).
  import type { SourceEvalSummary, SourceView } from '../core/types'
  import Icon from '../shell/Icon.svelte'

  let { sources, summary }: { sources: SourceView[]; summary: SourceEvalSummary } = $props()

  const COVERAGE_LABEL: Record<string, string> = {
    corroborated: 'corroborated',
    unique: 'single-source',
    contested: 'contested',
  }
  const shortVersion = $derived(summary.graph_version ? summary.graph_version.slice(0, 8) : '—')
</script>

<details class="eval" open>
  <summary>
    Source evaluation
    {#if summary.stale}
      <span class="stale" title="The concept graph was rebuilt after this assessment was computed.">
        <Icon name="triangle-alert" size={11} /> stale
      </span>
    {/if}
  </summary>

  <!-- The score column had no label, sitting under a heading that reads "Source evaluation" — so a
       bare 0.94 next to an epistemic assessment invited "is this a quality/review score?" (user
       report 2026-08-05). It is not: it is the cross-encoder reranker's relevance of that CHUNK to
       THIS question, the same number the retrieval ranking used. Nothing in the app scores a
       source's quality. Label the column rather than relying on a hover title. -->
  <p class="legend">
    <span class="legend-what">relevance</span> = how well the retrieved passage matches your
    question (reranker score, 0–1). Not a judgement of the source.
  </p>

  <ul class="rows">
    {#each sources as s (s.n)}
      <li>
        <span class="n">[{s.n}]</span>
        <!-- KI-33 / ADR-041: the coverage + superseded chips are WITHHELD, not deleted. Both are
             derived from Node-B stance, which is judged without the document in the prompt and
             whose verdict moves with a pair's index in a generated list — so `contested`,
             `corroborated` and `unique` alike encode list position, not the corpus. Restore the
             two blocks below once Node B is rebuilt on evidence (ADR-041 option 1); the wire
             field and the read model are deliberately left intact so nothing else has to change.
        {#if s.evaluation?.coverage}
          <span class="cov cov-{s.evaluation.coverage}">{COVERAGE_LABEL[s.evaluation.coverage]}</span>
        {:else}
          <span class="cov cov-none">not assessed</span>
        {/if}
        {#if s.evaluation?.superseded}
          <span class="sup" title="Newer sources dispute this — a superseded trend.">superseded</span>
        {/if}
        -->
        <span class="cov cov-none" title="Per-source epistemic assessment is withheld pending a rebuild of the stance extractor (KI-33).">assessment withheld</span>
        {#if s.evaluation?.year != null}
          <span class="year">{s.evaluation.year}</span>
        {/if}
        <span
          class="score"
          title="Relevance of this passage to your question (cross-encoder reranker score, 0–1). Not a quality or review score."
        >
          <span class="score-label">relevance</span>{s.reranker_score.toFixed(2)}
        </span>
      </li>
    {/each}
  </ul>

  <p class="foot" title={`Concept-graph build: ${summary.graph_version ?? 'none'}`}>
    {#if summary.stale}
      Assessed from an earlier graph ({shortVersion}) — rebuild the concept graph to refresh.
    {:else}
      Assessed as of graph {shortVersion}.
    {/if}
  </p>
</details>

<style>
  .eval {
    margin-top: 0.7rem;
    font-size: 0.76rem;
  }
  .eval > summary {
    cursor: pointer;
    color: var(--fg-2);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.15rem 0;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .eval > summary:hover {
    color: var(--fg);
  }
  .stale {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    color: var(--warn-fg);
    text-transform: none;
    letter-spacing: 0;
  }
  .rows {
    list-style: none;
    margin: 0.4rem 0 0;
    padding: 0;
    display: grid;
    gap: 0.2rem;
  }
  .rows li {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    min-width: 0;
    flex-wrap: wrap;
  }
  .n {
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    flex: none;
  }
  .cov {
    padding: 0.02rem 0.4rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    font-size: 0.72rem;
    white-space: nowrap;
    flex: none;
  }
  /* KI-33 / ADR-041: withheld alongside their markup above — restore both blocks together.
  .cov-contested {
    color: var(--warn-fg);
    border-color: var(--warn-border);
    background: var(--warn-bg);
  }
  .cov-corroborated {
    color: var(--ok-fg);
    border-color: var(--ok-border);
  }
  .cov-unique {
    color: var(--fg-2);
  }
  .sup {
    color: var(--danger);
    font-size: 0.72rem;
    flex: none;
  }
  */
  .cov-none {
    color: var(--fg-2);
    opacity: 0.7;
    font-style: italic;
  }
  .year {
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    flex: none;
  }
  .legend {
    margin: 0.35rem 0 0;
    color: var(--fg-2);
    font-size: 0.7rem;
    line-height: 1.35;
    opacity: 0.85;
  }
  .legend-what {
    font-variant: small-caps;
    letter-spacing: 0.03em;
    color: var(--fg);
  }
  .score {
    margin-left: auto;
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    flex: none;
    display: inline-flex;
    align-items: baseline;
    gap: 0.3rem;
  }
  /* Small-caps unit beside the number, so a lone decimal is never ambiguous even when the legend
     above has scrolled out of view. */
  .score-label {
    font-variant: small-caps;
    font-size: 0.68rem;
    letter-spacing: 0.03em;
    opacity: 0.65;
  }
  .foot {
    margin: 0.45rem 0 0;
    color: var(--fg-2);
    font-size: 0.7rem;
    line-height: 1.35;
  }
</style>
