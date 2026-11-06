<script lang="ts">
  // The Library pane's *Summary* layout (ROADMAP 112, docs/wireframes/library-summary.md, rev 2):
  // what went wrong at indexing and what is known about the documents, for the library or one folder.
  //
  // Renders `/api/library/summary`; every sentence and shape comes from `./summary.ts`, so the zero,
  // plural and "not read" rules are tested there, not here. A count with something behind it is
  // navigation (ADR-063): the whole row is the target, with a chevron, as the Graph's concept view
  // lists its documents. The only buttons are *Count again* (quiet) and, in the pane's bar, *Add
  // documents*. Nothing here changes the library: a row opens the fix, it never runs it.
  import Icon from '../../shell/Icon.svelte'
  import { openIndexing } from '../../lib/core/ingest.svelte'
  import { getCollectionSummary, listIssueDocuments } from '../../lib/core/api/summary'
  import type { CollectionSummary, SummaryIssue } from '../../lib/core/types/summary'
  import type { LibraryCollection } from '../../lib/library/library'
  import {
    contextLine,
    countedAt,
    fmt,
    foundRows,
    indexingRows,
    meterShare,
    metadataRows,
    tiles,
    typeBars,
    whatLine,
    yearBars,
    type Bar,
    type CardRow,
    type SummaryTarget,
  } from '../../lib/library/summary'
  import { tooltip } from '../../lib/shell/tooltip'

  interface Props {
    /** The open folder, or `null` for the whole library. */
    folderId: string | null
    /** Open the documents one count names, as a collection of the grid. */
    onOpenCollection: (c: LibraryCollection) => void
    /** The folder's note about files: open the library's own summary. */
    onOpenLibrary: () => void
    /** The files in the documents folder live in Settings > Documents. */
    onShowFiles: () => void
  }
  const { folderId, onOpenCollection, onOpenLibrary, onShowFiles }: Props = $props()

  let summary = $state<CollectionSummary | null>(null)
  let loading = $state(true)
  let error = $state<string | null>(null)
  let opening = $state<SummaryIssue | null>(null)
  let indexingHeading = $state<HTMLElement | null>(null)
  let request = 0

  async function load(id: string | null): Promise<void> {
    const mine = ++request
    loading = true
    error = null
    try {
      const s = await getCollectionSummary(id)
      if (mine !== request) return
      summary = s
      if (s === null) error = 'This folder is gone.'
    } catch (e) {
      if (mine !== request) return
      error = e instanceof Error ? e.message : String(e)
    } finally {
      if (mine === request) loading = false
    }
  }

  $effect(() => {
    void load(folderId)
  })

  async function show(issue: SummaryIssue): Promise<void> {
    opening = issue
    error = null
    try {
      const docs = await listIssueDocuments(issue, folderId)
      onOpenCollection({ kind: 'issue', value: issue, folder: folderId, ids: docs.map((d) => d.id) })
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      opening = null
    }
  }

  function go(target: SummaryTarget): void {
    if (target.kind === 'issue') void show(target.issue)
    else if (target.kind === 'files') onShowFiles()
    else if (target.kind === 'indexing') openIndexing()
    else onOpenLibrary()
  }

  function toIndexing(): void {
    indexingHeading?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const y = $derived(summary === null ? null : yearBars(summary))
  const metaRows = $derived(summary === null ? [] : metadataRows(summary))

  const title = $derived(
    summary === null
      ? 'Summary'
      : summary.scope === 'folder'
        ? `Summary of ${summary.folder_name ?? 'this folder'}`
        : 'Summary of your library',
  )
</script>

{#snippet bars(items: Bar[], noun: string)}
  <ul class="bars">
    {#each items as b (b.key)}
      <li title="{b.label}: {fmt(b.value)} {noun}">
        <span class="barlabel">{b.label}</span>
        <span class="barcell">
          <span class="barfill" style:width="calc((100% - 3.5rem) * {b.share})"></span>
          <span class="barvalue">{fmt(b.value)}</span>
        </span>
      </li>
    {/each}
  </ul>
{/snippet}

{#snippet cardRow(r: CardRow)}
  {#if r.kind === 'line'}
    <div class="notread">
      <strong>{r.title}</strong>
      <span>{r.detail}</span>
    </div>
  {:else}
    {@const issue = r.issue}
    {#snippet meterBody()}
      <span class="mlabel">{r.label}</span>
      <span class="mvalue">{fmt(r.have)} of {fmt(r.of)}</span>
      <span class="track" aria-hidden="true"
        ><span class="fill" style:width="{meterShare(r.have, r.of) * 100}%"></span></span
      >
    {/snippet}
    {#if issue !== null}
      <button
        class="meter nav"
        disabled={opening !== null}
        onclick={() => show(issue)}
        use:tooltip={`Open the ${r.tail}`}
        type="button"
      >
        {@render meterBody()}
        <span class="mtail">{r.tail}</span>
        <span class="chev"><Icon name="chevron-right" size={13} /></span>
      </button>
    {:else}
      <div class="meter">
        {@render meterBody()}
        <span class="mtail"></span>
        <span class="chev"></span>
      </div>
    {/if}
  {/if}
{/snippet}

<section class="summary" aria-busy={loading}>
  <div class="head">
    <div class="headtext">
      <h2>{title}</h2>
      {#if summary}
        <p class="context">{contextLine(summary)}</p>
        <p class="when">
          Counted at {countedAt(summary.counted_at)}. Nothing here changes your library.
        </p>
      {/if}
    </div>
    <button class="btn btn-quiet" disabled={loading} onclick={() => load(folderId)} type="button" use:tooltip={'Count the library again; nothing in it changes'}>
      <Icon name="rotate-ccw" size={13} /> Count again
    </button>
  </div>

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}

  {#if summary === null && loading}
    <p class="muted">Counting…</p>
  {:else if summary}
    {@const s = summary}
    {@const [need, docs] = tiles(s)}
    <div class="tiles">
      <!-- What needs action leads; it is navigation to its own rows, not an action. -->
      <button class="tile need" class:alert={need.alert} onclick={toIndexing} type="button" use:tooltip={'Jump to what needs doing, below'}>
        <span class="value">
          {#if need.alert}<Icon name="triangle-alert" size={18} />{/if}
          {need.value}
        </span>
        <span class="caption">{need.caption}</span>
      </button>
      <div class="tile">
        <span class="value">{docs.value}</span>
        <span class="caption">{docs.caption}</span>
      </div>
    </div>

    <h3 bind:this={indexingHeading}>From indexing</h3>
    <p class="sub">Each row: what happened, how many, and where to fix it. Nothing here runs a fix.</p>
    <ul class="rows">
      {#each indexingRows(s) as row (row.key)}
        {@const target = row.target}
        <li>
          {#if target !== null}
            <button
              class="navrow tone-{row.tone}"
              disabled={opening !== null}
              onclick={() => go(target)}
              type="button"
              use:tooltip={target.kind === 'issue' ? 'List these documents in the Library' : target.kind === 'files' ? 'Open Manage files in Settings' : target.kind === 'indexing' ? 'Open the indexing screen' : 'Show every document in the Library'}
            >
              {#if row.tone !== 'none'}
                <span class="mark" aria-label={row.tone === 'crit' ? 'Problem' : 'Warning'}
                  ><Icon name="triangle-alert" size={15} /></span
                >
              {/if}
              <span class="rowtext">
                <strong>{row.title}</strong>
                <span>{row.detail}</span>
                <span class="fix-under">{row.fix}</span>
              </span>
              <span class="fix">{row.fix}</span>
              <span class="chev"><Icon name="chevron-right" size={14} /></span>
            </button>
          {:else}
            <div class="plainrow">
              <span class="rowtext">
                <strong>{row.title}</strong>
                <span>{row.detail}</span>
              </span>
            </div>
          {/if}
        </li>
      {/each}
    </ul>

    {#if s.documents === 0}
      <p class="nothing">
        Nothing else to count: {s.scope === 'folder' ? 'this folder' : 'the library'} holds no documents.
        Metadata, what was found and the types appear once it does.
      </p>
    {:else}
      <div class="cards">
        <div class="card">
          <h3>Metadata</h3>
          {#if metaRows.length === 1 && metaRows[0].kind === 'line'}
            <!-- Nothing has read any field: one line says so; a sub and a footnote would repeat it. -->
            {@render cardRow(metaRows[0])}
          {:else}
            <p class="sub">Documents that have it, of {fmt(s.documents)}. A value you typed counts.</p>
            <div class="cardrows">
              {#each metaRows as r (r.key)}{@render cardRow(r)}{/each}
            </div>
            <p class="foot">
              Indexing does not read these. Re-run… &gt; Metadata reads them from the first page.
              {#if s.metadata.implausible_year > 0}
                {fmt(s.metadata.implausible_year)}
                {s.metadata.implausible_year === 1 ? 'year is' : 'years are'} set but cannot be a publication year.
              {/if}
            </p>
          {/if}
        </div>

        <div class="card">
          <h3>Found in each document</h3>
          <p class="sub">
            On record, of {fmt(s.documents)}{#if s.found.figures.pdfs > 0 && s.found.figures.pdfs !== s.documents}{' '}(figures
              and spacing: of {fmt(s.found.figures.pdfs)} PDFs){/if}.
          </p>
          <div class="cardrows">
            {#each foundRows(s) as r (r.key)}{@render cardRow(r)}{/each}
          </div>
          <p class="foot">None on record: none found, or never looked for. Re-run… looks.</p>
        </div>
      </div>

      <h3>What is in it</h3>
      <p class="sub">{whatLine(s)}</p>
      <div class="cards">
        <div class="card">
          <h3>Types</h3>
          {@render bars(typeBars(s), 'documents')}
        </div>
        <div class="card">
          <h3>Years</h3>
          {#if y === null || y.decades.length === 0}
            <!-- A lone "No year" bar would be a one-bar chart: a sentence says it. -->
            <p class="sub">No document has a year.</p>
          {:else}
            {@render bars(y.decades, 'documents')}
            {#if y.noYear}
              <div class="apart">{@render bars([y.noYear], 'documents')}</div>
            {/if}
          {/if}
          {#if s.metadata.implausible_year > 0}
            <p class="foot">
              {fmt(s.metadata.implausible_year)} with a year that cannot be one, counted in no decade.
            </p>
          {/if}
        </div>
      </div>
    {/if}
  {/if}
</section>

<style>
  .summary {
    container-type: inline-size;
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--space-4) var(--space-2) var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3);
    flex-wrap: wrap;
  }
  .headtext {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  h2 {
    margin: 0;
    font-size: var(--text-title);
  }
  h3 {
    margin: var(--space-3) 0 0;
    font-size: 1rem;
    scroll-margin-top: var(--space-3);
  }
  .card h3 {
    margin-top: 0;
  }
  .context {
    margin: 0;
    color: var(--fg);
  }
  .when,
  .sub,
  .muted {
    margin: 0;
    color: var(--fg-2);
    font-size: var(--text-sm);
  }
  .error {
    margin: 0;
    color: var(--danger);
    font-size: var(--text-sm);
  }
  .tiles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
  .tile {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.2rem;
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--fg);
    font: inherit;
    text-align: left;
  }
  button.tile {
    cursor: pointer;
  }
  button.tile:hover {
    background: var(--surface-2);
  }
  .tile.alert {
    background: var(--warn-bg);
    border-color: var(--warn-border);
  }
  .tile.alert:hover {
    background: color-mix(in srgb, var(--warn-bg) 80%, var(--fg) 6%);
  }
  .tile .value {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1.1;
  }
  .tile.alert .value {
    color: var(--warn-fg);
  }
  .tile .caption {
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  /* A count is navigation (ADR-063): the whole row is the target, as in the Graph's document list. */
  .navrow,
  .plainrow {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    width: 100%;
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--fg);
    font: inherit;
    text-align: left;
  }
  .navrow {
    cursor: pointer;
  }
  .navrow:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .plainrow {
    background: transparent;
    border-style: dashed;
  }
  .navrow.tone-warn {
    background: var(--warn-bg);
    border-color: var(--warn-border);
  }
  .navrow.tone-crit {
    background: color-mix(in srgb, var(--danger) 10%, var(--surface));
    border-color: color-mix(in srgb, var(--danger) 45%, var(--border));
  }
  .navrow.tone-warn:hover:not(:disabled),
  .navrow.tone-crit:hover:not(:disabled) {
    filter: brightness(0.97);
  }
  .mark {
    flex: none;
    display: inline-flex;
    color: var(--warn-fg);
  }
  .tone-crit .mark {
    color: var(--danger);
  }
  .rowtext {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
    flex: 1 1 auto;
  }
  .rowtext strong {
    font-weight: 600;
  }
  .rowtext span {
    font-size: var(--text-sm);
    color: var(--fg-2);
  }
  .fix {
    flex: none;
    max-width: 40%;
    font-size: var(--text-sm);
    color: var(--fg-2);
    text-align: right;
  }
  .chev {
    flex: none;
    display: inline-flex;
    color: var(--fg-2);
  }
  .fix-under {
    display: none;
  }
  /* Narrow (a phone, a split pane): the fix moves under the reason so the reason keeps its width. */
  @container (max-width: 34rem) {
    .fix {
      display: none;
    }
    .rowtext .fix-under {
      display: block;
      margin-top: 0.15rem;
      color: var(--fg);
    }
  }
  .nothing {
    margin: var(--space-2) 0 0;
    color: var(--fg-2);
  }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr));
    gap: var(--space-3);
    margin-top: var(--space-1);
  }
  .card {
    container-type: inline-size;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-3);
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    min-width: 0;
  }
  .cardrows {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .meter {
    display: grid;
    grid-template-columns: 6rem 6.5rem minmax(3rem, 1fr) auto 14px;
    grid-template-areas: 'label value track tail chev';
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    padding: 0.4rem 0.6rem;
    border: 1px solid transparent;
    border-radius: 8px;
    background: none;
    color: var(--fg);
    font: inherit;
    text-align: left;
  }
  .meter.nav {
    cursor: pointer;
    border-color: var(--border);
  }
  .meter.nav:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .mlabel {
    grid-area: label;
  }
  .mvalue {
    grid-area: value;
  }
  .mtail {
    grid-area: tail;
  }
  .meter .chev {
    grid-area: chev;
  }
  .track {
    grid-area: track;
  }
  /* A narrow card puts the bar on its own line under the words, rather than clipping them. */
  @container (max-width: 24rem) {
    .meter {
      grid-template-columns: 1fr auto auto 14px;
      grid-template-areas:
        'label value tail chev'
        'track track track track';
      row-gap: 0.35rem;
    }
  }
  .mvalue,
  .mtail {
    font-size: var(--text-sm);
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  /* A thin meter: the share that has it; fully rounded, no text on it. */
  .track {
    height: 6px;
    border-radius: 999px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .fill {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: var(--accent);
  }
  .notread {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: var(--space-2) var(--space-3);
    border-radius: 8px;
    background: var(--surface-2);
  }
  .notread span {
    font-size: var(--text-sm);
    color: var(--fg-2);
  }
  .foot {
    margin: 0;
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  /* Bars: horizontal, one hue, value at the tip; square at the baseline, a 4px rounded data end. */
  .bars {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .bars li {
    display: grid;
    grid-template-columns: 4.5rem 1fr;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-sm);
  }
  .barlabel {
    color: var(--fg-2);
  }
  .barcell {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    min-width: 0;
  }
  .barfill {
    display: block;
    height: 0.75rem;
    min-width: 2px;
    border-radius: 0 4px 4px 0;
    background: var(--accent);
  }
  .barvalue {
    color: var(--fg);
    font-variant-numeric: tabular-nums;
  }
  .apart {
    margin-top: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px dashed var(--border);
  }
</style>
