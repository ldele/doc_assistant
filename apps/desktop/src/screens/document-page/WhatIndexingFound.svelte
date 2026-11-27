<script lang="ts">
  // What indexing found (0.9.0, docs/specs/ingestion-record.md; docs/wireframes/extraction-record.md
  // S2 and S3, approved as drawn): the document's extraction record, under the header. Warnings
  // first, each with the fix beside it, then one line per pass. It informs: every sentence is the
  // record's, and a button opens a fix (Re-run… with that part ticked, or Edit metadata), never
  // runs one. Shape and wording live in `lib/library/extractionRecord.ts`, which is tested.
  import { untrack } from 'svelte'
  import type { DocumentExtractionRecord, LibraryDocument, RecordWarning } from '../../lib/core/types'
  import { getExtractionRecord } from '../../lib/core/api'
  import { factLines, fixFor, recordView, whenLine } from '../../lib/library/extractionRecord'
  import { openEditMetadata } from '../../lib/library/documents.svelte'
  import { rerunLabel } from '../../lib/library/reingest'
  import { openRerun } from '../../lib/library/reingest.svelte'
  import { tooltip } from '../../lib/shell/tooltip'
  import Icon from '../../shell/Icon.svelte'

  let {
    docId,
    listed,
  }: {
    docId: string
    /** This document's row in the library list. A new one means the list was refetched, which a
     *  finished re-run does, so the record may have changed. */
    listed: LibraryDocument | null
  } = $props()

  let data = $state<DocumentExtractionRecord | null | undefined>(undefined)
  let failed = $state(false)
  const view = $derived(recordView(data))

  let token = 0
  $effect(() => {
    const id = docId
    void listed
    untrack(() => {
      failed = false
      if (data?.document_id !== id) data = undefined
    })
    if (!id) return
    const mine = ++token
    void (async () => {
      try {
        const d = await getExtractionRecord(id)
        if (mine === token) data = d
      } catch {
        if (mine === token) failed = true
      }
    })()
  })

  function rerun(parts: string[] = []): void {
    openRerun([docId], rerunLabel(listed), parts)
  }

  function fix(w: RecordWarning): void {
    const f = fixFor(w)
    if (f?.kind === 'rerun') rerun(f.parts)
    else if (f?.kind === 'metadata') openEditMetadata(docId)
  }
</script>

<section class="found" aria-labelledby="found-h">
  <h3 id="found-h">What indexing found</h3>
  {#if failed}
    <p class="muted lead">What indexing found is unavailable right now.</p>
  {:else if view.kind === 'loading'}
    <p class="muted lead">Reading…</p>
  {:else if view.kind === 'never' || view.kind === 'stale'}
    <div class="nothing">
      <p class="lead">
        {#if view.kind === 'never'}
          Not looked for yet. This document was indexed before indexing read figures, references and tables.
        {:else}
          Recorded for an earlier reading of this file ({view.when}). Re-reading the text brings it up to date.
        {/if}
      </p>
      <button class="btn" type="button" onclick={() => rerun(view.kind === 'stale' ? ['text'] : [])} use:tooltip={'Choose what to read again for this document'}>Re-run…</button>
    </div>
  {:else}
    {@const r = view.record}
    <p class="muted when">{whenLine(r)}</p>
    {#if r.warnings.length}
      <ul class="warnings">
        {#each r.warnings as w (w.id)}
          {@const f = fixFor(w)}
          <li>
            <span class="wtext"><Icon name="triangle-alert" size={13} /> {w.text}</span>
            {#if f?.kind === 'rerun'}
              <button class="btn" type="button" onclick={() => fix(w)} use:tooltip={'Open Re-run… with this part ticked'}>Re-run…</button>
            {:else if f?.kind === 'metadata'}
              <button class="btn" type="button" onclick={() => fix(w)} use:tooltip={'Fill in the missing details yourself'}>Edit metadata</button>
            {/if}
          </li>
        {/each}
      </ul>
    {/if}
    <dl class="facts">
      {#each factLines(r) as line (line.label)}
        <dt>{line.label}</dt>
        <dd>{line.text}</dd>
      {/each}
    </dl>
  {/if}
</section>

<style>
  .found {
    margin-top: var(--space-3);
    padding: 0.6rem 0.85rem;
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  h3 {
    margin: 0 0 0.25rem;
    font-size: 0.9rem;
  }
  .lead,
  .when {
    margin: 0.2rem 0 0.5rem;
    font-size: 0.8rem;
    line-height: 1.45;
  }
  .nothing {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .warnings {
    list-style: none;
    margin: 0 0 0.6rem;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .warnings li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    padding: 0.35rem 0.6rem;
    border: 1px solid var(--warn-border);
    border-radius: 8px;
    background: var(--warn-bg);
  }
  .wtext {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--warn-fg);
    font-size: 0.85rem;
  }
  .facts {
    display: grid;
    grid-template-columns: 6.5rem 1fr;
    gap: 0.3rem var(--space-3);
    margin: 0;
    font-size: 0.85rem;
  }
  .facts dt {
    color: var(--fg-2);
  }
  .facts dd {
    margin: 0;
  }
</style>
