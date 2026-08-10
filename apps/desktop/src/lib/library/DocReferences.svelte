<script lang="ts">
  // The References block — the paper's own bibliography, at the bottom of the document view
  // (user request 2026-08-09). Distinct from Connections above it: Connections is what this
  // document is *near* (semantic neighbours) and who cites it; this is what it *cites*.
  //
  // Two rules the panel exists to honour:
  //  1. Every extracted reference is listed, including the ones that parsed no title and the
  //     ones that matched nothing — a bibliography with its failures removed would misstate
  //     what the paper cites.
  //  2. A reference that resolved to a document in the library is a link that opens it. On
  //     this corpus that fires rarely (16 of 4,374 rows), so the header states the count
  //     instead of leaving the reader to guess whether linking works at all.
  //
  // Read-only and advisory: a load failure degrades to one quiet line, never blocking the view.
  import type { DocReferences } from '../core/types'
  import { getDocumentReferences } from '../core/api'
  import { referenceLabel } from './library'

  let {
    docId,
    onOpenDocument,
  }: { docId: string; onOpenDocument?: (id: string) => void } = $props()

  let data = $state<DocReferences | null>(null)
  let error = $state<string | null>(null)

  // Last-write-wins token, mirroring DocConnections/DocFigures' own load guards.
  let token = 0
  $effect(() => {
    const id = docId
    data = null
    error = null
    if (!id) return
    const mine = ++token
    void (async () => {
      try {
        const d = await getDocumentReferences(id)
        if (mine === token) data = d
      } catch (e) {
        if (mine === token) error = String(e)
      }
    })()
  })
</script>

<section class="refs" aria-label="References">
  <h3>
    References
    {#if data && data.total > 0}
      <span class="count">
        {data.total}
        {#if data.in_library > 0}
          · {data.in_library} in your library
        {/if}
      </span>
    {/if}
  </h3>

  {#if error}
    <p class="quiet">References unavailable.</p>
  {:else if data === null}
    <p class="quiet">Loading…</p>
  {:else if data.total === 0}
    <!-- Honest empty: no bibliography was extracted from this document. Ordinary for a book,
         a scan, or any format whose references section the regex tier does not find. -->
    <p class="quiet">No references extracted from this document.</p>
  {:else}
    <ol class="list">
      {#each data.references as ref, i (i)}
        <li class:linked={ref.document_id !== null} title={ref.raw_text ?? ''}>
          {#if ref.document_id !== null}
            <button
              class="doclink"
              title="Open this document in your library"
              onclick={() => onOpenDocument?.(ref.document_id ?? '')}
            >
              {referenceLabel(ref) ?? ref.filename}
            </button>
            <span class="owned">in library</span>
          {:else}
            <!-- Clamped, not truncated: the regex regularly swallows an editor list into the
                 title ("… In Meeyoung Cha, … editors, Proceedings of the 34th ACM …"), which
                 turns one bibliography entry into a paragraph. The full line is the row's
                 tooltip. Same treatment as a figure caption. -->
            <span class="reftext">{referenceLabel(ref) ?? 'Unparsed reference'}</span>
          {/if}
          {#if ref.year != null}<span class="muted">({ref.year})</span>{/if}
          {#if ref.doi}<span class="muted doi">{ref.doi}</span>{/if}
        </li>
      {/each}
    </ol>
    {#if data.shown < data.total}
      <!-- No silent truncation (and the cap never drops a linked reference). -->
      <p class="quiet capnote">Showing the first {data.shown} of {data.total}.</p>
    {/if}
    <p class="quiet ordernote">
      Extracted from the document's reference section, newest first — the paper's own numbering
      isn't recorded.
    </p>
  {/if}
</section>

<style>
  .refs {
    margin-top: 1.25rem;
  }
  h3 {
    font-size: 0.9rem;
    margin: 0 0 0.5rem;
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
  .list {
    list-style: decimal;
    margin: 0;
    padding-left: 1.6rem;
  }
  li {
    font-size: 0.82rem;
    line-height: 1.5;
    padding: 0.14rem 0;
    overflow-wrap: anywhere;
  }
  /* A reference the user owns reads as actionable, not as one more bibliography line. */
  li.linked::marker {
    color: var(--accent);
  }
  .reftext {
    color: var(--fg);
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .doclink {
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    color: var(--accent);
    cursor: pointer;
    text-align: left;
    overflow-wrap: anywhere;
  }
  .doclink:hover {
    text-decoration: underline;
  }
  .owned {
    font-size: 0.68rem;
    color: var(--fg-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0 0.3rem;
    margin-left: 0.3rem;
    white-space: nowrap;
  }
  .muted {
    color: var(--fg-2);
    margin-left: 0.3rem;
  }
  .doi {
    font-size: 0.72rem;
  }
  .capnote,
  .ordernote {
    margin: 0.4rem 0 0;
    font-size: 0.75rem;
  }
</style>
