<script lang="ts">
  // The text-check block (A2b, docs/security.md S5): the paragraphs of this document that address
  // an AI rather than a reader, found once after extraction. It informs: nothing in the document
  // was changed, and a paper about AI that quotes such text is marked too. Hidden for a document
  // checked with nothing found, which is nearly every document.
  import { untrack } from 'svelte'
  import type { DocumentTextCheck, LibraryDocument } from '../../lib/core/types'
  import { getDocumentTextCheck } from '../../lib/core/api'
  import Icon from '../../shell/Icon.svelte'
  import { rerunLabel } from '../../lib/library/reingest'
  import { openRerun } from '../../lib/library/reingest.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    docId,
    listed,
    onShowPage,
  }: {
    docId: string
    /** This document's row in the library list. A new one means the list was refetched, which a
     *  finished re-run does, so the check may have changed. */
    listed: LibraryDocument | null
    /** Open the source pane at this page. */
    onShowPage: (page: number) => void
  } = $props()

  let data = $state<DocumentTextCheck | null>(null)
  let failed = $state(false)

  const visible = $derived(data !== null && (data.state !== 'current' || data.n_marked > 0))
  const more = $derived(data ? data.n_marked - data.findings.length : 0)

  let token = 0
  $effect(() => {
    const id = docId
    void listed
    untrack(() => {
      failed = false
      if (data?.document_id !== id) data = null
    })
    if (!id) return
    const mine = ++token
    void (async () => {
      try {
        const d = await getDocumentTextCheck(id)
        if (mine === token) data = d
      } catch {
        if (mine === token) failed = true
      }
    })()
  })

  function recheck(): void {
    openRerun([docId], rerunLabel(listed), ['checks'])
  }
</script>

{#if failed}
  <p class="aicheck-quiet muted">Text check unavailable.</p>
{:else if visible && data}
  <section class="aicheck" class:marked={data.state === 'current'} aria-label="Text addressed to an AI">
    {#if data.state === 'current'}
      <h3>
        <Icon name="triangle-alert" size={14} /> Text addressed to an AI
        <span class="muted aicheck-count">{data.n_marked} {data.n_marked === 1 ? 'paragraph' : 'paragraphs'}</span>
      </h3>
      <p class="aicheck-lead muted">
        Written to an AI assistant rather than to a reader, such as an instruction planted in the document. An
        answer citing it says so, and a link from it is not clickable. Nothing was removed: a paper about AI
        that quotes such text is marked too.
      </p>
      <ul class="aicheck-list">
        {#each data.findings as f, i (i)}
          <li>
            <blockquote>{f.excerpt}</blockquote>
            <div class="aicheck-meta">
              <span class="muted">{f.labels.join(' · ')}{f.page !== null ? ` · page ${f.page}` : ''}</span>
              {#if f.page !== null}
                {@const page = f.page}
                <button
                  class="btn"
                  type="button"
                  onclick={() => onShowPage(page)}
                  use:tooltip={'Open the document at the page this paragraph is on'}
                >
                  <Icon name="book-open" size={13} /> Show on page
                </button>
              {/if}
            </div>
          </li>
        {/each}
      </ul>
      {#if more > 0}
        <p class="aicheck-lead muted">{more} more {more === 1 ? 'paragraph is' : 'paragraphs are'} marked, not listed here.</p>
      {/if}
    {:else}
      <p class="aicheck-line">
        <span class="muted">
          {data.state === 'stale'
            ? 'The check for text addressed to an AI was made before this text or the check changed.'
            : 'Not checked for text addressed to an AI yet.'}
        </span>
        <button class="btn" type="button" onclick={recheck} use:tooltip={'Open the re-run dialog with Text checks chosen'}>
          <Icon name="rotate-ccw" size={13} /> Check now…
        </button>
      </p>
    {/if}
  </section>
{/if}

<style>
  .aicheck {
    margin-top: var(--space-3);
    padding: 0.6rem 0.85rem;
    border-radius: 10px;
    border: 1px dashed var(--border);
  }
  .aicheck.marked {
    border: 1px solid var(--warn-border);
    background: var(--warn-bg);
  }
  .aicheck h3 {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin: 0 0 0.25rem;
    font-size: 0.9rem;
    color: var(--warn-fg);
  }
  .aicheck-count {
    font-weight: 400;
    font-size: 0.78rem;
  }
  .aicheck-lead,
  .aicheck-quiet {
    margin: 0.2rem 0 0.4rem;
    font-size: 0.8rem;
    line-height: 1.45;
  }
  .aicheck-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.45rem;
  }
  .aicheck-list blockquote {
    margin: 0;
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--warn-border);
    background: var(--surface);
    font-size: 0.84rem;
    overflow-wrap: anywhere;
  }
  .aicheck-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
    margin-top: 0.2rem;
    font-size: 0.75rem;
  }
  .aicheck-line {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
    font-size: 0.8rem;
  }
</style>
