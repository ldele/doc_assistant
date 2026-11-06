<script lang="ts">
  // The letter-spacing block (ADR-056 points 5, 9, 10): what extraction did to this document's
  // letter-spaced runs, and the ignore-list entries that change it. Hidden for a document that
  // has none, which is nearly every document.
  //
  // Each row is a crop of the page beside the run as the text holds it, with a tick where a space
  // was removed (the page and the text side by side, as on the review page the rows come from).
  // Every change writes the ignore list only; it takes effect when this document's text is
  // re-run, and the row says so until then.
  import { untrack } from 'svelte'
  import type { DocumentSpacing, LibraryDocument, SpacingRow } from '../../lib/core/types'
  import { getDocumentSpacing, keepRunSpaced, removeRunEntry, setKeepAllSpaced, sourcePageUrl } from '../../lib/core/api'
  import Icon from '../../shell/Icon.svelte'
  import EditSpacing from './EditSpacing.svelte'
  import { markRemovedSpaces, rowCaption, shownText, splitRows } from '../../lib/library/spacing'
  import { rerunLabel } from '../../lib/library/reingest'
  import { openRerun } from '../../lib/library/reingest.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    docId,
    listed,
    onShowPage,
    onOpenSettings,
  }: {
    docId: string
    /** This document's row in the library list: the name *Re-run text now…* gives the dialog. A
     *  new one means the list was refetched, which a finished re-run does, so what this block read
     *  may be out of date. */
    listed: LibraryDocument | null
    /** Open the source pane at this page. */
    onShowPage: (page: number) => void
    /** Open Settings at its Documents category, where the whole list is. */
    onOpenSettings: () => void
  } = $props()

  // Re-runs this document's text, which is where every change below takes effect.
  function rerunText(): void {
    openRerun([docId], rerunLabel(listed), ['text'])
  }

  let data = $state<DocumentSpacing | null>(null)
  let error = $state<string | null>(null)
  let actionError = $state<string | null>(null)
  let busy = $state(false)
  let editing = $state<SpacingRow | null>(null)

  const rows = $derived(splitRows(data?.rows ?? []))
  const pending = $derived((data?.rows ?? []).filter((r) => r.pending !== null).length)
  const visible = $derived(data !== null && data.pageable && (data.rows.length > 0 || data.keep_all))

  // Last-write-wins token, like the other document blocks. A refetch of the same document keeps
  // its rows and an open editor on screen; only another document clears them.
  let token = 0
  $effect(() => {
    const id = docId
    void listed
    untrack(() => {
      error = null
      actionError = null
      if (data?.document_id !== id) {
        data = null
        editing = null
      }
    })
    if (!id) return
    const mine = ++token
    void (async () => {
      try {
        const d = await getDocumentSpacing(id)
        if (mine === token) data = d
      } catch (e) {
        if (mine === token) error = String(e)
      }
    })()
  })

  async function act(change: () => Promise<DocumentSpacing>): Promise<void> {
    busy = true
    actionError = null
    token++ // a read already on its way must not overwrite what this change returns
    try {
      data = await change()
    } catch (e) {
      actionError = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  function pendingNote(row: SpacingRow): string {
    if (row.pending === null) return ''
    const what = { closed: 'closes', kept: 'stays as extracted', yours: 'takes your spacing', maths: 'stays as extracted', unknown: '' }[row.pending]
    return what ? `${what} at the next re-run` : ''
  }
</script>

{#if error}
  <!-- Advisory: a failure here must not cost the reader the document. -->
  <p class="quiet">Letter-spacing unavailable.</p>
{:else if visible && data}
  <section class="spacing" aria-label="Letter-spacing on this document">
    <h3>
      Letter-spacing on this document
      <span class="count">{data.rows.length} {data.rows.length === 1 ? 'run' : 'runs'}</span>
    </h3>
    <p class="lead">
      The letter-spaced runs extraction closed here, and the ones you keep as extracted. A change takes
      effect when this document's text is re-run.
    </p>
    {#if pending > 0}
      <p class="pendingbar" role="status">
        {pending} {pending === 1 ? 'change waits' : 'changes wait'} for a re-run of this document's text.
        <button class="btn" type="button" onclick={rerunText}>Re-run text now…</button>
      </p>
    {/if}
    {#if actionError}<p class="err" role="alert">{actionError}</p>{/if}

    {#each [{ title: 'Closed', list: rows.closed, closed: true }, { title: 'Kept or spaced your way', list: rows.kept, closed: false }] as group (group.title)}
      <h4>{group.title} · {group.list.length === 0 ? 'none' : group.list.length}</h4>
      {#if group.list.length === 0}
        <p class="quiet">
          {group.closed
            ? 'Nothing on this document is closed.'
            : 'A run you keep spaced appears here, with Show on page, Edit spacing and Remove.'}
        </p>
      {:else}
        <ul class="rows">
          {#each group.list as row (row.spaced)}
            {@const text = shownText(row)}
            <li class="row" class:pending={row.pending !== null}>
              {#if row.crop && row.pages.length > 0}
                <img
                  class="crop"
                  src={sourcePageUrl(docId, row.pages[0], 144, row.crop)}
                  alt="The page around this run, page {row.pages[0]}"
                  loading="lazy"
                />
              {:else}
                <span class="crop none" aria-hidden="true"></span>
              {/if}
              <div class="txt">
                <!-- One line, no whitespace around the loop: `pre-wrap` would show it. A tick is
                     drawn on its glyph, not beside it, so the line breaks only at word spaces. -->
                <span class="line" title="As extracted: {row.spaced}">{#each markRemovedSpaces(row.spaced, text) as g, i (i)}{#if g.spaceBefore}{' '}{/if}<span class:tick={g.removedBefore}>{g.char}</span>{/each}</span>
                <span class="cap">
                  {rowCaption({ spaced: row.spaced, written: text, pages: row.pages })}{#if row.kind === 'maths'}{' · maths, left as extracted'}{:else if row.kind === 'url'}{' · web address'}{/if}{#if row.pending !== null}<span class="note">{' · '}{pendingNote(row)}</span>{/if}
                </span>
                <div class="acts">
                  <button
                    class="btn"
                    type="button"
                    disabled={row.pages.length === 0}
                    onclick={() => onShowPage(row.pages[0])}
                  >
                    <Icon name="book-open" size={13} /> Show on page
                  </button>
                  <button class="btn" type="button" disabled={busy} onclick={() => (editing = row)}>
                    Edit spacing
                  </button>
                  {#if group.closed}
                    <button
                      class="btn"
                      type="button"
                      disabled={busy || data.keep_all}
                      onclick={() => act(() => keepRunSpaced(docId, row.spaced))}
                    >
                      Keep spaced
                    </button>
                  {:else}
                    <button
                      class="btn"
                      type="button"
                      disabled={busy || data.keep_all}
                      use:tooltip={data.keep_all ? 'Every run of this document is kept: turn that off first' : 'Give this run back to the rule, so your choice for it no longer applies'}
                      onclick={() => act(() => removeRunEntry(docId, row.spaced))}
                    >
                      Remove
                    </button>
                  {/if}
                </div>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    {/each}

    <div class="foot">
      <button
        class="btn"
        type="button"
        aria-pressed={data.keep_all}
        disabled={busy}
        onclick={() => act(() => setKeepAllSpaced(docId, !(data?.keep_all ?? false)))}
      >
        {data.keep_all ? 'Stop keeping every run spaced' : 'Keep every run spaced'}
      </button>
      <button class="btn" type="button" onclick={rerunText}>
        <Icon name="rotate-ccw" size={13} /> Re-run text now…
      </button>
      <span class="spacer"></span>
      <button class="btn" type="button" onclick={onOpenSettings}>All entries in Settings</button>
    </div>
  </section>

  {#if editing}
    {#key editing.spaced}
    <EditSpacing
      {docId}
      row={editing}
      onSaved={(d) => {
        data = d
        editing = null
      }}
      onClose={() => (editing = null)}
    />
    {/key}
  {/if}
{/if}

<style>
  .spacing {
    /* The rows stack by the block's own width: the source pane halves it at any window size. */
    container-type: inline-size;
    margin-top: 1rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    background: var(--surface);
  }
  h3 {
    font-size: 0.9rem;
    margin: 0 0 0.3rem;
  }
  h4 {
    font-size: 0.8rem;
    margin: 0.75rem 0 0.35rem;
    color: var(--fg-2);
    font-weight: 600;
  }
  .count {
    font-weight: 400;
    opacity: 0.65;
    font-size: 0.8rem;
  }
  .lead,
  .quiet {
    margin: 0;
    font-size: 0.82rem;
    color: var(--fg-2);
    line-height: 1.5;
  }
  .pendingbar {
    margin: 0.5rem 0 0;
    padding: 0.4rem 0.6rem;
    font-size: 0.8rem;
    border-radius: 8px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-fg);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
  }
  .err {
    margin: 0.5rem 0 0;
    font-size: 0.8rem;
    color: var(--danger);
  }
  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .row {
    display: grid;
    grid-template-columns: minmax(0, 220px) minmax(0, 1fr);
    align-items: center;
    gap: 0.75rem;
    padding: 0.45rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .row.pending {
    border-style: dashed;
  }
  .crop {
    width: 100%;
    max-height: 56px;
    object-fit: contain;
    object-position: left center;
    background: #ffffff;
    border-radius: 4px;
    border: 1px solid var(--border);
  }
  .crop.none {
    display: block;
    height: 32px;
    background: var(--surface-2);
  }
  .txt {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
  }
  .line {
    font-size: 0.92rem;
    color: var(--fg);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    /* A kept space must read wider than a removed one's tick. */
    word-spacing: 0.3em;
  }
  /* A removed space: a hairline on the left edge of the letter after it. Positioned, so it adds
     no break opportunity (an inline-block would split words at every tick). */
  .tick {
    position: relative;
    margin-left: 2px;
  }
  .tick::before {
    content: '';
    position: absolute;
    left: -1.5px;
    top: 0.2em;
    bottom: 0.15em;
    width: 1px;
    background: var(--accent);
    opacity: 0.6;
  }
  .cap {
    font-size: 0.75rem;
    color: var(--fg-2);
  }
  .note {
    color: var(--warn-fg);
  }
  .acts,
  .foot {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    align-items: center;
  }
  .acts {
    margin-top: 0.25rem;
  }
  .foot {
    margin-top: 0.75rem;
  }
  .spacer {
    flex: 1;
  }
  @container (max-width: 480px) {
    .row {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
