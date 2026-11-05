<script lang="ts">
  // Edit spacing (ADR-056 point 10): choose where a run's words break. The letters are the page's
  // and cannot change; between every two of them is a gap you can join (a tick) or break (a
  // space). Saving writes the run's entry in the ignore list, which applies when the document's
  // text is re-run; "Back to the rule's reading" removes the entry instead.
  import { untrack } from 'svelte'
  import type { DocumentSpacing, SpacingRow } from '../../lib/core/types'
  import { removeRunEntry, setRunSpacing, sourcePageUrl } from '../../lib/core/api'
  import { gapsOf, glyphsOf, shownText, writeWithGaps } from '../../lib/library/spacing'

  let {
    docId,
    row,
    onSaved,
    onClose,
  }: {
    docId: string
    row: SpacingRow
    onSaved: (data: DocumentSpacing) => void
    onClose: () => void
  } = $props()

  const glyphs = $derived(glyphsOf(row.spaced))
  // Start from what the document will hold; a reading the gaps cannot express starts from the
  // rule's, which always can. Read once: the dialog edits one row, and the parent remounts it
  // (`{#key}`) for another.
  let gaps = $state<boolean[]>(
    untrack(() => gapsOf(row.spaced, shownText(row)) ?? gapsOf(row.spaced, row.rule) ?? []),
  )
  let busy = $state(false)
  let error = $state<string | null>(null)

  const preview = $derived(writeWithGaps(row.spaced, gaps))
  const isRule = $derived(preview === row.rule)

  function toggle(i: number): void {
    gaps[i] = !gaps[i]
  }

  async function save(): Promise<void> {
    busy = true
    error = null
    try {
      onSaved(await setRunSpacing(docId, row.spaced, preview))
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  async function backToRule(): Promise<void> {
    busy = true
    error = null
    try {
      onSaved(await removeRunEntry(docId, row.spaced))
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape' && !busy) onClose()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={() => !busy && onClose()} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="Edit spacing">
  <h2>Edit spacing</h2>
  <p class="lead">
    Choose where the words break. The letters stay as the page has them; only the spaces change.
  </p>
  <p class="src">
    {row.pages.length === 1 ? `Page ${row.pages[0]}` : row.pages.length > 1 ? `${row.pages.length} pages` : 'No page recorded'}
  </p>
  {#if row.crop && row.pages.length > 0}
    <!-- The page, so the breaks are chosen against what it shows. -->
    <img
      class="crop"
      src={sourcePageUrl(docId, row.pages[0], 144, row.crop)}
      alt="The page around this run, page {row.pages[0]}"
    />
  {/if}

  <div class="strip" role="group" aria-label="The run's letters and the gaps between them">
    {#each glyphs as g, i (i)}<span class="glyph">{g}</span>{#if i < glyphs.length - 1}<button
          type="button"
          class="gap"
          class:open={gaps[i]}
          aria-pressed={gaps[i]}
          aria-label={gaps[i] ? `Join ${g} and ${glyphs[i + 1]}` : `Break between ${g} and ${glyphs[i + 1]}`}
          title={gaps[i] ? 'A space: click to join' : 'Joined: click to break here'}
          disabled={busy}
          onclick={() => toggle(i)}
        ></button>{/if}{/each}
  </div>

  <p class="preview"><span class="k">Reads as</span> <span class="v">{preview}</span></p>
  <p class="note">
    {isRule
      ? "This is the rule's own reading."
      : "Save writes this spacing to the ignore list. Back to the rule's reading removes it again."}
  </p>
  {#if error}<p class="err" role="alert">{error}</p>{/if}

  <div class="mactions">
    <button class="primary" type="button" disabled={busy} onclick={save}>Save</button>
    <button class="ghost" type="button" disabled={busy} onclick={backToRule}>Back to the rule's reading</button>
    <button class="ghost" type="button" disabled={busy} onclick={onClose}>Cancel</button>
  </div>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--fg) 32%, transparent);
    z-index: 40;
  }
  .modal {
    position: fixed;
    z-index: 41;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(92vw, 640px);
    max-height: 90vh;
    overflow: auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  h2 {
    margin: 0;
    font-size: var(--text-title);
    font-family: var(--font-serif);
  }
  .lead,
  .src,
  .note {
    margin: 0;
    font-size: 0.82rem;
    color: var(--fg-2);
    line-height: 1.5;
  }
  .crop {
    display: block;
    width: 100%;
    max-height: 96px;
    object-fit: contain;
    object-position: left center;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .strip {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    padding: 0.6rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface-2);
    font-size: 1.15rem;
    row-gap: 0.4rem;
  }
  .glyph {
    padding: 0 1px;
    color: var(--fg);
  }
  /* A gap between two letters: a hairline tick when joined, a visible space when broken. */
  .gap {
    width: 7px;
    margin: 0 1px;
    padding: 0;
    border: none;
    border-radius: 3px;
    background: transparent;
    cursor: pointer;
    position: relative;
  }
  .gap::after {
    content: '';
    position: absolute;
    left: 3px;
    top: 15%;
    bottom: 15%;
    width: 1px;
    background: var(--accent);
    opacity: 0.55;
  }
  .gap.open {
    width: 0.8em;
    background: color-mix(in srgb, var(--accent) 14%, transparent);
  }
  .gap.open::after {
    display: none;
  }
  .gap:hover,
  .gap:focus-visible {
    background: color-mix(in srgb, var(--accent) 24%, transparent);
  }
  .preview {
    margin: 0;
    display: flex;
    gap: 0.6rem;
    align-items: baseline;
  }
  .preview .k {
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .preview .v {
    font-size: 1rem;
    color: var(--fg);
    overflow-wrap: anywhere;
  }
  .err {
    margin: 0;
    font-size: 0.8rem;
    color: var(--danger);
  }
  .mactions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: var(--space-2);
  }
  .ghost,
  .primary {
    font: inherit;
    font-size: 0.82rem;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    padding: 0.35rem 0.8rem;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }
  .primary {
    font-weight: 600;
    border-color: var(--accent);
    background: var(--accent);
    color: var(--accent-fg);
  }
  .ghost:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .ghost:disabled,
  .primary:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .ghost:focus-visible,
  .primary:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
</style>
