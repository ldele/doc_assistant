<script lang="ts">
  // Edit-metadata modal for one library document (ADR-013). Pre-filled with the *effective*
  // values (user override ?? auto-extracted); Save PATCHes the overrides, Reset drops them
  // (revert to auto). The backend dedups each field against its auto default, so saving an
  // untouched field creates no override.
  import type { LibraryDocument } from './types'
  import Icon from './Icon.svelte'

  let {
    doc,
    onSave,
    onReset,
    onClose,
  }: {
    doc: LibraryDocument
    onSave: (patch: { title: string; authors: string; year: number | null }) => void
    onReset: () => void
    onClose: () => void
  } = $props()

  // The modal is mounted fresh per document (App unmounts it on close), so seeding the editable
  // fields from the prop's initial value is intentional — they must not reactively overwrite edits.
  // svelte-ignore state_referenced_locally
  let title = $state(doc.title ?? '')
  // svelte-ignore state_referenced_locally
  let authors = $state(doc.authors ?? '')
  // svelte-ignore state_referenced_locally
  let yearStr = $state(doc.year != null ? String(doc.year) : '')

  function save(): void {
    const t = yearStr.trim()
    const year = t === '' ? null : Number.parseInt(t, 10)
    onSave({ title, authors, year: Number.isNaN(year as number) ? null : year })
  }
  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
  function autofocus(node: HTMLInputElement): void {
    node.focus()
    node.select()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="Edit document metadata">
  <div class="mhead">
    <h2>Edit metadata</h2>
    <button class="iconbtn" onclick={onClose} aria-label="Close" type="button">
      <Icon name="x" size={16} />
    </button>
  </div>
  <p class="fname" title={doc.filename}>{doc.filename}</p>

  <label class="field">
    <span>Title</span>
    <input use:autofocus bind:value={title} placeholder="Document title" />
  </label>
  <label class="field">
    <span>Authors</span>
    <input bind:value={authors} placeholder="e.g. Jane Doe, John Smith" />
  </label>
  <label class="field">
    <span>Year</span>
    <input bind:value={yearStr} inputmode="numeric" placeholder="e.g. 2024" />
  </label>

  <p class="hint">
    Auto-detected values are the default. Your edits take precedence; Reset restores them.
  </p>

  <div class="mactions">
    <button class="reset" onclick={onReset} type="button">
      <Icon name="rotate-ccw" size={14} /> Reset to default
    </button>
    <span class="spacer"></span>
    <button class="ghost" onclick={onClose} type="button">Cancel</button>
    <button class="primary" onclick={save} type="button">Save</button>
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
    width: min(92vw, 460px);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .mhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .mhead h2 {
    margin: 0;
    font-size: var(--text-title);
    font-family: var(--font-serif);
  }
  .iconbtn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.2rem;
    border: none;
    background: none;
    color: var(--fg-2);
    border-radius: 6px;
    cursor: pointer;
  }
  .iconbtn:hover {
    color: var(--fg);
    background: var(--surface-2);
  }
  .fname {
    margin: 0;
    font-size: 0.72rem;
    color: var(--fg-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .field span {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--fg-2);
  }
  .field input {
    font: inherit;
    font-size: 0.9rem;
    padding: 0.45rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--fg);
  }
  .field input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .hint {
    margin: 0.1rem 0 0;
    font-size: 0.72rem;
    color: var(--fg-2);
    line-height: 1.5;
  }
  .mactions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: var(--space-2);
  }
  .spacer {
    flex: 1;
  }
  .mactions button {
    font: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    border: 1px solid var(--border);
  }
  .reset {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: none;
    color: var(--fg-2);
  }
  .reset:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  .ghost {
    background: none;
    color: var(--fg);
  }
  .ghost:hover {
    background: var(--surface-2);
  }
  .primary {
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
  }
  .primary:hover {
    filter: brightness(1.05);
  }
</style>
