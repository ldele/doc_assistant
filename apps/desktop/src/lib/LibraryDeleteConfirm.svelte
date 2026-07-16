<script lang="ts">
  // Confirmation dialog for safe-delete (ADR-014). States clearly that the source file goes to the
  // Recycle Bin (recoverable) and the document leaves the library + search index. Delete is the
  // destructive (red) action; Cancel / Esc / scrim-click back out.
  import type { LibraryDocument } from './types'
  import Icon from './Icon.svelte'

  let {
    doc,
    busy = false,
    onConfirm,
    onClose,
  }: {
    doc: LibraryDocument
    busy?: boolean
    onConfirm: () => void
    onClose: () => void
  } = $props()

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="Delete document">
  <h2>Delete this document?</h2>
  <p class="target" title={doc.filename}>{doc.title ?? doc.filename}</p>
  <p class="body">
    The source file moves to your <strong>Recycle Bin</strong> (recoverable), and the document leaves
    your library{#if doc.chunk_count}, removing its {doc.chunk_count.toLocaleString()} chunks from the
      search index{/if}.
  </p>
  <div class="mactions">
    <button class="ghost" onclick={onClose} type="button" disabled={busy}>Cancel</button>
    <button class="danger" onclick={onConfirm} type="button" disabled={busy}>
      <Icon name="trash-2" size={14} />
      {busy ? 'Deleting…' : 'Delete'}
    </button>
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
    width: min(92vw, 420px);
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
  .target {
    margin: 0;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .body {
    margin: 0;
    font-size: 0.85rem;
    color: var(--fg-2);
    line-height: 1.6;
  }
  .mactions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: var(--space-2);
  }
  .mactions button {
    font: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    border: 1px solid var(--border);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .mactions button:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .ghost {
    background: none;
    color: var(--fg);
  }
  .ghost:hover {
    background: var(--surface-2);
  }
  .danger {
    background: color-mix(in srgb, var(--danger) 14%, transparent);
    color: var(--danger);
    border-color: var(--danger);
    font-weight: 600;
  }
  .danger:hover:not(:disabled) {
    background: color-mix(in srgb, var(--danger) 22%, transparent);
  }
</style>
