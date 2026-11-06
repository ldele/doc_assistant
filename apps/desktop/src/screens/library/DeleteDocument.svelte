<script lang="ts">
  // Confirmation dialog for delete (ADR-046 §2, amending ADR-014).
  //
  // ADR-014 made "delete" mean "delete the file", which is right for a copy the app made and wrong
  // for a file the user keeps in their own folder and merely pointed the library at. So this dialog
  // now *asks*, and the safe branch is preselected:
  //
  //   • Remove from library (default) — row + chunks go; the file is untouched.
  //   • Also delete the file          — ADR-014's behaviour, opt-in per deletion.
  //
  // **Naming the destination is part of the decision, not copy-writing** (ADR-046): the accepted
  // risk of a per-delete choice is a mis-click, and showing the path is what makes the click
  // informed. When no path is recorded the destructive option is not offered at all — refusing to
  // name the target is refusing the guarantee the ADR asked for.
  import type { LibraryDocument } from '../../lib/core/types'
  import Icon from '../../shell/Icon.svelte'
  import Dialog from '../../shell/Dialog.svelte'
  import { deleteFileDetail, removeOnlyDetail, targetName } from '../../lib/library/deletetarget'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    doc,
    busy = false,
    onConfirm,
    onClose,
  }: {
    doc: LibraryDocument
    busy?: boolean
    /** `deleteFile` is the user's choice for THIS deletion — never a remembered preference. */
    onConfirm: (deleteFile: boolean) => void
    onClose: () => void
  } = $props()

  // Defaults to the non-destructive branch every time the dialog opens. Deliberately not
  // persisted: a remembered "also delete the file" would turn a per-deletion decision back into
  // the unconditional rule ADR-046 removed.
  let deleteFile = $state(false)

  const fileDetail = $derived(deleteFileDetail(doc))
  const safeDetail = $derived(removeOnlyDetail(doc))
</script>

<Dialog label="Delete document" {onClose}>
  <h2 class="dialog-title">Delete this document?</h2>
  <p class="target" title={doc.filename}>{targetName(doc)}</p>

  <fieldset class="choice">
    <legend class="sr-only">What to delete</legend>
    <label class="opt">
      <input type="radio" bind:group={deleteFile} value={false} disabled={busy} />
      <span>
        <strong>Remove from library</strong>
        <span class="detail">{safeDetail}</span>
      </span>
    </label>

    {#if fileDetail}
      <label class="opt">
        <input type="radio" bind:group={deleteFile} value={true} disabled={busy} />
        <span>
          <strong>Also delete the file</strong>
          <!-- `title` carries the untruncated path; the visible form keeps both ends. -->
          <span class="detail" title={doc.source_path ?? undefined}>{fileDetail}</span>
        </span>
      </label>
    {/if}
  </fieldset>

  <div class="dialog-actions">
    <button class="btn" onclick={onClose} type="button" disabled={busy} use:tooltip={'Keep the document'}>Cancel</button>
    <button class="btn btn-primary btn-danger" onclick={() => onConfirm(deleteFile)} type="button" disabled={busy} use:tooltip={deleteFile ? 'Take the document out of the library and delete its file from disk' : 'Take the document out of the library; its file stays on disk'}>
      <Icon name="trash-2" size={14} />
      {busy ? 'Deleting…' : deleteFile ? 'Delete both' : 'Remove'}
    </button>
  </div>
</Dialog>

<style>
  .target {
    margin: 0;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .choice {
    margin: var(--space-2) 0 0;
    padding: 0;
    border: none;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }
  .opt {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    padding: 0.55rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
  }
  .opt:hover {
    background: var(--surface-2);
  }
  .opt input {
    margin-top: 0.2rem;
    flex: none;
  }
  .opt strong {
    display: block;
    font-size: 0.9rem;
  }
  .detail {
    display: block;
    margin-top: 0.15rem;
    font-size: 0.8rem;
    color: var(--fg-2);
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
</style>
