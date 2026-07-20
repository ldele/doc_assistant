<script lang="ts">
  // Manage-folders view (ADR-025 F1, docs/specs/feature-corpus-folders.md). Opened from the
  // library rail's Folders section. Create a folder, rename it inline, delete it, and pick
  // which documents belong to it (bulk, searchable). Reuses LibraryManageKeywords' modal shell
  // (scrim + centred dialog, Esc-to-close). Dumb by design — App owns the lists + the API calls.
  //
  // The honest line at the foot is not decoration (spec D8): F1 ships the Library filter but NOT
  // retrieval scoping (that is F2), and a user who narrows the Library would otherwise reasonably
  // assume their next question was narrowed too. That is the is_archived failure ADR-025 exists
  // to prevent. Delete the line in F2, when it stops being true.
  import { untrack } from 'svelte'
  import type { LibraryDocument, LibraryFolder } from './types'
  import { docLabel } from './library'
  import Icon from './Icon.svelte'

  let {
    folders,
    documents,
    selectedId = null,
    initialDocQuery = '',
    error = null,
    onCreate,
    onRename,
    onDelete,
    onAddDocuments,
    onRemoveDocument,
    onSelect,
    onClose,
  }: {
    folders: LibraryFolder[]
    documents: LibraryDocument[]
    selectedId?: string | null
    initialDocQuery?: string
    error?: string | null
    onCreate: (name: string) => void
    onRename: (folderId: string, name: string) => void
    onDelete: (folderId: string) => void
    onAddDocuments: (folderId: string, documentIds: string[]) => void
    onRemoveDocument: (folderId: string, documentId: string) => void
    onSelect: (folderId: string | null) => void
    onClose: () => void
  } = $props()

  const selected = $derived(folders.find((f) => f.id === selectedId) ?? null)

  let newName = $state('')
  function submitCreate(): void {
    const name = newName.trim()
    if (!name) return
    onCreate(name)
    newName = ''
  }

  let editingId = $state<string | null>(null)
  let editingValue = $state('')
  function startRename(f: LibraryFolder): void {
    editingId = f.id
    editingValue = f.name
  }
  function commitRename(): void {
    if (editingId === null) return
    const v = editingValue.trim()
    if (v) onRename(editingId, v)
    editingId = null
  }

  let confirmingDeleteId = $state<string | null>(null)

  // Membership picker. `pending` holds the ticks made since the panel opened for this folder;
  // nothing is written until Apply, so a mis-click costs nothing.
  // Seeded from the prop: the grid's "Add to folder…" opens this view pre-filtered to that one
  // document. Safe as a one-shot initialiser because the overlay is mounted fresh on every open.
  let docQuery = $state(untrack(() => initialDocQuery))
  let pending = $state<Set<string>>(new Set())
  const memberIds = $derived(
    new Set(selected ? documents.filter((d) => d.folder_ids.includes(selected.id)).map((d) => d.id) : []),
  )
  const shownDocs = $derived.by(() => {
    const q = docQuery.trim().toLowerCase()
    const list = q === '' ? documents : documents.filter((d) => docLabel(d).toLowerCase().includes(q))
    return [...list].sort((a, b) => docLabel(a).localeCompare(docLabel(b)))
  })
  function togglePending(id: string): void {
    const next = new Set(pending)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    pending = next
  }
  function applyPending(): void {
    if (!selected) return
    const ids = [...pending].filter((id) => !memberIds.has(id))
    if (ids.length > 0) onAddDocuments(selected.id, ids)
    pending = new Set()
    docQuery = ''
  }
  function selectFolder(id: string | null): void {
    pending = new Set()
    // Back to the seed, not to blank: in the "Add to folder…" flow the folder is chosen *after*
    // the document, so clearing here would throw away the filter that brought the user in.
    docQuery = initialDocQuery
    onSelect(id)
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
  function autofocus(node: HTMLInputElement): void {
    node.focus()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="Manage folders">
  <div class="mhead">
    <h2>Manage folders</h2>
    <button class="iconbtn" onclick={onClose} aria-label="Close" type="button">
      <Icon name="x" size={16} />
    </button>
  </div>

  <div class="body">
    <section class="block">
      <h3>New folder</h3>
      <div class="createrow">
        <input
          type="text"
          bind:value={newName}
          placeholder="Folder name (e.g. Demo corpus)"
          onkeydown={(e) => e.key === 'Enter' && submitCreate()}
        />
        <button class="primary" onclick={submitCreate} disabled={newName.trim() === ''} type="button">
          Create
        </button>
      </div>
      {#if error}<p class="error">{error}</p>{/if}
    </section>

    <section class="block">
      <h3>Folders ({folders.length})</h3>
      {#if folders.length === 0}
        <p class="hint">No folders yet. Create one above, then pick its documents.</p>
      {:else}
        <div class="list">
          {#each folders as f (f.id)}
            <div class="row" class:active={f.id === selectedId}>
              {#if editingId === f.id}
                <input
                  class="renameinput"
                  type="text"
                  bind:value={editingValue}
                  use:autofocus
                  onkeydown={(e) => {
                    if (e.key === 'Enter') commitRename()
                    if (e.key === 'Escape') editingId = null
                  }}
                  onblur={commitRename}
                />
              {:else}
                <button class="rowname" onclick={() => selectFolder(f.id === selectedId ? null : f.id)} type="button">
                  <Icon name="folder" size={14} />
                  <span class="name">{f.name}</span>
                  <span class="count">{f.doc_count}</span>
                </button>
                <button class="iconbtn" onclick={() => startRename(f)} aria-label="Rename {f.name}" type="button">
                  <Icon name="pencil" size={14} />
                </button>
                {#if confirmingDeleteId === f.id}
                  <button class="confirmdel" onclick={() => { confirmingDeleteId = null; onDelete(f.id) }} type="button">
                    Delete folder?
                  </button>
                {:else}
                  <button
                    class="iconbtn danger"
                    onclick={() => (confirmingDeleteId = f.id)}
                    aria-label="Delete {f.name}"
                    type="button"
                  >
                    <Icon name="trash-2" size={14} />
                  </button>
                {/if}
              {/if}
            </div>
          {/each}
        </div>
        <p class="hint deletehint">Deleting a folder never deletes documents — only the grouping.</p>
      {/if}
    </section>

    {#if selected}
      <section class="block">
        <h3>Documents in “{selected.name}” ({memberIds.size})</h3>
        <input
          class="search"
          type="search"
          bind:value={docQuery}
          placeholder="Search {documents.length} documents…"
        />
        <div class="picker">
          {#each shownDocs as d (d.id)}
            {@const isMember = memberIds.has(d.id)}
            <label class="pickrow" class:member={isMember}>
              <input
                type="checkbox"
                checked={isMember || pending.has(d.id)}
                disabled={isMember}
                onchange={() => togglePending(d.id)}
              />
              <span class="picklabel">{docLabel(d)}</span>
              {#if isMember}
                <button
                  class="iconbtn danger"
                  onclick={() => onRemoveDocument(selected.id, d.id)}
                  aria-label="Remove {docLabel(d)} from {selected.name}"
                  type="button"
                >
                  <Icon name="x" size={13} />
                </button>
              {/if}
            </label>
          {:else}
            <p class="hint">No document matches “{docQuery}”.</p>
          {/each}
        </div>
        <button class="secondary" onclick={applyPending} disabled={pending.size === 0} type="button">
          Add {pending.size} document{pending.size === 1 ? '' : 's'}
        </button>
      </section>
    {/if}

    <p class="scopenote">
      <Icon name="triangle-alert" size={13} />
      <span>
        Folders organise the Library. Chat still searches <strong>every</strong> document —
        scoping a conversation to a folder is not built yet.
      </span>
    </p>
  </div>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--fg) 32%, transparent);
    z-index: 42;
  }
  .modal {
    position: fixed;
    z-index: 43;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(94vw, 640px);
    max-height: min(86vh, 700px);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .mhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex: none;
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
    flex: none;
  }
  .iconbtn:hover {
    color: var(--fg);
    background: var(--surface-2);
  }
  .iconbtn.danger:hover {
    color: var(--danger, #c0392b);
  }
  .body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  .block h3 {
    margin: 0 0 0.3rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--fg-2);
  }
  .hint {
    margin: 0 0 0.5rem;
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .deletehint {
    margin: 0.4rem 0 0;
    font-style: italic;
  }
  .createrow {
    display: flex;
    gap: 0.4rem;
  }
  .createrow input,
  .renameinput,
  .search {
    flex: 1;
    min-width: 0;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.4rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--fg);
  }
  .createrow input:focus,
  .renameinput:focus,
  .search:focus {
    outline: none;
    border-color: var(--accent);
  }
  .search {
    width: 100%;
    font-size: 0.8rem;
    margin-bottom: 0.4rem;
  }
  .primary {
    font: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--accent-fg);
    background: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
    flex: none;
  }
  .primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .secondary {
    font: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--fg);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
    align-self: flex-start;
    margin-top: 0.5rem;
  }
  .secondary:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .secondary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .error {
    margin: 0.4rem 0 0;
    font-size: 0.78rem;
    color: var(--danger, #c0392b);
  }
  .list {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.45rem;
  }
  .row.active {
    border-color: var(--accent);
  }
  .rowname {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font: inherit;
    font-size: 0.85rem;
    text-align: left;
    cursor: pointer;
    color: var(--fg);
    background: none;
    border: none;
    padding: 0.1rem 0;
  }
  .rowname .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .rowname .count {
    flex: none;
    font-size: 0.72rem;
    color: var(--fg-2);
  }
  .confirmdel {
    font: inherit;
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--danger, #c0392b);
    background: none;
    border: 1px solid var(--danger, #c0392b);
    border-radius: 6px;
    padding: 0.2rem 0.45rem;
    flex: none;
  }
  .picker {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    max-height: 220px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem;
  }
  .pickrow {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.8rem;
    padding: 0.15rem 0.2rem;
    border-radius: 6px;
    cursor: pointer;
  }
  .pickrow:hover {
    background: var(--surface-2);
  }
  .pickrow.member {
    color: var(--fg-2);
  }
  .picklabel {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .scopenote {
    display: flex;
    align-items: flex-start;
    gap: 0.4rem;
    margin: 0;
    padding-top: var(--space-3);
    border-top: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--fg-2);
  }
  @media (max-width: 640px) {
    .modal {
      width: 94vw;
      max-height: 90vh;
    }
  }
</style>
