<script lang="ts">
  // The folder list of a chat's setup (A8a, W1): *All documents* set apart, then one tickbox per
  // folder, and a count line that says how many documents are searched, each once. Shown by
  // *Search in* on the start screen, the *New chat defaults* window and *Continue in a new chat*.
  // The tick rules are `lib/chat/setup.ts`; the union comes from the backend, which resolves the
  // folders as a turn would (`/api/conversations/scope`), so the count can never disagree.
  import type { LibraryFolder } from '../../lib/core/types'
  import { previewScope } from '../../lib/core/api'
  import { countLine, pruneChoice, tickFolder, type FolderChoice } from '../../lib/chat/setup'

  let {
    folders,
    libraryCount,
    choice = $bindable(),
    union = $bindable(null),
    preview = true,
    name,
  }: {
    folders: LibraryFolder[]
    libraryCount: number
    choice: FolderChoice
    /** The union's size for the ticked folders, once the backend has said it. */
    union?: number | null
    /** Ask the backend for the union here; off where the parent asks for it (the start screen,
     *  whose *Search in* button shows the count while this list is closed). */
    preview?: boolean
    /** Keeps two lists on one screen apart (ids of the tickboxes). */
    name: string
  } = $props()

  // A tick starts from the folders that still exist (a default folder deleted since is dropped).
  const known = $derived(folders.map((f) => f.id))
  function tick(id: string | null, checked: boolean): void {
    choice = tickFolder(pruneChoice(choice, known), id, checked)
  }

  // Ask the backend for the union whenever the ticks change; a stale answer is dropped.
  let asked = 0
  $effect(() => {
    if (!preview) return
    const ids = choice === null ? null : [...choice]
    const n = ++asked
    if (ids === null) {
      union = null
      return
    }
    union = null
    previewScope(ids)
      .then((s) => {
        if (n === asked) union = s?.doc_count ?? 0
      })
      .catch(() => {
        if (n === asked) union = null
      })
  })
</script>

<div class="ticks">
  <label class="tickrow all">
    <input
      type="checkbox"
      checked={choice === null}
      onchange={(e) => tick(null, e.currentTarget.checked)}
      id="{name}-all"
    />
    <span class="tickname">All documents · {libraryCount}</span>
  </label>
  <p class="ticknote muted">Ticking it unticks every folder</p>
  {#if folders.length > 0}
    <p class="tickhead muted">Folders</p>
    <div class="folderlist">
      {#each folders as f (f.id)}
        <label class="tickrow">
          <input
            type="checkbox"
            checked={choice !== null && choice.includes(f.id)}
            onchange={(e) => tick(f.id, e.currentTarget.checked)}
            id="{name}-{f.id}"
          />
          <span class="tickname">{f.name} · {f.doc_count}</span>
        </label>
      {/each}
    </div>
  {:else}
    <p class="ticknote muted">No folders yet: the Library makes them.</p>
  {/if}
  <p class="countline">{countLine(choice, folders, libraryCount, union)}</p>
</div>

<style>
  .ticks {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .tickrow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-sm);
    color: var(--fg);
    cursor: pointer;
    padding: 0.15rem 0;
  }
  /* T-015's styling half: the box, its tick and its focus ring follow the theme. */
  .tickrow input {
    accent-color: var(--accent);
    width: 1rem;
    height: 1rem;
    margin: 0;
    flex: none;
    cursor: pointer;
  }
  .tickrow.all {
    font-weight: 600;
  }
  .ticknote,
  .tickhead {
    margin: 0;
    font-size: 0.72rem;
  }
  .tickhead {
    margin-top: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px solid var(--border);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .folderlist {
    display: flex;
    flex-direction: column;
    max-height: 14rem;
    overflow-y: auto;
  }
  .countline {
    margin: var(--space-2) 0 0;
    padding-top: var(--space-2);
    border-top: 1px solid var(--border);
    font-size: 0.78rem;
    color: var(--fg-2);
  }
</style>
