<script lang="ts">
  // One family row of Manage keywords, rendered by both lists (Concepts and Terms): its name (click
  // to rename), its document count, the concept toggle, Delete… with the question it asks on the
  // row, its forms, and the field that adds one. A concept's row marks each of its forms exact or
  // broad; a term's row has no such control, because only a concept's forms are counted (ADR-054).
  //
  // Owns the rename and the add. What is one-at-a-time across rows, or must outlive the row while a
  // search hides it, stays with the view: which row is being renamed and the typed form (both
  // bound), the delete question (Escape answers it there), and the marks being saved.
  import type { KeywordFamily, KeywordFamilyDeletion, MemberBreadth } from '../../lib/core/types'
  import { deletionLosses, formToAdd, markKey, shownBreadth } from '../../lib/library/library'
  import { shownName } from '../../lib/graph/labels'
  import Icon from '../../shell/Icon.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    family: f,
    highlighted,
    editingId = $bindable(),
    editingValue = $bindable(),
    addValue = $bindable(),
    deleteAsk,
    savingMarks,
    failedMarks,
    onRename,
    onSetOnGraph,
    onAddMember,
    onRemoveMember,
    onSetMark,
    onAskDelete,
    onKeep,
    onTakeOffGraphInstead,
    onConfirmDelete,
  }: {
    family: KeywordFamily
    highlighted: boolean // the row the view last went to (a deep-link, or "Go to family")
    editingId: string | null // the one row being renamed, across both lists
    editingValue: string
    addValue: string | undefined // what is typed in this row's add field
    deleteAsk: { id: string; loading: boolean; info: KeywordFamilyDeletion | null } | null
    savingMarks: Record<string, MemberBreadth> // markKey -> the mark being saved
    failedMarks: Record<string, MemberBreadth> // markKey -> the mark that did not save
    onRename: (familyId: string, canonical: string) => void
    onSetOnGraph: (familyId: string, include: boolean) => void
    onAddMember: (familyId: string, keyword: string) => void
    onRemoveMember: (familyId: string, keyword: string) => void
    onSetMark: (f: KeywordFamily, alias: string, clicked: 'exact' | 'broad') => Promise<void>
    onAskDelete: (f: KeywordFamily) => Promise<void>
    onKeep: () => void
    onTakeOffGraphInstead: () => void
    onConfirmDelete: () => void
  } = $props()

  const name = $derived(shownName(f.canonical, f.written ?? null))

  function startRename(): void {
    editingId = f.id
    editingValue = f.canonical
  }
  function commitRename(): void {
    if (editingId === null) return
    const v = editingValue.trim()
    if (v) onRename(editingId, v)
    editingId = null
  }
  function cancelRename(): void {
    editingId = null
  }

  // The field takes any text and offers the unassigned keywords as you type, from the one shared
  // list the view renders (`mk-unassigned`) — not a <select> per row, which froze the view.
  function submitAdd(): void {
    const form = formToAdd(f, addValue)
    if (form === null) return
    onAddMember(f.id, form)
    addValue = ''
  }

  function autofocus(node: HTMLElement): void {
    node.focus()
  }
</script>

<div class="famrow" id="fam-{f.id}" class:highlight={highlighted}>
  <div class="famhead">
    {#if editingId === f.id}
      <input
        class="renameinput"
        bind:value={editingValue}
        aria-label="Rename {name}"
        onkeydown={(e) => {
          if (e.key === 'Enter') commitRename()
          if (e.key === 'Escape') cancelRename()
        }}
      />
      <button class="iconbtn" onclick={commitRename} aria-label="Save name" type="button" use:tooltip={'Keep the new name'}>
        <Icon name="check" size={14} />
      </button>
      <button class="iconbtn" onclick={cancelRename} aria-label="Cancel rename" type="button" use:tooltip={'Keep the old name'}>
        <Icon name="x" size={14} />
      </button>
    {:else}
      <button class="famname" onclick={startRename} type="button" use:tooltip={'Change this name'}>
        {name}
      </button>
      <!-- The library filter's count: documents that carry the keyword. The graph counts
           mentions in the text, so the two numbers differ for the same concept, and a
           bare "1 doc" beside a concept the graph shows in eleven reads as a fault. -->
      <span
        class="doccount"
        title="Documents whose keywords carry this name or one of its forms — the library filter's count. The graph counts mentions in the text instead."
      >
        {f.doc_count} doc{f.doc_count === 1 ? '' : 's'}
      </span>
      <button
        class="iconbtn graph"
        class:on={f.graph_include}
        aria-pressed={f.graph_include}
        onclick={() => onSetOnGraph(f.id, !f.graph_include)}
        aria-label={f.graph_include
          ? `Take ${name} off the concept graph`
          : `Take ${name} on as a concept`}
        use:tooltip={f.graph_include
          ? 'A concept, on the graph — click to make it a term again'
          : 'A term — click to take it on as a concept'}
        type="button"
      >
        <Icon name="waypoints" size={14} />
      </button>
      <button
        class="iconbtn danger"
        onclick={() => void onAskDelete(f)}
        aria-label="Delete {name}"
        use:tooltip={'Delete this and its forms, after you confirm'}
        type="button"
      >
        <Icon name="x" size={14} />
      </button>
    {/if}
  </div>
  {#if deleteAsk !== null && deleteAsk.id === f.id}
    {@const losses = deleteAsk.info ? deletionLosses(deleteAsk.info) : []}
    <div class="confirm" role="group" aria-label="Delete {name}?">
      {#if deleteAsk.loading}
        <p class="confirmtext">Checking what goes with “{name}”…</p>
      {:else}
        <p class="confirmtext">
          {#if deleteAsk.info === null}
            Delete “{name}”? What goes with it could not be read.
          {:else if losses.length === 0}
            Delete “{name}”? Nothing else is attached to it.
          {:else}
            Delete “{name}”? This also removes {losses.join(', ')}.
          {/if}
          {#if f.graph_include}
            Taking it off the graph keeps all of that and makes it a term again.
          {/if}
        </p>
        <div class="confirmrow">
          <button class="btn" use:autofocus onclick={onKeep} type="button" use:tooltip={'Keep it; nothing is deleted'}>Keep</button>
          {#if f.graph_include}
            <button class="btn" onclick={onTakeOffGraphInstead} type="button" use:tooltip={'Keep it and its forms, as a term off the graph'}>
              Take off the graph instead
            </button>
          {/if}
          <button class="btn btn-primary btn-danger" onclick={onConfirmDelete} type="button" use:tooltip={'Delete it and what goes with it'}>Delete</button>
        </div>
      {/if}
    </div>
  {/if}
  <div class="members" role="group" aria-label="Forms of {name}">
    {#each f.aliases as alias (alias)}
      {#if f.graph_include}
        {@const key = markKey(f.id, alias)}
        {@const mark = shownBreadth(savingMarks, f, alias)}
        {@const saving = key in savingMarks}
        <span class="form" class:broad={mark === 'broad'}>
          <span class="formtext">{alias}</span>
          <span class="seg" role="group" aria-label="How “{alias}” counts for {name}">
            <button
              class="segbtn"
              class:on={mark === 'exact'}
              aria-pressed={mark === 'exact'}
              disabled={saving}
              onclick={() => void onSetMark(f, alias, 'exact')}
              use:tooltip={`Exact: “${alias}” always means ${name}, so its documents count.`}
              type="button"
            >
              exact
            </button>
            <button
              class="segbtn"
              class:on={mark === 'broad'}
              aria-pressed={mark === 'broad'}
              disabled={saving}
              onclick={() => void onSetMark(f, alias, 'broad')}
              use:tooltip={`Broad: “${alias}” also means other things, so its documents are listed beside ${name} and not counted in it.`}
              type="button"
            >
              broad
            </button>
          </span>
          {#if key in failedMarks}
            <span class="formerr" role="status">not saved — click again</span>
          {/if}
          <button
            class="formx"
            onclick={() => onRemoveMember(f.id, alias)}
            aria-label="Remove “{alias}” from {name}"
            use:tooltip={'Remove this form'}
            type="button"
          >
            <Icon name="x" size={11} />
          </button>
        </span>
      {:else}
        <button
          class="chip"
          onclick={() => onRemoveMember(f.id, alias)}
          type="button"
          use:tooltip={`Remove “${alias}” from this family`}
        >
          <span>{alias}</span>
          <Icon name="x" size={11} />
        </button>
      {/if}
    {:else}
      <span class="nomembers">
        {f.graph_include
          ? 'No other forms yet — just the name.'
          : 'No member keywords yet — just the name.'}
      </span>
    {/each}
  </div>
  <div class="addrow">
    <input
      class="addinput"
      list="mk-unassigned"
      bind:value={addValue}
      placeholder={f.graph_include
        ? 'Add a form — type to search the keywords, or type a new one'
        : 'Add a keyword — type to search'}
      aria-label={f.graph_include ? `Add a form to ${name}` : `Add a keyword to ${name}`}
      onkeydown={(e) => {
        if (e.key === 'Enter') submitAdd()
      }}
    />
    <button
      class="btn"
      onclick={submitAdd}
      disabled={formToAdd(f, addValue) === null}
      type="button"
      use:tooltip={formToAdd(f, addValue) === null ? 'Type or pick a keyword first' : f.graph_include ? 'Add this as a form of the concept' : 'Add this keyword to the family'}
    >
      Add
    </button>
  </div>
</div>

<style>
  .famrow.highlight {
    outline: 2px solid var(--accent, var(--fg-2));
    outline-offset: 2px;
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
  /* Allow-listed (ui-rules.allow.ts): an icon-only danger button in a dense row. Red at rest, as
     every danger action is (ADR-063); its name is the aria-label, and the delete asks on the row. */
  .iconbtn.danger,
  .iconbtn.danger:hover {
    color: var(--danger);
  }
  /* ADR-018 — the graph-vocabulary toggle. It is a two-state control on a row of one-shot
     actions, so "on" has to read as a state and not as a hover: filled accent, not a tint. */
  .iconbtn.graph.on {
    color: var(--accent);
    background: var(--surface-2);
  }
  .iconbtn.graph.on:hover {
    color: var(--accent);
  }
  .famrow {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .famhead {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .famname {
    font: inherit;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--fg);
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    text-align: left;
  }
  .famname:hover {
    text-decoration: underline;
  }
  .renameinput {
    flex: 1;
    min-width: 0;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.2rem 0.4rem;
    border: 1px solid var(--accent);
    border-radius: 6px;
    background: var(--bg);
    color: var(--fg);
  }
  .doccount {
    font-size: 0.72rem;
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    margin-left: auto;
    flex: none;
  }
  .members {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .nomembers {
    font-size: 0.75rem;
    color: var(--fg-2);
    font-style: italic;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font: inherit;
    font-size: 0.72rem;
    cursor: pointer;
    color: var(--fg);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.14rem 0.5rem;
  }
  .chip:hover {
    border-color: var(--danger);
    color: var(--danger);
  }
  /* ADR-054 — a concept's form: its text, how it counts, and a way to remove it. Three controls
     where a term's chip has one, so it is a container and not a button. */
  .form {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    color: var(--fg);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.1rem 0.25rem 0.1rem 0.5rem;
  }
  /* A broad form sits beside the concept, not in it: the dashed edge says so before the mark is
     read. */
  .form.broad {
    border-style: dashed;
    background: none;
  }
  .seg {
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 999px;
    overflow: hidden;
  }
  .segbtn {
    font: inherit;
    font-size: 0.66rem;
    cursor: pointer;
    color: var(--fg-2);
    background: var(--bg);
    border: none;
    padding: 0.05rem 0.4rem;
  }
  .segbtn + .segbtn {
    border-left: 1px solid var(--border);
  }
  .segbtn:hover {
    color: var(--fg);
  }
  /* A two-state mark, so "on" reads as a state and not as a hover: filled, like the graph toggle. */
  .segbtn.on {
    color: var(--accent-fg);
    background: var(--accent);
    font-weight: 600;
  }
  /* While a mark is being saved its two buttons take no click: the second click of a toggle
     would undo the first. */
  .segbtn:disabled {
    cursor: progress;
  }
  .formerr {
    font-size: 0.66rem;
    color: var(--danger);
  }
  /* Set apart from the marks: removing a form is not the third choice after exact and broad. */
  .formx {
    margin-left: 0.3rem;
    display: inline-flex;
    align-items: center;
    padding: 0.1rem;
    border: none;
    background: none;
    color: var(--fg-2);
    border-radius: 999px;
    cursor: pointer;
  }
  .formx:hover {
    color: var(--danger);
  }
  /* The delete question, on the row it is about. */
  .confirm {
    border: 1px solid var(--danger);
    border-radius: 6px;
    padding: 0.4rem 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .confirmtext {
    margin: 0;
    font-size: 0.78rem;
    color: var(--fg);
  }
  .confirmrow {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 0.35rem;
  }
  .addrow {
    display: flex;
    gap: 0.35rem;
  }
  .addinput {
    flex: 1;
    min-width: 0;
    font: inherit;
    font-size: 0.78rem;
    padding: 0.3rem 0.4rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg);
    color: var(--fg);
  }
</style>
