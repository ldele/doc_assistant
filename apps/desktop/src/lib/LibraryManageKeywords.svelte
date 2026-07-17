<script lang="ts">
  // Manage-keywords view (feature-tag-families.md, PR-1). Opened from the keyword-filter overlay's
  // "Manage keywords…" link (ADR-015 — curation is a dedicated view, distinct from filtering).
  // Curate families by hand: create one from un-familied keywords, rename its canonical label,
  // move keywords in/out (a keyword belongs to at most one family — adding it here moves it off
  // any other family), or delete it. Reuses the overlay's modal shell (scrim + centered dialog,
  // Esc-to-close). Dumb by design — App owns the families list + calls the API, then refreshes.
  import type { KeywordFamily, KeywordFamilyProposal } from './types'
  import Icon from './Icon.svelte'

  let {
    families,
    allKeywords,
    proposals,
    detecting,
    detectError,
    onCreate,
    onRename,
    onAddMember,
    onRemoveMember,
    onDelete,
    onDetect,
    onAcceptProposal,
    onDismissProposal,
    onClose,
  }: {
    families: KeywordFamily[]
    allKeywords: string[] // every raw keyword name across the corpus
    proposals: KeywordFamilyProposal[] // zero-LLM detection results (PR-2); [] until Detect runs
    detecting: boolean
    detectError: string | null
    onCreate: (canonical: string, members: string[]) => void
    onRename: (familyId: string, canonical: string) => void
    onAddMember: (familyId: string, keyword: string) => void
    onRemoveMember: (familyId: string, keyword: string) => void
    onDelete: (familyId: string) => void
    onDetect: () => void
    onAcceptProposal: (p: KeywordFamilyProposal) => void
    onDismissProposal: (canonical: string) => void
    onClose: () => void
  } = $props()

  // A keyword belongs to at most one family (ADR-015) — "un-familied" = not a canonical or
  // an alias of any family, matched case-insensitively (mirrors the backend's own matching).
  const familiedLower = $derived.by(() => {
    const s = new Set<string>()
    for (const f of families) {
      s.add(f.canonical.toLowerCase())
      for (const a of f.aliases) s.add(a.toLowerCase())
    }
    return s
  })
  const unfamilied = $derived(
    [...allKeywords].filter((k) => !familiedLower.has(k.toLowerCase())).sort((a, b) => a.localeCompare(b)),
  )

  let newCanonical = $state('')
  let newMembers = $state<string[]>([])
  function toggleNewMember(k: string): void {
    newMembers = newMembers.includes(k) ? newMembers.filter((x) => x !== k) : [...newMembers, k]
  }
  function submitCreate(): void {
    const canonical = newCanonical.trim()
    if (!canonical) return
    onCreate(canonical, newMembers)
    newCanonical = ''
    newMembers = []
  }

  let editingId = $state<string | null>(null)
  let editingValue = $state('')
  function startRename(f: KeywordFamily): void {
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

  // One pending "add member" pick per family row (family id -> the keyword chosen in its select).
  let addSelection = $state<Record<string, string>>({})
  function submitAdd(familyId: string): void {
    const kw = addSelection[familyId]
    if (!kw) return
    onAddMember(familyId, kw)
    addSelection = { ...addSelection, [familyId]: '' }
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
<div class="modal" role="dialog" aria-modal="true" aria-label="Manage keyword families">
  <div class="mhead">
    <h2>Manage keywords</h2>
    <button class="iconbtn" onclick={onClose} aria-label="Close" type="button">
      <Icon name="x" size={16} />
    </button>
  </div>

  <div class="body">
    <section class="block">
      <h3>Detect proposals</h3>
      <p class="hint">
        Zero-LLM, on request — checks keyword spelling (morphology) and meaning (bge embedding)
        for near-duplicates. Nothing is written until you accept a proposal below.
      </p>
      <button class="secondary" onclick={onDetect} disabled={detecting} type="button">
        {detecting ? 'Detecting…' : 'Detect'}
      </button>
      {#if detectError}
        <p class="error">{detectError}</p>
      {/if}
      {#if proposals.length > 0}
        <div class="proplist" role="group" aria-label="Detected family proposals">
          {#each proposals as p (p.canonical)}
            <div class="proprow">
              <span class="proptier" class:embedding={p.tier === 'embedding'}>
                {p.tier === 'morphological' ? 'spelling' : 'meaning'}
              </span>
              <span class="proptext">
                <strong>{p.canonical}</strong>
                <span class="propmembers">+ {p.members.join(', ')}</span>
              </span>
              <span class="propconf">{Math.round(p.confidence * 100)}%</span>
              <button class="propaccept" onclick={() => onAcceptProposal(p)} type="button">
                Accept
              </button>
              <button
                class="iconbtn"
                onclick={() => onDismissProposal(p.canonical)}
                aria-label="Dismiss proposal for {p.canonical}"
                title="Dismiss"
                type="button"
              >
                <Icon name="x" size={14} />
              </button>
            </div>
          {/each}
        </div>
      {:else if !detecting && !detectError}
        <p class="hint">No proposals yet — run Detect to check for near-duplicate keywords.</p>
      {/if}
    </section>

    <section class="block">
      <h3>New family</h3>
      <p class="hint">
        Collapse near-duplicate keywords (e.g. “llm” / “llms”) into one canonical tag.
      </p>
      <div class="createrow">
        <input
          use:autofocus
          bind:value={newCanonical}
          placeholder="Canonical name, e.g. “large language models”"
          aria-label="New family canonical name"
          onkeydown={(e) => e.key === 'Enter' && submitCreate()}
        />
        <button class="primary" onclick={submitCreate} disabled={!newCanonical.trim()} type="button">
          Create
        </button>
      </div>
      {#if unfamilied.length > 0}
        <div class="pickrow" role="group" aria-label="Member keywords for the new family">
          {#each unfamilied as k (k)}
            <button
              class="pick"
              class:on={newMembers.includes(k)}
              onclick={() => toggleNewMember(k)}
              type="button"
              aria-pressed={newMembers.includes(k)}
            >
              {#if newMembers.includes(k)}<Icon name="check" size={11} />{/if}
              {k}
            </button>
          {/each}
        </div>
      {/if}
    </section>

    <section class="block">
      <h3>Families ({families.length})</h3>
      {#if families.length === 0}
        <p class="hint">No families yet — create one above.</p>
      {:else}
        <div class="famlist">
          {#each families as f (f.id)}
            <div class="famrow">
              <div class="famhead">
                {#if editingId === f.id}
                  <input
                    class="renameinput"
                    bind:value={editingValue}
                    aria-label="Rename family"
                    onkeydown={(e) => {
                      if (e.key === 'Enter') commitRename()
                      if (e.key === 'Escape') cancelRename()
                    }}
                  />
                  <button class="iconbtn" onclick={commitRename} aria-label="Save name" type="button">
                    <Icon name="check" size={14} />
                  </button>
                  <button class="iconbtn" onclick={cancelRename} aria-label="Cancel rename" type="button">
                    <Icon name="x" size={14} />
                  </button>
                {:else}
                  <button class="famname" onclick={() => startRename(f)} type="button" title="Rename">
                    {f.canonical}
                  </button>
                  <span class="doccount">{f.doc_count} doc{f.doc_count === 1 ? '' : 's'}</span>
                  <button
                    class="iconbtn danger"
                    onclick={() => onDelete(f.id)}
                    aria-label="Delete family {f.canonical}"
                    title="Delete family"
                    type="button"
                  >
                    <Icon name="x" size={14} />
                  </button>
                {/if}
              </div>
              <div class="members" role="group" aria-label="Members of {f.canonical}">
                {#each f.aliases as alias (alias)}
                  <button
                    class="chip"
                    onclick={() => onRemoveMember(f.id, alias)}
                    type="button"
                    title="Remove “{alias}” from this family"
                  >
                    <span>{alias}</span>
                    <Icon name="x" size={11} />
                  </button>
                {:else}
                  <span class="nomembers">No member keywords yet — just the canonical name.</span>
                {/each}
              </div>
              {#if unfamilied.length > 0}
                <div class="addrow">
                  <select
                    bind:value={addSelection[f.id]}
                    aria-label="Add a keyword to {f.canonical}"
                  >
                    <option value="">Add a keyword…</option>
                    {#each unfamilied as k (k)}
                      <option value={k}>{k}</option>
                    {/each}
                  </select>
                  <button
                    class="addbtn"
                    onclick={() => submitAdd(f.id)}
                    disabled={!addSelection[f.id]}
                    type="button"
                  >
                    Add
                  </button>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </section>
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
  .createrow {
    display: flex;
    gap: 0.4rem;
  }
  .createrow input {
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
  .createrow input:focus {
    outline: none;
    border-color: var(--accent);
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
  }
  .secondary:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .secondary:disabled {
    opacity: 0.6;
    cursor: wait;
  }
  .error {
    margin: 0.4rem 0 0;
    font-size: 0.78rem;
    color: var(--danger, #c0392b);
  }
  .proplist {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin-top: 0.5rem;
  }
  .proprow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.55rem;
  }
  .proptier {
    flex: none;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--fg-2);
    background: var(--surface-2);
    border-radius: 999px;
    padding: 0.1rem 0.45rem;
  }
  .proptier.embedding {
    color: var(--accent-fg);
    background: var(--accent);
  }
  .proptext {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    font-size: 0.82rem;
    overflow: hidden;
  }
  .propmembers {
    color: var(--fg-2);
    font-size: 0.75rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .propconf {
    flex: none;
    font-size: 0.7rem;
    font-variant-numeric: tabular-nums;
    color: var(--fg-2);
  }
  .propaccept {
    flex: none;
    font: inherit;
    font-size: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--accent);
    background: none;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
  }
  .propaccept:hover {
    color: var(--accent-fg);
    background: var(--accent);
  }
  .pickrow {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.5rem;
  }
  .pick {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font: inherit;
    font-size: 0.72rem;
    cursor: pointer;
    color: var(--fg-2);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.14rem 0.5rem;
  }
  .pick:hover {
    border-color: var(--accent);
  }
  .pick.on {
    color: var(--accent-fg);
    background: var(--accent);
    border-color: var(--accent);
    font-weight: 600;
  }
  .famlist {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
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
    border-color: var(--danger, #c0392b);
    color: var(--danger, #c0392b);
  }
  .addrow {
    display: flex;
    gap: 0.35rem;
  }
  .addrow select {
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
  .addbtn {
    font: inherit;
    font-size: 0.75rem;
    cursor: pointer;
    color: var(--fg-2);
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.25rem 0.55rem;
    flex: none;
  }
  .addbtn:hover:not(:disabled) {
    color: var(--fg);
    border-color: var(--accent);
  }
  .addbtn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  @media (max-width: 640px) {
    .modal {
      width: 94vw;
      max-height: 90vh;
    }
  }
</style>
