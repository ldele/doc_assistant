<script lang="ts">
  // Taxonomy view (docs/specs/feature-taxonomy-view.md, ADR-028 increment 2b). A dedicated modal
  // that renders the curated field forest and *places* concepts/documents onto it (ADR-019 D11 — a
  // dedicated taxonomy surface owns tree edits). Reuses the overlay shell (scrim + centred dialog +
  // Esc) like Manage-keywords. Dumb by design: App owns the data + calls the API, then refreshes.
  //
  // Placement-only (spec ledger #6): attach/detach a concept to a field (an `in_field` edge) and
  // attach a document to a field. No field→field re-parenting here — that is the only edit that can
  // trip the 409 cycle guard, and it stays API-only in 2b.
  //
  // Increment 3b (TX3b, ROADMAP 51): the review side of ADR-028 D8. Anything auto-proposed is
  // marked as a proposal wherever it appears and carries accept (a curated write of the same link,
  // which promotes it) and reject (the delete). `is_a` proposals hang under no field, so they get
  // their own pane — the list is the only place a concept→concept proposal is visible.
  import type {
    FieldDetail,
    LabelledOption,
    LibraryDocument,
    ProposedEdge,
    TaxonomyView,
  } from '../core/types'
  import { buildForest } from './taxonomy'
  import { docLabel } from './library'
  import Icon from '../shell/Icon.svelte'

  let {
    view,
    fieldDetail,
    proposals,
    loading,
    error,
    documents,
    concepts,
    focusConceptId,
    onSelectField,
    onAddEdge,
    onRemoveEdge,
    onAttachDocument,
    onDetachDocument,
    onClose,
  }: {
    view: TaxonomyView | null
    fieldDetail: FieldDetail | null
    proposals: ProposedEdge[] // auto-proposed edges awaiting accept-or-delete (ADR-028 D8)
    loading: boolean
    error: string | null
    documents: LibraryDocument[]
    concepts: LabelledOption[] // the attachable vocabulary (graph nodes, spec ledger #7 — no link,
    // so no origin: an *attached* member is a FieldMember, an offerable one is not
    focusConceptId: string | null // graph deep-link: preselect this concept for placement
    onSelectField: (fieldId: string) => void
    onAddEdge: (body: { source_id: string; target_id: string; type: 'is_a' | 'in_field' }) => void
    onRemoveEdge: (body: { source_id: string; target_id: string; type: 'is_a' | 'in_field' }) => void
    onAttachDocument: (docId: string, fieldId: string) => void
    onDetachDocument: (docId: string, fieldId: string) => void
    onClose: () => void
  } = $props()

  // The forest as ordered display rows (root→child, depth for indent). Empty until `view` loads.
  const rows = $derived(view ? buildForest(view) : [])

  // Field-label search. When active, matched fields render flat (depth 0) — the same "flatten the
  // pool on search" pattern Manage-keywords uses; the full indented forest returns when it clears.
  let query = $state('')
  const shownRows = $derived.by(() => {
    const q = query.trim().toLowerCase()
    if (q === '') return rows
    return rows
      .filter((r) => r.field.label.toLowerCase().includes(q))
      .map((r) => ({ ...r, depth: 0 }))
  })

  // Which pane the right column shows: one field's members, or the proposal review list.
  let pane = $state<'field' | 'proposals'>('field')

  // Local selection for immediate row highlight (the detail arrives a tick later from App).
  let selectedId = $state<string | null>(null)
  function selectField(id: string): void {
    pane = 'field'
    selectedId = id
    onSelectField(id)
  }

  // Accept = write the same link as curated; the backend promotes the proposed row in place
  // rather than duplicating it (ADR-028 D8). Reject = delete it. Both re-read from the server.
  // A document proposal is the same review unit through the other pair of routes — `source_kind`
  // is what tells them apart, since a document is not a node in the hierarchy table.
  function acceptProposal(p: ProposedEdge): void {
    if (p.source_kind === 'document') onAttachDocument(p.source_id, p.target_id)
    else onAddEdge({ source_id: p.source_id, target_id: p.target_id, type: p.type })
  }
  function rejectProposal(p: ProposedEdge): void {
    if (p.source_kind === 'document') onDetachDocument(p.source_id, p.target_id)
    else onRemoveEdge({ source_id: p.source_id, target_id: p.target_id, type: p.type })
  }
  // The detail is authoritative only when it matches the row the user last clicked.
  const detail = $derived(fieldDetail && fieldDetail.id === selectedId ? fieldDetail : null)

  const focusLabel = $derived(
    focusConceptId ? (concepts.find((c) => c.id === focusConceptId)?.label ?? null) : null,
  )

  // Attach-concept picker: the vocabulary minus what is already on this field.
  const attachedConceptIds = $derived(new Set((detail?.concepts ?? []).map((c) => c.id)))
  const attachableConcepts = $derived(
    concepts
      .filter((c) => !attachedConceptIds.has(c.id))
      .sort((a, b) => a.label.localeCompare(b.label)),
  )

  let conceptToAdd = $state('')
  let docToAdd = $state('')
  // A graph deep-link arrives with a concept to place — preselect it so placing is one field click
  // + one Add. Only writes when a focus id is actually supplied, so it never clobbers a manual pick.
  $effect(() => {
    if (focusConceptId) conceptToAdd = focusConceptId
  })

  function submitAddConcept(): void {
    if (!selectedId || !conceptToAdd) return
    onAddEdge({ source_id: conceptToAdd, target_id: selectedId, type: 'in_field' })
    conceptToAdd = ''
  }
  function submitAttachDocument(): void {
    if (!selectedId || !docToAdd) return
    onAttachDocument(docToAdd, selectedId)
    docToAdd = ''
  }
  function removeConcept(conceptId: string): void {
    if (!selectedId) return
    onRemoveEdge({ source_id: conceptId, target_id: selectedId, type: 'in_field' })
  }
  function acceptConcept(conceptId: string): void {
    if (!selectedId) return
    onAddEdge({ source_id: conceptId, target_id: selectedId, type: 'in_field' })
  }

  const docLabelById = $derived.by(() => {
    const m = new Map<string, string>()
    for (const d of documents) m.set(d.id, docLabel(d))
    return m
  })

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="Taxonomy">
  <div class="mhead">
    <div class="titlewrap">
      <h2>Taxonomy</h2>
      {#if view}
        <span class="totals">
          {view.n_concepts_total} graph concept{view.n_concepts_total === 1 ? '' : 's'} ·
          {view.n_documents_total} document{view.n_documents_total === 1 ? '' : 's'} ·
          {view.fields.length} fields
          {#if view.n_unassigned_concepts > 0}
            · <span class="unplaced">{view.n_unassigned_concepts} not yet placed</span>
          {/if}
        </span>
      {/if}
    </div>
    <button class="iconbtn" onclick={onClose} aria-label="Close" type="button">
      <Icon name="x" size={16} />
    </button>
  </div>

  {#if focusLabel}
    <p class="placing">
      <Icon name="tag" size={13} /> Placing <strong>{focusLabel}</strong> — pick a field, then
      Attach.
    </p>
  {/if}

  {#if error}
    <p class="error">{error}</p>
  {/if}

  {#if loading && !view}
    <p class="hint pad">Loading the taxonomy…</p>
  {:else if view && view.fields.length === 0}
    <div class="empty">
      <span class="state-mark"><Icon name="folder" size={26} /></span>
      <strong>No fields seeded yet</strong>
      <p>Run <code>scripts/seed_taxonomy --apply</code> to load the ANZSRC field trunk.</p>
    </div>
  {:else if view}
    <div class="cols">
      <!-- LEFT: the field forest -->
      <section class="forest" aria-label="Field forest">
        {#if proposals.length > 0}
          <button
            class="proposalsrow"
            class:sel={pane === 'proposals'}
            onclick={() => (pane = 'proposals')}
            type="button"
            aria-pressed={pane === 'proposals'}
          >
            <Icon name="tag" size={13} />
            <span>Proposed placements</span>
            <span class="pcount">{proposals.length}</span>
          </button>
        {/if}
        <div class="searchrow small">
          <Icon name="search" size={13} />
          <input bind:value={query} placeholder="Search fields" aria-label="Search fields" />
        </div>
        <div class="treescroll">
          {#each shownRows as row (row.field.id + ':' + row.depth)}
            <button
              class="fieldrow"
              class:sel={selectedId === row.field.id}
              style="padding-left: {0.5 + row.depth * 0.95}rem"
              onclick={() => selectField(row.field.id)}
              type="button"
              aria-pressed={selectedId === row.field.id}
            >
              <span class="flabel">{row.field.label}</span>
              <span
                class="fcount"
                class:zero={row.field.n_concepts_rollup === 0 && row.field.n_documents_rollup === 0}
                title="Concepts · documents under this field (including narrower fields)"
              >
                {row.field.n_concepts_rollup} · {row.field.n_documents_rollup}
              </span>
            </button>
          {:else}
            <p class="hint pad">No fields match “{query.trim()}”.</p>
          {/each}
        </div>
      </section>

      <!-- RIGHT: the selected field's detail + placement controls -->
      <section class="detail" aria-label="Field detail">
        {#if pane === 'proposals'}
          <div class="dhead">
            <h3>Proposed placements</h3>
            <span class="totals">{proposals.length} awaiting review</span>
          </div>
          <p class="hint">
            Suggested by a pass over the corpus, never written as fact. Accept one to make it
            yours, or reject it — nothing else changes until you do.
          </p>
          <div class="plist">
            {#each proposals as p (p.source_id + ':' + p.target_id + ':' + p.type)}
              <div class="prow">
                <span class="ptext">
                  <strong>{p.source_label}</strong>
                  <span class="ptype">{p.type === 'is_a' ? 'is a kind of' : 'belongs to'}</span>
                  <strong>{p.target_label}</strong>
                  {#if p.source_kind === 'document'}<span class="ptag">document</span>{/if}
                </span>
                <span class="pactions">
                  <button
                    class="okbtn"
                    onclick={() => acceptProposal(p)}
                    type="button"
                    title="Accept — keep this as your own placement"
                  >
                    <Icon name="check" size={13} /> Accept
                  </button>
                  <button
                    class="nobtn"
                    onclick={() => rejectProposal(p)}
                    type="button"
                    title="Reject — remove this proposal"
                  >
                    <Icon name="x" size={13} /> Reject
                  </button>
                </span>
              </div>
            {:else}
              <span class="nomembers">
                Nothing left to review — every proposal has been accepted or rejected.
              </span>
            {/each}
          </div>
        {:else if !selectedId}
          <p class="hint pad">Select a field to see or edit what is placed under it.</p>
        {:else if !detail}
          <p class="hint pad">Loading field…</p>
        {:else}
          <div class="dhead">
            <h3>{detail.label}</h3>
            <span class="totals">
              {detail.n_concepts_rollup} concept{detail.n_concepts_rollup === 1 ? '' : 's'} ·
              {detail.n_documents_rollup} document{detail.n_documents_rollup === 1 ? '' : 's'}
              <span class="rollnote">(incl. narrower)</span>
            </span>
          </div>

          <div class="block">
            <h4>Concepts here ({detail.concepts.length})</h4>
            <div class="chips">
              {#each detail.concepts as c (c.id)}
                {#if c.origin === 'proposed'}
                  <span class="chip proposed" title="Proposed, not yours yet — accept or reject">
                    <span>{c.label}</span>
                    <button
                      class="chipbtn"
                      onclick={() => acceptConcept(c.id)}
                      type="button"
                      aria-label="Accept “{c.label}” on this field"
                    >
                      <Icon name="check" size={11} />
                    </button>
                    <button
                      class="chipbtn"
                      onclick={() => removeConcept(c.id)}
                      type="button"
                      aria-label="Reject “{c.label}” on this field"
                    >
                      <Icon name="x" size={11} />
                    </button>
                  </span>
                {:else}
                  <button
                    class="chip"
                    onclick={() => removeConcept(c.id)}
                    type="button"
                    title="Remove “{c.label}” from this field"
                  >
                    <span>{c.label}</span>
                    <Icon name="x" size={11} />
                  </button>
                {/if}
              {:else}
                <span class="nomembers">No concepts placed here directly yet.</span>
              {/each}
            </div>
            <div class="addrow">
              <select bind:value={conceptToAdd} aria-label="Concept to attach to {detail.label}">
                <option value="">Attach a concept…</option>
                {#each attachableConcepts as c (c.id)}
                  <option value={c.id}>{c.label}</option>
                {/each}
              </select>
              <button
                class="addbtn"
                onclick={submitAddConcept}
                disabled={!conceptToAdd}
                type="button">Attach</button
              >
            </div>
          </div>

          <div class="block">
            <h4>Documents here ({detail.documents.length})</h4>
            <div class="doclist">
              {#each detail.documents as d (d.id)}
                {#if d.origin === 'proposed'}
                  <span class="docrow proposed" title="Proposed, not yours yet — accept or reject">
                    <span class="dlabel">{d.label}</span>
                    <button
                      class="chipbtn"
                      onclick={() => onAttachDocument(d.id, detail.id)}
                      type="button"
                      aria-label="Accept “{d.label}” on this field"
                    >
                      <Icon name="check" size={11} />
                    </button>
                    <button
                      class="chipbtn"
                      onclick={() => onDetachDocument(d.id, detail.id)}
                      type="button"
                      aria-label="Reject “{d.label}” on this field"
                    >
                      <Icon name="x" size={11} />
                    </button>
                  </span>
                {:else}
                  <span class="docrow" title={d.label}>{d.label}</span>
                {/if}
              {:else}
                <span class="nomembers">No documents placed here directly yet.</span>
              {/each}
            </div>
            <div class="addrow">
              <select bind:value={docToAdd} aria-label="Document to attach to {detail.label}">
                <option value="">Attach a document…</option>
                {#each documents as d (d.id)}
                  <option value={d.id}>{docLabelById.get(d.id)}</option>
                {/each}
              </select>
              <button
                class="addbtn"
                onclick={submitAttachDocument}
                disabled={!docToAdd}
                type="button">Attach</button
              >
            </div>
          </div>
        {/if}
      </section>
    </div>
  {/if}
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
    width: min(94vw, 880px);
    max-height: min(88vh, 760px);
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
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-2);
    flex: none;
  }
  .titlewrap {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
  }
  .mhead h2 {
    margin: 0;
    font-size: var(--text-title);
    font-family: var(--font-serif);
  }
  .totals {
    font-size: 0.76rem;
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
  }
  .unplaced {
    color: var(--warn-fg, var(--accent));
    font-weight: 600;
  }
  .rollnote {
    opacity: 0.7;
  }
  .placing {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin: 0;
    font-size: 0.8rem;
    color: var(--fg);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.55rem;
    flex: none;
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
  .error {
    margin: 0;
    font-size: 0.78rem;
    color: var(--danger, #c0392b);
    flex: none;
  }
  .hint {
    font-size: 0.78rem;
    color: var(--fg-2);
    margin: 0;
  }
  .hint.pad {
    padding: 0.6rem 0.2rem;
  }
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
    text-align: center;
    padding: 2rem 1rem;
    color: var(--fg-2);
  }
  .empty strong {
    color: var(--fg);
  }
  .empty code {
    font-size: 0.85em;
    background: var(--surface-2);
    padding: 0.05rem 0.3rem;
    border-radius: 4px;
  }
  .state-mark {
    color: var(--fg-2);
    opacity: 0.7;
  }
  .cols {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--space-3);
  }
  .forest,
  .detail {
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem;
    overflow: hidden;
  }
  .searchrow.small {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex: none;
    border: 1px solid var(--line, var(--border));
    border-radius: var(--radius-sm, 6px);
    padding: 0.2rem 0.45rem;
    color: var(--fg-2);
  }
  .searchrow.small input {
    flex: 1;
    min-width: 0;
    border: none;
    background: none;
    color: var(--fg);
    font: inherit;
    font-size: 0.8rem;
    outline: none;
  }
  .treescroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  .fieldrow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    text-align: left;
    font: inherit;
    font-size: 0.82rem;
    color: var(--fg);
    background: none;
    border: none;
    border-radius: 6px;
    padding: 0.28rem 0.4rem;
    cursor: pointer;
  }
  .fieldrow:hover {
    background: var(--surface-2);
  }
  .fieldrow.sel {
    background: var(--accent);
    color: var(--accent-fg);
  }
  .flabel {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .fcount {
    flex: none;
    font-size: 0.7rem;
    font-variant-numeric: tabular-nums;
    color: var(--fg-2);
  }
  .fieldrow.sel .fcount {
    color: var(--accent-fg);
  }
  .fcount.zero {
    opacity: 0.45;
  }
  .detail {
    overflow-y: auto;
  }
  .dhead {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    flex: none;
  }
  .dhead h3 {
    margin: 0;
    font-size: 0.95rem;
    font-family: var(--font-serif);
    color: var(--fg);
  }
  .block {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .block h4 {
    margin: 0;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--fg-2);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font: inherit;
    font-size: 0.74rem;
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
  /* A proposal is never styled as a placement the user made: dashed edge, its own two actions. */
  .chip.proposed,
  .docrow.proposed {
    border: 1px dashed var(--warn-fg, var(--accent));
    background: color-mix(in srgb, var(--warn-fg, var(--accent)) 8%, transparent);
  }
  .docrow.proposed {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    border-radius: 6px;
    padding: 0.1rem 0.4rem;
  }
  .dlabel {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .chipbtn {
    display: inline-flex;
    align-items: center;
    font: inherit;
    color: var(--fg-2);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
  }
  .chipbtn:hover {
    color: var(--fg);
  }
  .proposalsrow {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    width: 100%;
    font: inherit;
    font-size: 0.8rem;
    color: var(--fg);
    background: color-mix(in srgb, var(--warn-fg, var(--accent)) 10%, transparent);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.45rem;
    cursor: pointer;
    text-align: left;
  }
  .proposalsrow.sel {
    background: var(--accent);
    color: var(--accent-fg);
  }
  .pcount {
    margin-left: auto;
    font-variant-numeric: tabular-nums;
    font-size: 0.76rem;
  }
  .plist {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    overflow-y: auto;
  }
  .prow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.5rem;
  }
  .ptext {
    font-size: 0.8rem;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ptype {
    color: var(--fg-2);
    font-size: 0.74rem;
    margin: 0 0.25rem;
  }
  .ptag {
    margin-left: 0.35rem;
    font-size: 0.68rem;
    color: var(--fg-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0 0.32rem;
  }
  .pactions {
    display: flex;
    gap: 0.3rem;
    margin-left: auto;
    flex: none;
  }
  .okbtn,
  .nobtn {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font: inherit;
    font-size: 0.74rem;
    color: var(--fg);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.16rem 0.44rem;
    cursor: pointer;
  }
  .okbtn:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .nobtn:hover {
    border-color: var(--danger, #c0392b);
    color: var(--danger, #c0392b);
  }
  .nomembers {
    font-size: 0.76rem;
    color: var(--fg-2);
    font-style: italic;
  }
  .doclist {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .docrow {
    font-size: 0.78rem;
    color: var(--fg);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
    padding: 0.25rem 0.6rem;
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
    .cols {
      grid-template-columns: minmax(0, 1fr);
    }
    /* Stacked on mobile: cap the forest height so the detail stays reachable below it. */
    .forest {
      max-height: 40vh;
    }
  }
</style>
