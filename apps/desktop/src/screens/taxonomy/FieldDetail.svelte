<script lang="ts">
  // A field's detail, the right side of the Taxonomy view: what is placed under the selected field
  // — its concepts and documents, a proposed one marked as such with Accept and Reject — and the
  // two pickers that attach more. While the field forest's *Proposed placements* row is pressed,
  // the same side shows that review list instead (TX3b): every auto-proposed link, `is_a` ones
  // included, since those hang under no field. Owns the pickers and the writes each control makes;
  // which field is selected and which pane shows are the view's.
  import type {
    FieldDetail,
    LabelledOption,
    LibraryDocument,
    ProposedEdge,
  } from '../../lib/core/types'
  import { docLabel } from '../../lib/library/library'
  // A concept is shown under the name the library writes it by (ADR-054) — the same helper the
  // graph uses, so `Cre` here is `Cre` there. Fields and documents carry no written form.
  import { shownLabel, shownName } from '../../lib/graph/labels'
  import Icon from '../../shell/Icon.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    pane,
    proposals,
    selectedId,
    fieldDetail,
    documents,
    concepts,
    focusConceptId,
    onAddEdge,
    onRemoveEdge,
    onAttachDocument,
    onDetachDocument,
  }: {
    pane: 'field' | 'proposals'
    proposals: ProposedEdge[] // auto-proposed edges awaiting accept-or-delete (ADR-028 D8)
    selectedId: string | null // the row last clicked in the forest
    fieldDetail: FieldDetail | null
    documents: LibraryDocument[]
    concepts: LabelledOption[] // the attachable vocabulary (graph nodes, spec ledger #7)
    focusConceptId: string | null // graph deep-link: preselect this concept for placement
    onAddEdge: (body: { source_id: string; target_id: string; type: 'is_a' | 'in_field' }) => void
    onRemoveEdge: (body: { source_id: string; target_id: string; type: 'is_a' | 'in_field' }) => void
    onAttachDocument: (docId: string, fieldId: string) => void
    onDetachDocument: (docId: string, fieldId: string) => void
  } = $props()

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
</script>

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
            <strong>{shownName(p.source_label, p.source_written ?? null)}</strong>
            <span class="ptype">{p.type === 'is_a' ? 'is a kind of' : 'belongs to'}</span>
            <strong>{shownName(p.target_label, p.target_written ?? null)}</strong>
            {#if p.source_kind === 'document'}<span class="ptag">document</span>{/if}
          </span>
          <span class="pactions">
            <button
              class="okbtn"
              onclick={() => acceptProposal(p)}
              type="button"
              use:tooltip={'Accept — keep this as your own placement'}
            >
              <Icon name="check" size={13} /> Accept
            </button>
            <button
              class="nobtn"
              onclick={() => rejectProposal(p)}
              type="button"
              use:tooltip={'Reject — remove this proposal'}
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
              <span>{shownLabel(c)}</span>
              <button
                class="chipbtn"
                onclick={() => acceptConcept(c.id)}
                type="button"
                aria-label="Accept “{shownLabel(c)}” on this field"
                use:tooltip={'Keep this as your own placement'}
              >
                <Icon name="check" size={11} />
              </button>
              <button
                class="chipbtn"
                onclick={() => removeConcept(c.id)}
                type="button"
                aria-label="Reject “{shownLabel(c)}” on this field"
                use:tooltip={'Remove this proposal'}
              >
                <Icon name="x" size={11} />
              </button>
            </span>
          {:else}
            <button
              class="chip"
              onclick={() => removeConcept(c.id)}
              type="button"
              use:tooltip={`Remove “${shownLabel(c)}” from this field`}
            >
              <span>{shownLabel(c)}</span>
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
          type="button" use:tooltip={!conceptToAdd ? 'Choose a concept first' : 'Place this concept in this field'}>Attach</button
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
                use:tooltip={'Keep this as your own placement'}
              >
                <Icon name="check" size={11} />
              </button>
              <button
                class="chipbtn"
                onclick={() => onDetachDocument(d.id, detail.id)}
                type="button"
                aria-label="Reject “{d.label}” on this field"
                use:tooltip={'Remove this proposal'}
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
          type="button" use:tooltip={!docToAdd ? 'Choose a document first' : 'Place this document in this field'}>Attach</button
        >
      </div>
    </div>
  {/if}
</section>

<style>
  .totals {
    font-size: 0.76rem;
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
  }
  .rollnote {
    opacity: 0.7;
  }
  .hint {
    font-size: 0.78rem;
    color: var(--fg-2);
    margin: 0;
  }
  .hint.pad {
    padding: 0.6rem 0.2rem;
  }
  /* The field forest's frame, except that this side scrolls. */
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
</style>
