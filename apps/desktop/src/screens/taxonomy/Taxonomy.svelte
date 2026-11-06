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
  //
  // The view's two parts sit beside it (ADR-064): FieldForest (the left side) and FieldDetail (the
  // right side, the proposal review pane included). This file keeps the frame — header and totals,
  // close, the "Placing …" line, the error, the loading and empty states — and what the two sides
  // share: the selected field and which pane the right side shows.
  import type {
    FieldDetail as FieldDetailData,
    LabelledOption,
    LibraryDocument,
    ProposedEdge,
    TaxonomyView,
  } from '../../lib/core/types'
  import Icon from '../../shell/Icon.svelte'
  import FieldDetail from './FieldDetail.svelte'
  import FieldForest from './FieldForest.svelte'

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
    fieldDetail: FieldDetailData | null
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

  // Which pane the right column shows: one field's members, or the proposal review list.
  let pane = $state<'field' | 'proposals'>('field')

  // Local selection for immediate row highlight (the detail arrives a tick later from App).
  let selectedId = $state<string | null>(null)
  function selectField(id: string): void {
    pane = 'field'
    selectedId = id
    onSelectField(id)
  }

  const focusLabel = $derived(
    focusConceptId ? (concepts.find((c) => c.id === focusConceptId)?.label ?? null) : null,
  )

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
      <FieldForest {view} {proposals} {selectedId} bind:pane onSelectField={selectField} />

      <!-- RIGHT: the selected field's detail + placement controls -->
      <FieldDetail
        {pane}
        {proposals}
        {selectedId}
        {fieldDetail}
        {documents}
        {concepts}
        {focusConceptId}
        {onAddEdge}
        {onRemoveEdge}
        {onAttachDocument}
        {onDetachDocument}
      />
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
    width: min(94vw, 60rem);
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
  @media (max-width: 640px) {
    .modal {
      width: 94vw;
      max-height: 90vh;
    }
    .cols {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
