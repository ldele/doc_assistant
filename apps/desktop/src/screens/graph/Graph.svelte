<script lang="ts">
  // The Graph tab (docs/specs/feature-concept-graph.md PR-G2a, ADR-017). A DESTINATION, not a
  // modal: selecting a concept (in the sidebar's ConceptIndex rail, which App composes) opens the
  // concept panel (ConceptPanel.svelte): a depth-1 ego graph and the documents that lead from the
  // concept to the chunks where it appears. Selection + the under-connected lens are App-owned
  // props — the rail and the panel must agree. Read-only for the vocabulary (ADR-017 A1): the only
  // write is a deep-link to the Manage-keywords view. The graph observes; it never edits.
  //
  // This file is the tab around the panel: the staleness notice, the graph bar (coverage, Rebuild,
  // Manage keywords), the loading, error and empty states, and the selection wiring — what a new
  // selection resets, and the fetch of where the concept appears.
  import type {
    ConceptGraph,
    ConceptPresence,
    GraphRebuildStatus,
    LibraryDocument,
  } from '../../lib/core/types'
  import { graphCoverage } from '../../lib/graph/gaps'
  import { formsChangedNotice, shownLabel } from '../../lib/graph/labels'
  import Icon from '../../shell/Icon.svelte'
  import ConceptPanel from './ConceptPanel.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    graph,
    loading,
    error,
    documents,
    rebuildState,
    selectedId,
    showUnderConnected,
    onRebuild,
    onOpenDocument,
    onManageConcept,
    onCurateVocabulary,
    onPlaceConcept,
    onSelectConcept,
    loadPresence,
    offGraphConcept = null,
    onOpenPassage,
  }: {
    graph: ConceptGraph | null
    loading: boolean
    error: string | null
    documents: LibraryDocument[]
    rebuildState: GraphRebuildStatus['state']
    selectedId: string | null
    showUnderConnected: boolean
    onRebuild: () => void
    onOpenDocument: (docId: string) => void
    onManageConcept: (conceptId: string, label: string) => void
    /** Open the curation view with no concept preselected — the empty state's primary
     *  action. Separate from `onManageConcept` rather than called with empty arguments:
     *  "curate the vocabulary" and "edit this concept" are different intents, and a call
     *  site passing `('', '')` is the kind of thing that survives long enough to mislead. */
    onCurateVocabulary: () => void
    onPlaceConcept: (conceptId: string, label: string) => void
    onSelectConcept: (id: string) => void
    loadPresence: (conceptId: string) => Promise<ConceptPresence[]>
    /** A concept reached by the vocabulary search that is not a graph node (ADR-053): the
     *  panel shows its definition without a neighbourhood. */
    offGraphConcept?: { id: string; label: string } | null
    /** Open a passage where it was written — the path a chat citation takes. */
    onOpenPassage: (chunkKey: string) => void
  } = $props()

  // What a new selection resets. Kept here, not in the concept panel: the panel is unmounted while
  // the graph reloads (a rebuild passes through the loading state), and this state outlives that;
  // only a new selection resets it. The panel writes the open row and the pan back (`bind:`).
  let presence = $state<ConceptPresence[]>([])
  let presenceLoading = $state(false)
  let expandedDocId = $state<string | null>(null)
  // Pan is position-specific (it resets when the ego re-centres); zoom, a real preference, is the
  // panel's and persisted. Pan restarts at 0 on each new selection.
  let panX = $state(0)
  let panY = $state(0)

  // Selection arrives as a prop (the rail lives in the sidebar): each change resets the per-concept
  // panel state and fetches presence. The cancellation flag guards rapid re-selection — a stale
  // response must not overwrite the newer concept's data.
  $effect(() => {
    const id = selectedId
    expandedDocId = null
    presence = []
    panX = 0
    panY = 0
    if (id === null) {
      presenceLoading = false
      return
    }
    let cancelled = false
    presenceLoading = true
    loadPresence(id)
      .then((p) => {
        if (!cancelled) presence = p
      })
      .catch(() => {
        if (!cancelled) presence = [] // navigation degrades to doc-level; the ego still renders
      })
      .finally(() => {
        if (!cancelled) presenceLoading = false
      })
    return () => {
      cancelled = true
    }
  })

  const staleBehind = $derived(
    graph ? graph.staleness.added_labels.length + graph.staleness.removed_ids.length : 0,
  )
  // ADR-054: the same concepts, counted through different forms since the build. Named, because
  // "the graph is behind" with no concept attached gives nothing to check after the rebuild.
  const formsChanged = $derived.by(() => {
    if (!graph) return ''
    const ids = new Set(graph.staleness.forms_changed_ids ?? [])
    return formsChangedNotice(
      graph.nodes.filter((n) => ids.has(n.id)).map((n) => shownLabel(n)),
      graph.staleness.forms_recorded !== false,
    )
  })
  // Coverage, stated with the rule that produces it. Not "68 documents are missing from the
  // graph": they are not waiting for a rebuild, they mention none of the concepts on the graph,
  // and a number that sent the user to a button changing nothing would be worse than silence.
  // The sentence names the lever that does move it — the graph vocabulary (ADR-018, ROADMAP 23).
  const coverage = $derived(graph ? graphCoverage(graph.staleness) : '')
  const rebuilding = $derived(rebuildState === 'running')
</script>

<div class="graphview">
  {#if loading}
    <div class="state">
      <span class="spinner" aria-hidden="true"></span>
      <p>Loading the concept graph…</p>
    </div>
  {:else if error}
    <div class="state">
      <span class="state-mark err"><Icon name="triangle-alert" size={26} /></span>
      <strong>Couldn’t load the concept graph</strong>
      <p class="muted">{error}</p>
      <button class="primary" onclick={onRebuild} disabled={rebuilding} type="button">
        {rebuilding ? 'Rebuilding…' : 'Rebuild'}
      </button>
    </div>
  {:else if !graph}
    <!-- 404 / never built: the NORMAL first run (skeleton.json is gitignored). Inform, offer a build. -->
    <div class="state">
      <span class="state-mark"><Icon name="waypoints" size={28} /></span>
      <strong>No concept graph yet</strong>
      <p class="muted">
        The graph maps how the concepts in your library connect, and surfaces which are backed by
        only one source. Building it is a local, offline pass (~7&nbsp;seconds).
      </p>
      <button class="primary" onclick={onRebuild} disabled={rebuilding} type="button">
        {rebuilding ? 'Building…' : 'Build the graph'}
      </button>
    </div>
  {:else if graph.nodes.length === 0}
    <!-- Built, and empty. Distinct from the branch above, and the distinction is the whole point:
         offering "Build the graph" here would promise that a rebuild fills the page, and it will
         not. Graph vocabulary is curated (ADR-018) — a concept enters at `graph_include=true`, and
         nothing a rebuild does sets it. So the primary action is the curation view, and the
         rebuild is demoted to what it actually is: the step *after* choosing (REVIEW 2026-08-12
         §2b R4, which asked for a real empty state rather than a page that reads as broken). -->
    <div class="state">
      <span class="state-mark"><Icon name="waypoints" size={28} /></span>
      <strong>No concepts in the graph yet</strong>
      <p class="muted">
        The graph maps a <em>curated</em> vocabulary — concepts you have chosen — rather than every
        keyword in your library, which is why it starts empty instead of full of noise.
      </p>
      <p class="muted">
        Choose which concepts belong on it in <strong>Manage keywords</strong>, then rebuild.
      </p>
      <!-- The view has one name on every door that opens it (GLOSSARY U-001). -->
      <button class="primary" onclick={onCurateVocabulary} type="button">Manage keywords</button>
      <button class="ghost" onclick={onRebuild} disabled={rebuilding} type="button">
        {rebuilding ? 'Rebuilding…' : 'Rebuild'}
      </button>
    </div>
  {:else}
    {#if staleBehind > 0 || formsChanged}
      <div class="stale" role="status">
        <Icon name="triangle-alert" size={14} />
        <span>
          {#if staleBehind > 0}Graph is {staleBehind} concept{staleBehind === 1 ? '' : 's'} behind
            your vocabulary.{/if}
          {formsChanged}
        </span>
        <button class="linkish" onclick={onRebuild} disabled={rebuilding} type="button">
          {rebuilding ? 'Rebuilding…' : 'Rebuild'}
        </button>
      </div>
    {/if}
    <div class="graphbar">
      {#if coverage}
        <!-- Plain text, not a warning: partial coverage is how this feature works, not a fault. -->
        <p class="coverage muted">{coverage}</p>
      {/if}
      <div class="graphactions">
        <!-- A rebuild that does not wait to be asked for. The notice above offers one when the
             vocabulary moved; nothing told the graph when the *reading* of the library changed
             (a fix to what counts as a document's text), so there was a graph to rebuild and no
             control to do it with. Left out while the notice is up: that one says why. -->
        {#if !(staleBehind > 0 || formsChanged)}
          <button
            class="ghost sm"
            onclick={onRebuild}
            disabled={rebuilding}
            type="button"
            use:tooltip={'Build the graph again from your library and your keywords as they are now. A few seconds, no model.'}
          >
            <Icon name="rotate-ccw" size={13} />
            {rebuilding ? 'Rebuilding…' : 'Rebuild'}
          </button>
        {/if}
        <!-- The standing door to the vocabulary, under the view's own name (GLOSSARY U-001). Until
             now the view was reached from the Library's keyword filter, or from here only once a
             concept was selected or the graph was empty — while this is the screen where a
             concept's forms are seen to matter. A door, not an editor: the graph never writes the
             vocabulary (ADR-017 A1). -->
        <button
          class="ghost sm"
          onclick={onCurateVocabulary}
          type="button"
          use:tooltip={'Mark forms, rename or add concepts'}
        >
          <Icon name="pencil" size={13} /> Manage keywords
        </button>
      </div>
    </div>

    <!-- The concept panel: the ego graph + details for the concept selected in the sidebar's index. -->
    <ConceptPanel
      {graph}
      {documents}
      {selectedId}
      {showUnderConnected}
      {offGraphConcept}
      {presence}
      {presenceLoading}
      bind:expandedDocId
      bind:panX
      bind:panY
      {onOpenDocument}
      {onManageConcept}
      {onPlaceConcept}
      {onSelectConcept}
      {onOpenPassage}
    />
  {/if}
</div>

<style>
  .graphview {
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-4);
  }

  /* Empty / loading / error states — centred card, mirrors the chat empty state. */
  .state {
    margin: auto;
    max-width: 40ch;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-6) var(--space-4);
  }
  .state-mark {
    color: var(--accent);
    display: inline-flex;
  }
  .state-mark.err {
    color: var(--danger);
  }
  .muted {
    color: var(--fg-2);
  }
  .primary {
    background: var(--accent);
    color: var(--accent-fg);
    border: none;
    border-radius: 8px;
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-weight: 600;
    cursor: pointer;
  }
  .primary:disabled {
    opacity: 0.6;
    cursor: default;
  }

  .graphbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-3);
    margin: 0 0 0.5rem;
  }
  /* On a narrow pane the sentence takes its own line instead of a column one word wide. */
  .coverage {
    margin: 0;
    font-size: 0.74rem;
    flex: 1 1 16rem;
  }
  /* Pushed to the far side whether or not the coverage sentence is there. */
  .graphactions {
    margin-left: auto;
    flex: none;
    display: flex;
    gap: var(--space-2);
  }
  .stale {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-sm);
    color: var(--warn-fg);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 8px;
    padding: var(--space-2) var(--space-3);
  }
  .stale span {
    flex: 1;
  }
  /* `.muted`, `.ghost` and `.linkish` are also in ConceptPanel.svelte, which styles its own
     copies: scoped CSS stops at the component, and each is a few lines. */
  .ghost {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    border-radius: 7px;
    padding: var(--space-1) var(--space-3);
    font: inherit;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    white-space: nowrap;
  }
  .ghost.sm {
    font-size: var(--text-sm);
  }
  .linkish {
    border: none;
    background: none;
    color: var(--accent);
    cursor: pointer;
    font: inherit;
    padding: 0;
  }
  .linkish:disabled {
    opacity: 0.6;
    cursor: default;
  }

  .spinner {
    width: 18px;
    height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
