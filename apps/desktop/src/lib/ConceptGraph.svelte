<script lang="ts">
  // Concept-graph view (docs/specs/feature-concept-graph.md PR-G2a, ADR-017). A DESTINATION, not a
  // modal: selecting a concept (in the sidebar's GraphIndex rail, which App composes) opens a
  // depth-1 ego graph (hand-rolled SVG + the seeded force layout) and a details panel that navigates
  // concept → document → the chunks where it appears. Selection + the under-connected lens are
  // App-owned props — the rail and this panel must agree. Read-only for the vocabulary (ADR-017 A1):
  // the only write is a deep-link to the Manage-keywords view. The graph observes; it never edits.
  //
  // The gaps are the payload, not decoration (ADR-004). Per RG-014 the strong signal is `single_source`
  // (the corroboration thesis) — it leads; `under_connected` measures graph degree, is dominated by
  // vocabulary sparsity at this size, and is OFF by default behind a toggle.
  import type {
    ConceptGraph,
    ConceptGraphNode,
    ConceptPresence,
    Gap,
    GraphRebuildStatus,
    LibraryDocument,
  } from './types'
  import { authorLabel, docLabel } from './library'
  import { GAP_META, visibleConceptGaps } from './gaps'
  import { forceLayout, type Point } from './forceLayout'
  import Icon from './Icon.svelte'

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
    onPlaceConcept,
    onSelectConcept,
    loadPresence,
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
    onPlaceConcept: (conceptId: string, label: string) => void
    onSelectConcept: (id: string) => void
    loadPresence: (conceptId: string) => Promise<ConceptPresence[]>
  } = $props()

  const VIEW_W = 760
  const VIEW_H = 520
  const ZOOM_MIN = 0.4
  const ZOOM_MAX = 3

  // Gap taxonomy (ranks, tones, blurbs, the under-connected opt-in) lives in the pure, shared,
  // node-tested `./gaps` module (RG-014) — imported above so this lens and the rail's GraphIndex
  // agree (the index itself, with its rail mode/filter/lenses, lives in the sidebar now).
  let presence = $state<ConceptPresence[]>([])
  let presenceLoading = $state(false)
  let expandedDocId = $state<string | null>(null)

  // Pan is position-specific (it resets when the ego re-centres), so only zoom — a real preference —
  // is persisted, following libraryView/librarySort. Pan restarts at 0 on each new selection.
  function loadZoom(): number {
    try {
      const v = Number(localStorage.getItem('graphZoom'))
      return v >= ZOOM_MIN && v <= ZOOM_MAX ? v : 1
    } catch {
      return 1
    }
  }
  let zoom = $state(loadZoom())
  let panX = $state(0)
  let panY = $state(0)

  const nodeById = $derived.by(() => {
    const m = new Map<string, ConceptGraphNode>()
    for (const n of graph?.nodes ?? []) m.set(n.id, n)
    return m
  })
  const docById = $derived.by(() => {
    const m = new Map<string, LibraryDocument>()
    for (const d of documents) m.set(d.id, d)
    return m
  })
  // concept_id → its gaps (a concept can carry more than one kind).
  const gapsByConcept = $derived.by(() => {
    const m = new Map<string, Gap[]>()
    for (const g of graph?.gaps ?? []) {
      const list = m.get(g.concept_id) ?? []
      list.push(g)
      m.set(g.concept_id, list)
    }
    return m
  })

  // The effective gap lens (dismissed dropped, under-connected opt-in) — shared with GraphIndex via
  // the pure `./gaps` module so badges there and notes/dots here always agree.
  function visibleGaps(conceptId: string): Gap[] {
    return visibleConceptGaps(gapsByConcept.get(conceptId) ?? [], showUnderConnected)
  }

  const selectedNode = $derived(selectedId ? (nodeById.get(selectedId) ?? null) : null)

  // Depth-1 ego subgraph: the selected concept, its neighbours, and every edge among that set (so
  // neighbour-neighbour triangles show as context). Small by construction — median 6 nodes.
  const ego = $derived.by(() => {
    if (!graph || !selectedId) return null
    const neighbours = new Set<string>()
    for (const e of graph.edges) {
      if (e.source === selectedId) neighbours.add(e.target)
      else if (e.target === selectedId) neighbours.add(e.source)
    }
    const ids = [selectedId, ...neighbours]
    const idSet = new Set(ids)
    const edges = graph.edges.filter((e) => idSet.has(e.source) && idSet.has(e.target))
    return { ids, edges }
  })
  // Deterministic: depends only on the ego set (not zoom/pan), so it does not recompute on interaction.
  const layout = $derived.by<Map<string, Point>>(() => {
    if (!ego) return new Map()
    return forceLayout(ego.ids, ego.edges, { width: VIEW_W, height: VIEW_H, seed: 42 })
  })

  function nodeRadius(n: ConceptGraphNode): number {
    return Math.min(22, 7 + Math.sqrt(n.degree) * 3)
  }
  function commColor(n: ConceptGraphNode): string {
    return `var(--comm-${((n.community % 12) + 12) % 12})`
  }
  function docCount(n: ConceptGraphNode): number {
    return n.doc_ids.length
  }

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

  function persistZoom(): void {
    try {
      localStorage.setItem('graphZoom', String(zoom))
    } catch {
      /* ignore — zoom just won't persist */
    }
  }

  // Wheel-zoom, clamped. A DELIBERATE non-passive listener (unlike LibraryGrid's passive wheel): we
  // must preventDefault to stop the page scrolling under the cursor. Attached via an action so the
  // { passive: false } is explicit and cleaned up.
  function wheelZoom(node: SVGSVGElement) {
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1
      zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom * factor))
      persistZoom()
    }
    node.addEventListener('wheel', onWheel, { passive: false })
    return { destroy: () => node.removeEventListener('wheel', onWheel) }
  }

  // Pan by dragging the canvas. Same pointer-drag idiom as App.svelte's startResize: capture on
  // window, clamp nothing (pan is free), clean up on release.
  function startPan(e: PointerEvent): void {
    if (e.button !== 0) return
    const ox = e.clientX - panX
    const oy = e.clientY - panY
    const onMove = (ev: PointerEvent) => {
      panX = ev.clientX - ox
      panY = ev.clientY - oy
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
    }
    document.body.style.cursor = 'grabbing'
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }
  function resetView(): void {
    zoom = 1
    panX = 0
    panY = 0
    persistZoom()
  }

  const staleBehind = $derived(
    graph ? graph.staleness.added_labels.length + graph.staleness.removed_ids.length : 0,
  )
  const rebuilding = $derived(rebuildState === 'running')

  function presenceFor(docId: string): ConceptPresence | undefined {
    return presence.find((p) => p.document_id === docId)
  }
  function docTitle(docId: string): string {
    const d = docById.get(docId)
    return d ? docLabel(d) : docId
  }
  function docByline(docId: string): string {
    const d = docById.get(docId)
    if (!d) return ''
    const a = authorLabel(d)
    const y = d.year != null ? String(d.year) : ''
    return a && y ? `${a} · ${y}` : a || y
  }
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
  {:else}
    {#if staleBehind > 0}
      <div class="stale" role="status">
        <Icon name="triangle-alert" size={14} />
        <span>Graph is {staleBehind} concept{staleBehind === 1 ? '' : 's'} behind your vocabulary.</span>
        <button class="linkish" onclick={onRebuild} disabled={rebuilding} type="button">
          {rebuilding ? 'Rebuilding…' : 'Rebuild'}
        </button>
      </div>
    {/if}

    <!-- The ego graph + details for the concept selected in the sidebar's index. -->
    <section class="ego" aria-label="Concept neighbourhood">
        {#if !selectedNode}
          <div class="ego-hint muted">
            <Icon name="waypoints" size={24} />
            <p>Select a concept to explore its neighbourhood and sources.</p>
          </div>
        {:else}
          <div class="ego-head">
            <div class="eh-title">
              <span class="dot lg" style="background:{commColor(selectedNode)}" aria-hidden="true"></span>
              <h2>{selectedNode.label}</h2>
            </div>
            <div class="eh-actions">
              <span class="eh-meta">{selectedNode.degree} link{selectedNode.degree === 1 ? '' : 's'} · {docCount(selectedNode)} doc{docCount(selectedNode) === 1 ? '' : 's'}</span>
              <button class="ghost sm" onclick={resetView} type="button" title="Reset zoom & pan">Reset view</button>
              <button
                class="ghost sm"
                onclick={() => onManageConcept(selectedNode.id, selectedNode.label)}
                type="button"
                title="Edit this concept in Manage keywords"
              >
                <Icon name="pencil" size={13} /> Edit
              </button>
              <button
                class="ghost sm"
                onclick={() => onPlaceConcept(selectedNode.id, selectedNode.label)}
                type="button"
                title="Place this concept in the field taxonomy"
              >
                <Icon name="tag" size={13} /> Place
              </button>
            </div>
          </div>

          {#if visibleGaps(selectedNode.id).length > 0}
            <div class="gap-notes">
              {#each visibleGaps(selectedNode.id) as g (g.kind)}
                <div class="gap-note {GAP_META[g.kind].tone}">
                  <strong>{GAP_META[g.kind].label}.</strong>
                  <span>{GAP_META[g.kind].blurb}</span>
                </div>
              {/each}
            </div>
          {/if}

          <div class="canvas">
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <svg
              class="graph"
              viewBox="0 0 {VIEW_W} {VIEW_H}"
              use:wheelZoom
              onpointerdown={startPan}
              role="img"
              aria-label="Depth-1 neighbourhood of {selectedNode.label}"
            >
              <g transform="translate({panX} {panY}) scale({zoom})">
                {#each ego?.edges ?? [] as e (e.source + '::' + e.target)}
                  {@const a = layout.get(e.source)}
                  {@const b = layout.get(e.target)}
                  {#if a && b}
                    <line class="edge" x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
                  {/if}
                {/each}
                {#each ego?.ids ?? [] as id (id)}
                  {@const n = nodeById.get(id)}
                  {@const p = layout.get(id)}
                  {#if n && p}
                    {@const vg = visibleGaps(id)}
                    <!-- svelte-ignore a11y_click_events_have_key_events -->
                    <g
                      class="node"
                      class:center={id === selectedId}
                      transform="translate({p.x} {p.y})"
                      role="button"
                      tabindex="0"
                      aria-label={n.label}
                      onclick={() => onSelectConcept(id)}
                      onkeydown={(ev) => {
                        if (ev.key === 'Enter' || ev.key === ' ') {
                          ev.preventDefault()
                          onSelectConcept(id)
                        }
                      }}
                    >
                      <circle
                        class="ncircle"
                        r={nodeRadius(n)}
                        style="fill:{commColor(n)}"
                      />
                      {#if vg.length > 0}
                        <circle
                          class="gapdot {GAP_META[vg[0].kind].tone}"
                          r="4"
                          cx={nodeRadius(n) * 0.7}
                          cy={-nodeRadius(n) * 0.7}
                        />
                      {/if}
                      <text class="nlabel" y={nodeRadius(n) + 13} text-anchor="middle">{n.label}</text>
                    </g>
                  {/if}
                {/each}
              </g>
            </svg>
            <p class="canvas-hint muted">Scroll to zoom · drag to pan · click a neighbour to recentre</p>
          </div>

          <!-- concept → document → chunks -->
          <div class="sources">
            <h3>Appears in {docCount(selectedNode)} document{docCount(selectedNode) === 1 ? '' : 's'}</h3>
            {#if presenceLoading}
              <p class="muted">Loading sources…</p>
            {/if}
            <ul class="doclist">
              {#each selectedNode.doc_ids as docId (docId)}
                {@const pr = presenceFor(docId)}
                <li class="docitem">
                  <button
                    class="docrow"
                    onclick={() => (expandedDocId = expandedDocId === docId ? null : docId)}
                    aria-expanded={expandedDocId === docId}
                    type="button"
                  >
                    <Icon name="file-text" size={14} />
                    <span class="dtitle">{docTitle(docId)}</span>
                    {#if docByline(docId)}<span class="dby muted">{docByline(docId)}</span>{/if}
                    {#if pr}<span class="dmentions" title="mentions in this document">{pr.n_mentions}×</span>{/if}
                    <Icon name="chevron-right" size={13} />
                  </button>
                  {#if expandedDocId === docId}
                    <div class="chunks">
                      {#if pr && pr.chunk_keys.length > 0}
                        <p class="muted">Mentioned in {pr.chunk_keys.length} chunk{pr.chunk_keys.length === 1 ? '' : 's'}.</p>
                      {:else if !presenceLoading}
                        <p class="muted">No chunk-level record for this concept in this document.</p>
                      {/if}
                      <button class="linkish" onclick={() => onOpenDocument(docId)} type="button">
                        Open in Library →
                      </button>
                    </div>
                  {/if}
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      </section>
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
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex: none;
  }
  .dot.lg {
    width: 14px;
    height: 14px;
  }

  /* ---- ego (fills the main pane — the index rail lives in the sidebar) ---- */
  .ego {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .ego-hint {
    margin: auto;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-2);
  }
  .ego-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    flex-wrap: wrap;
  }
  .eh-title {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .eh-title h2 {
    margin: 0;
    font-size: var(--text-title);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .eh-actions {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .eh-meta {
    font-size: var(--text-sm);
    color: var(--fg-2);
  }
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
  }
  .ghost.sm {
    font-size: var(--text-sm);
  }
  .gap-notes {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .gap-note {
    font-size: var(--text-sm);
    border-radius: 8px;
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--border);
  }
  .gap-note.danger {
    color: var(--danger);
    border-color: var(--danger);
  }
  .gap-note.warn {
    color: var(--warn-fg);
    background: var(--warn-bg);
    border-color: var(--warn-border);
  }

  .canvas {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    overflow: hidden;
  }
  .graph {
    display: block;
    width: 100%;
    height: auto;
    aspect-ratio: 760 / 520;
    touch-action: none;
    cursor: grab;
  }
  .graph:active {
    cursor: grabbing;
  }
  .edge {
    stroke: var(--graph-edge);
    stroke-width: 1.5;
  }
  .node {
    cursor: pointer;
  }
  .ncircle {
    stroke: var(--graph-node-stroke);
    stroke-width: 1.5;
  }
  .node.center .ncircle {
    stroke: var(--accent);
    stroke-width: 3;
  }
  .node:focus-visible .ncircle {
    stroke: var(--accent);
    stroke-width: 3;
  }
  .gapdot.danger {
    fill: var(--danger);
  }
  .gapdot.warn {
    fill: var(--warn-fg);
  }
  .nlabel {
    fill: var(--fg);
    font-size: 11px;
    font-family: var(--font-sans);
    paint-order: stroke;
    stroke: var(--surface);
    stroke-width: 3px;
  }
  .canvas-hint {
    font-size: var(--text-meta);
    text-align: center;
    padding: var(--space-1) 0 var(--space-2);
    margin: 0;
  }

  /* ---- sources ---- */
  .sources h3 {
    margin: 0 0 var(--space-2);
    font-size: var(--text-sm);
    color: var(--fg-2);
    font-weight: 600;
  }
  .doclist {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .docrow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    text-align: left;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: var(--space-2);
    background: none;
    color: var(--fg);
    font: inherit;
    cursor: pointer;
  }
  .docrow:hover {
    background: var(--surface);
  }
  .dtitle {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dby {
    font-size: var(--text-meta);
    white-space: nowrap;
  }
  .dmentions {
    font-size: var(--text-meta);
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
  }
  .chunks {
    padding: var(--space-1) var(--space-2) var(--space-3) calc(var(--space-4) + var(--space-2));
    font-size: var(--text-sm);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    align-items: flex-start;
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
