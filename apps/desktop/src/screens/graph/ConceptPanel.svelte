<script lang="ts">
  // The concept panel (ADR-053, GLOSSARY U-004): the Graph tab's main pane, under the graph bar.
  // With a concept selected in the sidebar's concept index it shows the concept's name and counts,
  // *Reset view* / *Edit* / *Place*, its gap notice, the depth-1 ego graph (hand-rolled SVG + the
  // seeded force layout: wheel-zoom, drag-pan, click a neighbour to recentre), its Definition card
  // and the documents it appears in (concept → document → the chunks where it appears). A term the
  // vocabulary search reached shows its name and definition without a neighbourhood; with nothing
  // selected, a hint.
  //
  // Owns what is drawn from the graph (the ego subgraph, its layout, the counts, the gap lens) and
  // the zoom preference. What a new selection resets (the presence fetch, the open document row,
  // the pan) is Graph.svelte's selection wiring; the panel writes the row and the pan back through
  // `$bindable`.
  //
  // The gaps are the payload, not decoration (ADR-004). Per RG-014 the strong signal is `single_source`
  // (the corroboration thesis) — it leads; `under_connected` measures graph degree, is dominated by
  // vocabulary sparsity at this size, and is OFF by default behind a toggle.
  import type {
    ConceptGraph,
    ConceptGraphNode,
    ConceptPresence,
    Gap,
    LibraryDocument,
  } from '../../lib/core/types'
  import { authorLabel } from '../../lib/library/library'
  import { GAP_META, visibleConceptGaps } from '../../lib/graph/gaps'
  import { forceLayout, type Point } from '../../lib/graph/forceLayout'
  import { broadSummary, shownLabel } from '../../lib/graph/labels'
  import Icon from '../../shell/Icon.svelte'
  import Definition from './Definition.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    graph,
    documents,
    selectedId,
    showUnderConnected,
    offGraphConcept,
    presence,
    presenceLoading,
    expandedDocId = $bindable(),
    panX = $bindable(),
    panY = $bindable(),
    onOpenDocument,
    onManageConcept,
    onPlaceConcept,
    onSelectConcept,
    onOpenPassage,
  }: {
    graph: ConceptGraph
    documents: LibraryDocument[]
    selectedId: string | null
    showUnderConnected: boolean
    /** A concept reached by the vocabulary search that is not a graph node (ADR-053): the
     *  panel shows its definition without a neighbourhood. */
    offGraphConcept: { id: string; label: string } | null
    /** Where the selected concept appears, chunk by chunk; fetched by Graph.svelte on each
     *  selection, empty until it lands or when it fails. */
    presence: ConceptPresence[]
    presenceLoading: boolean
    /** The document row opened in the list; a new selection closes it. */
    expandedDocId: string | null
    /** The canvas pan; a new selection puts it back to 0. */
    panX: number
    panY: number
    onOpenDocument: (docId: string) => void
    onManageConcept: (conceptId: string, label: string) => void
    onPlaceConcept: (conceptId: string, label: string) => void
    onSelectConcept: (id: string) => void
    /** Open a passage where it was written — the path a chat citation takes. */
    onOpenPassage: (chunkKey: string) => void
  } = $props()

  const VIEW_W = 760
  const VIEW_H = 520
  // A margin around the layout box, in the same units. The layout places node centres up to its
  // edge and a label hangs below its node, so a node at the bottom lost its name to the canvas
  // edge (seen on *hard negatives* at y=496 of 520, 2026-11-05; ROADMAP 111).
  const VIEW_PAD = 30
  const ZOOM_MIN = 0.4
  const ZOOM_MAX = 3

  // Gap taxonomy (ranks, tones, blurbs, the under-connected opt-in) lives in the pure, shared,
  // node-tested `./gaps` module (RG-014) — imported above so this lens and the rail's ConceptIndex
  // agree (the index itself, with its rail mode/filter/lenses, lives in the sidebar now).

  // Only zoom — a real preference — is persisted, following libraryView/librarySort, and read again
  // on each mount. Pan is position-specific: Graph.svelte puts it back to 0 on each new selection.
  function loadZoom(): number {
    try {
      const v = Number(localStorage.getItem('graphZoom'))
      return v >= ZOOM_MIN && v <= ZOOM_MAX ? v : 1
    } catch {
      return 1
    }
  }
  let zoom = $state(loadZoom())

  const nodeById = $derived.by(() => {
    const m = new Map<string, ConceptGraphNode>()
    for (const n of graph.nodes) m.set(n.id, n)
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
    for (const g of graph.gaps ?? []) {
      const list = m.get(g.concept_id) ?? []
      list.push(g)
      m.set(g.concept_id, list)
    }
    return m
  })

  // The effective gap lens (dismissed dropped, under-connected opt-in) — shared with ConceptIndex via
  // the pure `./gaps` module so badges there and notes/dots here always agree.
  function visibleGaps(conceptId: string): Gap[] {
    return visibleConceptGaps(gapsByConcept.get(conceptId) ?? [], showUnderConnected)
  }

  const selectedNode = $derived(selectedId ? (nodeById.get(selectedId) ?? null) : null)

  // Depth-1 ego subgraph: the selected concept, its neighbours, and every edge among that set (so
  // neighbour-neighbour triangles show as context). Small by construction — median 6 nodes.
  const ego = $derived.by(() => {
    if (!selectedId) return null
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
  // The graph is a build artifact over the corpus *as it was*, so a node can cite a document the
  // library no longer resolves — every pre-ADR-047 re-extraction minted a new id. Counting and
  // listing only what resolves is what stops a bare UUID reaching the title slot; the count that
  // did not resolve is reported beside it rather than quietly dropped.
  function resolvedDocIds(n: ConceptGraphNode): string[] {
    return n.doc_ids.filter((id) => docById.has(id))
  }
  function docCount(n: ConceptGraphNode): number {
    return resolvedDocIds(n).length
  }
  function unresolvedDocCount(n: ConceptGraphNode): number {
    return n.doc_ids.length - resolvedDocIds(n).length
  }
  // ADR-054: the documents only a *broad* form reaches. They sit beside the count above and are
  // never added to it — a broad form (`distillation`) also matches other things. Listed so the
  // difference is something to open, not a number to take on trust.
  function broadDocIds(n: ConceptGraphNode): string[] {
    return (n.broad_doc_ids ?? []).filter((id) => docById.has(id))
  }

  function persistZoom(): void {
    try {
      localStorage.setItem('graphZoom', String(zoom))
    } catch {
      /* ignore — zoom just won't persist */
    }
  }

  // Wheel-zoom, clamped. A DELIBERATE non-passive listener (unlike DocumentGrid's passive wheel): we
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

  function presenceFor(docId: string): ConceptPresence | undefined {
    return presence.find((p) => p.document_id === docId)
  }
  // Title ONLY — never `docLabel`, which appends "· first author" for the breadcrumb/search case
  // where there is no separate byline. This row renders `docByline` right beside it, so docLabel
  // printed the authors twice ("A Primer… · Mathis et al." then "Mathis et al. · 2020").
  function docTitle(docId: string): string {
    const d = docById.get(docId)
    // An identifier is not a label. The list only renders resolvable ids, so this is the
    // belt-and-braces arm — and returning `docId` is precisely how UUIDs used to reach the user.
    if (!d) return 'No longer in your library'
    return d.title || d.filename
  }
  function docByline(docId: string): string {
    const d = docById.get(docId)
    if (!d) return ''
    const a = authorLabel(d)
    const y = d.year != null ? String(d.year) : ''
    return a && y ? `${a} · ${y}` : a || y
  }
</script>

<section class="ego" aria-label="Concept neighbourhood">
  {#if !selectedNode && offGraphConcept}
    <div class="ego-head">
      <div class="eh-title">
        <h2>{offGraphConcept.label}</h2>
        <span class="termtag">term</span>
      </div>
      <div class="eh-actions">
        <button
          class="btn"
          onclick={() => onManageConcept(offGraphConcept.id, offGraphConcept.label)}
          type="button"
          use:tooltip={'Take this term on as a concept in Manage keywords'}
        >
          <Icon name="pencil" size={13} /> Manage keywords
        </button>
      </div>
    </div>
    <!-- ADR-054: a term is a word the library uses that nobody has taken on. It can be read
         here; a definition is chosen for a concept, so the write starts in Manage keywords. -->
    <p class="muted offgraph">
      A term: your library uses this word, and you have not taken it on as a concept. Below is
      what the library says about it. To choose or write its definition, put it on the concept
      graph in Manage keywords.
    </p>
    <Definition conceptId={offGraphConcept.id} {onOpenPassage} />
  {:else if !selectedNode}
    <div class="ego-hint muted">
      <Icon name="waypoints" size={24} />
      <p>Select a concept to explore its neighbourhood and sources.</p>
    </div>
  {:else}
    <div class="ego-head">
      <div class="eh-title">
        <span class="dot lg" style="background:{commColor(selectedNode)}" aria-hidden="true"></span>
        <h2>{shownLabel(selectedNode)}</h2>
      </div>
      <div class="eh-actions">
        <span class="eh-meta">{selectedNode.degree} link{selectedNode.degree === 1 ? '' : 's'} · {docCount(selectedNode)} doc{docCount(selectedNode) === 1 ? '' : 's'}</span>
        <button class="btn btn-quiet" onclick={resetView} type="button" use:tooltip={'Put the graph back to its first zoom and position'}>Reset view</button>
        <button
          class="btn"
          onclick={() => onManageConcept(selectedNode.id, selectedNode.label)}
          type="button"
          use:tooltip={'Edit this concept in Manage keywords'}
        >
          <Icon name="pencil" size={13} /> Edit
        </button>
        <button
          class="btn"
          onclick={() => onPlaceConcept(selectedNode.id, selectedNode.label)}
          type="button"
          use:tooltip={'Place this concept in the field taxonomy'}
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

    <!-- The graph first, where it is seen on opening a concept: below a long definition it was
         off-screen, and before it kept its height it was squashed to 2 px (ROADMAP 111). -->
    <div class="canvas">
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <svg
        class="graph"
        viewBox="{-VIEW_PAD} {-VIEW_PAD} {VIEW_W + 2 * VIEW_PAD} {VIEW_H + 2 * VIEW_PAD}"
        use:wheelZoom
        onpointerdown={startPan}
        role="img"
        aria-label="Depth-1 neighbourhood of {shownLabel(selectedNode)}"
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
                aria-label={shownLabel(n)}
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
                <text class="nlabel" y={nodeRadius(n) + 13} text-anchor="middle">{shownLabel(n)}</text>
              </g>
            {/if}
          {/each}
        </g>
      </svg>
      <p class="canvas-hint muted">Scroll to zoom · drag to pan · click a neighbour to recentre</p>
    </div>

    <Definition conceptId={selectedNode.id} {onOpenPassage} />

    <!-- concept → document → chunks -->
    <div class="sources">
      <h3>Appears in {docCount(selectedNode)} document{docCount(selectedNode) === 1 ? '' : 's'}</h3>
      {#if unresolvedDocCount(selectedNode) > 0}
        <!-- Inform, don't block: the resolvable half is still listed below. Said here rather
             than only in the staleness banner because this is where the missing rows would
             otherwise be, and a count that silently shrank would read as a bug. -->
        <p class="muted">
          {unresolvedDocCount(selectedNode)} more
          {unresolvedDocCount(selectedNode) === 1 ? 'document is' : 'documents are'} cited by the
          graph but no longer in your library — rebuild the graph to bring it up to date.
        </p>
      {/if}
      {#if presenceLoading}
        <p class="muted">Loading sources…</p>
      {/if}
      <ul class="doclist">
        {#each resolvedDocIds(selectedNode) as docId (docId)}
          {@const pr = presenceFor(docId)}
          <li class="docitem">
            <button
              class="docrow"
              onclick={() => (expandedDocId = expandedDocId === docId ? null : docId)}
              aria-expanded={expandedDocId === docId}
              type="button"
              use:tooltip={expandedDocId === docId ? 'Hide where this document mentions the concept' : 'Show where this document mentions the concept'}
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
                <button class="linkish" onclick={() => onOpenDocument(docId)} type="button" use:tooltip={'Open this document in the Library'}>
                  Open in Library →
                </button>
              </div>
            {/if}
          </li>
        {/each}
      </ul>
      {#if broadDocIds(selectedNode).length > 0}
        <!-- Beside the count, never in it (ADR-054). -->
        <div class="beside">
          <p class="muted">
            {broadSummary(broadDocIds(selectedNode).length, selectedNode.broad_forms ?? [])}.
            A broad form also matches other things, so these are not counted above. Change it
            in Manage keywords.
          </p>
          <ul class="doclist">
            {#each broadDocIds(selectedNode) as docId (docId)}
              <li class="docitem">
                <button class="docrow" onclick={() => onOpenDocument(docId)} type="button" use:tooltip={'Open this document in the Library'}>
                  <Icon name="file-text" size={14} />
                  <span class="dtitle">{docTitle(docId)}</span>
                  {#if docByline(docId)}<span class="dby muted">{docByline(docId)}</span>{/if}
                  <Icon name="chevron-right" size={13} />
                </button>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    </div>
  {/if}
</section>

<style>
  /* Also in Graph.svelte, which styles its own copies of these three: scoped CSS stops at the
     component, and each is a few lines. */
  .muted {
    color: var(--fg-2);
  }
  .linkish {
    border: none;
    background: none;
    color: var(--accent);
    cursor: pointer;
    font: inherit;
    padding: 0;
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
  /* The column scrolls; nothing in it shrinks. A flex child with `overflow: hidden` has no
     minimum height, so the canvas took every pixel the definition and the document list needed
     and was drawn 2 px tall (the user, 2026-11-05: "the text squashes the graph"; ROADMAP 111).
     `:global` because the definition card is a child component's root. */
  .ego > :global(*) {
    flex-shrink: 0;
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
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .eh-meta {
    font-size: var(--text-sm);
    color: var(--fg-2);
    white-space: nowrap;
  }
  .offgraph {
    margin: 0;
    font-size: var(--text-sm);
  }
  /* ADR-054 — a term is not a concept: say so beside its name, quietly. */
  .termtag {
    flex: none;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--fg-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0 0.45rem;
  }
  /* The documents a broad form reaches: beside the concept's own, set apart by a rule. */
  .beside {
    margin-top: var(--space-3);
    padding-top: var(--space-2);
    border-top: 1px dashed var(--border);
  }
  .beside p {
    margin: 0 0 var(--space-2);
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
  /* Allow-listed (ui-rules.allow.ts), with `.gapdot.danger` below: a gap's severity on a note and
     on a node's dot, named by GAP_META's `tone`, not a button's role. */
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
  /* The layout's proportions (the padded viewBox, 820 × 580), capped so a wide pane does not
     push everything else a screen down; when capped, the viewBox centres the layout in the box. */
  .graph {
    display: block;
    width: 100%;
    height: auto;
    aspect-ratio: 820 / 580;
    max-height: min(55vh, 520px);
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
</style>
