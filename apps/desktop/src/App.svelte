<script lang="ts">
  import type {
    CompareResult,
    ConceptGraph as ConceptGraphData,
    ConversationDetail,
    ConversationSummary,
    GraphRebuildStatus,
    Health,
    KeywordFamily,
    KeywordFamilyProposal,
    LibraryDocument,
    LibraryFolder,
    RagOverrides,
    TurnResult,
  } from './lib/types'
  import {
    addDocumentsToFolder,
    addFamilyMember,
    compareRetrieval,
    createFolder,
    createKeywordFamily,
    deleteFolder,
    deleteKeywordFamily,
    detectKeywordFamilies,
    getConceptGraph,
    getConceptPresence,
    getConversation,
    getGraphRebuildStatus,
    getHealth,
    exportConversation,
    listConversations,
    listFolders,
    listKeywordFamilies,
    deleteDocument,
    listLibraryDocuments,
    rebuildConceptGraph,
    removeDocumentFromFolder,
    removeFamilyMember,
    renameFolder,
    renameKeywordFamily,
    resetDocumentMeta,
    revealDocument,
    streamChat,
    updateConversationMeta,
    updateDocumentMeta,
  } from './lib/api'
  import Turn from './lib/Turn.svelte'
  import ReadonlyTurn from './lib/ReadonlyTurn.svelte'
  import Settings from './lib/Settings.svelte'
  import SourcePanel from './lib/SourcePanel.svelte'
  import Sidebar from './lib/Sidebar.svelte'
  import LibraryBrowser from './lib/LibraryBrowser.svelte'
  import LibraryGrid from './lib/LibraryGrid.svelte'
  import LibraryFilterStrip from './lib/LibraryFilterStrip.svelte'
  import LibraryKeywordFilter from './lib/LibraryKeywordFilter.svelte'
  import LibraryManageKeywords from './lib/LibraryManageKeywords.svelte'
  import LibraryManageFolders from './lib/LibraryManageFolders.svelte'
  import LibraryMetaEditor from './lib/LibraryMetaEditor.svelte'
  import LibraryDeleteConfirm from './lib/LibraryDeleteConfirm.svelte'
  import CompareCard from './lib/CompareCard.svelte'
  import ConceptGraph from './lib/ConceptGraph.svelte'
  import GlobalSearch from './lib/GlobalSearch.svelte'
  import Icon from './lib/Icon.svelte'
  import {
    type LibraryCollection,
    type LibrarySort,
    collectionLabel,
    docLabel,
    folderNameMap,
    docsFor,
    facetFilter,
    familyCanonicalMap,
    familyUnitsOf,
    remapSelection,
    unitDocCounts,
    filterDocs,
    keywordFacets,
    sortDocs,
  } from './lib/library'
  import { searchEverything } from './lib/search'
  import appMark from './assets/brand/app-mark.png'

  interface TurnState {
    id: number
    question: string
    answer: string
    result: TurnResult | null
    streaming: boolean
    error: string | null
  }

  function freshSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  }
  // $state: the sidebar's "current" marker + the citation-source derivation read this, so a fresh
  // id from ↻ New must trigger updates.
  let sessionId = $state(freshSessionId())

  let health = $state<Health | null>(null)
  let status = $state<'connecting' | 'ready' | 'down'>('connecting')
  let turns = $state<TurnState[]>([])
  let input = $state('')
  let sending = $state(false)
  let showSettings = $state(false)
  // ADR-010: the RAG-sandbox overrides for this app session. In-memory only — a fresh
  // launch always starts from {} (locked defaults), never persisted to disk.
  let overrides = $state<RagOverrides>({})
  let nextId = 0

  // A/B-compare (U6): a per-turn retrieval diff (locked defaults vs the session override). $0 —
  // retrieval only, no answer generation. The result is an ephemeral card, not a chat turn.
  let compareResult = $state<CompareResult | null>(null)
  let comparing = $state(false)
  // The Test-override button only exists while a retrieval-affecting override is set — with none,
  // both sides retrieve identically and the button is dead weight (2026-07-13 UX review). Settings
  // writes these fields only when touched; Reset returns overrides to {}.
  const hasRetrievalOverride = $derived(
    overrides.top_k != null || overrides.use_multi_query != null,
  )

  // Conversation history (feature-conversation-history.md). `viewing` is the session_id shown as a
  // read-only transcript; `null` means the live chat (composer + claims bound to `sessionId`).
  let conversations = $state<ConversationSummary[]>([])
  let viewing = $state<string | null>(null)
  let viewedConvo = $state<ConversationDetail | null>(null)
  // Resume (fresh-context): a reopened past chat the user chose to *continue*. Its turns render
  // read-only above the composer for reference; `sessionId` is switched to it so new turns thread
  // to the same conversation and persist. The in-memory backend session starts fresh (empty
  // history), so new questions are standalone corpus queries — no replay of the old turns.
  let resumedHistory = $state<ConversationDetail | null>(null)
  let sidebarOpen = $state(false) // mobile drawer

  // Global-search overlay (docs/specs/feature-app-shell-search-collapse.md, sub-item a). A
  // navigation search over chats + documents, opened from the header or Cmd/Ctrl-K. App owns the
  // query + derives the results (searchEverything is pure/tested); GlobalSearch just renders.
  let searchOpen = $state(false)
  let searchQuery = $state('')

  // Library space (feature-library-browser.md L1; nav redesign feature-library-redesign.md L4
  // Phase A). `mode` swaps the sidebar + main pane between Chat and Library; the chat state
  // (turns/viewing/sessionId) is untouched by the switch. Navigation model: the rail picks the
  // active *collection*, the main pane shows it as an inventory grid, and opening a document
  // drills down in place to the chunk view (breadcrumb + Back walk back up).
  let mode = $state<'chat' | 'library' | 'graph'>('chat')
  let documents = $state<LibraryDocument[]>([])
  let libraryCollection = $state<LibraryCollection>({ kind: 'all' })
  let libraryDocId = $state<string | null>(null)
  let libraryQuery = $state('')
  // Selected keyword facets (AND). Orthogonal to the collection — session-scoped, non-persistent
  // (a filter, not a preference), resets on reload like the search query. `keywordFilterOpen` toggles
  // the two-pane picker overlay; the inline strip shows only the selected keywords + the trigger.
  let libraryKeywords = $state<string[]>([])
  let keywordFilterOpen = $state(false)
  let documentsLoaded = false
  // The overlay's results, derived from the live chat + document lists (both already client-side).
  const searchResults = $derived(searchEverything(searchQuery, conversations, documents))

  // Folders (ADR-025 F1, docs/specs/feature-corpus-folders.md). Manual Library organisation.
  // The rail renders this list rather than deriving groups from `documents`, so a folder with
  // zero members is still visible and therefore fillable (spec D3). `folderError` surfaces a
  // 400 (blank/collision) in the Manage view without blocking anything else.
  let folders = $state<LibraryFolder[]>([])
  // ADR-025 F2 — the chat retrieval scope. Sticky across turns, in memory ONLY: a reload
  // returns to the whole library. Persisting it is the rejected option — a scope you forgot
  // you set silently narrows every future answer, which is the exact lie the integrity layer
  // exists to prevent. Deliberately separate from `libraryCollection`: filtering the Library
  // grid and scoping a conversation are two different intentions.
  let chatScopeFolderId = $state<string | null>(null)
  let manageFoldersOpen = $state(false)
  let manageFolderId = $state<string | null>(null)
  let manageFolderQuery = $state('')
  let folderError = $state<string | null>(null)

  // Tag families (feature-tag-families.md, PR-1). Loaded alongside the document list;
  // `manageKeywordsOpen` opens the curation view (from the keyword-filter overlay's link).
  let keywordFamilies = $state<KeywordFamily[]>([])
  let manageKeywordsOpen = $state(false)
  // Detection (PR-2) — proposals live only while the Manage view is open; nothing is written
  // until a proposal is accepted (routes into createFamily below), so staleness is harmless.
  let detectProposals = $state<KeywordFamilyProposal[]>([])
  let detecting = $state(false)
  let detectError = $state<string | null>(null)

  // Concept graph (feature-concept-graph.md PR-G2a, ADR-017). Lazy-loaded on first entry to the
  // Graph mode; `null` after a load means "never built" (a 404 — the normal first run), which the
  // view renders as a build affordance. Rebuild is a 202 + poll job (B1/ADR-017 B1); while it runs
  // `graphRebuildState` is 'running' and the graph is refetched once it settles.
  let conceptGraph = $state<ConceptGraphData | null>(null)
  let graphLoading = $state(false)
  let graphError = $state<string | null>(null)
  let graphLoaded = false
  let graphRebuildState = $state<GraphRebuildStatus['state']>('idle')

  async function loadConceptGraph(): Promise<void> {
    graphLoading = true
    graphError = null
    try {
      conceptGraph = await getConceptGraph()
    } catch (e) {
      graphError = e instanceof Error ? e.message : String(e)
    } finally {
      graphLoading = false
      graphLoaded = true
    }
  }

  // Kick a rebuild and poll the status route until it settles, then refetch the graph. Deterministic
  // and ~7s; the view stays usable throughout (inform, don't block).
  async function rebuildGraph(): Promise<void> {
    if (graphRebuildState === 'running') return
    graphRebuildState = 'running'
    try {
      await rebuildConceptGraph()
    } catch (e) {
      graphRebuildState = 'error'
      graphError = e instanceof Error ? e.message : String(e)
      return
    }
    const poll = async (): Promise<void> => {
      try {
        const st = await getGraphRebuildStatus()
        graphRebuildState = st.state
        if (st.state === 'running') {
          setTimeout(() => void poll(), 700)
          return
        }
        if (st.state === 'error') {
          graphError = st.message ?? 'rebuild failed'
          return
        }
        await loadConceptGraph() // 'done' → pull the fresh graph
      } catch (e) {
        graphRebuildState = 'error'
        graphError = e instanceof Error ? e.message : String(e)
      }
    }
    void poll()
  }

  // Deep-link from a graph node to curate its concept (ADR-017 A1 — the graph never writes the
  // vocabulary; the Manage-keywords view owns every edit). Switches to Library and opens the view.
  function manageConcept(_conceptId: string, _label: string): void {
    selectMode('library')
    manageKeywordsOpen = true
  }

  // Grid ⇄ list toggle — a client-only view preference, persisted like theme/panel widths.
  function loadLibraryView(): 'grid' | 'list' {
    try {
      const v = localStorage.getItem('libraryView')
      if (v === 'grid' || v === 'list') return v
    } catch {
      /* ignore — fall back to default */
    }
    return 'grid'
  }
  let libraryView = $state<'grid' | 'list'>(loadLibraryView())
  function setLibraryView(v: 'grid' | 'list'): void {
    libraryView = v
    try {
      localStorage.setItem('libraryView', v)
    } catch {
      /* ignore — view just won't persist */
    }
  }

  // Library sort — a client-only preference, persisted like the view toggle.
  const LIB_SORTS: { key: LibrarySort; label: string }[] = [
    { key: 'title-az', label: 'Title (A–Z)' },
    { key: 'author-az', label: 'Author (A–Z)' },
    { key: 'pub-desc', label: 'Publication date (newest)' },
    { key: 'added-desc', label: 'Added date (newest)' },
  ]
  function loadLibrarySort(): LibrarySort {
    try {
      const v = localStorage.getItem('librarySort')
      if (LIB_SORTS.some((s) => s.key === v)) return v as LibrarySort
    } catch {
      /* ignore — fall back to default */
    }
    return 'title-az'
  }
  let librarySort = $state<LibrarySort>(loadLibrarySort())
  let libSortOpen = $state(false)
  function setLibrarySort(v: LibrarySort): void {
    librarySort = v
    libSortOpen = false
    try {
      localStorage.setItem('librarySort', v)
    } catch {
      /* ignore — just won't persist */
    }
  }

  // Pipeline: active collection → search filter (Decision 5a) → keyword facets (AND) → sort.
  // Facets are orthogonal to the collection: switching collection keeps them, and the facet chips
  // grey out relative to the current searched pool. `facetList` drives the facet bar; the selected
  // keywords also drive the tile highlight + first-position ordering.
  // Family collapse (PR-1) sits ahead of the facet math: `keywordsOf` maps each doc's raw
  // keywords through the family canonical map (identity when no families exist — byte-identical
  // to pre-PR-1 behavior), then `keywordFacets`/`facetFilter` operate on those collapsed units.
  const familyCanonicalOf = $derived(familyCanonicalMap(keywordFamilies))
  const keywordsOf = $derived(familyUnitsOf(familyCanonicalOf))
  const folderNames = $derived(folderNameMap(folders))
  // A folder deleted elsewhere must not stay silently selected as the chat scope.
  const chatScopeFolder = $derived(folders.find((f) => f.id === chatScopeFolderId) ?? null)
  const collectionDocs = $derived(docsFor(documents, libraryCollection, new Date()))
  const searchedDocs = $derived(filterDocs(collectionDocs, libraryQuery))
  const facetList = $derived(keywordFacets(searchedDocs, libraryKeywords, keywordsOf))
  // Documents per unit over the pre-facet pool — the rare-tail split (PR-2.7 F4) must not
  // shift as the user toggles keywords, and `KeywordFacet.count` is relative to the faceted
  // pool, so it cannot be reused for this.
  const facetDocCounts = $derived(unitDocCounts(searchedDocs, keywordsOf))
  // The Manage view's pool lists *raw* keyword names (a family's members), so its rare split
  // counts raw keywords over the whole library rather than family units over a collection.
  const rawKeywordDocCounts = $derived(unitDocCounts(documents))
  const visibleDocs = $derived(
    sortDocs(facetFilter(searchedDocs, libraryKeywords, keywordsOf), librarySort),
  )
  // The corpus's full raw-keyword universe (unfiltered by collection/search), for the Manage view.
  const allKeywords = $derived.by(() => {
    const s = new Set<string>()
    for (const d of documents) for (const k of d.keywords) s.add(k)
    return [...s]
  })
  // Breadcrumb label for the open document, from the cached list (the chunk view fetches its
  // own detail; a stale/missing entry just hides the crumb).
  const openDoc = $derived(
    libraryDocId ? (documents.find((d) => d.id === libraryDocId) ?? null) : null,
  )

  // Which citation panel is open — keyed by a turn *key* (a live turn's id as string, or a past
  // turn's record_id) so a click resolves against the right turn in either mode.
  let activeCitation = $state<{ turnKey: string; n: number } | null>(null)
  const activeSource = $derived.by(() => {
    if (!activeCitation) return null
    // Read-only transcripts (a viewed chat, or a resumed chat's history) key by record_id;
    // a resumed chat also has live turns below, so fall through to those if not found here.
    const detail = viewedConvo ?? resumedHistory
    if (detail) {
      const t = detail.turns.find((t) => t.record_id === activeCitation!.turnKey)
      const s = t?.sources.find((s) => s.n === activeCitation!.n)
      // A rehydrated source is degraded — no markers/figures/evaluation (not persisted). Shape it
      // as a SourceView so SourcePanel/SourceCard render it unchanged.
      if (s) {
        return { n: s.n, citation: s.citation, excerpt: s.excerpt, figure_id: null, chunk_key: null, markers: [], reranker_score: 0, evaluation: null }
      }
    }
    const t = turns.find((t) => String(t.id) === activeCitation!.turnKey)
    return t?.result?.sources.find((s) => s.n === activeCitation!.n) ?? null
  })

  let convoEl = $state<HTMLElement | null>(null)
  let taEl = $state<HTMLTextAreaElement | null>(null)
  // Follow a streaming answer only while the reader is already at the bottom, so a long
  // response never yanks them away from something they scrolled up to re-read.
  let pinned = true

  // Re-pull /api/health after an ingest so the header chunk count + the empty-corpus banner
  // reflect the new corpus (the backend rebuilds the controller before reporting "done").
  async function refreshHealth(): Promise<void> {
    try {
      health = await getHealth()
      status = 'ready'
    } catch {
      // leave the prior health/status; a transient blip shouldn't blank the header
    }
  }

  // History is a sidecar read — a failure must never break the chat (inform, don't block).
  async function refreshConversations(): Promise<void> {
    try {
      conversations = await listConversations()
    } catch {
      // keep the prior list
    }
  }

  // Conversation management (pin / archive / soft-delete): PATCH the flag, then refresh the list.
  async function pinConversation(sid: string, pinned: boolean): Promise<void> {
    try {
      await updateConversationMeta(sid, { pinned })
      await refreshConversations()
    } catch (e) {
      console.error('pin failed', e)
    }
  }
  async function archiveConversation(sid: string, archived: boolean): Promise<void> {
    try {
      await updateConversationMeta(sid, { archived })
      await refreshConversations()
    } catch (e) {
      console.error('archive failed', e)
    }
  }
  async function renameConversation(sid: string, title: string): Promise<void> {
    try {
      await updateConversationMeta(sid, { title })
      await refreshConversations()
    } catch (e) {
      console.error('rename failed', e)
    }
  }

  // Resizable left sidebar. Width is a client-only view preference (like the theme toggle),
  // persisted in localStorage; clamped so it can't be dragged uselessly narrow or wide.
  const SIDEBAR_MIN = 200
  const SIDEBAR_MAX = 480
  function loadSidebarWidth(): number {
    try {
      const v = Number(localStorage.getItem('sidebarWidth'))
      return v >= SIDEBAR_MIN && v <= SIDEBAR_MAX ? v : 260
    } catch {
      return 260
    }
  }
  let sidebarWidth = $state(loadSidebarWidth())

  // Collapsible sidebar (spec sub-item b). A desktop-only view preference — the same class as the
  // theme + width above (client-only, localStorage, never a backend setting). Collapsing hides the
  // rail via `.app.collapsed` under a min-width guard; the mobile off-canvas drawer (`sidebarOpen`)
  // is untouched. Expanding restores the persisted width unchanged (collapse ≠ resize).
  function loadSidebarCollapsed(): boolean {
    try {
      return localStorage.getItem('sidebarCollapsed') === '1'
    } catch {
      return false
    }
  }
  let sidebarCollapsed = $state(loadSidebarCollapsed())
  function toggleSidebar(): void {
    sidebarCollapsed = !sidebarCollapsed
    try {
      localStorage.setItem('sidebarCollapsed', sidebarCollapsed ? '1' : '0')
    } catch {
      /* ignore — collapse state just won't persist */
    }
  }

  function startResize(e: PointerEvent): void {
    e.preventDefault()
    const onMove = (ev: PointerEvent) => {
      sidebarWidth = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, ev.clientX))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      try {
        localStorage.setItem('sidebarWidth', String(Math.round(sidebarWidth)))
      } catch {
        /* ignore — width just won't persist */
      }
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }
  async function deleteConversation(sid: string): Promise<void> {
    // Soft-delete is reversible, but there's no restore UI yet — confirm to avoid a mis-click.
    if (
      !window.confirm(
        'Delete this conversation? It is removed from your history; the underlying records are kept.',
      )
    )
      return
    try {
      await updateConversationMeta(sid, { deleted: true })
      // If the deleted chat is the one on screen (viewed, resumed, or live), start fresh.
      if (viewing === sid || resumedHistory?.session_id === sid || sessionId === sid) {
        newConversation()
      }
      await refreshConversations()
    } catch (e) {
      console.error('delete failed', e)
    }
  }

  // Readiness gate (PR-M4): the frozen sidecar takes a few seconds to load models before
  // it accepts requests. Poll /api/health until it answers (or give up after ~60s), then
  // load the conversation history.
  $effect(() => {
    let cancelled = false
    void (async () => {
      for (let i = 0; i < 60 && !cancelled; i++) {
        try {
          const h = await getHealth()
          if (!cancelled) {
            health = h
            status = 'ready'
            void refreshConversations()
            // The composer's scope selector needs the folder list even if the user never
            // opens the Library.
            void refreshFolders()
          }
          return
        } catch {
          await new Promise((r) => setTimeout(r, 1000))
        }
      }
      if (!cancelled) status = 'down'
    })()
    return () => {
      cancelled = true
    }
  })

  // A scope naming a folder that no longer exists would silently become "search nothing" on the
  // next turn. Drop it the moment the folder leaves the list, so the selector can't lie.
  $effect(() => {
    if (chatScopeFolderId !== null && !folders.some((f) => f.id === chatScopeFolderId)) {
      chatScopeFolderId = null
    }
  })

  // Keep the newest content in view as tokens stream in / a turn is added / a chat is opened —
  // but only when the reader is pinned to the bottom (see `pinned`).
  $effect(() => {
    const last = turns[turns.length - 1]
    void last?.answer
    void turns.length
    void viewing
    if (pinned && convoEl) convoEl.scrollTop = convoEl.scrollHeight
  })

  // A reader who has scrolled up to re-read is no longer pinned; snap back on once they
  // return to the bottom (small slack so it engages just before the exact edge).
  function onConvoScroll(): void {
    if (!convoEl) return
    pinned = convoEl.scrollHeight - convoEl.scrollTop - convoEl.clientHeight < 80
  }

  // Grow the composer with its content up to a cap, then let it scroll. Reset to the base
  // height after a send (measuring `scrollHeight` on empty content would keep the tall size).
  function autogrow(): void {
    if (!taEl) return
    taEl.style.height = 'auto'
    taEl.style.height = `${Math.min(taEl.scrollHeight, 160)}px`
  }
  function resetComposer(): void {
    if (taEl) taEl.style.height = 'auto'
  }

  // Sample questions for the empty state — corpus-agnostic openers that run one-click on any
  // library. Picking one only prefills the existing composer (no turn sent, no new behavior); the
  // reader still presses Send. Kept generic because the corpus topics aren't known at this layer.
  const sampleQuestions = [
    'What are the main themes across my documents?',
    'Where do my sources agree and disagree?',
    'What are the key findings, with citations?',
  ]
  function useSample(q: string): void {
    input = q
    taEl?.focus()
  }

  async function send(): Promise<void> {
    const text = input.trim()
    if (!text || sending) return
    input = ''
    resetComposer()
    pinned = true // sending jumps the reader to their own new turn
    sending = true
    const idx =
      turns.push({
        id: nextId++,
        question: text,
        answer: '',
        result: null,
        streaming: true,
        error: null,
      }) - 1
    try {
      for await (const ev of streamChat(text, sessionId, overrides, undefined, chatScopeFolderId)) {
        if (ev.event === 'token') turns[idx].answer += ev.data
        else if (ev.event === 'result') turns[idx].result = JSON.parse(ev.data) as TurnResult
        // `step` events are advisory; ignored for now.
      }
    } catch (e) {
      turns[idx].error = String(e)
    } finally {
      turns[idx].streaming = false
      sending = false
      // The finished turn is now persisted — refresh the sidebar so this chat appears/updates.
      void refreshConversations()
    }
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send()
    }
  }

  // A/B-compare (U6): retrieve the current question under the locked defaults and the session
  // override, and show the source-set diff. $0 (no LLM); the composer text is left intact.
  async function doCompare(): Promise<void> {
    const text = input.trim()
    if (!text || sending || comparing) return
    comparing = true
    try {
      compareResult = await compareRetrieval(text, overrides, chatScopeFolderId)
      pinned = true // bring the fresh compare card into view
    } catch (e) {
      console.error('compare failed', e)
    } finally {
      comparing = false
    }
  }

  async function doExport(): Promise<void> {
    try {
      // Export the conversation on screen: the viewed past chat, else the live/resumed session.
      // The backend sources the transcript from the durable records by id, so both work.
      await exportConversation(viewing ?? sessionId, false)
    } catch (e) {
      console.error('export failed', e)
    }
  }

  // Clear the conversation and start a fresh question (U4). Resets the on-screen turns, any open
  // citation panel, the read-only view, and the composer — and mints a new sessionId so the
  // backend doesn't thread the previous conversation's context into the next question. Session
  // overrides (ADR-010) are left as-is: a deliberate sandbox setting, not conversation state.
  function newConversation(): void {
    if (sending) return
    turns = []
    activeCitation = null
    compareResult = null
    viewing = null
    viewedConvo = null
    resumedHistory = null
    input = ''
    resetComposer()
    nextId = 0
    sessionId = freshSessionId()
    pinned = true
    sidebarOpen = false
    taEl?.focus()
  }

  // Continue a viewed past chat (fresh-context resume). Switch the live session to it: its turns
  // become read-only reference above the composer, and new turns thread to the same session_id
  // (so they append + persist). The backend session for this id starts empty — new questions are
  // standalone corpus queries, not a replay of the old conversation (memory is a later increment).
  function resumeConversation(): void {
    if (!viewedConvo || !viewing) return
    resumedHistory = viewedConvo
    sessionId = viewing
    viewing = null
    viewedConvo = null
    turns = []
    nextId = 0
    activeCitation = null
    compareResult = null
    input = ''
    resetComposer()
    pinned = true
    sidebarOpen = false
    taEl?.focus()
  }

  // Open a past conversation read-only (H2). Selecting the live chat returns to it; the live
  // chat's in-memory state is never destroyed by viewing an old one.
  async function openConversation(sid: string): Promise<void> {
    sidebarOpen = false
    activeCitation = null
    if (sid === sessionId) {
      viewing = null
      viewedConvo = null
      return
    }
    try {
      viewedConvo = await getConversation(sid)
      viewing = sid
      pinned = true
    } catch (e) {
      console.error('open conversation failed', e)
    }
  }

  function backToCurrent(): void {
    viewing = null
    viewedConvo = null
    activeCitation = null
  }

  // Library documents are a sidecar read — a failure must never break the app (inform, don't block).
  async function refreshDocuments(): Promise<void> {
    try {
      documents = await listLibraryDocuments()
      documentsLoaded = true
    } catch {
      // keep the prior list
    }
  }

  // Tag families are a sidecar read, same inform-don't-block rule as documents.
  async function refreshFamilies(): Promise<void> {
    try {
      keywordFamilies = await listKeywordFamilies()
      // PR-2.5 D5 — a family write changes what a facet *unit* is, so a live selection has to be
      // re-pointed or the grid silently empties behind a chip that still looks selectable. The
      // Manage view is opened from the overlay, i.e. exactly where a selection is live. Mapped
      // against the whole library, not the active collection, so an out-of-collection selection
      // survives (it must stay removable).
      libraryKeywords = remapSelection(
        libraryKeywords,
        familyCanonicalMap(keywordFamilies),
        documents,
      )
    } catch {
      // keep the prior list
    }
  }

  // Folders are a sidecar read, same inform-don't-block rule as documents.
  async function refreshFolders(): Promise<void> {
    try {
      folders = await listFolders()
    } catch {
      // keep the prior list
    }
  }

  // Folder writes (ADR-025 F1). Write-then-refetch, like the family mutations above: the server
  // is the authority on names + counts, so we never patch the local list by hand. Membership
  // changes also refresh `documents`, whose `folder_ids` drive the grid filter.
  async function folderWrite(op: () => Promise<unknown>, alsoDocuments = false): Promise<void> {
    folderError = null
    try {
      await op()
    } catch (e) {
      folderError = e instanceof Error ? e.message : String(e)
      return
    }
    await refreshFolders()
    if (alsoDocuments) await refreshDocuments()
  }

  function createLibraryFolder(name: string): void {
    void folderWrite(() => createFolder(name))
  }

  function renameLibraryFolder(folderId: string, name: string): void {
    void folderWrite(() => renameFolder(folderId, name), true)
  }

  function deleteLibraryFolder(folderId: string): void {
    // A deleted folder can't stay the active collection or the Manage selection.
    if (libraryCollection.kind === 'folder' && libraryCollection.value === folderId) {
      libraryCollection = { kind: 'all' }
    }
    if (manageFolderId === folderId) manageFolderId = null
    void folderWrite(() => deleteFolder(folderId), true)
  }

  function addDocsToFolder(folderId: string, documentIds: string[]): void {
    void folderWrite(() => addDocumentsToFolder(folderId, documentIds), true)
  }

  function removeDocFromFolder(folderId: string, documentId: string): void {
    void folderWrite(() => removeDocumentFromFolder(folderId, documentId), true)
  }

  function openManageFolders(): void {
    manageFolderQuery = ''
    manageFoldersOpen = true
    folderError = null
    sidebarOpen = false
    void refreshFolders()
  }

  // The grid tile's "Add to folder…": same view, opened pre-filtered to that one document so the
  // picker shows it alone once a folder is chosen.
  function openManageFoldersForDoc(id: string): void {
    const doc = documents.find((d) => d.id === id)
    openManageFolders()
    manageFolderQuery = doc ? docLabel(doc) : ''
  }

  // Switch between Chat and Library. Entering Library closes any open citation panel and lazy-loads
  // the document list once; the live chat's in-memory state is preserved across the switch.
  function selectMode(m: 'chat' | 'library' | 'graph'): void {
    mode = m
    sidebarOpen = false
    activeCitation = null
    if (m === 'chat' && folders.length === 0) void refreshFolders()
    if (m === 'library' && !documentsLoaded) {
      void refreshDocuments()
      void refreshFamilies()
      void refreshFolders()
    }
    if (m === 'graph') {
      // The ego panel resolves doc_ids → titles from the library list, so it must be loaded too.
      if (!documentsLoaded) void refreshDocuments()
      if (!graphLoaded) void loadConceptGraph()
    }
  }

  // Rail ↔ main sync (Decision 4a): selecting a collection makes it the grid's content and
  // returns to grid level; clicking a document anywhere drills the main pane into it.
  function selectCollection(c: LibraryCollection): void {
    libraryCollection = c
    libraryDocId = null
    sidebarOpen = false
  }

  function openDocument(id: string): void {
    libraryDocId = id
    sidebarOpen = false
  }

  // Global search (spec sub-item a). Opening refreshes both lists (inform-don't-block): documents
  // lazy-load only on entering the Library, so a chat-only user must still be able to find a paper.
  function openSearch(): void {
    searchQuery = ''
    searchOpen = true
    void refreshConversations()
    if (!documentsLoaded) void refreshDocuments()
  }
  function closeSearch(): void {
    searchOpen = false
  }
  // Reuse the existing entry points (spec A6): a chat opens in Chat mode, a document in Library
  // mode. selectMode already lazy-loads what each mode needs; opening a doc in chat mode shows
  // nothing. Close the overlay on select.
  function searchOpenChat(sid: string): void {
    searchOpen = false
    if (mode !== 'chat') selectMode('chat')
    void openConversation(sid)
  }
  function searchOpenDoc(id: string): void {
    searchOpen = false
    if (mode !== 'library') selectMode('library')
    openDocument(id)
  }
  // Cmd/Ctrl-K toggles the overlay (spec A2). preventDefault so the browser's own find/location
  // bar never steals it; toggling closed is why it's not just `openSearch()`.
  function onGlobalKey(e: KeyboardEvent): void {
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault()
      if (searchOpen) closeSearch()
      else openSearch()
    }
  }

  // Metadata editing (ADR-013 — first browse-time write path). `editingDocId` opens the modal;
  // Save/Reset write then re-fetch the list so the tile reflects the new effective values
  // (mirrors how the conversation actions re-fetch). Reveal opens the OS file manager server-side.
  let editingDocId = $state<string | null>(null)
  const editingDoc = $derived(
    editingDocId ? (documents.find((d) => d.id === editingDocId) ?? null) : null,
  )
  async function saveDocMeta(patch: {
    title: string
    authors: string
    year: number | null
  }): Promise<void> {
    if (editingDocId === null) return
    try {
      await updateDocumentMeta(editingDocId, patch)
      await refreshDocuments()
    } catch {
      // inform-don't-block: a write failure leaves the prior values in place
    }
    editingDocId = null
  }
  async function resetDocMeta(): Promise<void> {
    if (editingDocId === null) return
    try {
      await resetDocumentMeta(editingDocId)
      await refreshDocuments()
    } catch {
      // keep the prior list on failure
    }
    editingDocId = null
  }
  async function revealDoc(id: string): Promise<void> {
    try {
      await revealDocument(id)
    } catch {
      // the source file may have moved since ingest — surface nothing, never crash the UI
    }
  }

  // Safe-delete (ADR-014): the ⋯ Delete opens a confirmation; on confirm the source file goes to
  // the Recycle Bin and the doc leaves the library + index. If the open doc was deleted, drop back
  // to its collection grid, then re-fetch the list.
  let deletingDocId = $state<string | null>(null)
  let deleteBusy = $state(false)
  const deletingDoc = $derived(
    deletingDocId ? (documents.find((d) => d.id === deletingDocId) ?? null) : null,
  )
  async function confirmDelete(): Promise<void> {
    if (deletingDocId === null) return
    deleteBusy = true
    try {
      await deleteDocument(deletingDocId)
      if (libraryDocId === deletingDocId) libraryDocId = null
      await refreshDocuments()
      deletingDocId = null
    } catch {
      // e.g. the file couldn't be moved to the Recycle Bin — leave the dialog open, doc intact
    }
    deleteBusy = false
  }

  // Back walks one level up: doc → its collection's grid, then collection → All documents.
  function libraryBack(): void {
    if (libraryDocId !== null) libraryDocId = null
    else libraryCollection = { kind: 'all' }
  }

  // The 0-match escape (Decision 5a): widen to All documents, keeping the query.
  function searchAll(): void {
    libraryCollection = { kind: 'all' }
    libraryDocId = null
  }

  // Keyword facets: toggle one on/off (AND semantics), or clear the whole selection.
  function toggleKeywordFacet(value: string): void {
    libraryKeywords = libraryKeywords.includes(value)
      ? libraryKeywords.filter((k) => k !== value)
      : [...libraryKeywords, value]
  }
  function clearKeywordFacets(): void {
    libraryKeywords = []
  }

  // Tag-family curation (feature-tag-families.md, PR-1). Each write refreshes the family list
  // (inform-don't-block: a failure just leaves the prior list, same as the document writes above).
  async function createFamily(canonical: string, members: string[]): Promise<void> {
    try {
      await createKeywordFamily(canonical, members)
      await refreshFamilies()
    } catch {
      // leave the prior list — the create form keeps its typed values for a retry
    }
  }
  async function renameFamily(familyId: string, canonical: string): Promise<void> {
    try {
      await renameKeywordFamily(familyId, canonical)
      await refreshFamilies()
    } catch {
      // keep the prior name
    }
  }
  async function addFamilyMemberKeyword(familyId: string, keyword: string): Promise<void> {
    try {
      await addFamilyMember(familyId, keyword)
      await refreshFamilies()
    } catch {
      // keep the prior membership
    }
  }
  async function removeFamilyMemberKeyword(familyId: string, keyword: string): Promise<void> {
    try {
      await removeFamilyMember(familyId, keyword)
      await refreshFamilies()
    } catch {
      // keep the prior membership
    }
  }
  async function deleteFamily(familyId: string): Promise<void> {
    try {
      await deleteKeywordFamily(familyId)
      await refreshFamilies()
    } catch {
      // keep the prior list
    }
  }

  // Detection (PR-2): a zero-LLM proposal pass, run on request (never automatically — the API
  // call loads/runs the embedder, not something to fire on every Manage-view open).
  async function runDetectFamilies(): Promise<void> {
    detecting = true
    detectError = null
    try {
      detectProposals = await detectKeywordFamilies()
    } catch (e) {
      detectError = e instanceof Error ? e.message : 'Detection failed.'
    }
    detecting = false
  }
  function dismissProposal(canonical: string): void {
    detectProposals = detectProposals.filter((p) => p.canonical !== canonical)
  }
  async function acceptProposal(p: KeywordFamilyProposal): Promise<void> {
    await createFamily(p.canonical, p.members)
    dismissProposal(p.canonical)
  }
  function closeManageKeywords(): void {
    manageKeywordsOpen = false
    detectProposals = []
    detectError = null
  }
</script>

<svelte:window onkeydown={onGlobalKey} />

<div class="app" class:collapsed={sidebarCollapsed} style="--sidebar-width: {sidebarWidth}px">
  <Sidebar
    {mode}
    {conversations}
    {documents}
    {folders}
    liveSessionId={sessionId}
    viewingSessionId={viewing}
    {libraryCollection}
    bind:libraryQuery
    open={sidebarOpen}
    onNew={newConversation}
    onSelect={openConversation}
    onSelectMode={selectMode}
    onSelectCollection={selectCollection}
    onManageFolders={openManageFolders}
    onClose={() => (sidebarOpen = false)}
    onPin={pinConversation}
    onArchive={archiveConversation}
    onDelete={deleteConversation}
    onRename={renameConversation}
  />
  <div
    class="resizer"
    role="separator"
    aria-orientation="vertical"
    aria-label="Resize sidebar"
    onpointerdown={startResize}
  ></div>

  <div class="content">
    <main class:wide={mode === 'library' || mode === 'graph'}>
      <header>
        <button class="hamburger" onclick={() => (sidebarOpen = true)} aria-label="Open conversations">
          <Icon name="menu" />
        </button>
        <button
          class="collapse-toggle"
          onclick={toggleSidebar}
          aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-pressed={sidebarCollapsed}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          type="button"
        >
          <Icon name="panel-left" size={16} />
        </button>
        <div class="brand">
          <span class="mark"><img src={appMark} alt="" width="32" height="32" /></span>
          <div class="brandtext">
            <span class="wordmark">proven<span class="wm-accent">ote</span></span>
            {#if status === 'ready' && health}
              <span class="meta">
                {health.chunk_count.toLocaleString()} chunks · {health.model} · {health.embedding_model}
              </span>
            {:else if status === 'connecting'}
              <span class="meta">starting the engine…</span>
            {:else}
              <span class="meta err">backend unreachable. Run <code>just api</code></span>
            {/if}
          </div>
        </div>
        <div class="actions">
          <button class="ghost" onclick={openSearch} aria-label="Search chats and documents" title="Search  (Ctrl/⌘ K)">
            <Icon name="search" size={15} />
          </button>
          <button
            class="ghost"
            onclick={doExport}
            disabled={mode !== 'chat' ||
              (viewing === null && resumedHistory === null && turns.length === 0)}
            ><Icon name="download" size={15} /> Export</button
          >
          <button class="ghost" onclick={() => (showSettings = true)} aria-label="Settings">
            <Icon name="settings" />
          </button>
        </div>
      </header>

      {#if mode === 'library'}
        <div class="library">
          <div class="libnav">
            {#if libraryDocId !== null || libraryCollection.kind !== 'all'}
              <button class="libback" onclick={libraryBack} aria-label="Back" title="Back">
                <Icon name="arrow-left" size={15} />
              </button>
            {/if}
            <nav class="crumbs" aria-label="Library location">
              <button
                class="crumb"
                onclick={() => selectCollection({ kind: 'all' })}
                disabled={libraryDocId === null && libraryCollection.kind === 'all'}
                type="button">Library</button
              >
              {#if libraryCollection.kind !== 'all'}
                <span class="crumbsep"><Icon name="chevron-right" size={13} /></span>
                <button
                  class="crumb"
                  onclick={() => (libraryDocId = null)}
                  disabled={libraryDocId === null}
                  type="button">{collectionLabel(libraryCollection, folderNames)}</button
                >
              {/if}
              {#if openDoc}
                <span class="crumbsep"><Icon name="chevron-right" size={13} /></span>
                <span class="crumb current" title={openDoc.filename}>{docLabel(openDoc)}</span>
              {/if}
            </nav>
            {#if libraryDocId === null}
              <div class="libsort">
                <button
                  class="sortbtn"
                  onclick={() => (libSortOpen = !libSortOpen)}
                  aria-haspopup="menu"
                  aria-expanded={libSortOpen}
                  title="Sort documents"
                  type="button"><Icon name="arrow-up-down" size={15} /></button
                >
                {#if libSortOpen}
                  <div
                    class="sort-backdrop"
                    onclick={() => (libSortOpen = false)}
                    role="presentation"
                  ></div>
                  <div class="sortmenu" role="menu">
                    {#each LIB_SORTS as s}
                      <button
                        class="sortitem"
                        class:on={librarySort === s.key}
                        role="menuitemradio"
                        aria-checked={librarySort === s.key}
                        onclick={() => setLibrarySort(s.key)}
                        type="button"
                      >
                        <span class="tick"
                          >{#if librarySort === s.key}<Icon name="check" size={13} />{/if}</span
                        >
                        {s.label}
                      </button>
                    {/each}
                  </div>
                {/if}
              </div>
              <div class="viewtoggle" role="group" aria-label="Layout">
                <button
                  class:active={libraryView === 'grid'}
                  onclick={() => setLibraryView('grid')}
                  aria-label="Grid view"
                  aria-pressed={libraryView === 'grid'}
                  title="Grid view"
                  type="button"><Icon name="layout-grid" size={15} /></button
                >
                <button
                  class:active={libraryView === 'list'}
                  onclick={() => setLibraryView('list')}
                  aria-label="List view"
                  aria-pressed={libraryView === 'list'}
                  title="List view"
                  type="button"><Icon name="list" size={15} /></button
                >
              </div>
            {/if}
          </div>

          {#if libraryDocId !== null}
            <LibraryBrowser docId={libraryDocId} onOpenDocument={openDocument} />
          {:else}
            <section class="libmain">
              {#if documents.length === 0}
                <div class="libempty">
                  <span class="state-mark"><Icon name="library" size={26} /></span>
                  <strong>No documents indexed yet</strong>
                  <p>Point doc_assistant at a folder of your documents to fill the library.</p>
                </div>
              {:else}
                <LibraryFilterStrip
                  selected={libraryKeywords}
                  resultCount={visibleDocs.length}
                  hasKeywords={facetList.length > 0}
                  onOpen={() => (keywordFilterOpen = true)}
                  onRemove={toggleKeywordFacet}
                  onClear={clearKeywordFacets}
                />
                {#if visibleDocs.length === 0}
                  <div class="libempty">
                    <span class="state-mark"><Icon name="search" size={26} /></span>
                    {#if libraryQuery.trim() !== '' || libraryKeywords.length > 0}
                      <strong>No documents match your filters</strong>
                      <p>
                        Nothing in {collectionLabel(libraryCollection, folderNames)} matches{#if libraryQuery.trim() !== ''}
                          “{libraryQuery.trim()}”{/if}.
                      </p>
                      {#if libraryKeywords.length > 0}
                        <button class="widen" onclick={clearKeywordFacets} type="button">
                          Clear keyword filters
                        </button>
                      {/if}
                      {#if libraryCollection.kind !== 'all'}
                        <button class="widen" onclick={searchAll} type="button">
                          Search all {documents.length} documents
                        </button>
                      {/if}
                    {:else}
                      <strong>Nothing in {collectionLabel(libraryCollection, folderNames)}</strong>
                      <p>This collection is empty right now.</p>
                    {/if}
                  </div>
                {:else}
                  <LibraryGrid
                    documents={visibleDocs}
                    view={libraryView}
                    activeKeywords={libraryKeywords}
                    {keywordsOf}
                    onOpenDocument={openDocument}
                    onEditMetadata={(id) => (editingDocId = id)}
                    onReveal={revealDoc}
                    onAddToFolder={openManageFoldersForDoc}
                    onDelete={(id) => (deletingDocId = id)}
                  />
                {/if}
              {/if}
            </section>
          {/if}
        </div>
      {:else if mode === 'graph'}
        <ConceptGraph
          graph={conceptGraph}
          loading={graphLoading}
          error={graphError}
          {documents}
          rebuildState={graphRebuildState}
          onRebuild={rebuildGraph}
          onOpenDocument={(id) => {
            selectMode('library')
            openDocument(id)
          }}
          onManageConcept={manageConcept}
          loadPresence={getConceptPresence}
        />
      {:else}
      <section class="conversation" bind:this={convoEl} onscroll={onConvoScroll}>
        {#if viewing && viewedConvo}
          <p class="readonly-note">
            Viewing a past conversation (read-only).
            <button class="linkish" onclick={resumeConversation}>Continue this chat</button>
            ·
            <button class="linkish" onclick={backToCurrent}>Back to current chat</button>
          </p>
          {#each viewedConvo.turns as t (t.record_id)}
            <ReadonlyTurn
              question={t.question}
              answer={t.answer}
              scope={t.scope}
              onCitationClick={(n) => (activeCitation = { turnKey: t.record_id, n })}
              activeCitationN={activeCitation?.turnKey === t.record_id ? activeCitation.n : null}
            />
          {/each}
        {:else}
          {#if resumedHistory}
            <p class="readonly-note resumed">
              Continuing <strong>{resumedHistory.title}</strong> · earlier turns are shown for
              reference. New questions start fresh — grounded in your corpus, not the old chat.
            </p>
            {#each resumedHistory.turns as t (t.record_id)}
              <ReadonlyTurn
                question={t.question}
                answer={t.answer}
                scope={t.scope}
                onCitationClick={(n) => (activeCitation = { turnKey: t.record_id, n })}
                activeCitationN={activeCitation?.turnKey === t.record_id ? activeCitation.n : null}
              />
            {/each}
            <div class="resume-divider"><span>continuing below</span></div>
          {/if}
          {#if status === 'ready' && health && health.chunk_count === 0}
            <div class="banner">
              <span class="state-mark"><Icon name="library" size={26} /></span>
              <strong>No documents indexed yet</strong>
              <p>
                Point doc_assistant at a folder of your documents to get started. It'll index them
                locally, then you can ask questions grounded in them.
              </p>
              <button class="primary" onclick={() => (showSettings = true)}>Choose a folder…</button>
            </div>
          {:else if turns.length === 0 && !resumedHistory}
            <div class="empty">
              <span class="state-mark"><Icon name="book-open-text" size={26} /></span>
              <h2>Ask your library a question</h2>
              <p>
                Every answer is grounded in your own documents, with inline citations, provenance,
                and per-claim review.
              </p>
              <div class="chips">
                {#each sampleQuestions as q}
                  <button class="chip" onclick={() => useSample(q)}>{q}</button>
                {/each}
              </div>
            </div>
          {/if}
          {#each turns as t (t.id)}
            <Turn
              question={t.question}
              answer={t.answer}
              result={t.result}
              streaming={t.streaming}
              error={t.error}
              onCitationClick={(n) => (activeCitation = { turnKey: String(t.id), n })}
              activeCitationN={activeCitation?.turnKey === String(t.id) ? activeCitation.n : null}
            />
          {/each}
          {#if compareResult}
            <CompareCard result={compareResult} onClose={() => (compareResult = null)} />
          {/if}
        {/if}
      </section>

      <footer>
        {#if viewing}
          <div class="viewing-bar">
            <button class="back" onclick={backToCurrent}
              ><Icon name="arrow-left" size={15} /> Back to current chat</button
            >
            <button class="resume" onclick={resumeConversation}
              ><Icon name="rotate-ccw" size={15} /> Continue this chat</button
            >
          </div>
        {:else}
          <textarea
            bind:this={taEl}
            bind:value={input}
            onkeydown={onKey}
            oninput={autogrow}
            placeholder="Ask your documents…  (Enter to send, Shift+Enter for newline)"
            rows="2"
            disabled={sending}
          ></textarea>
          {#if hasRetrievalOverride}
            <button
              class="compare"
              onclick={doCompare}
              disabled={sending || comparing || input.trim() === ''}
              title="See how your override changes retrieval for this question: locked defaults vs override, sources only, no answer ($0)"
              type="button"
            >
              {comparing ? 'Comparing…' : 'Test override'}
            </button>
          {/if}
          {#if folders.length > 0}
            <!-- ADR-025 F2 scope selector. Session-sticky, never persisted (see chatScopeFolderId).
                 "All documents" is always the first option, so returning to the whole library is
                 one click and never a hidden state. -->
            <label class="scopepick" class:scoped={chatScopeFolderId !== null}>
              <Icon name="folder" size={13} />
              <select
                bind:value={chatScopeFolderId}
                disabled={sending}
                aria-label="Search scope"
                title="Which documents this question searches"
              >
                <option value={null}>All documents</option>
                {#each folders as f (f.id)}
                  <option value={f.id}>{f.name} ({f.doc_count})</option>
                {/each}
              </select>
            </label>
          {/if}
          <button class="send" onclick={send} disabled={sending || input.trim() === ''} aria-busy={sending}>
            {#if sending}<span class="spinner" aria-hidden="true"></span>{:else}Send{/if}
          </button>
        {/if}
      </footer>
      {/if}
    </main>
  </div>
</div>

{#if showSettings}
  <Settings onClose={() => (showSettings = false)} onCorpusChanged={refreshHealth} bind:overrides />
{/if}

{#if activeCitation && activeSource}
  <SourcePanel source={activeSource} onClose={() => (activeCitation = null)} />
{/if}

{#if editingDoc}
  <LibraryMetaEditor
    doc={editingDoc}
    onSave={saveDocMeta}
    onReset={resetDocMeta}
    onClose={() => (editingDocId = null)}
  />
{/if}

{#if deletingDoc}
  <LibraryDeleteConfirm
    doc={deletingDoc}
    busy={deleteBusy}
    onConfirm={confirmDelete}
    onClose={() => (deletingDocId = null)}
  />
{/if}

{#if keywordFilterOpen}
  <LibraryKeywordFilter
    facets={facetList}
    docCounts={facetDocCounts}
    previewDocs={visibleDocs}
    selectedCount={libraryKeywords.length}
    families={keywordFamilies}
    onToggle={toggleKeywordFacet}
    onClear={clearKeywordFacets}
    onClose={() => (keywordFilterOpen = false)}
    onManage={() => (manageKeywordsOpen = true)}
  />
{/if}

{#if manageFoldersOpen}
  <LibraryManageFolders
    {folders}
    {documents}
    selectedId={manageFolderId}
    initialDocQuery={manageFolderQuery}
    error={folderError}
    onCreate={createLibraryFolder}
    onRename={renameLibraryFolder}
    onDelete={deleteLibraryFolder}
    onAddDocuments={addDocsToFolder}
    onRemoveDocument={removeDocFromFolder}
    onSelect={(id) => (manageFolderId = id)}
    onClose={() => (manageFoldersOpen = false)}
  />
{/if}

{#if manageKeywordsOpen}
  <LibraryManageKeywords
    families={keywordFamilies}
    {allKeywords}
    keywordDocCounts={rawKeywordDocCounts}
    proposals={detectProposals}
    {detecting}
    {detectError}
    onCreate={createFamily}
    onRename={renameFamily}
    onAddMember={addFamilyMemberKeyword}
    onRemoveMember={removeFamilyMemberKeyword}
    onDelete={deleteFamily}
    onDetect={runDetectFamilies}
    onAcceptProposal={acceptProposal}
    onDismissProposal={dismissProposal}
    onClose={closeManageKeywords}
  />
{/if}

{#if searchOpen}
  <GlobalSearch
    bind:query={searchQuery}
    results={searchResults}
    onSelectChat={searchOpenChat}
    onSelectDoc={searchOpenDoc}
    onClose={closeSearch}
  />
{/if}

<style>
  .app {
    display: flex;
    height: 100vh;
  }
  /* Drag handle between the sidebar and the content — a thin hit area with a hover cue. */
  .resizer {
    flex: none;
    width: 5px;
    margin-left: -3px;
    cursor: col-resize;
    background: transparent;
    z-index: 5;
    transition: background 0.15s ease;
  }
  .resizer:hover,
  .resizer:active {
    background: var(--accent);
    opacity: 0.5;
  }
  @media (max-width: 720px) {
    .resizer {
      display: none;
    }
  }
  /* Collapsed sidebar (spec sub-item b) — desktop only. The rail + its drag handle are removed
     from flow so the content fills the width; the header's collapse toggle brings them back at the
     persisted width. `:global(.sidebar)` reaches the child component's root (the class lives in
     Sidebar.svelte); the min-width guard leaves the mobile off-canvas drawer untouched. */
  @media (min-width: 721px) {
    .app.collapsed :global(.sidebar) {
      display: none;
    }
    .app.collapsed .resizer {
      display: none;
    }
  }
  .content {
    flex: 1;
    min-width: 0;
    display: flex;
    justify-content: center;
    overflow: hidden;
  }
  main {
    width: 100%;
    max-width: 820px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 0 1rem;
  }
  /* The 820px cap is the chat reading measure (~68ch). The library is an inventory
     grid, not prose — let it use the width so the grid reflows into more columns
     instead of floating in a centered column with empty margins in fullscreen. */
  main.wide {
    max-width: 1500px;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    padding: var(--space-3) 0;
    border-bottom: 1px solid var(--border);
  }
  .hamburger {
    display: none;
    font: inherit;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg);
    border-radius: 8px;
    padding: 0.2rem 0.55rem;
  }
  /* Desktop collapse toggle (spec b). Shares the header-left slot with the hamburger, split by the
     720 px breakpoint: hamburger on mobile, collapse toggle on desktop. */
  .collapse-toggle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: none;
    font: inherit;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg-2);
    border-radius: 8px;
    padding: 0.3rem;
  }
  .collapse-toggle:hover {
    color: var(--fg);
    background: var(--surface);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex: 1;
    min-width: 0;
  }
  .mark {
    flex: none;
    width: 32px;
    height: 32px;
    border-radius: 9px;
    overflow: hidden;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--shadow-1);
  }
  .mark img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .brandtext {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .wordmark {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    line-height: 1.15;
    color: var(--fg);
  }
  .wm-accent {
    color: var(--accent-wordmark);
  }
  .meta {
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .meta.err {
    color: var(--warn-fg);
  }
  .conversation {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-2) 0;
  }
  /* Library pane (L4): breadcrumb/Back/view-toggle bar over the grid or the drilled chunk view. */
  .library {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .libnav {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) 0;
    border-bottom: 1px solid var(--border);
    min-height: 2.4rem;
  }
  .libback {
    flex: none;
    display: inline-flex;
    align-items: center;
    padding: 0.28rem 0.5rem;
    color: var(--fg-2);
  }
  .libback:hover {
    color: var(--fg);
  }
  .crumbs {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 0.2rem;
    overflow: hidden;
  }
  .crumb {
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    border: none;
    background: none;
    color: var(--accent);
    padding: 0.15rem 0.25rem;
    border-radius: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .crumb:hover:not(:disabled) {
    background: var(--surface);
  }
  /* The current location isn't a link — a disabled tail crumb reads as "you are here". */
  .crumb:disabled,
  .crumb.current {
    color: var(--fg);
    cursor: default;
    opacity: 1;
    font-weight: 600;
  }
  .crumbsep {
    color: var(--fg-2);
    display: inline-flex;
    flex: none;
  }
  .libsort {
    position: relative;
    flex: none;
  }
  .sortbtn {
    display: inline-flex;
    align-items: center;
    padding: 0.28rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--fg-2);
    cursor: pointer;
  }
  .sortbtn:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  .sort-backdrop {
    position: fixed;
    inset: 0;
    z-index: 20;
  }
  .sortmenu {
    position: absolute;
    z-index: 21;
    top: calc(100% + 4px);
    right: 0;
    min-width: 200px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: var(--shadow-2);
    padding: 0.25rem;
  }
  .sortitem {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    width: 100%;
    padding: 0.4rem 0.5rem;
    border: none;
    background: none;
    color: var(--fg);
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 0.82rem;
    text-align: left;
    white-space: nowrap;
  }
  .sortitem:hover {
    background: var(--surface-2);
  }
  .sortitem.on {
    color: var(--accent);
  }
  .sortitem .tick {
    display: inline-flex;
    width: 13px;
    flex: none;
  }
  .viewtoggle {
    flex: none;
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .viewtoggle button {
    border: none;
    border-radius: 0;
    background: var(--surface);
    color: var(--fg-2);
    padding: 0.28rem 0.55rem;
    display: inline-flex;
    align-items: center;
  }
  .viewtoggle button.active {
    background: var(--surface-2);
    color: var(--accent);
  }
  .libmain {
    flex: 1;
    overflow-y: auto;
    min-width: 0;
  }
  .libempty {
    max-width: 540px;
    margin: var(--space-6) auto 0;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .libempty strong {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    font-weight: 600;
    color: var(--fg);
  }
  .libempty p {
    color: var(--fg-2);
    font-size: var(--text-sm);
    line-height: 1.6;
    max-width: 46ch;
    margin: var(--space-2) 0 var(--space-4);
  }
  .widen {
    font-size: var(--text-sm);
    color: var(--accent);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: var(--space-2) var(--space-3);
  }
  .widen:hover {
    border-color: var(--accent);
  }
  /* Empty + first-run states share one centered, mark-led layout (V2). */
  .empty,
  .banner {
    max-width: 540px;
    margin: var(--space-6) auto 0;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .state-mark {
    flex: none;
    width: 46px;
    height: 46px;
    border-radius: 12px;
    background: var(--surface-2);
    color: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: var(--space-4);
  }
  .empty h2,
  .banner strong {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    font-weight: 600;
    color: var(--fg);
    margin: 0;
  }
  .empty p,
  .banner p {
    color: var(--fg-2);
    font-size: var(--text-sm);
    line-height: 1.6;
    max-width: 46ch;
    margin: var(--space-2) 0 var(--space-4);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    justify-content: center;
  }
  .chip {
    font-size: var(--text-sm);
    color: var(--accent);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: var(--space-2) var(--space-3);
  }
  .chip:hover {
    border-color: var(--accent);
  }
  .readonly-note {
    font-size: 0.78rem;
    color: var(--fg-2);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    margin: 0 0 0.5rem;
  }
  .linkish {
    font: inherit;
    font-size: inherit;
    background: none;
    border: none;
    color: var(--accent);
    cursor: pointer;
    padding: 0;
    text-decoration: underline;
  }
  /* Resume banner: tinted with the accent so "continuing" reads distinct from "viewing". */
  .readonly-note.resumed {
    background: color-mix(in srgb, var(--accent) 8%, var(--surface));
    border-color: color-mix(in srgb, var(--accent) 25%, var(--border));
    color: var(--fg);
  }
  .readonly-note.resumed strong {
    font-weight: 600;
  }
  .resume-divider {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.4rem 0 0.8rem;
    color: var(--fg-2);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .resume-divider::before,
  .resume-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }
  .actions {
    display: flex;
    gap: 0.4rem;
    align-items: center;
  }
  .banner {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    box-shadow: var(--shadow-1);
    padding: var(--space-6) var(--space-5);
  }
  .banner .primary {
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
    font-weight: 600;
    padding: 0.45rem 1.1rem;
  }
  footer {
    display: flex;
    gap: var(--space-2);
    padding: var(--space-3) 0;
    border-top: 1px solid var(--border);
  }
  textarea {
    flex: 1;
    resize: none;
    font: inherit;
    padding: 0.5rem 0.6rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    min-height: 3.4rem;
    max-height: 160px;
    overflow-y: auto;
  }
  button {
    font: inherit;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg);
    padding: 0 1rem;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .viewing-bar {
    display: flex;
    gap: 0.5rem;
    width: 100%;
  }
  .back {
    flex: 1;
    padding: 0.6rem;
    color: var(--fg-2);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
  }
  .resume {
    flex: 1;
    padding: 0.6rem;
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
  }
  .send {
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
    font-weight: 600;
    min-width: 4.4rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  /* ADR-025 F2 scope selector — reads as a quiet control until a scope is set, then it is
     tinted so a narrowed conversation is visible without opening anything. */
  .scopepick {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    flex: none;
    padding: 0.25rem 0.4rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--fg-2);
    background: var(--surface);
  }
  .scopepick.scoped {
    color: var(--accent);
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
  }
  .scopepick select {
    font: inherit;
    font-size: 0.78rem;
    max-width: 11rem;
    border: none;
    background: none;
    color: inherit;
    cursor: pointer;
  }
  .scopepick select:focus {
    outline: none;
  }
  .scopepick select:disabled {
    cursor: not-allowed;
  }
  .spinner {
    width: 0.95em;
    height: 0.95em;
    border: 2px solid var(--accent-fg);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .spinner {
      animation: none;
    }
  }
  .ghost {
    font-size: 0.82rem;
    padding: 0.3rem 0.7rem;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }
  .compare {
    font-size: 0.82rem;
    white-space: nowrap;
    color: var(--fg-2);
  }
  @media (max-width: 720px) {
    .hamburger {
      display: inline-flex;
    }
    .collapse-toggle {
      display: none;
    }
  }
</style>
