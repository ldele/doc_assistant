<script lang="ts">
  import { untrack } from 'svelte'
  import type {
    ConversationDetail,
    KeywordFamily,
    KeywordFamilyProposal,
    LibraryDocument,
    LibraryFolder,
    TurnResult,
  } from './lib/core/types'
  import {
    addDocumentsToFolder,
    addFamilyMember,
    compareRetrieval,
    createFolder,
    createKeywordFamily,
    deleteFolder,
    deleteKeywordFamily,
    detectKeywordFamilies,
    getConceptPresence,
    getConversation,
    getHealth,
    getSetup,
    exportConversation,
    listFolders,
    listKeywordFamilies,
    deleteDocument,
    listLibraryDocuments,
    removeDocumentFromFolder,
    removeFamilyMember,
    renameFolder,
    renameKeywordFamily,
    resetDocumentMeta,
    revealDocument,
    streamChat,
    updateConversationMeta,
    updateDocumentMeta,
  } from './lib/core/api'
  import ChatPane from './lib/chat/ChatPane.svelte'
  import Settings from './lib/settings/Settings.svelte'
  import SourcePanel from './lib/chat/SourcePanel.svelte'
  import Sidebar from './lib/shell/Sidebar.svelte'
  import LibraryPane from './lib/library/LibraryPane.svelte'
  import LibraryGrid from './lib/library/LibraryGrid.svelte'
  import LibraryKeywordFilter from './lib/library/LibraryKeywordFilter.svelte'
  import LibraryManageKeywords from './lib/library/LibraryManageKeywords.svelte'
  import LibraryManageFolders from './lib/library/LibraryManageFolders.svelte'
  import LibraryMetaEditor from './lib/library/LibraryMetaEditor.svelte'
  import LibraryDeleteConfirm from './lib/library/LibraryDeleteConfirm.svelte'
  import ConfirmDialog from './lib/shell/ConfirmDialog.svelte'
  import ConceptGraph from './lib/graph/ConceptGraph.svelte'
  import GraphIndex from './lib/graph/GraphIndex.svelte'
  import ShortcutsDialog from './lib/shell/ShortcutsDialog.svelte'
  import AboutDialog from './lib/shell/AboutDialog.svelte'
  import LibraryTaxonomy from './lib/library/LibraryTaxonomy.svelte'
  import GlobalSearch from './lib/shell/GlobalSearch.svelte'
  import Topbar from './lib/shell/Topbar.svelte'
  import StatusBar from './lib/shell/StatusBar.svelte'
  import {
    type LibraryCollection,
    type LibrarySort,
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
    sameCollection,
  } from './lib/library/library'
  import { searchEverything } from './lib/shell/search'
  // Per-domain reactive state (`.svelte.ts` = a rune module). Cross-domain orchestration —
  // selectMode, the nav-history observer, the readiness gate, the chat-scope guard — stays in
  // this file on purpose: splitting it would only hide the coupling across import graphs.
  import {
    graph,
    graphLoaded,
    loadConceptGraph,
    rebuildGraph,
    useGraphHygiene,
  } from './lib/graph/graph.svelte'
  import {
    closeTaxonomy,
    openTaxonomy,
    selectTaxonomyField,
    taxonomy,
    taxonomyAddEdge,
    taxonomyAttachDocument,
    taxonomyRemoveEdge,
  } from './lib/library/taxonomy.svelte'
  import { LIB_SORTS, libPrefs, setLibrarySort, setLibraryView } from './lib/library/prefs.svelte'
  import {
    SIDEBAR_MAX,
    SIDEBAR_MIN,
    sidebarPrefs,
    startSidebarResize,
    toggleSidebarCollapsed,
  } from './lib/shell/prefs.svelte'
  // Shell chrome state (leaf module — imports no sibling state). Phase 2 lets the pane components
  // import this directly instead of taking ~20 props each.
  import { shell } from './lib/shell/shell.svelte'
  import {
    chat,
    freshSessionId,
    nextTurnId,
    resetComposer,
    resetTurnIds,
    setPinned,
    useChatAutoscroll,
  } from './lib/chat/chat.svelte'
  import {
    archiveConversation,
    conversations,
    pinConversation,
    refreshConversations,
    renameConversation,
  } from './lib/chat/conversations.svelte'

  // The Test-override button only exists while a retrieval-affecting override is set — with none,
  // both sides retrieve identically and the button is dead weight (2026-07-13 UX review). Settings
  // writes these fields only when touched; Reset returns chat.overrides to {}.
  const hasRetrievalOverride = $derived(
    chat.overrides.top_k != null || chat.overrides.use_multi_query != null,
  )

  // Conversation history (feature-conversation-history.md). `viewing` is the session_id shown as a
  // read-only transcript; `null` means the live chat (composer + claims bound to `chat.sessionId`).
  let viewing = $state<string | null>(null)
  let viewedConvo = $state<ConversationDetail | null>(null)
  // Resume (fresh-context): a reopened past chat the user chose to *continue*. Its chat.turns render
  // read-only above the composer for reference; `chat.sessionId` is switched to it so new chat.turns thread
  // to the same conversation and persist. The in-memory backend session starts fresh (empty
  // history), so new questions are standalone corpus queries — no replay of the old chat.turns.
  let resumedHistory = $state<ConversationDetail | null>(null)

  // Global-search overlay (docs/specs/feature-app-shell-search-collapse.md, sub-item a). A
  // navigation search over chats + documents, opened from the header or Cmd/Ctrl-K. App owns the
  // query + derives the results (searchEverything is pure/tested); GlobalSearch just renders.

  // Library space (feature-library-browser.md L1; nav redesign feature-library-redesign.md L4
  // Phase A). `mode` swaps the sidebar + main pane between Chat and Library; the chat state
  // (chat.turns/viewing/chat.sessionId) is untouched by the switch. Navigation model: the rail picks the
  // active *collection*, the main pane shows it as an inventory grid, and opening a document
  // drills down in place to the chunk view (breadcrumb + Back walk back up).
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
  const searchResults = $derived(searchEverything(shell.searchQuery, conversations.list, documents))

  // Folders (ADR-025 F1, docs/specs/feature-corpus-folders.md). Manual Library organisation.
  // The rail renders this list rather than deriving groups from `documents`, so a folder with
  // zero members is still visible and therefore fillable (spec D3). `folderError` surfaces a
  // 400 (blank/collision) in the Manage view without blocking anything else.
  let folders = $state<LibraryFolder[]>([])
  // ADR-025 F2 — the chat retrieval scope. Sticky across chat.turns, in memory ONLY: a reload
  // returns to the whole library. Persisting it is the rejected option — a scope you forgot
  // you set silently narrows every future answer, which is the exact lie the integrity layer
  // exists to prevent. Deliberately separate from `libraryCollection`: filtering the Library
  // grid and scoping a conversation are two different intentions.
  let chatScopeFolderId = $state<string | null>(null)
  let manageFoldersOpen = $state(false)
  let manageFolderId = $state<string | null>(null)
  let manageFolderQuery = $state('')

  // Library select mode (batch add-to-folder). Selection is App-owned (LibraryGrid stays dumb);
  // bulk delete is deliberately NOT here — deferred pending its own ADR (ui-checklist).
  let libSelectMode = $state(false)
  let libSelected = $state<string[]>([])
  let libAddMenuOpen = $state(false)
  function toggleLibSelected(id: string): void {
    libSelected = libSelected.includes(id) ? libSelected.filter((x) => x !== id) : [...libSelected, id]
  }
  function exitLibSelect(): void {
    libSelectMode = false
    libSelected = []
    libAddMenuOpen = false
  }
  function addSelectionToFolder(folderId: string): void {
    if (libSelected.length > 0) addDocsToFolder(folderId, libSelected)
    exitLibSelect()
  }
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

  // Concept graph — state + its loader/rebuild live in `lib/graph/graph.svelte.ts`; lazy-loaded on
  // first entry to Graph mode (see selectMode). Only the cross-domain bits stay here.
  function selectGraphConcept(id: string): void {
    graph.selectedId = id
    shell.sidebarOpen = false // mobile drawer: selecting navigates, like selectCollection
  }
  useGraphHygiene() // intra-domain: a rebuild can drop the selected concept
  // Chat transcript autoscroll. `viewing` is passed as a getter because it is conversation-view
  // state owned here, and the effect must re-run when it changes (opening a past chat scrolls
  // to the bottom too).
  useChatAutoscroll(() => viewing)

  // Taxonomy view (docs/specs/feature-taxonomy-view.md, ADR-028 2b). A dedicated modal that renders
  // the curated field forest + *places* concepts/documents onto it. App owns the data; LibraryTaxonomy
  // is a dumb renderer. Opened from the Library rail, or from a graph node's Place action (which
  // preselects that concept via `taxonomy.focusConceptId`). Decoupled from the top-level nav — it's a
  // global overlay like Settings/Search, so it opens from any mode.

  // Deep-link from a graph node to curate its concept (ADR-017 A1 — the graph never writes the
  // vocabulary; the Manage-keywords view owns every edit). Switches to Library and opens the view.
  function manageConcept(_conceptId: string, _label: string): void {
    selectMode('library')
    manageKeywordsOpen = true
  }

  // Cross-domain wrapper over the taxonomy module: the attach picker lists *documents*, which
  // lazy-load only on entering the Library, so a user who opens the modal from Chat or Graph would
  // otherwise get an empty picker. The document list is library state, so the pull stays here
  // rather than reaching across from `taxonomy.svelte.ts`.
  function openTaxonomyView(focusConceptId: string | null = null): void {
    if (!documentsLoaded) void refreshDocuments()
    void openTaxonomy(focusConceptId)
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
    sortDocs(facetFilter(searchedDocs, libraryKeywords, keywordsOf), libPrefs.sort),
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
  const activeSource = $derived.by(() => {
    if (!chat.activeCitation) return null
    // Read-only transcripts (a viewed chat, or a resumed chat's history) key by record_id;
    // a resumed chat also has live chat.turns below, so fall through to those if not found here.
    const detail = viewedConvo ?? resumedHistory
    if (detail) {
      const t = detail.turns.find((t) => t.record_id === chat.activeCitation!.turnKey)
      const s = t?.sources.find((s) => s.n === chat.activeCitation!.n)
      // A rehydrated source is degraded — no markers/figures/evaluation (not persisted). Shape it
      // as a SourceView so SourcePanel/SourceCard render it unchanged.
      if (s) {
        return { n: s.n, citation: s.citation, excerpt: s.excerpt, figure_id: null, chunk_key: null, markers: [], reranker_score: 0, evaluation: null }
      }
    }
    const t = chat.turns.find((t) => String(t.id) === chat.activeCitation!.turnKey)
    return t?.result?.sources.find((s) => s.n === chat.activeCitation!.n) ?? null
  })


  // Re-pull /api/health after an ingest so the header chunk count + the empty-corpus banner
  // reflect the new corpus (the backend rebuilds the controller before reporting "done").
  async function refreshHealth(): Promise<void> {
    try {
      shell.health = await getHealth()
      shell.status = 'ready'
    } catch {
      // leave the prior health/status; a transient blip shouldn't blank the header
    }
    // ADR-034: indexing a folder can complete the setup checklist, so the banner must re-read it.
    void refreshSetup()
  }

  // First-run readiness (ADR-034). Failure is silent on purpose: this drives an advisory banner,
  // and a blip must not put a "finish setup" card in front of a working install.
  async function refreshSetup(): Promise<void> {
    try {
      shell.setup = await getSetup()
    } catch {
      // keep whatever we last knew
    }
  }

  // App menu (☰) + its two info modals (keyboard shortcuts, about). The menu is the top-toolbar's
  // "more" surface; Settings has its own gear too (a fast path), so it appears in both.

  // Browser-style navigation history (top-toolbar ← →). A "view" is the navigable snapshot: the
  // mode plus each mode's location (library collection + open document, graph selection). A passive
  // $effect observes those four fields and records a new entry whenever they change; ← / → replay a
  // recorded entry through the real navigation paths. Chat is tracked at mode granularity (opening a
  // past conversation is its own in-rail affordance, not a history step).
  type NavEntry = {
    mode: 'chat' | 'library' | 'graph'
    collection: LibraryCollection
    docId: string | null
    graphId: string | null
  }
  const NAV_CAP = 50
  let navStack = $state<NavEntry[]>([])
  let navIndex = $state(-1)
  const canNavBack = $derived(navIndex > 0)
  const canNavForward = $derived(navIndex >= 0 && navIndex < navStack.length - 1)
  function navEq(a: NavEntry, b: NavEntry): boolean {
    return (
      a.mode === b.mode &&
      a.docId === b.docId &&
      a.graphId === b.graphId &&
      sameCollection(a.collection, b.collection)
    )
  }
  $effect(() => {
    // Tracked reads (the deps): a change to any of these is a navigation.
    const entry: NavEntry = {
      mode: shell.mode,
      collection: libraryCollection,
      docId: libraryDocId,
      graphId: graph.selectedId,
    }
    // Untracked: reading/writing the stack here must not feed back into this effect.
    untrack(() => {
      const top = navStack[navIndex]
      if (top && navEq(top, entry)) return
      let base = navStack.slice(0, navIndex + 1)
      base.push(entry)
      if (base.length > NAV_CAP) base = base.slice(base.length - NAV_CAP)
      navStack = base
      navIndex = base.length - 1
    })
  })
  // Replay through the real navigation functions so their side effects (lazy-loads, closing the
  // citation panel) fire; setting navIndex first means the observer sees the restored state already
  // matches navStack[navIndex] and does not re-record it.
  function applyNav(e: NavEntry): void {
    selectMode(e.mode)
    libraryCollection = e.collection
    libraryDocId = e.docId
    graph.selectedId = e.graphId
  }
  function navBack(): void {
    if (navIndex <= 0) return
    navIndex -= 1
    applyNav(navStack[navIndex])
  }
  function navForward(): void {
    if (navIndex >= navStack.length - 1) return
    navIndex += 1
    applyNav(navStack[navIndex])
  }

  // Soft-delete is reversible, but there's no restore UI yet — confirm via an in-app dialog (not the
  // native window.confirm, which shows OS "localhost says" chrome) to avoid a mis-click.
  let pendingDeleteConvId = $state<string | null>(null)
  let deleteConvBusy = $state(false)
  function deleteConversation(sid: string): void {
    pendingDeleteConvId = sid
  }
  async function confirmDeleteConversation(): Promise<void> {
    const sid = pendingDeleteConvId
    if (sid === null) return
    deleteConvBusy = true
    try {
      await updateConversationMeta(sid, { deleted: true })
      // If the deleted chat is the one on screen (viewed, resumed, or live), start fresh.
      if (viewing === sid || resumedHistory?.session_id === sid || chat.sessionId === sid) {
        newConversation()
      }
      await refreshConversations()
      pendingDeleteConvId = null
    } catch (e) {
      console.error('delete failed', e)
      // Leave the dialog open so the user can retry or cancel.
    }
    deleteConvBusy = false
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
            shell.health = h
            shell.status = 'ready'
            void refreshSetup() // ADR-034 — what this install still needs, if anything
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
      if (!cancelled) shell.status = 'down'
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


  // Sample questions for the empty state — corpus-agnostic openers that run one-click on any
  // library. Picking one only prefills the existing composer (no turn sent, no new behavior); the
  // reader still presses Send. Kept generic because the corpus topics aren't known at this layer.
  const sampleQuestions = [
    'What are the main themes across my documents?',
    'Where do my sources agree and disagree?',
    'What are the key findings, with citations?',
  ]
  function useSample(q: string): void {
    chat.input = q
    chat.taEl?.focus()
  }

  async function send(): Promise<void> {
    const text = chat.input.trim()
    if (!text || chat.sending) return
    chat.input = ''
    resetComposer()
    setPinned(true) // chat.sending jumps the reader to their own new turn
    chat.sending = true
    const idx =
      chat.turns.push({
        id: nextTurnId(),
        question: text,
        answer: '',
        result: null,
        streaming: true,
        error: null,
      }) - 1
    try {
      for await (const ev of streamChat(text, chat.sessionId, chat.overrides, undefined, chatScopeFolderId)) {
        if (ev.event === 'token') chat.turns[idx].answer += ev.data
        else if (ev.event === 'result') chat.turns[idx].result = JSON.parse(ev.data) as TurnResult
        // `step` events are advisory; ignored for now.
      }
    } catch (e) {
      chat.turns[idx].error = String(e)
    } finally {
      chat.turns[idx].streaming = false
      chat.sending = false
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
    const text = chat.input.trim()
    if (!text || chat.sending || chat.comparing) return
    chat.comparing = true
    try {
      chat.compareResult = await compareRetrieval(text, chat.overrides, chatScopeFolderId)
      setPinned(true) // bring the fresh compare card into view
    } catch (e) {
      console.error('compare failed', e)
    } finally {
      chat.comparing = false
    }
  }

  async function doExport(): Promise<void> {
    try {
      // Export the conversation on screen: the viewed past chat, else the live/resumed session.
      // The backend sources the transcript from the durable records by id, so both work.
      await exportConversation(viewing ?? chat.sessionId, false)
    } catch (e) {
      console.error('export failed', e)
    }
  }

  // Clear the conversation and start a fresh question (U4). Resets the on-screen chat.turns, any open
  // citation panel, the read-only view, and the composer — and mints a new chat.sessionId so the
  // backend doesn't thread the previous conversation's context into the next question. Session
  // chat.overrides (ADR-010) are left as-is: a deliberate sandbox setting, not conversation state.
  function newConversation(): void {
    if (chat.sending) return
    chat.turns = []
    chat.activeCitation = null
    chat.compareResult = null
    viewing = null
    viewedConvo = null
    resumedHistory = null
    chat.input = ''
    resetComposer()
    resetTurnIds()
    chat.sessionId = freshSessionId()
    setPinned(true)
    shell.sidebarOpen = false
    chat.taEl?.focus()
  }

  // Continue a viewed past chat (fresh-context resume). Switch the live session to it: its chat.turns
  // become read-only reference above the composer, and new chat.turns thread to the same session_id
  // (so they append + persist). The backend session for this id starts empty — new questions are
  // standalone corpus queries, not a replay of the old conversation (memory is a later increment).
  function resumeConversation(): void {
    if (!viewedConvo || !viewing) return
    resumedHistory = viewedConvo
    chat.sessionId = viewing
    viewing = null
    viewedConvo = null
    chat.turns = []
    resetTurnIds()
    chat.activeCitation = null
    chat.compareResult = null
    chat.input = ''
    resetComposer()
    setPinned(true)
    shell.sidebarOpen = false
    chat.taEl?.focus()
  }

  // Open a past conversation read-only (H2). Selecting the live chat returns to it; the live
  // chat's in-memory state is never destroyed by viewing an old one.
  async function openConversation(sid: string): Promise<void> {
    shell.sidebarOpen = false
    chat.activeCitation = null
    if (sid === chat.sessionId) {
      viewing = null
      viewedConvo = null
      return
    }
    try {
      viewedConvo = await getConversation(sid)
      viewing = sid
      setPinned(true)
    } catch (e) {
      console.error('open conversation failed', e)
    }
  }

  function backToCurrent(): void {
    viewing = null
    viewedConvo = null
    chat.activeCitation = null
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
    shell.sidebarOpen = false
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
    shell.mode = m
    shell.sidebarOpen = false
    chat.activeCitation = null
    if (m === 'chat' && folders.length === 0) void refreshFolders()
    if (m === 'library' && !documentsLoaded) {
      void refreshDocuments()
      void refreshFamilies()
      void refreshFolders()
    }
    if (m === 'graph') {
      // The ego panel resolves doc_ids → titles from the library list, so it must be loaded too.
      if (!documentsLoaded) void refreshDocuments()
      if (!graphLoaded()) void loadConceptGraph()
    }
  }

  // Rail ↔ main sync (Decision 4a): selecting a collection makes it the grid's content and
  // returns to grid level; clicking a document anywhere drills the main pane into it.
  function selectCollection(c: LibraryCollection): void {
    libraryCollection = c
    libraryDocId = null
    shell.sidebarOpen = false
  }

  function openDocument(id: string): void {
    libraryDocId = id
    shell.sidebarOpen = false
  }

  // Global search (spec sub-item a). Opening refreshes both lists (inform-don't-block): documents
  // lazy-load only on entering the Library, so a chat-only user must still be able to find a paper.
  function openSearch(): void {
    shell.searchQuery = ''
    shell.searchOpen = true
    void refreshConversations()
    if (!documentsLoaded) void refreshDocuments()
  }
  function closeSearch(): void {
    shell.searchOpen = false
  }
  // Reuse the existing entry points (spec A6): a chat opens in Chat mode, a document in Library
  // mode. selectMode already lazy-loads what each mode needs; opening a doc in chat mode shows
  // nothing. Close the overlay on select.
  function searchOpenChat(sid: string): void {
    shell.searchOpen = false
    if (shell.mode !== 'chat') selectMode('chat')
    void openConversation(sid)
  }
  function searchOpenDoc(id: string): void {
    shell.searchOpen = false
    if (shell.mode !== 'library') selectMode('library')
    openDocument(id)
  }
  // Cmd/Ctrl-K toggles the overlay (spec A2). preventDefault so the browser's own find/location
  // bar never steals it; toggling closed is why it's not just `openSearch()`.
  function onGlobalKey(e: KeyboardEvent): void {
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault()
      if (shell.searchOpen) closeSearch()
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

<!-- The graph-mode sidebar rail. App composes it (data + selection are App-owned) and hands it to
     Sidebar as a snippet, so Sidebar stays a dumb renderer without ~8 more graph props. -->
{#snippet graphRail()}
  <GraphIndex
    nodes={graph.data?.nodes ?? []}
    gaps={graph.data?.gaps ?? []}
    selectedId={graph.selectedId}
    bind:showUnderConnected={graph.showUnderConnected}
    loading={graph.loading}
    built={graph.data !== null}
    graphError={graph.error}
    onSelectConcept={selectGraphConcept}
  />
{/snippet}

<div class="app" class:collapsed={sidebarPrefs.collapsed} style="--sidebar-width: {sidebarPrefs.width}px">
  <!-- Unified top toolbar (browser-chrome shell): one bar across the whole window carrying the app
       menu, sidebar toggle, back/forward, brand, the mode tabs, and search/settings — the pattern
       replaces the old split of mode-pills-in-sidebar + actions-in-header. -->
  <Topbar
    canBack={canNavBack}
    canForward={canNavForward}
    onNavBack={navBack}
    onNavForward={navForward}
    onSelectMode={selectMode}
    onOpenSearch={openSearch}
    exportDisabled={viewing === null && resumedHistory === null && chat.turns.length === 0}
    onExport={doExport}
  />

  <div class="below">
  <Sidebar
    mode={shell.mode}
    conversations={conversations.list}
    {documents}
    {folders}
    liveSessionId={chat.sessionId}
    viewingSessionId={viewing}
    {libraryCollection}
    bind:libraryQuery
    open={shell.sidebarOpen}
    {graphRail}
    onNew={newConversation}
    onSelect={openConversation}
    onSelectCollection={selectCollection}
    onManageFolders={openManageFolders}
    onOpenTaxonomy={() => openTaxonomyView()}
    onClose={() => (shell.sidebarOpen = false)}
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
    onpointerdown={startSidebarResize}
  ></div>

  <div class="content">
    <div class="viewport">
    <main class:wide={shell.mode === 'library' || shell.mode === 'graph'}>
      {#if shell.mode === 'library'}
        <LibraryPane
          {documents}
          {visibleDocs}
          {facetList}
          {keywordsOf}
          {openDoc}
          {folderNames}
          {libraryCollection}
          {libraryDocId}
          {libraryQuery}
          {libraryKeywords}
          {folders}
          {libSelectMode}
          {libSelected}
          {libAddMenuOpen}
          onLibraryBack={libraryBack}
          onOpenDocument={openDocument}
          onSearchAll={searchAll}
          onSelectCollection={selectCollection}
          onSetDocId={(id) => (libraryDocId = id)}
          onToggleKeywordFacet={toggleKeywordFacet}
          onClearKeywordFacets={clearKeywordFacets}
          onOpenKeywordFilter={() => (keywordFilterOpen = true)}
          onEnterSelectMode={() => (libSelectMode = true)}
          onExitSelectMode={exitLibSelect}
          onToggleSelected={toggleLibSelected}
          onSetAddMenuOpen={(open) => (libAddMenuOpen = open)}
          onAddSelectionToFolder={addSelectionToFolder}
          onClearSelection={() => (libSelected = [])}
          onOpenManageFolders={openManageFolders}
          onEditDoc={(id) => (editingDocId = id)}
          onDeleteDoc={(id) => (deletingDocId = id)}
          onRevealDoc={revealDoc}
          onManageFoldersForDoc={openManageFoldersForDoc}
        />
      {:else if shell.mode === 'graph'}
        <ConceptGraph
          graph={graph.data}
          loading={graph.loading}
          error={graph.error}
          {documents}
          rebuildState={graph.rebuildState}
          selectedId={graph.selectedId}
          showUnderConnected={graph.showUnderConnected}
          onRebuild={rebuildGraph}
          onOpenDocument={(id) => {
            selectMode('library')
            openDocument(id)
          }}
          onManageConcept={manageConcept}
          onPlaceConcept={(id) => openTaxonomyView(id)}
          onSelectConcept={selectGraphConcept}
          loadPresence={getConceptPresence}
        />
      {:else}
        <ChatPane
          {viewing}
          {viewedConvo}
          {resumedHistory}
          {folders}
          bind:chatScopeFolderId
          {hasRetrievalOverride}
          {sampleQuestions}
          onSend={send}
          {onKey}
          onCompare={doCompare}
          onUseSample={useSample}
          onResume={resumeConversation}
          onBackToCurrent={backToCurrent}
        />
      {/if}
    </main>
    </div>
  </div>
  </div>

  <StatusBar />
</div>

{#if shell.showSettings}
  <Settings onClose={() => (shell.showSettings = false)} onCorpusChanged={refreshHealth} bind:overrides={chat.overrides} />
{/if}

{#if shell.showShortcuts}
  <ShortcutsDialog onClose={() => (shell.showShortcuts = false)} />
{/if}

{#if shell.showAbout}
  <AboutDialog
    chunks={shell.health?.chunk_count ?? null}
    model={shell.health?.model ?? null}
    embedding={shell.health?.embedding_model ?? null}
    onClose={() => (shell.showAbout = false)}
  />
{/if}

{#if chat.activeCitation && activeSource}
  <SourcePanel source={activeSource} onClose={() => (chat.activeCitation = null)} />
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

{#if pendingDeleteConvId !== null}
  <ConfirmDialog
    title="Delete this conversation?"
    body="It is removed from your history; the underlying records are kept."
    confirmLabel="Delete"
    busy={deleteConvBusy}
    onConfirm={confirmDeleteConversation}
    onClose={() => (pendingDeleteConvId = null)}
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

{#if shell.searchOpen}
  <GlobalSearch
    bind:query={shell.searchQuery}
    results={searchResults}
    onSelectChat={searchOpenChat}
    onSelectDoc={searchOpenDoc}
    onClose={closeSearch}
  />
{/if}

{#if taxonomy.open}
  <LibraryTaxonomy
    view={taxonomy.view}
    fieldDetail={taxonomy.fieldDetail}
    loading={taxonomy.loading}
    error={taxonomy.error}
    {documents}
    concepts={taxonomy.concepts}
    focusConceptId={taxonomy.focusConceptId}
    onSelectField={selectTaxonomyField}
    onAddEdge={taxonomyAddEdge}
    onRemoveEdge={taxonomyRemoveEdge}
    onAttachDocument={taxonomyAttachDocument}
    onClose={closeTaxonomy}
  />
{/if}

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  /* The window below the full-width toolbar: sidebar │ resizer │ content. Positioned so the mobile
     off-canvas drawer anchors here (below the toolbar) rather than over it. */
  .below {
    flex: 1;
    min-height: 0;
    display: flex;
    position: relative;
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
  /* Collapsed sidebar — desktop only. The toolbar keeps the mode tabs + search reachable, so
     collapsing simply hides the rail + its handle (no mini-rail needed). The min-width guard
     leaves the mobile off-canvas drawer untouched. */
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
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }
  .viewport {
    flex: 1;
    min-height: 0;
    display: flex;
    justify-content: center;
    overflow: hidden;
  }
  main {
    width: 100%;
    max-width: 820px;
    height: 100%;
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


</style>
