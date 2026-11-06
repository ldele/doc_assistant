<script lang="ts">
  import { untrack } from 'svelte'
  import type {
    ConversationDetail,
    KeywordFamilyProposal,
    MemberBreadth,
    TurnResult,
  } from './lib/core/types'
  import {
    compareRetrieval,
    getConceptPresence,
    getConversation,
    getHealth,
    getSetup,
    exportConversation,
    listKeywordFamilies,
    deleteDocument,
    locateChunk,
    streamChat,
    bulkUpdateConversations,
    updateConversationMeta,
  } from './lib/core/api'
  import { GRAPH_TAB_ENABLED } from './lib/core/features'
  import Chat from './screens/chat/Chat.svelte'
  import Settings from './screens/settings/Settings.svelte'
  import SourcePanel from './screens/chat/SourcePanel.svelte'
  import Sidebar from './shell/Sidebar.svelte'
  import Library from './screens/library/Library.svelte'
  import Icon from './shell/Icon.svelte'
  import AddDocumentsReview from './screens/library/AddDocumentsReview.svelte'
  import AddDocumentsChooser from './screens/library/AddDocumentsChooser.svelte'
  import FilterByKeyword from './screens/library/FilterByKeyword.svelte'
  import ManageKeywords from './screens/manage-keywords/ManageKeywords.svelte'
  import ManageFolders from './screens/library/ManageFolders.svelte'
  import EditMetadata from './screens/library/EditMetadata.svelte'
  import DeleteDocument from './screens/library/DeleteDocument.svelte'
  import RerunIngestion from './screens/document-page/RerunIngestion.svelte'
  import ConfirmDialog from './shell/ConfirmDialog.svelte'
  import Graph from './screens/graph/Graph.svelte'
  import ConceptIndex from './screens/graph/ConceptIndex.svelte'
  import ChatHistory from './screens/chat/ChatHistory.svelte'
  import KeyboardShortcuts from './shell/KeyboardShortcuts.svelte'
  import About from './shell/About.svelte'
  import Taxonomy from './screens/taxonomy/Taxonomy.svelte'
  import GlobalSearch from './shell/GlobalSearch.svelte'
  import Topbar from './shell/Topbar.svelte'
  import StatusBar from './shell/StatusBar.svelte'
  import {
    type LibraryCollection,
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
    invalidateGraph,
    loadConceptGraph,
    rebuildGraph,
    restoreGraph,
    useGraphHygiene,
  } from './lib/graph/graph.svelte'
  import {
    closeTaxonomy,
    openTaxonomy,
    selectTaxonomyField,
    taxonomy,
    taxonomyAddEdge,
    taxonomyAttachDocument,
    taxonomyDetachDocument,
    taxonomyRemoveEdge,
  } from './lib/library/taxonomy.svelte'
  import { closeRerun, rerun } from './lib/library/reingest.svelte'
  import {
    closeDeleteDocument,
    closeEditMetadata,
    library,
    refreshDocuments,
    resetMetadata,
    restoreDocuments,
    saveMetadata,
  } from './lib/library/documents.svelte'
  import {
    addDocsToFolder,
    closeManageFolders,
    createLibraryFolder,
    deleteLibraryFolder,
    folders,
    refreshFolders,
    removeDocFromFolder,
    renameLibraryFolder,
    restoreFolders,
  } from './lib/library/folders.svelte'
  import {
    addFamilyMemberKeyword,
    closeManageKeywords,
    createFamily,
    deleteFamily,
    dismissProposal,
    keywords,
    previewFamilyDeletion,
    removeFamilyMemberKeyword,
    renameFamily,
    runDetectFamilies,
    setFamilyOnGraph,
    setFormBreadth,
  } from './lib/library/keywords.svelte'
  import { libPrefs } from './lib/library/prefs.svelte'
  import { sidebarPrefs, startSidebarResize } from './lib/shell/prefs.svelte'
  // Shell chrome state (leaf module — imports no sibling state). Phase 2 lets the pane components
  // import this directly instead of taking ~20 props each.
  import { shell } from './lib/shell/shell.svelte'
  import {
    accept,
    clearPending,
    loadAccepts,
    watchDrops,
  } from './lib/library/accept.svelte'
  import { formatLine, limitLine } from './lib/library/formats'
  import { previewNames, remainderLabel, summarise } from './lib/library/accept'
  import { ingestRun, watchIngest } from './lib/core/ingest.svelte'
  import Indexing from './shell/Indexing.svelte'
  import LibrarySwitchFailed from './shell/LibrarySwitchFailed.svelte'
  import { clearLibrarySwitch, pendingLibrarySwitch } from './lib/core/tauri'
  import { backoffDelayMs, startupPhase } from './lib/shell/startup'
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
    archiveConversations,
    conversations,
    pinConversation,
    refreshConversations,
    renameConversation,
    restoreConversations,
  } from './lib/chat/conversations.svelte'

  // The Test-override button only exists while a retrieval-affecting override is set — with none,
  // both sides retrieve identically and the button is dead weight (2026-07-13 UX review). Settings
  // writes these fields only when touched; Reset returns chat.overrides to {}.
  const hasRetrievalOverride = $derived(
    chat.overrides.top_k != null || chat.overrides.use_multi_query != null,
  )

  // Conversation history (feature-conversation-history.md). `viewing` is the session_id shown as a
  // read-only transcript; `null` means the live chat (composer + claims bound to `chat.sessionId`).
  // ADR-061 choice 7: the library a switch just asked for ('' = the app's own), read once across the
  // reload the switch needs; cleared when the new backend answers.
  let switchAsked = $state<string | null>(pendingLibrarySwitch())
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
  // The document list, its dialogs and the select mode are `lib/library/documents.svelte.ts`.
  let libraryCollection = $state<LibraryCollection>({ kind: 'all' })
  let libraryDocId = $state<string | null>(null)
  let libraryQuery = $state('')
  // Selected keyword facets (AND). Orthogonal to the collection — session-scoped, non-persistent
  // (a filter, not a preference), resets on reload like the search query. `keywordFilterOpen` toggles
  // the two-pane picker overlay; the inline strip shows only the selected keywords + the trigger.
  let libraryKeywords = $state<string[]>([])
  let keywordFilterOpen = $state(false)
  // Families need their own latch, not `library.loaded`. The graph path loads documents (the ego
  // panel resolves doc_ids → titles) without the family list, so once it has run, the library's
  // `!library.loaded` guard skips the family fetch forever — and Manage keywords opens listing
  // zero families on a library with 357. Found 2026-08-30 by making that view the graph empty
  // state's primary action; the graph's per-concept "Edit" has had the same hole since it shipped.
  let familiesLoaded = false
  // The overlay's results, derived from the live chat + document lists (both already client-side).
  const searchResults = $derived(searchEverything(shell.searchQuery, conversations.list, library.list))

  // The last session's library, folders, chat list and graph, shown at launch until the engine
  // answers (lib/core/lastSession.ts): the packaged backend takes tens of seconds to start, and
  // the window used to sit empty meanwhile. `library.loaded` stays false, so every list is
  // still fetched for real; the readiness gate refreshes them all once the engine is up.
  const restoredDocuments = restoreDocuments()
  const restoredFolders = restoreFolders()
  const restoredChats = restoreConversations()
  const restoredGraph = restoreGraph()
  let lastSession = $state(
    restoredDocuments || restoredFolders || restoredChats || restoredGraph,
  )
  // ADR-025 F2 — the chat retrieval scope. Sticky across chat.turns, in memory ONLY: a reload
  // returns to the whole library. Persisting it is the rejected option — a scope you forgot
  // you set silently narrows every future answer, which is the exact lie the integrity layer
  // exists to prevent. Deliberately separate from `libraryCollection`: filtering the Library
  // grid and scoping a conversation are two different intentions.
  let chatScopeFolderId = $state<string | null>(null)

  // Concept graph — state + its loader/rebuild live in `lib/graph/graph.svelte.ts`; lazy-loaded on
  // first entry to Graph mode (see selectMode). Only the cross-domain bits stay here.
  function selectGraphConcept(id: string): void {
    graph.selectedId = id
    graph.offGraph = null
    shell.sidebarOpen = false // mobile drawer: selecting navigates, like selectCollection
  }
  // A concept the graph does not map, reached by the index's vocabulary search (ADR-053): the
  // panel shows its definition without a neighbourhood.
  function selectOffGraphConcept(id: string, label: string): void {
    graph.selectedId = null
    graph.offGraph = { id, label }
    shell.sidebarOpen = false
  }
  useGraphHygiene() // intra-domain: a rebuild can drop the selected concept
  // Chat transcript autoscroll. `viewing` is passed as a getter because it is conversation-view
  // state owned here, and the effect must re-run when it changes (opening a past chat scrolls
  // to the bottom too).
  useChatAutoscroll(() => viewing)

  // Taxonomy view (docs/specs/feature-taxonomy-view.md, ADR-028 2b). A dedicated modal that renders
  // the curated field forest + *places* concepts/documents onto it. App owns the data; Taxonomy
  // is a dumb renderer. Opened from the Library rail, or from a graph node's Place action (which
  // preselects that concept via `taxonomy.focusConceptId`). Decoupled from the top-level nav — it's a
  // global overlay like Settings/Search, so it opens from any mode.

  // Deep-link from a graph node to curate its concept (ADR-017 A1 — the graph never writes the
  // vocabulary; the Manage-keywords view owns every edit). Switches to Library and opens the view
  // on that row: with terms listed too (ADR-054) the view is a few hundred rows long, and "take
  // this term on" that lands at the top of them leaves the user to find the term again.
  function manageConcept(conceptId: string, _label: string): void {
    keywords.focusId = conceptId
    curateVocabulary()
  }

  // The graph's empty state sends people here (ADR-018's curation lives in the keywords view, and
  // ADR-017 A1 keeps the graph read-only over the vocabulary). Same destination as the per-concept
  // "Edit" above, which is why that one now delegates — one door, one implementation.
  // Reached from the Graph tab only, so the view opens on its Concepts section (a row to focus on
  // wins over that: `manageConcept` above sets one first).
  //
  // The view is an overlay, and it opens over the Graph: it used to switch to the Library first,
  // only because entering the Library is what loads the two lists it shows — so closing it left
  // the user on a tab they had not asked for. The lists are loaded here instead.
  function curateVocabulary(): void {
    if (!library.loaded) void refreshDocuments()
    if (!familiesLoaded) void refreshFamilies()
    keywords.onConcepts = true
    keywords.manageOpen = true
  }

  // Cross-domain wrapper over the taxonomy module: the attach picker lists *documents*, which
  // lazy-load only on entering the Library, so a user who opens the modal from Chat or Graph would
  // otherwise get an empty picker. The document list is library state, so the pull stays here
  // rather than reaching across from `taxonomy.svelte.ts`.
  function openTaxonomyView(focusConceptId: string | null = null): void {
    if (!library.loaded) void refreshDocuments()
    void openTaxonomy(focusConceptId)
  }

  // Pipeline: active collection → search filter (Decision 5a) → keyword facets (AND) → sort.
  // Facets are orthogonal to the collection: switching collection keeps them, and the facet chips
  // grey out relative to the current searched pool. `facetList` drives the facet bar; the selected
  // keywords also drive the tile highlight + first-position ordering.
  // Family collapse (PR-1) sits ahead of the facet math: `keywordsOf` maps each doc's raw
  // keywords through the family canonical map (identity when no families exist — byte-identical
  // to pre-PR-1 behavior), then `keywordFacets`/`facetFilter` operate on those collapsed units.
  const familyCanonicalOf = $derived(familyCanonicalMap(keywords.families))
  const keywordsOf = $derived(familyUnitsOf(familyCanonicalOf))
  const folderNames = $derived(folderNameMap(folders.list))
  const collectionDocs = $derived(docsFor(library.list, libraryCollection, new Date()))
  const searchedDocs = $derived(filterDocs(collectionDocs, libraryQuery))
  const facetList = $derived(keywordFacets(searchedDocs, libraryKeywords, keywordsOf))
  // Documents per unit over the pre-facet pool — the rare-tail split (PR-2.7 F4) must not
  // shift as the user toggles keywords, and `KeywordFacet.count` is relative to the faceted
  // pool, so it cannot be reused for this.
  const facetDocCounts = $derived(unitDocCounts(searchedDocs, keywordsOf))
  // The Manage view's pool lists *raw* keyword names (a family's members), so its rare split
  // counts raw keywords over the whole library rather than family units over a collection.
  const rawKeywordDocCounts = $derived(unitDocCounts(library.list))
  const visibleDocs = $derived(
    sortDocs(facetFilter(searchedDocs, libraryKeywords, keywordsOf), libPrefs.sort),
  )
  // The corpus's full raw-keyword universe (unfiltered by collection/search), for the Manage view.
  const allKeywords = $derived.by(() => {
    const s = new Set<string>()
    for (const d of library.list) for (const k of d.keywords) s.add(k)
    return [...s]
  })
  // Breadcrumb label for the open document, from the cached list (the chunk view fetches its
  // own detail; a stale/missing entry just hides the crumb).
  const openDoc = $derived(
    libraryDocId ? (library.list.find((d) => d.id === libraryDocId) ?? null) : null,
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
  // AD1 — window-level drag-drop. Tauri intercepts the OS drag before the DOM sees it, so this
  // is the only route to real paths; in a browser `watchDrops` subscribes to nothing and the
  // returned teardown is a no-op, so no branch is needed here.
  // AD2 — the review sheet. Opens itself when files arrive: a user who just dropped a folder
  // has already asked the question, so making them find a button would be a second ask.
  //
  // The staged paths are ALSO what keep the sheet mounted, so clearing them belongs to
  // `onClose`, never to `onAdded`: clearing on success unmounted the sheet before the outcome
  // (and the Undo button) could be seen. Caught by the AD3 live test, invisible to svelte-check.
  let reviewingAdd = $state(false)
  $effect(() => {
    if (accept.pending.length > 0) reviewingAdd = true
  })

  $effect(() => watchDrops())

  // Ambient ingest watch. Runs for the app's whole life rather than being owned by whichever
  // screen started a run, because an ingest kicked off from the Manage files list — or from the CLI
  // against the same backend — is just as worth showing as one started from the add sheet.
  $effect(() => watchIngest())

  // When a run *finishes*, re-read what it changed. Everything that shows the corpus previously
  // refreshed on the 202 — i.e. before the run had indexed anything — so a just-added document was
  // absent from the grid and the chunk count was stale until something else happened to refresh
  // them. `lastSeenRun` is a plain `let`, not `$state`, so writing it here cannot re-trigger this
  // effect; only `completedRuns` is tracked.
  let lastSeenRun = 0
  $effect(() => {
    const n = ingestRun.completedRuns
    if (n <= lastSeenRun) return
    lastSeenRun = n
    void refreshDocuments()
    // The corpus size lives on `health`, and a first run can also finish the setup checklist.
    void refreshHealth()
  })

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

  // Bulk delete from the sidebar's select mode. Same soft delete, one transaction — and the
  // confirmation names the export, because "the records are still in the database" is not a
  // restore path the user can act on and a file is.
  let pendingDeleteConvIds = $state<string[]>([])
  async function confirmDeleteConversations(): Promise<void> {
    const ids = pendingDeleteConvIds
    if (ids.length === 0) return
    deleteConvBusy = true
    try {
      await bulkUpdateConversations(ids, true)
      if (
        (viewing !== null && ids.includes(viewing)) ||
        (resumedHistory !== null && ids.includes(resumedHistory.session_id)) ||
        ids.includes(chat.sessionId)
      ) {
        newConversation()
      }
      await refreshConversations()
      pendingDeleteConvIds = []
    } catch (e) {
      console.error('bulk delete failed', e)
    } finally {
      deleteConvBusy = false
    }
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

  // Readiness gate (PR-M4): the frozen sidecar unpacks ~1.5 GB and loads models before it accepts
  // requests. Poll /api/health until it answers — **without a deadline** (KI-39).
  //
  // This loop used to stop after 60 tries and set status 'down' for good, which made a slow first
  // launch indistinguishable from a broken install and unrecoverable without a relaunch. It now
  // backs off and keeps going; `startupPhase` only changes what the status bar SAYS. Timing and
  // thresholds are in the pure, tested `startup.ts`.
  $effect(() => {
    let cancelled = false
    const startedAt = Date.now()
    void (async () => {
      for (let attempt = 0; !cancelled; attempt++) {
        try {
          const h = await getHealth()
          if (cancelled) return
          shell.health = h
          shell.status = 'ready'
          shell.startupPhase = 'connecting' // reset, so a later drop-out starts its story afresh
          // A library switch that answered is done (ADR-061): no *Go back* for it any more.
          clearLibrarySwitch()
          switchAsked = null
          void refreshSetup() // ADR-034 — what this install still needs, if anything
          void loadAccepts() // formats + limits the uploader states (security S-1/S-2)
          void refreshConversations()
          // The composer's scope selector needs the folder list even if the user never
          // opens the Library.
          void refreshFolders()
          // Replace whatever the last session left on screen with the live lists.
          if (lastSession) {
            lastSession = false
            if (restoredDocuments) void refreshDocuments()
            if (restoredGraph || shell.mode === 'graph') void loadConceptGraph()
          }
          return
        } catch {
          if (cancelled) return
          const phase = startupPhase(Date.now() - startedAt)
          shell.startupPhase = phase
          // 'stalled' shows the fault colour, but the loop does NOT end — if the backend turns up
          // at minute three, the app goes ready by itself.
          shell.status = phase === 'stalled' ? 'down' : 'connecting'
          await new Promise((r) => setTimeout(r, backoffDelayMs(attempt)))
        }
      }
    })()
    return () => {
      cancelled = true
    }
  })

  // A scope naming a folder that no longer exists would silently become "search nothing" on the
  // next turn. Drop it the moment the folder leaves the list, so the selector can't lie.
  $effect(() => {
    if (chatScopeFolderId !== null && !folders.list.some((f) => f.id === chatScopeFolderId)) {
      chatScopeFolderId = null
    }
  })


  // Sample questions for the empty state — corpus-agnostic openers that run one-click on any
  // library. Picking one only prefills the existing composer (no turn sent, no new behavior); the
  // reader still presses Send. Kept generic because the corpus topics aren't known at this layer.
  const EMPTY_TURN =
    'No answer came back, and no reason with it. Settings → Getting started shows what the app still needs.'
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
        // The turn stopped: the API's sentence names the engine that failed.
        else if (ev.event === 'error') chat.turns[idx].error = ev.data
        // `step` events are advisory; ignored for now.
      }
      const t = chat.turns[idx]
      // A stream that ends with nothing said used to leave the question alone on screen.
      if (!t.answer && !t.result && !t.error) t.error = EMPTY_TURN
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

  // Tag families are a sidecar read, same inform-don't-block rule as documents. Glue: the list is
  // `keywords.svelte.ts`'s, the selection it re-points is the Library's facets (below).
  async function refreshFamilies(): Promise<void> {
    try {
      keywords.families = await listKeywordFamilies()
      familiesLoaded = true
      // PR-2.5 D5 — a family write changes what a facet *unit* is, so a live selection has to be
      // re-pointed or the grid silently empties behind a chip that still looks selectable. The
      // Manage view is opened from the overlay, i.e. exactly where a selection is live. Mapped
      // against the whole library, not the active collection, so an out-of-collection selection
      // survives (it must stay removable).
      libraryKeywords = remapSelection(
        libraryKeywords,
        familyCanonicalMap(keywords.families),
        library.list,
      )
    } catch {
      // keep the prior list
    }
  }

  // Glue over `folders.svelte.ts`: a deleted folder can't stay the Library's open collection
  // (navigation); the module then drops it as the Manage selection and writes.
  function leaveAndDeleteFolder(folderId: string): void {
    if (libraryCollection.kind === 'folder' && libraryCollection.value === folderId) {
      libraryCollection = { kind: 'all' }
    }
    deleteLibraryFolder(folderId)
  }

  // Switch between Chat and Library. Entering Library closes any open citation panel and lazy-loads
  // the document list once; the live chat's in-memory state is preserved across the switch.
  function selectMode(m: 'chat' | 'library' | 'graph'): void {
    // The Graph tab is hidden for 0.6 (core/features.ts). The tab is gone, but this function is
    // also the nav-history restore path (line ~373), so it is guarded here too: landing on a mode
    // with no way to leave it is the one failure a hidden tab can still cause.
    if (m === 'graph' && !GRAPH_TAB_ENABLED) m = 'chat'
    shell.mode = m
    shell.sidebarOpen = false
    chat.activeCitation = null
    if (m === 'chat' && folders.list.length === 0) void refreshFolders()
    if (m === 'library') {
      if (!library.loaded) void refreshDocuments()
      if (!familiesLoaded) void refreshFamilies()
      if (folders.list.length === 0) void refreshFolders()
    }
    if (m === 'graph') {
      // The ego panel resolves doc_ids → titles from the library list, so it must be loaded too.
      if (!library.loaded) void refreshDocuments()
      // Before the engine answers, a load would only fail; the readiness gate loads it then.
      if (!graphLoaded() && shell.status === 'ready') void loadConceptGraph()
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
    if (!library.loaded) void refreshDocuments()
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

  // A chat citation asking to be shown in its source (ROADMAP 18). The chat side knows only a
  // `chunk_key`; the server turns that into a document and a page, because the key's shape is a
  // contract and a second copy of it in the client would rot.
  let sourceJump = $state<{ docId: string; page: number | null; nonce: number } | null>(null)
  let sourceJumpNonce = 0

  async function openCitationInDocument(chunkKey: string): Promise<void> {
    const found = await locateChunk(chunkKey)
    // A chunk the store does not know cannot be shown anywhere; leaving the reader where they
    // are beats navigating them to an empty Library.
    if (!found) return
    chat.activeCitation = null
    if (shell.mode !== 'library') selectMode('library')
    openDocument(found.document_id)
    sourceJump = { docId: found.document_id, page: found.page, nonce: ++sourceJumpNonce }
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

  // The two dialogs a tile's ⋯ menu opens (`documents.svelte.ts` holds their state and Edit
  // metadata's writes); each mounts while its document is still in the list.
  const editingDoc = $derived(
    library.editingId ? (library.list.find((d) => d.id === library.editingId) ?? null) : null,
  )
  const deletingDoc = $derived(
    library.deletingId ? (library.list.find((d) => d.id === library.deletingId) ?? null) : null,
  )
  // Safe-delete (ADR-014), glue: on confirm the source file goes to the Recycle Bin and the doc
  // leaves the library + index. If the open doc was deleted, drop back to its collection grid
  // (navigation), then re-fetch the list and the engine's health.
  // `deleteFile` is the choice the dialog just collected, per deletion — never a stored
  // preference (ADR-046 §2 replaced an unconditional rule with a per-deletion question, and
  // remembering the answer would quietly restore the rule).
  async function confirmDelete(deleteFile: boolean): Promise<void> {
    if (library.deletingId === null) return
    library.deleteBusy = true
    try {
      await deleteDocument(library.deletingId, deleteFile)
      if (libraryDocId === library.deletingId) libraryDocId = null
      await refreshDocuments()
      void refreshHealth()
      library.deletingId = null
    } catch {
      // e.g. the file couldn't be moved to the Recycle Bin — leave the dialog open, doc intact
    }
    library.deleteBusy = false
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

  // Per-part re-ingest (ADR-048, ROADMAP 20/21): the dialog's state and its four doors are in
  // `lib/library/reingest.svelte.ts`. What runs after is glue (the document list and the engine's
  // health). A finished re-run can have changed the metadata, the chunk count, the figures or the
  // references — so the list is refetched wholesale rather than guessing which. The open
  // document's own blocks refetch from their `docId` when the list identity changes.
  async function afterReingest(): Promise<void> {
    await refreshDocuments()
    refreshHealth()
  }

  // Tag-family curation (feature-tag-families.md, PR-1), glue over `keywords.svelte.ts`: a saved
  // write refetches the families, which re-points the facet selection (`refreshFamilies`). A
  // concept's name, forms and graph flag are what the graph counts through (ADR-018, ADR-054), so
  // every write but a create also drops the graph's latch: the graph's staleness is derived from
  // the live vocabulary at read time, so the next open re-reads it and names the concept the
  // graph is now behind on. Dropping the latch instead of refetching: paying for a fetch the user
  // may never look at is the wrong default. A deleted row may have been a concept too.
  async function afterFamilyWrite(write: Promise<boolean>, graphToo: boolean): Promise<void> {
    if (!(await write)) return
    await refreshFamilies()
    if (graphToo) invalidateGraph()
  }
  // A form's mark (ADR-054) replaces its row in place; it changes what the next build counts and
  // nothing the loaded graph shows, so the latch is dropped like the writes above.
  async function setFamilyFormBreadth(
    familyId: string,
    keyword: string,
    breadth: MemberBreadth,
  ): Promise<boolean> {
    const ok = await setFormBreadth(familyId, keyword, breadth)
    if (ok) invalidateGraph()
    return ok
  }
  async function acceptProposal(p: KeywordFamilyProposal): Promise<void> {
    await afterFamilyWrite(createFamily(p.canonical, p.members), false)
    dismissProposal(p.canonical)
  }
  // A write in the view drops the graph's latch (`invalidateGraph`), and the graph is re-read on
  // entering its tab. Closed over the Graph there is no entering: re-read it now, so the notice
  // naming what changed is there when the view goes away.
  function leaveManageKeywords(): void {
    closeManageKeywords()
    if (shell.mode === 'graph' && !graphLoaded() && shell.status === 'ready') void loadConceptGraph()
  }
</script>

<svelte:window onkeydown={onGlobalKey} />

<!-- The chat-mode sidebar rail (Chat history), composed here like the graph rail below: the
     conversation list and its actions are App-owned, and the sidebar frame stays a dumb renderer. -->
{#snippet chatRail()}
  <ChatHistory
    conversations={conversations.list}
    conversationsLoaded={conversations.loaded}
    liveSessionId={chat.sessionId}
    viewingSessionId={viewing}
    onNew={newConversation}
    onSelect={openConversation}
    onPin={pinConversation}
    onArchive={archiveConversation}
    onDelete={deleteConversation}
    onRename={renameConversation}
    onDeleteMany={(ids) => (pendingDeleteConvIds = ids)}
    onArchiveMany={archiveConversations}
  />
{/snippet}

<!-- The graph-mode sidebar rail. App composes it (data + selection are App-owned) and hands it to
     Sidebar as a snippet, so Sidebar stays a dumb renderer without ~8 more graph props. -->
{#snippet graphRail()}
  <ConceptIndex
    nodes={graph.data?.nodes ?? []}
    gaps={graph.data?.gaps ?? []}
    selectedId={graph.selectedId}
    bind:showUnderConnected={graph.showUnderConnected}
    loading={graph.loading || (shell.status !== 'ready' && !graph.data)}
    built={graph.data !== null}
    graphError={graph.error}
    onSelectConcept={selectGraphConcept}
    offGraphId={graph.offGraph?.id ?? null}
    onSelectOffGraph={selectOffGraphConcept}
  />
{/snippet}

<div class="app" class:collapsed={sidebarPrefs.collapsed} style="--sidebar-width: {sidebarPrefs.width}px">
  <!-- AD1: the drop target is the whole window, because a user dragging a paper has no reason to
       aim. It is presentational only — `accept.dragging` is driven by tauri://drag-enter/leave and
       never appears in a browser, where those events do not exist. -->
  {#if accept.dragging}
    <div class="dropveil" aria-hidden="true">
      <div class="dropcard">
        <Icon name="plus" size={22} />
        <strong>Drop to add documents</strong>
        {#if accept.accepts}
          <span>{formatLine(accept.accepts.extensions)}</span>
          <span>{limitLine(accept.accepts)}</span>
        {/if}
      </div>
    </div>
  {/if}
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

  <!-- AD1 ends here on purpose: files are STAGED, never applied. The review sheet that turns
       these into per-file verdicts is AD2, and copy/register/index is AD3. Showing the count and
       the names (rather than a disabled "Continue") keeps this build node honest — it reports
       exactly what it can do. -->
  {#if accept.pending.length > 0}
    <div class="staged" role="status">
      <Icon name="file-text" size={15} />
      <strong>{summarise(accept.pending)}</strong>
      <span class="stagednames">
        {previewNames(accept.pending).join(' · ')}
        {#if remainderLabel(accept.pending)}<em>{remainderLabel(accept.pending)}</em>{/if}
      </span>
      <button class="stagedreview" onclick={() => (reviewingAdd = true)} type="button">Review</button>
      <button class="stagedclear" onclick={clearPending} type="button">Clear</button>
    </div>
  {/if}

  <!-- AD1 entry. Mounted at the root, not inside the Library pane, because the same dialog is
       reachable from the app menu — and it must survive a mode switch mid-choice. -->
  {#if accept.chooser}
    <AddDocumentsChooser />
  {/if}

  {#if reviewingAdd && accept.pending.length > 0}
    <AddDocumentsReview
      paths={accept.pending}
      referenceRoot={accept.referenceRoot}
      onClose={() => {
        reviewingAdd = false
        clearPending()
      }}
      onAdded={() => {
        void refreshDocuments()
        void refreshHealth()
      }}
    />
  {/if}

  <div class="below">
  <Sidebar
    mode={shell.mode}
    documents={library.list}
    folders={folders.list}
    {libraryCollection}
    bind:libraryQuery
    open={shell.sidebarOpen}
    {chatRail}
    {graphRail}
    onSelectCollection={selectCollection}
    onOpenTaxonomy={() => openTaxonomyView()}
    onClose={() => (shell.sidebarOpen = false)}
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
      {#if lastSession && shell.status !== 'ready'}
        <p class="last-session" role="status">
          Showing your library as it was last time while the engine starts. Changes work once it
          is ready.
        </p>
      {/if}
      {#if shell.mode === 'library'}
        <Library
          {sourceJump}
          documents={library.list}
          documentsLoaded={library.loaded}
          {visibleDocs}
          {facetList}
          {keywordsOf}
          {openDoc}
          {folderNames}
          {libraryCollection}
          {libraryDocId}
          {libraryQuery}
          {libraryKeywords}
          folders={folders.list}
          onLibraryBack={libraryBack}
          onOpenDocument={openDocument}
          onSearchAll={searchAll}
          onSelectCollection={selectCollection}
          onSetDocId={(id) => (libraryDocId = id)}
          onToggleKeywordFacet={toggleKeywordFacet}
          onClearKeywordFacets={clearKeywordFacets}
          onOpenKeywordFilter={() => (keywordFilterOpen = true)}
        />
      {:else if shell.mode === 'graph'}
        <Graph
          graph={graph.data}
          loading={graph.loading || (shell.status !== 'ready' && !graph.data)}
          error={graph.error}
          documents={library.list}
          rebuildState={graph.rebuildState}
          selectedId={graph.selectedId}
          showUnderConnected={graph.showUnderConnected}
          onRebuild={rebuildGraph}
          onOpenDocument={(id) => {
            selectMode('library')
            openDocument(id)
          }}
          onManageConcept={manageConcept}
          onCurateVocabulary={curateVocabulary}
          onPlaceConcept={(id) => openTaxonomyView(id)}
          onSelectConcept={selectGraphConcept}
          loadPresence={getConceptPresence}
          offGraphConcept={graph.offGraph}
          onOpenPassage={(key) => void openCitationInDocument(key)}
        />
      {:else}
        <Chat
          {viewing}
          {viewedConvo}
          {resumedHistory}
          folders={folders.list}
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

{#if ingestRun.panelOpen}
  <Indexing />
{/if}

{#if switchAsked !== null && shell.status !== 'ready' && shell.startupPhase === 'stalled'}
  <LibrarySwitchFailed asked={switchAsked} onDismiss={() => (switchAsked = null)} />
{/if}

{#if shell.showSettings}
  <Settings
    onClose={() => (shell.showSettings = false)}
    onCorpusChanged={refreshHealth}
    bind:overrides={chat.overrides}
  />
{/if}

{#if shell.showShortcuts}
  <KeyboardShortcuts onClose={() => (shell.showShortcuts = false)} />
{/if}

{#if shell.showAbout}
  <About
    chunks={shell.health?.chunk_count ?? null}
    model={shell.health?.model ?? null}
    embedding={shell.health?.embedding_model ?? null}
    version={shell.health?.version ?? null}
    onClose={() => (shell.showAbout = false)}
  />
{/if}

{#if chat.activeCitation && activeSource}
  <SourcePanel
    source={activeSource}
    onClose={() => (chat.activeCitation = null)}
    onOpenInDocument={openCitationInDocument}
  />
{/if}

{#if editingDoc}
  <EditMetadata
    doc={editingDoc}
    onSave={saveMetadata}
    onReset={resetMetadata}
    onClose={closeEditMetadata}
  />
{/if}

{#if deletingDoc}
  <DeleteDocument
    doc={deletingDoc}
    busy={library.deleteBusy}
    onConfirm={confirmDelete}
    onClose={closeDeleteDocument}
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

{#if pendingDeleteConvIds.length > 0}
  <ConfirmDialog
    title="Delete {pendingDeleteConvIds.length} conversation{pendingDeleteConvIds.length === 1
      ? ''
      : 's'}?"
    body="They are removed from your history; the underlying records are kept. To keep a readable copy, export your history from Settings first."
    confirmLabel="Delete {pendingDeleteConvIds.length}"
    busy={deleteConvBusy}
    onConfirm={confirmDeleteConversations}
    onClose={() => (pendingDeleteConvIds = [])}
  />
{/if}

{#if keywordFilterOpen}
  <FilterByKeyword
    facets={facetList}
    docCounts={facetDocCounts}
    previewDocs={visibleDocs}
    selectedCount={libraryKeywords.length}
    families={keywords.families}
    onToggle={toggleKeywordFacet}
    onClear={clearKeywordFacets}
    onClose={() => (keywordFilterOpen = false)}
  />
{/if}

{#if folders.manageOpen}
  <ManageFolders
    folders={folders.list}
    documents={library.list}
    selectedId={folders.manageId}
    initialDocQuery={folders.manageQuery}
    error={folders.error}
    onCreate={createLibraryFolder}
    onRename={renameLibraryFolder}
    onDelete={leaveAndDeleteFolder}
    onAddDocuments={addDocsToFolder}
    onRemoveDocument={removeDocFromFolder}
    onSelect={(id) => (folders.manageId = id)}
    onClose={closeManageFolders}
  />
{/if}

{#if rerun.targets !== null}
  <RerunIngestion
    documentIds={rerun.targets}
    label={rerun.label}
    preselected={rerun.parts}
    onClose={closeRerun}
    onFinished={() => void afterReingest()}
  />
{/if}

{#if keywords.manageOpen}
  <ManageKeywords
    families={keywords.families}
    focusId={keywords.focusId}
    openOnConcepts={keywords.onConcepts}
    {allKeywords}
    keywordDocCounts={rawKeywordDocCounts}
    proposals={keywords.proposals}
    detecting={keywords.detecting}
    detectError={keywords.detectError}
    onCreate={(canonical, members) => afterFamilyWrite(createFamily(canonical, members), false)}
    onRename={(id, canonical) => afterFamilyWrite(renameFamily(id, canonical), true)}
    onSetOnGraph={(id, include) => afterFamilyWrite(setFamilyOnGraph(id, include), true)}
    onAddMember={(id, keyword) => afterFamilyWrite(addFamilyMemberKeyword(id, keyword), true)}
    onRemoveMember={(id, keyword) => afterFamilyWrite(removeFamilyMemberKeyword(id, keyword), true)}
    onSetBreadth={setFamilyFormBreadth}
    previewDelete={previewFamilyDeletion}
    onDelete={(id) => afterFamilyWrite(deleteFamily(id), true)}
    onDetect={runDetectFamilies}
    onAcceptProposal={acceptProposal}
    onDismissProposal={dismissProposal}
    onClose={leaveManageKeywords}
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
  <Taxonomy
    view={taxonomy.view}
    fieldDetail={taxonomy.fieldDetail}
    proposals={taxonomy.proposals}
    loading={taxonomy.loading}
    error={taxonomy.error}
    documents={library.list}
    concepts={taxonomy.concepts}
    focusConceptId={taxonomy.focusConceptId}
    onSelectField={selectTaxonomyField}
    onAddEdge={taxonomyAddEdge}
    onRemoveEdge={taxonomyRemoveEdge}
    onAttachDocument={taxonomyAttachDocument}
    onDetachDocument={taxonomyDetachDocument}
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
  /* ---- AD1 accept surface ---- */
  .dropveil {
    position: fixed;
    inset: 0;
    z-index: 60;
    display: flex;
    align-items: center;
    justify-content: center;
    background: color-mix(in srgb, var(--bg) 78%, transparent);
    pointer-events: none;
  }
  .dropcard {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
    padding: 2rem 3rem;
    border: 2px dashed var(--accent);
    border-radius: 12px;
    background: var(--bg);
    color: var(--accent);
    box-shadow: var(--shadow-2);
  }
  .dropcard span {
    font-size: var(--text-meta);
    color: var(--fg-2);
  }
  .staged {
    flex: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.7rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    font-size: var(--text-sm);
  }
  .stagednames {
    color: var(--fg-2);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .stagednames em {
    font-style: normal;
    opacity: 0.75;
  }
  .stagedreview {
    margin-left: auto;
    flex: none;
    background: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font: inherit;
    font-size: var(--text-meta);
    color: var(--accent-fg);
    cursor: pointer;
  }
  .stagedclear {
    flex: none;
    background: none;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font: inherit;
    font-size: var(--text-meta);
    color: var(--fg-2);
    cursor: pointer;
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
  /* The last session's lists on screen before the engine answers (lib/core/lastSession.ts). */
  .last-session {
    margin: 0 0 0.75rem;
    padding: 0.45rem 0.75rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--fg-2);
    font-size: 0.85rem;
  }
  main.wide {
    max-width: 1500px;
  }


</style>
