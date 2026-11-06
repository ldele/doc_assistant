<script lang="ts">
  // Manage-keywords view (feature-tag-families.md, PR-1). Opened from the keyword-filter overlay's
  // "Manage keywords…" link (ADR-015 — curation is a dedicated view, distinct from filtering).
  // Curate families by hand: create one from un-familied keywords, rename its canonical label,
  // move keywords in/out (a keyword belongs to at most one family — adding it here moves it off
  // any other family), or delete it. Reuses the overlay's modal shell (scrim + centered dialog,
  // Esc-to-close). Dumb by design — `lib/library/keywords.svelte.ts` owns the families list and
  // the writes; App passes them in, and refreshes after a write (that also re-points the
  // Library's keyword filter, so it is App's glue).
  //
  // ADR-054: the list is two lists. A *concept* is a row the user has taken on (it is on the
  // concept graph); a *term* is a word the library uses that nobody has. A concept is shown under
  // its name, and each of its forms is marked exact or broad here. Deleting asks first and says
  // what goes with the row.
  //
  // The view's parts sit beside it, one file each (ADR-064): DetectProposals, NewFamily, Concepts,
  // Terms, and FamilyRow — one row, rendered by both lists. This file keeps the frame (scrim,
  // header, close, Escape) and what crosses the parts: the one search of both lists, going to a
  // row, the one rename and the one delete question at a time, and the marks being saved.
  import { untrack } from 'svelte'
  import type {
    KeywordFamily,
    KeywordFamilyDeletion,
    KeywordFamilyProposal,
    MemberBreadth,
  } from '../../lib/core/types'
  import {
    filterByQuery,
    graphVocabulary,
    markKey,
    memberBreadth,
    nextBreadth,
    splitInheritedFamilies,
    termFamilies,
    withoutKey,
  } from '../../lib/library/library'
  import Icon from '../../shell/Icon.svelte'
  import Dialog from '../../shell/Dialog.svelte'
  import Concepts from './Concepts.svelte'
  import DetectProposals from './DetectProposals.svelte'
  import FamilyRow from './FamilyRow.svelte'
  import NewFamily from './NewFamily.svelte'
  import Terms from './Terms.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    families,
    focusId = null,
    openOnConcepts = false,
    allKeywords,
    keywordDocCounts,
    proposals,
    detecting,
    detectError,
    onCreate,
    onRename,
    onSetOnGraph,
    onAddMember,
    onRemoveMember,
    onSetBreadth,
    previewDelete,
    onDelete,
    onDetect,
    onAcceptProposal,
    onDismissProposal,
    onClose,
  }: {
    families: KeywordFamily[]
    focusId?: string | null // the row to open on (a deep-link from the graph); null = the top
    openOnConcepts?: boolean // open on the Concepts section (the Graph tab's door), not the top
    allKeywords: string[] // every raw keyword name across the corpus
    keywordDocCounts: Map<string, number> // documents per raw keyword (PR-2.7 F4)
    proposals: KeywordFamilyProposal[] // zero-LLM detection results (PR-2); [] until Detect runs
    detecting: boolean
    detectError: string | null
    onCreate: (canonical: string, members: string[]) => void
    onRename: (familyId: string, canonical: string) => void
    onSetOnGraph: (familyId: string, include: boolean) => void
    onAddMember: (familyId: string, keyword: string) => void
    onRemoveMember: (familyId: string, keyword: string) => void
    // ADR-054: mark one form of a concept exact or broad; `null` clears the mark. Resolves to
    // whether the mark was saved, so the row can say when it was not.
    onSetBreadth: (familyId: string, keyword: string, breadth: MemberBreadth) => Promise<boolean>
    // What a delete would remove, asked before the confirmation is shown. `null` = could not load.
    previewDelete: (familyId: string) => Promise<KeywordFamilyDeletion | null>
    onDelete: (familyId: string) => void
    onDetect: () => void
    onAcceptProposal: (p: KeywordFamilyProposal) => void
    onDismissProposal: (canonical: string) => void
    onClose: () => void
  } = $props()

  // A keyword belongs to at most one family (ADR-015) — "un-familied" = not a canonical or
  // an alias of any family, matched case-insensitively (mirrors the backend's own matching).
  // New family picks its members from it, and every row's add field offers it.
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

  // ADR-054 — two lists. Concepts are the rows the user has taken on (the graph vocabulary,
  // ADR-018); every one is listed, whether or not it has member keywords or documents, because
  // hiding one would hide a meaning the user chose. Terms are the rest.
  let famQuery = $state('')
  const familyText = (f: KeywordFamily): string =>
    `${f.canonical} ${f.written ?? ''} ${f.aliases.join(' ')}`
  const onGraph = $derived(graphVocabulary(families))
  const conceptsShown = $derived(filterByQuery(onGraph, famQuery, familyText))

  // A term with no member keywords and no documents filters nothing today. Hidden by default,
  // never deleted — the search and the toggle both reach it.
  let showInherited = $state(false)
  const terms = $derived(termFamilies(families))
  const famSplit = $derived(splitInheritedFamilies(terms))
  const termsShown = $derived(
    filterByQuery(
      showInherited ? [...famSplit.real, ...famSplit.inherited] : famSplit.real,
      famQuery,
      familyText,
    ),
  )

  // What is typed in New family's name field. Kept here because going to a row clears it.
  let newCanonical = $state('')
  let highlightId = $state<string | null>(null)
  function goToFamily(f: KeywordFamily): void {
    newCanonical = ''
    famQuery = ''
    if (famSplit.inherited.some((i) => i.id === f.id)) showInherited = true
    highlightId = f.id
    queueMicrotask(() => {
      document.getElementById(`fam-${f.id}`)?.scrollIntoView({ block: 'center' })
    })
  }

  // A deep-link from the graph opens the view on one row: "take this term on" has to land on the
  // term, not at the top of a few hundred. Applied once, when the row is in the list — the list
  // can still be loading when the view opens.
  let focusApplied = false
  $effect(() => {
    if (focusApplied || focusId === null) return
    const target = families.find((f) => f.id === focusId)
    if (!target) return
    focusApplied = true
    untrack(() => goToFamily(target))
  })
  // The Graph tab's door lands on the Concepts section: from there the top of this view (detect
  // proposals, a new family) is the part that is not about the graph.
  // Applied once, when the rows are in: while the list is still loading the section has nothing
  // below it, and a scroll made then stops short.
  let conceptsShownOnce = false
  $effect(() => {
    if (conceptsShownOnce || !openOnConcepts || focusId !== null) return
    if (families.length === 0) return
    conceptsShownOnce = true
    queueMicrotask(() => {
      document.getElementById('mk-concepts')?.scrollIntoView({ block: 'start' })
    })
  })

  // A family created with no members starts as a term with 0 aliases / 0 docs, which is exactly
  // the shape the unused group hides — so creating one would look like it silently failed.
  // Reveal the group in that case (and only that case: with members it lands in the visible
  // list).
  function createFamily(canonical: string, members: string[]): void {
    if (members.length === 0) showInherited = true
    onCreate(canonical, members)
  }

  // One rename at a time, across both lists: the row that starts one takes it (bound into the row).
  let editingId = $state<string | null>(null)
  let editingValue = $state('')

  // One pending "add a form" text per family row (family id -> what is typed in its field). The
  // field takes any text and offers the unassigned keywords as you type, from ONE shared list.
  // Each row used to carry its own <select> of the whole pool: on a 104-document library that was
  // 254 rows by 1,229 options, 312,166 elements, and the view froze for seconds on opening — while
  // a form that is not an extracted keyword (`DPR`, a bare `contrastive`) could not be added at all.
  // Kept here, not in the row, so a search that hides the row keeps what was typed.
  let addSelection = $state<Record<string, string>>({})

  // A mark is set by toggling, so its control shows the mark the moment it is clicked and takes
  // no second click until the server has answered: a control that waited looked as if nothing had
  // happened, and the click that followed cleared what the first had set (seen 2026-10-09). A save
  // that fails is said on the form, since nothing else on the screen would show it.
  let savingMarks = $state<Record<string, MemberBreadth>>({})
  let failedMarks = $state<Record<string, MemberBreadth>>({})
  async function setMark(f: KeywordFamily, alias: string, clicked: 'exact' | 'broad'): Promise<void> {
    const key = markKey(f.id, alias)
    if (key in savingMarks) return
    const target = nextBreadth(memberBreadth(f, alias), clicked)
    savingMarks = { ...savingMarks, [key]: target }
    failedMarks = withoutKey(failedMarks, key)
    const saved = await onSetBreadth(f.id, alias, target)
    savingMarks = withoutKey(savingMarks, key)
    if (!saved) failedMarks = { ...failedMarks, [key]: target }
  }

  // ADR-054 — a delete asks first. Pressing the button reads what goes with the row from the
  // server and shows it on the row; nothing is removed until the second press. One row at a time:
  // asking about another row drops the first question.
  let deleteAsk = $state<{
    id: string
    loading: boolean
    info: KeywordFamilyDeletion | null
  } | null>(null)
  async function askDelete(f: KeywordFamily): Promise<void> {
    deleteAsk = { id: f.id, loading: true, info: null }
    const info = await previewDelete(f.id)
    // The answer for a row the user has since left is dropped rather than shown on another row.
    if (deleteAsk === null || deleteAsk.id !== f.id) return
    deleteAsk = { id: f.id, loading: false, info }
  }
  function confirmDelete(): void {
    if (deleteAsk === null) return
    const id = deleteAsk.id
    deleteAsk = null
    onDelete(id)
  }
  function keepFamily(): void {
    deleteAsk = null
  }
  // The other way out for a concept: it goes back to being a term and keeps everything it has.
  function takeOffGraphInstead(): void {
    if (deleteAsk === null) return
    const id = deleteAsk.id
    deleteAsk = null
    onSetOnGraph(id, false)
  }

  // Escape answers the open question before it closes the view.
  function onEscape(): void {
    if (deleteAsk !== null) keepFamily()
    else onClose()
  }
</script>

<Dialog
  label="Manage keyword families"
  width="48rem"
  height="min(86vh, 700px)"
  gap="var(--space-3)"
  {onClose}
  {onEscape}
>
  <div class="mhead">
    <h2 class="dialog-title">Manage keywords</h2>
    <button class="iconbtn" onclick={onClose} aria-label="Close" type="button" use:tooltip={'Close Manage keywords'}>
      <Icon name="x" size={16} />
    </button>
  </div>

  <div class="body">
    <DetectProposals
      {proposals}
      {detecting}
      {detectError}
      {onDetect}
      {onAcceptProposal}
      {onDismissProposal}
    />

    <NewFamily
      {families}
      {unfamilied}
      {keywordDocCounts}
      bind:newCanonical
      onGoToFamily={goToFamily}
      onCreate={createFamily}
    />

    <!-- One row, rendered by both lists below. Wired here once, so Concepts and Terms only place
         it; the row itself is FamilyRow. -->
    {#snippet familyRow(f: KeywordFamily)}
      <FamilyRow
        family={f}
        highlighted={highlightId === f.id}
        bind:editingId
        bind:editingValue
        bind:addValue={addSelection[f.id]}
        {deleteAsk}
        {savingMarks}
        {failedMarks}
        {onRename}
        {onSetOnGraph}
        {onAddMember}
        {onRemoveMember}
        onSetMark={setMark}
        onAskDelete={askDelete}
        onKeep={keepFamily}
        onTakeOffGraphInstead={takeOffGraphInstead}
        onConfirmDelete={confirmDelete}
      />
    {/snippet}

    <!-- The unassigned keywords, once, for every row's add field to offer as you type. -->
    <datalist id="mk-unassigned">
      {#each unfamilied as k (k)}
        <option value={k}></option>
      {/each}
    </datalist>

    <Concepts concepts={onGraph} {conceptsShown} bind:famQuery {familyRow} />

    <Terms {terms} {famSplit} {termsShown} {famQuery} bind:showInherited {familyRow} />
  </div>
</Dialog>

<style>
  .mhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
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
  .body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
</style>
