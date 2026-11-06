// Keyword families and the *Manage keywords* view (feature-tag-families.md, ADR-015, ADR-054;
// GLOSSARY C-003, U-001; ADR-064 choice 5): the family list, the family writes, the view's open
// flag and where it opens, and detection's proposals.
//
// A `.svelte.ts` rune module. Each write here is the API call alone and resolves to whether it
// was saved; what follows a saved write is glue and lives in App.svelte, because it spans
// domains: refetching the families also re-points the Library's live keyword selection (a
// family write changes what a facet *unit* is, PR-2.5 D5), and a change to what the concept
// graph counts drops the graph's latch. App's `refreshFamilies` writes `keywords.families` for
// the same reason, and holds the fetch-once latch. So are the Graph tab's two doors to the view
// (a concept's *Edit*, *Manage keywords* on the graph bar), which load the lists first, and
// closing the view over the Graph, which re-reads it.

import {
  addFamilyMember,
  createKeywordFamily,
  deleteKeywordFamily,
  describeFamilyDeletion,
  detectKeywordFamilies,
  removeFamilyMember,
  renameKeywordFamily,
  setFamilyGraphInclude,
  setFamilyMemberBreadth,
} from '../core/api'
import type {
  KeywordFamily,
  KeywordFamilyDeletion,
  KeywordFamilyProposal,
  MemberBreadth,
} from '../core/types'

export const keywords = $state({
  families: [] as KeywordFamily[],

  /** *Manage keywords* is open. */
  manageOpen: false,
  /** The row the view opens on (a concept's *Edit* on the graph); `null` = the top. */
  focusId: null as string | null,
  /** Open on the Concepts section (the Graph tab's doors), not the top. */
  onConcepts: false,

  /** Detection (PR-2): proposals live only while the view is open; nothing is written until one
   *  is accepted, so staleness is harmless. */
  proposals: [] as KeywordFamilyProposal[],
  detecting: false,
  detectError: null as string | null,
})

/** *Manage keywords…* in *Filter by keyword*. */
export function openManageKeywords(): void {
  keywords.manageOpen = true
}

export function closeManageKeywords(): void {
  keywords.manageOpen = false
  keywords.focusId = null
  keywords.onConcepts = false
  keywords.proposals = []
  keywords.detectError = null
}

// --- family writes: the call alone (see the header for what follows one)

async function saved(op: () => Promise<unknown>): Promise<boolean> {
  try {
    await op()
    return true
  } catch {
    // inform-don't-block: the prior list stays (the create form keeps its typed values for a
    // retry, a row its prior name, flag or membership)
    return false
  }
}

export function createFamily(canonical: string, members: string[]): Promise<boolean> {
  return saved(() => createKeywordFamily(canonical, members))
}
export function renameFamily(familyId: string, canonical: string): Promise<boolean> {
  return saved(() => renameKeywordFamily(familyId, canonical))
}
/** ADR-018's curation flag, from the view the ADR named as its home. The graph never writes its
 *  own vocabulary (ADR-017 A1) — this is the library side of that line, and the graph finds out
 *  by re-reading it (its "N concepts behind your vocabulary" banner) rather than by being told. */
export function setFamilyOnGraph(familyId: string, include: boolean): Promise<boolean> {
  return saved(() => setFamilyGraphInclude(familyId, include))
}
export function addFamilyMemberKeyword(familyId: string, keyword: string): Promise<boolean> {
  return saved(() => addFamilyMember(familyId, keyword))
}
export function removeFamilyMemberKeyword(familyId: string, keyword: string): Promise<boolean> {
  return saved(() => removeFamilyMember(familyId, keyword))
}
export function deleteFamily(familyId: string): Promise<boolean> {
  return saved(() => deleteKeywordFamily(familyId))
}

/** ADR-054: how one form of a concept counts. The route answers with the family as it now
 *  stands, and a mark moves no membership, so that one row is replaced in place: refetching every
 *  family after each click was a second request and a re-render of the whole view before the
 *  button could show what had been clicked. Resolves to whether the mark was saved — the row
 *  says so when it was not. */
export async function setFormBreadth(
  familyId: string,
  keyword: string,
  breadth: MemberBreadth,
): Promise<boolean> {
  try {
    const updated = await setFamilyMemberBreadth(familyId, keyword, breadth)
    keywords.families = keywords.families.map((f) => (f.id === updated.id ? updated : f))
    return true
  } catch {
    return false
  }
}

/** What a delete would remove, read before the row asks. `null` on a failure: the row then says
 *  it could not be read, and the choice stays the user's. */
export async function previewFamilyDeletion(familyId: string): Promise<KeywordFamilyDeletion | null> {
  try {
    return await describeFamilyDeletion(familyId)
  } catch {
    return null
  }
}

// --- detection (PR-2): a zero-LLM proposal pass, run on request (never automatically — the API
// call loads/runs the embedder, not something to fire on every Manage-view open).

export async function runDetectFamilies(): Promise<void> {
  keywords.detecting = true
  keywords.detectError = null
  try {
    keywords.proposals = await detectKeywordFamilies()
  } catch (e) {
    keywords.detectError = e instanceof Error ? e.message : 'Detection failed.'
  }
  keywords.detecting = false
}

export function dismissProposal(canonical: string): void {
  keywords.proposals = keywords.proposals.filter((p) => p.canonical !== canonical)
}
