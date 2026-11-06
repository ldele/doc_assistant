// The allow-lists of the style checks in ui-rules.test.ts (ADR-063, *Enforced by*). An entry is a
// file (its path from src/), how many times it breaks the check's rule, and why that is allowed.
//
// A ratchet entry holds the count measured on 2026-11-06 and the piece of ROADMAP 110 that brings
// it down; an exception holds a reason of its own, also written in a comment where it lives
// (ADR-063 choice 8). A count may only fall. The check fails when a file goes over its entry (a
// new break: fix it, do not raise the number) and when it goes under (lower the entry, or delete
// it at zero, so the list stays the truth). Check 1 is exceptions only since piece (b).
//
// Measured with the scanner in ui-rules.test.ts, which reads a tag past an arrow function's `=>`
// (ADR-063's regex counts of 2026-11-06 did not, and missed 47 button titles) and counts named
// colours too (`green`, `gray`).

export interface Allowed {
  count: number
  why: string
}

const SWEEP_HOVER = 'ratchet, 2026-11-06: ROADMAP 110 (c) writes these buttons their hover text'
const SWEEP_COLOURS = 'ratchet, 2026-11-06: ROADMAP 110 (d) moves these colours to tokens in app.css'

const MENU_ROW =
  "a menu row (`.menuitem.danger`), not a button bar: the WAI-ARIA menu pattern keeps its rows' look; red names the act and a confirm follows"
const GAP_TONE = "a gap's severity (GAP_META's `tone`) on a badge, a note or a node's dot, not a button"
const ICON_DANGER =
  'icon-only danger buttons in a dense row (`.iconbtn.danger`): red at rest, named by their aria-label'

/** Check 1: class names `.primary`, `.ghost`, `.danger` a component defines for itself (0 to 3
 *  per file). 28 files, 48 names on 2026-11-06; after piece (b), only these exceptions. */
export const LOCAL_BUTTON_CLASSES: Record<string, Allowed> = {
  'screens/chat/ChatHistory.svelte': { count: 1, why: MENU_ROW },
  'screens/library/DocumentGrid.svelte': { count: 1, why: MENU_ROW },
  'screens/graph/ConceptIndex.svelte': { count: 1, why: GAP_TONE },
  'screens/graph/ConceptPanel.svelte': { count: 1, why: GAP_TONE },
  'screens/graph/GapList.svelte': { count: 1, why: GAP_TONE },
  'screens/library/ManageFolders.svelte': { count: 1, why: ICON_DANGER },
  'screens/manage-keywords/FamilyRow.svelte': { count: 1, why: ICON_DANGER },
}

/** Check 2: `<button>` tags without `use:tooltip`. 237 of them in 59 files on
 *  2026-11-06 (the 73 that had a native `title` moved onto the tooltip the same day). */
export const BUTTONS_WITHOUT_HOVER_TEXT: Record<string, Allowed> = {
  'App.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/chat/Chat.svelte': { count: 7, why: SWEEP_HOVER },
  'screens/chat/ChatHistory.svelte': { count: 14, why: SWEEP_HOVER },
  'screens/chat/ChunkContextView.svelte': { count: 1, why: SWEEP_HOVER },
  'screens/chat/ClaimReview.svelte': { count: 5, why: SWEEP_HOVER },
  'screens/chat/CompareCard.svelte': { count: 1, why: SWEEP_HOVER },
  'screens/chat/SourceCard.svelte': { count: 1, why: SWEEP_HOVER },
  'screens/chat/SourcePanel.svelte': { count: 1, why: SWEEP_HOVER },
  'screens/document-page/Chunks.svelte': { count: 5, why: SWEEP_HOVER },
  'screens/document-page/Connections.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/document-page/EditSpacing.svelte': { count: 3, why: SWEEP_HOVER },
  'screens/document-page/FigureViewer.svelte': { count: 5, why: SWEEP_HOVER },
  'screens/document-page/LetterSpacing.svelte': { count: 7, why: SWEEP_HOVER },
  'screens/document-page/PageIndex.svelte': { count: 1, why: SWEEP_HOVER },
  'screens/document-page/RerunIngestion.svelte': { count: 3, why: SWEEP_HOVER },
  'screens/document-page/SourceViewer.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/graph/ConceptIndex.svelte': { count: 4, why: SWEEP_HOVER },
  'screens/graph/ConceptPanel.svelte': { count: 3, why: SWEEP_HOVER },
  'screens/graph/Definition.svelte': { count: 12, why: SWEEP_HOVER },
  'screens/graph/GapList.svelte': { count: 4, why: SWEEP_HOVER },
  'screens/graph/Graph.svelte': { count: 5, why: SWEEP_HOVER },
  'screens/library/AddDocumentsChooser.svelte': { count: 5, why: SWEEP_HOVER },
  'screens/library/AddDocumentsReview.svelte': { count: 6, why: SWEEP_HOVER },
  'screens/library/DeleteDocument.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/library/DocumentGrid.svelte': { count: 4, why: SWEEP_HOVER },
  'screens/library/EditMetadata.svelte': { count: 4, why: SWEEP_HOVER },
  'screens/library/FilterByKeyword.svelte': { count: 5, why: SWEEP_HOVER },
  'screens/library/FilterStrip.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/library/HeaderBar.svelte': { count: 9, why: SWEEP_HOVER },
  'screens/library/Library.svelte': { count: 3, why: SWEEP_HOVER },
  'screens/library/ManageFolders.svelte': { count: 9, why: SWEEP_HOVER },
  'screens/library/Summary.svelte': { count: 3, why: SWEEP_HOVER },
  'screens/manage-keywords/DetectProposals.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/manage-keywords/FamilyRow.svelte': { count: 6, why: SWEEP_HOVER },
  'screens/manage-keywords/ManageKeywords.svelte': { count: 1, why: SWEEP_HOVER },
  'screens/manage-keywords/NewFamily.svelte': { count: 4, why: SWEEP_HOVER },
  'screens/manage-keywords/Terms.svelte': { count: 1, why: SWEEP_HOVER },
  'screens/settings/Documents.svelte': { count: 6, why: SWEEP_HOVER },
  'screens/settings/General.svelte': { count: 5, why: SWEEP_HOVER },
  'screens/settings/GettingStarted.svelte': { count: 1, why: SWEEP_HOVER },
  'screens/settings/ManageFiles.svelte': { count: 3, why: SWEEP_HOVER },
  'screens/settings/OpenLibrary.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/settings/ProviderForm.svelte': { count: 11, why: SWEEP_HOVER },
  'screens/settings/ProviderList.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/settings/ReindexFolder.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/settings/Retrieval.svelte': { count: 3, why: SWEEP_HOVER },
  'screens/settings/Settings.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/settings/SpacingIgnoreList.svelte': { count: 5, why: SWEEP_HOVER },
  'screens/taxonomy/FieldDetail.svelte': { count: 6, why: SWEEP_HOVER },
  'screens/taxonomy/FieldForest.svelte': { count: 2, why: SWEEP_HOVER },
  'screens/taxonomy/Taxonomy.svelte': { count: 1, why: SWEEP_HOVER },
  'shell/About.svelte': { count: 1, why: SWEEP_HOVER },
  'shell/ConfirmDialog.svelte': { count: 2, why: SWEEP_HOVER },
  'shell/GlobalSearch.svelte': { count: 4, why: SWEEP_HOVER },
  'shell/Indexing.svelte': { count: 8, why: SWEEP_HOVER },
  'shell/KeyboardShortcuts.svelte': { count: 1, why: SWEEP_HOVER },
  'shell/LibrarySwitchFailed.svelte': { count: 2, why: SWEEP_HOVER },
  'shell/Sidebar.svelte': { count: 6, why: SWEEP_HOVER },
  'shell/Topbar.svelte': { count: 7, why: SWEEP_HOVER },
}

/** Check 3: literal colours (hex, rgb(), hsl(), a named colour) in a component's styles.
 *  49 in 20 files on 2026-11-06. */
export const LITERAL_COLOURS: Record<string, Allowed> = {
  'screens/chat/ChunkContextView.svelte': { count: 1, why: SWEEP_COLOURS },
  'screens/chat/SourcePanel.svelte': { count: 2, why: SWEEP_COLOURS },
  'screens/document-page/EditSpacing.svelte': { count: 1, why: SWEEP_COLOURS },
  'screens/document-page/FigureViewer.svelte': { count: 3, why: SWEEP_COLOURS },
  'screens/document-page/Figures.svelte': { count: 5, why: SWEEP_COLOURS },
  'screens/document-page/LetterSpacing.svelte': { count: 1, why: SWEEP_COLOURS },
  'screens/document-page/RerunIngestion.svelte': { count: 2, why: SWEEP_COLOURS },
  'screens/document-page/SourceViewer.svelte': { count: 1, why: SWEEP_COLOURS },
  'screens/library/AddDocumentsChooser.svelte': { count: 6, why: SWEEP_COLOURS },
  'screens/library/AddDocumentsReview.svelte': { count: 1, why: SWEEP_COLOURS },
  'screens/manage-keywords/DetectProposals.svelte': { count: 1, why: SWEEP_COLOURS },
  'screens/settings/Settings.svelte': { count: 2, why: SWEEP_COLOURS },
  'screens/taxonomy/FieldDetail.svelte': { count: 4, why: SWEEP_COLOURS },
  'screens/taxonomy/Taxonomy.svelte': { count: 1, why: SWEEP_COLOURS },
  'shell/Indexing.svelte': { count: 1, why: SWEEP_COLOURS },
  'shell/IngestChip.svelte': { count: 1, why: SWEEP_COLOURS },
  'shell/Sidebar.svelte': { count: 2, why: SWEEP_COLOURS },
  'shell/StatusBar.svelte': { count: 1, why: SWEEP_COLOURS },
}
