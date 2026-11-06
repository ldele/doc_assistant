// The allow-lists of the style checks in ui-rules.test.ts (ADR-063, *Enforced by*). An entry is a
// file (its path from src/), how many times it breaks the check's rule, and why that is allowed.
//
// A ratchet entry holds the count measured on 2026-11-06 and the piece of ROADMAP 110 that brings
// it down; an exception holds a reason of its own, also written in a comment where it lives
// (ADR-063 choice 8). A count may only fall. The check fails when a file goes over its entry (a
// new break: fix it, do not raise the number) and when it goes under (lower the entry, or delete
// it at zero, so the list stays the truth). Check 1 is exceptions only since piece (b), check 2
// is empty since piece (c), check 3 is empty and check 4 is the frame alone since piece (d).
//
// Measured with the scanner in ui-rules.test.ts, which reads a tag past an arrow function's `=>`
// (ADR-063's regex counts of 2026-11-06 did not, and missed 47 button titles) and counts named
// colours too (`green`, `gray`).

export interface Allowed {
  count: number
  why: string
}

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
 *  2026-11-06 (the 73 that had a native `title` moved onto the tooltip the same day); piece (c)
 *  wrote the other 236 their hover text the same day, so the list is empty. An exception would
 *  go here with its reason; none is expected. */
export const BUTTONS_WITHOUT_HOVER_TEXT: Record<string, Allowed> = {}

/** Check 3: literal colours (hex, rgb(), hsl(), a named colour) in a component's styles.
 *  49 in 20 files on 2026-11-06, 36 in 18 after piece (b); piece (d) moved the rest onto tokens in
 *  app.css the same day (`--scrim`, `--paper`, `--shadow-side`, and the fallbacks of tokens that
 *  already existed), so the list is empty. A component that needs a colour the tokens lack adds a
 *  token, in both themes. */
export const LITERAL_COLOURS: Record<string, Allowed> = {}

/** Check 4: elements that carry `aria-modal`, a modal dialog a component draws for itself. 20 on
 *  2026-11-06, each with its own backdrop (four looks of it) and its own Esc; piece (d) moved them
 *  all onto the frame the same day. */
export const MODAL_FRAMES: Record<string, Allowed> = {
  'shell/Dialog.svelte': {
    count: 1,
    why: 'the one dialog frame: every dialog renders it, so its backdrop, Esc and focus rules are written once',
  },
}

/** Check 5: rules a component writes that another component also writes, word for word (each copy
 *  past the first). One number, not a list: a new copy fails the check, so the second component to
 *  need a rule moves it to shared.css. 211 on 2026-11-06; 179 after piece (d) the same day, which
 *  moved the dialog frame's copies (backdrop, frame, title, head, buttons) and the plainest `.muted`. */
export const COPIED_RULES: Allowed = {
  count: 179,
  why: 'ratchet, 2026-11-06: the copies left after ROADMAP 110 (d); a change that touches a copy moves it to shared.css',
}
