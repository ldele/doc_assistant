<!-- status: active · updated: 2026-11-06 (As built: ROADMAP 110 piece (d), one dialog frame, no literal colours, the shared stylesheet; piece (c), hover text on every button and the 50 other titles sorted; piece (b), the roles across the app, the five confirms and two more; piece (a), the fill pair, the tooltip and the checks, with the counts corrected; accepted as proposed: all eight choices; revised while proposed: choices 6 to 8 and three checks that fail, from the user's words of 2026-11-06; 2026-11-05: proposed, the two select bars built to it) · class: append-only -->

# ADR-063 — A button's look says what it does: four action roles, and a delete always asks first

- **Status:** proposed (Claude Code, 2026-11-05); **accepted as proposed by the user on
  2026-11-06** (*"Accept as proposed"*): option A with all eight choices as recommended. Built for
  the two select bars only; the rest is ROADMAP 110. The request is the user's, 2026-11-05, in one
  batch of UI fixes: a selection's *Clear* was unclear (*"if it deletes, call it Delete, show it in
  red and ask for confirmation"*), *"make this a design rule for the whole app: destructive actions
  use a danger colour and a confirm step, and the other kinds of action get suitable colours too"*,
  *Done* should read as finishing the task, and *"buttons that do the same thing look the same
  everywhere"*, in the Graph tab's style. Built so far: the rule's classes and the two select bars
  (ROADMAP 78 and 80). The rest of the app moves over in ROADMAP **110**.
- **Revised 2026-11-06, still proposed:** choices 6 to 8 and the checks in *Enforced by*. The user,
  2026-11-06: *"Buttons should share design. We should have a short explanation of what they do
  when you mouse hover. They should have the relevant color when needed (red to delete -> meaning
  dangerous, + confirm delete or action)"*, *"+ Other Industry-wide conventions and design"*,
  *"Make sure of Styling (css, other) conventions across project (We may want to have some
  exceptions to this but generally we don't)"* and *"We should be indeed stricter here in the
  project itself"*.
- **Date:** 2026-11-05; revised 2026-11-06
- **Deciders:** user + Claude Code
- **Extends** `docs/specs/feature-visual-identity.md` (the palette and its tokens; it set colours,
  not what a button's colour means) and the ConfirmDialog convention in
  `apps/desktop/src/shell/ConfirmDialog.svelte` (a confirm gates a destructive act). Works with
  [ADR-064](ADR-064-the-frontends-folders-follow-the-screens.md) (the frontend's folders), whose
  file names this ADR uses from 2026-11-06 on.
- **In one sentence:** every button in the app plays one of four roles (*ordinary*, *primary*,
  *quiet*, *danger*), each role has one look defined once in `app.css`, and a danger button names
  what it removes, is red, and always asks first; every button also says what it will do when
  hovered or focused, styles live in one place, and checks fail when a component breaks a rule.

## To decide — eight choices, each with a recommendation

1. **The roles and their looks.** *Recommended:* four roles, one class each in `app.css`:
   - **ordinary** (`btn`): a filled box (`--surface`), a border, full-strength text (`--fg`). This
     is the Graph tab's button (`screens/graph/Graph.svelte` `.ghost`), the one the user picked as the
     clearest. Most buttons are this.
   - **primary** (`btn btn-primary`): the accent fill. The one action a screen or a mode is for:
     *Send*, *Add documents*, *Save* in a dialog, *Done*. At most one per bar.
   - **quiet** (`btn btn-quiet`): no box until hovered, secondary text (`--fg-2`). It changes only
     what is shown or selected and never changes the library: *Select all*, *Select none*,
     *Reset view*.
   - **danger** (`btn btn-danger`): red text and a red border (`--danger`). It removes something.
     The label names the act (*Delete*, *Remove*), never *Clear*, and ends in "…" because a
     confirm follows. The confirm names what goes and how many, and its own button is filled red
     (`ConfirmDialog`'s `tone="danger"`, as today).

   Navigation is not an action and keeps its own look: tabs, the document page's index (*Metadata*,
   *Connections*, …), view toggles. Choice 5 is where that line bites.
2. **"The Graph's style" for the primary, in the dark theme.** The user runs the dark theme (the
   app follows Windows, which is dark). There, the Graph's buttons are the *ordinary* look above, a
   dark box with near-white text, while the app's primary today is a light lavender fill with
   *dark* text (`--accent-fg: #1b1813`), so no primary has white text in the dark theme.
   *Recommended:* both. The ordinary look is the Graph's, everywhere (the Library's *Select*, sort
   and view buttons drop their grey text). The primary gets white text in both themes, from a deeper
   accent fill in the dark theme (about `#6b5fd6`: white on it is 4.9:1, where white on today's
   `#9a8ff0` is 2.8:1, below the 4.5:1 reading floor). A new token pair, `--accent-fill` and
   `--accent-fill-fg`, so the accent used for links and rings keeps its brightness.
3. **Done.** *Recommended:* *Done* ends a mode, so it is that mode's primary, at the trailing end of
   its bar, set apart from the actions by the free space between them. Both select bars have one
   (the chat rail had none: the way out was the toggle in its toolbar).
4. **Which actions confirm.** *Recommended:* every danger action, with no exception for the cheap
   ones, because the user asked for exactly that and a rule with exceptions has to be remembered
   per button. An action whose result the same screen undoes in one click (dismissing a chosen
   definition, ADR-053) still confirms. If that friction shows in use, the next ADR replaces its
   confirm with an undo; this one does not guess.
5. **The document page's *Source* and *Re-run…*.** They sit in the same row and the same pill as
   the page's index (*Metadata*, *Connections*, *Chunks*, *Figures*, *References*), which scrolls
   the page, while these two open a viewer and a dialog. *Recommended:* the index stays pills;
   *Source* and *Re-run…* become ordinary buttons (`btn`), at the row's trailing end.
6. **Hover text on every button** (added 2026-11-06). *Recommended:* every button has hover text
   that says what it does. It completes the sentence "this will …" and never repeats the label:
   *Done* → "Leave select mode"; *Delete…* → "Delete the 4 selected conversations, after you
   confirm". It shows on keyboard focus as well as on hover, and it meets WCAG 2.2's rule for
   content shown on hover or focus (SC 1.4.13): *Esc* dismisses it, the pointer can move onto it,
   and it stays until then. The native `title` attribute does none of the three, so one shared
   tooltip (a Svelte action, applied the same way everywhere) replaces it. Nothing the user must
   know lives only in hover text: a warning or a consequence is on the screen or in the confirm.
   An icon-only button also carries an `aria-label` that names it; the hover text explains it.
7. **Industry conventions, and which one wins** (added 2026-11-06). *Recommended:* the standards
   decide what is correct, and published design systems explain why and settle what the standards
   leave open. The standards are WCAG 2.2 level AA (contrast 4.5:1 for text and 3:1 for the parts
   of a control that identify it, a visible focus that nothing covers, targets at least 24 by 24
   CSS pixels, every action reachable from the keyboard) and the WAI-ARIA Authoring Practices
   patterns. Where a pattern applies, the code names it in a comment and follows its keys: a
   confirm before a destructive act is the *alert dialog* pattern, a select bar is a *toolbar*, a
   list with a select mode follows the *listbox* pattern's multi-select model. That last one is
   only half built: Shift-click and Ctrl-click work (ROADMAP 78, 80), Shift+Arrow and Ctrl+A do
   not, because the rows are toggle buttons today.
8. **Styling in one place** (added 2026-11-06). *Recommended:*
   - tokens (colour, spacing, type, radius, shadow) and the role classes live in `app.css`, and a
     component uses tokens only: no literal colour in a `.svelte` file;
   - small rules that repeat across parts (a `.hint` line, a section `h3`, ADR-064's list) go in
     one shared stylesheet beside `app.css`, loaded once;
   - the dialog frame that 20 files draw for themselves (their own `.scrim`, 14 also a `.modal`)
     becomes one shell component, because it is markup and *Esc* handling as well as style;
   - a component's own `<style>` is for its own layout;
   - an exception says why, in a comment where it lives, and is listed in the allow-list of the
     check it would fail (*Enforced by*). The user expects few: *"generally we don't"*.

## In practice — the two select bars on 2026-11-05

Read in the running app on the user's library (105 documents, the installed 0.7.2 engine), before
the change:

| Bar | Buttons, left to right | What each did |
|---|---|---|
| Library, select mode | `0 selected` · *Add to folder…* · *Re-run…* · *Clear* · *Done* | *Clear* emptied the selection (2 selected → 0) and nothing else; *Done* left the mode. All four in one look, *Done* touching *Clear*. |
| Chat rail, select mode | *All* / *Clear* · `0 selected` · *Archive* · *Delete* | *All*/*Clear* toggled every shown chat. *Delete* was red, from a fallback colour (`--danger-fg`, which no theme defines), and opened a confirm, but left select mode **before** the answer, so *Cancel* lost the selection. No *Done*. |

The user read *Clear* as a possible delete. It never was one, but a button that might remove
things and might not is the failure this rule is for. After the change (both bars built to
choices 1 and 3):

| Bar | Buttons |
|---|---|
| Library | `6 selected` · *Select none* (quiet) · │ · *Add to folder…* · *Re-run…* · · · *Done* (primary, trailing) |
| Chat rail | line 1: *Select none* (quiet) · · · *Done* (primary); line 2: `4 selected` · *Archive* · *Delete…* (danger) |

Driven the same day: a Ctrl-click on a tile outside select mode started it with that tile; a
Shift-click on the sixth tile then selected tiles 1 to 6 in reading order; *Select none* → 0,
*Select all* → 105; *Done* left the mode. In the rail, Ctrl-click then Shift-click selected 4
chats; *Delete…* asked *"Delete 4 conversations?"*; *Cancel* kept all 4 selected. A confirmed
delete was driven against an in-page stand-in for the conversation routes (no request reached the
engine): one bulk request carrying the 3 selected ids, the rows left the list, the mode closed.

**The rest of the app, measured the same day** (what ROADMAP 110 moves over; the components
were renamed on 2026-11-06 by ADR-064, and the new names follow in brackets):

- 16 components each define their own `.primary`, 16 their own `.ghost`, 8 their own `.danger`, in
  at least four sizes and three corner radii.
- Five destructive actions act at once, with no confirm and no red: *Remove key* and *Remove
  server* (`ProviderSetup.svelte` [`screens/settings/ProviderForm.svelte`]), *Remove* and *Remove
  all* on the letter-spacing ignore list (`SpacingIgnoreList.svelte`
  [`screens/settings/SpacingIgnoreList.svelte`]), and *Clear* on a chosen definition
  (`ConceptDefinition.svelte` [`screens/graph/Definition.svelte`]). *Clear* there is a dismissal,
  the very word this ADR retires for a removal.
- The four confirms that exist (`ConfirmDialog`: delete one chat, delete several, stop indexing,
  switch library) and the document delete (`LibraryDeleteConfirm`
  [`screens/library/DeleteDocument.svelte`]) already name what goes; they keep their dialogs.

**Measured 2026-11-06, for choices 6 to 8** (a regex over `apps/desktop/src/**/*.svelte`, after
ADR-064's pass 2a):

- 312 `<button>` tags, 26 of them with a `title`: 286 buttons say nothing on hover, and none says
  anything on keyboard focus. The most, per file: `screens/graph/Definition.svelte` and
  `screens/chat/ChatHistory.svelte` 16 each, `screens/library/HeaderBar.svelte` 14,
  `screens/manage-keywords/FamilyRow.svelte` 12, `shell/Topbar.svelte` and
  `screens/settings/ProviderForm.svelte` 11 each.
- 45 literal colours (a hex value, `rgb(…)` or `hsl(…)`) in the `<style>` blocks of 20 components.
- 20 components draw their own dialog backdrop (`.scrim`), 14 of them their own frame (`.modal`).

## Context

The palette has had a `--danger` token since the visual identity pass, and a confirm dialog since
the chat delete. What it never had is a statement of what a colour *means* on a button. Each
component chose: the Library's primary was an accent fill, its secondary a surface box with grey
text, the Graph's secondary a surface box with full-strength text, the chat rail's a transparent
pill. So the same act looked different in two places, and two different acts looked the same in
one place (*Clear* and *Done* beside *Add to folder…*). The user's reading of *Clear* as a delete
is the cost.

Svelte scopes each component's styles, so a global class only wins where the component has no
class of its own. That is why the new classes are prefixed (`btn-…`): they cannot collide with a
component's `.primary` while the migration is under way, and a component that has moved over
deletes its own.

## Options

- **A. Four roles defined once, every button on one of them** (recommended). One place to change
  a look; a reviewer can tell a button's kind from its class alone.
- **B. Fix the cases the user named and leave the rest.** Cheapest, and the reason the app has 16
  primaries: each fix is local and the next screen invents its own.
- **C. A component library (`<Button role="danger">`).** The same rule with a wrapper; it buys
  nothing a class does not, and every `.svelte` file would change its markup, not only its class.

## Decision

Option A, with the eight choices above as recommended: accepted as proposed by the user on
2026-11-06. Built on 2026-11-05: the four
classes in `apps/desktop/src/app.css` and both select bars. The `--accent-fill` pair (choice 2) is
not built yet: it is decided, and until ROADMAP 110 adds it `btn-primary` uses `--accent` and
`--accent-fg` as every primary does today. Choices 6 to 8 and the checks are not built; they are
ROADMAP 110, which starts once ADR-064's work in the same components is committed (its pass 2
is, in `b57601b`).

## Consequences

- A new button picks a role, not a colour. The GLOSSARY names the roles (U-003), so a spec or a
  review can say "a quiet button" and mean one thing.
- *Clear* is retired as the label of anything that removes. It stays for emptying a text field (the
  × in a filter box) and for a filter (*Clear keyword filters*), which change only what is shown.
- Every danger action gains a confirm (choice 4). Five do not have one today (*In practice*).
- The migration (ROADMAP 110) is a sweep over 16 components. Until it lands, two looks coexist:
  the new classes on the select bars, the old local ones elsewhere.
- Range selection (Shift-click, Ctrl/Cmd-click) is one function for every list with a select mode
  (`apps/desktop/src/lib/shell/selection.ts`); the next list with one reuses it.
- Every new button needs hover text, and writing it is part of adding the button: if the sentence
  "this will …" is hard to finish, the button's job is unclear (choice 6).
- A rule that repeats moves to the shared stylesheet the second time it is written, not the
  twentieth (choice 8). A component that needs a colour the tokens lack adds a token, in both
  themes.
- The checks make the rules cost something to break: an exception is a line in an allow-list with
  its reason, read in review, not a silent local override.

## Enforced by

- `apps/desktop/src/app.css` (the classes) and `GLOSSARY.md` U-003 (the names).
- `apps/desktop/src/lib/shell/selection.test.ts` (range and select-all semantics).
- Three checks that fail, as `node:test` files run by `npm test` (and so by CI's frontend job), each
  with an allow-list whose every entry gives its reason:
  1. a `.svelte` file defines its own `.primary`, `.ghost` or `.danger` button class (16 files on
     2026-11-05);
  2. a `<button>` has no hover text (286 of 312 on 2026-11-06);
  3. a literal colour appears in a component's styles (45 in 20 files on 2026-11-06).

  *Recommended order:* each check lands first as a ratchet that fails when its count rises above
  the measured one, so nothing new breaks a rule from the day it lands; ROADMAP 110's sweep then
  brings each count to its allow-list, and the ratchet becomes the list.

## Confidence

High on the roles: they are the four kinds of button the app already has, named. Medium on choice
2: "the Graph's style" was read from the user's words in a restatement, and the dark-theme fill is
a contrast calculation, not yet a look the user has seen. High on choices 6 to 8: they restate the
user's words of 2026-11-06 and the standards they point to; the counts behind them are from a
regex, so a tag split across lines in an unusual way may be miscounted by one or two.

## As built — ROADMAP 110 piece (a), 2026-11-06

ROADMAP 110 is cut in four pieces, one per session (the user picked the order on 2026-11-06):
(a) the fill pair, the tooltip and the checks; (b) the button roles, the five confirms, *Source*
and *Re-run…*; (c) hover text on the rest of the buttons; (d) the dialog frame, the shared
stylesheet and the colours. Piece (a) is built:

- **Choice 2.** `--accent-fill` / `--accent-fill-fg` in all four palette blocks of `app.css`:
  `#4a3fa6` in the light theme (white on it 8.2:1), `#6b5fd6` in the dark (5.0:1); `btn-primary`
  uses them. Its hover now mixes the fill toward black: toward `--fg`, as before, it lightened in
  the dark theme and dropped white text to 3.9:1 (now 6.4:1 dark, 9.9:1 light).
- **Choice 6.** One tooltip, `apps/desktop/src/lib/shell/tooltip.ts`, a Svelte action
  (`use:tooltip={text}`) over one shared `role="tooltip"` element styled in `app.css` (inverse ink,
  16:1 light, 14:1 dark). Driven in the running app: Esc hides it without moving the pointer or
  the focus, the pointer can cross onto it, and it stays until both leave; the button's
  `aria-describedby` points at it while it shows; it works on a disabled button, so a reason
  still shows; it stays inside the window at 375 px. Three calls the ADR left open: focus shows
  it only when a navigation key (Tab, an arrow, Home, End) moved the focus, so a click or a
  dialog's scripted focus does not pop it up; the first Esc dismisses the tooltip alone (it is
  the top layer, as in the WAI-ARIA pattern) and the next reaches the dialog underneath; touch
  shows nothing, and pressing the button hides it.
- **Every native `title` on a button moved onto it: 73, not 26.** About 30 texts only repeated
  their label (*Back*, *Zoom in*, *Settings*, *More*, *Dismiss*) and were rewritten to finish
  "this will …". Three icon-only buttons had the `title` as their only name, so they gained an
  `aria-label` (the indexing note's ×, Chat history's select toggle, the Library's sort).
- **The checks:** `apps/desktop/src/ui-rules.test.ts`, allow-lists in `ui-rules.allow.ts`, every
  entry a ratchet at the count below with the piece that brings it down. The test fails when a
  file goes over its entry and when it goes under (lower the entry), so the list stays the truth.
  A fourth check has no allow-list: a `<button>` never carries `title`.

**The counts, corrected.** The regex behind *In practice* stopped at the `>` of an arrow
function (`onclick={() => …}`), so it missed every `title` written after one. The checks read
each tag with a scanner that skips `{…}` and quoted values. On 2026-11-06, before the move:
310 buttons (two of the 312 were in comments), 73 with a `title`; after it, **237 buttons in 59
files have no hover text**. Local role classes: **28 files, 48 class names** (ADR-064's pass 2 split
some of the 16). Literal colours: **49 in 20 files**, the 45 plus four named colours (`green`,
`gray` in the two figure views), which check 3 counts too.

**Left open by piece (a).** 50 `title` attributes on elements that are not buttons, in 31 files
(35 on a `span`: a truncated file name, a marker's meaning). All but one (the Chat tab's scope
`<select>`) are on elements that cannot take focus, so what they say reaches a mouse only, which
choice 6's last sentence rules out for anything the user must know. They are outside the button
rule; piece (c) or (d) decides each: on the screen, or the tooltip on a focusable element.

## As built — ROADMAP 110 piece (b), 2026-11-06

- **The roles across the app.** The 28 components with their own `.primary` / `.ghost` / `.danger`
  are on `btn` / `btn-primary` / `btn-quiet` / `btn-danger`, and so are the local primaries under
  other names (`.send`, `.resume`, `.addbtn`, `.secondary`, `.neutral`, `.dangerbtn`). Each
  component's bare `button { … }` rule went too: it outranked `.btn` on specificity. *Send*, *Add
  documents*, *Finish setup…*, a dialog's *Save* and every empty state's call to action now use the
  dark theme's violet fill with white text. The ordinary look has one size, so the four sizes
  and three radii are gone. What stays in a component is layout (`.below`, `.again`, `.grow`).
- **Two additions to `app.css`, each defined once.** `btn-primary btn-danger` is a confirm's own
  button when it removes something, filled red: white on `#c0392b` is 5.4:1 in both themes
  (`--danger-fill` / `--danger-fill-fg`), where the tinted fill it replaces read 4.1:1 in the
  light theme. `btn-icon` is the same role for an icon-only button (about 26px a side), and any
  `btn` with `aria-pressed="true"` shows as pressed.
- **`ConfirmDialog` follows the alert dialog pattern** (choice 7): `role="alertdialog"` when the
  act removes something, labelled by its title and described by its body, focus on *Cancel*.
- **The five confirms** (choice 4): *Remove key…* (two places) and *Remove server…*, *Remove…* and
  *Remove all…* on the letter-spacing ignore list, *Dismiss…* on a chosen definition (it was
  *Clear*). Each is red and asks through `ConfirmDialog`, saying what goes and how it comes back.
- **Two more found and confirmed the same way**, both acting at once with no way back: *Reset to
  default…* in *Edit metadata* (it deletes your title, authors and year overrides) and *Undo all…*
  in the add-documents review (it deletes the copies the review made).
- **Choice 5.** The document page's *Source* and *Re-run…* are ordinary buttons outside the
  index's `nav`, at the strip's trailing end; the index keeps its pills.
- **Calls made while building.** *Pause* on the indexing screen was amber; it holds the run and
  removes nothing, so it is ordinary. *Reset* in the gap list writes a status, so it is ordinary,
  not quiet. *Select new + changed* and *Clear* in *Manage files* are selection helpers, so they
  are quiet, and *Clear* is now *Select none* (U-003). A second primary in *Manage folders* (*Add N
  documents*) became ordinary, leaving *Create* the dialog's one primary. Chat history's ⋯ *Delete*
  gained its "…".
- **Check 1 is exceptions only:** seven files, each with its reason in the allow-list and a comment
  in the file: two menu rows (`.menuitem.danger`), three gap severities on badges, notes and
  dots, and two dense icon-only danger buttons, now red at rest.
- **Seen in the running app** over an invented library, in dark, light and 375 px: *Remove server…*
  end to end (*Cancel* kept the server, the confirm removed it), both ignore-list removals (each
  with its *Undo*), *Reset to default…*'s confirm (Esc closed it and left the editor), the
  document delete, a folder's inline delete. Not seen live, for want of data in a new library: the
  definition's *Dismiss…*, *Remove key…*, *Undo all…*, the indexing screen.
- **Not done here.** Removing a document from a folder (the × in *Manage folders*) acts at once.
  The same row's checkbox undoes it, but choice 4 allows no exception, so it waits on the user.
  The Library header did not wrap at 375 px and cut *Add documents* off; it wraps now.

## As built — ROADMAP 110 piece (c), 2026-11-06

- **Every button has hover text** (choice 6): the 236 that had none, in 59 files, so check 2's
  allow-list is empty. Each text finishes "this will …" and says what the label cannot: what a
  quiet act leaves alone (*Clear* on the staged files: "nothing was added, and the files
  themselves are not touched"), what comes back and how (*Dismiss* on a definition option: "Show
  dismissed brings it back"), and which ones confirm first ("after you confirm"). A text follows
  the button's state, so a disabled button says why: *Send* "Type a question first", *Add to
  folder…* "Select documents first", a figure's *Next* "This is the last figure". The Settings
  sections take their existing one-line blurbs.
- **Icon-only buttons.** All of them had an `aria-label` already (piece (a) added the last three).
  The scan cannot see a label that CSS hides: the top bar's *Chat*, *Library* and *Graph* tabs drop
  their labels under 780px and had no name there, so each carries an `aria-label` equal to its
  label (WCAG 2.5.3, the name contains the visible label). A fifth check in `ui-rules.test.ts`
  fails an icon-only `<button>` without `aria-label` or `aria-labelledby`.
- **The 50 `title`s on other elements, decided one by one.** One is on an element that takes focus,
  the Chat tab's scope `<select>`; it moved onto the tooltip, so it shows on keyboard focus too. Of
  the other 49:
  - 17 give the whole of a name or path the screen cuts short (a file name, a folder, a source
    path). The start is on screen and assistive technology reads the element's whole text; they
    stay native `title`s, which WCAG leaves out of SC 1.4.13 as the browser's own tooltip.
  - 12 add detail nobody needs to act: what a count counts, a token breakdown, a graph build id, a
    reference's raw text, the chips past the cap. They stay too.
  - 20 explain what a marker means, and choice 6's last sentence rules those out: the current-chat
    dot, the disputed-topic chip's caveat, the source evaluation's stale, withheld and relevance
    notes, a turn's markers, *Excluded from retrieval* on a chunk, a figure's searchable mark and
    the reason it is not, a gap's blurb on a badge, *Has a chosen definition*, *Edited metadata*
    on a tile, *Proposed* on a field placement, the status bar's two engine-start notes. Each needs
    its meaning on the screen or as an accessible name, which is a design call per marker: not
    built here; proposed for piece (d) or a row of its own.
- **Seen in the running app** over an invented library (the installed app held port 8001, so the
  check ran on 8011 behind a throwaway Vite config): hover text in dark and light, a disabled
  button's reason, a text that changed when a tile was selected, the tooltip on Tab focus with
  `aria-describedby` set, the top bar's icon-only tabs at 375 px with their names and the tooltip
  inside the window.
- **Found, not done.** Three components keep a button look of their own that check 1 does not
  catch, because the classes have other names: the staged-files bar in `App.svelte`, the claim
  review's *Accept* and *Reject*, *Detect* and *Accept* in Manage keywords. The staged bar's *Clear*
  drops a list (nothing is added or deleted), but U-003 retires the word for a removal. A
  definition option's *Dismiss* acts at once, as the gap list's does; both can be undone from the
  same screen. Each is piece (d)'s or the user's call.

## As built — ROADMAP 110 piece (d), 2026-11-06

- **One dialog frame (choice 8).** `shell/Dialog.svelte` draws the backdrop and the frame, closes on
  *Esc* and a click on the backdrop, starts focus inside (on what the dialog's content focuses, else
  on the frame), keeps *Tab* inside, and hands focus back to what opened it when it closes (the
  WAI-ARIA dialog pattern). It has three shapes: a `card` centred on the window, a `side` panel
  along the right edge (Settings, a source) and a `screen` that takes most of the window
  (Indexing). All 20 components that drew a modal of their own render it now: the 14 with a
  `.modal` (ConfirmDialog among them), the figure viewer, both add-documents dialogs, the source
  panel, Settings and Indexing. What a dialog shows stays its own.
- **The list of open dialogs** (`lib/shell/dialogs.ts`, tested) settles what spans dialogs. *Esc*
  closes the one in front, and only it, so the guards Settings, Indexing and Edit metadata kept
  against the questions they open are gone. A dialog opened from another sits at the next layer
  (40, then 2 more per dialog under it), which replaces the fixed 40/41 and 42/43, and also the
  10/11 (Settings, a source) and 30/31 (Indexing) that the menus at 20 to 31 could cover. The
  frame is centred without a transform, so a `position: fixed` inside it measures from the window.
- **What changed for the user:**
  - *Esc* in a rename field (Manage folders, Manage keywords) cancels the rename and the dialog
    stays. It used to close the dialog as well.
  - The add-documents review closes on *Esc*; it had no *Esc* before. Once files are added, *Esc*
    and the backdrop finish like *Done*, so the library hears about them, and nothing closes it
    while it adds or undoes.
  - Every dialog keeps *Tab* inside it (only the three panels did) and gives focus back.
  - One backdrop, `--scrim`. In the light theme it is the cards' backdrop as before. In the dark
    theme it now darkens the screen: the 11 centred dialogs mixed `--fg` into it, which there laid
    a light veil behind a darker card, while the panels used black.
- **Colours: 36 in 18 files to 0.** New tokens in `app.css`: `--scrim` (per theme), `--shadow-side`
  (the side panels and the narrow window's sidebar) and `--paper` (the white behind a page image, a
  crop or a figure, in both themes). The fallbacks written after existing tokens went
  (`var(--danger, #c0392b)` and the like). The status bar's ready dot read `var(--ok, #2e9e5b)`,
  and `--ok` was never defined, so it was always that fallback green; it is `--ok-fg` now. The
  figure badges' `green` and `gray` became mixes of `--ok-fg` and `--fg-2`.
- **The shared stylesheet** is `src/shared.css`, loaded once after `app.css`. It holds a dialog's
  title, head row and buttons (`.dialog-title`, `.dialog-head`, `.dialog-actions`, written word for
  word in up to 11 files) and the plain `.muted`. A fifth style check counts the rules still copied
  between components: 211 before this piece, 179 after. It is one number, a ratchet: a new copy
  fails, so the second component to need a rule moves it there. The rest (`.hint` in three
  variants, `.err`, `.iconbtn`, the search row) is left to the change that next touches each.
- **Checks.** Check 3's list is empty. Check 4 fails an element with `aria-modal` anywhere but
  `shell/Dialog.svelte`, and check 5 counts the copies. `layout.test.ts` lets a screen take
  `Dialog` from `shell/`, beside `Icon` and `ConfirmDialog`.
- **Seen in the running app** over an invented library (API on 8011, the dev server on 5733):
  - the search near the top of the window, focus in its field, *Esc* closing it and focus back on
    the button;
  - the confirm over *Edit metadata* at the next layer, focus on *Cancel*, *Tab* wrapping, the
    first *Esc* closing only the confirm (focus back on *Reset to default…*), the second the
    editor;
  - *Esc* in the folder rename field;
  - Keyboard shortcuts, About, the add-documents chooser, Taxonomy, Manage keywords and Re-run
    ingestion, each centred, with focus inside, closing on *Esc*;
  - Settings full height at 884px, Indexing at 72rem, the source panel's resize handle above its
    frame and dragged from 420 to 580px;
  - light and dark, and 375px (a card at 94vw, Settings at 96vw, the sidebar's backdrop).

  The browser pane was hidden, which freezes a transition where it starts: the side panels'
  slide-in was seen to begin (760px off), and their layout was checked with reduced motion.
- **Not seen live:** Filter by keyword (its open flag is App's own state), the figure viewer and
  *Edit spacing* (an invented library has no figures or spacing rows), the add-documents review
  (it needs files on disk), *Open library* and *Re-index folder* (they need an existing folder).
  Each is the same frame, moved the same way; `svelte-check` and the build pass.
- **Still open, with the user:** the 20 markers whose meaning lives only in a `title`, the three
  button looks check 1 cannot see, the staged bar's *Clear*, and the × in *Manage folders*.
  `.iconbtn` (10 files) looks the same as `btn btn-quiet btn-icon`; moving it is the same kind of
  sweep as the copies and is left with them.
