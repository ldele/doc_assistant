<!-- status: active · updated: 2026-11-06 (revised while proposed: choices 6 to 8 and three checks that fail, from the user's words of 2026-11-06; 2026-11-05: proposed, the two select bars built to it) · class: append-only -->

# ADR-063 — A button's look says what it does: four action roles, and a delete always asks first

- **Status:** proposed (Claude Code, 2026-11-05). The request is the user's, 2026-11-05, in one
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

Option A, with the choices above as recommended, once the user accepts them. Built now: the four
classes in `apps/desktop/src/app.css` and both select bars. The `--accent-fill` pair (choice 2) is
not built: it waits on the user's answer, and until then `btn-primary` uses `--accent` and
`--accent-fg` as every primary does today. Choices 6 to 8 and the checks are not built; they are
ROADMAP 110, which waits for ADR-064's pass 2 to be committed, because both sweep the same
components.

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
