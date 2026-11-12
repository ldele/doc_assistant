<!-- status: archived · updated: 2026-11-09 · class: append-only -->

# DEVLOG — archive 010 (2026-11-05 (3) → 2026-11-06 (3))

Rotated DEVLOG entries — moved here from `docs/DEVLOG.md` once it held more than 20, **verbatim** per cpc ADR-023 rule 13b and cpc ADR-053. Newest entry on top. Append-only; never edited.

## 2026-11-10 — cpc 1.18.1 → 1.22.1: two files leave `.claude/`, cpc's templates leave `docs/`

**What changed.** Re-vendored cpc 1.18.1 -> 1.22.1 from a worktree of the `v1.22.1` tag
(`tools/conventions/cpc/`, 64 modules; `tools/conventions/templates/` is new). The known issues
and the friction register moved to `docs/KNOWN_ISSUES.md` and `docs/FRICTION.md` with `cpc-move`,
and `.gitignore` names both, so they stay local (ADR-029). The four template copies cpc had laid
(`docs/decisions/ADR-000-template.md` and its pivot, `docs/specs/SPEC-000-template.md`,
`docs/sprints/SPRINT-000-template.md`) are deleted: cpc's are in `tools/conventions/templates/`.
`AGENTS.md` numbers the three files every session reads and lists the rest as read on need;
`docs/local-only.md` and three `.claude/` files name the new paths.

**Why.** cpc 1.22.0 sorts documents by when they are read, with `.claude/` for every session and
`docs/` for a task, and its init check fails until the known issues move. 1.22.1 keeps cpc's
templates in cpc's folder. The user, 2026-11-10: update this project first, and keep the two
moved files out of git.

**Rejected.** Keeping the known issues in `.claude/` with a `[paths]` line: it would leave this
project off the layout every other one moves to.

**Verified.** `init_check`: 0 errors. `docs_check --strict`: 0 errors and 0 warnings, where the
re-vendor alone gave 4 errors and 16 warnings. A baton rotation of 6 entries cleared 5 warnings.
The user ruled the other 11 the same day: caps of 650 tokens for one baton or DEVLOG entry,
transitional, until 8 entries written before cpc's caps rotate out; 9,000 for the session-start
files, measured at ~8,458; a `## Releases` section in the ROADMAP with no entry, so cpc's release
checkpoints stay off; a `## Concept map` in `docs/architecture.md`, drafted from the import graph.

**What it opens.** The concept map's grouping is a draft and its May-import column records
today's imports, some of which run upward; correct it before wiring `cpc-dod-lint`. The two entry
caps go back to 300 and 500 once the long entries rotate out. Three shipped specs still name the
old sprint template path.

## 2026-11-09 (3) — New README media: a three-beat demo GIF and five screenshots

**What changed.** `docs/assets/provenote-demo.gif` is re-recorded on the current UI (1280×800,
21 s, 18 frames, 0.79 MB, light theme) in three beats: a question streams into a cited answer; a
citation opens its passage and *Show the page* opens the printed page beside the document page;
back in the answer, its sources and one flagged *uncited* claim. Five PNGs are new in
`docs/assets/` (112–223 KB each): the answer with a flagged claim and the source viewer under
*Why Provenote*, and the Library grid, the library summary and the concept graph with a chosen
definition in a two-column table under *What you can do with it*. The GIF's alt text and the
`docs/DEMO.md` line that describes it now say what is filmed.

**Why.** The user, 2026-11-09: new media for the rewritten README; the old GIF predated the title
bar, the settings cog and the library summary.

**How.** Recorded over the public demo corpus (28 arXiv papers) in a new scratch data home outside
the repo, never `data/`, with Ollama `llama3.1:8b` forced (the recorder aborts unless the page
reports that data home and an Ollama engine). The answer is real model output, not a mock; the
recorder retries a fresh chat until an answer carries a flagged claim (3 attempts). Toolkit:
puppeteer-core + installed Chrome + Pillow, kept out of the repo as before.

**Rejected.** Mocking `/api/chat` with a written answer (a staged answer is not the product). The
library summary and the grid in one frame (the summary is a separate view that replaces the grid).
The keyword-filter dialog as the grid shot (it covers the grid). The optional add-documents review
sheet (not recorded).

**What it opens.** The definition panel names its source and links the passage but shows no page
number, though the README says definitions are quoted "at their page". At 375 px the two-column
table's images are 144 px wide.

## 2026-11-09 (2) — ROADMAP 113: the eval harness is reinforced before the chat work

**What changed.** A new row, 113 (F11 Quality), and one sentence each in the ROADMAP's *Next* and
F11 blocks. No code. The sessions are in the local plan.

**Why.** The user, 2026-11-09: *"Add to the plan for 0.8.0 [...] a reinforcement of the harness
and benchmarks. Since one of the planned goals is to work on the chatbot part, it seems
appropriate."* Read against the plan, four chat sessions are gated on a measurement the harness
cannot take: the prompt fence on 10 questions with no hostile passage, context handling with no
conversation case, lookups with no record or not-in-the-library case, and modes with citation
coverage that is not a scorer.

**Rejected.** Reinforcing the harness after the chat work (its *before* runs would be lost).
Adopting an evaluation framework (*What NOT to do*: no framework, no new dependency).

**What it opens.** The spec decides the case families, the statistic, the model panel and the
paid budget. Two pieces need the user's own reading: verifying the new references and grading a
sample of answers for the judge check. Row 15 (the harness as its own repository) stays after.

## 2026-11-09 — The README leads with the product; the full limitations move to `docs/limitations.md`

**What changed.** `README.md` goes from 365 lines to about 190. It opens on a one-line pitch, the
demo and a download link, then *Why Provenote*, *What you can do with it*, *Get it*, one benchmark
table (*Measured, not asserted*), *How it works*, a six-item *Limitations* list, a short *Status*
and the documentation table. Three badges are new (CI, latest release, licence). The Limitations
section moves verbatim, links re-based, to the new `docs/limitations.md`. The README's *How long
does indexing take?* section is removed: `docs/performance.md` § 1 already holds each of its
figures. `docs/OVERVIEW.md` and `docs/QUICKSTART.md` point at the new file.

**Why.** The user, 2026-11-09: a better description of the project in the README, more
"marketable", with the technical detail moved to linked files.

**Rejected.** Dropping the benchmark caveats from the README (results stay indicative, with their
limits beside them). Cutting Limitations from the README altogether.

**What it opens.** The demo GIF predates the title bar and the library summary; new media is
planned, not recorded. The local-model citation figures still predate the 0.7.0 fix. The removed
indexing section was the only plain-language version of those figures.

## 2026-11-06 (10) — The settings cog moves beside the menu (ADR-065 choice 7)

**What changed.** The top bar's *Settings* button moves from the right end, between the wordmark
and the window controls, to the left, right after the menu (`shell/Topbar.svelte`). Its name and
hover text are unchanged. The bar's right end now holds only the wordmark and, on Windows, the
window controls. Two comments that described the old place are updated (`Topbar.svelte`,
`screens/library/HeaderBar.svelte`), and ADR-065 choice 7 gains its *As built* note.

**Why.** The user, 2026-11-06: *"The settings cog should be moved to a better place now."* Beside
the window controls it read as part of the window's frame. Of the four places offered, the user
chose beside the menu.

**Rejected.** The sidebar's footer, before the wordmark, the status bar (ADR-065 choice 7).

**What it opens.** Checked in the Vite preview over the invented library: the button order, the
hover text, the click opening the Settings drawer, and the light theme at 375, 640 and 1280 px
with no sideways scroll. Along the bar, with the drag script's rules applied point by point, the
cog never starts a drag and the empty space still does. Not checked in a Tauri window, nor by a
screenshot (the browser pane was hidden and did not draw). The app menu still has its own
*Settings* entry, so Settings now has two doors a few pixels apart.

## 2026-11-06 (9) — The top bar is the window's title bar on Windows (ADR-065)

**What changed.** On Windows the window has no native title bar any more
(`src-tauri/tauri.windows.conf.json`: the main window restated with `decorations: false`,
`shadow: true`; other platforms keep their frame). The top bar takes its place: its empty space
drags the window and a double-click maximises or restores it (`data-tauri-drag-region="deep"`,
`"false"` on the app menu's wrapper), and `shell/WindowControls.svelte` draws minimise,
maximise/restore and close at its right end, full height, the platform's look, Close red on hover
(`--caption-close`). The window calls go through `lib/core/tauri.ts` and degrade to nothing in a
browser. Four window permissions join the capability set. Built in `f35cabe`; the hover texts came
with ROADMAP 110's merge (`5ed8bf5`). ADR-065 records it, written after the build at the user's
request. The user then chose where the settings cog goes, beside the menu; not built yet.

**Why.** The user, 2026-11-06: the native title bar is *"not modern (dead space)"*: a row of its
own (31 px at 96 DPI) above a top bar that already had empty space.

**Rejected.** Keeping the native frame; frameless on every platform (Windows-style buttons on a
macOS window); native hit testing to keep the snap layouts on hovering maximise (window-procedure
code Tauri does not provide; Win+Z still opens them). ADR-065 § Options.

**What it opens.** Driven in a dev window over an invented library: drag, double-click maximise,
restore, a maximised frame equal to the work area, Close at the corner pixel, Win+Z. Not driven
yet: minimise and close by click, edge resizing, keyboard focus, the light theme, a 640 px window,
a dialog over the bar. They are a row in `docs/release-ux-checklist.md` §6, before 0.7.3's tag.
The cog's move to the left of the top bar (ADR-065 choice 7).

## 2026-11-06 (8) — The library summary stops saying indexing reads no metadata

**What changed.** The *Metadata* card's two lines (`lib/library/summary.ts`,
`screens/library/Summary.svelte`) now say indexing fills metadata for new or changed documents
only, and that *Re-run… > Metadata* reads it for documents indexed earlier. ROADMAP 112's
"Exposed" note says the same. The merge of ROADMAP 110 (`5ed8bf5`) also gave the title bar's three
window buttons their hover text, because `ui-rules.test.ts` check 2 failed without it: *Hide the
window in the taskbar*, *Make the window fill the screen* / *Return the window to its earlier
size*, *Quit Provenote*.

**Why.** Since `3a1966f` (merged `1123b3d`) indexing fills a new document's blank title, authors,
year and DOI, so "Indexing does not read metadata yet" was wrong for every document indexed after
it. It stays true for documents indexed before, and nothing re-reads those on its own.

**Rejected.** Naming a version in the copy ("since 0.7.3"): the reader of the card knows when a
document was added, not which build added it.

**Opens.** ADR-065 (the title bar) is cited by the code but not yet written; the window buttons'
texts belong in it.

## 2026-11-06 (7) — ROADMAP 110 (d): one dialog frame, and no colour outside the tokens

**What changed.** `shell/Dialog.svelte` is the one dialog frame: backdrop, frame, *Esc*, focus in
and back, *Tab* kept inside, as a centred card, a side panel or a screen. The 20 components that
drew a modal of their own render it, and a list of open dialogs (`lib/shell/dialogs.ts`, tested)
makes *Esc* close the one in front and puts a dialog opened from another above it, which replaced
three hand-written *Esc* guards and the fixed layers. The 36 literal colours are tokens
(`--scrim`, `--paper`, `--shadow-side`, the fallbacks dropped). `src/shared.css` holds the dialog's
title, head and buttons. `ui-rules.test.ts` gains checks 4 (no `aria-modal` outside the frame)
and 5 (rules copied between components: 211 → 179, a ratchet).

**Why.** ADR-063 choice 8 and the user's go-ahead for piece (d) on 2026-11-06. The frame is markup
and *Esc* handling as well as style, so it is a component, not a class. Seen live over an invented
library: the nested confirm's layer, focus and two *Esc*s, the rename field, Settings, Indexing,
the source panel's handle, light, dark and 375 px. `npm test` 409/409, `svelte-check` 0, build ok.

**Rejected.** The native `<dialog>` with `showModal()`: its top layer would sit above the shared
tooltip, which must show over every dialog. Centring with `transform`: it makes a fixed element
inside the frame measure from the frame. A per-file list for the copied rules: 60 files long, and
a total that may only fall says the same. Moving `.hint`, `.err` and `.iconbtn` now: three
variants each, so unifying them changes spacing on many screens; left to the change that touches
each.

**What it opens.** Behaviour that changed and the user has not seen: the dark theme's backdrop
now darkens the screen behind a centred dialog (it lightened it), the review closes on *Esc*, and
focus returns to the opener. Still with the user: the 20 marker meanings, three button looks check
1 cannot see, the staged bar's *Clear*, the × in *Manage folders*.

## 2026-11-06 (6) — ROADMAP 110 (c): every button says what it does

**What changed.** The 236 buttons in 59 files that had no hover text have it, on the shared
tooltip, so check 2's allow-list is empty. Each text finishes "this will …" without repeating
the label, says what a quiet act leaves alone and what comes back, and follows the state: a
disabled button says why (*Send* "Type a question first"). The top bar's *Chat*, *Library* and
*Graph* tabs carry an `aria-label`, since under 780px their labels are hidden and they had no
name. `ui-rules.test.ts` gains a fifth check: an icon-only `<button>` needs `aria-label` or
`aria-labelledby`. Of the 50 `title`s on other elements, the Chat tab's scope `<select>` (the one
that takes focus) moved onto the tooltip; the rest are sorted in ADR-063's *As built (c)*.

**Why.** ADR-063 choice 6, and the user's go-ahead for piece (c) on 2026-11-06. The texts were
written from each button's handler, not from its label, after reading what it does: the
"Promote" on a gap marks it as being acted on, a sample question fills the box and does not send.
Seen in the running app over an invented library, dark, light and 375 px: hover, Tab focus with
`aria-describedby`, a disabled reason, a text changing with the selection. `npm test` 402/402,
`svelte-check` 0.

**Rejected.** Moving all 49 other `title`s onto the tooltip: on an element that cannot take focus
the tooltip never shows to a keyboard or a screen reader, and it removes the `title`, which a
screen reader can read today, so 20 marker meanings would get worse, not better. Generic texts
("Click to open"): they say nothing the label does not. An allow-list entry for the list rows:
every one had something to say.

**What it opens.** Piece (d): the dialog frame, the shared stylesheet, 36 colours; and, with the
user, the 20 markers whose meaning lives only in a `title`, three button looks check 1 cannot see
(the staged-files bar, the claim review, *Detect* in Manage keywords) and the staged bar's *Clear*.

## 2026-11-06 (5) — ROADMAP 110 (b): every button on a role, and a delete always asks

**What changed.** The 28 components that styled their own `.primary` / `.ghost` / `.danger`, and
the local primaries under other names (`.send`, `.resume`, `.addbtn`, `.secondary`, `.neutral`,
`.dangerbtn`), are on `btn` / `btn-primary` / `btn-quiet` / `btn-danger`. Their bare `button`
rules are gone too, since they outranked `.btn`. `app.css` gains `btn-primary btn-danger` (a
confirm's own button when it removes, filled red, `--danger-fill` in all four palettes),
`btn-icon`, and a pressed look for any `btn` with `aria-pressed`. `ConfirmDialog` is an alert
dialog: labelled, described, focus on *Cancel*. The five destructive actions ADR-063 counted
(*Remove key…*, *Remove server…*, *Remove…*, *Remove all…*, *Dismiss…* on a chosen definition,
which was *Clear*) are red and confirm, and so are *Reset to default…* in *Edit metadata* and
*Undo all…* in the add-documents review. The document page's *Source* and *Re-run…* left the
index's `nav` for its trailing end. The Library header wraps at 375 px. Check 1's allow-list is
seven exceptions with reasons; hover text 237 → 236 and colours 49 → 36 (two files' dead
`var(--danger, #c0392b)` fallbacks).

**Why.** ADR-063 choices 1, 4, 5 and 7, and the user's go-ahead for piece (b) on 2026-11-06. Seen
in the running app over an invented library, dark, light and 375 px; *Remove server…* and both
ignore-list removals driven end to end, cancel and confirm. `npm test` 401/401, `svelte-check` 0,
`npm audit --audit-level=high` clean.

**Rejected.** Keeping the confirm's tinted red fill: 4.1:1 in the light theme (3.7:1 hovered),
below WCAG's 4.5:1. A small-button size: one size meets the 24px target everywhere it was
used. Keeping *Pause* amber: it removes nothing, so it is ordinary. Renaming the gap badges'
`.danger` to pass the check: it names a severity from the data, so it is an exception with its
reason, not a disguise. A confirm on removing a document from a folder: the row's checkbox
undoes it, and choice 4 says that still confirms, so it is left for the user to decide.

**What it opens.** Piece (c), hover text on the 236 buttons left; piece (d), the dialog frame,
shared stylesheet and 36 colours. Not seen live for want of data in a new library: the
definition's *Dismiss…*, *Remove key…*, *Undo all…*, the indexing screen.

## 2026-11-06 (4) — ROADMAP 110 (a): the primary's fill, one tooltip, and checks that fail

**What changed.** The first of ROADMAP 110's four pieces. `app.css` gains `--accent-fill` and
`--accent-fill-fg` in all four palette blocks (`#6b5fd6` with white in the dark theme, 5.0:1;
the accent itself in the light, 8.2:1), and `btn-primary` uses them; its hover now darkens toward
black in both themes. One tooltip, `apps/desktop/src/lib/shell/tooltip.ts`, is a Svelte action
over one shared `role="tooltip"` element: it shows on hover and on keyboard focus, Esc dismisses
it, the pointer can move onto it, and the button's `aria-describedby` points at it while it
shows. All 73 native `title`s on buttons moved onto it (about 30 rewritten to say what the button
will do rather than repeat its label), and three icon-only buttons whose only name was that
`title` gained an `aria-label`. `src/ui-rules.test.ts` holds ADR-063's three checks as ratchets
at today's counts, per file, with reasons in `src/ui-rules.allow.ts`, plus a fourth with no
allow-list (no `title` on a button). ADR-063 gains *As built*; ROADMAP 110 lists the four pieces.

**Why.** ADR-063 choices 2, 6 and 8 and its *Enforced by*, accepted by the user on 2026-11-06;
the user picked this piece first from the proposed split. Seen working in the running app over an
invented library (a new data home, no `.env`, no key): dark, light and 375 px, a disabled button
included. `npm test` 401/401, `svelte-check` 0 errors, `npm audit --audit-level=high` clean.

**Rejected.** Moving only the 26 `title`s the ADR counted: its regex stopped at an arrow
function's `=>`, and a real scanner finds 73, so the rest would have kept two kinds of hover text.
Keeping a native `title` beside the tooltip as a fallback: two texts on one hover. Showing the
tooltip on every focus: a click focuses a button too, and a dialog focuses one by script on
opening, so it shows only after a navigation key moved the focus. Letting Esc through to the
dialog underneath: the tooltip is the top layer (the WAI-ARIA pattern), so the first Esc closes
it alone. A total count per check: a file could break a rule while another improved, so each
file has its own entry, and an entry above the measured count also fails.

**What it opens.** Pieces (b) roles and confirms, (c) hover text on the 237 buttons left, (d)
dialog frame, shared stylesheet and colours, each lowering its list. 50 `title`s on elements that
are not buttons reach a mouse only (ADR-063 *As built*). The corrected counts replace the ADR's
26, 286, 312 and 45 with 73, 237, 310 and 49.

## 2026-11-06 (3) — ADR-063 accepted as proposed

**What changed.** ADR-063 is accepted with all eight choices as recommended: the four action roles,
the Graph tab's look for ordinary buttons, a deeper accent fill with white text for the primary in
the dark theme (`--accent-fill`, `--accent-fill-fg`), *Done* as a mode's trailing primary, a
confirm on every danger action, *Source* and *Re-run…* as ordinary buttons beside the document
page's index, hover text on every button through one tooltip, the standards ranked over design
systems, and styling in one place with three checks that fail. ROADMAP 110 moves from waiting to
planned now, and its index line in `docs/decisions.md` says accepted.

**Why.** The user, 2026-11-06: *"Accept as proposed"*, after the revision in entry (2).

**Rejected.** Nothing new: the options are in the ADR.

**What it opens.** ROADMAP 110, the sweep over the components, once no other session has edits
in flight in them.

## 2026-11-06 (2) — ADR-063 revised: hover text, styles in one place, checks that fail

**What changed.** ADR-063, still proposed, gains three choices and an *Enforced by* that fails.
Choice 6: every button has hover text that says what it will do, shown on keyboard focus too and
meeting WCAG 2.2 SC 1.4.13, through one shared tooltip instead of the native `title`. Choice 7:
WCAG 2.2 AA and the WAI-ARIA Authoring Practices patterns decide what is correct, design systems
explain why, and the code names the pattern it follows (alert dialog, toolbar, listbox). Choice 8:
tokens and role classes in `app.css`, repeated small rules in one shared stylesheet, one dialog
frame, no literal colour in a component, every exception commented and allow-listed. Three
`npm test` checks with allow-lists (local button classes, a button without hover text, a literal
colour), landed first as ratchets. ROADMAP 110 widens to carry them; GLOSSARY U-003 names *hover
text*; the ADR's file names follow ADR-064's move.

**Why.** The user, 2026-11-06, relayed by the session that started this one: buttons share a
design, explain themselves on hover, carry the right colour and a confirm when dangerous, follow
industry conventions, keep styling conventions across the project with few exceptions, and the
project is stricter about it. Measured the same day: 26 of 312 buttons have a `title`, and the
`<style>` blocks of 20 components hold 45 literal colours.

**Rejected.** The native `title` as the hover text: it never shows on focus and cannot be
dismissed. Checks that fail at full strength on the day they land: 286 buttons would fail at once,
so each starts as a ratchet at today's count. Building the sweep now: ADR-064's pass 2 is editing
the same components in this checkout.

**What it opens.** ROADMAP 110, after ADR-064's pass 2 is committed and the user answers
ADR-063 (choice 2, the dark-theme primary; choices 6 to 8).

## 2026-11-06 — A dialog's door calls its module, not a callback through App (ADR-064, pass 2b)

**What changed.** Four rune modules in `lib/library/` take App's one-domain state:
`reingest.svelte.ts` (*Re-run ingestion*), `documents.svelte.ts` (the list, *Edit metadata*,
*Delete this document?*, select mode), `folders.svelte.ts` (*Manage folders* and the folder
writes) and `keywords.svelte.ts` (*Manage keywords* and the family writes). Each door calls its
module (`PageIndex` calls `openRerun`, as ADR-064's worked example drew it), and the props that
only carried a door up through two components are gone. App keeps the glue: anything that also
touches another domain (the health refresh after a re-run, the facet re-point after a family
write, the graph latch). App: 1,778 → 1,478 lines. Driven in the preview, read-only: every door
opens its dialog with the right label and parts and cancels (Re-run from the page names the title,
from the select bar "2 documents"); *Manage keywords* opens at the top from the filter, on Concepts
from the Graph tab, on its row from *Edit*; the chat delete confirm still names the count; 375 px
light and dark, no script error. npm test 391/391, svelte-check 0/0.

**Why.** ADR-064 choice 5, accepted: a person who sees a dialog finds its state beside its
domain, and a new dialog adds a module function, not a flag, a callback prop and a mount in App.

**Rejected.** Moving conversation deletes: both confirms read the chat view and start a new chat,
which `conversations.svelte.ts` already leaves to App. Pushing App to choice 5's 600–800 by moving
the chat lifecycle or navigation: choice 5 keeps those, and the estimate (~830 one-domain lines)
was read from function names; ~300 were.

**What it opens.** The writes the modules hold (create a folder, save metadata, delete a document,
write a family, a re-run to the end) were not driven live: same API calls, moved, but a check needs
a scratch copy of the library. Smaller App candidates are in ADR-064's "As built — pass 2b".

## 2026-11-05 (10) — Eight screens cut into the parts they show (ADR-064, pass 2a)

**What changed.** Eight of the nine large files are cut along their headings and sections, each
part a component beside its entry file, with the entry's props unchanged: Settings (its four
sections; the provider list and forms), Manage keywords (five parts), the Sidebar (Chat history
leaves for `screens/chat/ChatHistory.svelte`, which App hands to the frame as a `chatRail`
snippet), Graph (the concept panel), Taxonomy (the forest, a field's detail), the document page
(its index, header, Chunks) and the Library tab (the header bar). ADR-064's "As built" has the
sizes and every place the plan was not followed. Driven in the preview against the dev API:
each screen and part opens and acts as before (sorts, select modes, the ⋯ menu, wheel zoom, a
Chunks block to its page, the delete question answered *Keep*, Settings keeping a typed folder
across a section switch), at 375 px in light and dark, no script error; npm test 389/389,
svelte-check 0/0.

**Why.** The user accepted choice 6. Choice 5, App's state, changes the props of the screens whose
doors it rewires, so it comes after (2b), on smaller files.

**Rejected.** A library rail cut out of the Sidebar: it would copy ~120 of its 150 CSS lines.
Shared rail components in `shell/` (one tree row, one filter box): the layout test allows a screen
only `Icon` and `ConfirmDialog` from `shell/`, and widening that is a decision, not a split.
`:global` styles from the frame: they would outrank a rail's own rules.

**What it opens.** 2b. Five parts stay above 300 lines, each its own CSS or logic (ADR-064 names
them). The four splits other than the Sidebar ran in parallel subagents, each checked by
svelte-check, tests and a script comparison of markup and CSS against `HEAD`; the ADR's "one file
at a time" became one live pass over all of them.

## 2026-11-05 (9) — `oauthlib` 4.0.1: an advisory in a package that never ships

**What changed.** `uv.lock`: `oauthlib` 4.0.0 → 4.0.1 (`uv lock -P oauthlib==4.0.1`); nothing else
moved. **Found by:** the pip-audit gate on 2026-11-05: CVE-2026-49272.

**Why.** It comes in through `chromadb` 1.5.9 → `kubernetes` 36.0.0 → `requests-oauthlib` 2.0.0,
Chroma's cluster client. Nothing here loads it: `src/`, `apps/`, `scripts/` and `tests/` never
import it, a local `PersistentClient` round-trip loads none of the three, and the 0.7.2 backend's
`PYZ-00.toc` holds 157 `chromadb` modules and no `oauthlib` or `kubernetes` one. So it reaches the
dev venv and CI only, and no CHANGELOG line. 4.0.1 is the first fixed version, 5 days old.

**Rejected.** A reviewed ignore in `pip-audit-ignore.toml`: the code never runs, but a patch
release that moves one package is cheaper than an entry someone has to revisit.

**Checked.** The gate is OK against an export of the new lock with CI's extras (197 packages,
0 unreviewed, 6 reviewed ignores matched, 0 stale). The 677 tests in the 49 files that use Chroma
pass with 4.0.1 overlaid on the main venv. The full suite was not run.

**What it opens.** The main venv keeps 4.0.0, and the local gate fails there, until its next
`uv sync`.

## 2026-11-05 (8) — A sentence is never a document's authors (KI-89)

**What changed.** `metadata_extractor._looks_like_author_line` gains `_reads_as_prose`: a candidate
with four or more lowercase words in a row, which also has more lowercase words than capitalised ones
or opens like a sentence (one capitalised word, then the run), is not an author line, so the scan
goes on to the next line or returns None. The public case: `pointer_networks_vinyals_2015.pdf` stored
*"Nonetheless, these methods still require the size of the output dictionary…"* as its authors,
because the head's 3,000-character cut leaves an Introduction fragment short enough to pass every
existing test. Fifteen new tests; the five prose ones fail without the guard.

**Why.** A sentence has commas and an *and* too, and since the index run fills blank metadata
(entry (7)) this would reach every new document. Re-measured on a
copy of the library (KI-89): every prose list gone (one falls through to a venue footnote, still
wrong), no correct list lost, and three documents whose title had been taken as their authors now
get the real list by falling through.

**Rejected.** A list of English function words: it is the measuring stick in KI-89, not a rule,
because it assumes a language. A run limit alone: *"Jane Doe, John Smith, on behalf of the X
Consortium"* has a run of four and is a correct list. A majority of lowercase words alone: *"Smit and
van der Berg"* has one. The three-word limit is how names are built (a connective plus *van der* or
*de la*), not a number fitted to this library. Cutting the author scan at the abstract: a layout
that prints the abstract before the names would lose its authors, and it was not measured.

**What it opens.** The stored rows stay: neither the re-run nor the enrichment runner writes None
over a value, so a *Re-run… › Metadata* clears no prose list the extractor now rejects; whether it
should is KI-89's open question. A self-citation line that begins with the right authors (ACM's
*"Name and Name. 2020. Title…"*) is not prose and is kept whole.

## 2026-11-05 (7) — The app's own index run fills title, authors, year and DOI (KI-87's library)

**What changed.** `ingest.main` gains a post-loop step, `_fill_missing_metadata`, right after the
catalogue's pass (`_apply_imported_metadata`). It takes the hashes the run indexed (new documents,
and documents re-read because their text changed), and `metadata_enrich.fill_missing_metadata`
runs the deterministic extractor over their cached text. It writes a column only while it is blank.
It works in batches of 100: one read of the four columns, one read of the texts that still have a
blank, one write. A document with nothing blank is not read. Like the steps beside it, it logs a
warning and never fails the ingest. `_persist_metadata` (the `scripts/enrich_metadata.py` runner)
now shares the column rule and, like the catalogue's fill-the-blanks mode, counts an
all-whitespace string as blank. Seven tests in `tests/integration/ingest/test_ingest_fills_metadata.py`:
0 documents, a fill, a re-index that keeps an extracted value and one that keeps a catalogue's, a
text that yields nothing, and a failing step. The fill and the two re-index tests fail without the
step, and the re-index ones also fail when it overwrites.

**Why.** The app's index run and *Add documents* never read metadata, so a library indexed only in
the app had no title, author or year on any document (KI-87's library; found inventorying the
library summary's signals, ROADMAP 112). No ADR: ADR-049 D3 already orders the slots (extractor <
catalogue < the user's edit) and already puts a post-loop metadata pass in `ingest.main`; this adds
the lowest slot under that order. KI-54's own measurement recommends running the extractor on blank
titles (14 right, 1 journal name, of 16). **Cost, measured** on a copy of the 2026-11-05 backup
library (104 documents, cached text median 71,704 characters, all four columns blanked, served
through `DOC_DATA_DIR`): **102 ms for all 104 (98.6 to 110.4 ms over 5 runs), about 1 ms per
document**, of which the extractor is 0.36 ms. It filled 102 titles, 76 author lists, 93 years and
26 DOIs. A second pass, with 84 documents still holding a blank, took 73 ms; with nothing blank,
0.9 ms. Extracting one PDF takes 14.7 s on average (`docs/performance.md`), so the step is under
0.01% of a first index; a re-read of 10,000 documents would add about 10 s. End to end, two public
PDFs (64 and 9 pages) indexed into an empty data home took 52.2 s, of which the step took 43.6 ms,
most of it the first import of its modules. On the copy the step's values match what the library
already stored from earlier runs on 97 of 104 titles, 97 of 104 author lists, and every year and
DOI.

**Rejected.** Filling every document on every run: a Re-index would change documents it did not
index, with nothing on the run's screen saying so. Replacing the extractor's earlier answer on a
re-read: replacing it is what *Re-run…* › *Metadata* is for. Calling `enrich_metadata` per document:
it opens a session per document, and with no bound on how many texts it loads, a re-read of the
whole library would hold every text at once. Extracting inside `process_document`: that puts it in
the locked ingest core and ahead of the catalogue's pass.

**What it opens.** A library already indexed in the app is not filled by a Re-index (its documents
are skipped as unchanged): one *Re-run…* › *Metadata* over them fills it. KI-54's journal-name title
now arrives at index time on about one new document in sixteen, to be corrected as an edit. **A
second defect, not filed yet:** the author picker takes a sentence of body text on 9 of the 76
author lists it returned on the copy (for *Pointer Networks*: "Nonetheless, these methods still
require…"). The stored library held 10 such lists, 8 of them the same, so the step brings the
defect forward to index time rather than adding it; a structural guard in `_extract_authors` is
its own fix.
`reingest._rerun_metadata` on an imported document re-applies the catalogue and leaves the fields it
lacks blank; the index step fills them. The library summary's *missing metadata* now means the
extractor found nothing, for documents indexed from here on. The spec's *Opens* line and a ROADMAP
row are the spec author's to update.

## 2026-11-05 (6) — Every component moves to the screen it shows (ADR-064 accepted, pass 1)

**What changed.** The user accepted ADR-064's seven choices as recommended. Pass 1, moves only: all
54 components left `apps/desktop/src/lib/` for `src/screens/<screen>/` (chat, library,
document-page, graph, manage-keywords, taxonomy, settings) or `src/shell/`, 28 of them renamed to
what the app shows (`LibraryBrowser` → `DocumentPage`, `ReingestDialog` → `RerunIngestion`,
`Sources` → `ManageFiles`, `LibraryTaxonomy` → `screens/taxonomy/Taxonomy`). Imports, component
tags and the comments naming a component follow; git records all 54 as renames (93–100%
similar). `src/layout.test.ts` holds the folder rules (no `.ts` in `screens/`/`shell/`, no
component in `lib/`, nothing in `lib/` or `shell/` imports a screen, a screen takes only `Icon` and
`ConfirmDialog` from `shell/`, every relative import resolves); it failed four of five on a copy with
planted violations. The security test's sink path, `docs/architecture.md` (the spine table's last
column, and a new section: every screen and dialog with its doors), GLOSSARY U-001/U-002 and the
new U-004, `docs/security.md` S2, ROADMAP row 111 and `apps/desktop/CLAUDE.md` follow.

**Why.** The user asked for a layout where a person who sees a screen finds its code in one place.
The rewrite ran first on a copy of `src/` with an import-resolution check, then for real once
`c2b2fef` left `apps/desktop/` clear.

**Rejected.** Splitting in the same commit: a file renamed and cut in one commit can fall under git's
50% rename threshold and lose its history, so pass 2 starts after this is committed. Renaming the
wire types that share a component's old name (`ConceptGraph`, `DocConnections`, `DocReferences`):
they mirror `apps/api/models/` and stay.

**What it opens.** Pass 2: the nine files of 690 lines or more, and App's one-domain state into rune
modules. ADRs, dated specs and archives keep the old paths as the record of their time.

## 2026-11-05 (5) — The library summary follows the user's seven recommendations (ROADMAP 112)

**What changed.** Six of the seven, in the wireframe (revision 2, four screens, rev 1001), the spec
(§ As built) and the screen. A count is navigation: each count row is its own target (hover,
chevron), a zero row is plain text, and *Show the N* buttons are gone. *Need attention* leads and
scrolls to *From indexing*; passages and pages are a context line. Meters read "N of M" with a thin
bar. Each *From indexing* row names its fix: files failed in this session's run open the indexing
screen, other files Settings › Documents, document issues the grid. "Not read" is said once: the
wire gains `metadata.with_any_field`, and at 0 the Metadata card is one line; references, figures
and keywords say *not looked for* while none is on record anywhere. Bars are horizontal, one hue,
value at the tip, square at the baseline. Driven again on invented libraries (60 documents; 140
with nothing read), at 1280 px dark and 375 px light, where the fix moves under the reason and a
meter's bar under its words (container queries).

**Why.** The user approved revision 1 and then sent the seven (*"Send all"*), on the condition that
the build follows ADR-063.

**Rejected.** *Years* sorted by count (recommendation 6): decades are an ordered scale and the dataviz
guide keeps an ordinal axis in order; *Types* are ranked. A lone *No year* bar when no document has a
year: a one-bar chart, so a sentence. Recommendation 7 (keep each count run for a trend): later work.

**What it opens.** The not-read rule is a heuristic on counts until a re-run records that it looked;
the canvas toolkit read revision 2 after the user's two open tabs had written revision 1 back into
the restarted canvas (T-023's "gives a restarted server back whatever it lacks"), so it was redrawn
on a fresh port.

## 2026-11-05 (4) — The library summary, built (ROADMAP 112)

**What changed.** The Library pane has a third layout, *Summary*, for the whole library or the open
folder, built to the wireframe the user approved with no markup. `library/summary.py` reads one fact
record per document and defines ten *issues* once (`no_text`, `flagged`, `missing_year`, …); the
counts sum them and `list_documents(issue=…)` selects by them, so *Show the N* opens exactly N, as a
new grid collection (`issue`). `GET /api/library/summary[?folder_id=]` is read-only and adds this
session's run failures; `GET /api/library/documents` takes optional `issue` and `folder_id`. Driven
in the preview on invented libraries (60 documents, then 0 with 6 files waiting): the three screens,
*Show the 7* → 7 documents with no year, a folder's summary and its door back, Settings opening at
Documents, 375 px in the light theme with no sideways scroll; dark at 1280 px.

**Why.** The user, 2026-11-05 (ROADMAP 112). The numbers say *on record* where the app never looks
(the spec's signal table): its indexing reads no metadata, references or figures.

**Rejected.** SQL predicates per issue: the effective-metadata rule (ADR-013) lives in Python, and a
second copy in SQL would be the drift this feature's one-definition rule exists to prevent. An `IN`
clause for the filter: SQLite's variable limit at 10,000 ids. Persisting *Summary* as the layout:
the Library would open on counts instead of documents. A per-document `EXISTS` on `citations`: no
index on the column.

**What it opens.** Measured at 10,000 invented documents: the route 0.68 s, but the unfiltered
document list the pane loads is 6.5 s and 4.2 MB (`docs/performance.md` § 3 and § 6). The
registry's "last look" cannot tell a folder emptied of every file from an old scan (the spec's known
limit). Indexing that reads metadata is its own session.

## 2026-11-05 (3) — The library summary, gate first: signals read, wireframe drawn, spec written (ROADMAP 112)

**What changed.** No code. `docs/wireframes/library-summary.md` (+ `.excalidraw`): three screens,
the library's summary, a folder's, and the same view at 0 documents, as a third layout (*Summary*)
of the Library pane beside *Grid* and *List*; drawn at rev 1060, gate `fallbacks 0 · inferred 0 ·
unnamedScreens 0 · orphans 0`, waiting on the user's markup. `docs/specs/feature-library-summary.md`
(draft): which signals exist and how far each can be trusted, with file:line; one read-only route,
`GET /api/library/summary[?folder_id=]`, logic in `library/summary.py`; an `issue` filter on the
document list for *Show the N*; how each number behaves at 0 and ~10,000 documents.

**Why.** The user, 2026-11-05: a summary per library or folder of what went wrong at indexing and
facts about the documents. Read first, it turned out that the app's own indexing and *Add documents*
never read metadata, references, figures or keywords (only a catalogue import brings metadata), so
"missing" there means "not on record", and a document parsed to no references looks exactly like
one never parsed. The drawing says so in its footnotes instead of presenting either as a fact.

**Rejected.** Counting through `GET /api/sources`: it scans and writes registry rows, so the summary
reads the rows as of the folder's last look instead. A per-document `EXISTS` over `citations`: the
column has no index; one grouped pass instead. Ratios on the wire: a count of 0 cannot be divided.

**What it opens.** Two follow-ups the screen exposes: indexing that reads metadata (milliseconds per
document), and a record that a *Re-run…* part ran, so *found none* splits from *never looked*. The
canvas reading numbers the pane column by column under a full-height sidebar: the toolkit's T-006,
a sighting added to its ledger; the wireframe doc states the intended order.
