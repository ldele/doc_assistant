<!-- status: active · updated: 2026-11-06 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps at most 20 entries** (`devlog_max_entries = 20` in `scripts/conventions.toml`,
> cpc rule 13b, user standard 2026-09-10; `tests/unit/test_doc_sizes.py` pins the same number).
> Past 20, one batch cuts it to the newest 10 (`devlog_rotate_to`, cpc 1.12.0, since 2026-09-29).
> Rotate with `python tools/conventions/rungate.py rotate --root . --file devlog --write` (the shim —
> calling `tools/conventions/cpc/rotate.py` directly fails under `.venv`; corrected 2026-09-16) — it moves the
> oldest entries **verbatim** into the highest-numbered archive, or starts the next one when that is
> past `archive_max_tokens`, and verifies the bytes; then update the range below by hand (cpc ticket
> T-003), and check a new archive's title (cpc 1.12.0 titles every new one "archive 001"). A day may
> be split across two files at the cut. Every archived heading is listed in
> [`docs/archive/DEVLOG-INDEX.md`](archive/DEVLOG-INDEX.md). Older entries, newest-first, unedited:
> **2026-10-24 → 2026-11-05 (2)** in [`docs/archive/DEVLOG-archive-009.md`](archive/DEVLOG-archive-009.md)
> (rotated 2026-10-31: 11 entries, 2026-11-02: 11 more, 2026-11-05: 11 more, and 2026-11-06: 11 more) ·
> **2026-09-30 (3) → 2026-10-22** in [`docs/archive/DEVLOG-archive-008.md`](archive/DEVLOG-archive-008.md)
> (rotated 2026-10-17: 11 entries, 2026-10-21: 11 more, and 2026-10-27: 11 more) ·
> **2026-09-07 → 2026-09-30 (2)** in [`docs/archive/DEVLOG-archive-007.md`](archive/DEVLOG-archive-007.md)
> (rotated 2026-09-30, the first batch: 11 entries, and 2026-10-09: 11 more) ·
> **2026-08-12 (1) → 2026-09-04 (2)** (the first 2026-09-02 and 2026-09-04 entries are unnumbered) in [`docs/archive/DEVLOG-archive-006.md`](archive/DEVLOG-archive-006.md)
> (rotated 2026-09-04, 2026-09-10, four times on 2026-09-16, on 2026-09-17, 2026-09-18, 2026-09-20, twice on 2026-09-21, twice on 2026-09-22 and on 2026-09-29) ·
> **2026-08-08 (1) → 2026-08-11 (4)** in [`docs/archive/DEVLOG-archive-005.md`](archive/DEVLOG-archive-005.md)
> (rotated 2026-08-30) ·
> **2026-08-05 → 2026-08-07** in [`docs/archive/DEVLOG-archive-004.md`](archive/DEVLOG-archive-004.md)
> (rotated 2026-08-28) · **2026-08-02 → 2026-08-03** in
> [`docs/archive/DEVLOG-archive-003.md`](archive/DEVLOG-archive-003.md) (rotated 2026-08-26 — this
> row was missing until 2026-08-28) · **2026-07-15 → 2026-08-01** in
> [`docs/archive/DEVLOG-archive-002.md`](archive/DEVLOG-archive-002.md)
> (rotated 2026-08-15) · **2026-05-21 → 2026-07-14** in
> [`docs/archive/DEVLOG-archive-001.md`](archive/DEVLOG-archive-001.md) (rotated 2026-07-21).
>
> **The 4,000-line cap in `tests/unit/test_doc_sizes.py` stays as the backstop** (an entry cap cannot
> see an entry that is itself an ADR in disguise). When either trips, rotate — **do not raise
> the cap.** The cap exists because this log reached 8,244 lines before anyone noticed: every entry
> is individually small and correct, so unbounded growth is invisible per commit.

---

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
