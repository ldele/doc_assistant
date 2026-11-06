<!-- status: active · updated: 2026-11-05 · class: append-only -->

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
> **2026-10-24 → 2026-11-02 (2)** in [`docs/archive/DEVLOG-archive-009.md`](archive/DEVLOG-archive-009.md)
> (rotated 2026-10-31: 11 entries, 2026-11-02: 11 more, and 2026-11-05: 11 more) ·
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

## 2026-11-05 (2) — The concept view keeps its graph in sight (ROADMAP 111)

**What changed.** In the Graph tab's concept view (`ConceptGraph.svelte`) the canvas now sits
above the definition card, and nothing in the scrolling column shrinks (`flex-shrink: 0` on its
children). The graph takes the layout's proportions up to `min(55vh, 520px)`. The view box has a
30-unit margin, so a label below a node on the layout's edge shows. The coverage sentence and the
concept's buttons wrap on a narrow pane instead of squeezing. Half of this landed in `54f77f9`,
committed while the fix was being built; this entry covers both halves.

**Why.** The user, 2026-11-05: text and other elements squash the graph until it cannot be seen.
Measured on *BM25* at 1280×800: the canvas was a flex child with `overflow: hidden`, which gives
it no minimum height, so the 827 px definition and the document list took its space and it was
drawn 2 px tall. After the fix: 440 px at 1280×800; at 375 px, 216 px with no sideways scroll.
*hard negatives* sat at y=496 of 520 and its label was cut until the margin.

**Rejected.** A fixed height from `55vh` alone: at 375 px it left a 500 px box around a 216 px
graph. Moving the definition into a tab: a layout decision for the graph re-pass (row 75), not
a fix.

**What it opens.** Pan moves in screen pixels while the graph is drawn in view-box units, so a
drag and the graph move at different speeds when the canvas is scaled; it did before this change,
and it is row 75's.

## 2026-11-05 — Selection in both lists, and a button's look says what it does (ADR-063 proposed)

**What changed.** The chat rail and the Library grid and list share one selection rule
(`apps/desktop/src/lib/shell/selection.ts`, tested): a click or Ctrl/Cmd-click toggles a row, a
Shift-click applies the clicked row's new state to the range from the last row clicked, and a
Shift- or Ctrl-click outside select mode starts it (ROADMAP 78 and 80). Both select bars are
rebuilt to four action roles defined once in `app.css` (`btn`, `btn-primary`, `btn-quiet`,
`btn-danger`): *Select all* / *Select none* replaces the Library's *Clear* and the rail's
*All*/*Clear*; *Done* is the primary at the trailing end of both bars (the rail had none); the
rail's *Delete…* is red from the theme's `--danger` (it used an undefined `--danger-fg` with a
fallback hex) and a *Cancel* on its confirm now keeps the selection. ADR-063 (proposed) records
the roles and GLOSSARY U-003 names them. ROADMAP rows 110 to 112 hold the rest of the user's
batch.

**Why.** The user, 2026-11-05: Shift-click in Chat, Shift- and Ctrl-click in the Library, a
*Clear* that might be a delete, a *Done* that does not read as finishing, one look per kind of
button. In the running app *Clear* only emptied the selection; the rail's *Delete* left select
mode before the confirm was answered, so *Cancel* lost the selection.

**Rejected.** A *Delete* for the Library selection: the user asked to rename *Clear* if it
deletes, and it never did; a bulk document delete still waits on its own decision (the comment
in `App.svelte` says so). Fixing the five unconfirmed destructive actions now (*Remove key*,
*Remove server*, *Remove* and *Remove all* on the ignore list, *Clear* on a definition): they
apply a rule the user has not accepted yet, and their components are the ones row 110 restyles.
Re-styling every button now: one PR per session, and choice 2 of ADR-063 is the user's.

**What it opens.** ADR-063's five choices (the dark-theme primary is the open one), then row 110
(the sweep), 111 (the squashed concept graph: the canvas is 2 px tall with a concept open) and
112 (the library summary, gate first). `docs/decisions.md` needs ADR-063's row; it was left alone
because ADR-062's row is staged there with an unstaged correction on top.

## 2026-11-04 (7) — RG-012's sixth trap: the host must not hold the run's log open (0.7.2 gate)

**What changed.** `docs/desktop-packaging.md` §5 gains trap 6, and `docs/RELEASE.md` §5 a line:
never open `out\run-*\rg012.log` from the host while the sandbox writes it; list the folder until
`turn-3-result.json` appears, then read the log. Both now bind Ollama to the Default Switch
address instead of `0.0.0.0`, with the recipe. The 0.7.2 known limits gain one line (KI-85): an
answer from a server that reports no token counts, such as Ollama on another computer, says
*0 tokens* in its header.

**Why.** The first RG-012 run on the 18:30 installer was watched with `tail -f` on its log; every
later `Add-Content` in the sandbox failed, so the log stopped at *installer chosen* while the three
turns ran and saved their answers. A run's log is never edited, so the gate ran again in a fresh
sandbox and passed (preflight: packaging and citation ok, the cut run listed beside it as a
failure). Ollama's firewall rule allows public networks, so `0.0.0.0` would have opened it to the
Wi-Fi for the length of the run. The `0 tokens` header came from the chat rows driven in the same
sandbox: since ADR-060 an Ollama on another computer is not local, and the header prints the sum
of no counts.

**Rejected.** Reading the log with a polling `grep`: it opens the file too, and a collision loses
a line silently. Fixing the `0 tokens` header before the tag: the line under the answer is right,
and a rebuild re-runs every installed-build gate.

**What it opens.** KI-85's fix (print no count rather than a zero).

## 2026-11-04 (6) — The status bar says when files failed, and stays until dismissed (0.7.2)

**What changed.** A run that ends with failed files reads *Indexing done · 2 failed* (or *· 3 new ·
1 failed*, *Indexing stopped · 16/22 done · 2 failed*) instead of *Indexing done · nothing new*,
its bar turns red, and it stays on the status bar until dismissed, like an error or a stopped run
already did. `failedCount` and `staysUntilDismissed` live in `core/ingest.ts` with the label, so
`node:test` covers them; `IngestChip` and the rune module only read them.

**Why.** Driving the installed 0.7.2 build in Windows Sandbox: two unreadable files failed, the
indexing screen listed them with *Retry the 2 that failed*, and the status bar said *nothing new*;
in the dev build the chip then went quiet after the 8 s linger. Once the screen is closed the bar is
the only trace of the run, so it hid the failures twice over. The user asked for the fix before the
tag. The count is the per-file one (`counts.failed`), the same the screen shows: the run's own
tally (`errors`) counts an empty file as skipped, so it can read 2 where the screen lists 3.

**Checked.** Three new `node:test` cases (361 pass), svelte-check clean; the dev build over a
scratch data home with three failing files: *Indexing done · 3 failed*, red, still there 15 s
later with its dismiss button. Gate 6 of ADR-056, re-run on the copy cleaned of the stale chunks
(entry (5)): every case cites the same sources as the control; the local record holds the numbers.

**Rejected.** Folding the failures into the screen only: the screen is closed most of the time.
An amber bar for partial failure: red is the one failure colour the bar already has.

**What it opens.** Settings' one-line run result still reads the run's tally (*2 errors* beside
*3 failed*); KI-84 (an ingest run builds its own embedding model, ~1.8 GB committed) was filed
from the same walkthrough.

## 2026-11-04 (5) — A changed document is indexed once: `process_document` drops its superseded chunks (0.7.2 blocker)

**What changed.** When a document keeps its identity (ADR-047, by path) but its text, and so its
hash, moves, `process_document` now removes the previous hash's chunks from both stores once the
new chunks and the row are written, unless another row still holds that hash. Before, only the CLI's
full sweep (`cleanup_orphans_*`, run when neither a scope nor a file list is given) or the
per-document re-run's own purge (`reingest._purge_superseded_chunks`) ever did. ADR-056 gate 3's
local test (`tests/integration/test_letter_spacing_library.py`) lands in the same change.

**Why.** Found by ADR-056's gate 6 on a copy of the development library: after the re-extraction
the eval's comparability check reported the index's document count up by two. The two documents
whose text changed kept their old chunks, letter-spaced, with no document row behind them. The app
reaches the same state through *Ingest selected* and *Re-index* (both pass `files=` or `scope=`),
which is how this release's one-time re-read happens, so every user's changed documents would have
been searchable twice. A lie in the release's headline promise: a blocker by the runbook.

**Checked.** `test_a_changed_file_ingested_by_selection_is_indexed_once`; the re-run and ingest
suites (89). Gate 3 passed on the copy against a reference that applies ADR-056 points 1 and 6 (the
two scanned-book places the accepted rule leaves spaced). Gate 6: keywords changed on one document
exactly as predicted, the graph identical; the eval within noise of its control (numbers in the
local record); to be run once more on the copy cleaned of the stale chunks.

**Rejected.** Running the full sweep from the app's routes: it walks every source and re-hashes,
which is what scoping exists to avoid.

**What it opens.** Libraries indexed before this keep any stale chunks until a CLI full ingest; the
copy shows how many a re-read leaves (two documents there).

## 2026-11-04 (4) — ADR-061 built: an indexed library opens where it is, nothing written outside it (0.7.2, piece 5)

**What changed.** The user accepted ADR-061 as proposed, and it is built in its order. **Choice 4
first:** `stored_paths.py` decides whether a stored path may be acted on (cache under the cache
folder, figure images under the figures folder, sources under the library's folders or a
registered root), and `resolve_source_path`, `purge_document_record`, the text and crops re-runs,
the orphan sweep and `load_figure_image_paths` ask it; the viewer says when a file is outside.
**Choice 3:** `library_home.inspect_library` reads a candidate folder read-only (SQLite URI mode),
served at `POST /api/libraries/inspect`. **Choices 2, 7, 9:** the shell keeps `library.json`, sets
`DOC_DATA_DIR` at spawn, and restarts the backend on `open_library`; Settings > Documents > Corpus
offers *Open another library…* and *Use the app's own library* in an installed build only; a
stalled switch offers *Go back*. `config.chroma_base_for` lifts KI-11's rule out of `_chroma_base`
so the check counts chunks where the app would read them. S-7 is done (`docs/security.md`).

**Why.** The user's request of 2026-11-01, and the review's must-hold G1: a copied library's rows
name the original's files, and four writes acted on them unchecked.

**Checked.** `tests/integration/test_stored_paths_confined.py` drives delete, reveal, the viewer,
a text re-run and the orphan sweep over a library whose rows name another folder, and the crops
and figure-route cases sit beside their own tests; each leaves the other folder's files untouched.
`tests/unit/test_library_home.py` covers the check, including that it changes no file. Three
fixtures that kept "library" files outside the library's folders now say where those are. Driven
in the browser over the real API with a stand-in shell: the copied scratch library read as the
ADR's step 4 says, *Open* asked the shell for it, a stalled switch offered *Go back*, and the
pending marker cleared when a backend answered. Rust: `cargo check`, `clippy`, `fmt`.

**Rejected.** Gating the reads that only take text for keywords: on a copy they read the same text.
Refusing every accented library path: one the installed app made opens fine (its stores are
already relocated). Building the installer now to drive the restart: the release builds from the
committed tree anyway.

**What it opens.** The shell's restart is driven in the installed build at the release (piece 6).
`_rerun_text` read `stats["errors"]`, never set, so a failed re-extraction reported success; fixed
here with a test. Bringing a copy in and libraries from another computer stay open (ADR-061).

## 2026-11-04 (3) — The release check sees every build input and every chat reply (review N1, KI-83)

**What changed.** `scripts/release_preflight.py`: `SHIPPED_PATHS` gains `package-lock.json`,
`public/`, `svelte.config.js`, `tsconfig.json`, Tauri's `capabilities/` and `nsis/`, and
`apps/__init__.py` and `apps/cli.py` (the spec freezes every submodule of `apps`). A new
`NOT_SHIPPED` names each other tracked file under `apps/` with its reason (`Cargo.lock`, the icon
sources, the notes), and `unclassified_app_files` lists any file in neither. `check_dev_commands`
reads `.ts` files and the string literals of `commands.py` (docstrings aside), with `python -m` and
`scripts/….py`. Six chat replies in `commands.py` that named a script or `.env` now name the app's
control, or say the pass is not in the app yet.

**Why.** Both checks are release criteria for 0.7.2 (plan piece 6). The DOMPurify fix touched only
the npm lock, which was in no list, so an older installer read as fresh; and the replies sent an
installed-app user to a `scripts/` folder they do not have, past a check that read `.svelte` only.

**Checked.** A test that every tracked file under `apps/` is in one list or the other (it is today);
fake trees that plant a `python -m` reply, a `scripts/….py` reply and a `.ts` string and fail,
while docstrings, comments and test files pass; the replies' own tests, rewritten to pin that no
command is named.

**Rejected.** Reading `.ts` comments as well: explaining a command to a maintainer is fine.
Shipping `Cargo.lock`: cargo rewrites it during every build (the constant's own comment).

**What it opens.** Similarity (`compute_doc_vectors`) has no in-app control; the reply now says so.

## 2026-11-04 (2) — The indexing screen: each file's outcome, Pause and Stop, a Re-index estimate (0.7.2, piece 4)

**What changed.** ROADMAP 31 (Track B's per-file outcome and retry), from the wireframe approved at
rev 858 (`docs/wireframes/indexing.md`). A new `ingest/run.py` holds one run as the app sees it:
each file's state (*waiting*, *reading*, *added*, *unchanged*, *failed*, *not started*) with what
happened, and the controls. `ingest.main` takes an optional `run`, records each document on it, and
asks it between two documents whether to wait or to end; `process_document` returns a
`DocumentResult` (the old tally, the passage count, a failure's reason) and `process_one_document`
keeps its string for its callers. A stop leaves the loop, so the post-run steps and the controller
reload still run; the extraction warm-up cancels what it has not started. Routes: pause, resume,
stop, a paged `GET /api/ingest/files`, `POST /api/ingest {rerun: failed|remaining}` from the
server's own record of the last run, `GET /api/ingest/plan`, and `{only_new: true}`. The status
gains `control` and `counts`, and a `stopped` state. The screen (`shell/IndexingScreen.svelte`)
opens from the status bar and from Settings' *View indexing*. Three changes by the user's word
during the build: the status bar reads *Indexing… 14/30 done*; Stop is red and asks first (the
wireframe had no confirmation), Pause amber; and Re-index opens a confirmation with the counts, an
estimate and *Only index new files* (`ingest/estimate.py`, `settings/ReindexConfirm.svelte`).

**Why.** A run showed only its position, and a failure's reason reached the log alone. The user
asked for the list, pause and stop on 2026-11-01, and for the rest while it was driven.

**Checked.** Unit tests for the run object (two real threads for pause and stop); a real
`ingest.main` over temp files: stop, pause and resume, a failed file's reason, a second run reading
*unchanged*, a follow-up run over the rest; route tests for every control, the page, re-runs, the
plan and `only_new`. Driven in the app on a new data home (`C:/Projects/pv-idx`, generated files, a
dummy key): paused before and during a run, stopped while paused, *Index the remaining*, *Retry*,
the Stop question and its Escape, the Re-index question before and after a run measured its speed;
light, dark, 375 px, no failed request since the page load. Pause was asked at 14 done and took
effect after the 15th, read in the server log.

**Rejected.** Discarding the file in flight on Pause (the user asked to check): extraction, 14.7 s
mean cold (`docs/performance.md`), is one library call that cannot be interrupted in-process, so
dropping the file would save only its embedding, about 11 s on the shipped CPU build; the wait is
named instead. A per-document constant for the estimate: the speed is measured per machine, and
none is shown before a run measures it. Sending the follow-up runs' paths from the client: the
server already holds them, and a paused run would otherwise ship 10,000 paths per click.

**What it opens.** The run's record is in memory: a restart forgets the last run's list. A document
recorded with no passages shows as failed but counts as added in the tally. Row 77 (parallel
extraction in the frozen build) is still to observe on the installed build at the release.

## 2026-11-04 — The test suite sees CI's provider environment; `main` was red since ADR-060

**What changed.** `tests/conftest.py` gains an autouse fixture that clears `config.ANTHROPIC_API_KEY`
and unpins `REVIEWER_PROVIDER`, as CI has them. `test_compatible_turn.py` reads the answer's request
and then the review's: both passages come from one paper, so the answer is flagged and the reviewer
follows the chat provider to the same server (ADR-011).

**Why.** CI failed six tests on `afb3313` and `20d52db` (`too many values to unpack`) that passed
here. Here `.env` pins the reviewer to Anthropic, so the fake server saw one request; and with the
key in `.env`, that whole-turn test built a real Anthropic client and called it on each flagged
answer, a billed call from the suite (KI-4's shape).

**Rejected.** Pinning the reviewer in the test alone: the next whole-turn test would meet the same
`.env`. Asserting one request: the review is the behaviour ADR-011 decided.

**What it opens.** Nothing pins that a test cannot reach a paid API in general; this closes the one
door found.

## 2026-11-02 (4) — The project review's routing applied to the tracked docs; archived paths re-pointed

**What changed.** The tracked half of the 2026-11-02 project review (its §8), checked against the code
first. `docs/decisions.md` indexes ADR-058 and ADR-059, which no index, row or plan cited.
`docs/ROADMAP.md`: row 109 (the installed app makes its own derived layers, `gate first`), row 106's
spec names ADR-059 and ADR-058, and five rows' paths point at `docs/archive/local/`.
`docs/security.md`: §2 says what the gates really cover (`scripts/` is not type-checked or scanned;
pre-commit's mypy and bandit see `src/` only), that the f-string SQL includes one `INSERT…SELECT`,
and that S-6's rule by method holds only while no GET has a side effect; S9 is widened to *reveal*,
*delete*, `purge_document_record` and `/api/figures/{id}`; new findings S15 (three GETs with side
effects), S16 (`Cargo.lock` and the packaging extra unaudited) and S17 (CI's uv floats, no
`permissions:`), with steps S-15 and S-16; S-7 moves into 0.7.2's piece 5; §5 no longer names a test
file as if it existed. `knowledge-layer.md`, `local-only.md` and `specs/feature-add-documents.md`
cite the archived plans at their new path, and six ADRs carry an appended note instead (append-only).

**Why.** A review the docs do not carry is lost at the next session. And the archived plans were
still cited at paths that no longer exist.

**Checked.** Six of the review's claims re-read in the code before any text was written: the delete
path that unlinks a stored cache path, the dead keyword index at 0 documents, the developer commands
in chat replies, the unindexed ADRs, the release check's file list, and the model-list route now a
POST. Each security line checked against `ci.yml`, `.pre-commit-config.yaml`, `migrations.py`,
`launch_token.py`, `routers/chat.py` and `library/add.py`; 445 packages in `Cargo.lock`. No moved path
is cited unannotated in a tracked doc (a grep: the docs gate reads backticked paths in `docs/**` by
markdown links only, the review's N12).

**Rejected.** Editing the ADRs' own lines: they are append-only, so each gains a note. Putting the
review's other additions to 0.7.2 into the tracked docs: they are plan decisions, and they live in the
local plan.

**What it opens.** S-7 in piece 5, with the review's must-hold that no write over an opened library
touches a path outside its data home. Row 109 for release B. Steps S-15 and S-16.

## 2026-11-02 (3) — ADR-060 built: other answer providers through one compatible client (0.7.2, piece 3)

**What changed.** New `providers.py`: the six presets of the wireframe as rows (LM Studio, OpenAI,
Google Gemini, Mistral, Groq, OpenRouter), servers of your own in `settings.json`, where the text
goes per model (`route_of`, with Ollama's `remote_host`), who bills (`Billing`: a vendor, nobody, not
known, or the user's word), model lists kept in memory, and the prompt-cut note. New
`compatible.py`: our own client over `httpx`, sending `model`, `messages` and `stream` and nothing
else, reading only `delta.content`, with a LangChain chat model for answers and an `LLMClient` for
reviews. Keys for these providers are bound to their address in `credentials.json` (`keys`) and never
read from the environment. `config.PAID_PROVIDERS` is gone: the CLI guard trips on billed or not known
and names the host and who bills. `build_chat_model` and `make_client` raise on an unknown id instead
of answering on Ollama. Routes keyed by provider (`/api/setup/providers/{id}/key|models|billing`,
`/api/setup/servers`) replace `/api/setup/anthropic-key`. The answer record gains
`answer_route_json` (additive), and the cost line says who bills instead of pricing every turn at
Haiku's rates. Provider & model draws one form for presets and servers: *Test connection*, the
*This server does not bill me* tick, the address-change warning, a model box over the server's list.

**Why.** The user, 2026-11-01: more options than Claude and Ollama, by the norm. ADR-060 and its
amendment are the contract. OpenAI's address and Mistral's model-list path were confirmed against
the vendors' docs before their rows went in; OpenRouter's model list answers without a key (checked),
so its row checks a key on `GET /key`.

**Proved at $0.** Fake-server tests (`tests/fixtures/compatible_server.py`): the body, the stream
reader on recorded streams, model lists on every status, the address and billing tables, every preset
row through a key-guarded route, and a whole turn through the real `ChatController` with provenance
and the cost line. Live, on a new data home holding two public RAG papers: a server of your own at
Ollama's `http://localhost:11434/v1` answered the ADR's question on `llama3.1:8b` with a valid
citation; the card read *stay on this computer* and *Billing not known*, then *Not billed (your
word)* once ticked. `qwen3.5:9b` through `/v1` keeps its reasoning in a separate field (nothing
inline, so no known issue), and its first visible token came after 32 s warm. No paid call was made.

**Departures from the ADR.** `*.localhost` is not *this computer*: the OS resolves it, and
`lab.localhost` did not resolve on this machine, so the sentence could not promise it. The models
route is a POST, so the launch token covers a call that sends a key (the whole-project review's
point). Local probes connect within 0.5 s: Windows retries a port nobody listens on, and LM Studio's
probe took 4 s.

**Rejected.** `langchain-openai` (ADR-060's Options). Returning an unprobed setup state from the
billing tick to make it faster: an unprobed Ollama row reads *Ready*.

**What it opens.** The RG-012 turn in the frozen build (test 9) belongs to the release. A paid message
per preset waits for the user's word. `remote_host` is pinned by a test on the documented shape only;
a real Ollama cloud model needs a sign-in. On Windows, `localhost` tries `::1` first, so a server
listening on IPv4 only costs about 2.2 s per connection: Ollama's own probe takes 2.38 s at
`localhost` and 0.22 s at `127.0.0.1` (older than this change). The status bar shows a server's id,
not its name.
