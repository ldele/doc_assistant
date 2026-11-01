<!-- status: active · updated: 2026-11-01 · class: append-only -->

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
> **2026-10-24 → 2026-10-27 (3)** in [`docs/archive/DEVLOG-archive-009.md`](archive/DEVLOG-archive-009.md)
> (rotated 2026-10-31: 11 entries) ·
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

## 2026-11-01 (3) — Wireframes approved: indexing, Getting started, Provider & model

**What changed.** `docs/wireframes/indexing.excalidraw` and
`docs/wireframes/getting-started-and-providers.excalidraw`, each with its reading in an `.md` beside
it. No code.

**Why.** The drawing comes before the code. The indexing screen lists every file with its status and
adds pause and stop. Getting started and Provider & model overlapped: both held provider setup, so a
first run had two places to look.

**Decided in the drawings.** Getting started is two rows, Model and Documents, each linking to its
tab; ready means both green, with the model, *Engine online* and the documents indexed. Provider &
model holds everything about answer models and says where the text goes. *Stop* has no confirmation,
because the next run skips what finished.

**Rejected.** Warning rows and extra buttons in Getting started: the user wants other options only if
they are needed. A confirmation on *Stop*: it loses nothing. One adapter per vendor: most providers now
document the same OpenAI-compatible interface, so presets are data over one adapter.

**Checked.** Indexing at rev 858 and Getting started and Provider & model at rev 176, each
`fallbacks 0 · inferred 0 · unnamedScreens 0 · orphans 0`, both approved by the user with no markup on
the canvas.

**What it opens.** An ADR for the adapter, extending ADR-011 and ADR-034. Per-file status, pause and
stop in the backend. Nothing here is built.

## 2026-11-01 (2) — Settings closes during a run (0.7.2, piece 1)

**What changed.** `Settings.svelte` and `Sources.svelte` drop their private `pollUntilDone` and read
the shared `ingestRun`. `busy` is derived in both: `indexLocked(requesting, status)` in
`core/ingest.ts`, true while a panel's own request is out or any run is in flight, so the folder
input, Browse and a second run stay guarded. Escape, the backdrop and the close button no longer
read it. `useRunEnd` in `core/ingest.svelte.ts` tells a mounted panel a run ended. Both panels call
`noteIngestStarted`, which now takes the run's folder. `initialSection` opens Documents during a
run. Opened mid-run, focus goes to the panel, because the input is disabled. The sentence now says
Settings can be closed. Settings reports `lostContact`, which used to come from its own poll. Tests:
`indexLocked` and the landing section, 331 in all.

**Why.** The user's report: once a run starts, Settings cannot be closed. `busy` covered all of
`indexFolder`, which awaited the poll. Freeing the three routes alone would have left three faults,
all gone by reading one run. A reopened panel held no run, so it showed an idle form that offered a
second run the API refuses with a 409. A run shorter than the 4 s idle poll ended unseen, because
Settings never told the watcher, and with the panel closed nothing refreshed the library. A first
run reopened on Getting started, which does not show it. `onCorpusChanged` no longer carries ingest
completion: App's `completedRuns` effect refreshes the library and chunk count with or without
Settings.

**Rejected.** Keeping both polls and adopting the run on mount: a second poller per panel, the
disagreement `core/ingest.svelte.ts` exists to prevent, and a run from the other panel would stay
unseen. A "started here" flag so Sources keeps its own lines: it forgets on reopen, and its result
would sit beside Settings' identical one. Sources sits directly under Settings' line, and its own
line was below the file list, often off screen, so it goes.

**Checked.** `npm run check` 0 errors; `npm test` 331 passed. Driven in the app in a browser preview
over a new data home and generated documents, four runs: 60 documents started with Enter, 240 with
a click on *Re-index*, one selected file closed at once, one started by a direct `POST /api/ingest`
with Settings left open. Escape, the backdrop and the close button each closed Settings mid-run at
1280 px in dark; Escape again in light, the close button again at 375 px. Escape works with focus on
the control that has just been disabled. The chip kept counting, and reopening showed the run on
Documents with focus on the panel. With Settings closed, the chunk count, the library and the setup
card refreshed. The outside run locked the open panel within one idle poll and ended in its result
line with fresh corpus numbers. No console error came from the app, and the real `data/` kept its
10,080 files at the same size and mtime. A new data home rather than a copy of the library because a
Re-index is a full ingest: `cleanup_orphans_chroma(also_clean_cache=True)` unlinks the cache paths
stored in Chroma, which on a copy name the real files. **Not driven:** the lost-contact line, Browse
(a browser does not render it), and the packaged build.

**What it opens.** Piece 3 replaces this line with the indexing screen, and `useRunEnd` is what it
needs for a result. A status poll already in flight when the POST lands can read the previous
`done` after `noteIngestStarted`, which fires a harmless extra refresh: the window is milliseconds,
the next poll corrects it, and the add sheet has the same gap. While Settings is open it covers the
chip on a wide window, so the count is only visible once it closes.

## 2026-11-01 — Letter-spaced text closes at extraction, with a record and a control (ADR-056, ROADMAP 108)

**What changed.** Built 2026-10-23 and 2026-10-24 in a side session and parked through the 0.7
release; this entry is the account of all of it, and of what landing changed.

*The rule.* `extractors.py`: `letter_spacing_joins` reads one PyMuPDF `rawdict` line for runs of
one-glyph tokens whose spaces the page does not show (the spaces PyMuPDF added inside letter-spaced
text, or space glyphs between letters that touch); `apply_letter_spacing` writes each run into the
page's markdown; `extract_pdf_pymupdf` runs both per page and skips a page whose glyphs cannot be
read. It leaves alone a page one image covers (`_SCANNED_PAGE_MIN`), any run in a formula, picture or
table box, and maths; a URL closes as kind `url`. *The ignore list.* `config.REJOIN_IGNORE_PATH`
(`rejoin_ignore.json` in the data home): per PDF, by sha256 with its size, the runs to keep, `*`, and
the user's `spacing`, applied only when it moves spaces alone. `document_fingerprint` adds a
document's entries to its cache fingerprint, so an edit re-extracts that document alone. *The
record.* `write_cache` writes `<cache>.md.spacing.json` between the markdown and the fingerprint,
each run with its state and its span in the text; a document without runs has no file. *The control.*
`library/spacing.py`, `/api/library/documents/{id}/spacing` (+ `keep`, `edit`, `remove`, `keep-all`),
`/api/library/spacing` (+ `remove`, `restore`), `?clip=` on the page route; `DocSpacing.svelte`,
`SpacingEditor.svelte`, `SpacingIgnoreList.svelte` and the pure `spacing.ts`; the re-run picker takes
parts to tick on open. *The CLI.* `scripts/list_letter_spacing.py`, read-only.

*Found by reading it again as new, and fixed, each with a test* (ADR-056, amendment 2026-11-01): a
bold or italic run never closed, because the pattern wanted whitespace where markdown writes `**` or
`_`; a kept or maths run recorded a form the text does not hold, so it had no span; the block did
not follow a finished re-run; `{"documents": null}` raised in the reader every freshness check
calls, and an entry of the wrong shape was a 500; a removed document left its record; `list_entries`
read every document's row and, like the CLI, missed a moved file; Settings could say "the 0
documents' text".

**Why.** ADR-056: a space the PDF does not hold, or holds with no visible width, is this project's
reading error, not the document's content, so it is fixed where it is made (as KI-64 was). The user
asked for points 7, 9 and 10 with the control from the sketch. Point 8 (keywords) waits for ADR-055,
and its span is in the record now because the record must be right before the library is
re-extracted.

**Checked.** The full suite, 2,937 passed; ruff, `mypy src` and CI's `mypy src/ apps/ --platform linux` clean; svelte-check 0 errors and 0 warnings, the 327 frontend tests pass. Gate 2 on the final rule, all 104 documents, against a copy of the live cache:
102 byte-identical, 2 changed by deleted spaces only (`cajal-lecture` 479, `s41598-025-33686-0` 306), no errors, as at every earlier run (those two compared with the live cache itself, the drive having re-extracted their copies). 36 runs recorded in 4 documents, 33 of them in the text, each at a distinct span that holds its text. The bold and italic fix moves nothing here: the same 3 runs still never reach the text.
Driven in the app on a scratch copy with every stored path pointed at it (`GET /api/settings` read
first, and a before/after listing of the real `data/`: unchanged): a document from before this
change shows no block; a real *Text & chunks* re-run wrote the text and the record, and the block
appeared without reopening the document; *Keep spaced*, *Edit spacing* and Save, *Show on page*,
*Keep every run spaced*, *All entries in Settings*, *Show runs*, *Remove all* and *Undo*; a second
re-run from *Re-run text now…*, which opens with *Text & chunks* ticked, wrote the user's spacing into
the text and kept the running head spaced, only spaces differing from the live cache. Dark, light and
375 px; no failed request. Earlier, in the side session: 48 joins reviewed by the user (33 places: 6
close, 27 not text, 0 keep spaced), which the four context rules reproduce on 31 of 33 decisions.

**Rejected.** The rule in a module of its own: the fingerprint walks `extractors` only, so a change
to it would re-extract nothing (KI-40). A set for the closing marks: its repr order follows the hash
seed, and the fingerprint hashes the repr. Skipping page headers: both letter-spaced runs of the
document that asked for this sit in header boxes. A threshold under that document's tracking for the
label-row gray zone: fitted to one library. `scrollIntoView` for *All entries in Settings*: it moves
every scrolling ancestor. An empty record file per document: files that say nothing.

**Not done, on purpose.** The count per document in `ingestion_events.notes` (ADR-056, point 3): the
store writes the event from the text alone, so the ingest log carries it. `data/rejoin_ignore.json`
is not written: this library's list starts empty. The library is not re-extracted: that is the
user's, with its backups, rebuilds and the private eval.

**Costs once.** The ignore entries go through `extract_to_markdown`, so every format's fingerprint
moves (checked on all six), and the first full ingest after this reads every document, about 24
minutes here. A document whose text comes out the same is not embedded again.

**What it opens.** ROADMAP 108: the re-extraction (gates 3 and 6), more letter-spaced PDFs (gate 4,
each download the user's call), and point 8 after ADR-055. Re-extracting before point 8 brings the
running head's keyword on the next keyword rebuild. The other document blocks (figures, references)
read their document once too, so they may also lag a finished re-run until it is reopened; not
checked here.

## 2026-10-31 (6) — The roadmap cleaned up; the order set for the next moves

**What changed.** `docs/ROADMAP.md` went from 90 rows to 86. Ten finished rows moved verbatim to
`docs/archive/ROADMAP-done-002.md`, 12 long rows were cut to what is left with their old text kept
there, and rows 101–106 were added. Row 65, the MCP server, is unparked and sits in F3. The table is
sorted by feature. Statuses were checked against the code: rows 34 and 86 closed, EX1's gate has been
closed since 2026-08-08, row 43 waits on KL3 only. `AGENTS.md` and `docs/OVERVIEW.md` §5 say 0.7.1 is
published.

**Why.** The user's order: the short-term work, then MCP and the chat, then extraction, the Library
and every instance of what the user chose. The file had also outgrown its own rule that a row's why
lives in its spec: one row ran to 6,145 characters.

**Rejected.** Release entries in the ROADMAP: sessions and release criteria stay in a local plan.
Scrubbing the archived rows of the private library's figures: an archive is verbatim (ADR-043).

**Opens.** ADR-055's parked patch takes row 107. KL2 and row 25 move behind both releases, unless
row 106's spec needs row 25.

---

## 2026-10-31 (5) — Preflight reads true after the tag

**What changed.** `check_checklists_refreshed` takes the version and leaves the release's own tag
out of `git describe`, with three tests. On this repository at `v0.7.1` it now compares against
`v0.7.0` and passes. `docs/RELEASE.md` §6 says so.

**Why.** Once the tag existed, the check compared the release with itself and failed, which is the
false alarm of v0.7.0.

**Rejected.** Keeping the runbook's "pre-tag only" rule: a check that lies after the tag is a trap
for whoever reads it there.

**Opens.** Nothing.

---

## 2026-10-31 (4) — Settings' Ollama card states its trade-off in words

**What changed.** *Expect fewer citations* in `ProviderSetup.svelte` no longer quotes the run's size
or its rates. It says local models carry fewer inline citations than Claude Haiku 4.5, and that this
predates 0.7.0's fix. The comment above it says why there are no figures.

**Why.** It was the one sentence the app shows that came from the private library. It stays in the
0.7.1 installer, and this change ships with the next build, by the user's choice over a 0.7.2.

**Rejected.** Deleting the warning: a local user still meets many *uncited* claims and needs the
reason where the engine is chosen.

**Opens.** A re-measure for KI-36 can no longer go on the card as figures from that library.

---

## 2026-10-31 (3) — 0.7.1 published; its notes say nothing about the private library

**What changed.** 0.7.1 is published, with the installer from `83c9af1`. The release notes and the
CHANGELOG's 0.7.0 section lost every figure, name and example taken from the private library. Each
limit stays, in words, and the notes' CHANGELOG link now reads `main`.

**Why.** The library is private, so a release says nothing about it. The clean-machine gate's three
arXiv papers are public and its numbers stay.

**Rejected.** Moving `v0.7.1` so its own CHANGELOG matches: a pushed tag is not moved
(`docs/RELEASE.md` §6), so that copy keeps the old text.

**Opens.** Settings' Ollama card still names the size of that library and its citation rates, in
the published installer too. Older CHANGELOG sections, the README and `docs/OVERVIEW.md` were left.

---

## 2026-10-31 (2) — 0.7.1's installer passes every upgrade route and its gate

**What changed.** No code. Installer built from `83c9af1` (15:35). RG-012 `run-20261031-160140`
passed and preflight is all green. Over the real 0.6.0 with its engine left running, silently and by
double-click, nothing of the old build stayed, the app was up in 13 s and every record matched. It
also held silently with 0.6.0's window open and over a running 0.7.1. `docs/RELEASE.md` §6 now says
preflight is a pre-tag check and a pushed tag is not moved.

**Why.** A double-click runs a different path from `/S`: the reinstall page calls 0.6.0's own
uninstaller first, which reports success and leaves the locked file. The hook had to clear that too,
and only driving the wizard could show it. The two runbook lines come from this release: preflight
failed falsely after `v0.7.0` was pushed, and that tag stayed.

**Rejected.** Repeating the walkthrough on this build: against `4a22951` the code changes only the
version and the hook, and the notes say which build each row stands on.

**Opens.** `release_preflight.py:691` could compare with the tag before HEAD's, which would end that
false failure; left for after the release, so the tag's code stays the tested code.

## 2026-10-31 — 0.7.1: the installer clears what 0.6.0 left (KI-80)

**What changed.** New `apps/desktop/src-tauri/nsis/hooks.nsh`, wired as `installerHooks`: after
files are copied it ends `doc-assistant-api.exe`, deletes the old file of that name from the install
root, and removes `%TEMP%\_MEI*` folders holding our reranker model. Version 0.7.1 in all nine
places. `docs/release-ux-checklist.md` §7 gains an upgrade row, and §2's "never sent to the bin"
now says what ADR-046 does: the bin only by choice.

**Why.** Driving the upgrade after `v0.7.0` was pushed, before anything was published: 0.6.0's
engine outlives its window, the 0.7.0 installer finished with exit 0 around it, and the app stayed on
"still starting" for over four minutes with a hint that cannot help. The library was untouched. Both
engine files stayed, 1.5 GB at the root, and each force-ended engine left 2.4 GB in `%TEMP%`, a
restart included, so the note the notes first carried ("restart Windows first") left all of that
behind. RG-012 installs fresh and the checklist had no upgrade row, so nothing could have seen it.

**Rejected.** PREINSTALL, the obvious hook: it runs before `CheckIfAppIsRunning`, so it could end the
engine under a window the user then declines to close. A second template change: the note on
`installer.nsi` promises one line, and a hook survives re-taking Tauri's template. Deleting every
`_MEI*`: any PyInstaller program uses that name, so the hook proves a folder is ours first. Moving the
pushed `v0.7.0` tag: 0.7.1 costs one more backend build and rewrites nothing public.

**Opens.** A `v0.7.0` tag with no release stays in the list. The hook ends the engine by name, so a
frozen build run by hand from `dist/` goes too; acceptable on the one machine that does that.

## 2026-10-30 (6) — The rebuilt installer passes its gate; a stage must keep the build's time

**What changed.** Installer rebuilt from `4a22951` (19:57). RG-012 `run-20261030-200429` passed,
preflight is all green, and KI-74 to KI-78, the rail, *Re-run* and both status-bar rows were driven
again on the installed build and held. `docs/RELEASE.md` §5 now says the staged copy keeps the
build's time. `CHANGELOG.md` carries the release date.

**Why.** Entries (3) to (5) left those fixes unproved on an installed build. The stage rule comes
from this run: a plain `cp` stamped the copy 19:58, the harness logged that, and preflight rightly
failed a real PASS as "a DIFFERENT build".

**Rejected.** Editing the run's log to the build's minute, or restamping the artifact to fit it:
either would make preflight's link say something nobody measured. Setting the copy's time and
running the gate again cost five minutes.

**Opens.** Three rows stay with the user (a wrong key, the larger text, unticking in the add
sheet). §2 to §6 rows that passed on `d8a6b3c` were not repeated on this build.

## 2026-10-30 (5) — Keyword search survives an add and forgets a delete (KI-77, KI-78); rail and Re-run fit

**What changed.** `ingest_start` now calls `release_keyword_index` on the old controller before
`controller_factory()`; `RAGPipeline.release_sparse_index` closes the handle and clears `_sparse`.
`SparseIndex.forget` keeps a set of removed `doc_hash` values that `search` and `doc_hashes` leave
out, and `purge_document_record` takes a `forget` callback that delete and undo pass from
`ChatController.forget_document`. In the desktop app: the concept rail's badge takes its own line,
the re-run dialog's actions are pinned to the sheet's bottom edge, and a delete refreshes the count.

**Why.** The 0.7.0 walkthrough, sections 2 to 6, on the installed `d8a6b3c` build over the 104-document
copy. KI-77: Settings showed the keyword index "unavailable". *Rebuild* cured it, and one added note
moved `/api/settings` from `on_disk` back to `unavailable`. The new controller builds its index while
the old one holds the file open, Windows refuses the `path.unlink` (`PermissionError [WinError 32]`,
reproduced alone), and the failure is swallowed into `keyword_index_unavailable`, which only Settings
reports. Every add to a non-empty library lost keyword matching until a relaunch, since `26ff0990`
(2026-07-30). An add from empty has no old handle, so RG-012 and first-run walks never saw it. KI-78:
after *Remove from library*, with the library at 104 documents, "What is walkite-001?" still answered
from the removed note and cited it, because a delete reached Chroma and never the keyword index. The
rail showed "B…", "c…", "d…" at 1024×768. *Re-run* was cut by the sheet's edge, and a click on its
lower half reached the backdrop, closed the dialog and ran nothing, twice in a row.

**Checked.** Unit and API tests for each: forgetting (search, scope, `doc_hashes`), releasing then
replacing the file, the order of release and factory, the delete and undo wiring. The full suite
2,842 passed, 9 of them new. The dev app at 1024×768 over the library copy: rail names whole, *Re-run* inside the
sheet by its measured box, nothing run. **Not checked yet:** both index fixes in the installed app,
which needs the rebuild, and that is the check that matters: add, read `keyword_index`; delete, ask.

**Rejected.** Writing the delete into the index file: the answer path opens it read-only and a moved
corpus rebuilds it at the next launch anyway. Rebuilding it on every delete: tens of seconds on the
request, or a background job for bulk deletes to race. Filtering hits against the `documents` table at
query time: a database read on every turn, and a dependency this module has never had. A unique file
name per build: disk and cleanup, to work around one ordering mistake. A chat notice when the arm is
down: right, and separate; with this the state should be rare.

**What it opens.** A `text` re-run that changes a document's hash leaves its old chunks in the keyword
index until a relaunch: not fixed, not driven. The walkthrough's other findings are in KI-79 and the
CHANGELOG's *Known limits*.

## 2026-10-30 (4) — The add review warns about a scan and catches the same file twice (KI-76)

**What changed.** `library/add.py`, in `inspect`. A new PDF with no text on five pages spread through
it stays an `add`, stays ticked, and carries an advisory saying it looks like a scan and nothing in
it can be searched. No warning where `tesseract` is on `PATH`, because the extractor then OCRs the
page (KI-47). A file with the same bytes as an earlier one of the selection is a `duplicate` naming
it. `sort_for_review` puts a warned add above the clean ones, so "and N more" still means clean
files. `AddDocuments.svelte` colours the warned row. No wire change: `advisory` already existed.

**Why.** The 0.7.0 walkthrough, section 1, on the installed `d8a6b3c` build. Both scans of the test
folder read as clean rows, then "Added", "0 errors", 0 chunks; the card said "0 chunks" and the
document page "· broken", with no reason. The spec had drawn this row since 2026-08 ("ticked,
warned"), gated on a cost measurement (RG-030) nobody ran, and the checklist went on asking for it.
The same bytes twice read "Added 8", ingest skipped the second as unchanged, and seven documents
appeared. The user chose both before the tag.

**Checked.** Measured for RG-030 on the 104 PDFs of the real library: 1.9 s, 18 ms each (median 15,
max 78), one PDF flagged, `middleton-2001.pdf`, the known scan, so no false alarm. A review opens
each new PDF, so a 10,000-PDF add would spend about three minutes on this before the sheet shows,
and only with no OCR engine. The real `inspect` over the 11-file test folder with `tesseract` hidden:
both scans warned and ticked, the twin refused by name. `test_library_add.py` and the inspect API
tests, 76 passed; the full suite 2,833 passed, 10 of them new. **Not checked yet:** the warned row on screen, which needs the desktop
window and so the rebuilt installer.

**Rejected.** A fifth verdict: a scan can still be added, and every `add` rule already applies. A
page-one probe as the spec drew it: a blank or image-only first page would call a whole paper a
scan, and a sample through the file costs little more. Refusing the file: inform, don't block. Reading OCR availability
from anything but `PATH`: it would disagree with what the extractor does. Hashing every file of a
selection: only same sizes can match, the rule the library already follows.

**What it opens.** A scan already in the library still says only "· broken" on its page. One page
of text in the sample clears a file, so a scan with a typed cover gets no warning. Two different scans with the same page count extract to the same
page markers, so by `doc_hash` the second may read as unchanged: not tried.

## 2026-10-30 (3) — The setup card and the chunk count follow an add (KI-75)

**What changed.** `App.svelte`: the run-finished effect calls `refreshHealth` instead of reading
`/api/health` itself, and the add sheet's `onAdded` calls it too. `refreshHealth` also re-reads
setup. `getHealth` stays where the readiness gate needs it.

**Why.** The 0.7.0 walkthrough, section 1, driven on the installed `d8a6b3c` build. With the library
at 281 chunks and answering with citations, the chat still said "One step to go — Add your
documents — Nothing indexed yet", with *Finish setup…*, until a relaunch: every new user's first
add. The status bar stayed a green "565 chunks" 60 s after *Undo all*, while `/api/health` said 0.
`onAdded` refreshed the documents only.

**Checked.** `svelte-check` 0 errors, 317 desktop tests. The dev app over an empty scratch data
folder: "One step to go", then `POST /api/ingest` on one note, and the card gave way to the
sample questions with "1 chunks" in the status bar; its eight console 500s were the dev proxy
before the API was up (ECONNREFUSED in the Vite log). **Not checked yet:** the status bar after
*Undo all*, which needs the add sheet and so the desktop window; the rebuilt installer covers it.

**Rejected.** A separate `refreshSetup` call in each place: `refreshHealth` exists for exactly this
and its comment says so.

**What it opens.** The same walkthrough found two gaps this leaves open: a scanned PDF is not warned
about before indexing, and the same bytes twice in one batch read as two adds.

## 2026-10-30 (2) — `langgraph-sdk` 0.4.6: an advisory published during the release

**What changed.** `uv.lock`: `langgraph-sdk` 0.4.5 → 0.4.6 (`uv lock --upgrade-package
'langgraph-sdk==0.4.6'`); nothing else moved. **Found by:** the pip-audit gate, run here before
`841fc75`: CVE-2026-118667 (GHSA-3fx2-cqw9-xr3c). It had passed on 2026-10-27.

**Why.** It ships: `collect_all("langchain")` in `scripts/doc_assistant_api.spec` takes `langgraph`
and this SDK into the frozen backend (84 entries in the last build's `PYZ-00.toc`). The flaw is in
the SDK's v3 streaming client, which talks to a remote LangGraph server. This app has none: `src/`
never imports `langgraph`, and neither package is loaded when the API starts. A patch release was
still cheaper than a reviewed ignore. 0.4.6 is the first fixed version, 24 days old.

**Checked.** The gate is OK (197 packages, 0 unreviewed), the whole suite passes (2,823), and the
venv moved one package. The app never loads the SDK, so no run here exercises the new version.

**Correction to the entry below** (this file is append-only): its *Checked* line says 8 tests "new
or extended"; the true count is 7 new cases and 3 extended tests.

**What it opens.** The eighth advisory round since 2026-10-20. The release backend must be frozen
from this commit or later.

## 2026-10-30 — A turn that retrieves nothing is answered without a model (KI-74); About shows the version

**What changed.** `_handle_rag` now stops at `if not scored:` and returns `_no_sources_result`: one
sentence from `_no_sources_answer` (`chat_controller/helpers.py`), for four cases: an empty library,
nothing matched, an empty folder, a folder that no longer exists. No model and no reviewer run. The
turn is recorded, because the history sidebar is a read over those records, and it stays out of
`session.history`, so a second question has nothing to be rewritten against. Its usage is a real
zero (`0 tokens · $0.0000`, not "tokens not reported") and the *Provenance & usage* card says no
model was asked. Human-mode turns keep their own path. `/api/health` gains `version`, which About
shows first.

**Why.** The 0.7.0 walkthrough, driven at the user's word on the installed `d8a6b3c` build: at 0
chunks, `llama3.1:8b` answered that the library "does describe a technique where "a random subset of
units is not updated during backpropagation" [4]". Nothing was retrieved: Ollama's log shows a
348-token prompt, the instructions and the question alone. The user caught it on screen. The
citation check flagged the answer as 0 valid, but beside a sentence already standing as the answer.
`feature-corpus-folders-scope.md` S3 had recorded "a zero-source turn already degrades honestly",
which was true only if the model obeyed an empty `Sources:` block. The checklist's About row expects
a version and there was none.

**Checked.** Full suite 2,823 passed, 8 of them new or extended for this; ruff, `mypy src`, bandit, `svelte-check`
0 errors, 317 desktop tests; `docs_check --strict` and `integrity_check` OK. The real API and
pipeline over an empty data folder, four questions (empty library twice in one chat, empty folder,
unknown folder): each got its sentence, 0 tokens streamed, all in `/api/conversations`, and Ollama's
log took no request. The dev app in the browser pane: the answer, the card line, `0 tokens ·
$0.0000`, *Version 0.7.0* in About; its three console 500s were the dev proxy while I restarted the
API. **Not checked yet:** the installed build (the fix needs the rebuild), and a first question with
no engine set up.

**Rejected.** A firmer sentence in `ANSWER_PROMPT`: it would still leave an honest state to a model,
and it moves `prompt_version` and the eval baselines. Hiding the answer when the citation check
fails: that check is advisory on every other turn, and "surface, don't rewrite" is its rule. A third
`mode` value on `TurnResult`: a wire change for a field the window does not read. `/api/updates` for
About's version: reading it can start a network check when one is due; health already feeds the
dialog.

**What it opens.** RG-012 and the walkthrough on a new build. Polish left from the same walkthrough:
the rail's "1 setup steps left" against the panel's count; "Couldn't check for updates" before any
check has run; "Claude needs an API key first" showing with Ollama active.

---

## 2026-10-27 (6) — The add review sheet: a checkbox per file, as the spec and the wireframe had it

**What changed.** `AddDocuments.svelte`: an *Add* row carries a checkbox in place of its check icon,
ticked, with the file name as its label, so the whole name is the click target; an unticked row
stays listed, dimmed. *Select all* / *Select none* sit on the headline row once two files can be
added. The headline, the time estimate and *Add N documents* count what is ticked, and "and N more"
says how many of the hidden rows still are. The logic is pure, in `lib/library/addselection.ts`
(6 tests): only a row the server starts ticked (`selected_by_default`, a clean `add`) can be ticked,
so the choice is subtractive and never reaches a duplicate, an unsupported or an unreadable file.
The spec's AD2 section records it as built.

**Why.** The walkthrough: picking a folder or several files gave no way to leave some out. The
spec's verdict table (AD2) and the 2026-08-20 plan's wireframe (§6.2) both had a tick per row; the
sheet shipped with the verdicts only. `/api/documents/add` already takes the list of paths, so the
server does not change.

**Checked.** The dev app, ten files from the walkthrough folder: 8 addable rows ticked, the damaged
EPUB and the `.xyz` listed without a box. Unticking one by its name and one by its box gave
"6 would be added" and *Add 6 documents*; *Select none* gave 0 and a disabled button. *Add* with one
unticked sent exactly the 7 ticked paths (the add and index calls were answered by a stand-in in
the page, so nothing was written), and the boxes locked once added. `svelte-check` 0 errors;
317 tests.

**Rejected.** Ticking a duplicate to add it again: `apply_add` would refuse it on the registry's
uniqueness and report a failure the user asked for. A tri-state checkbox in a header: the sheet is
a list, not a table, and a word says what the click does.

**What it opens.** Options per format on a row, which the user mentioned and has not specified yet.

## 2026-10-27 (5) — Larger text, controls and panels; the document page's section strip centred

**What changed.** `app.css`: the base size 15 px → 17 px. Sizes in the app are in `rem`, so text,
buttons, pills and padding grow by 13% together. Panel widths move from px to `rem` so they grow
with their text, and the ones the walkthrough named get more room: *Add documents* 680 px → 52rem
(884 px at the new base); *Manage folders*, *Manage keywords* and search 640 px → 48rem (816 px);
the keyword filter and Settings 760 px → 52rem; the taxonomy 880 px → 60rem; *Re-run* 520 px →
36rem. The small confirm, about and edit dialogs only follow the base. The sidebar's *Manage…*
(0.7rem, 11.9 px at the new base, the one visible control still under 12.5 px) takes `--text-sm`.
The document page's section strip (`.blocknav`, `LibraryBrowser.svelte`): its pills sat on the
library header's rule above (0 px) with 7.5 px below; now centred, 7.6 px on each side.

**Why.** The walkthrough, twice (2026-10-26 and today): buttons such as *Manage…* too small for
weak eyesight, the panels a menu opens too small, and the section pills touching a line. At the
15 px base the commonest small size, 0.72rem, was 10.8 px, and the smallest 9.3 px.

**Checked.** The dev app over the 104-document scratch copy, at 1024×768 (the window's default) and
1440×900 in light and at 375 px in dark: no horizontal scroll at 375 px, where the strip wraps to
two rows; *Manage folders*, search and Settings open wider and nothing clips. The fixed-pixel
heights in the app are icons and marks, not text boxes. `svelte-check` 0 errors; 311 tests.

**Rejected.** Raising each small size by hand (about 300 declarations between 0.62 and 0.78rem):
the same result with 300 chances to miss one and no single value to tune. A floor on every small
size: the base already lifts the smallest to 10.5 px and the pills to 12.6 px; what stays small are
counts and labels, not targets. A text-size choice in Settings: the next step, since the base is
one value, but a new control; the default had to change first.

**What it opens.** A *Text size* choice in Settings. Less fits in the default 1024×768 window.

## 2026-10-27 (4) — `msgpack` 1.2.3: an advisory published during the release

**What changed.** `uv.lock`: `msgpack` 1.2.1 → 1.2.3 (`uv lock --upgrade-package 'msgpack==1.2.3'`);
nothing else moved. **Found by:** CI on `167ef5d`, whose pip-audit gate failed on CVE-2026-114613
(GHSA-j586-36cw-2gc2, published 2026-10-22); the same gate had passed here less than an hour
before. Tests, lint, mypy and bandit passed in that run.

**Why.** It ships: `wordfreq`, which the keyword pass reads word frequencies from
(`knowledge/keywords.py`), depends on it, and the frozen backend bundles it. The leak needs
msgpack input an attacker supplies, and `wordfreq` unpacks only the frequency files it ships, so
nothing here reaches it; a patch release was still cheaper than a reviewed ignore. 1.2.3, not the
newest 1.2.4: 1.2.3 carries the fix and was a month old, 1.2.4 a day. Checked: `wordfreq` reads
its data (`zipf_frequency("the", "en")` 7.73), the gate is OK, the whole suite passes.

**What it opens.** The seventh advisory round since 2026-10-20. The release backend is frozen
from this commit; the freeze from `167ef5d` was stopped for it.
