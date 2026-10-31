<!-- status: active · updated: 2026-10-31 · class: append-only -->

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
