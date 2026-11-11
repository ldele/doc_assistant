<!-- status: active · updated: 2026-11-11 · class: append-only -->

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
> **2026-11-05 (3) → 2026-11-06 (3)** in [`docs/archive/DEVLOG-archive-010.md`](archive/DEVLOG-archive-010.md)
> (rotated 2026-11-09: 11 entries) ·
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

## 2026-11-11 (4) — The PDF fingerprint no longer carries the app's version (KI-91)

**What changed.** `ingest/cache.py::_extraction_closure` treats a class defined in `extractors`
as code, not a dependency: a new `_class_parts` walks its methods (and any base written there)
like the module's functions, hashes its class-level values as constants (`_module_attr` resolves
`Class.name`), and adds only a foreign base's package to the module list. Two guard tests: no
format's fingerprint moves with `doc_assistant.__version__`, and an edit to
`_CellTextOfBlocks.__call__` moves `.pdf` and nothing else. A CHANGELOG `[Unreleased]` *Fixed*
line says the next update marks every PDF *changed* one last time.

**Why.** The KI-64 ligature fix's two wrapper classes live in `extractors`, and the walk mapped
any class it reached to its package's version, so `doc_assistant=<version>` was in the PDF
fingerprint and every release since 0.7.0 marked every PDF *changed*. Both tests failed first; the
second showed the converse hole: an edit to those classes' methods between releases was invisible
(KI-40), masked only because a release moved the version anyway.

**Rejected.** Dropping `doc_assistant` from the module list alone: it fixes the release noise and
keeps the KI-40 hole. Hashing the whole `extractors` module when a local class is reached: every
EPUB fix would re-read every PDF again (KI-48).

**What it opens.** Measured: `.pdf` moved once (`72f4aab8da6266fa` → `b4d61cccbf2cbc78`), the
other eight formats unchanged, and the PDF closure holds six methods of the three KI-64 classes.
The 0.8.0 upgrade run (A9) should show every PDF *changed* once; the one after it, none.

## 2026-11-11 (3) — 0.7.3's notes say every PDF reads *changed* after the update

**What changed.** The 0.7.3 CHANGELOG no longer says nothing is read again: its summary and a new
first *Known limits* entry say every PDF is listed as *changed* after the update, though none is,
and that reading them again is not needed. The release checklist's upgrade row (§7) expects the
same, naming KI-91.

**Why.** The upgrade run from the real 0.7.2 installer in the sandbox (2026-11-11) kept every
record (documents, titles, the folder, chunks, conversations), then listed the 4 PDFs *changed*
and the text file *ingested*. Cause: the PDF extraction fingerprint hashes the version of the
package each reached class comes from, and the KI-64 ligature fix's two helper classes live in
`extractors`, so `doc_assistant`'s own version is hashed (KI-91). Every release since 0.7.0 does it.

**Rejected.** Fixing it before the tag: 0.7.3 would mark every PDF *changed* either way, because
the fix moves the fingerprint once more, and it would mean a change to the cache code after the
gates, then the gate, the build, RG-012 and the upgrade run again (the user's call: ship).

**What it opens.** KI-91's fix in release A; that upgrade marks every PDF *changed* one last time.
RG-012 on this build passed (`run-20261111-152202`, 3 of 3 turns cited); the build is unchanged.

## 2026-11-11 (2) — Preflight counts the Windows window config as a build input; CI is green again

**What changed.** `scripts/release_preflight.py` lists
`apps/desktop/src-tauri/tauri.windows.conf.json` in `SHIPPED_PATHS`, and
`test_the_shipped_path_list_is_pinned` pins it.

**Why.** ADR-065 (`f35cabe`, merged in `5ed8bf5`) added the file: Tauri merges it over
`tauri.conf.json` on Windows, where it removes the native frame, so it changes what ships. Left
out, `test_every_tracked_app_file_is_shipped_or_named_as_not` failed on every CI run of `main`
from `5ed8bf5` to the release commit `823bf3b`, and `artifact_fresh` could not see a change to it
(KI-58's sixth instance). Found by 0.7.3's full gate: 1 failed, 3,229 passed.

**Rejected.** `NOT_SHIPPED`: the file is in the installed app's window behaviour.

**What it opens.** The release is tagged at this commit, not at `823bf3b`; the build is made from
it.

## 2026-11-11 — An engine error reaches the installed window as a 500, not as "Failed to fetch"

**What changed.** `apps/api/server_errors.py` adds `ServerErrorCorsMiddleware`, mounted innermost
in `create_app`, inside CORS. A route that raises before sending now gets a 500 that carries the
CORS headers; the error is re-raised, so the server still logs it. A response already started (the
chat's event stream) is left alone. Tests: the window-origins test gains a 500 for both window
origins (it fails without the fix); `test_api_server_errors.py` pins the before/mid-stream split.

**Why.** The 0.7.3 report *"Couldn't load connections: TypeError: Failed to fetch"*. The installed
window calls the engine from another origin, and Starlette answers an unhandled exception outside
CORS, so the 500 was discarded and read like a dead engine. In dev, Vite proxies on the same origin,
so the same failure showed as a 500. The report's own cause is not known: no document reproduces
it on a copy of the library, and the installed engine keeps no log (KI-90).

**Rejected.** A FastAPI handler for `Exception`: Starlette mounts it on the outermost layer, outside
CORS, so the header would still be missing. A JSON body: the window's clients fall back to
"X failed: 500" only on a non-JSON body, so Starlette's plain text is kept.

**What it opens.** KI-90 (a log file for the installed engine) and KI-79's health poll, for
release A. The CHANGELOG line is written at `docs/RELEASE.md` §2.

## 2026-11-10 (2) — A digest is registered, and Docker joins the release gates

**What changed.** `[generate]` registers `.claude/DIGEST.md`: every known issue's heading and
status, the open tickets and the newest DEVLOG headings, ~1,600 tokens. `AGENTS.md` numbers it
fourth in what every session reads. `docs/RELEASE.md` §3 gains a Docker row: read the image's CI
job green on the release commit, and re-read the `Dockerfile`, the compose file and the setup
guide when the CHANGELOG names a new dependency, variable, port or data path.

**Why.** The entry file told each session to scan `docs/KNOWN_ISSUES.md`, ~31,500 tokens. The
user, 2026-11-10: the digest should be standard, and Docker is part of release work.

**Verified.** `generate --check` and `docs_check --strict` pass; `keypoint session-start` lists
the digest fifth. No release was cut, so the Docker row has not been used.

**What it opens.** A triage of the 29 open known issues, drafted for the user's ruling.

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
