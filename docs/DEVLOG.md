<!-- status: active · updated: 2026-10-09 · class: append-only -->

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

## 2026-10-09 (3) — The knowledge layer read figure chunks as the document's own text; it now reads prose (ROADMAP 97)

**What changed.**
- **`concept_skeleton.prose_parents`** (new, pure): child-row metadata → one entry per prose
  parent, a `chunk_type == "figure"` parent left out. `load_presence_inputs` returns what it
  keeps, so presence and co-occurrence, the written-form vote and the library-wide definitions
  scan all read prose.
- **`definitions.chunks_mentioning`** (the one-concept path): a figure's block in the keyword
  index neither picks a document nor comes back as one of its parents.
- **`epistemics.load_pc_parent_chunks`** is unchanged and now says why: it reads for what
  retrieval can return, and a retrieved figure needs its markers.
- Retrieval, both stores and the keyword index are untouched. No locked setting moved.
- `docs/knowledge-layer.md` (the presence row); ROADMAP 97, and row 95 restated on prose.

**Why.** A described figure is stored as a parent chunk of its own, after the document's prose:
its caption, the vision model's description, and a whole copy of the passage that cites it. That
is what retrieval wants. The knowledge layer's reader returned every parent, so it took those
chunks for text the document wrote, with three effects. A cited passage was counted twice, and a
link needs only two shared chunks. A model's wording could make a term present in a document that
never uses the word, where presence is meant to be decided by the text alone. And appended after
the prose, the figure text moved the References heading before the halfway mark the bibliography
cut wants, so the cut missed nine documents. Found while listing, for the user, how the library
uses the forms they were about to mark: one sentence showed twice in three documents, and the
extraction cache held it once.

**Measured** (`tests/eval/baselines/prose_parents_2026-10-09.md` — the committed code against the
change, dry runs on the working library with the user's marks in place):
- 615 figure parents in 83 of 104 documents, 6.6% of the text that was read; 181 prose parents
  stored more than once, one of them nine times.
- **The 13 concepts: the same documents, the same 25 links, the same 12 gap rows.** Three links
  rest on one or two fewer shared chunks, so the graph version changes.
- The 357 rows: 1,103 concept–document pairs → 1,079. The 24 are all terms, and each is a word
  found in a model's description and in no caption (`plateau` in 8 documents).
- Written forms: four move, all on terms.
- The bibliography cut fires on 85 documents, not 76. Section 5 of the baseline replaces section 7
  of `vocabulary_shape_2026-10-01.md`: what the cut wrongly removes after a reference list is 3.7%
  of the text in 32 documents, not 4.5% in 39.
- Definition candidates: 20 for the concepts and 110 over every row, the same sentences before
  and after.
- Five new tests; the two that go through the readers fail on the code before. Suite: 2,643
  passed.

**Rejected.**
- *Keep the caption, drop only the description and the copy.* The caption is already in the prose
  parent it was extracted with: no pair came from a caption alone.
- *Filter in each caller.* Presence, the vote and two definition paths are four places to forget;
  there is one reader.
- *A `chunk_type` column on the index's `parents` table.* It changes retrieval's index file for a
  rule the reader can apply from the block's own metadata.
- *Align the marker reader with this one.* A retrieved figure would lose its markers.
- *Take the copy out of the figure chunk.* It is what lets an answer read a figure inside the
  passage that argues from it.

**What it opens.** It applies at the next graph rebuild, which is the user's: the graph version
changes, three links lose one or two shared chunks, and the written forms are voted on prose.
ROADMAP 95 (a mention in a reference list stops counting) starts from the prose figures. Not
measured: retrieval returning a figure and the passage it cites together, so that an answer reads
that passage twice.

## 2026-10-09 (2) — The audit gates' second catch: four upgrades for advisories published since the last green run

**What changed.** Four versions, in two lock files, and nothing else.
- `uv.lock`: fsspec 2026.4.0 → 2026.6.0, multidict 6.7.1 → 6.9.1.
- `apps/desktop/package-lock.json`: source-map-js 1.2.1 → 1.2.2, dompurify 3.4.15 → 3.4.16.
  `package.json` is unchanged.

**Why.** CI on `b50941a` — a frontend change that touched no dependency — failed in both audit
steps (run 37933120201), on advisories published in the eight days since the last green run.
pip-audit reported two it had not seen: fsspec (CVE-2026-104851) and multidict
(CVE-2026-104874). `npm audit --audit-level=high` reported source-map-js at high
(GHSA-68fv-2mgg-jv7q) and listed two low ones on dompurify (GHSA-p98j-92pf-mc4p,
GHSA-6688-9rhm-gjv2). Lint, types and the tests passed; the secret scan, which runs after
pip-audit, was skipped.

**Where the four sit.** Nothing under `src/`, `apps/` or `scripts/` imports fsspec or multidict:
fsspec arrives through huggingface-hub and torch, multidict through aiohttp and yarl.
source-map-js is build-time only (vite → postcss). dompurify is the one direct dependency: it
sanitises rendered answers (`lib/chat/Markdown.svelte`), on a string, and both of its advisories
are about the `IN_PLACE` mode, which the app does not use.

**Read and measured.**
- `uv lock -P fsspec==2026.6.0 -P multidict==6.9.1`: each pinned to the smallest version that
  clears its advisory. 236 packages before and after, two version lines changed, no package
  added, removed or downgraded, no dependency edge added or removed, every changed artifact from
  `files.pythonhosted.org`.
- `npm audit fix`: two packages changed in the lock, six lines each way.
- The gates as CI runs them: `scripts.pip_audit_gate` — 196 audited, 5 reviewed, **0 unreviewed**;
  `npm audit --audit-level=high` — 0 vulnerabilities.
- The unit and integration suite passed on the upgraded packages: 2,643, in a tree that also held
  an unrelated change under `src/` with five new tests (CI runs this commit alone). Frontend: 286
  tests pass, `svelte-check` 0 errors. No trial freeze of the sidecar; the release sessions
  freeze anyway.

**Rejected.**
- *Reviewing the four into the ignore lists.* Each has a fix the lock takes without moving
  anything else.
- *An unpinned `uv lock -P <package>`.* On 2026-09-30 that pulled a major SDK version and three
  new packages for the same kind of fix.

**What it opens.** This is the second push in eight days that changed no dependency and turned
`main` red (the first was `a5bb634`, 2026-10-01). A blocking audit with nothing that looks between
pushes makes each push pay for what was published since the last one: the second data point for
S-12, the scheduled or alert-driven check.

## 2026-10-09 (1) — Manage keywords built 312,166 dropdown options on opening: a row now has one text field, a mark shows at once, and the Graph tab opens the view (ROADMAP 96)

**What changed.**
- **One field to add a form.** Each row of `LibraryManageKeywords.svelte` carried an
  "Add a form…" `<select>` listing every unassigned keyword. It is now a text field over one
  shared `<datalist>`: type to search the keywords, or type any form and press Enter.
  `formToAdd` (`library.ts`) refuses a blank, the row's own name and a form the row already has.
- **A mark shows when it is clicked.** The exact and broad buttons show the mark that is being
  saved (`shownBreadth`) and take no click until the server has answered. A save that fails
  leaves "not saved — click again" on the form. `setFamilyFormBreadth` (`App.svelte`) replaces
  the one row with the route's answer instead of fetching every family again, and resolves to
  whether the mark was saved.
- **The Graph tab opens the view.** A *Manage concepts* button beside the coverage sentence
  opens Manage keywords on its Concepts section (`ConceptGraph.svelte`, `openOnConcepts`).
- The × that removes a form sits a little apart from the two marks.

**Why.** The user marked the 22 forms of ADR-054 on 2026-10-09 and met three faults in an hour.
The view froze on opening. "Add a form" listed 1,229 uncurated keywords and could not add a
string that is not one of them, so a form removed by a slip (`contrastive`) had no way back
through the screen. And a mark looked unsaved for as long as the view took to redraw: a mark is
set by toggling, so the second click cleared what the first had set. Two attempts on one form
left no mark, and nothing said so. They also asked for a way into the view from the graph: "It
would be more logical to 'manage keywords' too in graph."

**Measured** — the dev app on the working library (104 documents, 254 rows shown), opened from
the Library's keyword filter with the family list already loaded:

| | Before | After |
|---|---:|---:|
| Elements in the page once the view is open | 322,887 | 11,958 |
| `<option>` elements | 312,166 (254 rows × 1,229) | 1,228 (one list) |
| Time the page is blocked by the click that opens it | 5.8 s | 0.1 s |

Measured in a browser pane that was not on screen, so the times are what the script cost, not
what a person waited. Opening makes no request of its own, before or after. 286 frontend tests
pass (two new), `svelte-check` reports 0 errors; at 375 px in the light theme and at desktop
width in the dark one, neither the new field nor the new button overflows.

**How the writes were checked.** With `fetch` replaced in the page so that no request other
than a GET could leave it: a slow save (the mark shows at once, both buttons are disabled, a
second click sends nothing), a failed save (the mark goes back and the message appears), a
retry (the message goes), and adding `DPR` to `dense retrieval` (one POST carrying that
string). The library file's write time did not move during the checks. That the route accepts a
string that is no keyword was shown the same day by a real call: `contrastive`, put back at the
user's word.

**Rejected.**
- *A lazily filled `<select>` per row.* Still a list of uncurated strings to scroll, and still
  unable to add a form that is not a keyword.
- *Marking forms inside the graph's concept panel.* A second place that writes the vocabulary,
  against ADR-017 A1. The graph gets a door to the one place.
- *A confirmation before a form is removed.* With any string addable again, a slip now costs one
  field and Enter. The removal itself is still silent.

**What it opens.** The fix for the figure chunks the knowledge layer reads as document text is
next. From the same hour of use and not done here: a wider left margin, a model picker in
Settings, and saying on the screen why *Add documents* is unavailable in a browser tab.

## 2026-10-01 (7) — The secret gate's first CI run was red, and right: on Windows the scanner had been skipping 47 files (S-14)

**What changed.**
- **The scanner reads every file as UTF-8, on both platforms.** `scripts/secret_scan_gate.py`
  starts the scanner with `python -X utf8`, and `.pre-commit-config.yaml` overrides the hook's
  `entry` to do the same (same repository, same rev).
- **`.secrets.baseline` records three more findings**, added by a scan run in UTF-8 mode: a fake
  id in `tests/unit/test_chat_controller.py` and two fake keys quoted in an archived DEVLOG entry.
  Its plugins, filters and existing entries are unchanged.
- The gate prints a finding's path with forward slashes, so a Windows run and CI name it alike.

**Why.** CI on `07f4b50` failed at the new secret-scan step with three unrecorded findings, while
the same gate passed here. `detect-secrets` opens a file in the platform's default encoding and
treats a decode error as "binary, skip". On Windows that encoding is cp1252, and a UTF-8 file
with one byte cp1252 does not define — a curly closing quote is enough — is skipped whole, with
no message. Linux read every file. So the hook that runs before every commit on this machine had
never looked inside those files, and the gate's "837 files scanned" was not true here.

**Measured.**
- 47 of 838 tracked files are UTF-8 text that cp1252 cannot decode: 20 under `apps/`, 16 under
  `docs/`, 5 tests, 3 under `src/`, 2 data files, 1 at the root; 28 are code or tests.
- The scanner's plugins match all three lines when handed them directly; its file scan reported
  none of them in the default encoding and all three with `-X utf8`. Line endings and path
  separators make no difference.
- After the change the gate gives the same verdict on Windows and in a Linux run (WSL) of this
  tree: 838 files, nothing unrecorded. The suite: 2,638 passed.
- The reconfigured hook fails on a scratch file holding a curly quote and the published example
  key, names the file and not the key, and passes over every tracked file.

**Rejected.**
- `# pragma: allowlist secret` on the two DEVLOG lines: the archive is kept verbatim.
- Setting `PYTHONUTF8=1` for the whole machine: it changes every Python program on it, and the
  repository's rule is an explicit encoding where the file is opened.
- Replacing the upstream hook with a local one in the project's environment: the `entry`
  override keeps the pinned rev and changes one thing.
- Leaving the local hook as it was because CI now catches it: CI runs after the push, and in a
  public repository that is after the key is out.

**What it opens.**
- Until this is pushed, `main` is red.
- The same default hides in any tool that opens a file without naming an encoding. The
  repository's own code is covered by its encoding rule; third-party tools are not, and the way
  this one was found was a second platform disagreeing with the first.
- A baseline entry is a hash at a line: the three new ones are fake values, read before they were
  recorded.

---

## 2026-10-01 (6) — The test suite runs on a data directory of its own (KI-60 closed)

**What changed.**
- **One redirect for everything.** `tests/conftest.py` sets `DOC_DATA_DIR` to a fresh, empty
  directory before `doc_assistant.config` is first imported: `.pytest-data/run-<pid>-<token>/`
  in the repository (gitignored), made for the run and removed when it ends. The library
  database, the vector stores, the extraction cache, the session logs, the settings file and the
  graph sidecars all resolve inside it, in the test process and in any subprocess a test starts.
- **It fails closed.** If `config` was imported before the conftest could set the variable, the
  run stops before any test and names what `config` resolved.
- **Cleaning up.** At the end the default database engine is disposed and every Chroma system the
  run opened is stopped, then the directory is removed. A directory left by a killed run is swept
  by a later run once it is a day old. `DOC_TESTS_KEEP_DATA_HOME=1` keeps the directory and prints
  its path, to see what the tests wrote.
- The session fixture of entry (5) is gone: the default engine is bound inside the run's
  directory by construction.
- `tests/unit/test_suite_data_home.py` (14) pins the mechanism: no path in `config` points into
  the repository's `data/`, each default store is inside the run's directory, a subprocess lands
  in the same one, a run whose `config` was imported first is refused, and only a stale run
  directory is swept.

**Why.** Entry (5) fixed the database and measured what was left. The same run-by-run debris had
been landing in the working library for months: 945 of the 946 folders under
`data/cache/referenced/` are named after test fixtures (`a.md`, `b.md`, `gone.md` …), and the
4,740 timestamped session logs in `data/exports/` include every test run's (about 37 each). None
of it failed a test, so nothing would have stopped the next thing a test wrote.

**Measured.**
- **Before the change, with the data directory pointed at an empty folder:** the suite passes
  unchanged (2,623 tests) and writes 76 files there — 38 extraction-cache files in 19 folders,
  37 session logs, and one vector store. `tests/unit/test_wiki.py` is the test that opens the
  default store, which creates it when it is absent.
- **After it:** every file under `data/` except `sources/` and the backups, compared by size and
  modification time before and after a full run — 7,815 files, none added, removed or changed;
  the working database and both vector stores keep their timestamps. 2,636 tests pass (coverage 92.7%), and `.pytest-data/` is
  gone when the run ends.
- **Chroma holds its file.** On chromadb 1.5.9 a directory with an open store cannot be removed
  on Windows, and dropping the client does not release it; stopping the cached systems does.

**Rejected.**
- The system temp directory. `config` moves the vector stores to `%PROGRAMDATA%` when the data
  path is not ASCII (KI-11), and the temp path of a Windows account with an accented name is not:
  the tests' stores would land in the machine-wide directory the installed app uses, outside
  anything a run removes. In the repository the path is ASCII whenever the checkout is.
- `os.environ.setdefault`: a developer who exports `DOC_DATA_DIR` to run the app against a library
  elsewhere would hand that library to the tests.
- Keeping the session engine fixture beside the redirect: two mechanisms for one guarantee.
- Failing the session when `data/` changed during it: it would fail whenever the app is used
  while the tests run.
- Changing `test_wiki.py` so it stops opening the default store: it is contained now.

**What it opens.**
- The debris earlier runs left in the working `data/` is still there; removing it is the user's
  call.
- ROADMAP 76's shared-fixture item is done, and with it KI-58's second open half: a test can no
  longer pass on this machine because the working library exists. The Windows CI job is open.
- `wiki.sample_chunks` creates an empty vector store when it reads a missing one — a read with a
  side effect, left as found.

---

## 2026-10-01 (5) — The test suite no longer reaches the working library's database (KI-60)

**What changed.** `tests/conftest.py` gains a session fixture that binds the database layer's
default engine to a throwaway file for the whole run. A test that reaches the database without
swapping the engine now lands on an empty database, as it does on a fresh checkout.

**Why.** `db.session` binds its engine to the configured `library.db` at import. Two route test
files (`test_reingest_routes.py`, `test_ingest_progress.py`) enter the app's lifespan without
redirecting it, and the lifespan's first act is the schema migration. On 2026-10-01 a full run
therefore added row 94's new column to the working library, before the app had ever been started
on that code. Read back afterwards: integrity check `ok`, no form marked, and 593 labels and 404
alias rows identical to the backup of 2026-09-01 — the column is the only change, and it is the
one the app would have made at its next start. Reproduced against an empty data directory:
before the fixture those two files create a `library.db` there, after it they do not.

**Rejected.**
- Stubbing the migration in the two files: it fixes today's two and leaves the next unredirected
  test free to do the same.
- Pointing the whole suite at a temporary data directory (`DOC_DATA_DIR`): the right end state,
  but two tracked files live under `data/` and which tests depend on that directory is unmeasured.
- A session-end check that the working database's modification time did not move: it would fail
  whenever the app is used while the tests run.

**Measured after the fix** (one full run, every file under `data/` compared before and after):
`library.db` untouched. The run still added 58 extraction-cache files for the tests' own documents
(`cache/referenced/`), two session logs under `exports/` and one row in the vector store's
write-lock table; the store's content is unchanged (18,011 embeddings).

**What it opens.** This is the database half of ROADMAP 76's shared-fixture item, and of KI-58's
second open half. The debris above is KI-60's open part: nothing the library holds changes, but it
accumulates, and a test that can open the working vector store could write to it. A temporary
data directory for the suite would close it, and belongs to row 76.

---

## 2026-10-01 (4) — A concept has a name and exact or broad forms, and the other rows are terms (ROADMAP 94, ADR-054); CI's secret scan can fail (S-13)

**What changed.**
- **A form is exact or broad.** One additive column, `concept_aliases.breadth`; unset reads as
  exact. `concept_skeleton.load_concepts()` — the loader that presence, edges and the gap list's
  claim attribution count through — leaves a broad form out. `load_broad_forms()` and
  `match_broad_presence()` match it on its own, and a node carries the documents only a broad form
  reaches (`broad_doc_ids`, `broad_forms`) beside `doc_ids`, never in them. The name is always
  exact: a broad mark on an alias that repeats the label is ignored.
- **Concepts and terms.** Stored definition candidates, merge suggestions, `is_a` proposals and
  field-placement proposals read concepts (rows with `graph_include`). In all four runners
  `--include-terms` reads every row and cannot be applied (for placement it replaces
  `--all-concepts`, kept as an alias). A term refuses a definition write (409) and returns the
  sentences found for it without storing them.
- **The name on every screen.** `written_forms.shown_labels` is the one helper. The vocabulary
  search, the gap list, the taxonomy view and Manage keywords carry `written`, as the graph did.
- **Manage keywords** lists concepts apart from terms. A concept's row has an exact | broad control
  per form. Delete asks first, says what goes with the row (forms, the chosen definition and the
  other options, field placements, gap decisions, the graph's documents) and offers "Take off the
  graph instead". The graph's Edit button, and a term's Manage keywords button, open the view on
  that row.
- **The graph says what it is behind on.** A build records a fingerprint of each concept's name
  and forms in `skeleton.meta["forms"]`; the view compares it with the live vocabulary and names
  the concepts that changed. A graph built before the record says it cannot tell, and offers the
  rebuild.
- **S-13.** CI's secret step is `python -m scripts.secret_scan_gate` (`just secret-scan`): the
  `detect-secrets` hook over every tracked file against a temporary copy of the baseline. It fails
  on a secret the baseline does not record, never rewrites the tracked baseline, makes no network
  call, prints a finding's type and place and never its value, and exits 2 when it could not look.
  It is stricter than the local hook in one way: the hook can ask a provider whether a candidate
  key is live and drop it if not, and the gate keeps it. It adds about 19 seconds to a CI run.

**Why.** ADR-054, accepted 2026-10-01: a count should say what it rests on, and until now it could
rest on a row nobody read or on an alias that means something wider. S-13: the step CI ran,
`detect-secrets scan --baseline`, is the command that writes a baseline — it recorded a new finding
and exited 0 (found 2026-09-30).

**Measured.**
- **Nothing moved.** `build_concept_skeleton(apply=False)` on the working library with the
  committed code and with this change: the same graph version (`cb99a7f9f36545da`), 13 concepts,
  30 edges, 86 concept–document pairs, equal field by field
  (`tests/eval/baselines/names_and_forms_2026-10-01.md`).
- **One mark, on a copy of the library.** `distillation` marked broad: `knowledge distillation`
  11 documents → 4, with 7 listed beside; no edge lost, six of its edges on fewer shared chunks;
  3 documents leave the graph's coverage. The graph named the concept until it was rebuilt. This
  is one mark to exercise the path. The user's reading of the forms has not happened.
- **Terms.** `viral` and `SPECTER` opened in the app on the copy: usage passages for both, two
  defining sentences for `SPECTER`, no control to choose one, and no row stored for either.
- **The count to classify is 22, not 31.** The 13 concepts hold 31 alias rows; 9 repeat the name.
- **S-13.** In a scratch repository the published example key fails the gate (exit 1) and passes
  the command CI ran before (exit 0, the key written into the baseline). The repository passes:
  834 files, about 19 seconds.
- Checked live on the copy in light and dark and at 375 px: no failed request after a clean load.
- The suite: 2,622 passed, coverage 92.8%. The gate's test file gained one test and had one
  rewritten after that run, and passes on its own (24). The desktop's 284 tests and its type
  check pass.

**Rejected.**
- Adding a broad form's documents to presence behind a flag: edges and gaps would count them.
- `graph_version` as the "forms changed" signal: it fingerprints the result, and the gap and
  epistemics sidecars key their own staleness on it.
- Calling a graph built before the record "changed": there is nothing to compare, so it says that.
- Deleting a row that has nothing attached without asking: one rule for every delete.
- For S-13, failing on `git diff --exit-code .secrets.baseline` after the old scan: it turns the
  scan's side effect into the signal, and a recorded line that only moved would fail it.
- Running the hook on the tracked baseline in CI: the hook rewrites the file it is given.
- Naming the `just` recipe `secrets`: the scanner reads that name and the command under it as a
  keyword with a value, so the commit hook failed on the recipe (and on a sample report line in
  the gate's own test). The recipe is `secret-scan`.

**What it opens.**
- The user's part: mark the 22 forms and name the concepts in Manage keywords, then rebuild.
  RG-032 is measured after that, before ADR-053 decision 4's signals (the next session).
- The working library's graph predates the record, so it will show the "cannot tell" notice until
  its next rebuild. That rebuild also applies the case-aware matching of 2026-09-30 (`cre` 7 → 6).
- The Graph tab offers Rebuild only when it reports itself behind. A standing control is a design
  choice left open.
- Found on the way: the test suite migrated the working library (the next entry, KI-60).
- `docs/specs/feature-tag-families.md` calls the hidden group "glossary-only"; the view says
  "unused" now, and the spec carries a dated note.

---

## 2026-10-01 (3) — ADR-054 accepted, with the user's amendment: a base vocabulary per field is a source of terms

**What changed.** No code.
- **ADR-054 is accepted** (user, 2026-10-01). It gains *Amendment 2026-10-01*, and its Status and
  its "to decide" line now say what was accepted. A curated base
  vocabulary for a field is a third source of rows beside the user's additions and the keyword
  extractor; its entries stay terms until the user takes them on. One sentence of the Decision is
  widened: a term is a candidate row nobody has taken on, whatever proposed it, because a base
  entry may not occur in the library at all.
- **ROADMAP KL2** names it as the concept-level half of "the taxonomy as the reference class for
  expected coverage"; row 94 (ADR-054's build) moves from *gate first* to *planned — next*; the
  index line in `docs/decisions.md` follows.

**Why.** The user, on reading ADR-054: a curated default vocabulary for a given topic "would go
hand in hand with the taxonomy"; otherwise "the text is clear". With the amendment in front of
them the user accepted the ADR: "Okay, let's go with this". The idea continues the user's
question of 2026-09-21 (experts have already listed and defined a field's terms), which ADR-053
answered for definitions only. It also supplies what `docs/knowledge-layer.md` §1 says a gap needs:
an expected structure to deviate from. The taxonomy says which fields exist; a base vocabulary says
what a field contains.

**Rejected.**
- Making a base list's entries concepts by default: that is the bulk promotion of 2026-07-05 with
  a better source, and ADR-054 exists because 344 rows became concepts without being read.
- Importing a vocabulary whole: ADR-028 decision 7 measured it as "a facet that partitions
  nothing" (30,000 MeSH descriptors for a small library).
- Rewriting ADR-054's Decision in place: the file is append-only; the amendment states the one
  widened sentence.
- Designing the feature here: which vocabularies, their licences and sizes, and whether a topic's
  list is taken on entry by entry or as a reviewed whole belong to their own ADR.

**What it opens.** The ADR-032 grill (KL2) decides what the base list is for and how it is taken
on; 93c's local copies are its data step. Known limit before any build: of the 19 priority
concepts, expert vocabularies define 8, gloss 6 and have nothing for 5
(`tests/eval/baselines/reference_vocabularies_2026-09-21.md`), so a base list is solid for the
established fields and thin for the youngest ones. The acceptance opens row 94: names and forms on
the 13 concepts. CI on `7357fcc`, the commit that carried the dependency fix and ADR-054 as
proposed, is green.

---

## 2026-10-01 (2) — The vocabulary is read against the library; concepts and terms, names and forms (ADR-054, proposed)

**What changed.** No code. Three documents:
- **`tests/eval/baselines/vocabulary_shape_2026-10-01.md`** (new): a read-only snapshot of the 357
  text-bearing concept rows against the library's text — where they came from, how far each
  spreads, what the words beside a one-word label are, what each alias of the 13 graph concepts
  contributes on its own, and what the bibliography cut removes.
- **`docs/decisions/ADR-054-concepts-and-terms-names-and-forms.md`** (new, proposed) and its index
  line: a *concept* is a row the user has taken on (`graph_include`, 13 today) and the other 344 are
  *terms*; a concept's label is its name, and each matched form is *exact* (counts as presence) or
  *broad* (counted beside it). One additive column, no second table. Not built.
- **`docs/ROADMAP.md`:** row 94 (ADR-054's build) and row 95 (the bibliography cut, body-text
  presence); row 93's abbreviation signal moves after 94.

**Why.** The user, before the abbreviation and fragment signals of ADR-053 decision 4 were built:
"we will need to think more about vocabulary". The open items — a label shorter than its concept, a
homograph, an alias that means something else, a label that is an author's surname — turned out to
share one cause: nothing in the data says whether a row is a meaning someone chose or a string an
extractor produced. The four choices were put to the user one at a time, each with a recommendation
and rows from the library, and the user took the recommended option each time.

**Measured** (104 documents, 8,861 parent chunks; the shipped matcher; no model):
- **Provenance.** 13 rows are hand-made and are the only ones on the graph. 344 were created in
  one promotion on 2026-07-05; 113 of them are a keyword of no document in today's extraction.
- **Spread.** 197 of the 344 occur in the prose of at most one document, 30 in none; 18 occur
  nowhere in the text (`comput vis`, `koonce emerson`).
- **Neighbours.** Of 161 one-word labels with at least 5 prose mentions, 22 are followed by the same
  word in at least half of them: 16 are the front of a longer term (`pose` → "estimation" 326 of
  498) and 5 are surnames followed by "et al.". The user's three examples do not resolve this way:
  "viral vector" is 7 of 103 uses of `viral`.
- **Forms.** `distillation` alone reaches 7 of the 9 documents counted for
  `knowledge distillation`. 89 of the 99 `passage ranking` mentions are a benchmark's name. "Cre
  recombinase" occurs once against 255 for `Cre`.
- **The cut.** 113 of 1,103 concept–document pairs exist only past the bibliography cut. The cut
  drops everything after the References heading: in 39 of 76 cut documents that includes content
  sections (appendices, methods, sections emitted late by the PDF extractor), about 555,000
  characters or 4.5% of the library — text the keyword extractor, the definition scan and the
  written-form vote never see.

**Decided by the user.** (1) The 13 are the concepts; the 344 are terms. (2) Name plus exact and
broad forms. (3) The names-and-forms build comes before decision 4's signals, which then propose
into it. (4) Reference-list mentions stop counting as presence after the next release, in one
measured change with the cut's fix; the release notes state the limit.

**Rejected.**
- Keeping all 357 as concepts and cleaning them with signals: 344 reviews to reach the state the 13
  already have, with the meaning features running over unread strings meanwhile.
- A second table for terms: ADR-018 reserved that for vocabularies that differ in shape, and the
  difference found is membership, which its flag already records.
- A display name with matching unchanged: `distillation` would keep counting in full.
- Fixing the cut before the release: it moves keywords, definition candidates and written forms
  at once.
- Writing the snapshot's numbers into the ADR alone: a decision file cites its evidence, so the
  counts went into a baseline a reader can check.

**What it opens.** Row 94 is gated on the user accepting ADR-054. The 31 aliases of the 13 need
the user's exact-or-broad call, and the effect on presence, edges and gaps is unmeasured until then.
The snapshot's counts are observations, not graded signals: whether a neighbour share or an "et al."
share separates a term from a concept needs the user's labels on a sample (decision 4). Found on the
way: Manage keywords deletes a concept row and its definitions with no confirmation — in row 94.

---

## 2026-10-01 — The pip-audit gate's first catch: three upgrades for eight advisories published overnight

**What changed.** `uv.lock`: sentence-transformers 5.5.1 → 5.6.0 · urllib3 2.7.0 → 2.8.0 ·
virtualenv 21.3.3 → 21.7.13, and python-discovery 1.3.1 → 1.6.1, which virtualenv 21.7.13
requires. Each named package is pinned to the smallest version that clears its advisories
(`uv lock -P <package>==<version>`). Read package by package: 228 packages before and after, none
added, removed or downgraded, every artifact from `files.pythonhosted.org`, and no new dependency
edge (python-discovery drops its `platformdirs` edge).

**Why.** CI on `a5bb634` (the written-case commit below) failed in the pip-audit step and nowhere
else: eight advisories were published after the gate went blocking the day before (2026-09-30 (2)).
One is in sentence-transformers (CVE-2026-68770: `trust_remote_code=False` is bypassed when the
model path exists on disk, so Python files inside a model directory run at load). Three are in
urllib3 (an HTTPS proxy's TLS settings mixed with the target server's; two ways a server can stall
or bloat a streamed response). Four are in virtualenv, which is here only through pre-commit (a
downloaded wheel was not verified; a prompt or a path could reach `pyvenv.cfg` and the activation
scripts unescaped). All eight have a fix release, so none went into `pip-audit-ignore.toml`.

**Measured.** The gate on the tree before the change: 8 unreviewed advisories, exit 1 — the list
CI printed. On the upgraded tree: 196 packages audited, the 5 reviewed ignores matched, 0
unreviewed, exit 0. **The embedder and the reranker give identical output on both versions:**
through the app's own factories (`embeddings.get_embeddings`, the `CrossEncoder` the pipeline
builds), on the CPU, 8 texts and 1 query embedded with bge-base and 8 query–text pairs scored with
bge-reranker-base differ by at most 0.0 between sentence-transformers 5.5.1 and 5.6.0. A second run
on 5.5.1 also differed by 0.0, so the comparison is repeatable. The unit + integration suite on the
dev venv: **2,561 passed, 0 failed**, coverage 92.7%.

**Not done.** No trial freeze. The sidecar spec collects sentence-transformers whole
(`collect_all`, `scripts/doc_assistant_api.spec`), the lock adds no package and no edge, and the
release runbook freezes and smoke-tests the bundle before any release. CI's CPU venv was not
rebuilt locally; the push's CI run is that check.

**Rejected.**
- Reviewing the eight into the ignore file: each has a fix the lock takes without moving anything
  else.
- An unpinned `uv lock --upgrade-package`: on 2026-09-30 it pulled anthropic 1.9, langsmith 0.14
  and three new packages for the same kind of fix.

**What it opens.** The gate reads a live advisory database, so `main` can turn red on a push that
changed no dependency — here one day after the gate became blocking. That is what S-8 asked for,
and the red run was seen only because the session looked (KI-58's shape). S-12 (Dependabot pull
requests, and the alerts toggle) is the planned control that would raise a new advisory before a
push finds it; this run is the first data point for that call.

---

## 2026-09-30 (3) — A label keeps its written case (ADR-053 decision 3); the API refuses a foreign Host (S-5)

**What changed.**
- **`knowledge/written_forms.py`** (new) + the derived table **`concept_written_forms`**: the full
  skeleton build votes, for every surface form of every text-bearing concept, how the library
  spells it — whole-word uses in body prose (`definitions.document_sentences`, now public), not at
  a sentence start; each document votes for its most-used spelling, most documents win. Stored
  beside `Concept.label`, never into it (ADR-043; a build test holds the label byte-identical), and
  on each `ConceptNode` as `written` (in `skeleton.json` only when set).
- **One matcher for all four callers** (`concept_skeleton.form_matcher`): presence, the gap list's
  claim attribution, definitions (passages and usage examples) and epistemics. A form written in
  lower case matches in any case, exactly as before; a form written with a capital matches only
  spellings that differ from it in **word-initial** letters (`word_case_key`). `surface_forms` is
  public now; substring mode, the RG-008 A/B lever, ignores written forms.
- **Display:** the graph payload carries `written`, and the Graph tab shows it on the nodes, the
  panel heading and the rail (`lib/graph/labels.ts`, tested).
- **S-5, the host guard:** `create_app` adds `TrustedHostMiddleware` last, so it runs first; hosts
  from `DOC_API_ALLOWED_HOSTS`, else `127.0.0.1,localhost`; a blank setting means loopback, never
  every host. `tests/conftest.py` allows the `TestClient`'s `testserver`; `docker-compose.yml` and
  `.env.example` name the setting.
- `GLOSSARY.md` C-013 *written form*; ADR-053 amendment 2026-09-30; `docs/knowledge-layer.md`
  (presence row); `docs/security.md` (S1, S-5, floor row 6); ROADMAP rows 60 and 93.

**Why.** The user's labels (2026-09-22): "din is not the same as dIN. Important of being
case-sensitive." Presence feeds `single_source`, the trust table's one trustworthy gap signal, so
two words sharing a lower-cased label were being counted as one concept. S-5: the API listens on
loopback, but DNS rebinding lets a web page reach it under a hostile name (S1, T2).

**Measured** (`tests/eval/baselines/written_forms_2026-09-30.md`, read-only, the working library):
367 written forms, 155 with a capital. Presence moves for **16 of 357 concepts, all losing
documents**: nine lose another word, a surname or OCR noise (`CRE` from `cre`, "vitamin Din" from
`din`, the numpy paper's author Colbert from `colbert`, "StS" initials from `sts`), seven lose the
same name lower-cased, mostly in bibliography titles (`gpt-4`, `deeplabcut`). On the graph only
`cre` moves, 7 → 6 documents. **The 24 labelled definition candidates are unchanged (11 usable)**;
the definitions layer's one change is that "vitamin Din" is no longer a usage example of `dIN`.
Nothing moves in the stored graph until the user rebuilds. S-5 checked live: the running API
answers a foreign `Host` with 400, both loopback names with 200, and the desktop dev app loads
through it. The Graph tab showed `Cre`, `DBS`, `Ntsr1` with the graph response rewritten in the
page, since the stored skeleton predates written forms.

**Rejected.**
- The most-used spelling matched exactly (R1, 27 concepts move): `assistant` fell from 10 documents
  to 4 and `plateau` from 12 to 2 — ordinary words capitalised just over half the time.
- Occurrences voting: one paper repeating `PERSONA` outvoted the three that say `persona`.
- Exact spellings for anything but a capitalised word (R2, 20 move): Title Case quoted from titles
  won the vote, and prose such as "a sentence encoder LSTM" stopped counting. The word-initial rule
  restores those four and keeps every correct split.
- Rewriting `Concept.label` to the written form: ADR-043 — derived data sits beside curated data.
- Deriving the written form inside each caller: epistemics sees one text at a time and the usage
  examples read six documents, so each would vote differently. One stored vote, one rule.
- Showing the written form everywhere in this change: vocabulary-search hits and the gap list read
  their label server-side from other payloads; left as a follow-up.

**What it opens.** The seven remaining losses mostly come from reference lists: presence reads the
whole text, the vote only body prose. Matching body text only would remove them, and would also
stop reference-only mentions counting — its own measurement. ADR-053 decision 4, the abbreviation
signal, now has its input (141 labels with a capital). The user's rebuild applies the 16 changes;
the baseline lists each one with what stopped counting.

---
