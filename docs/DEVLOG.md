<!-- status: active · updated: 2026-11-04 · class: append-only -->

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
> **2026-10-24 → 2026-10-31 (2)** in [`docs/archive/DEVLOG-archive-009.md`](archive/DEVLOG-archive-009.md)
> (rotated 2026-10-31: 11 entries, and 2026-11-02: 11 more) ·
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

## 2026-11-02 (2) — ADR-060 accepted in a grill: who bills is a fact of its own

**What changed.** ADR-060's status: accepted by the user. An amendment holds the grill's ledger of 11
forks. `docs/decisions.md`'s line follows. No code.

**Why.** Seven choices stand as recommended. The user asked whether more effort on *paid* was worth
it, and checking found that a local Ollama serves cloud models from ollama.com, marked by
`remote_host` in `/api/tags` (Ollama's docs, fetched 2026-11-02). So where the text goes is now
worked out per model, and *who bills* is a separate fact: a preset's vendor, nobody, Ollama for a
remote model, or a *does not bill me* tick on a server of your own. The guard trips on billed or not
known.

**Rejected.** Real prices per turn: they need a price catalogue and usage reports. `temperature` per
preset: one fact per vendor to keep current, for reviews that are unmeasured anyway. Editable preset
addresses. Exported vendor keys. A model list kept on disk.

**What it opens.** Piece 2's *stays on this computer* is wrong today for an Ollama cloud model; the
build fixes it. The build confirms OpenAI's address and Mistral's model-list path first, and checks
`remote_host` on a real cloud model, which needs an Ollama sign-in.

## 2026-11-02 — ADR-060 proposed: other answer providers through one OpenAI-compatible adapter (0.7.2, piece 3)

**What changed.** `docs/decisions/ADR-060-other-answer-providers-through-one-compatible-adapter.md`,
proposed, and its line in `docs/decisions.md`. No code.

**Why.** The user asked for more providers than Claude and Ollama, and the norm is an
OpenAI-compatible `/v1/chat/completions` with `/v1/models`. The ADR decides nine things: the client,
the presets, the route that takes a secret, where the text goes, what counts as paid, what is sent,
an unknown context window, model lists, and the tests at $0. Its worked example follows one question
through Groq and then through a server of your own.

**Rejected.** `langchain-openai`: its 1.7.3 needs `langchain-core>=1.7` against the locked 1.4.9, and
brings `openai`, which is on `httpx2`, and `tiktoken`, which fetches its files on first use (PyPI,
2026-11-02). A presets file: the build freezes `doc_assistant` as code only. Looking up a name to
decide where the text goes. Ollama through `/v1`: its docs say the context size cannot be set there,
which is KI-66 again.

**Found.** Eight places name the two providers, not five. `app_settings.set_llm_selection:107`
refuses any other name. `helpers._is_local` decides *no metered cost* from the name. The cost chip
prices every non-Ollama turn at Haiku's rates. `pipeline.build_chat_model` builds an Ollama model
for any name that is not `anthropic`. `docs/decisions.md` has no line for ADR-058 or ADR-059.

**What it opens.** The user's decision on the nine choices (`grill-me` if wanted), then one or two
build sessions. A paid call only comes last, at the user's word.

## 2026-11-01 (5) — Getting started is two rows; models live in Provider & model (0.7.2, piece 2)

**What changed.** New `GettingStarted.svelte`: Model and Documents, each a step from `/api/setup` with
a status and a link to its tab, and no form. `ProviderSetup.svelte` moved under Provider & model and
gained the *Answering with* card and a provider list grouped by where the text goes, with one form
open at a time. Save, verify and remove, the `.env` note, the Ollama probe and *Expect fewer
citations* all came across. The select and Apply are gone: each form has a free-text model box behind
*Use for answers*. `readiness.py` words the rows: `SetupStep` gains `status` and `offline`, and
`ProviderReadiness` gains `name`, `group` and `destination`, mirrored in `models/setup.py` and
`types/setup.ts`. `set_llm_selection`'s refusal now names Provider & model. ADR-034 amended; README,
QUICKSTART, setup, DEMO and `.env.example` send a reader to the right tab.

**Why.** The user: "Gettings Started vs Provider/Model is still confusing on initial setup". Both tabs
held provider setup and each pointed at the other.

**Decided while building.** *Engine online* for Claude means a key is in place, since ADR-034 does not
probe it; the amendment says so. Where the text goes comes from the address, so an Ollama on another
machine reads "go to" that host and is listed under *Your own server*, and `ANTHROPIC_BASE_URL` is
honoured. Only a chosen engine that does not answer is red: an Ollama with no models and a mismatched
model read *Not set* with the reason. The row shows the model id, as the status bar does: the drawing's
"Claude Haiku 4.5" is marked as an example, and a table of display names would go stale. The drawing's
line about every cloud provider having one form waits for those providers. `costLabel` and
`displaySteps` went with their only callers. The Model row's detail joined an alternative straight on
to a sentence with no full stop; fixed.

**Rejected.** A display-name mapper; a probe for Claude; warning rows or extra buttons on Getting
started; keeping the provider select as a second way in.

**Checked.** On new empty data homes, the repository's `.env` hidden and a local verifier, so no request
left the machine. `data_home` read through the page before each write. Driven: a fresh install from
*Finish setup…* with no key field on Getting started, a refused key, a saved key, removal, Ollama
chosen, a folder indexed, and a question answered with citations on `llama3.1:8b`; green; Ollama at a
dead port, red, the link opening its form. Light, dark and 375 px, no sideways scroll, 0 console errors
in the two clean final tabs. Real `data/` unchanged. **Mocked:** the `.env` note, by one in-page
response, because the check hid `.env`. **Not driven:** the packaged build; Ollama coming back under
Re-check; *Answering with* for an Ollama off this computer, which has tests.

**What it opens.** Piece 3 adds providers: `PROVIDERS` and `group` place a row with no change to the
list, and the ADR decides the form. `step.action` for Documents has no reader left in the app; it stays
on the wire because the API's contract and tests rest on it.

## 2026-11-01 (4) — Claude models from Opus 4.7 on answer and review

**What changed.** `llm.is_older_claude` tells the models before Opus 4.7 from the rest. On the rest,
`AnthropicClient.complete` sends no `temperature`, asks for `output_config.effort: "low"` and adds
`THINKING_ROOM` (4,096) to `max_tokens`; `build_chat_model` adds the same room to the chat's 1,024.
`llm.content_text` reads the text out of a string or a list of blocks of any shape, and the three
chat reads in `pipeline.py` (`rewrite`, `stream_answer`, `expand_query`) and
`_extract_anthropic_text` all use it. Haiku 4.5 is untouched: same request, temperature 0, 1,024.
Tests: `tests/unit/test_newer_claude_models.py`, the real `ChatAnthropic` over a fake stream of the
SDK's own event types; the three chat reads failed there before the fix. 2,983 tests pass.

**Why.** Asked whether the parked `temperature` patch could wait, the check found a gap it does not
cover. The patch keys on the API version; this keys on the model, and the API reference lists
`temperature` as a 400 under `2023-06-01` on Opus 4.7 and later, Sonnet 5 and later, Haiku 5.5 and
Fable. The model field is free text, and the reviewer follows it unless `REVIEWER_PROVIDER` is set
in the environment, which an installed copy has no `.env` to do. Tracing the fix found two more
faults behind the first. From Claude Opus 5 a reply opens with a thinking block: LangChain emits it
as list-shaped chunks and `controller.py:631` does `full_answer += tok`, a `TypeError`, so the chat
fails before the reviewer runs. `_extract_anthropic_text` read `content[0]` and returned the
response's repr. And thinking spends `max_tokens`, so a 400-token review could end with no text.

**Rejected.** The reverse list, naming the models that refuse: a new model would be a 400 until
someone added it, and sending no `temperature` is valid on every model, so an unknown name takes that
side. Trying with `temperature` and retrying on a 400: it depends on an error's wording and spends a
round trip on every call. Dropping `temperature` everywhere: it moves the reviewer and judge on Haiku
4.5 from 0 to 1.0, and their numbers stop being comparable. Turning thinking off: Opus 5.5 and Fable
refuse that, and Sonnet 5.5 needs a different value, so it would be a table per model.

**What it opens.** **Not checked against the API:** no paid call was made. The model list comes from
the API reference, and `THINKING_ROOM` is a judgement: if a model thinks past it, the answer is cut
at `max_tokens` and a review reports it could not parse. Chat runs at the model's default effort.
`figures.py` forces a tool, which the reference lists as a 400 on Opus 5.5, Sonnet 5.5 and Fable 5.1;
only `FIGURE_VLM_MODEL` reaches it, so it stays. The parked SDK 2.0 patch no longer applies to
`llm.py`; when that bump comes, its version test has to combine with `is_older_claude`.

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
