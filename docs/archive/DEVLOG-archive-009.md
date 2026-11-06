<!-- status: archived · updated: 2026-11-06 · class: append-only -->

# DEVLOG — archive 009 (2026-10-24 → 2026-11-05 (2))

Rotated DEVLOG entries — moved here from `docs/DEVLOG.md` once it held more than 20, **verbatim** per cpc ADR-023 rule 13b and cpc ADR-053. Newest entry on top. Append-only; never edited.

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

## 2026-10-27 (3) — The backend ships as a folder: 9–10 s to a working window instead of 72 (KI-73, part 2)

**What changed.**
- **Build.** `scripts/doc_assistant_api.spec` builds one-dir (`EXE(exclude_binaries=True)` +
  `COLLECT`); `scripts/build_sidecar.py` copies `dist/doc-assistant-api/` to
  `apps/desktop/src-tauri/backend/` (gitignored), which `tauri.conf.json` ships as
  `bundle.resources` instead of `bundle.externalBin`.
- **Shell.** `lib.rs` `bundled_backend` finds `backend/doc-assistant-api.exe` under the resource
  folder and starts it with the shell plugin's `command(path)`: the same launch token, parent PID
  and KI-71 tree-kill as the sidecar had, and no console window. Without the folder (dev) it says
  so and starts nothing, as before.
- **Capability.** The window's `shell:allow-execute` for the sidecar is gone — Rust starts the
  backend, and capabilities do not govern Rust. The window holds `core:default` and
  `dialog:allow-open` only (`test_desktop_security_config.py`, `docs/security.md`).
- **Installer.** `src-tauri/nsis/installer.nsi` is Tauri CLI 2.11.4's own NSIS template with one
  line changed, `SetCompressor` without `/SOLID`; `bundle.windows.nsis` sets it and `zlib`.
- **Preflight.** `backend_contents` replaces the one-file size floor: the folder must hold
  `_internal/pymupdf/layout/resources/onnx` with its `.yaml` files (what KI-34 lost) and
  `_internal/hf_cache/hub`, and weigh at least 2,490 MiB (built: 2,515.9 MiB, 17,855 files).
  KI-34's 17 MB is under the 1% margin, so the paths, not the floor, catch it. 3 tests replace 1.
- **Local gates.** The built folder puts third-party `.py` files under `apps/desktop/src-tauri/`
  (`backend/`, and its copy in `target/`), so CI's own `mypy src/ apps/` and `bandit -r src/ apps/`
  failed on this PC after a build; `pyproject.toml` excludes `src-tauri` from both (CI never has
  the folder). `.secrets.baseline` records the template's one detect-secrets hit, a registry path
  (`SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate`, line 621) read as high-entropy text.
- Paths in `clean_build.ps1`, the justfile and `docs/desktop-packaging.md`.

**Why.** KI-73: the one-file backend unpacked 1.6 GB to a temp folder at every launch — 29.4 s
of its 38.5 s to health alone, 52 s of 72 with the window. Part 1 (entry (2)) filled the wait;
this removes most of it.

**Measured** (dev PC, scratch data folder, `/api/health` polled every 200 ms). The backend alone:
38.3 and 38.8 s one-file, 6.1 and 7.3 s one-dir. The built app, window start to health: 72.2 s
one-file; one-dir 28.9 s on the first launch after the build (Windows scanning the new files),
then 9.3 and 9.9 s. One backend process each time, and nothing left after the window closed.
Two runs per arm: indicative, not a benchmark. Installer: 1,642,519,979 bytes in ~7 min (one-file:
1,574 MiB); installed, 2.5 GB instead of 1.6.

**The installer detour.** The first build failed in makensis: "Internal compiler error #12345:
error mmapping datablock". Tauri's template sets `SetCompressor /SOLID`, which gathers every
file uncompressed in one block, and the 32-bit makensis cannot map one past ~2 GB — with LZMA
and with zlib. An `installerHooks` file cannot change it ("can't change compressor after data
already got compressed or header already changed!"), and Tauri has no non-solid setting. So the
template: extracted from `cli.win32-x64-msvc.node` (978 lines, CRLF, 86 handlebars tags) and
checked against the script Tauri generated for this app — a sorted-line diff differs only in the
compressor line, its configured value, comments and the include of the hook file tried first.

**Rejected.** fp16 models to shrink the folder: changes the retrieval numbers, so it needs an eval
run first. An MSI (WiX): a per-machine, admin install, and RG-012 installs the NSIS build
silently per user. One-file kept, with an archive unpacked on first launch: the same files on
disk plus a first-launch path to build and test. A hook overriding the compressor: makensis
refuses it.

**What it opens.** The template is a copy: on a Tauri CLI upgrade, re-take it and re-apply the
one line (its header says so; `test_the_installer_compresses_file_by_file` fails if `/SOLID`
returns). 2.5 GB installed. The release build and RG-012 rerun come from this commit.

## 2026-10-27 (2) — The window shows the last session at once: a start screen, then the library, chats and graph until the engine answers (KI-73, part 1)

**What changed.** Committed with the Settings fixes in `a5d3231`; this entry and its CHANGELOG
line follow. `index.html` draws a start screen inside `#app` ("Starting Provenote…", the icon)
styled by `public/boot.css` — a file, since the window's CSP admits only its own; it colours
`html` in the system's light or dark `--bg`, which the app's `:root` rule outranks at mount, and
`main.ts` clears `#app` before `mount()`. `lib/core/lastSession.ts` (pure, 4 tests) keeps a
versioned copy of a list in the window's `localStorage` and reads it back; a copy over 2,000,000
characters is not kept and an older one is removed. The library documents, folders, chat list and
concept graph save a copy whenever they load for real; at launch `App.svelte` shows them
(`restoreConversations`, `restoreGraph`) with the notice "Showing your library as it was last time
while the engine starts. Changes work once it is ready.", and the readiness gate refreshes them
all once `/api/health` answers. A never-built graph is not kept. Before the engine answers, the
Graph tab no longer fetches (it would show an error) and reads as loading rather than "No
concept graph yet".

**Why.** The installed app takes 38–72 s before its backend answers (KI-73: 29–52 s of one-file
unpack, ~7.5 s of imports), and the window drew an empty shell meanwhile. The user, in the
walkthrough: load the library, past chats and graphs before the rest.

**Checked.** Live in the dev app over the 104-document scratch copy: the copies came to 75 KB
(documents), 20 KB (chats), 19 KB (graph). With the API stopped and the page reloaded, the Library
grid, the chat list and the Graph ("Covers 29 of your 104 documents") came from the copies under
the notice, with "starting the engine…" in the status bar; started again, the window went ready by
itself 16 s later, the notice went, and every failed request was from the down period. The start
screen is served, and nothing of it is left after mount.

**Rejected.** IndexedDB: a larger store for lists that fit in a few hundred KB, and a second async
path at startup. Disabling every write while the copy is on screen: many controls for a wait the
packaging change cuts to seconds; the notice says changes work once ready. A window colour in
`tauri.conf.json`: one fixed colour cannot follow the system theme, and in light mode it is the
same white.

**What it opens.** At 75 KB per 104 documents the cap is reached near 2,800 documents; a library
past it starts as before. Part 2 of KI-73: package the backend as a folder.

## 2026-10-27 — Setting up an answer engine from Settings: an instant panel, verdicts in place, no impossible model (KI-72)

**What changed.**
- **Server.** `llm.model_mismatch(provider, model, installed)`: the one rule for a pair that can
  never answer — a Claude name on Ollama, any other name on Anthropic, or a model an answering,
  probed Ollama lacks (`llama3` and `llama3:latest` are the same). `app_settings.set_llm_selection`
  refuses the first two (names only, no network) and no longer tells an installed-app user to
  "add it to .env". `readiness._with_selection` marks the active engine not ready when its
  selection is impossible, so *Getting started* names it and says what fixes it.
- **Getting started** (`ProviderSetup.svelte`) draws from `GET /api/setup?probe=false` first and
  replaces it with the probed state; while the probe is out, the Ollama card reads *checking…* and,
  when Ollama is the active engine, its step reads "Checking whether Ollama is answering…"
  (`setup.ts` `displaySteps`, `ollamaPending`). Each card has its own result line under its own
  buttons; a refused key reads "Anthropic refused this key, so nothing was saved. <reason>"
  (`keyRefusal`). The Ollama model box is seeded with a model Ollama can serve, and *Use Claude
  for answers* also shows when the active selection is the broken one.
- **Provider & model** (`Settings.svelte`): a keyless provider gets a line and a button to
  *Getting started*; changing the provider fills in a model it serves (`modelForProvider`); errors
  read without "Error:"; the hint no longer says *Getting started* is "above". After a switch from
  either section the status bar's health is re-read: it kept naming the old model.

**Found by:** the walkthrough's §0 in a clean sandbox (the user). *Getting started* said
"Checking…" until the Ollama probe gave up (4.1 s there, against 0.5 s without the probe); a key
seemed not to register — the verdict was drawn at the panel's foot, under the Ollama card; *Provider
& model* could not reach Anthropic, and the sandbox's own `settings.json` held `ollama` with
`claude-haiku-4-5-20251001`. Checked then: the installed app reads its key from the environment or
the data folder's `credentials.json` and its selection from `settings.json`, both hand-editable, and
the `.env` the dev setup uses is looked for beside the unpacked program, so it never applies —
Settings is in practice the only door.

**Checked.** Server tests (rule, refusal, readiness; two that matched the old wording updated);
frontend helpers (3 tests). Live in the dev app over a scratch folder seeded with the stuck
selection: the panel drew in 69 ms with Ollama *checking…*, the probed verdict named the Claude
model on Ollama at 2.4 s, *Use this model* offered `llama3.1:8b` and fixed it; switching provider
filled `claude-haiku-4-5-20251001` and back `llama3.1:8b`; the Claude-on-Ollama pair was refused with
its sentence; the status bar followed both switches. A refused key's line appeared under the key
field — the request intercepted in the page, so no key, fake or real, was sent.

**Rejected.** Checking with Ollama at switch time whether the model is installed: it puts the
network in the save and makes the tests depend on the box; *Getting started* reports it instead.
A key field in *Provider & model* too: two places to enter a secret; one button to the existing
one. Clearing a refused key from its box: it is masked, and keeping it lets a typo be fixed.

**What it opens.** The real key flow on the installed build (the user's own key, in the sandbox).

## 2026-10-26 (4) — Closing the app ends its backend; the backend stops when the app is gone (KI-71)

**What changed.** `apps/desktop/src-tauri/src/lib.rs`: the shell keeps the sidecar's
`CommandChild` (`Sidecar`, managed state) and on `RunEvent::Exit` ends its process tree:
`taskkill /PID <pid> /T /F`, by full path under `%SystemRoot%\System32`, with `CREATE_NO_WINDOW`;
`kill()` when that fails or off Windows. `run()` now builds the app and passes an event handler.
It also sets `DOC_PARENT_PID` to its own process id. `apps/api/parent_watch.py` (new): a daemon
thread waits for that process (on Windows on its handle, `WaitForSingleObject`; elsewhere by
polling) and `__main__` then sets uvicorn's `should_exit`, with a forced exit 15 s later if a
pool or a stream holds the shutdown open. `__main__` now runs `uvicorn.Server` directly to reach
`should_exit`. `rustfmt` also wrapped one existing line in `launch_token`.
`tests/unit/test_api_parent_watch.py`: the pid parsing, a real process watched before and after
it ends, `main` watching only when a pid is named, and the Rust contract pinned in its source.

**Found by:** the 0.7.0 walkthrough in the RG-012 sandbox. After the window closed, both
`doc-assistant-api` processes — the one-file launcher and the server it starts — still ran and
held 8001. A reopened window's own backend could not bind, so the window read through the old one
with its new launch token, and *new folder* was refused ("needs the token the app was started
with"). A plain launch from the desktop icon did the same: on any machine, the second launch could
change nothing until a reboot. The cause: the shell plugin ends on exit only the children the
window spawns (`tauri-plugin-shell` 2.3.5 `lib.rs`), and `lib.rs` dropped the handle at spawn.

**Checked.** The new tests (10). `cargo check` and `cargo fmt --check`. The server half end to
end: `python -m apps.api` with `DOC_PARENT_PID` naming a stand-in process — healthy after 16 s,
then `desktop_shell_gone` and a clean shutdown 1.0 s after the stand-in was killed. The shell half
needs the installed build: the sandbox after the rebuild.

**Rejected.** Killing only the launcher (`CommandChild::kill`): a one-file launcher's server may
survive it, which is the state we were in. A single-instance lock instead: it would stop a second
window but not free a backend whose window had closed. Accepting any token from a backend that
outlived its window: it would re-open the hole S-6 closed.

**What it opens.** A second Provenote window opened while one runs still gets a backend it cannot
write to (its own cannot bind 8001); a single-instance guard is the follow-up.

## 2026-10-26 (3) — A failed turn says why; quotes drop the extractor's emphasis markers; two claims corrected

**What changed.** From a pre-launch check of the 0.7.0 rows of `docs/release-ux-checklist.md`, run
in the dev app over a scratch copy of the working library (`DOC_DATA_DIR`, nothing written to
`data/`):
- **A failed turn ends with a sentence.** `chat_controller.events.Failure` is a terminal event:
  `handle_message` wraps the answer path and yields one when it raises, its text from
  `helpers._turn_failure_sentence` (engine unreachable, by exception name through the cause chain:
  "No answer: Ollama is not answering at <host>…"; another failure: the provider, model and
  exception type, never its message). The router maps it to an SSE `error` event, and its worker
  thread catches anything that escapes the controller (`_WORKER_FAILED`); the window shows the
  sentence on the turn, and a stream that ends with no answer, result or error says so
  (`EMPTY_TURN`). The CLI prints it; `measure_latency` exits with it. Tests: the controller turn
  with `httpx.ConnectError`, the sentence rules, and two SSE tests that fail without the router
  change.
- **Quotes read as text.** `graph/emphasis.ts` marks the characters that are emphasis delimiters
  (`**` runs, an `_` outside a word); `ConceptDefinition.svelte` leaves them out of every quote and
  passes them to `markedSegments`, which now takes a drop set so the server's highlight offsets
  still land on their words. Display only (ADR-043). Five frontend tests.
- **Settings' Ollama card** says its 36% / 14% citation figures predate KI-66's fix.
- **`CHANGELOG`**: "known junk is gone" becomes "most"; the keyword limit lists what remains.

**Found by the check.** With Ollama stopped, a question got nothing under it: the worker thread
died on `httpx.ConnectError` and the stream ended with only `done` (release checklist §6, the
user's row). The Definition card and term panels showed `_**Jakobs et al., 2019**_` verbatim.
On the library's 1,560 keywords, about 50 are not terms in classes the filters do not cover:
own author names from title pages and headers (23 in 14 documents, a few false hits), "Cell Press"
split by letter-spacing (`ce ress`, `ll primer`), accents (`g ttingen`, `ram n`), possessives
(`cajal s`), notation (`t cat`) — which the CHANGELOG's "letter-spaced text … no longer" denied.
The Ollama card stated the pre-KI-66 rates unqualified. Also filed, not fixed: KI-68 (the
staleness notice names the concept, not the form), KI-69 (no sentence for zero `is_a`
proposals), KI-70 (rail names cut at 800 px). Passed: re-ingest hint; Manage keywords at size
(95 ms), forms, exact/broad, the delete preview; staleness → Rebuild (7 s); the Definition card's
every action; term signals (5 cited names, 31 offers, 26 places over 333 rows); proposals accept
and reject, an accepted placement surviving a rebuild; KL1 labels; S-5; Settings with the
engine down.

**Rejected.** Rendering emphasis as `<em>`/`<strong>`: italics would survive, but a quote is
plain text everywhere else and the markers sit mostly on citations. A fallback to another
provider when the engine is unreachable: it would spend money the user did not choose to.

**What it opens.** Row 115 still wants Settings to take another Ollama address (ROADMAP 100).
Keyword classes above: ADR-056 (letter-spacing) and ADR-057 (accents) are parked work.

## 2026-10-26 (2) — The API answers the Windows window's origin, and RG-012 asks as the window does (KI-67)

**What changed.** `apps/api/main.py` `_ALLOWED_ORIGINS` gains `http://tauri.localhost`, the origin
Tauri 2 serves the window from on Windows; `tauri://localhost` (macOS, Linux) and the two dev-server
origins stay. `tests/integration/test_api_window_origins.py` pins a read and a token-carrying write
from both packaged origins, and that any other origin gets no `Access-Control-Allow-Origin`; its two
Windows cases fail on the old list. The RG-012 harness (`scripts/rg012/rg012-run.ps1`) asks
`/api/health` with `Origin: http://tauri.localhost` and logs `window origin …: allowed|REFUSED`;
REFUSED fails its packaging verdict. `release_preflight.rg012_packaging` fails a run whose line says
REFUSED and a run with no line at all (every run before this one); `test_rg012_harness` pins the
line against the reader and the harness's origin against the API's list. `CHANGELOG` [0.7.0] Fixed
leads with it.

**Found by:** the walkthrough's first look at the installed 0.7.0 build, in the RG-012 sandbox after
a passing run (`run-20261026-104551`: 3 of 3 answered, 3 of 3 cited). The window "did not respond,
Settings included". A read-only diagnostic run inside the sandbox (`wsb exec`, at the user's go)
showed otherwise: Windows called the window responsive, the webview was idle, and the screenshot
showed "Loading…" and the status bar's *still starting — retrying*, while `/api/health` answered
200 in 0.5 s from the same box. `/api/health` with `Origin: http://tauri.localhost` came back without
`Access-Control-Allow-Origin` (with `tauri://localhost`, with it), and the window's storage was
written under `http://tauri.localhost`. So the readiness loop in `App.svelte` never saw a health
answer it could read. 0.6.0 had the same list on the same Tauri (2.11.3): its installed window
cannot have worked on Windows either, and KI-39's report was probably this. RG-012 missed it on
every release because PowerShell ignores CORS.

**Also seen, not fixed here.** The window process running at the diagnostic had started at 10:56,
while the backend still running had started with the run at 10:49, so a backend outlived its first
window (`lib.rs` never stops the sidecar). A window opened next would talk to that old backend with
a different launch token, so its writes would get 401. The walkthrough's §6 row "closing the window
ends the sidecar process" checks it on the fixed build.

**Rejected.** `allow_origin_regex` for any `*.localhost`: wider than the two origins Tauri uses. Routing
the window's calls through a Rust-side HTTP client to skip CORS: a new plugin and permission surface
for what one list entry fixes. Checking every packaged origin in RG-012: the sandbox is Windows, so it
can only see the Windows window's.

**What it opens.** The release's walkthrough can now be driven at all. Nothing yet checks the
macOS/Linux origin on a real window; no installer is built for either.

## 2026-10-26 — Every Ollama call asks for an 8,192-token window: answers no longer lose the instructions to cite

**What changed.** `config.OLLAMA_CONTEXT_TOKENS` (8,192) is passed as `num_ctx` to the streamed
answer (`pipeline.build_chat_model`'s `OllamaLLM`) and to every one-shot client
(`llm.make_client`: reviewer, eval judge, taxonomy, gap suggestions). The wiki narrows it through
`fit_window` as before. Two tests pin it, one per path. `CHANGELOG` [0.7.0] gains the fix and
moves to 2026-10-26, the date the release commit will carry; README's local-model figures are
marked as predating it.

**Found by:** RG-012 Tier-2 for 0.7.0 (run `run-20261026-090139`). Packaging passed on a clean
sandbox, and 0 of 3 answers cited anything; preflight read all three as *uncited*. Ollama's own log
gave the cause: `vram-based default context … default_num_ctx=4096`, then `truncating input prompt
limit=2050` on the answer prompts of 4,032–5,176 tokens (10 sources). On overflow llama.cpp keeps
only the end, so the system prompt, which carries the instructions to cite, never reached the
model, and the call still succeeded. It is KI-65's cause, on the answer path: `828fddc` gave only
the wiki a window.

**Checked.** The same three questions on the host, over a scratch library of the same three PDFs
(322 chunks), with the dev API on Ollama `llama3.1:8b`. Before: 0 of 20 sentences cited, and four
truncation warnings. After: all three answers cite, 11 of 13 sentences ([1]–[5]; [5]; [3], [4],
[6], [9]); Ollama starts its runner with `-c 8192`, and every prompt runs whole (`truncated = 0`);
0.6 GB more VRAM (5,675 against 5,035 MiB).

**Rejected.** Sizing the window per prompt from an estimate (the wiki's `material_budget`): the
answer prompt is bounded by the locked `TOP_K`, and one fixed window keeps Ollama from reloading the
model between the answer and the reviewer. Leaving it to `OLLAMA_CONTEXT_LENGTH` on the server: an
installed app's user would have to know to set it.

**What it opens.** KI-36's local citation rates (`llama3.1:8b` 37.6%, `qwen2.5:7b` 18.0%,
2026-08-07) were measured with prompts that were probably being cut, so they likely overstate the
gap to Haiku: re-measure. Nothing yet warns when an answer prompt exceeds 8,192 tokens.

## 2026-10-25 — 0.7.0 release commit: the version in nine places, a dated CHANGELOG, and no developer command in the re-ingest dialog

**What changed.** `docs/RELEASE.md` §1–§2. The version reads 0.7.0 in all nine places
(`pyproject.toml`, `src/doc_assistant/__init__.py`, `package.json`, `tauri.conf.json`,
`Cargo.toml`, and the three locks re-locked by their own tools, `uv lock`, `npm install
--package-lock-only`, `cargo update -p doc-assistant-desktop --offline`). `CHANGELOG.md`: the
Unreleased section becomes `[0.7.0] — 2026-10-25`, with a line for the libraries upgraded over
advisories during the release. `README.md` Status names 0.7.0 with its date and today's test count
(2,776 Python, 299 desktop). The 0.7.0 notes are drafted locally; their install size, hash and
verification are filled in the release's second session.

**Found on the way, and fixed here.** The re-ingest dialog, changed in session 5c, told the user to
run `python -m scripts.extract_keywords --apply --force` — a command an installed app has no Python
or checkout to run. It now says the app does not run those passes yet and that they are redone from
a source checkout. Behind it: nothing in `apps/` or `src/` calls `extract_keywords`, so keyword
extraction exists only on the command line — a library the installed app builds has none, and one
carried over from 0.6.0 keeps its old keywords, listed alphabetically. The CHANGELOG line and the
README limitation for keywords now say so. `release_preflight`'s `dev_commands` check reads `just`
recipes only, which is why it passed this.

**What it opens.** Keyword extraction inside the app (an action, or a step of ingest) — after the
release, with the LLM scan design (ADR-058) that would replace its candidates.

## 2026-10-24 (3) — Keyword cleanup (ROADMAP 99): junk filtered, keywords listed by rank, and the first precision baseline shows no gain

**What changed.** The code is `344b3d6`. `knowledge/keywords.py` has five filters, each with a
unit test of what it drops and keeps. They drop a `br` token (table residue), a phrase with two or
more one-letter tokens (letter-spaced text, initials), a token with no letter (pages, volumes,
counts, years), and a surname the document itself cites with "et al." on at least half of its uses
(`cited_authors`, the term signals' reading; `majority` moved here and is re-imported by
`term_signals.py`). Web addresses and DOIs are removed before tokenising (`strip_urls`).
`document_keywords` gained `score` and `rank` (additive, through `init_db`; old links read NULL).
`Document.keywords` is ordered by rank, so the Library lists a document's keywords best first.
`scripts/extract_keywords.py` runs `init_db()` before `--apply`. The re-ingest dialog no longer
sends keywords to Settings: Rebuild on the Graph tab redoes gaps, and keywords are redone from the
command line. The working library was re-extracted 2026-10-22 (backup
`data/library.db.bak-20261022-pre5c`). This entry adds the baseline,
`tests/eval/baselines/keyword_cleanup_2026-10-23.md`. **Found by:** the 2026-10-20 assessment:
keyword scores computed and thrown away, about a quarter of keywords noise in a 12-document probe,
precision never measured.

**Measured.** The five patterns went from 123 links to 0 across 104 documents. Every document has 15
keywords and none has fewer than before. Nothing the graph reads moved (row hashes, graph version,
the 22 marked forms). On 15 seeded documents labelled by the user, precision was 62% [46, 78]
before and 64% [50, 77] after, +1 point [−4, +6]; with the user's own labels only, +0 [−4, +4]. The
filters took out 44 keywords, 37 of them noise, but 34 of the 44 that replaced them were noise too.

**Why.** The probe found mechanical junk worth removing, and an unordered list hides which
keyword is the document's main subject. The baseline is what any later keyword change is
measured against.

**Rejected.** A list of names (`cajal`, `dbs`, `cre` are real vocabulary here); dropping
abbreviations outright (ADR-055 shows the long form instead); keeping numbers inside a phrase
(`type 2 diabetes`): none was seen, so the rule stays simple, and the cost is recorded (`sam 2`,
`mrr 10`); replacing `<br>` before tokenising, which would join words across table cells; a
percentile bootstrap at 15 documents.

**What it opens.** Filtering cannot raise precision while the candidates stay TF-IDF windows, so
better candidates come next (ADR-058, an LLM scan measured on these 15 documents). The label rule
gains a *Too vague* class: the user's point is that `connexion` is a term but not a precise one.
Also open: ADR-055, ADR-056 point 8, and two gaps in the baseline, alternating recto/verso running
heads under the furniture floor and accent-split names (the tokeniser is eval-gated).

## 2026-10-24 (2) — `filelock` 4 and OpenTelemetry 1.45.1 upgraded; a `setuptools` advisory reviewed

**What changed.** `uv.lock`: `filelock` 3.29.0 → 4.2.4 (a new major; 4.1.0 for the Python 3.10 fork)
and the OpenTelemetry stack 1.42.0 → 1.45.1, which `chromadb` pulls in (`uv lock
--upgrade-package filelock --upgrade-package opentelemetry-api`; one exporter package added).
`pip-audit-ignore.toml`: GHSA-xmj3-9gf7-h52w on `setuptools`, reviewed. **Found by:** the local
audit while staging the entry below — `filelock` CVE-2026-112907, -112913 and -112940;
`opentelemetry-api` CVE-2026-114603; the `setuptools` advisory (low).

**Why.** `filelock` guards the Hugging Face model cache and torch's caches, so its major was tested
where it is used: the embedder and the reranker load offline from the local cache (768
dimensions; 0.999 against 0.0 on a matching and an unrelated pair); both Chroma stores open on the
new OpenTelemetry with their full counts (18,012 and 45,581); the whole suite passes, 2,216 unit and
560 integration tests. The `setuptools` fix (84.1.0) is blocked by torch 2.12's `setuptools<82`,
and the advisory concerns the legacy wheel-to-egg installer, which nothing here calls — the
reason and what would reverse it are in the ignore entry.

**What it opens.** The sixth advisory round since 2026-10-20 and the second `setuptools` entry held
by torch's cap; both ignore entries reverse when torch lifts it.

## 2026-10-24 — `dompurify` 3.4.17: a critical advisory in the app's answer sanitiser

**What changed.** `apps/desktop/package-lock.json` only: `dompurify` 3.4.16 → 3.4.17
(`npm audit fix`, inside `package.json`'s `^3.4.15`). **Found by:** CI's frontend audit on
`344b3d6` (run 40041172523) — three bypasses rated critical together (a template-expression
bypass in shadow-root output, armed handlers left on a detached subtree, an SMIL `href` guard
skipped under an XHTML parser type).

**Why.** Unlike the `vite` advisory, this one ships: DOMPurify cleans every chat answer before the
one `{@html}` in `Markdown.svelte` (security S-3). Checked: `npm audit` 0; `svelte-check` 0 errors
over 229 files; 299 unit tests; a production build; `tests/unit/test_desktop_security_config.py`
(6). **Not checked here:** the sanitiser against hostile input in a DOM — no DOM library is
installed and the browser pane does not run local scripts; the release walkthrough's S-3 row
drives exactly that in the installed build.

**What it opens.** The same run failed one worker test on Linux
(`test_the_frozen_guard_can_be_lifted_deliberately`, `BrokenProcessPool`), which `344b3d6` cannot
reach (no module it changed is on the worker's import path) and which passes 3 of 3 here; the run
on this commit shows whether it is intermittent.
