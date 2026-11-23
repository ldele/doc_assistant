<!-- status: active · updated: 2026-11-23 (A8″a built: As built added; the spec's two builds done); 2026-11-23 (one list of items, not jobs: decisions 2, 4, 7 revised by the owner); 2026-11-23 (A8″b built: As built added; A8″a waits on its wireframe); 2026-11-23 (accepted by the owner as written: all ten recommendations) · class: living -->

# Spec — One queue for every ingestion, adds while the engine starts, and the letter-spacing block (session A8″)

> The executor brief. Target executor: **Claude Code**. Session A8″ of the plan
> (`docs/plans/PLAN_2026-10-31_next-two-releases.md`), tickets T-007, T-009, T-010 and T-011, with
> S-17 as its security step (`docs/security.md` §4). **Accepted by the owner, 2026-11-23, as
> written; then revised the same day:** the queue is one list of items, not a line of jobs
> (decisions 2, 4 and 7). The contract below is the build's.

## In one sentence

Every file to index and every re-run of a document's parts is one item in a single list in the
engine, run in order, so nothing is refused while another runs: an add puts its files at the end of
the list. An add made while the engine is starting waits in the window and is sent when it is ready.

## A worked example

Forty papers are indexing; the status bar reads *Indexing 12 of 40*. The person adds three more.

- **Today:** the add sheet copies the three files, then `POST /api/ingest` answers 409 *ingest
  already running*. The files sit in the library folder, not indexed, until the person finds Manage
  files and indexes them by hand after the forty finish. A re-run of one document's text, asked for
  meanwhile, is refused the same way.
- **After:** the three join the end of the list. The status bar reads *Indexing 12 of 43*; the
  indexing screen shows them as three more *Waiting* rows, each with *Remove*. The re-run joins as a
  row of its own (*survey-2024.pdf · Text & chunks*). The list runs on, in order, to its end.

## Today (read in the code and measured, 2026-11-23)

- **Three job systems, each refusing the others.** `POST /api/ingest` (`routers/sources.py:44`)
  runs `ingest.main` on a thread and answers 409 while one runs. `POST /api/documents/add`
  (`sources.py:371`) copies or registers files and answers 409 while an ingest runs; indexing them
  is the window's next call, `POST /api/ingest` with `paths`. `POST /api/library/documents/reingest`
  (`routers/library/documents.py:333`) runs `reingest.rerun` on its own thread and answers 409 while
  an ingest or another re-run runs. Each has its own status object (`apps/api/services.py`).
- **The window locks little.** Only the index buttons in Settings and Manage files are disabled
  during a run (`lib/core/ingest.ts` `indexLocked`). The chat and the Library are not.
- **The engine keeps answering, and retrieval slows.** Measured on the dev machine over a scratch
  data home (the ten public papers; three indexed, seven indexing; one extraction worker, as the
  installed build runs; probes every two seconds). With the GPU hidden, which is how the installed
  build runs (it bundles `torch 2.12.0+cpu`):

  | Request | Idle, median | During the ingest, median / worst | Failed |
  |---|---|---|---|
  | `GET /api/health` | 0.00 s | 0.01 / 0.03 s | 0 of 44 |
  | `GET /api/library/documents` | 0.02 s | 0.01 / 0.06 s | 0 of 44 |
  | an evidence-only chat turn (retrieval and rerank, no model) | 1.05 s | 3.49 / 15.5 s | 0 of 44 |

  The ingest took 308 s; with the GPU, 95 s, and the chat turn 0.14 s idle, 0.64 / 1.72 s during.
  So what stops a person is the 409s, and a chat that is several times slower while the CPU embeds;
  a turn that also calls a model adds its own time. The probe was a stdlib loop over these three
  requests, not kept in the repository; decision 6 records the numbers in `docs/performance.md`.
- **Adds while the engine starts.** The window's connection state is `connecting · ready · down`
  (`lib/shell/shell.svelte.ts:20`); while it is `connecting`, an add fails at its first request.
- **Letter spacing.** `DocumentPage.svelte:149` places the block third, under the header and the
  text check, always open. A rule is the ignore list keyed by the PDF's sha256
  (`library/spacing.py`, `config.REJOIN_IGNORE_PATH`) and enters the cache fingerprint, so it
  should survive a re-ingest: read in the code, never tested through one.

## Decided (the owner, 2026-11-23: every recommendation accepted)

1. **Where the queue lives.** *Recommended:* in the engine, in memory: one runner, jobs in order.
   A job left waiting when the engine stops is not lost: its files are already in the library
   folder, which Manage files lists as *not indexed*, with *Index new files*. *Instead:* a queue
   stored in `library.db`, which survives a restart at the cost of a table and a recovery path.
2. **What the list holds.** *Revised by the owner, 2026-11-23:* one list of **items**, not a line of
   jobs: each item is a file to index, or one document's parts to re-run. An add or a selection puts
   its files at the end; *index the folder* puts the folder's new and changed files there (and its
   sweep of vanished files runs with it). A file or a (document, part) already waiting is not added
   twice. The two status objects become one, and the three 409s go. *(Accepted first: three kinds of
   job, each waiting whole behind the one running.)*
3. **The add's copy step.** *Recommended:* unchanged: the files are copied or registered when the
   person confirms the add sheet; only the indexing waits. *Undo all* stays possible while the job is
   waiting, and is refused once it runs (as today, for the same reason: the extractor reads them).
4. **Stop, and what waits.** *Revised by the owner, 2026-11-23:* *Pause* and *Stop* act on the whole
   list, as on today's run: after a Stop, what did not start shows *Not started*, with *Index the
   remaining N*, as on the approved stopped screen. Any waiting item can be removed. *(Accepted
   first: waiting jobs held after a Stop until Start.)*
5. **Adds while the engine starts (T-007).** *Recommended:* the window keeps the picked files as a
   pending add (*Waiting for the engine*, in the status bar) and sends them once it is `ready`; if
   it goes `down`, the status bar says the add was not made and keeps the files to try again. Held
   in memory, not across a restart of the app. *Instead:* the shell copies the files itself, which
   would put add logic in Rust.
6. **The chat during an ingest.** *Recommended:* say it, do not throttle yet: while a job runs, the
   chat line notes *Indexing: answers take longer*. Throttling (the ingest yields between batches
   while a turn runs, or uses fewer threads) is a measured trade-off for later, recorded in
   `docs/performance.md` with today's numbers.
7. **The indexing screen and the status bar.** *Revised by the owner, 2026-11-23:* a wireframe round
   first, as for every screen change (house skill *wireframe-first*): added items as *Waiting* rows
   at the end of the one list, *Remove* on a waiting row, *Add to queue* in the header, the count
   growing with the list (*Indexing 14 of 33*), and the status bar's states (S7).
8. **Two builds.** *Recommended:* A8″a, the queue (decisions 1 to 7, T-007, T-010, T-011); A8″b, the
   letter-spacing block (T-009) and S-17. One commit each; the queue's wireframe gates A8″a only.
9. **S-17, the four GETs.** *Recommended:* each GET reads what is stored, and the work it did moves
   to a POST the window calls: `POST /api/sources/scan` (the walk and its registry rows; Manage
   files and the plan call it on open), `POST /api/updates/auto` (the opted-in daily check, sent once
   at launch), `POST /api/setup/probe` (the Ollama probe). `GET /api/ingest/plan` reads the registry
   as the last scan left it.
10. **The letter-spacing block (T-009).** *Recommended:* at the bottom of the document page; closed
    by default once the document has a rule, open while it has spacing to judge and no rule; open or
    closed is not remembered per document. A test drives a rule through a per-part text re-run and a
    full re-ingest and checks the rule still applies.

## The contract (follows the recommendations)

**Engine (A8″a).** `src/doc_assistant/ingest/queue.py`: `IngestQueue` holds one list of `Item`s
(a file to index, or one document's parts to re-run; `added_at`; `state`: waiting · running · added
· unchanged · failed · removed · not started). One thread runs it: it takes the waiting file items
in a batch through `ingest.main(files=…)` and the re-run items through `reingest.rerun`, in list
order, and items added meanwhile wait their turn in the same list. The steps a run does once happen
once, when the list is empty: the text check of the new documents, forgetting replaced hashes, and
the engine's reload (as S18's fix does after a re-run today). Pause and Stop act between documents,
on the whole list. `apps/api` keeps thin routes: `POST /api/ingest` and the re-ingest route add
items and answer 202; `GET /api/ingest/status` and `/api/ingest/files` report the one list;
`DELETE /api/ingest/items/{id}` removes a waiting item. `POST /api/documents/add` no longer refuses
during a run.

**Window (A8″a).** The status bar chip and the indexing screen render the one list, per the
approved wireframe; the add sheet adds to it; a pending add is held while the engine is
`connecting`.

**A8″b.** The block's place and default; the re-ingest test; the four routes per decision 9 and
their window clients; `test_get_routes_have_no_side_effects.py`'s `KNOWN` list emptied.

## Hard constraints

- All queue logic in `src/doc_assistant/`; `apps/api` and the window only render and call it
  (AGENTS.md).
- Nothing is refused because something else is running, except an *Undo all* of the running job's
  files.
- Every write route stays a POST, PUT, PATCH or DELETE under the launch token (S-6); no new GET may
  write or call out (the floor test).
- Honest at 0 documents and at 10,000: the waiting list is paged like the files list.

## Done when

- The plan's line: all three driven in the installed build (the chat answers during a slow
  ingestion), and the re-ingest check has a test.
- An add, a folder index and a re-run made while a job runs each wait and then run, in order (a
  test per kind, and driven once in `tauri dev`).
- An add made while the engine starts runs once it is ready, and says so if it does not come up.
- `KNOWN` in the GET floor test is empty.

## Out of scope

Throttling the ingest while a chat turn runs (decision 6); a queue that survives a restart (decision
1's alternative); parallel jobs.

## Open questions

None beyond *To decide*.

## As built — A8″b (2026-11-23)

- **T-009.** `LetterSpacing` sits at the foot of the document page, after References, with a fold in
  its heading (`aria-expanded`, hover text). `lib/library/spacing.ts` `spacingOpenByDefault`: open
  while the document has runs and no rule of the person's (nothing kept, not every run kept), or a
  change waits for a re-run; closed otherwise. A folded block with a waiting change says how many
  in its heading. The person's toggle holds until another document is shown.
- **The rule through a re-ingest.** `tests/integration/test_spacing_rule_survives_reingest.py`: a
  real PDF with a letter-spaced title is closed without a rule; *Keep every run spaced* then holds
  through a per-part text re-run, an index of the folder, a rebuild, and an extraction from scratch.
- **S-17, as decided.** `GET /api/sources` lists the registry read-only (`registry.stored_sources`;
  a library never scanned lists nothing), and `POST /api/sources/scan` walks every root.
  `GET /api/ingest/plan` counts the stored rows (`plan_library(scan=False)`); the window's
  `getIngestPlan` scans first. `GET /api/updates` answers from the last check; `POST
  /api/updates/auto` runs the opted-in daily check, and the window's `getUpdateStatus` calls it.
  `GET /api/setup` never probes; `POST /api/setup/probe` does, and `getSetup(probe)` picks one.
  `KNOWN` in `test_get_routes_have_no_side_effects.py` is empty.
- **Not foreseen.** `GET /api/setup?probe=true` no longer probes: the parameter is gone.

## As built — A8″a (2026-11-23)

- **Engine.** `ingest/queue.py` `IngestQueue` (made on first use by `apps/api/services.ingest_queue`,
  so importing the API still loads no torch). Rows live in `ingest/run.py` `IngestRun`: each has a
  stable `id` and a `kind` (`file`, `folder`, `rerun`); `append_files`, `append_folder`,
  `append_rerun`, `remove`, `next_batch`, `close_if_drained`. `ingest.main` takes a `RunSink`, and
  `BatchView` maps one batch's indexes to the rows by path, turns a folder row into its files, and
  ends a batch early when the next file's row was removed (a waiting file in the running batch can
  still be taken out). New states `done` (a finished re-run) and `removed` (never counted).
- **The run's end, once.** Forgetting the replaced hashes and the engine's reload happen once, when
  the list is empty. The text check and the other per-document steps stay inside each
  `ingest.main` batch, scoped to that batch's new documents, as before.
- **Routes.** `POST /api/ingest`, the folder index and `POST /api/library/documents/reingest` add
  rows and answer 202; none answers 409 because something else runs. `DELETE
  /api/ingest/items/{id}` (409 once started). `/api/documents/add` is allowed during a run; *Undo
  all* takes back files still waiting and answers 409 once one is being read. The re-run status
  follows the latest request's rows and adds `waiting_behind`; one row per document (`total`
  counts documents, no longer document × part).
- **Window.** The indexing screen: *Add to queue* in the header while the list runs (the usual add
  flow), *Remove* on every waiting row (decision 4 says any waiting item; the drawing showed it on
  the added rows only), a `Done` pill. The status bar's *Waiting for the engine · N papers to add*
  and *… not added: the engine did not start* with *Try again* and *Dismiss*
  (`PendingAddChip.svelte`, `accept.pendingAddText`); the review sheet opens once the engine is
  ready. The re-run dialog says how many items run first and is not locked while it waits. The chat
  line says *Indexing: answers take longer* while the list runs.
- **Found by driving it.** Removing a waiting file that was already in the running batch did not
  stop it being read; fixed in `BatchView.checkpoint`, with a test.
- **Not driven.** *Add to queue* and the engine-start chip need the desktop shell (the file picker,
  and an engine that is still starting); they are covered by tests here and wait for the installed
  build at A9.
