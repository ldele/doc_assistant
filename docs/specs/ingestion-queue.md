<!-- status: active · updated: 2026-11-23 (accepted by the owner as written: all ten recommendations) · class: living -->

# Spec — One queue for every ingestion, adds while the engine starts, and the letter-spacing block (session A8″)

> The executor brief. Target executor: **Claude Code**. Session A8″ of the plan
> (`docs/plans/PLAN_2026-10-31_next-two-releases.md`), tickets T-007, T-009, T-010 and T-011, with
> S-17 as its security step (`docs/security.md` §4). **Accepted by the owner, 2026-11-23, as
> written:** every recommendation under *To decide* holds, so the contract below is the build's.

## In one sentence

Every ingestion (an add, a whole-folder index, a re-run of a document's parts) joins one queue in
the engine and runs in turn, so nothing is refused while another runs; an add made while the engine
is starting waits in the window and is sent when the engine is ready.

## A worked example

Forty papers are indexing; the status bar reads *Indexing 12 of 40*. The person adds three more.

- **Today:** the add sheet copies the three files, then `POST /api/ingest` answers 409 *ingest
  already running*. The files sit in the library folder, not indexed, until the person finds Manage
  files and indexes them by hand after the forty finish. A re-run of one document's text, asked for
  meanwhile, is refused the same way.
- **After:** the three join the queue behind the forty. The status bar reads *Indexing 12 of 40 ·
  1 waiting*; the indexing screen lists the waiting job (*3 papers, added 14:02*) with *Remove*. The
  re-run joins as a second waiting job. When the forty finish, the three start, then the re-run.

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
2. **What a job is.** *Recommended:* three kinds, run by the one runner: *index these files* (an
   add, or a selection), *index the folder*, and *re-run these parts of these documents*. The two
   status objects become one, and the three 409s go. A file or a (document, part) already waiting is
   not queued twice.
3. **The add's copy step.** *Recommended:* unchanged: the files are copied or registered when the
   person confirms the add sheet; only the indexing waits. *Undo all* stays possible while the job is
   waiting, and is refused once it runs (as today, for the same reason: the extractor reads them).
4. **Stop, and what waits.** *Recommended:* *Pause* and *Stop* act on the running job as today. After
   a Stop, the waiting jobs do not start by themselves: the screen says *2 waiting* with *Start*, so
   a Stop never sets off more work. Any waiting job can be removed.
5. **Adds while the engine starts (T-007).** *Recommended:* the window keeps the picked files as a
   pending add (*Waiting for the engine*, in the status bar) and sends them once it is `ready`; if
   it goes `down`, the status bar says the add was not made and keeps the files to try again. Held
   in memory, not across a restart of the app. *Instead:* the shell copies the files itself, which
   would put add logic in Rust.
6. **The chat during an ingest.** *Recommended:* say it, do not throttle yet: while a job runs, the
   chat line notes *Indexing: answers take longer*. Throttling (the ingest yields between batches
   while a turn runs, or uses fewer threads) is a measured trade-off for later, recorded in
   `docs/performance.md` with today's numbers.
7. **The indexing screen and the status bar.** *Recommended:* a wireframe round first, as for every
   screen change (house skill *wireframe-first*): the waiting list, *Remove*, *Start*, *Add to queue*
   on the screen, and the status bar's *· N waiting*.
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

**Engine (A8″a).** `src/doc_assistant/ingest/queue.py`: `IngestQueue` holds `Job`s (`kind`,
`files` or `(document_ids, parts)`, `added_at`, `state`: waiting · running · done · failed ·
removed), runs one at a time on one thread through `ingest.main` and `reingest.rerun`, and after a
job that changed text forgets the replaced hashes and reloads the engine (as S18's fix does today).
`apps/api` keeps thin routes: `POST /api/ingest` and the re-ingest route enqueue and answer 202 with
the job; `GET /api/ingest/status` reports the running job and the waiting ones; `DELETE
/api/ingest/jobs/{id}` removes a waiting job; `POST /api/ingest/start` starts the queue after a Stop.
`POST /api/documents/add` no longer refuses during a run.

**Window (A8″a).** The status bar chip and the indexing screen render the queue, per the approved
wireframe; the add sheet enqueues; a pending add is held while the engine is `connecting`.

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
