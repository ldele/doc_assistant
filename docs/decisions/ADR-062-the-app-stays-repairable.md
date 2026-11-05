<!-- status: active · updated: 2026-11-05 (proposed; revised the same day: choice 8, the start-up, and the settings docs; step 7's third row and step 5 corrected for KI-88) · class: append-only -->

# ADR-062 — The app stays repairable: Settings opens in every state, and a repair never destroys a library

- **Status:** proposed (Claude Code, 2026-11-05). Not accepted, not built. This is ROADMAP row 100,
  *the app stays repairable when it cannot work*, as the user widened it on 2026-11-05: Settings must
  open whatever state the app is in, with fewer features rather than none; *Check for updates* at
  least; and perhaps a *Repair* that puts back the files the app needs, with defaults, the way some
  apps regenerate missing default files and data. Plan: `docs/plans/PLAN_2026-10-31_next-two-releases.md`
  (local), *Now*, item 4. **Revised the same day** with the user's second comment: at launch the app
  waits for the engine before Settings can be used, and a start of about 30 s can make a first-time
  user think the app is broken. That is choice 8.
- **Date:** 2026-11-05
- **Deciders:** user + Claude Code
- **Extends** [ADR-034](ADR-034-in-app-provider-setup.md) (first run, the readiness gate, *inform,
  don't block*), [ADR-044](ADR-044-update-notification-not-delivery.md) (the update check: notify
  only, never on its own unless the person opted in) and
  [ADR-061](ADR-061-an-indexed-library-opens-where-it-is.md) (the shell holds the library choice and
  restarts the engine; *Go back*). **Revisits** start-up rule E0.5a in `apps/api/main.py`, which
  makes the engine refuse to start when a migration fails. Choice 1 keeps the reason for that rule
  (never serve a half-migrated schema) but replaces the refusal (the engine still starts, with the
  library closed). Takes KI-86 and two of the leftovers in KI-79: a taken port 8001, and the
  "still starting" hint.
- **In one sentence:** the engine (the backend program the desktop shell starts) answers within
  about a second of launch and loads the library behind it, so Settings works from the start; when
  the library cannot be opened, the engine stays up with the library closed and says why; when no
  engine can run, the window shows its own screen with the engine's log, the program holding port
  8001, the library folder in use, *Retry* and a *Check for updates* that needs no engine; and
  *Repair* only renames a damaged part aside and rebuilds from the library's own documents, never
  deleting or overwriting anything.

## To decide — eight choices, each with a recommendation

1. **Where the app degrades: the engine, the shell, or both.** *Recommended:* both, as two layers.
   - **The engine starts in repair mode instead of not at all.** When the schema migration
     (`init_db`) or the search index fails at start-up, the engine stays up with the library
     *closed*. `/api/health` answers `status: "repair"` with what failed and why. Every route that
     reads or writes the library answers 503 with that same sentence. The routes that need no
     library keep working: health, settings, provider setup, updates and repair. E0.5a's reason
     still holds, because no route serves a half-migrated schema.
   - **A start with `DOC_SAFE_START=1` takes that same path without opening the library at all.**
     The shell uses it once, on its own, when a normal start died before answering (a crash below
     Python, for example). The window then says it did so, and why.
   - **The shell owns a screen for when no engine answers.** It covers what repair mode cannot:
     - the engine exits even when started with `DOC_SAFE_START`;
     - port 8001 is taken;
     - the bundled engine is missing or will not start (today a failed spawn ends the shell,
       `lib.rs:246`, and a missing one is taken for a development build, `lib.rs:248`);
     - the remembered library folder is gone (choice 7).

     The screen shows:
     - what happened, in one sentence;
     - the last lines of the engine's log;
     - the program holding port 8001;
     - the library folder in use;
     - *Retry*, *Check for updates* (choice 2), *Choose another library…*, *Use the app's own
       library*, *Open the log folder* and *Copy details*.
   - **The shell writes the engine's log to a file,** so that screen has something to show. The
     engine's stderr goes to `engine.log` in the app's log folder, and the previous launch's log is
     kept beside it. Today the shell passes that stderr to its own console (`lib.rs:142-148`), and a
     release build has no console (`main.rs:2`).
   - **The window hears of an exit when it happens.** The shell forwards the engine's exit as an
     event. An engine that dies after start-up is then reported at once. Today it never is: this is
     KI-79's first leftover, where the status bar stays green and every panel says *Failed to fetch*.
2. **Check for updates without the engine.** *Recommended:* the shell makes ADR-044's check itself,
   under the same rules:
   - one HTTPS GET to the fixed `releases/latest` URL;
   - on the person's click only, with a 5 s timeout;
   - three states (*current*, *update available*, *unknown* with a reason);
   - the running version is the shell's own;
   - the only action is a link to the release page.

   TLS goes through the operating system. On Windows this means the `native-tls` backend for the
   `reqwest` 0.13.4 already in the shell's tree, which today comes through `tauri` with no TLS backend
   (`Cargo.lock`). A proxy's root certificate then works, as KI-10 made it work for the engine.

   The window uses this check whenever the engine does not answer. The engine's route
   (`POST /api/updates/check`) stays for the normal Settings and for the automatic daily check.
   One shared table of version pairs, read by both the Python tests and the Rust tests, keeps the
   two comparisons from drifting. An *Open the releases page* link and the running version are shown
   beside the check at all times, so even an *unknown* leaves the person a way to find out.

   The GET lives in a Rust command with its URL fixed in the code. The window gains no network reach
   and no `http` capability, which `docs/security.md` lists as not granted.

   *Rejected:* a fetch made by the window itself. It would need `https://api.github.com` in the
   window's `connect-src` (`tauri.conf.json:25`, which today allows only the engine, and
   `tests/unit/test_desktop_security_config.py` holds it there). That widens the window's outbound
   reach in every state, including while it renders passages and model output as HTML
   (`docs/security.md` S2, sanitised since step S-3).
3. **What Repair may do.** *Recommended:* four actions. Each shows what moves where before anything
   moves, each is the person's click, and none runs automatically.
   - **Rebuild the search index.** For a damaged search index; `library.db` is kept.
     - It moves the vector stores and the keyword index aside, then reads every document listed in
       `library.db` again, from all its roots.
     - Everything `library.db` holds is kept: folders, chats, concepts, definitions and marks.
     - A Re-index walks one folder (`ingest/__init__.py:527-540`), so this action hands the ingest
       its list of documents instead.
     - The ingest's duplicate check takes the intersection of the two vector stores
       (`ingest/__init__.py:670`), so empty stores mean every document is read again. That takes
       about as long as the first indexing.
   - **Rebuild the library from its documents.** For a damaged `library.db`.
     - It moves `library.db` and the index aside, keeps the documents (`sources\`) and their
       extracted text (`cache\`), makes a new `library.db` and reads the library folder again.
     - The rebuilt library loses what `library.db` held: folders, chats, curated concepts and
       definitions, marks, and the documents added by reference (their folders were registered in
       it). All of it stays in the moved file.
   - **Start a new library.** Moves the whole folder aside, next to where it was, and starts an
     empty library at the same path.
   - **Reset settings.** Renames `settings.json` to `settings.json.bak-<date>`, so the app reads its
     defaults. The keys file (`credentials.json`) is not touched: Provider & model has its own
     action for removing a key.

   **Rules for every repair:**
   - A move is a rename on the same drive: instant, with no extra disk space.
   - Nothing is deleted or overwritten. A name already in use gets a suffix.
   - Moved parts go to `repair\<date-time>\` inside the library. A whole moved folder goes next to
     the library, as `<folder>.repair-<date-time>`.
   - The screen names where things went and offers *Open folder*. Only the person deletes them.
   - A rename that fails (a file held open) changes nothing, and the screen says which file.

   **The engine carries out a repair at its next start, before it opens anything.** The window asks,
   and the shell restarts the engine with `DOC_REPAIR=<action>`, as it already restarts the engine
   for a library switch (`open_library`, `lib.rs:192-223`). The repair logic lives in
   `src/doc_assistant/` (non-negotiable 3). It cannot run inside an engine that is already up:
   that engine holds open the very files a repair renames, and Windows refuses to rename them (KI-77
   is that refusal).

   *Out of scope:* salvaging rows from a damaged `library.db` (SQLite's recovery tools). The moved
   file is kept for that.
4. **Which faults offer which repair.** *Recommended:* name the fault first, and offer a repair only
   for damage. The engine sorts a start-up failure into one of five faults:

   | Fault | Example | Offers a repair? |
   |---|---|---|
   | *missing* | no folder, or no `library.db` (choice 7) | no: *Look again*, choose a folder |
   | *in use* | `database is locked`: another app on the same library (ADR-061 choice 6) | no: *Retry*, with the reason |
   | *unreadable* | permission denied, a drive error | no: *Retry*, with the reason |
   | *damaged* | `file is not a database`, a failed integrity check, a failed migration | yes |
   | *search index damaged* | `library.db` opens, the vector store does not | yes |

   *In use* and *unreadable* offer no repair because that is how a repair would do harm: by moving
   aside a library that another program is writing, or one on a drive that is slow to wake. A
   failure the engine cannot sort is shown as it is, with *Retry* and no repair. A wrong sort
   therefore fails safe.
5. **What stays reachable in each state.** *Recommended:* this table. Settings always opens: its
   list of sections always shows, and a section that needs the engine shows that sentence in place
   of its contents. Today a single panel-wide error replaces the whole of Settings
   (`Settings.svelte:477-478`).

   | State | How the window knows | Works | Closed, each with its sentence |
   |---|---|---|---|
   Settings' sections are Getting started, Documents, Provider & model, Retrieval and General
   (`sections.ts`); the theme, chat history and Updates are in General. *Repair* would be a new
   block in Documents, where the library folder is chosen.

   | State | How the window knows | Works | Closed, each with its sentence |
   |---|---|---|---|
   | Ready | health `ok`, an answer engine connected | everything | nothing |
   | Starting (choice 8) | health `starting`, with its stage | Settings: Getting started, Provider & model, General. The last session's lists stay on screen, marked as last time's (`lastSession.ts`, KI-73) | chat, Library, graph, adding, indexing: *your library is still opening* |
   | No answer engine ([ADR-034](ADR-034-in-app-provider-setup.md)) | health `ok`, setup steps left | everything but answering; Getting started leads to Provider & model | answering |
   | Library closed (choice 1) | health `repair`, with the fault | as *Starting*, plus Documents › Repair and the data home | as *Starting*, with the fault's sentence |
   | No engine | the engine exited, the port is taken, there is no engine in the install, or 90 s with no answer | the shell's screen; Settings: General (theme, Updates by the shell's check), the library choice, the log | everything the engine serves |
   | Library folder missing (choice 7) | the shell's check at launch | as *No engine*, plus *Look again* and *Choose the folder…* | as *No engine* |
6. **Who answers on port 8001** (KI-79). *Recommended:*
   - The shell names the program holding the port: its name and process id. It reads them with
     System32 tools, called by full path and with no console window, the way `end_sidecar` already
     runs `taskkill` (`lib.rs:25-47`).
   - If the holder is an older Provenote engine (the case in KI-71 and KI-80), the screen offers
     *End it and retry*.
   - Any other program is named and never ended; *Retry* follows once the person has closed it.
   - The window no longer trusts whatever answers on 8001. Health says whether the request carried
     this launch's token. Today a `GET` needs no token (S-6), so a development engine or an older
     one answering health reads as *ready*.
   - The status bar's *Restart the app if it never arrives* is removed.
7. **KI-86: a remembered folder with no `library.db`.** *Recommended:* at launch, this is a state,
   not a new library.
   - Before starting the engine on a remembered folder, the shell makes the check `open_library`
     already makes (`lib.rs:199-203`).
   - If the check fails, the shell starts nothing on that folder and shows its screen: *The library
     at <folder> was not found*, with *Look again* (for a drive not yet connected), *Choose the
     folder…* and *Use the app's own library*. Nothing is created.
   - A second guard: with a remembered folder, the shell also passes `DOC_LIBRARY_MUST_EXIST=1`, and
     `init_db` then reports the fault *missing* instead of making the folder
     (`db/migrations.py:386-387`).
   - Two cases still create a library: the app's own library on a first run, which is a fresh
     install; and a folder a developer names by hand with `DOC_DATA_DIR`.
8. **Start-up: Settings works from the first seconds, and the wait says what it is.**
   *Recommended:* the engine answers first and loads the library after.
   - The engine opens its port as soon as its light part is imported: the web server, settings,
     credentials, provider setup and updates. It answers `/api/health` with `status: "starting"`
     and the stage it has reached. The search stack (embedding model, vector store, keyword index)
     loads behind it, in a thread. In the development environment the light part imports in
     0.6–0.7 s and the whole app in 9.4–17.8 s. In the installed build, imports are about 8 of the
     10 s to health (*In practice*, step 7).
   - While it starts, the engine serves what repair mode serves (choice 1). Library routes answer
     503 *your library is still opening* until it is open. One gate, *library not open*, covers
     both states, so a library that fails to open turns *starting* into *repair* rather than into a
     dead engine.
   - Settings opens at any moment and loads each section once the engine can serve it, without being
     closed and reopened. Today it loads once, when opened (`Settings.svelte:262`). Before the
     engine answers, that one load fails and puts *Couldn't load settings: …* in place of the whole
     panel (`Settings.svelte:477-478`), and nothing retries.
   - The status bar names the stage and the time so far (*opening your library*, *loading the
     search model*) instead of *starting the engine…* for the whole wait. When the window has no
     last session to show, which is a first run, it says from the start that a first launch can
     take a minute, not only after 20 s.
   - A first-time user can open Getting started and Provider & model within seconds, which is what
     they need first ([ADR-034](ADR-034-in-app-provider-setup.md)).
   - **Ollama is not the cause of the wait.** Building the chat model makes no call
     (`pipeline.py:95-104`). The Ollama probe runs only for `/api/setup`, after health
     (`readiness.py:245`). And the engine reached health in 9.7 s with Ollama unreachable, the same
     as with it running (step 7).

   *Alternatives:*
   - **The window alone:** Settings shows its shell-backed parts and fills in the rest when the
     engine answers, with no engine change. This fixes the impression, but Provider & model still
     waits the whole start.
   - **Faster imports only:** this shortens the wait, but a cold first launch still waits and the
     window still says nothing while it does.

## In practice — a library database that cannot be read, 2026-11-05

This was made on a throwaway folder (`C:\Projects\pv-repair-probe`, deleted afterwards), never on a
real library. The folder's `library.db` was replaced by one line of text, and the engine was started
from the development checkout with `DOC_DATA_DIR` set to that folder and on a spare port.

1. **What the engine does today (observed).** The start-up migration fails, and the engine refuses
   to serve. Its log reads:
   - `schema_migration_failed … (sqlite3.DatabaseError) file is not a database`;
   - `RuntimeError: database migration failed at startup; refusing to serve a stale schema …
     Fix the DB or run python -m doc_assistant.db.migrations`;
   - `Application startup failed. Exiting.`

   The process ends 41 s after launch with **exit code 0**. Most of those seconds are imports in a
   development checkout; the installed engine reaches health in 9–10 s (KI-73). Because the exit
   code is 0, the shell cannot use it to tell a failure from a clean stop. The error message names a
   command that an installed-app user does not have.
2. **What the window does today (read in the code, not driven in the installed build).** Nobody
   sees that log: the shell copies the engine's stderr to its own console (`lib.rs:142-148`), and a
   release build has none (`main.rs:2`). The status bar reads, in turn:
   - *starting the engine…*;
   - at 20 s, *starting the engine — a first launch can take a minute…*;
   - at 90 s, *still starting — retrying. Restart the app if it never arrives.*

   (`StatusBar.svelte:29-38`, `startup.ts:17-21`.) Settings opens to one line, *Couldn't load
   settings: TypeError: Failed to fetch* (`Settings.svelte:477-478`). There is no Updates section,
   no library choice and nothing to repair with. Restarting the app repeats all of it. The only way
   out is a file manager, and knowing which file to move.
3. **With choice 1.** The same migration fails. The engine logs it and stays up with the library
   closed. `/api/health` answers within the same few seconds, in the proposed shape
   `{"status": "repair", "fault": "damaged", "part": "database", "reason": "file is not a database"}`
   plus the data home. The window stops waiting at once and shows a card: *This library's database
   cannot be read (file is not a database). Your documents are still in its folder. Settings ›
   Repair shows what can be done.* The last session's library and chats stay on screen, marked as
   last time's. In Settings:
   - General works (the theme, and *Check now* answers), and so do Getting started and Provider &
     model;
   - Documents opens on its *Repair* block, and its library facts and indexing say that they need
     the library.
4. **Repair (choices 3 and 4).** The fault is *damaged*, in `library.db`. Two actions are offered:
   *Rebuild the library from its documents* and *Start a new library*. *Rebuild the search index*
   is not offered, because it keeps `library.db`, the broken part. Before anything moves, the
   screen says: *`library.db` and the search index move to `<library>\repair\2026-11-05-1013\`.
   Your documents (`sources\`) and their extracted text (`cache\`) stay where they are. The rebuilt
   library will not have your folders, chats, concepts or marks; they are in the moved file.* When
   the person clicks *Rebuild*:
   - the shell restarts the engine with `DOC_REPAIR=rebuild-library`;
   - before opening anything, the engine renames the two parts (same drive, so nothing is copied)
     and makes a new `library.db`;
   - the engine starts normally, and the window opens the Re-index confirmation.

   Nothing has been deleted.
5. **The same on an account whose name has an accent** (this computer's account is one). The app's
   own library then sits at a path with an accent, so its vector stores live under
   `%PROGRAMDATA%\doc_assistant\chroma\<hash of the library's path>` (`config.py:98-106`, KI-11).
   Suppose *Start a new library* renamed only the library folder. The new, empty library would sit
   at the same path, so it would get the same hash and read the old library's passages. So the
   repair renames that folder under `%PROGRAMDATA%` too. This is the one case where moving the
   library folder aside is not enough. The sharing is already real on this computer (KI-88): an
   agent's shell, inside an MSIX package, opens that package's stale copy of `library.db`
   (0 documents) at the app's path, and the copy reads the real library's 41,989 chunks.
6. **When even repair mode cannot run (observed with a held port).** Another program held the port,
   and the engine was started on an empty folder. It migrated, loaded its models, then failed to
   bind: `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8012)`, exit code
   1, after 16 s. The server binds its port only after start-up has run, so a held port shows up
   only once the models have loaded.
   - **Today** the window waits forever, and KI-79's hint says to restart, which changes nothing.
   - **With the shell's screen and choice 6:** *The engine could not start: another program is
     using port 8001: <name> (process <id>).* An older Provenote engine gets *End it and retry*;
     any other program gets *Close it, then Retry*. *Check for updates* works from this screen
     (choice 2).

**A launch, measured (2026-11-05).** The installed 0.7.2 engine
(`%LOCALAPPDATA%\Provenote\backend`) was started on a throwaway folder and a spare port, and timed
from process start to its first health answer. Each line of its log was stamped with the seconds
elapsed.

7. **Today.**

   | Library | Time to health | Before start-up begins (imports) |
   |---|---|---|
   | empty, Ollama `llama3.1:8b` selected | 10.2 s, then 9.7 s | 8.4 s, 7.7 s |
   | the same, Ollama unreachable | 9.7 s | 8.0 s |
   | this computer's own search index (760 MB, 41,989 chunks), copied from an agent's shell, which sees an MSIX package's stale `library.db` (0 documents) at the app's path (KI-88) | 9.9 s, twice | 7.6 s, twice |

   The search model, vector store and keyword index then take about 1.5 s, and the rest is under a
   second. The window's gate polls every second for the first 10 s (`startup.ts:30-34`), so it
   sees such an engine within a second. These are warm starts. KI-73 measured 28.9 s on a first
   launch after a build, when the engine folder's 17,855 files are read cold. A start of about 30 s
   is therefore most likely a cold start (not measured today), not Ollama and not the library's
   size. Throughout those seconds, Settings shows one error line (choice 8).
8. **With choice 8.** The engine's light part answers within about a second, as *starting ·
   loading the search model*. Settings opens on Getting started: the Model row is probed as today,
   and the Documents row reads *your library is opening*. Provider & model works. When the search
   stack is up, the library routes open and the window refreshes its lists as it does today. A
   first-time user is never left with a window that has nothing to click.

## Context

- **What already comes back with defaults** (read 2026-11-05):
  - an unreadable `settings.json` or `credentials.json` reads as empty, so the defaults apply
    (`app_settings.py:30-38`, `credentials.py:71-79`);
  - a missing library folder and `library.db` are created (`db/migrations.py:386-396`), which is
    exactly the fresh-install path;
  - an unreadable shell `library.json` means the app's own library (`lib.rs:102-109`).

  So regenerating default files, as the user described, is already what the app does for every file
  it needs, with three exceptions: a `library.db` that exists but cannot be read, a search index
  that cannot be opened, and the engine itself.
- **The engine is the single point of failure.** Its start-up turns a failed `init_db` into an exit
  (`apps/api/main.py:138-146`). The search pipeline opens the vector store and the keyword index
  while it is built (`pipeline.py:177-200`), during the same start-up. The server runs that start-up
  before it binds the port (step 6). The dev log of the failed start says *exit code 0* (step 1).
- **The window.** The readiness gate polls `/api/health` with no deadline (`App.svelte:560-612`),
  and its phases change at 20 s and 90 s (`startup.ts:17-21`). Settings depends on the engine for
  every section (`Settings.svelte:477-478`). The update check is an engine route
  (`apps/api/routers/updates.py`, `update_check.py`). The automatic check is off by default,
  because the app makes no outbound call the person did not ask for (`app_settings.py:275-281`).
  The only recovery screen is the card shown after a failed library switch
  (`LibrarySwitchFailed.svelte`, ADR-061 choice 7).
- **The shell** (`apps/desktop/src-tauri/src/lib.rs`):
  - it starts the engine and panics if the spawn fails (`:246`);
  - it reads a missing bundled engine as a development build (`:78-86`, `:248`);
  - it copies the engine's stderr to a console that a release build does not have (`:142-148`,
    `main.rs:2`);
  - it checks for `library.db` only when the person opens a library (`:199-203`), never at launch
    (KI-86);
  - the window's content-security policy allows connections only to itself and the engine
    (`tauri.conf.json:25`).
- **What the settings docs already say** (read 2026-11-05):
  - [the Getting started wireframe](../wireframes/getting-started-and-providers.md): Getting
    started is two rows, Model and Documents, each with a status and a link; *Engine online* there
    means the answer engine;
  - [the release walkthrough](../release-ux-checklist.md): §6's *Answer engine unreachable* row
    (Settings must open when Ollama stops) is still to be driven and is assigned to ROADMAP 100;
    its port row expects *the app reports the conflict*; and its console row says that a console
    engine would need its stderr sent "to the app log";
  - `docs/security.md` S12 and step S-11 plan security events "in the app log". In the installed
    build that log is the engine's stderr, which goes nowhere today (above), so S-11's events would
    have no reader without the log file in choice 1;
  - [the UI gate](../ui-checklist.md): three states per surface, never two (from ADR-044).
- **Where a library's parts live.** `library.db` holds the documents' records, folders, keywords,
  concepts and definitions, chats (`answer_records`, `conversation_meta`) and the registered roots
  (`db/models.py`). The vector stores sit inside the library folder, except on a path with an
  accent, where they sit under `%PROGRAMDATA%`, keyed by a hash of the path (`config.py:98-106`).
  The keyword index sits next to the vector stores.

## Options

- **A — Both layers (recommended).** The engine's repair mode covers a damaged library and keeps
  Settings, provider setup and the update check reachable; with choice 8 the same gate covers the
  seconds while the library opens at every launch. The shell's screen covers what no engine
  can report: no process, no port, no folder. Repair stays in Python.
- **B — The shell's screen only.** The engine still refuses to start. Simpler in the engine, but
  Repair would then have to run in Rust, which puts library logic in a shell (non-negotiable 3), or
  start the engine anyway. And every Settings section that needs no library, Provider & model among
  them, stays out of reach while a library is damaged.
- **C — The engine's repair mode only.** It covers a damaged library and, with the flag in choice 7,
  a missing one. A held port, a missing engine and an engine that cannot import would remain the
  90 s stall with no reason given, and *Check for updates* would still need the engine.
- **D — Repair on its own at start-up, without asking,** which is the user's comparison. That is
  already the behaviour for files that hold no work, such as `settings.json`, the library folder and
  `library.json`. For a library it would mean deciding for the person to set their folders, chats
  and concepts aside. *Inform, don't block* and *never overwrite a library* both rule it out, so it
  is rejected for the library.

## Decision

Proposed: **A**, with choices 1 to 8 as recommended. Build order:

1. Choice 7 and the engine's log file first. Both are small, and KI-86 can make a library look lost
   today.
2. Choices 8 and 1 together, with choice 4's fault names. They share one gate, *library not open*:
   the library is still opening (*starting*) or could not be opened (*repair*). Choice 8 is what
   every first-time user sees, so it leads.
3. The shell's screen, choice 6 and choice 2.
4. Repair (choice 3) last, since it is the only part that moves files.

## Consequences

- **E0.5a changes.** The engine no longer exits on a failed migration.
  `test_boot_fails_when_migration_fails` (`tests/integration/test_api_startup_migration.py`) is the
  guard that holds the old rule, so it is rewritten on purpose: the boot succeeds in repair mode,
  and every library route refuses. That test must not simply be deleted, because its point (never
  serve a stale schema) stands.
- **The shell gains logic** that cannot live in the engine, because it serves the time when the
  engine is down: a log file, a port check, a folder check and an HTTPS GET. Its scope is bounded:
  it reports and restarts, and it never reads or changes what a library holds. This is a named
  exception to "apps are thin shells", limited to the no-engine states.
- **The shell gains crates:** a TLS backend, audited with the rest.
- **There are two version comparisons,** in Python and in Rust, held together by one shared table.
- **The app writes a log file for the first time.** It holds file names and may hold question text.
  It stays on this computer, and *Copy details* copies only what the screen shows. It is also where
  security step S-11's events land in the installed build.
- **The engine's import graph is split** into a light part that answers and a heavy part loaded
  behind it (choice 8). Today the routers import the search stack at module level, so this is a
  restructuring of `apps/api/` start-up, not a flag. A route that needs the library must say so,
  and a test holds the light part light.
- **Moved-aside parts take disk space** until the person deletes them. The screen says how much.
- **One word needs pinning before it ships.** *Engine* already means three things on screen: the
  backend (*starting the engine…*, `StatusBar.svelte`), the answer engine (Getting started's
  *Engine online*, ADR-034's amendment), and Retrieval's *Engine (read-only)* block. The start-up
  and repair screens need a `GLOSSARY.md` entry first.
- **Done when:**
  - each state in choice 5's table is driven in the installed build, each on a copy: a `library.db`
    replaced by text, port 8001 held, the remembered folder renamed, and the engine folder renamed
    in a copy of the install;
  - in each of those states, Settings opens and *Check for updates* answers;
  - on a warm launch of the installed build, health answers *starting* within 2 s, and Getting
    started and Provider & model work before the library is open; a cold first launch is timed and
    recorded beside KI-73's 28.9 s;
  - the walkthrough's *Answer engine unreachable* row is driven;
  - a test proves that no repair deletes or overwrites a file, by content hash before and after.

## Enforced by

Planned; nothing is built yet.

- A test for each start-up fault: the engine answers `/api/health` with `status: "repair"` and the
  fault named, and the library routes refuse.
- A test with a library that is still opening (a controller factory that blocks): health answers
  `starting`, settings, provider setup and updates work, and library routes answer 503.
- A test that importing the light part of the engine does not import the search stack (no
  `torch`, no `chromadb` in `sys.modules`).
- A test that every repair action leaves every file it found, by content hash, inside the library,
  its `repair\` folder or the moved folder.
- The shared table of version pairs, read by `tests/unit/test_update_check.py` and the shell's
  `cargo test`.
- `docs/release-ux-checklist.md` §6: one row per state, replacing the port row's "reports the
  conflict".

## Confidence

- **Observed** on this computer (2026-11-05, development checkout, throwaway folder):
  - a `library.db` holding text stops the engine with `file is not a database`, and the process
    exits with code 0;
  - a held port surfaces only after the models load, and the process exits with code 1;
  - the installed engine reaches health in 9.7–10.2 s on a warm launch, with or without Ollama
    reachable, on an empty library and with this computer's 760 MB search index (beside a stale
    `library.db`, KI-88); about 8 s of it is imports;
  - the light part imports in 0.6–0.7 s in the development environment.
- **Read in the code, not driven:**
  - what the window does in the installed build for these faults;
  - that a Re-index after the stores are moved aside reads every document again (the intersection
    check). The Re-index estimate (`/api/ingest/plan`) counts from the registry, not from the
    stores, so it may call every document *unchanged*. Whether the registry's *unchanged* skips a
    file before the intersection check is not read. Check both on a copy before building choice 3.
- **Not measured:** a cold first launch today; the light part's import time in the installed build;
  how long the window itself takes to appear.
- **Medium:**
  - how much of the import graph can move behind the port (choice 8). The routers import the search
    stack at module level today, and that has not been traced module by module;
  - sorting faults (choice 4). SQLite's messages are stable; the vector store's are not a contract.
    A fault the engine cannot sort offers no repair, so a wrong sort fails safe.
  - naming the port's holder with System32 tools, without a new crate.
- **Not measured:** the 5 s timeout is ADR-044's, and the log's size cap is structural.
