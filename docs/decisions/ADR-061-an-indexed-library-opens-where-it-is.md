<!-- status: active · updated: 2026-11-04 (proposed) · class: append-only -->

# ADR-061 — An indexed library opens where it is, and nothing is written outside it

- **Status:** proposed (Claude Code, 2026-11-04), for the user to decide. Not built. The request is
  the user's, 2026-11-01: *"Add already indexed library to app."* On 2026-11-04 the user put it in
  0.7.2 and left its meaning to this ADR. It is 0.7.2's piece 5
  (`docs/plans/PLAN_2026-10-31_next-two-releases.md`, local).
- **Date:** 2026-11-04
- **Deciders:** user + Claude Code
- **Extends** [ADR-034](ADR-034-in-app-provider-setup.md) (first run and the readiness gate) and
  [ADR-046](ADR-046-added-documents-copy-or-reference.md) (roots: files the library references
  where they live). Carries `docs/security.md` step **S-7** (path confinement) and the 2026-11-02
  review's must-hold **G1**.
- **In one sentence:** the app can switch to a library folder that was indexed elsewhere on this
  computer, such as a development checkout's `data/`, by restarting its backend on that folder
  after a read-only check of what it holds; and every file the app deletes, overwrites or serves
  through a path stored in a library must sit inside that library's folder or one of its registered
  roots, whichever library is open.

## To decide — nine choices, each with a recommendation

1. **What "add an already indexed library" means.** *Recommended:* **open it where it is**, on this
   computer: the app works on that folder as its library until told otherwise, and copies nothing.
   *Bring a copy in* and *open a library from another computer* come later. Both need every stored
   path rewritten first, because a library names its own files by absolute path (choice 4 and *In
   practice*, step 4), and that rewrite is its own design.
2. **Who holds the choice of library.** *Recommended:* the desktop shell. It remembers the folder in
   its own settings and starts the backend with `DOC_DATA_DIR` set to it, the variable the backend
   already reads first (`config.py:66`). Switching restarts the backend. The backend never changes
   its data home while running: it is read once at import, and the database engine, the vector
   stores and every derived path hang off it (`config.py:79-127`, `db/session.py:11`).
3. **What is checked before switching.** *Recommended:* a read-only `POST /api/library/inspect
   {path}` and a screen that shows its answer, with *Open* as the person's call. It reports the
   documents and passages the folder holds, whether its vector store has passages for the app's
   embedding model, how many stored source files are found where the library says they are, and
   whether the library's stored cache paths name this folder or another one. A folder that is not a
   library, or holds nothing the app can search, is named as such. Nothing is written by the check.
4. **Path confinement (G1, S-7).** *Recommended:* one function decides whether a stored path may be
   touched. A cache file or a figure image must sit under the open library's folder; a source file
   must sit under one of its registered roots. Every write and every file the API serves through a
   stored path goes through it (the list is in *Context*). Anything outside is refused with a
   sentence and logged, and the file is left alone. It holds for every library, the app's own
   included, not only one that was opened.
5. **A library at a path with a non-ASCII character.** The vector store library cannot read such a
   path, so the app keeps that library's vector stores under `%PROGRAMDATA%` instead, keyed by a
   hash of the path (`config.py:82-100`, KI-11). A library indexed at an ASCII path and opened at a
   non-ASCII one would find no passages. *Recommended:* the check says so and *Open* is refused
   with the reason and the remedy (a folder whose path has no accent). Copying the stores across is
   later: it is a hidden copy of hundreds of megabytes.
6. **Two copies of the app on one library.** A development backend and the installed app on the
   same folder would both write to it. *Recommended:* the screen says one app at a time, and the
   check reports a library whose database is held open by another process when it can tell.
   Nothing stronger in 0.7.2.
7. **The way back.** *Recommended:* the shell keeps the previous folder. If the backend does not
   answer after a switch within the wait the start-up gate already allows, the window says the
   library could not be opened, with its reason when the backend logged one, and offers *Go back to
   the previous library*. *Use the app's own library* is always offered once another is open.
8. **KI-79 and ROADMAP 100** (the backend stops after start-up; port 8001 is taken). *Recommended:*
   only the switch's own failure rides here (choice 7). The general states stay with row 100.
9. **Development builds.** They run without the bundled backend, so the shell has nothing to
   restart. *Recommended:* the control is not shown there, as the folder picker is not shown in a
   browser (`canPickFiles`). `DOC_DATA_DIR` stays the developer's route.

## In practice — opening the scratch library from 2026-11-04

The library used to check 0.7.2's indexing screen: 36 generated markdown documents, 1,080 passages,
100 MB, indexed by a development backend into `C:\Projects\pv-idx\data` from documents kept in
`C:\Projects\pv-idx\docs`. No real library's data is used here.

1. **Today.** The installed app always opens `%LOCALAPPDATA%\doc_assistant\data` (`config.py:69-75`):
   the desktop shell sets only a launch token and its own process id when it starts the backend
   (`src-tauri/src/lib.rs:106-108`). There is no way to reach the scratch library without setting a
   variable by hand, which `docs/desktop-packaging.md:66-71` already calls not a shippable UX.
2. **Open it where it is (recommended).** Settings › Documents › *Open another library…*, the
   folder `C:\Projects\pv-idx\data`. The check reads `library.db` and the vector stores, writes
   nothing, and the screen says: *36 documents, 1,080 passages, searchable with this app's model
   (bge-base). 36 of 36 source files found where the library says. Its cache is in this folder.*
   *Open*: the shell remembers the folder and restarts the backend with
   `DOC_DATA_DIR=C:\Projects\pv-idx\data`. The path is ASCII, so the vector stores are read from
   inside the folder. Each document's stored source path,
   `C:\Projects\pv-idx\docs\00-retrieval.md`, exists; its cache file is found at the path the app
   derives from that source path, so every document reads *indexed*. Nothing is re-read. Settings
   shows *Data home: C:\Projects\pv-idx\data* and the library's own source folder, which lives in
   that library's `settings.json`.
3. **A write after opening.** Re-running *Text & chunks* on `00-retrieval.md` deletes its cache file
   before extracting again (`reingest.py:561-565`). The stored path is
   `C:\Projects\pv-idx\data\cache\referenced\34d333ca82e8b9ca\00-retrieval.md`, inside the open
   library: allowed.
4. **The same library copied with File Explorer to `D:\libs\pv-copy` and opened there.** Every
   stored cache path still names `C:\Projects\pv-idx\data\cache\…`, the original. **Today** a *Text
   & chunks* re-run on the copy would delete the original's cache file, a delete would do the same
   (`purge_document_record`), and a full command-line ingest's orphan sweep too (G1; it happened to
   a scratch copy on 2026-11-01). **With choice 4** each of those writes is refused, the original
   is untouched, and the run says the cached text is outside this library. The check screen says it
   up front: *its cache paths name another folder, `C:\Projects\pv-idx\data`; it was copied or
   moved. Opening it is safe, nothing outside it will be written, but re-running a document's text
   needs the library where it was made.* This is why *bring a copy in* waits for a rewrite of every
   stored path.
5. **The source folder moved too.** For a file outside the library's own `sources/` folder, the
   cache file's name is a hash of the file's absolute path (`ingest/cache.py:57-65`). If
   `C:\Projects\pv-idx\docs` moves, the stored source paths are dead, the derived cache names
   change, and every document would read *changed* and be read again. That is *a library from
   another computer*: later, with the rewrite.
6. **Going back.** *Use the app's own library* restarts the backend on
   `%LOCALAPPDATA%\doc_assistant\data`. On a Windows account whose name has an accent, that path is
   not ASCII, and its vector stores are under `%PROGRAMDATA%` already; nothing about that changes.

## Context

- **The data home is read once, at import** (`config.py:57-79`): `DOC_DATA_DIR`, else
  `%LOCALAPPDATA%\doc_assistant\data` in the installed app, else the checkout's `data/`. Everything
  derives from it at import: the cache, the vector stores, `library.db` and its engine, the figures,
  the wiki, the concept skeleton, `settings.json`, the stored keys file (`credentials.py:38`). The
  keyword index sits beside the vector stores, so it follows their relocation (`sparse_index.py:59`).
- **The shell starts the backend** (`src-tauri/src/lib.rs:78-121`) and ends it on exit (`:25-47`).
  It sets no data home. The port is fixed at 8001 in the backend, the frontend and the
  content-security policy. `apps/api/CLAUDE.md` says `__main__.py` resolves the data home; it does
  not, and the note is out of date.
- **Absolute paths a library stores:** `documents.source_original` and `source_cache`, the same two
  in every passage's vector-store metadata, `figures.image_path`, `source_roots.path`, and
  `external_metadata.path_key` (`db/models.py:152-153`, `:358`, `:1193`, `:1138`).
  `source_files.rel_path` is relative to its root. `settings.json` holds the source folder.
- **What acts on a stored path today, with no check that it is inside the library or a root:**
  `reingest._rerun_text` deletes `source_cache`; `_rerun_crops` writes to `image_path`;
  `purge_document_record` deletes `source_cache`; `delete_document` sends the resolved source to the
  Recycle Bin; `reveal_document_source` opens it in File Explorer; the source viewer reads it;
  `cleanup_orphans_chroma(also_clean_cache=True)` deletes the `source_cache` named in vector-store
  metadata (the command-line full ingest only: the app's runs pass a scope or a file list, and
  `ingest.main` sweeps only without either); `/api/figures/{id}` serves `image_path`. Only the undo
  of an add is gated (`library/add.py:885-917`: the library root, a copied file, the undo window).
- **`docs/security.md`:** S9 lists these paths; step S-7 (planned in this piece) is *refuse to open
  a `source_original` under no registered root*, with the check inside `resolve_source_path`, and
  the cache and figure paths confined in the same step. S9 adds that a root check alone does not
  stop a row added by reference, since a reference add registers the file's folder as a root.
- **Nothing exists** to switch, copy, import or export a library (a search of `src/`, `apps/` and
  `scripts/`). `GET /api/settings` reports `data_home` and Settings shows it read-only.

## Options

- **A — Open where it is** (recommended). Small: the shell remembers a folder and sets one
  variable, the backend already honours it, and a read-only check guards the switch. The library's
  stored paths stay valid on the computer that made it. It fits the request's likely case, a
  library a development checkout built. It does not cover a library from another computer.
- **B — Bring a copy in.** Safer for the original, but every stored path in the copy names the
  original and must be rewritten before any write (*In practice*, step 4), the copy doubles the
  disk (about 6 MB per document, `docs/performance.md`), and on an account whose name has an accent
  the copied vector stores must also move under `%PROGRAMDATA%` (choice 5). It is a rewrite with a
  test of its own; it can follow A without changing it.
- **C — Index again.** Always works, needs nothing new, and is what the user asked to avoid: a
  library of a hundred PDFs takes on the order of the first ingest again.

## Decision

Proposed: **A**, with choices 2 to 9 as recommended. Choice 4 is built first and holds for every
library, so the app's own library gains the same protection whatever the user decides about A.

## Consequences

- A library opened elsewhere is used in place: what the app writes (new documents, caches, a
  migration of `library.db` to the app's schema, settings) goes into that folder. The check screen
  says so before *Open*.
- Confinement can refuse a write the app makes today. On a library where everything sits inside its
  folder and roots, nothing changes; on a copied or moved library, a re-run of a document's text is
  refused with a reason instead of deleting a file elsewhere.
- The shell owns one more setting and one more restart path. A failed switch is visible and
  reversible (choice 7); KI-79's general states stay open.
- *Bring a copy in* and *a library from another computer* stay open, behind one design: rewriting
  every stored path, and the cache names that derive from source paths.
- **Done when** (plan, piece 5): the user's own library opens in the installed app with its counts
  unchanged and no document re-read, driven; a test proves no write over an opened library touches
  a path outside its folder and roots; and *Go back* works after a switch that fails.

## Confidence

High on the facts (each read in the code, 2026-11-04) and on choice 4. Medium on choice 2's restart:
the shell's start and end of the backend exist, a restart between them is new, and port 8001 must
be free again before the new backend binds it. Choice 6 rests on a check that may not always be able
to tell; the screen's sentence does not depend on it.
