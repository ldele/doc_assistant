<!-- status: active · updated: 2026-11-12 · class: append-only -->

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
> **2026-11-05 (3) → 2026-11-10** in [`docs/archive/DEVLOG-archive-010.md`](archive/DEVLOG-archive-010.md)
> (rotated 2026-11-09: 11 entries, and 2026-11-12: 11 more) ·
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

## 2026-11-13 (2) — ADR-066 accepted as proposed

**What changed.** ADR-066 (*the library is offered to a model as records and passages*) is
accepted: option A, the nine choices and the seven read tools as written. Its status, its Decision
section, the index line in `docs/decisions.md` and ROADMAP rows 65, 102 and 37 say so. No code.

**Why.** It was the open decision that closes A1 in the plan. The user read it with the steer
that the MCP endpoint need not be complete in its first form and will grow; the ADR's table of
seams already carries that.

**Rejected.** A smaller first tool set (three or four tools, without the two conversation tools
and their switch), offered because choice 8 freezes what ships first: the user kept the seven.

**What it opens.** H1, the harness spec. Two texts waited on this acceptance and are still to
write: row 65's description (it still says a stdio server over `pipeline.py`) and the MCP-client
row in `docs/security.md`.

## 2026-11-13 — Repository upkeep: four tickets, and the easy fixes they found

**What changed.** Tickets T-003 to T-006 (the glossary's missing words, security.md's tables,
tables and status headers across the living docs, plan labels in code comments), and three reports
filed in the inboxes of the projects they concern: cpc's template justfile, cpc's docs gate, and
the code-descriptions skill. Fixed now: GLOSSARY's C-011 and C-012 read as built and sit in
order; security.md's §6 is a list, since an unescaped pipe in a code span split seven of its rows
on GitHub, and S3 no longer says it waits on S-6; the ADR-048 row in `docs/decisions.md` has its
Status cell; the CLI's `/help` stops saying *(Phase 4)*; the justfile goes from 184 to 133 lines
with every recipe unchanged; GLOSSARY and security.md lose the changelog in their status line.

**Why.** The owner asked for the upkeep to be ticketed and the easy part done. Each fix is a
statement that was wrong or a page that rendered broken; anything that needs a decision (a
canonical word, a table's new shape) is a ticket.

**Rejected.** Escaping the pipes in §6 rather than making it a list: the raw file would then
show `\|` in commands meant to be copied. Trimming comments across `src/` now: T-006 rides on the
0.8.0 chat and extraction work, which touches those files anyway.

The ticket ledger, `docs/TICKETS.md`, leaves the repository: it is gitignored like the known
issues, and a public report is a GitHub issue. T-001 and T-002 stay in the history. `soupsieve`
moves from 2.9 to 2.10 for CVE-2026-126192. CVE-2026-126190 is fixed only in 4.0.0, which needs
Python 3.11.5 and forks the lock; it is a reviewed ignore in `pip-audit-ignore.toml`, since
nothing here runs a CSS selector.

**What it opens.** T-003 needs the owner's words for the engine and for the library's folders.
The archive tables that render broken stay as they are, since archives are verbatim. The
`soupsieve` ignore lifts when the Python floor reaches 3.11.5.

## 2026-11-12 (3) — ADR-066 proposed: the library is offered to a model as records and passages

**What changed.** A new ADR for ROADMAP rows 65, 102 and 37, with its line in
`docs/decisions.md` and the three rows pointing at it. No code.

**Why.** The chat and MCP work needed its design before any build. A grill put nine forks to the
user, one at a time, and the ADR records each answer: a search returns passages and never an
answer from the app's model; the server runs inside the engine with a stdio bridge; seven
read-only tools; writes in the next release behind a second switch; the chat picks a lookup by
rule and command; a mode is a name and its framing; a conversation's labels are read from its
answer records; the tools' shapes are what freezes; the endpoint is off by default with its own
token. The user also asked that each piece stay manageable now and be able to grow, so the
Decision names the seam each one leaves.

**Rejected.** The model choosing its own tools, and an `ask` tool: different behaviour per model
and a model call at an outside program's request. A server in its own process, as row 65 was
written: a second embedding model in memory and a second program on one library.

**What it opens.** The user's read, which is what accepts it. Not verified: that the engine can
serve MCP beside its routes, and that a stdio bridge starts from an installed Windows build. Row
65's scope text and `docs/security.md`'s attacker row for an MCP client change when it is
accepted.

## 2026-11-12 (2) — ADR-062 accepted as proposed

**What changed.** ADR-062 (*the app stays repairable*) is accepted: option A with all eight
choices as recommended. Its status, its Decision section, the index line in `docs/decisions.md`
and ROADMAP row 100 say so. No code.

**Why.** The plan held it as an open question before the chat and MCP design. The user took each
choice on its own: the engine and the shell both degrade; the shell makes the update check when
no engine answers; four repairs that rename aside and never delete; a repair offered only for
damage; the reachability table as written; the program on port 8001 named, and only an older
engine of ours ended; a remembered folder with no `library.db` is a state; the engine answers
first and opens the library behind it.

**Rejected.** On choice 5, also showing the last session's settings read-only while a section
waits: a stale folder or provider shown as current can mislead. On choice 6, starting the engine
on a free port: the security policy and the dev setup both fix 8001.

**What it opens.** The build, in the ADR's order (choice 7 and the engine's log first), before the
MCP server. The entry before this one built the window half of choice 8's third bullet without
citing the ADR; the ADR's status now names it, and its `Settings.svelte` line numbers are those of
2026-11-05.

## 2026-11-12 — Settings opens before the engine answers (KI-93)

**What changed.** `screens/settings/Settings.svelte` renders its rail and pane at once. General
and Getting started read nothing from `/api/settings`, so they are whole straight away; Documents,
Provider & model and Retrieval say *Waiting for the engine* until the settings land. The read is
made again when `shell.status` turns `ready`, so the open drawer fills in place, and a read that
fails on a running engine shows its error with *Try again*. `GettingStarted.svelte` says it is
waiting, not *Couldn't read the setup state*, while the engine is not up.

**Why.** The user's 0.7.3 walkthrough: Settings showed nothing until the engine was online, and
the installed engine takes tens of seconds to start. The drawer asked once at opening, replaced
itself with *Loading…* or the error, and never asked again, so it had to be closed and reopened.

**Rejected.** Showing the last session's settings read-only, as the library is shown
(`lib/core/lastSession.ts`): three sections would each need a read-only mode, and a stale provider
or folder shown as current is worse than a line that says what is awaited. Polling from the
drawer: the readiness gate already polls, and `shell.status` is its answer.

**What it opens.** Checked in the dev preview with the engine cut off in the page and then let
back: rail, General and the waiting line, then the open page filled; `npm test` 416/416,
`svelte-check` 0/0. Not seen: a cold start of an installed build, and Getting started's waiting
line (it needs a start with no setup state yet).

## 2026-11-11 (6) — A broken extraction pool no longer stops an ingest run (KI-92)

**What changed.** `ingest/workers.py::warm_extraction_cache` wraps the whole pool block: a failure
of the pool itself logs `extraction_warmup_pool_failed` and returns, with every document it did not
finish counted as failed and left to the serial pass. `tests/unit/test_ingest_workers.py` gains a
stub pool and a test that breaks it at the first, second and third `submit`; the frozen-guard test
now uses the stub too.

**Why.** CI failed on the `v0.7.3` commit with `BrokenProcessPool` raised at `pool.submit`, outside
the per-result guard. The test fakes `sys.frozen`, so every real child starts with
`--multiprocessing-fork` and dies (30 of 30 jobs in a probe here); it passed when the break reached
`fut.result()` and failed when it reached the next `submit`. The test was flaky, and the code under
it was wrong: the warm-up's docstring promises a failed pool leaves ingest as slow as serial.

**Rejected.** Catching only `BrokenProcessPool`: a child that cannot start raises other errors, and
the step is optional either way. Retrying with a fresh pool: what killed one child may kill the
next, and the serial pass already does the work.

**What it opens.** CI's Linux runner, where the race showed, is the real check; the two failures
were on runs 40041172523 and 43070327008.

## 2026-11-11 (5) — The state lines say v0.7.3

**What changed.** AGENTS.md's *State*, the README's *Status* and the ROADMAP's *Where the product
stands* and *Releases* now name v0.7.3 (2026-11-11) and what it added; the README's *Next* names
the chat and MCP, as the ROADMAP's order does. ROADMAP row 60's current security step is S-9, the
prompt fence: S-7 was done on 2026-11-04 and the row still named it.

**Why.** `docs/RELEASE.md` §0 asks for it at each release, and three surfaces still said v0.7.1.
The test counts are the tag's: 3,233 Python tests collected (CI's own count on `a8408c0`: 3,226
passed, 6 skipped, 1 failed, KI-92) and 416 desktop tests.

**What it opens.** KI-92, filed beside this: the worker test that failed CI on the tagged commit
is a real escape at `pool.submit`, not only a flaky test.

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
