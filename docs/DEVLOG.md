<!-- status: active · updated: 2026-11-26 · class: append-only -->

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
> **2026-11-05 (3) → 2026-11-20 (2)** in [`docs/archive/DEVLOG-archive-010.md`](archive/DEVLOG-archive-010.md)
> (rotated 2026-11-09: 11 entries, 2026-11-12: 11 more, 2026-11-16: 11 more, 2026-11-20: 11 more, and 2026-11-23: 11 more) ·
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

## 2026-11-27 — 0.8.0 published; the front docs say so

**What.** The README's *Status*, `docs/OVERVIEW.md` §5 and the ROADMAP's *Where the product
stands* and *Releases* move to v0.8.0, published 2026-11-27 (tag on `979ce4b`, installer built
2026-11-26 12:31 from `ed3316b`, its SHA-256 on the release page). Their *Next* lines carry the
user's order from the release walk: extraction first (an in-depth mode, and an assisted one where
the user corrects what extraction got wrong), then the shape of answers, then concept curation.

**Why.** RELEASE.md's after-release steps; OVERVIEW §5 still named 0.7.1 as the latest release.
The README counts the tag's suites: 3,741 Python tests collected, 459 desktop tests passing.

**Rejected.** Listing the walk's 29 design points in the ROADMAP: they are planned locally
(`docs/plans/PLAN_2026-11-24_release-0.8.0.md`), and the PR table gains rows when the next
release is planned.

**Opens.** Planning the next release from that order.

## 2026-11-26 (5) — The 0.8.0 walk closes: four known limits, the release day, a secret ignored

**What.** `CHANGELOG.md` § 0.8.0 is dated 2026-11-26, the tag's day, and gains four *Known limits*
from the owner's walk on the 12:31 build: *Check now* keeps its last *up to date* beside its error
when the window cannot reach the engine (KI-100); Ollama's state in Settings is the last check, not
a live one; a claim citing several passages opens only the first; the amber of a weak claim is hard
to see in the dark theme. Separately, `.gitignore` names `data/credentials.json`.

**Why.** The walk found no blocker; what a user would meet goes under *Known limits* (RELEASE.md
§5b). KI-100 was reproduced in Windows Sandbox with networking off: with the engine reachable the
check says it could not check and records it, as ADR-044 requires; only a window cut from its own
engine keeps the old verdict. `credentials.json` holds the API keys the app saves
(`credentials.py`); `.gitignore` lists `data/` files by name and this one was never added, so a
`git add .` would have staged it. It was never committed.

**Rejected.** Fixing KI-100 before the tag: a window change re-freezes the backend and repeats the
sandbox runs (RELEASE.md §4), for a case a real offline user does not meet. Ignoring all of
`data/`: the tracked files under it (the vendored data) would have to be re-listed as exceptions.

**Opens.** The 29 design points of the walk, in `docs/plans/PLAN_2026-11-24_release-0.8.0.md`,
ordered by the owner for after the release: ingestion first.

## 2026-11-26 (4) — RG-012 repairs the sandbox's WebView2 registry

**What.** `scripts/rg012/rg012-run.ps1` points the sandbox's WebView2 registry entry (`pv`, and
ClientState's `pv` and `EBWebView`) at the one version folder on disk when the registered one is
missing, before it installs, and logs `webview2: …`. The repair is `walk080-logon.ps1`'s, in use
by the walk sandboxes since 2026-10-29. A test pins it ahead of the install and the launch;
`desktop-packaging.md` §5 gains trap 7.

**Why.** The first RG-012 run on the 12:31 0.8.0 installer (session 3 redone for KI-99) installed
in 77 s, then waited 600 s for a health that never came: the sandbox registry named WebView2
157.0.4341.60, the shared files are 158.0.4370.33 since the host updated, so no window and no
engine. Archived as `out-2026-11-26-v0.8.0-FAIL-webview2-registry`. A sandbox artefact: a real
machine updates the registry and the files together.

**Rejected.** A logon step in the `.wsb` before the harness: two scripts to keep in step, and the
gate is meant to run the one tracked file. Waiting for the sandbox image to catch up: it follows
the host's servicing, not WebView2's own updates.

**Opens.** Nothing; the installer is unchanged, so no rebuild.

## 2026-11-26 (3) — cpc re-vendored, 1.22.1 to 1.26.1

**What.** 2026-11-26 — re-vendored cpc 1.22.1 -> 1.26.1 (tools/conventions/cpc/, 83 modules), from a
worktree of the `v1.26.1` tag, on Lucas's "update Provenote"; nothing laid. `cpc-verdict-diff --base
v1.22.1 --head v1.26.1` named one new warning: ADR-064's `Enforced by` names `src/layout.test.ts`,
which is at `apps/desktop/src/layout.test.ts`; an amendment gives the path. The cpc gates and
`init_check --strict` pass. CONTEXT.md and the `conventions.toml` header name 1.26.1.

**Why.** cpc 1.26.1 carries T-100, reported from here: the security template now has the shape
`docs/security.md` took on 2026-11-23, so nothing changes in that file. And sprint-close stops and
asks for `/house-skills:sprint-review` and `/house-skills:deep-review`, which the agent cannot start.

**Rejected.** `cpc-rotate --split` on the five archives over `archive_max_tokens` (a NOTE each,
nothing required), and the ADR-101 NOTEs on the plan's lettered headings: both left as they are.

**Opens.** The 8 deny rules `--add deny-rules` would add, and `## Resolved index`, the
`docs/KNOWN_ISSUES.md` heading `cpc-drift` says a cpc tool reads: each on Lucas's yes.

## 2026-11-26 (2) — virtualenv 21.15.13 for an advisory

**What.** `uv lock --upgrade-package virtualenv`: 21.14.12 → 21.15.13, nothing else re-resolved.
The 21.14.5 entry beside it serves Python below 3.11 only and is never installed here.

**Why.** `scripts.pip_audit_gate` failed on PYSEC-2026-4402 (fixed in 21.14.13), published since
the last run; CI runs the same gate, so the next push would have gone red. Dev only: virtualenv
comes in through pre-commit (`--extra dev`) and is not in the installer.

**Rejected.** Reviewing it into `pip-audit-ignore.toml`: a fix exists, and the bump is one package.
Pinning 21.14.13: uv took the newest release; the hooks ran on it.

**Opens.** Nothing.

## 2026-11-26 — The window's saved lists are tied to the app version (KI-99)

**What.** `apps/desktop/src/lib/core/lastSession.ts` stamps each saved list (documents, folders,
chats, graph) with the app version and reads back only a copy from the same version. Vite sets
`__APP_VERSION__` from `package.json` (`vite.config.ts`, declared in `vite-env.d.ts`); `npm test`
sees `dev`. Two tests in `lastSession.test.ts`: another version is ignored both ways, and a copy
shaped like 0.7.3's chat list (format 1, no stamp, rows without `labels`) is not drawn.

**Why.** The owner's walk on the 18:40 0.8.0 build: installed over 0.7.3 on the real machine, the
window stayed blank on every launch while the engine was healthy. Read through a WebView2 debug
port: `TypeError … reading 'folders'` in the chat list's label line (`lib/chat/setup.ts`
`labelLine`). 0.8.0 drew 0.7.3's saved chat list before the engine answered; its rows lack the
fields A7b/A8a added, the first row threw at mount, and the fresh fetch that would replace the copy
never ran. The format number had stayed 1 since the copies landed, and nothing asks anyone to bump
it when a list's shape changes. Every sandbox run started from a fresh window profile, so none
saw it. A blocker: a data loss or a lie blocks the tag, and this left the app unusable.

**Rejected.** Bumping `SNAPSHOT_VERSION` to 2: fixes this release, leaves the next one to the same
memory. Guarding `labelLine` against a missing field: fixes one reader of four lists. A per-type
hash pinned in a test: catches the change but makes every type edit a ritual. The cost of the
chosen fix: the first launch after an upgrade shows the start screen until the engine answers, as
before 0.7.x kept copies.

**Opens.** Any error at mount still blanks the whole window (no boundary at the root); KI-99 keeps
that half open. The fix goes back through sessions 2 to 3 of the 0.8.0 plan: a new build, RG-012,
and an upgrade from 0.7.3 whose window has saved lists; then the owner's walk again from step 1.

## 2026-11-24 (2) — The 0.8.0 chat baseline, reproduced from a fresh clone (H6)

**What changed.** `tests/eval/baselines/release_0.8.0_chat_2026-11-24.md`: five families, two
local models, three trials, on `7d29dd6`, paired against 391ea0e. `TESTING.md` gains § Running the
chat families, the commands the clone ran; `evals/README.md` § The chat benchmark (0.8.0); the
README's results and `OVERVIEW.md` §4 from it; `RELEASE.md` §3 gains the chat baseline and the
stage benchmark, with this machine's reference in `stage_profile_public_2026-11-24.md`. KI-96 filed.

**Why.** H6: a stranger rebuilds the public numbers, and every README number has a baseline file.

**Measured.** $0, 62 min clone to last run; index digest `614c1241…`, 391ea0e's. Records 3 to 8 of
8 by rule on both models; citation coverage there falls to 0 by design. Llama's absent
`contains_all` −0.267 [−0.467, −0.067] on n=5 is a one-word check matched by guessing answers
before; `no_invention` 6 to 8 held. Nothing else moved beyond its interval.

**Rejected.** Running the walkthrough's models through the LLM judge (H5: none passed); copying the
dev checkout's data homes (not a stranger's run).

**What it opens.** KI-96. The README's August Haiku numbers stay, labelled as the older single-turn
run; no billed model ran on the chat families.

## 2026-11-24 — The app's wording, first pass for AI-writing tells (T-014)

**What changed.** 63 strings in 33 files reworded: em-dash asides in hints, empty states, tooltips
and confirmations became plain sentences, colons or semicolons; the chat's provenance lines lost
their emoji section markers (🔍 🔎 🧪 📊 🖥 📡 🧑 📄), keeping ⚠ where it marks a warning; a few
phrases lost jargon ("Zero-LLM … (bge embedding)", "Low graph degree"). Five tests that pinned the
old text follow.

**Why.** The owner still found AI-written tells in the app (T-014), to fix before 0.8.0's tag.

**Measured.** $0. A read-only sweep of `apps/desktop/src` (attributes, text nodes, script strings),
`apps/api` and the engine modules whose sentences reach the window found 193 hits; most were
comments or markup. After: none in visible text but the kept ⚠. Touched tests 215 passed; npm test
457; svelte-check clean.

**Rejected.** Dropping ⚠ too (it carries meaning, not decoration); rewording command-line-only
messages (`--doc`'s prefix error).

**What it opens.** The sweep reads one line at a time, so text split across template lines and
engine sentences outside the listed modules were read by eye only in part. `.claude/NORTH_STAR.md`
is still the unfilled template, so there is no written register to check against. Raw exception
text still reaches the chat in four messages (`Edit failed: {e}` and three like it).

## 2026-11-23 (10) — The Rust tree is audited in CI, and Tauri is 2.12 (S-16)

**What changed.** `scripts/cargo_audit_gate.py` runs `cargo audit --json` over
`apps/desktop/src-tauri/Cargo.lock` and fails on any vulnerability or warning (unmaintained,
unsound, yanked) not reviewed in the new `cargo-audit-ignore.toml`. It reuses the pip-audit gate's
ignore parser and fields, and fails closed with no report or a broken list. A new CI job,
`rust-audit`, builds `cargo-audit` 0.22.2 with `--locked` and runs it; `just audit-rust` does the
same locally. CI's sync adds `--extra packaging`, so `pip_audit_gate` sees PyInstaller and its
hooks. Tauri 2.11.3 → 2.12.7 with its plugins and `@tauri-apps/cli` 2.12.6; `plist`, `cookie`,
`url`, `tokio`, `anyhow` updated. `rust-version` 1.90 (Tauri's MSRV). `nsis/installer.nsi`
re-taken from the CLI 2.12.6 source with the one KI-73 line.

**Why.** S16: ~450 crates and the packaging extra went into every installer with nothing auditing
them. The update cleared both `quick-xml` advisories and two found today (RUSTSEC-2026-0594 in
`cookie`, -0592 in `url`), and the `anyhow` and `tokio` warnings. 2.12.0 also fixes
GHSA-w28w-mhc8-qvjv (channel IPC payloads readable across webviews). The template's header asks
for a re-take on every CLI upgrade: 2.12.3 and 2.12.5 changed downgrade blocking and a language
string in Tauri's own copy, and the old copy would have kept the 2.11.4 behaviour.

**Measured.** $0. Gate: 453 crates, 3 reviewed warnings (`chrono` unmaintained, in the lock only
as an unused optional dependency; `glib` unsound and `proc-macro-error` unmaintained, both in the
Linux GTK stack, absent from `--target x86_64-pc-windows-msvc`), 0 unreviewed. The packaging set,
Windows-only parts included, has no known pip-audit advisory. `tauri dev` builds with no warning;
the dev window polled the scratch API at `C:\Projects\pv-a8pp-cpu`. 13 new gate tests.

**Rejected.** `.cargo/audit.toml`'s own ignore list (no field for why or what reverses it, and it
lets warnings pass); a third-party install action (one more thing to trust and pin); a full
`cargo update` (229 crates, past what the release notes were read for).

**What it opens.** The installed build of this tree, driven at A9 with A8 and A8″ (the owner,
2026-11-23). `pefile` and `pywin32-ctypes` are not installed on the Linux runner, so CI's pip
audit does not see them. In dev mode the shell starts the stale frozen sidecar in `binaries/`
against the real `data/`; with the dev API already on 8001 it exits at the bind.

## 2026-11-23 (9) — The commit hook type-checks `apps/` as CI does

**What changed.** The mypy hook runs `mypy src apps` and triggers on a staged `src/` or `apps/`
file (`.pre-commit-config.yaml`); `just typecheck` runs the same. AGENTS.md, the backend's
CLAUDE.md, `docs/RELEASE.md` and CONTEXT §8 name the new command.

**Why.** CI has run `mypy src/ apps/` since 2026-09-10; the hook checked `src` only, so (8)'s error
first showed on main.

**Measured.** $0. Warm runs: `mypy src apps` and `mypy src`, in either order, both clean on the
same cache. The old `.mypy_cache` gave a false `doc_assistant.mcp_server` error on every warm
`src apps` run, which no source change explained; it was moved aside and rebuilt (46 s cold).

**Rejected.** A separate hook for `apps/` (two runs, two caches to keep warm).

**What it opens.** Nothing.

## 2026-11-23 (8) — CI's type check on `apps/` fixed after A8″a

**What changed.** `GET /api/ingest/files` keeps a removed item out of its rows by a check the type
checker can read (`apps/api/routers/sources.py`). `run.page()` never lists one, so nothing changes
at run time; but its rows carry the full state type, `removed` included, which the payload's state
type leaves out.

**Why.** A8″a's push failed CI: CI runs `mypy src/ apps/`, while the commit hook and
`just typecheck` run `mypy src`, so the error never showed before the push.

**Measured.** $0. `mypy src/ apps/` clean on a cold cache (a warm cache built from `mypy src`
alone reported a false `mcp_server` error); the ingest route tests and the GET floor test, 63 passed.

**Rejected.** Adding `removed` to the payload's state type: the API never sends it, and the
window's `IngestFileState` would drift from it.

**What it opens.** The hook and `just typecheck` still check only `src`: the same gap can let
another `apps/` type error through to CI.

## 2026-11-23 (7) — Every ingestion is one list in the engine (A8″a, T-007, T-010, T-011)

**What changed.** An add, a selection, the library folder and a document's re-run are rows of one
list (`ingest/queue.py`, `ingest/run.py`): nothing is refused because something else runs, what
arrives joins the end, and one thread runs it in order (files in batches through `ingest.main`, a
re-run one document at a time). The reload happens once, when the list is empty. A waiting row
can be removed (`DELETE /api/ingest/items/{id}`), even inside the running batch. The window: *Add to
queue* and *Remove* on the indexing screen, the engine-start chip, the re-run dialog that says what
it waits behind, and the chat line's *Indexing: answers take longer*.

**Why.** T-007, T-010, T-011 (the owner, 2026-11-19); spec `docs/specs/ingestion-queue.md`,
revised the same day to one list of items rather than a line of jobs. The trace (spec § Today):
nothing locked; three job routes refused each other, and retrieval ran 3 to 15 times slower on the
CPU build during an ingest.

**Measured.** $0. Queue tests 7 (one for the removal bug), run tests 12, the route tests moved to
the new contract (156 in the affected files); window 457, `svelte-check` 0. Driven over a scratch
home on the CPU build: one list counted whole, added papers waiting at its end, a removal skipped,
a re-run waiting behind three others, the chat line's note.

**Rejected.** A line of jobs (the owner: one list); a queue stored in `library.db` (decision 1);
one `ingest.main` call per file (each call scans both stores for the dedup set).

**What it opens.** *Add to queue* and the engine-start chip are not driven: they need the desktop
shell, so A9's installed-build walkthrough covers them.

## 2026-11-23 (6) — No GET route writes or calls out (A8″b, S-17)

**What changed.** The four GETs with a side effect lost it, each to a POST the window calls:
`POST /api/sources/scan` (the walk that registers new files; `GET /api/sources` now lists the
registry read-only, `registry.stored_sources`), `GET /api/ingest/plan` counts the stored rows
(`plan_library(scan=False)`) after the window's scan, `POST /api/updates/auto` (the opted-in daily
check; `GET /api/updates` answers from the last one), `POST /api/setup/probe` (the model-server
probe; `GET /api/setup` never probes). The floor test's `KNOWN` list is empty.

**Why.** `docs/security.md` S15 and S-17 (T2: a page can make the engine run a GET); spec
`docs/specs/ingestion-queue.md` decision 9.

**Measured.** $0. The floor test (every GET route, 49) passes with nothing listed; route tests moved
to the POSTs, plus a test each that a GET never checks for updates and never probes. Window 456.

**Rejected.** Keeping the walk in the GET behind a header: a page cannot send one, but any other
caller still triggers writes on a read.

**What it opens.** Nothing reachable from a web page is open in `docs/security.md`. S-16 is next.

## 2026-11-23 (5) — The letter-spacing block at the foot of the page, folded once ruled (A8″b, T-009)

**What changed.** The block moved below References on the document page, with a fold in its
heading: open while the document has spacing to judge and no rule, or a change waiting for a re-run;
closed once ruled (`lib/library/spacing.ts` `spacingOpenByDefault`, tested). A test drives a *Keep
every run spaced* rule on a real PDF through a per-part text re-run, a folder index, a rebuild and
an extraction from scratch.

**Why.** T-009 (the owner, 2026-11-19); spec `docs/specs/ingestion-queue.md` decision 10.

**Measured.** $0. Window 456 (one new), `svelte-check` 0; the re-ingest test passes in about 10 s.
Driven in the dev window over a scratch home (a letter-spaced PDF): last on the page, open with no
rule, open while the change waited, folded by hand with the count shown, closed once the re-run
applied it; dark, light and 375 px, no console error. Not yet in the installed build.

**Rejected.** Remembering open or closed per document: the default already follows the rule.

**What it opens.** Nothing further for T-009 once it is driven in the installed build (A9).

## 2026-11-23 (4) — `docs/security.md` rewritten around the questions a reader brings

**What changed.** The file keeps its section numbers, so every citation of §1 to §7 still holds, but
each section now answers one question. §0 is a status (open findings by reach, the current step,
what is due). §2 states each control in the present tense with its test, grouped by boundary,
instead of a log by release. §3 and §4 hold open findings and steps only, in short cells; a closed
finding is one line in §8 naming its step and DEVLOG date, and what it left accepted moves to §7.
`withGlobalTauri` is recorded as not done on purpose (the owner, 2026-11-23). The gap in `scripts/`
gating, stated in §2 since 2026-11-02 with no step, is now finding S25. The cpc proposal that was §8
goes to the cpc ticket.

**Why.** The owner, 2026-11-23: the file was practically unreadable. T-004 had named the cause.

**Measured.** 43,501 → 23,198 characters, 6,933 → 3,740 words; table rows median 537 → 112
characters, longest 1,316 → 314. Every file, test and function the file names exists.

**Rejected.** Splitting the file: at this length one file reads in one sitting, and a split would
put the open findings and the plan in different places again.

**What it opens.** The cpc template update, once the owner approves this shape.

## 2026-11-23 (3) — Security controls say when they fire, and GETs are held to no side effects (A8′, S-11)

**What changed.** Four structlog events, each with its control's name and never the content:
`security_host_refused` (a subclass of the host guard, `apps/api/host_guard.py`),
`security_path_refused` (`stored_paths.confine`, replacing `stored_path_outside_library`),
`security_cap_hit` (the three ingest caps in `extractors.ingest_refusal` and the add cap),
`security_markup_sanitised` (the prompt fence's defusal). A floor test walks every GET route over a
library where a write or a call-out would show (SQL writes, files under the data home, outbound
connections refused) and holds the known writers in a list that can only shrink.

**Why.** `docs/security.md` S-11 (findings S12, S15), in A8′.

**Measured.** $0, no model. `test_security_events.py` 5; the floor test 49 routes. It found a
fourth GET with a side effect that S15 had not named: `GET /api/ingest/plan` commits registry rows.

**Rejected.** A window-to-engine log route for DOMPurify's removals: a new write route for a log
line. Fixing the four GETs here: they need the window's clients changed too (S-17).

**What it opens.** S-17, the four GETs. §6's periodic check can now read its events.

## 2026-11-23 (2) — The small security ones: export file, Explorer, pinned models (A8′, S-10)

**What changed.** The history export writes a `mkstemp` file and removes it once sent, so a fixed
name in the shared temp folder can be neither planted nor left behind. *Reveal in folder* runs
`%WINDIR%\explorer.exe` by full path. The embedders and the reranker load a pinned commit
(`EmbeddingModelConfig.revision`, `RERANKER_REVISION`), the commits this machine's cache already
held, and the release build stages those same snapshots. `docs/setup.md` § Reporting a problem.

**Why.** `docs/security.md` S-10 (finding S11), in A8′.

**Measured.** $0. Tests: the export (a planted file untouched, nothing left), Explorer's path,
each model pinned to a 40-character commit and loaded at it, and the spec staging the same pairs.
The installed build was not built or run: the pinned snapshot is the one the bundle already holds.

**Rejected.** `withGlobalTauri: false` here: Tauri 2 injects its internal IPC object either way,
so it narrows little, and it adds two npm packages and reverses W0's choice. Left to the owner.

**What it opens.** The A9 walkthrough loads models offline from the pinned snapshot (RG-012).

## 2026-11-23 (1) — The MCP path and A6's lookups read against T6: six fixes, two logged (A8′)

**What changed.** An end-to-end test over a real ingest and a real retrieval pipeline (fake
embedder, word-overlap reranker) runs the plan's five evidence cases. Two failed and are fixed. A
text re-run left the superseded passages in the running engine's keyword index until a relaunch:
the run now reports the hashes it replaced, and the engine forgets them and reloads, as after an
ingest (S18). The parent-child dedup keyed on the file name, so two documents named the same
merged (S20): it keys on the text's hash. Read in the code and fixed: passage ids carry the
text's version, `{key}@{hash}`, and `get_passage` finds nothing for an older one (S19); a chunk's
stored cache path is read only inside the library's cache folder (S21); MCP arguments are checked
against each tool's schema and a failing tool answers `isError` with no message (S22); captions,
reference lines and cited excerpts carry the text check (S23, additive fields, version 1 kept).
List orders get a tie-break. An answer record's references survive a per-part and a full
re-ingest: the id is kept (ADR-047), the excerpt is its own copy, the page is re-placed or said
to be unknown.

**Why.** The plan's A8′ line; `docs/security.md` T6.

**Measured.** $0, no model. `test_read_surface_evidence.py` 6 (cases 1 and 2 red before the fixes),
MCP wire tests +11, read-surface +1, chunk context +1.

**Rejected.** Refusing unversioned passage ids: the chat's own keys and a definition's
`passage_id` use them. A tie-break in the keyword index's SQL: a repeat call is stable (case 5).

**What it opens.** S23's titles and authors and S24 (record answers' Markdown), both low, with B2.

## 2026-11-20 (4) — Claims coloured in the answer, and what a turn looked up and carried (A8b)

**What changed.** An answer's sentences are coloured by their retrieval marker (G4): lilac for a
claim citing a well-ranked passage, amber with "Cites a passage that ranked low for this question",
plain for an uncited one; matched to the shown text in order, a claim not found left plain;
clicking one opens its first cited passage, drawn darker while open. A records answer shows *What
it looked up* (lookup, arguments, rows, its first five rows opening their document, folder or
concept). The fold under an answer is now *How this answer was made* and holds *What the answer
carried* (earlier messages, carried exchanges, passages, the window and its source, mode and prompt
version). A reopened chat shows its claims, the decisions recorded on them and the reviewer's
recorded verdict. New `turn_details.py`; `TurnResult` and the replayed turn gain `claims`,
`records`, `context`, `chat_mode`, `prompt_version` (and `review`). Nothing new is stored.

**Why.** ROADMAP 103 (session A8b), rows 42 and 44; `docs/specs/chat-interface-pass.md`.

**Measured.** Tests and a driven check, $0, no eval run. `test_turn_details.py` 8, the
turn-parity, A8a setup and API chat tests unchanged and passing; desktop 455, `svelte-check` 0 errors. Driven over the scratch data home
with a seeded chat and a mocked live turn: `[s1]`, `[s7]`, `[s8]`, `[s10]`, both themes.

**Rejected.** Marking a sentence inside the cited passage (no character offsets); colouring by the
reviewer (G4); a second store for the records card's rows.

**What it opens.** Nothing left of row 103. Not foreseen, in the spec's As built: the fold renamed
from *Provenance & usage*, and a claim's highlight split by its citation buttons.

## 2026-11-20 (3) — A chat's mode and folders set when it starts, and Continue in a new chat (A8a)

**What changed.** A chat's mode and folders are chosen on the start screen *New chat* opens (mode
chips, *Search in* with the folder tickboxes, *All documents* set apart, the union count, sample
questions per mode, *Make this my default*) and stored on the chat by its first turn
(`conversation_meta.scope_json`, `continued_from`, `carried_exchanges`, additive); later turns
ignore the request's scope except a one-off `/folder <name>: <question>`. New module
`chat_setup.py`. The chat line at the top of a chat names the mode and folders with their count,
with *Continue in a new chat…*, whose dialog carries the old chat's exchanges, read from its
records, into the new chat's memory, fitted by A5; each record keeps `context_json.carried`.
*Continue this chat* on a reopened chat carries its own exchanges (open question 1). `/mode`
follows G2. *New chat defaults* (⚙ beside *New chat*) live in `settings.json` → `new_chat`. The
composer is the text box and *Send*. The rail's sort menu gains *Group* and *Only chats that
cited*; rows' date lines carry their labels. ADR-025's addendum amends S9.

**Why.** ROADMAP 103, session A8a; `docs/specs/chat-interface-pass.md` (accepted 2026-11-20) and
the approved wireframe (`docs/wireframes/chat.md`): the composer's pickers got in the way.

**Measured.** Tests and a driven check, $0, no eval run. Full suite: 3,626 passed, 3 skipped (one A7b-era test of S9's
unstored scope rewritten to the amendment).
`tests/integration/test_chat_setup.py` 28; the turn-parity test unchanged. Desktop: 444 tests,
`svelte-check` 0 errors, `vite build` clean. Driven over a scratch data home of 30 invented
documents in four folders, `/api/chat` mocked in the page: each of `[s2]`–`[s6]` and `[s9]`, both
themes, 375 px (the spec's As built lists what was not foreseen).

**Rejected.** Keeping a picker in the composer (the user circled it); summarising carried
exchanges (G3: never); dropping every gone default folder to an empty choice, which reads as the
whole library; a second label or grouping derivation in the window.

**What it opens.** A8b (claims in the answer, the records card, the carried card, a reopened
chat's claims). Not foreseen by the spec, in its As built: A5's history cap means a turn holds at
most the newest three carried exchanges, and only the rewrite call holds them.
