<!-- status: active · updated: 2026-11-27 · class: append-only -->

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
> **2026-11-20 (3) → 2026-11-23 (10)** in [`docs/archive/DEVLOG-archive-011.md`](archive/DEVLOG-archive-011.md)
> (rotated 2026-11-27: 12 entries) ·
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

## 2026-11-27 (3) — S-15: CI installs a named uv, and its token reads only

**What.** `.github/workflows/ci.yml` gains a top-level `permissions: contents: read`, and
`setup-uv` installs `version: "0.11.14"` (the dev machine's) instead of `"latest"`. Two tests in
`tests/unit/test_ci_actions_pinned.py` fail on a `setup-uv` step without an exact version and on a
workflow without read-only permissions; both fail on the previous file. `docs/security.md` closes
S17 (open findings 5 → 4), ROADMAP row 60 names the periodic full check as next.

**Why.** The plan's one security step per session; S-15 was the placed step left after S-16.

**Rejected.** Per-job `permissions:`: no job writes anything, so one top-level line says it once.
A version range for uv: a range is still a new uv on the day.

**Opens.** The first CI run on this file is the proof that the GHA build cache still writes under
`contents: read` (it uses the runtime token, not `GITHUB_TOKEN`; not seen yet). Raising uv is a
hand edit, since Dependabot bumps the action, not its `version:` input.

## 2026-11-27 (2) — The walk's four small fixes: a failed check, Ollama's state, every cited passage, dark highlights

**What.** Session 1 of `PLAN_2026-11-27_release-0.9.0-ingestion.md`, desktop only:
- **KI-100.** A *Check now* that never reaches the engine shows `unreachableStatus()`
  (`lib/settings/updates.ts`), *Couldn't check for updates*, not the last green verdict.
- **Ollama's state (walk items 11, 12).** `engineDown()` (`lib/settings/setup.ts`) reads the last
  probe; the status bar says *Ollama is not answering*, with *Check now*. A window focus probes
  again, at most once per 30 s (`recheckDue`); Settings' card follows `shell.setup`.
- **Every cited passage (item 25).** `passagesFor()` (`lib/chat/claims.ts`) resolves all of a
  claim's sources in its order; the panel stacks one card each under *Sources [1] [6]*.
- **Dark highlights (item 6).** A new `--mark-bg` token: the weak claim and the passage mark use it,
  `--warn-bg` stays the notes' fill. Dark `#54421a`: 1.83:1 off `--bg` (was 1.20), text 7.7:1;
  light unchanged. `src/theme-contrast.test.ts` pins both ratios from `app.css` (`lib/core/contrast.ts`).

**Why.** The owner's 0.8.0 walk; the plan's session 1. Each has a test that fails without it
(471 desktop tests, 7 new); svelte-check clean.

**Driven** in the Vite preview over the dev engine (not `tauri dev`: its frozen sidecar is stale),
the page's `fetch` faking the failures; all four seen, both themes. **Not driven:** the passage mark
in its surrounding text (only a live answer's source has a chunk key).

**Measured.** One probe takes 2.6 s here (three runs: 2,623, 2,649, 2,635 ms), waiting on the
compatible servers nobody runs; so no timer, as the plan said: focus and the button only.

**Rejected.** Changing `--warn-bg`: twenty notes use it, where faint is right. A 3:1 bar for the
dark fill: text on it would fall under 4.5:1.

**Opens.** The lilac *cited* fill is 1.24:1 in dark too; the walk did not name it. Walk item 11's model panel (Scribe's
Settings first) stays for the Settings pass.

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
