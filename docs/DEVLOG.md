<!-- status: active · updated: 2026-09-10 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps the newest 20 entries** (`devlog_max_entries = 20` in `scripts/conventions.toml`,
> cpc rule 13b, user standard 2026-09-10; `tests/unit/test_doc_sizes.py` pins the same number).
> Rotate with `python tools/conventions/cpc/rotate.py --root . --file devlog --write` — it moves the
> oldest entries **verbatim** into the highest-numbered archive and verifies the bytes; then update
> the range below by hand (cpc ticket T-003). A day may be split across two files at the cut.
> Older entries, newest-first, unedited:
> **2026-08-12 (1) → 2026-08-30 (9)** in [`docs/archive/DEVLOG-archive-006.md`](archive/DEVLOG-archive-006.md)
> (rotated 2026-09-04 and 2026-09-10) ·
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

## 2026-09-10 (2) — Working docs get folders, the DEVLOG keeps twenty, the release checklists are stale by default, and security becomes one step per session

**What changed.** Second session of the day, on the user's reading of the review. **Folders
(ADR-051):** the five live plans moved to `docs/plans/`, the two reviews to `docs/reviews/`; each
folder has a tracked README; the dated files stay local-only by *pattern* (a folder-level ignore
would have taken the README with it); fifteen living files repointed (ADRs 028/030–033/046, the
add-documents spec, ROADMAP, ui-checklist, knowledge-layer, local-only, README, the three local
trackers) — append-only logs and the CHANGELOG keep their old paths as history. **Entry budgets:**
`devlog_max_entries = 20` in `scripts/conventions.toml`, `cpc-rotate --file devlog --write` moved
56 entries (2026-08-15 → 2026-08-30 (8)) into archive 006 byte-verified, and
`tests/unit/test_doc_sizes.py` pins the same number so a clone or CI sees it. **Release:** the two
checklists are assumed stale — `docs/RELEASE.md` §0 is now the first step (list what shipped, add
rows, bump), and `release_preflight` gained `check_checklists_refreshed` (both files must carry a
commit or an uncommitted edit since the previous tag; six tests, a pinned file list). **Security:**
`docs/security.md` restructured — §0 how the file is worked, §4 an ordered plan S-1 … S-12 sized
one step per session with a `done when` per step, §6 the periodic full check (eleven lenses, each
with its command) whose log is `.claude/REVIEWS.md` row 7; S-12 (Dependabot + SHA pins) marked as
the user's call. **Roadmap:** re-sequenced to the user's order — gaps first with one security step
per session (row 60 is now the standing "current step" row), the knowledge layer made honest
(KL1 → 53 + 54 → 51 → KL4 → KL2), then row 25, then a **graph re-pass** (new row 75: 52 · 57 ·
50 · 43 · a UX pass); a twelve-session sequence table; row 62 done → archive. **cpc:** tickets
T-011 (the folders) and T-012 (the four consumer-grown conventions) opened with `cpc-ticket` in
the cpc checkout — uncommitted there, for the next cpc session.

**Why.** Each is a user decision from the second session: folders for plans and reviews as the
standard going forward; twenty entries as the DEVLOG standard; checklists that "would never
otherwise be up to date"; security worked a little at a time with a thorough check at the end and
a standard place to log it; and the order gaps → knowledge → ingestion → graph.

**Rejected.** Ignoring the two folders wholesale (kills the tracked README); judging checklist
freshness by the `updated:` header (bumpable without a refresh — git history, like
`artifact_fresh`); hand-cutting the DEVLOG on a day boundary (the tool rotates by count and
verifies bytes; the split day is recorded in both headers); a separate security log file (the
REVIEWS ledger already has the shape and the "not covered" discipline); editing cpc's CONVENTIONS
directly (a ticket is the channel the ledger asks for).

**What it opens.** The DEVLOG header's archive range is still updated by hand (cpc T-003). Session
1 of the sequence is row 46 + security step S-1 (ingest size caps). `docs/sprints/` is laid by cpc
and unused — the T-011 answer may settle it. Stamps: the first draft of this session said
2026-09-11 in fourteen files; corrected to 2026-09-10 before staging.

---

## 2026-09-10 — Whole-project review: the roadmap goes per feature, the security floor is written down, and a ninth version file turns up at 0.4.2

**What changed.** A review session, not a feature session. Three read-only audits (repo health,
security surface, file architecture) and a read of every coordination doc produced
`docs/REVIEW_2026-09-10_project-review.md` (local, ADR-029) and these tracked changes:
`docs/ROADMAP.md` restructured into one table of **open** rows with a `Feature` column and twelve
feature sections — the 87 done rows moved **verbatim** to `docs/archive/ROADMAP-done-001.md`; the
UI checklist's queue and verification debt folded into it (`docs/ui-checklist.md` keeps only the
per-feature gate); three closed working docs archived under the new, gitignored
`docs/archive/local/`; `docs/security.md` (threat model · what holds · eleven ranked findings each
with a roadmap row · the deterministic floor · a cpc gate proposal); `docs/release-ux-checklist.md`
(seven surfaces to drive in the installed build, with a record table whose "not driven" column is
mandatory) wired into `docs/RELEASE.md` §5b and the gates table; `docs/architecture.md` corrected
(the taxonomy layer is *built*, ~2,600 lines; the keyword arm is the on-disk FTS5 index, not the
retired in-RAM BM25; the legacy `tests/eval/run_eval.py` dropped from the layout).
Code and config: the lint/type/SAST gates now cover `apps/` and `scripts/` in CI and pre-commit
(they were `src/`-only — the HTTP boundary was gated by nothing; two fixes for mypy: the
`sse_starlette` import moved to the package root, and `routers/updates.py` names its `UpdateState`);
`[tool.bandit]` exists so `-c pyproject.toml` means something; `npm audit --audit-level=high` is a
frontend CI step and its one high (`nanoid` < 3.3.18) is fixed in the lock;
`tests/unit/test_desktop_security_config.py` pins the Tauri CSP (set, never unsafe, network only to
the sidecar) and the exact capability set; `docker-compose.yml` binds the port to loopback;
`scripts/conventions.toml` registers `gh run list --branch main --limit 1` in the session-start floor
and the security floor + walkthrough as sprint-close checklist lines; the dead `langchain.*` mypy
override is gone; `docs/features/` (a template nobody used, not laid by cpc) is removed. Local:
`.claude/CONTEXT.md` 316 → 245 lines (the July narrative and the resolved questions moved verbatim),
`.claude/KNOWN_ISSUES.md` 1,726 → 867 (18 closed bodies → archive 002, index rows auto-derived from
headings — refine when next touched; **KI-58 filed**: the CI-blind shape), `.claude/RIGOR_TODO.md`
history block archived and its "gate fails" line corrected (no gate exists).

**Why.** The planning surfaces disagreed (the ROADMAP said row 24 next, the UI checklist said P2;
87 of 108 rows were done and restated the DEVLOG); the phase framing had stopped describing the
work; security had never been written down as a threat model, so every finding would have been
"critical" or ignored; and the session-start protocol never read CI, which is how `main` stayed red
for seven pushes.

**Found on the way, and the one that matters:** `apps/desktop/package-lock.json` records the
project version twice and npm rewrites it only when npm runs — it sat at **0.4.2 through v0.5.0,
v0.5.1 and v0.6.0**. Same shape as the Cargo files: a version-carrying file nothing was looking at.
`release_preflight.collect_versions` reads both fields now, the pinned-file-list test carries them,
and `docs/RELEASE.md` §1 says nine places, not eight.

**Rejected.** Making `pip-audit` blocking today (66 advisories; it would redden every run until
row 61 triages them — the CI step stays advisory and the `sprint-close` checklist names it);
adding Dependabot and SHA-pinning the actions (bot PRs are the user's call); DOMPurify, the CSP
`form-action`/`base-uri`/`object-src` additions, `TrustedHostMiddleware` and the prompt fence
(each a code change that needs the installed build or an eval run to verify — rows 60/61, not a
review session); rotating the DEVLOG now (3,905 of 4,000 lines — due at the next close);
hand-writing the 18 KI index rows (verbatim archive first, judgment when each is next touched).

**What it opens.** Row 46 before the next release (the citation gate is a coin flip); a Windows CI
job (KI-58's other half — the shipped platform is the one no job runs on); the `pins.py` reachability
question; the 22 headerless specs; `docs/sprints/` is laid by cpc and unused since 07-18 — decide.
The review's §11 lists what it did not cover; rows 2 and 3 of `REVIEWS.md` stay `never`.

---

## 2026-09-07 (3) — CI on `main` was red for five days and seven pushes, including the release; two test-only fixes

**What changed.** `tests/unit/test_source_view.py`: the two "unknown document" tests get an
`empty_library` fixture — a temp SQLite with `Base.metadata.create_all` swapped into
`db.session._engine/_SessionLocal` via `monkeypatch`. `tests/integration/test_external_metadata.py::
test_matching_survives_a_differently_written_path`: the "differently written" variant is now
`parent / "sub" / ".." / name` (upper-cased on Windows) instead of a `/`→`\` swap. Both files,
+41/−4, test-only; shipped code untouched.

**Why.** `gh run list --branch main` shows every run since `3230703` (2026-09-02) red at `pytest
with coverage`, last green `5405d44` (2026-08-28) — through the merge, the tag and the docs commit
this morning — while the suite was 2,389/0 on the Windows dev box. Two causes. (1) The source-view
tests read the real `session_scope()`: locally `data/library.db` has the schema so the lookup
misses and passes; on CI `SQLITE_URL` names a file SQLAlchemy creates empty on first connect, so
`no such table: documents`. Reproduced here with `DOC_DATA_DIR=<empty dir>` on the old file (both
fail with exactly that error) and the fixed file (24/24). (2) `pathkey` is
`normcase(abspath(...))`. The old variant, on POSIX, produced a relative filename with literal
backslashes — a different file, so the catalogue entry never matched, red since the test was
written on 2026-08-31; on Windows `str(Path)` contains no `/`, so the swap never fired and the
test passed without testing anything. `abspath` collapses `..` lexically on both `posixpath` and
`ntpath` (checked), so the new variant exercises the normalisation everywhere.

**Rejected.** *An autouse DB-isolation fixture in `tests/conftest.py`* — the right end state, but
it touches every test that reaches the database, which is not a fix to make blind on the day the
suite is discovered to differ by platform. *`skipif(os.name != "nt")` on the path test* — hides
the platform difference instead of testing it. *Dropping the case half* — kept on Windows only,
because `normcase` folds case only there and a case-different path on ext4 *is* a different file.

**What it opens.** Nobody looked at Actions for five days while three baton entries said "every
gate green locally": the third instance of the CI-blind shape (uv.lock at 0.4.0 · the container
until 2026-09-04 · this) and worth a KI with a session-start check (`gh run list --branch main
--limit 1`). The two DB-isolating fixtures (`env` in the integration file, `empty_library` here)
want a shared one in `conftest.py`.

---

## 2026-09-07 (2) — v0.6.0 published; a README tell pass; §7b learns what the publish command actually does

**What changed.** The GitHub release for `v0.6.0` exists and is Latest: id 383972486, asset
`Provenote_0.6.0_x64-setup.exe` at 1,648,765,716 bytes with GitHub's own digest
`sha256:2280…5bc4` equal to the local hash, body = `.claude/release-notes-0.6.0.md` (11,099
chars, ends in the SHA-256 line). `releases/latest` now answers 0.6.0, so ADR-044's update check
on a 0.5.1 install has something to compare against. `README.md`: a pass for writing tells — the
contrastive slogans ("measured, not asserted", "estimates, not promises", "costs one paper, not
the library"), the aphorisms ("megabytes tell you almost nothing"), three "deliberately", one
"surprisingly", and two headline phrases added earlier today ("Your files, added your way", "the
answer can show you its source"); every number and the bold-lead structure stayed. `docs/RELEASE.md`
§7b: the command now carries the asset and `--verify-tag`, and the section says where the notes
live, why they need a status header, and that the command drafts, uploads, then publishes.

**Why.** Two `gh release create` runs raced: one from this session, one started a minute earlier
from the Run button on the command shown in the previous message. The earlier one drafted,
uploaded and published at 09:36:44Z; this session's uploaded a second copy to its own draft and
failed the final flip with HTTP 422 "Release.tag_name already exists", after which gh deleted its
draft — zero drafts remain, one release object, correct. The runbook did not say the command drafts
first, so a draft with zero assets read as a failure mid-way, and nothing said to check
`gh release list` before running it a second time.

**Rejected.** Killing the second process once both were seen — one had already published by then,
and gh's own cleanup handled the loser. Editing the entry below to say "published" — append-only;
this entry is the correction.

**What it opens.** `README.md`, `docs/DEVLOG.md`, `docs/RELEASE.md` are uncommitted. The GIF
(storyboards in the baton) and the README alt text that goes with it. `docs/DEMO.md` and the KI-48
heading, as noted below.

---

## 2026-09-07 — 0.6.0 release notes and README brought level with the tag; the release object itself is the user's command

**What changed.** `README.md` re-read against the tagged code, the way `docs/RELEASE.md` §2 asks
of the CHANGELOG — and two Limitations bullets were false. The OCR one said a pure scan "is
unreachable" and that OCR is "deliberately not built"; KI-47 measured the opposite (PyMuPDF uses any
`tesseract` on PATH, 0 → 34,600 characters on the same file), and the bullet now carries the
corrected 0.6.0 text. The reference-links one still described the surname+year matcher, "4 links
where 16 are stored", and named KI-45 as *next* — 0.6.0 requires title agreement (16 → 41 links,
DEVLOG 2026-08-26), so it now says that, plus the real remaining limit: links are computed at first
read and not revisited (`extract_citations --reresolve` refreshes them; not a button). The first
ingest is no longer "single-threaded" (two workers by default). **Added:** a Windows-installer route
at the top of Quick start — the README had no download pointer at all through two releases that
shipped an installer; the source-pane / *In context* bullet; the add-documents / Zotero /
per-part re-run bullet; the graph's coverage line; and a condition under the indexing-time table,
whose two OCR rows are only true with `tesseract` on PATH. Status → v0.6.0, 2,389 tests.

**The release notes** are drafted in the 0.5.1 body's shape at `.claude/release-notes-0.6.0.md`
(gitignored, like the 0.5.1 body was never committed). They carry the corrected OCR limit and say
in one sentence that the tagged `CHANGELOG.md` has it wrong; state the RG-012 verdict in two halves
(packaging strong; citation one sample — 1 failure in 4 byte-identical runs on 0.5.1, RIGOR_TODO
2026-08-14); and explain why a 0.5.1 library shows every document as *changed* after upgrading
(`is_cache_fresh` compares the recorded fingerprint against the per-format one 0.6.0 computes, so
every 0.5.1 cache reads stale — and ADR-047 means that re-read no longer orphans sidecars). The
upgrade path itself was not gated and the notes say so. Gates re-run at HEAD, which equals the tag
on `src apps scripts tests`: pytest **2,389/0** (9:33) · node:test 257/257 · mypy 98 files clean ·
svelte-check 219 files 0/0 · `docs_check --strict` OK · docs-encoding guard 5/5. Installer
SHA-256 `2280180840548023735044020e9a0e05ec418d06522909af66dbe4db9a5c5bc4`, 1,648,765,716 bytes.

**Why.** `docs/RELEASE.md` §7b: since ADR-044 the app reads `releases/latest`, so a pushed tag with
no release object is invisible to every install — 0.5.1 users cannot learn 0.6.0 exists until the
object is there, and `gh release list` shows only 0.5.1 and 0.4.2. The README drifted the same way
the CHANGELOG had: its Limitations were written for 0.5.0 and nobody re-read them at 0.6.0.

**Rejected.** *Re-tagging onto `b5ad5de`* to carry the corrected CHANGELOG — the tag is public and
peels to the source the installer was built from; moving it breaks the tested-equals-tagged diff §6
depends on. The correction lives in the release body, which is what people read. *Creating the
release from the agent session* — `gh release create` was refused by the permission classifier as
a publish, which is the right call: §7 makes publishing the one deliberate, irreversible step. The
exact command is in the baton. *Committing the notes under `docs/`* — the release page is the
body's home and the CHANGELOG is the durable record; a third copy rots.

**What it opens.** The release is one command away. The README GIF is a release stale (the 0.5.0
storyboard: no source pane, no add-documents) — three slideshow storyboards are in the baton, per
the user's ask. `docs/DEMO.md` still calls Connections *scored* (ranked since 0.5.1) and does not
mention the source pane — not touched. `KNOWN_ISSUES.md` still heads KI-48 as OPEN while DEVLOG
2026-08-25 (3) fixed it at the cause; the heading wants reconciling.

---

## 2026-09-04 (2) — CI builds the container, and checks the two things a green build does not prove

**What changed.** A third job in `.github/workflows/ci.yml`, alongside `ci` and `frontend`: free
~10 GB on the runner, build the image through buildx with the GitHub Actions cache, then assert
**(a)** torch is the `+cpu` wheel with zero `nvidia-*` distributions and **(b)** `apps.api` and
`doc_assistant` import inside the image.

**Why.** Nothing referenced the Dockerfile between `a052703` (2026-08-01) and today, so the
container was the one of this project's three shipping paths — desktop installer, source checkout,
headless image — that no gate touched. The pinned `ghcr.io/astral-sh/uv:0.12.1` base had never been
exercised on any machine (the dev box runs uv 0.11.14), and a `uv.lock` that had drifted would have
failed `uv sync --locked` in the image and surfaced only when somebody needed the container. Which
is exactly how it came up: the user asked whether Docker still worked, and the honest answer was
that nothing had checked since August.

**The two assertions are the job, not the build.** A green build says the layers assembled. It does
not say the image is the right one: `pip install ".[cpu]"` ignores `[tool.uv.sources]`, resolves
torch from PyPI, and that Linux wheel bundles CUDA — several GB of `nvidia-*` in an image with no
GPU, which still builds and still runs. KI-34 is the standing version of this lesson at the desktop
end: an artifact that started cleanly, served `/api/health`, reported a healthy chunk count, and
could not read a single PDF.

**Two bugs found in this job while writing it, both by running it rather than reading it.**

1. The nvidia count was `ls /app/.venv/lib/python*/site-packages | grep -c "^nvidia" || true`, which
   prints `0` — a **pass** — when the glob matches nothing at all. Relocating the venv would have
   turned the check off silently instead of failing it. It now asks `importlib.metadata` inside the
   image, and that the scan is not blind is itself checked: pointed at `torch`, it fails with
   `packages found: ['torch']`.
2. Rewriting it to a single line made the whole workflow **unparseable YAML** — a plain scalar
   cannot contain `": "`, and the f-string is `f"nvidia packages in the image: {nv}"`. It parsed
   before the edit and not after; only re-validating caught it. Both `run:` steps are block scalars
   now, with the reason recorded inline.

**Verified by extracting the commands from the parsed YAML and executing those exact strings**
against the built image, rather than retyping them: `torch 2.12.0+cpu | nvidia packages: 0` and
`apps.api ok; doc_assistant 0.6.0`.

**Rejected.** *Booting to a green `/api/health`* — first run downloads the embedder and reranker,
which is why the Dockerfile's `HEALTHCHECK` carries a 300 s start period; that belongs in the
release gate, not on every push. *A path-filtered trigger* — the Dockerfile's inputs are
`pyproject.toml`, `uv.lock`, `src/`, `apps/api/` and `scripts/`, which is most of the repo, so the
filter would have saved nothing and hidden the cases it did skip. *Plain `docker build` with no
buildx cache* — the dependency layer is ~8 minutes and the Dockerfile already orders its `COPY`s to
make it cacheable; not using that would have made the job the slowest thing in CI for no reason.

**What it opens.** The image is ~6.3 GB and the GHA cache is capped at 10 GB per repository, so the
`mode=max` export may thrash once other caches compete. If it does, the fix is `mode=min` or
dropping the cache export and paying the eight minutes. Left as-is because the first failure will
say so plainly, and guessing at it now would be tuning against an imagined problem.

---

## 2026-09-04 — The 0.6.0 known limits, checked line by line: one was inverted, one stale, and one I broke

**What changed.** Three bullets under `## [0.6.0] → Known limits` in `CHANGELOG.md` (`6da294b`).
The 0.5.1 section was deliberately left alone — a released section records what was true then, so
the exception belongs in the 0.6.0 entry, which is where somebody deciding whether to install 0.6.0
reads.

**Why.** `docs/RELEASE.md` §2 makes this a judgment step and names the failure it guards: the 0.4.1
draft claimed a clean-machine install was unverified for three days after it had been verified — "a
limit that silently becomes a lie". Checking all eight bullets against the code rather than
re-reading them found six sound and two not — and produced a third error of my own, recorded below
because it is the sharpest instance of this session's recurring failure.

- **The OCR limit is TRUE, and I broke it before restoring it — the most useful thing in this
  entry.** I read it as fiction and rewrote it to say the app has no OCR. Four "confirmations"
  agreed: no OCR package in `pyproject.toml` or `uv.lock`; `tesseract` never in `src/` in the whole
  history (`git log -S`); `pymupdf4llm.to_markdown` takes no `ocr` argument and the library's own
  `check_ocr` import is commented out; ADR-039's Context says a scan "extracts to nothing". Every
  one of those is about **code this repo can read**, and the mechanism is not in this repo's code:
  PyMuPDF discovers a `tesseract` binary on PATH by itself. `KI-47` had the measurement all along —
  the same scan yielding **0 characters on 2026-08-08 and 34,600 on 2026-08-19**, nothing in the
  repo changed — and `C:\Program Files\Tesseract-OCR\tesseract.EXE` v5.4.0 is on this box's PATH
  right now. ADR-039's Context is not counter-evidence either: it was written 2026-08-01, eighteen
  days before the behaviour was found. The corrected bullet now carries the measurement, so it is
  stronger than what it replaced.
- **The moved-file limit was inverted.** It led with "is treated as a new document", while its own
  next sentence said the content is recognised. `_existing_document_id` (`ingest/store.py:67`)
  matches on `doc_hash` **first** and falls back to path, so a moved file keeps its id, its figures
  and its corrections. Only path-keyed state is lost — exclusions live on registry rows keyed by
  `pathkey` — which is what the bullet now says.
- **The inherited update-check limit was resolved and nobody noticed.** 0.6.0 said "everything
  under 0.5.1 still applies"; 0.5.1 said the update check cannot compare until releases are cut.
  `gh release list` shows Provenote 0.5.1 (Latest) and 0.4.2 (Pre-release) published.

**Re-measured rather than assumed**, for the two limits carrying numbers: KI-57 is still open at
0.2% of pages corpus-wide, and the keyword figure is **97.0%** — 1,358 of 1,400 attached keywords
on exactly one document, across 98 documents — so 0.5.1's "97%" is still exactly right at a corpus
one document larger.

**Rejected.** *Editing the 0.5.1 section* — Keep a Changelog treats a released section as a record,
and rewriting it would falsify history to fix a forward reference. *Deleting the OCR bullet* — the
underlying limit is real and load-bearing (a scanned PDF is unreadable and marked broken); only the
mechanism was invented.

**What it opens.** A blanket "everything from the previous release still applies" inherits claims
without naming them, which is how the update-check line was re-shipped unread; naming each
carried-over limit would cost a few lines and make each one checkable.

The larger opening is the OCR mistake. **`.claude/KNOWN_ISSUES.md` was never consulted** during a
review whose entire job was "is this still true", and KI-47 answered the question directly, with a
measurement. Reading the code proves what the code does; it does not prove what the *running system*
does, and the gap between those is exactly where an undeclared external dependency lives. The
verification order should be: what did we already measure, then what does the code say — not the
reverse. This is the same fail-open shape as the four gates fixed on 2026-09-02, committed in a
release note by the person who had just spent two days cataloguing it.

---

## 2026-09-02 (3) — RG-012 passed on 0.6.0, and the preflight could not see it

**What changed.** One character class in `scripts/release_preflight.py`: `_CHOSEN` now reads
`([\d,]+(?:\.\d+)?)` where it read `([\d,]+)`. Five parametrised tests pin the parse.

**Why.** RG-012 Tier-2 passed on the 0.6.0 installer at 17:49 — clean Windows Sandbox, `python on
PATH? False`, 181 s silent install, health at 210 s, 3 PDFs to 322 chunks, a 14 s turn with 10
sources and 4 resolved citations, 0 unresolvable. `release_preflight` then reported **"no RG-012
run matches this installer — the gate ran against a DIFFERENT build"**, with the archive count
correctly up from 9 to 10. It had found the run and rejected it.

The harness logs its size as `[math]::Round($bytes/1MB, 1)`. Every installer before this one
happened to land on a whole number — 0.5.1 is 1572.0318 MiB, which renders as `1572` — so a pattern
accepting digits and commas matched for four releases running. 0.6.0 is 1572.3855 MiB, renders
`1572.4`, and the line stopped matching. The check had never been right; it had been lucky, with
roughly a one-in-ten chance of exposure per build.

**The failure direction is what makes this worth a log entry.** A parser that drops the record it
is looking for reports *absence of evidence* — and this check's absence message is an accusation
("the gate ran against a DIFFERENT build"). The rational response to it is to re-run a 20-minute
clean-machine gate that has already passed, or to override the check by hand. Both are worse than
the check not existing. This is the third instance today of one shape: `versions` never opened two
files, `artifact_fresh` compared the wrong quantity, `rg012` could not parse its own harness. In
each case the check *ran*, and what it silently failed to see was the thing it was for.

**Rejected.** *Loosening to `([^)]+)`* — it would parse, but the size is the one field that makes
the log line self-describing, and a pattern that accepts anything stops being a guard. *Making the
harness print an integer* — the harness is the record; changing what it writes to suit a reader is
backwards, and the archived logs would still not parse.

**On what the PASS is worth.** The packaging half is strong: it is the half that found KI-34, and
nothing else exercises the frozen artifact end to end. The citation half is one sample of a
measurement `.claude/RIGOR_TODO.md` reopened on 2026-08-14 as unreliable — a coin flip on
`llama3.1:8b`, where 0.5.1 failed once and passed twice on the same installer. Recorded in
`docs/desktop-packaging.md` §5 so the next reader does not take "4 resolved citations" for a
stability claim.

**Host state.** The run needs Ollama reachable from the sandbox. Rather than the documented
persistent `OLLAMA_HOST` user variable, this run set it **process-scoped** on a directly launched
`ollama serve`, so there was nothing persistent to revert — verified afterwards: both env scopes
empty, listener back to `127.0.0.1` only, gateway address refused, 9 models still served locally.
Worth preferring next time: the documented procedure leaves a variable that has to be remembered.

---

## 2026-09-02 (2) — `artifact_fresh` judges git history, not file mtimes

**What changed.** `check_artifact_fresh` no longer asks "is any tracked source file's mtime newer
than the artifact?". It asks `_newest_shipped_change()`, which splits the question in two:

- **committed** files are dated by the **committer date of the newest commit touching a shipped
  path** — never by the file on disk;
- **uncommitted** files are dated by mtime, which is the one place an mtime means what it looks
  like it means: a person edited the file, and it is not in history yet to be dated any other way.

`SOURCE_GLOBS` (dead — defined, never referenced) and `_newest_source` are gone. In their place
`SHIPPED_PATHS` names the fifteen paths the artifact is actually built from, and `_is_shipped`
matches them exactly: a `/`-terminated entry by prefix, a file entry only against itself.

**Why.** The old comparison demanded a rebuild after a plain `git checkout main`, which
re-materialises files with today's date and byte-identical content — `src/doc_assistant/__init__.py`
was blob `a789456…` at both the built commit and HEAD, and the preflight called it a source edit
(2026-09-02). A gate that cries wolf on a branch switch is a gate that gets overridden by hand,
which is how it stops working. Content makes an artifact stale; a checkout is not an edit.

The old path list was also short: it covered `src/`, `apps/api/` and `apps/desktop/src/` and
**not** the Rust shell, `tauri.conf.json`, the PyInstaller spec or `build_sidecar.py` — so an edit
to the spec, which is exactly where KI-34 lived, could not have marked the artifact stale. Same
failure as the version check's two missing Cargo files, so it gets the same guard: the list is
pinned by `test_the_shipped_path_list_is_pinned`, which also asserts every entry exists on disk.

**`Cargo.lock` is deliberately excluded, and it is a judgment call.** Cargo rewrites the lock
*while building*, so a release build necessarily ends with a lock newer than the artifact it just
produced; counting it would fail this check on every release, which is what happened at 0.6.0. Its
one release-relevant field (the crate version) is covered by `versions` instead. **Residual gap,
stated rather than hidden:** a dependency version changed in the lock without a rebuild is not
caught here. `uv.lock` stays *in* the list — nothing in the build rewrites it, so the asymmetry has
a reason.

**A bug found by probing rather than reasoning.** The first implementation read `git status
--porcelain` and sliced `line[3:]` for the path. `_run` ends in `.stdout.strip()`, which eats the
leading space of an unstaged ` M path`, shifting every offset by one: the path came out as
`rc/doc_assistant/__init__.py`, failed `is_file()`, and the entire uncommitted branch was a silent
no-op. One test caught it; two wrong guesses at the cause (a fatal pathspec, then a timestamp tie)
were both disproved by running the thing in isolation. It now splits on whitespace and never
depends on a column.

**Six behavioural tests**, in a throwaway git repo with `ROOT` monkeypatched: a docs commit does
not move the bar; an mtime-only bump does not (the regression test, which asserts its own premise —
that the bumped file *is* the newest thing on disk — so it cannot quietly stop testing anything);
an uncommitted shipped edit does; a committed shipped edit does; a `Cargo.lock` commit does not.

**Rejected.** *Recording the built commit in a stamp file at build time* — the correct answer in
the abstract, and still the better one if this ever needs to be exact. It was rejected here because
existing artifacts carry no stamp, so the check would have to degrade to "cannot tell" for the very
release that motivated the fix, and back-filling a stamp by hand is the "a PASS from a previous
build reads as evidence" hazard this file already warns about. *Keeping mtimes and special-casing
the checkout* — there is no way to tell a checkout from an edit by mtime, which is the whole point.

**What it opens.** `artifact_fresh` now trusts commit dates, so a rebased or amended history with
rewritten committer dates could in principle move the bar backwards. Committer dates are set at
commit time and a rebase rewrites them to "now", so this is monotonic in practice on one machine;
it would need revisiting if releases were ever cut from a rewritten branch.

---

## 2026-09-02 — The version check now reads the two Cargo files, and its file list is a test

**What changed.** `scripts/release_preflight.py`'s `versions` check went from five sources to
**seven**: `apps/desktop/src-tauri/Cargo.toml` (`[package] version`) and `Cargo.lock` (the
`doc-assistant-desktop` entry, found by name in the package list) now join the five it already
read. `collect_versions()` is split out of `check_versions()` so the *list of files* is importable
and therefore testable, and `docs/RELEASE.md` §1 grew from six rows to eight.

Three tests, in `tests/unit/test_release_preflight.py`:

- **the source list, pinned by equality** — adding a version-carrying file means adding it here;
- **no source may read as a sentinel** — `(not found)` and `(missing)` compare equal to each
  other, so seven simultaneously-broken readers would have "agreed";
- **a drift in any single file must FAIL**, parametrised over the source list rather than
  spot-checked, so a file added to the list gets its negative case for free.

**Why.** The `versions` check reported green while `Cargo.toml` and `Cargo.lock` held `0.4.1`
through **v0.4.2, v0.5.0 and v0.5.1** — three tagged releases (verified by reading each tag:
`git show vX.Y.Z:apps/desktop/src-tauri/Cargo.toml`). It never opened them, and neither did the
runbook table. This is the *inverse* of the `uv.lock` incident that created the check: not a file
someone forgot to edit, but a file nothing was looking at. An agreement check is worth exactly as
much as its file list, and until now that list existed only inside a function body.

Surfaced at 0.6.0 the hard way: the release build regenerated `Cargo.lock` from 0.4.1 to 0.6.0
*after* the release commit, and `tree_clean` — not `versions` — was what caught it.

**Verified by reverting the fix.** With the two Cargo sources removed from `collect_versions()`,
exactly three tests fail (the list test and both Cargo drift cases) and the other 14 pass. The
guard reproduces the historical bug rather than merely describing it.

**The re-lock command is verified too.** `docs/RELEASE.md` §1 now carries
`cargo update --manifest-path apps/desktop/src-tauri/Cargo.toml -p doc-assistant-desktop --offline`
— run against a deliberately desynced tree, exit 0, one line changed, no network. `cargo metadata`
was tried first and rejected on evidence: it wants metadata for every locked package including
Android-only ones this box has never downloaded, so it exits **101** under `--offline` (after
writing the lock) and needs the network without it; `--no-deps` exits 0 and updates nothing.

**Rejected.** *Deriving the file list from the runbook table* — a docs parser is a second thing to
break, and the table is prose. *Hand-editing `Cargo.lock`* — it is a lock; cargo overwrites it at
build time anyway, which is precisely the failure being fixed. *Extending the existing
"do they agree?" test* — it structurally cannot catch a missing source, which was the bug.

**What it opens.** `artifact_fresh` has the same shape of weakness one layer over: it compares
**mtimes**, so `git checkout main` re-materialising a byte-identical file (blob `a789456…` at both
`ef4a6d8` and `663c290`) fails it. Comparing `git diff <built-commit> HEAD` instead would say what
the check means. Not done here — it needs the built commit recorded next to the artifact, which is
a change to the build, not to the check.

---

## 2026-09-01 (6) — Two UI corrections from using the app: controls too small to find, and a dropdown painted by the OS

**What changed.** Both reported after driving the merged build in the native Tauri window.

1. **The source pane's header controls were too small to find.** 0.15rem of padding under a 0.7rem
   label, drawn in the muted `--fg-2`, inside a border that reads as part of the pane frame. Three
   changes, no redesign: a real hit target (**26px** row, steppers squared to 26x26 from 23px wide,
   close 29x26), the resting colour moved off `--fg-2` onto **`--fg`**, and hover that fills the
   button rather than only tinting the glyph. The active fit preset now takes the accent as a
   *fill* (`--accent` / `--accent-fg`) instead of tinting its text, so which mode is on is legible
   at a glance. Focus rings added on all three groups.
2. **The chat folder-scope dropdown did not match the app.** Its options and popup were painted by
   the user agent in the **OS** scheme — a light menu over a dark app.

**The second one was not a colour bug in that component.** `.scopepick select` sets
`background: none`, so the closed control was already correct; what was wrong is that **the app
never declared `color-scheme`**. Without it the UA paints every native widget in the system scheme
regardless of the page's palette. So the fix is one declaration per theme state in `app.css`
(`:root`, `[data-theme='dark']`, `[data-theme='light']`, and the `prefers-color-scheme` block) —
next to the palettes, not on the one control, because the same mismatch was in **all five**
`<select>`s and every native scrollbar fallback.

**Verified live** in the running app, both themes: `color-scheme` resolves `dark` / `dark` / `light`
across system-default, forced-dark and forced-light, and the scope `<select>` inherits it in all
three. Control contrast checked in both — active preset indigo-on-white in light, and the close
button now `#ece5d6` on dark where it was the muted `#a79e8b`.

**Rejected.** *Styling `option` backgrounds directly* — works in Chromium, does nothing for the
popup chrome or the scrollbars, and would need repeating in five places. *Hardcoding
`color-scheme: dark` on the control* — correct in one theme and wrong in the other.

**Gates.** node:test 257/257 · svelte-check 219/0. CSS-only plus one icon size; no logic touched.
**$0 — no model call.**

## 2026-09-01 (5) — Row 18 closed out: a citation now opens its page, and the two branches no test could reach were driven for real

**What changed.** Two gaps, both named in the 2026-09-01 (1) baton as unfinished.

1. **A chat citation can open its page.** The source card gains **Show the page**, which navigates
   to the Library and opens the pane where the passage is. `GET /api/library/chunk-page` now
   returns `{document_id, page}` rather than a bare page: a chat citation carries a `chunk_key` and
   **no document id**, and turning one into the other means reading the chunk store — so it happens
   on the server rather than by parsing the key's shape in the client, where the second copy of
   that contract would rot. New `library.locate_chunk` + `ChunkLocation`; `page_for_chunk` is now a
   thin wrapper on it.
   **It is offered for a figure too**, which the card's own comment had anticipated: a figure has
   no position in the text, so the page image is the only place it can be shown.
2. **The unavailable and text-only arms were driven**, against the live library, with a backup taken
   first — a file moved out from under a document, then a document's `format` flipped to `epub`.
   Both restored; the library matches `data/library.db.bak-20260901-183252-prearms` row for row.

**And that is where the two real defects were, neither of which a test could have caught** — the
corpus is 98/98 PDF with every file present, so no test fixture stands in for driving it:

- **The size and zoom controls rendered over a document that has no page.** "Fit page | Width |
  − 100% +" sat above the sentence *"The file is not where the library expects it"*. A dead
  control, and this project's own rule is that a dead control is worse than none. Now gated on a
  renderable page, with the close button taking the right-hand margin when they are absent.
- **Two pieces of copy that were wrong.** The backend said *"a epub document has no pages"* — an
  article cannot agree with a value read from the database, so it is now *"a document in EPUB
  format"*. And the pane said the extracted text was **below** while its own hint said **beside**:
  Chunks is beside the pane in the split layout and above it when stacked, so any direction is
  wrong half the time. Both directions dropped.

**A false defect, avoided by the project's own rule.** After the jump, the citation panel appeared
to stay open over the Library — the DOM still held it 2 s later and a screenshot showed it. It had
in fact closed; the node was mid-transition, exactly the stranding the baton warns about
(2026-08-31 (3)). **When the DOM and the state disagree, believe the state**: querying again showed
the card gone. Nothing was "fixed".

**A test that had to be rewritten, for a reason worth keeping.** The first version of the route
test monkeypatched `source_view.locate_chunk` and failed — the route resolves the name through the
**package** re-export (`from doc_assistant.library import locate_chunk`), which is a separate
binding. That is the trap in `src/doc_assistant/CLAUDE.md`, met from the other side. Rather than
patch the re-export, the fake chunk store now holds real rows, so the tests exercise the real
derivation — cache markers on disk, offset in the metadata, page derived.

**Verified live.** A mocked `/api/chat` turn (fabricated sources, a **real** `chunk_key`; no model
call, no cost — the discipline from the 2026-08 chat-UI note) → click the citation → **Show the
page** → the Library opens `03-Zuo2014_NBR_fconn.pdf` at **"Page 10 of 19 · cited here"**, the page
computed independently beforehand. Unavailable and text-only arms both render a sentence, no image,
no broken image, no page nav, no size controls.

**Gates.** pytest source-viewer suites **44/44** (+7) · node:test 257/257 · svelte-check 219/0 ·
mypy 98/0 · ruff + format clean · bandit 0 · `detect-secrets` clean against the baseline.
**$0 — no model call.**

## 2026-09-01 (4) — The page fits because the reader decides how: a real zoom, a draggable split, and renders that get sharper instead of bigger

**What changed.** The source pane stops being a fixed picture in a fixed box.

1. **Zoom** — a `− 76% +` stepper beside the fit presets, **Ctrl/Cmd + wheel** (a bare wheel still
   scrolls), and the reading itself is a button that returns to the chosen fit.
2. **A draggable split** — a `separator` between the document and the pane, dragged with pointer
   events (so trackpad, pen and touch all work), moved with ← → (Shift for a coarse step, Home to
   centre), double-clicked to reset. Persisted, clamped to 25-75% so neither side can be dragged away.
3. **Sharper renders, not magnification** — `GET …/page/{n}` takes a `dpi`, and the pane climbs a
   ladder (110 → 150 → 200 → 260 → 330 → 400) as the page is drawn larger.

**Why (3) is the part that matters.** Zoom on a fixed image is just blur. Asking the server to
draw the page again at the resolution it is being displayed at is what makes zoom mean anything —
verified live: at 354% the pane requested **260 dpi** and got a **1831px** render in place of the
775px one, and stepping back out returned to 200 then 110.

**Three decisions worth keeping.**

- **Zoom is a multiple of the pane's width**, not of "actual size" — a page that was never on
  paper here has no actual size, and a percentage of one would shift under the reader every time
  they dragged the split. `Width` is therefore always 100%, and `Fit page` is whatever fits.
- **The dpi ladder is quantised.** Requesting exactly what each zoom level needs would issue a
  render per frame of a drag. Snapping **up** to a rung keeps it to a handful of fetches and never
  asks for an image blurrier than the one it replaces. The ceiling is enforced server-side
  (`clamp_dpi`, 72-400): render cost grows with the square of dpi, so an unbounded query parameter
  is a work generator. Out-of-range is **clamped, not refused** — a zoom level is not a validation
  error.
- **`dpi` is clamped in the library, not the route.** One expression of the bound, called by the
  route, so no caller can reach the renderer around it.

**Two things the work corrected in itself.**

- **A test disproved a claim in my own comment.** `renderDpi` said a 2x display makes the default
  soft at rest. It does not: at the pane's real width (433 CSS px, 612pt page) the render needs 51
  dpi at 1x and 102 at 2x, both under the 110 served. Device pixel ratio starts to bite once
  *zoomed* (153 dpi at 1.5x on a 2x display) or on a pane dragged wide (212 dpi at 900px). Comment
  corrected; the number is now pinned by a test, because nothing else would have caught it.
- **The fit was inferred and raced.** The first version decided "has the reader zoomed?" by
  comparing `zoom` against the computed fit — which is 1 before the box is measured, and 1 is also
  a legitimate zoom, so the pane opened at 100% instead of fitted. Replaced by an explicit
  `userZoomed` flag with the fit *derived*; a flag cannot race.

**Also fixed while verifying.** The fit was measured against the body's **border** box, so a
"fitted" page still needed 16px of scroll — `ResizeObserver`'s `contentRect` and the image's own
2px border now give a true fit (measured: `scrollY: 0`). And `setPointerCapture` is wrapped: it
throws for a pointer the browser no longer considers active, and the exception would abort
`onSplitDown` *before* its listeners attach — a handle that looks grabbed and does nothing.

**Verified live** on `cajal-lecture.pdf`: fitted at 72% with **0px scroll on both axes**; +/- and
Ctrl+wheel step through 38% → 188% → 354% with the dpi ladder following; a plain wheel scrolls
without zooming; a real mouse drag moved the pane 667 → 433px, persisted **0.438**, and the zoom
re-fitted 49% → 76% on its own. Both themes.

**Rejected.** *A zoom slider* — a stepper plus Ctrl+wheel covers coarse and fine without a control
that is hard to hit at the pane's size. *Re-rendering at exactly the needed dpi* — see the ladder.
*Refusing an out-of-range dpi with a 4xx* — the honest answer to "sharper than we draw" is the
sharpest we draw.

**Gates.** pytest source-viewer suites 37/37 (3 new dpi tests) · node:test **257/257** (+20) ·
svelte-check **219/0** · mypy 98/0 · ruff + format clean · bandit 0. **$0 — no model call.**

## 2026-09-01 (3) — No page in the corpus actually fit the source pane, including the ordinary ones

**What changed.** The source pane gains a **Fit page / Width** toggle in its header, defaulting to
**Fit page**, persisted client-side (`libPrefs.sourceFit`, localStorage, the same class as the
theme toggle and the grid/list switch — never a backend setting).

**Why.** The pane sized a page to the pane's *width* and let height scroll. Measured on the running
app at 1280x720, that means **no page fits**, at any shape in the corpus:

| Page aspect (h/w) | Example | Visible at fit-width | Width if fitted |
|---|---|---:|---:|
| 1.29 (US Letter, **57 of 98 docs**) | most of the corpus | **94%** | 405px of 433 |
| 1.41 (A4, 19 docs) | European journals | 86% | 371px |
| 1.57 | `cajal-lecture.pdf` | 77% | 333px |
| 1.79 | `middleton-2001.pdf` | **67%** | 292px |

Confirmed live rather than computed: the Cajal page rendered 433x679 into a 519px body — 178px of
scroll to see the bottom of a page. Even the most common size in the library needed a scroll to
show its last inch.

**The default is the argument, not the toggle.** ADR-050 D1 already settled what this pane is for:
the image carries *fidelity and provenance* — "this is the page it came from" — while row 19's
extracted text is the reading and searching surface. A view whose job is **where** should show the
whole page; a view that cannot show the whole page cannot answer that question. Fitting costs
little on the common case — 405px against 433px, a 6% loss of width for the last 6% of the page —
and the reader who does want to read has one click to Width, remembered thereafter.

It also removes a prerequisite from ROADMAP 24: a highlight band low on a page is worth nothing if
the pane opens showing the top two-thirds. Fit page means the band is on screen the moment it exists.

**Verified live** on `cajal-lecture.pdf` (the 1.57 case): default **100% visible, 0px scroll**;
toggled to Width **76%, 178px**; toggled back, 100%; the choice persisted. Both themes, and the
stacked (<900px) layout, where the pane's own `max-height: 60vh` becomes the binding constraint and
the page still fits inside it.

**Rejected.** *Reclaiming chrome* — the pane spends 51px of 574 on its header and footer; even
deleting both would not fit the 1.79 page. *Widening the pane* — a wider page is a **taller** one,
so it makes fitting strictly worse. *Fit page with no escape* — at 41% scale the body text is not
readable, and pretending otherwise would push people back to the OS viewer.

**What it opens.** In the stacked layout the height cap binds while horizontal room goes unused, so
the fitted page sits small between wide margins; raising the cap trades that against how far the
reader must scroll past the pane. Left alone deliberately — it fits, which was the requirement.

**Gates.** node:test 237/237 · svelte-check 219/0 · ruff clean · mypy 98/0. Frontend-only; the
Python suite is unmoved from 2026-09-01 (1). **$0 — no model call.**

## 2026-09-01 (2) — ADR-050 D5 measured: the on-image highlight is viable, and twice the measurement lied before it told the truth

**What changed.** No code. ADR-050 gains a dated **Addendum** answering the question D5 left open —
*can a cited passage be located as rectangles on its page image, and how accurately?* — and
ROADMAP row 24 files the follow-on with what it actually costs. Read-only, $0, on the live corpus.

**Why now.** D5 scoped the highlight out and called its accuracy "unmeasured", naming that the
follow-on's first question. Answering it before anyone commits to building is cheaper than
answering it afterwards, and the answer changes what the follow-on is.

**What it found.**

*Recall inverts with anchor length.* A **3-word** anchor places **91%** of single-page prose
sentences; a **12-word** one places **69%** (730 sentences). Longer anchors cross line breaks,
where hyphenation and the extractor's reflow stop matching. At 4 words: 90% placed, and only **5%**
genuinely ambiguous.

*The design the numbers point to is an envelope, not per-sentence rects.* Highlighting sentences
individually leaves ~10% unlit and scattered through the passage, and a reader cannot read a gap as
anything but "this part was not the evidence". A parent chunk is contiguous text, so highlighting
the band between the first and last unambiguous anchor gives **97% median purity** (highlighted
words that really are the passage; >= 90% on 88% of passages). Coverage measured 45% median, but
that is a **floor, not a verdict** — the probe grouped anchors into columns by `int(x0 // 60)`,
which splits an indented paragraph across two bands.

**Two measurement traps, both of which produced a confident wrong answer first, and both worth
keeping.** (1) The first run scored **68%** — because its needles still carried the cache's list
markers and table pipes. It was measuring the probe. Cleaned, the same method scores **94%**.
(2) "More than one rect" was then read as ambiguity, which made *longer* anchors look *less*
precise — an inversion that should have been the tell. `search_for` returns one rect **per line a
match spans**, so a wrapped phrase is indistinguishable from a repeated one until you separate them
geometrically — and the rects of a wrapped phrase are horizontally **disjoint** (tail of one line,
head of the next), so the natural test, "do they overlap in x?", misclassifies every one of them.
Only a vertical test works. Ambiguity fell from a fictional 22-32% to a real **5-7%**.

**Why it was not built this session.** The row implies a detail; the measurement says increment.
Three things have to be solved that nothing had named: real column detection (the probe's proxy is
not shippable), **43% of parent chunks straddle a page break** so the opening page can only ever
show part of the passage and the pane must say so, and a stated policy for the 5% ambiguous anchors
(decline, never guess). Filed as ROADMAP 24 with those three named, rather than started and left
half-done.

**Rejected.** *Building it on the 94% figure* — that number is single-page prose with tables
excluded, and quoting it for the feature as a whole would be the same error the first probe made,
one level up. *Per-sentence highlighting* — higher coverage, but its gaps make a false claim about
what the evidence was. *Treating the 45% coverage as the answer* — it is an artifact of the probe's
column proxy, and shipping a "known 45%" would bake in a limit that was never measured.

**What it opens.** ROADMAP 24. Also a question worth asking before that is built: with 43% of
parents crossing a page break, the highlight's honest unit may be *the passage across two pages*
rather than one page's band — which is a pane-layout decision, not a locating one.

**Gates.** Docs-only: `docs_check --strict` 0/0 · doc guards 9/9. No code changed, so the code
gates are unmoved from 2026-09-01 (1). **$0 — no model call.**

## 2026-09-01 (1) — ROADMAP 18: the document beside its library entry — and the row's stated reason for it being free was wrong

**What changed.** A source pane on the Library document view (`SourceViewer.svelte`, opened from a
new **Source** button beside Re-run), rendering the file itself one page at a time. Backend:
`library/source_view.py` + three routes — `GET /api/library/documents/{id}/source` (can this be
shown, and why not), `.../page/{n}` (PNG, rendered on demand), and `GET /api/library/chunk-page`
(which page a cited chunk sits on). Behind **ADR-050**, which row 18 did not have.

Each open parent block in **Chunks** now carries *"Show this page in the document"*, which resolves
that block's chunk key — the same `{document_id}:p{parent_index}` a chat citation carries — and
opens the pane there. ROADMAP 19 shows a passage in the extracted *text*; this shows the page of the
original it came off.

**Why.** Row 18 asked for it in 2026-08-25, and row 19 shipped the text half already noting the page
image was 18's job.

**The measurement that changed the design.** The row asserted *"page-level jump costs no ingest
change — chunks already carry `page`"*. It does not hold for the path the app retrieves on:
`USE_PARENT_CHILD` defaults true, and the parent-child store carries `page` on **615 of 39,705
chunks (1.5%)** — all of them figure chunks, whose page comes from figure detection. The flat
baseline store is 100%, and it is not the retrieval path. Building on the row as written would have
produced a feature that worked on figures and nothing else.

The conclusion survives for a different reason: the **cache** is page-annotated (`<!-- page:N -->`,
`extractors.py:99`) on **98/98** documents, with marker count equal to `Document.page_count`
exactly and sequential from 1, and chunks carry `parent_char_start` at 100% after row 19's re-chunk.
So the page is a read-time scan of markers against an offset — the rule `chunking.extract_chunk_metadata`
already applies at ingest for the flat store. `ChunkContext.page` therefore goes from **2.0% to
98.0%** populated on the live path (measured over 300 sampled parents), which also fills in a field
row 19's payload documented as permanently sparse. The remaining 2% are figure chunks, which have no
text span to place — and `page_for_chunk` still gives them a page from the stored value, so a figure
citation opens correctly where the *text* view honestly cannot show anything.

**Cost, measured before choosing.** A page render is 19-31 ms and 140-261 KB (median over 18 pages of
the 6 longest documents; 110 dpi ships). Nothing is pre-rendered or cached: the whole corpus is 2,973
pages, or ~760 MB and ~90 s to render up front, to save 19 ms.

**What driving it found — KI-57, and it is not this feature's bug.** Block 400 of `hebb_1949`
resolves to page 202, but its text is visibly on page 201. The cause is upstream: markers 201 and 202
delimit **byte-identical** segments — the cache holds page 201 twice and page 202 not at all.
Measured: **13 of 355 pages (3.7%) in `hebb_1949`, all 13 exact duplicates**, against **1 of 657
(0.2%)** across a 25-document sample. The marker *rule* is sound (342/355 and 656/657 segments match
their own page); what is occasionally wrong is the text placed under a marker. Filed rather than
fixed — the fix is an extraction change that re-invalidates every cache, and this is 0.2% of pages.
The suspicion that `_recover_lost_page` causes it is **wrong**: the other two recovery documents are
clean, 0 of 61.

**Rejected.** *PDF.js in the frontend* — better on selectable text and in-page find, but puts
document parsing in the thin shell, adds a worker and a Tauri CSP fight, and ships whole files to
show one page; the searchable surface already exists as the extracted text. *Tauri asset protocol* —
bypasses the ADR-002 boundary and dies in browser dev mode. *Backfilling `page` onto the
parent-child store* — a 39,705-chunk re-chunk to persist something derivable for free and
invalidated by the next extraction change. *Converting non-PDFs to PDF to give them pages* — invents
pages a document never had; they degrade to their extracted text instead, which is what they are.

**What it opens.** The passage highlight *on the page image* (ADR-050 D5, scoped out): offsets are
not coordinates, so it needs `page.search_for`, whose accuracy against normalised extraction is
**unmeasured** — that measurement is the follow-on's first question. Also: the pane is most of the
substrate an annotation layer would need, and nothing about it is speculative yet. And KI-57 has a
cheap exact detector if anyone picks it up — a page segment byte-identical to its predecessor found
13 of 13 with no false positives.

**Gates.** pytest **2349/0** (2315 + 34) · mypy 98/0 · ruff + format clean · bandit 0 ·
svelte-check **219/0** · node:test **237/237** (216 + 21) · doc guards 9/9 · `docs_check --strict`
0/0 · `test_api_check` 0/0 (240 files). Driven live on the real 98-document library in both themes
and at 820px. **$0 — no model call.**

## 2026-08-31 (4) — The graph now says how much of the library it covers, and why the obvious version of that number would have lied

**What changed.** `GraphStaleness` gains `n_documents_in_library`, and the Graph workspace states
**"Covers 30 of your 98 documents — a document appears once it mentions one of the 13 concepts on
your graph."** One field, one pure helper (`graph.graphCoverage`), 5 node:tests, 1 pytest case. No
extra query: the live document set was already being read for `missing_document_ids`.

**Entry (3) closed with the wrong open item, and checking it is what corrected the design.** It
said *"nothing watches the inverse — documents the corpus has that the graph has never seen … a
count of it would tell a user whether a rebuild is worth 10 seconds."* Measured before building it:
the library holds **98** documents, the graph cites **30**, and the other **68** are not waiting for
anything — they mention none of the **13** concepts in the graph vocabulary (of **593** curated). A
rebuild would return the same 30. So "68 documents not yet in the graph" would have been a number
that reads as a backlog, dressed a no-op button as the fix, and sent the user away from the lever
that actually moves it: **curating vocabulary** (ADR-018, ROADMAP 23).

**So the number is coverage, and it ships with the rule that produces it.** A fraction plus the
sentence explaining the fraction, in plain text rather than a warning — partial coverage is how the
feature works, not a fault. The test that matters asserts the *absence* of the misleading framing:
the string must not contain "missing", "not yet", "rebuild" or "pending".

**Rejected: a `built_at` timestamp in the skeleton.** The honest form of "documents added since the
build" needs one, and `_graph_version` is documented as a **timestamp-free** fingerprint precisely
so identical inputs produce a byte-identical `skeleton.json` (Decision 3). Stamping the artifact
would trade a verified determinism property for a number that coverage already answers well enough.

**Rejected: folding coverage into the staleness banner.** `stale` means *the graph is wrong* —
vocabulary drift or a reference it cannot resolve. Coverage is neither, and putting it behind a
warning icon would teach the user to dismiss the icon.

**What it opens.** The 68 uncited documents are a **vocabulary** signal, not a graph one: 13 of 593
curated concepts are on the graph, and that ratio — not a rebuild — is what decides coverage. The
Manage-keywords view is where that would be worth surfacing.

---

## 2026-08-31 (3) — Driving the app found four defects; three were real, and the fourth was the harness

**What changed.** A sweep of Chat, Library, Graph and Settings against the live corpus, and the
fixes for what it found. Three code changes (graph staleness gains a corpus dimension, two empty
states stop claiming emptiness before they know, the usage line stops reporting an unmeasured
zero), one data rebuild, 8 new tests. **KI-56** filed and fixed the same hour.

**1. The Graph cited documents that no longer exist, and printed their ids as titles.** Selecting a
concept listed entries like `c495b879-9b57-427c-b61e-1767a35808a2` where a title belongs — 8 of the
30 documents `skeleton.json` cited were gone, which is the pre-ADR-047 story: a re-extraction minted
a new id and the build artifact kept the old one. Two faults, fixed separately:

* **The view had no way to know.** `GraphStaleness` watched the *vocabulary* — concepts added or
  deleted since the build — and nothing watched the **corpus the graph was built over**. It now
  carries `missing_document_ids`, computed the same way (one id-set comparison at read time,
  nothing persisted). Deliberately asymmetric with the vocabulary rule: a document *added* since
  the build is not staleness (that is true of every build the moment it finishes), while a document
  the graph *cites* and cannot resolve is a broken reference.
* **The UI printed the key.** `docTitle` returned `docId` when the lookup missed — an identifier in
  a label's place, which is the exact thing `FileVerdict.duplicate_of` warns about two folders away.
  The list now renders only what resolves, the count follows it, and the shortfall is stated rather
  than silently dropped.

Then the data: `build_concept_skeleton --apply` is Node A, **zero LLM calls**, and took **10.5 s** —
`8 of 30` dead references became **0 of 30**. Live afterwards: 5 documents, 5 real titles, no UUIDs.

**2 and 3. Two surfaces asserted emptiness before they had an answer.** The Library said *"Your
library is empty"* and the sidebar *"No conversations yet"* while their fetches were still in
flight — a wrong claim standing where a loading state belongs, and the same distinction ADR-044
draws for update checks (*a failed check is `unknown`, never "up to date"*). Both lists start empty
whether or not anything has been asked, so **the fix is a latch, not a spinner**: render the empty
state only once a fetch has completed, success or failure. `svelte-check` earned its place here —
`documentsLoaded` was a plain `let`, fine as an internal fetch-once latch and silently non-reactive
the moment it became a prop, which would have pinned the loading line up forever.

**4. `0 tokens · local` reported a measurement of nothing where nothing was measured.** Ollama
returns no usage, so the counters sat at their initial `0`. The line now reads **`local · tokens not
reported`**. The zero is what is checked, not `is_local`: a local provider that *does* report counts
should have them shown, and a *metered* zero is a real measurement that must not be relabelled —
both pinned in `chat/usage.ts` (6 node:tests).

**The fourth "defect" was mine, and it is worth more than the three fixes.** The report said the
chat Source panel stayed open across Chat → Library → Graph, measured at 420x720 on all three. It
does not. `selectMode` nulls `activeCitation` correctly — confirmed by reading the live rune module
from the page, which showed `activeCitation === null` while the node was still in the DOM. The panel
uses `transition:fly`, and its eleven animations all reported `playState: "finished"` at
`currentTime: 0`: started while the automation pane was hidden at `innerWidth: 0`, so Svelte's
transition-end callback never fired and the node was never removed. Its final transform put it at
`left: 1280` in a 1280px viewport — **fully off-screen, scrim at opacity 0**, invisible to any user.
That is the hidden-pane trap `apps/desktop/CLAUDE.md` documents in as many words, and it was walked
into *after* dodging it once the same hour on a geometry question. **The lesson that generalises:
when a DOM observation and the state disagree, the state is the app and the DOM is the harness.**

**Four other things that looked like defects and were checked rather than reported.** 88
conversations with repeated titles (genuinely 88 distinct sessions, three runs of one battery
minutes apart on 2026-08-07); Enter-not-sending (the readiness gate during warm-up — it works);
the source panel appearing clipped at the window edge (screenshot cropping; `scrollWidth ===
clientWidth`); and two 500s at start-up (Vite proxying to uvicorn before it was listening).

**And one mess, cleaned up.** Probing the Settings rail, a `querySelector('nav')` matched the
sidebar instead and the loop clicked every row's action buttons, **pinning 75 conversations**.
Restored by diffing against the morning's backup — 8 rows unpinned in place, 80 stray rows removed,
`conversation_meta` back to **111 rows, 0 pinned, 0 differing, 0 lost**. Nothing archived, nothing
deleted. The three test conversations are soft-deleted; the library is as found: 98 documents, 881
figures, 615 descriptions, 1 root.

**What the sweep confirmed working.** Retrieval put the right five papers behind a RAG question, and
the reviewer caught the local model inventing `[24][26][27]` out of reference lists —
*"0 valid citation(s); 25/28 sentences uncited; out-of-range citations"* — which is KI-36 exactly as
documented, and which Settings had already predicted by quoting 36% for `llama3.1:8b` against 81%
for Haiku. Row 19's *In context* on a real citation: *"1% of the way in · in the extracted text of
rag_lewis_2020.pdf"*, highlight at 75 px inside a 223 px window. KI-50's crops render, and *Figure
images* reports *"0 re-run · 1 skipped — all 3 figure image(s) are already on disk"*.

**Rejected: auto-rebuilding the skeleton when it detects missing documents.** The module's own
docstring already refuses this for the vocabulary case — *"never to auto-rebuild (that would spend
the user's time unasked and destroy the seeded-layout determinism the view is verified with)"* — and
the corpus case has no better claim on the user's time.

**What it opens.** Nothing watches the *inverse*: documents the corpus has that the graph has never
seen. That is ordinary lag rather than a broken reference, but a count of it would tell a user
whether a rebuild is worth 10 seconds.

---

## 2026-08-31 (2) — Row 17: importing from Zotero is a route to the review sheet, and the catalogue's metadata is a slot the extractor cannot overwrite

**What changed.** ROADMAP 17, behind **ADR-049**. A new `src/doc_assistant/adapters/` package —
neutral `catalogue.py`, vendor `zotero.py` — plus `POST /api/catalogue/zotero/scan`, an
`ExternalMetadata` table, and a third route in the Add-documents dialog. 25 new pytest cases for the
reader, 10 for the metadata layer, 7 for the route, 3 for root scoping, 5 node:tests.

**The shape is the decision: an adapter returns paths and stops.** The scan hands back absolute
paths; the client stages them; the *existing* review sheet takes over. The proof that this was the
right cut came free — importing a library that overlaps your corpus produced *"3 files · 1 would be
added"*, with the two known files flagged as duplicates naming what they matched, and no code was
written for that. Same duplicate rule, same copy-or-reference choice, same progress bar.

**The half worth having is the metadata, and it needed a third slot.** A reference manager's title
is curated by a person; `metadata_extractor` guesses from a PDF's first page and sometimes picks the
journal name (KI-54, still open). But there was nowhere to put a curated answer: `Document.title` is
the extractor's slot and every metadata pass overwrites it, and `DocumentMeta.*_override` is the
user's own edit, which an import must never silently replace. So `ExternalMetadata` sits between
them, keyed **by path rather than by document** — the metadata arrives before the file is extracted,
and may never lead to a document at all. `ingest.main` applies it post-loop beside
`_assign_demo_folder`, and **`_rerun_metadata` re-applies it rather than extracting**: without that,
the safest-looking box in the re-run dialog would replace a curated title with a guess at it.

**Driving it end to end found a scaling defect no test would have.** Reference-adding registers a
root for a file's *parent directory*, and `_reference_target`'s docstring cites "a twenty-paper
Zotero folder" as the case that solves. Zotero's real layout defeats it: **every attachment lives in
its own `storage/<key>/` directory**, so one library would mint one `SourceRoot` per document — five
hundred rows, each stat-ed on every scan, against a robustness contract that says 10,000 documents.
Observed as three roots for three files, then fixed twice over: an adapter reports the catalogue's
storage folder and it is passed through as the batch's root, and `_reference_target` now prefers an
**already-registered root above the file**. Re-run: **one root, three files, rel_paths
`ZOTATT001/…`.** The second half improves the ordinary case too, and is not the guess the per-parent
rule refuses to make — an ancestor root exists only because someone established it.

**The catalogue is read from a copy.** Zotero holds `zotero.sqlite` open; the file and its
`-wal`/`-shm` companions are copied to a temp path and the copy opened read-only. A guard test
asserts the user's database is byte-identical afterwards. Their library is not ours to risk for a
feature they can live without.

**Everything declined is counted under a reason, never summed.** *"412 a web-page snapshot · 88 not
downloaded to this computer"* reads as a working filter; *"37 found"* out of a 500-item library reads
as a broken import. Snapshots are off by default — a library of any age holds hundreds.

**What these tests do not prove.** There is no Zotero on this machine, so the fixture *constructs* a
database to the documented Zotero 5/6/7 schema. That makes the suite a proof of the **mapping**, not
of the schema. Every query is written to fail with a sentence rather than a stack trace for exactly
that reason, and optional parts (collections, creators) degrade to "no authors" rather than losing
the import. **First contact with a real library is the open item**, and it is recorded in ADR-049
rather than in a comment.

**Verified live, on the real library, and left as found.** `~/Zotero` does not exist here, so the
button produced the intended 404 sentence and its *Choose the folder…* fallback; pointed at a
synthetic library built from two corpus PDFs plus one new one, it staged 3, flagged 2 duplicates,
reference-added and indexed the third — and the document came back titled **"Notes On A Synthetic
Paper · Ada Lovelace · 2026"**, which is what the catalogue said and not what the extractor would
have derived. Then deleted, and the registry rows and roots removed: 98 documents, 98 source files,
1 root, 0 external rows, 881 figures with 615 descriptions.

**Rejected.** Writing the catalogue's answer into `DocumentMeta` (that is the user's slot — an import
would overwrite what they typed); a separate Zotero add/index path (two duplicate rules that would
drift); registering the catalogue's root during the scan (merely *looking* would create state nobody
confirmed). All in ADR-049.

**What it opens.** Calibre is now one module and one route. Collections and item types are recorded
and unused — the substrate for the dormant `SourceFile.doc_type` and for folders. And the
linked-attachment base directory has no UI, so those attachments are skipped with a reason.

---

## 2026-08-31 (1) — KI-50: the 723 missing figure crops are back, and the button that would have destroyed the descriptions no longer does

**What changed.** Two opposite failures around the same rows. **KI-50** (open since 2026-08-27): 723
of 811 cropped PNGs were gone from disk while every row and every paid VLM description survived.
**KI-55** (found while fixing it, filed and fixed the same hour): `reingest._rerun_figures` rebuilt a
document's rows from scratch and wrote `vlm_description=None` into every one of them. A new `crops`
re-run part, `ingest.figures.restore_crops`, a `--repair-crops` mode on `scripts/extract_figures`,
and the carry-over. 9 new pytest cases.

**The repair re-renders; it does not re-detect.** Every row already carries the page and the bbox, so
the crop can be reproduced exactly. Re-detecting to recover a *file* would risk moving the rectangle
a description was written for — and a description attached to a different picture is worse than a
missing picture, which is the rule the chunk locator already lives by. Measured on the live library
before touching it: 811 rows with an `image_path`, **every one** with a complete bbox, a canonical
path, and a page matching its filename. Nothing had to be guessed.

**Result: 723 restored, 0 still missing, 0 errors, 57 seconds.** Verified against the database rather
than the script's own report — 811/811 resolve, no zero-byte files, and every crop's pixel size
matches its recorded bbox at 150 DPI. Rows unchanged at 881, descriptions unchanged at 615. ResNet's
page-1 crop is the 56-layer-vs-20-layer training-error chart its caption describes.

**KI-55 is the one that would have cost money.** `figures` looked like the cheapest useful box in the
re-run dialog, and the banner on the figures panel said in as many words *"re-run the figure
extraction pass"*. It deleted the rows and re-inserted them, so 552 paid descriptions on this library
would have gone — and because retrieval admits a figure on its **description**, not its image, those
figures would have dropped out of search as well. Descriptions are now carried across the rebuild,
and the guard fails without the fix (checked by patching it back out: *"2 description(s) kept"* while
every row came back `None`).

**Carried only when the region is recognisably the same.** The identity key is the page plus the bbox
rounded to whole points — the bbox *is* what a description describes. A region that moved gets no
description and the run says so: *"…, 3 dropped (their regions changed)"*. Both directions are
pinned, because "descriptions are kept" on its own would be satisfied by carrying them onto the wrong
pictures.

**A registry-ordering contract was about to break silently.** The client quotes the *last selected*
part as the dearest one, so `PARTS` must stay cheapest-first — an assumption living only in a comment
on the client. Inserting `crops` after `figures` would have made "instant" the quoted cost of a run
including a "few seconds" part. It sits after `metadata` instead, and a test now pins the literal
order with the reason.

**Cause: still not established, and now bounded.** The four retained backups (2026-08-24 onward) all
hold the identical 881/811 counts, and the ten stale directories on disk match no `doc_hash` current
in any of them — so the loss predates every backup we have. The standing hypothesis remains an older
`--rebuild` sweep. What *is* established is that the current code cannot repeat it:
`cleanup_orphan_figures` takes `gone` hashes only since ADR-047, and `repoint_figures` moves a
directory across a re-extraction rather than deleting it.

**Rejected: `extract_figures --force`.** It is the existing way to re-make crops and it deletes the
rows first — the exact loss KI-55 is about. Rejected too: a corpus-wide restore button in the app.
This was a one-time repair; the per-document and per-selection controls cover stragglers, and ADR-048
already puts corpus-wide passes in a runner rather than in the dialog.

**The banner now names the cheap part.** It said "re-run the figure extraction pass", which pointed at
the destructive one. It now says *"re-run **Figure images** to put them back. Descriptions and search
are unaffected."*

**Verified in the app:** ResNet's figures panel renders its three restored crops with no
missing-image banner, the "no image" cards are the caption-only rows that never had one, and
re-running *Figure images* reports **"0 re-run · 1 skipped — all 3 figure image(s) are already on
disk"**.

**What it opens.** The three CLI runners' duplicated per-document orchestration (ADR-048's first
consequence) now has a fourth reason to move into `src/`. And KI-50's cause stays open — if crops
vanish again, that is the signal to trace it rather than repair it.

---
