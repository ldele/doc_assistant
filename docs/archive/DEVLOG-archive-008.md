<!-- status: archived · updated: 2026-10-27 · class: append-only -->

# DEVLOG — archive 008 (2026-09-30 (3) → 2026-10-22)

Rotated DEVLOG entries — moved here from `docs/DEVLOG.md` once it held more than 20, **verbatim** per cpc ADR-023 rule 13b and cpc ADR-053. Newest entry on top. Append-only; never edited.

## 2026-10-22 — `virtualenv` 21.14.12: five advisories in a dev-only tool

**What changed.** `uv.lock` only: `virtualenv` 21.7.13 → 21.14.12 with `packaging` 26.2 → 26.3 and
`distlib` 0.4.0 → 0.4.4 (`uv lock --upgrade-package virtualenv`). As with `pillow` (2026-10-20
(5)), the lock keeps older versions for the Python 3.10 that `requires-python` still declares;
3.12 gets the fixed ones. **Found by:** the audit gate on `4655aae`, reported by session 5c —
CVE-2026-113722, -113725, -113726, -113732 and -113776, all fixed by 21.14.9.

**Why.** `virtualenv` reaches the tree only through `pre-commit` (the `dev` extra), so the app never
shipped it; CI's gate audits the whole environment. Checked: the gate 0 unreviewed; the commit
hooks run on the new version; 29 tooling tests (version, secret-scan gate, action pins) pass.

**What it opens.** The fifth advisory round in three days. The 3.10 fork now holds a `virtualenv`
below the fixes; raising `requires-python` to the 3.12 the project runs would end these forks.

## 2026-10-21 — `vite` 6.4.5: a high advisory in the dev server's editor launcher

**What changed.** `apps/desktop/package-lock.json` only: `vite` 6.4.3 → 6.4.5 (`npm audit fix`,
inside `package.json`'s `^6.0.0`). **Found by:** `npm audit` the morning after `96685c4`, before
any push — three `launch-editor` advisories (GHSA-698v-4r3x-3vcv, GHSA-r825-w8mm-5pxw,
GHSA-587h-7wmj-h5pm): Windows argument escaping, editor flags read as commands, an existence check
bypassed on Windows.

**Why.** The editor launcher runs only in the dev server, so the installed app never carried it;
but the dev loop runs on Windows here, and CI's frontend job fails on a high advisory. Checked:
`npm audit` 0; `svelte-check` 0 errors over 229 files; 299 unit tests; a production build.

**What it opens.** Nothing new — the fourth advisory round in two days, all fixed inside their
version ranges except `pillow`'s.

## 2026-10-20 (5) — `pillow` 13.0.0: eight advisories, fixed only in a new major

**What changed.** `uv.lock` only: `pillow` 12.3.0 → 13.0.0 (`uv lock --upgrade-package pillow`).
The lock keeps 12.3.0 for Python 3.10 alone, which `requires-python = ">=3.10"` still declares and
pillow 13 (`>=3.11`) no longer serves; the project runs 3.12 (KI-2), where 13.0.0 installs.
**Found by:** the local audit gate while staging entry (4), the same evening as entries (2) and (3)
— eight advisories, seven of them high: out-of-bounds reads in the TIFF decoder and encoder, a GIF
decompression-bomb DoS, unbounded recursion in pillow's PDF parser.

**Why.** Hostile documents are in the threat model (`docs/security.md` §1), so reviewing these into
`pip-audit-ignore.toml` was rejected. The app reaches pillow only through `pdfplumber` (its sole
dependent, `Pillow>=9.1`); its own code never imports it outside `scripts/debug_tables.py`, and
figure crops go through PyMuPDF. Checked: the gate 0 unreviewed; 406 PDF, table, figure, extractor
and image tests pass; a library page renders through `pdfplumber.to_image` on 13.0.0.

**What it opens.** Raising `requires-python` to the version the project actually runs would remove
the 3.10 fork from the lock — a decision for the KI-2 pin, not this fix.

## 2026-10-20 (4) — Release refresh for 0.7.0: the walkthrough, the security record and the CHANGELOG brought level with what shipped

**What changed.** `docs/RELEASE.md` §0 and §2, done ahead of the release session while session 5c
waits on the user's labels; no code. `docs/release-ux-checklist.md`: a row per surface shipped
since `v0.6.0` (20 product commits) — the extraction pool, Manage keywords at library size,
proposed placements, keywords in score order (5c), the KL1 labels, names and written case, the
Definition card, term signals — plus the "no console window" row and S-4, S-5 and S-6 checks.
`docs/security.md` §2 lists S-4, S-5, S-6, S-8 and S-12 as holding; the launch token is a rule by
method, so the write routes added since are covered without a line each. `CHANGELOG.md`
`[Unreleased]`: an *Added* block, the security, change and fix lines that were missing, and
*Known limits*, each of 0.6.0's re-read against the code (row 24 still planned; since the tag, the
code behind the others changed only for the ligature fix and the size caps).

**Why.** `preflight`'s `checklists` check fails a release whose walkthrough and security record
have no commit since the previous tag, and the Unreleased section stopped before the definitions
work: release criterion 5 needs it to say what the README says.

**What it opens.** Session 5c adds its own line (keywords in score order, junk filtered) and the
measured keyword numbers to the CHANGELOG. The version bump, the 0.7.0 notes, the build and the
gates stay with the release session.

## 2026-10-20 (3) — `certifi` 2026.10.19: one more advisory, published the same day

**What changed.** `uv.lock` only: `certifi` 2026.5.20 → 2026.10.19 (`uv lock --upgrade-package
certifi`). **Found by:** CI's blocking audit on `2ab7895` (run 39487577705), GHSA-6q48-6c7w-9gw2 —
entry (2)'s three `aiohttp` advisories were gone; the tests passed there (2,754).

**Why.** A newer certificate bundle from the same maintainer; nothing to review into
`pip-audit-ignore.toml`. Checked: the gate 0 unreviewed; two HTTPS connections verified with the new
bundle.

**What it opens.** Two red runs in one evening from advisories published between pushes; the
alerts toggle (entry (2)) still decides whether the next one is seen before a push.

## 2026-10-20 (2) — `aiohttp` 3.14.5: three advisories published overnight turned `main` red

**What changed.** `uv.lock` only: `aiohttp` 3.14.3 → 3.14.5 and its dependency `yarl` 1.24.2 →
1.25.1 (`uv lock --upgrade-package aiohttp`); nothing else moved. **Found by:** CI's blocking audit
on `c27ff9f` (run 39479200471) — CVE-2026-111063, CVE-2026-111064 and GHSA-2g87-pp6c-29x5, all fixed
in 3.14.4; the four Dependabot merges after it fail at the same step.

**Why.** An upgrade inside the same minor series, so no review into `pip-audit-ignore.toml` is
needed. Checked: `scripts.pip_audit_gate` 0 unreviewed; the API integration tests (184) pass on
the `cu130` venv.

**What it opens.** The sixth time a push that touched no dependency met a red audit (KI-58's
shape). Dependabot alerts, still off in the repository settings, would have flagged these between
pushes.

## 2026-10-20 — The README opens on its reader and stops over-stating how settings were chosen and how often local models cite; a project overview gives every figure its source

**What changed.** `README.md`: the opening says who the app is for and what sets it apart (the
printed page behind a citation, the sources kept apart from the model's interpretation, a local
model that keeps documents on the machine), then one result with its size (the public benchmark's
`contains_all` 0.932 ± 0.014 over 5 trials, and why `citation_overlap` at 1.000 cannot rank), then
Quick start, moved up. "Why it's built this way" moves below Architecture, and its first bullet no
longer says `TOP_K`, parent-child, chunk sizes and the BM25 mix "were each picked by measuring
alternatives": it names what the harness measured, says the chunk-size and weight sweeps kept the
defaults, that `TOP_K` and parent-child predate the harness, and that the reranker had no
experiment. Limitations quotes the shipped prompt's citation coverage, 83.5 / 37.6 / 18.0% (one run
each, `tests/eval/baselines/citation_coverage_2026-08-07.md`), in place of the earlier prompt's
81 / 36 / 14%, and replaces "Answers stay grounded either way" with the count behind it: 12 of 27
`llama3.1:8b` answers cited nothing. New `docs/OVERVIEW.md`: what the project is, what sets it
apart, how it works, each measured result with its size, conditions and source, and its state;
linked from the Documentation table. The old opening's second paragraph is gone; its map of the
library is already the concept bullet under What it does.

**Why.** Read against the record on 2026-10-19 and 2026-10-20, two README statements claimed more
than it holds: settings chosen by experiment, and citation rates from a prompt that no longer
ships. The overview keeps those facts in one place, each with its source, so a reader who reuses
them does not re-derive them from the README.

**Rejected.** Rewriting the 0.4.1 and 0.5.0 CHANGELOG entries that quote 81/36/14%: released
entries stay as they shipped. Leading with the citation-coverage figure: it comes from the private
library, one run each, where the public benchmark is one anyone can re-run. A new alt text for the
demo GIF: the GIF dates from 2026-08-11, before the page pane, and its alt text still describes
what it shows.

**What it opens.** The provider card (`apps/desktop/src/lib/settings/ProviderSetup.svelte`
l.254-256) still shows 36/14/81% and "Answers stay grounded in your documents"; the 2026-08-07
baseline planned that refresh for the next release that rebuilds, which 0.7.0 is, with a line under
the CHANGELOG's Unreleased beside it. `tests/eval/TESTING.md` says five scorers where `scorers.py`
has six (`figure_retrieval`). The demo GIF predates the page pane (0.6.0). ROADMAP row 98 (the docs
revamp after 0.7.0) starts from the README's new order. Docs only; no code or test changed.

## 2026-10-18 (3) — The README leads with what a reader gets and states 0.7.0's limits; CI's actions are pinned to commits (security S-12)

**What changed.** `README.md`: the opening says what you get (answers you can check, the page one
click away, what leaves the machine with each engine) before how it is built; the concept bullet
covers names, forms and definitions; Status names 0.7.0 as in preparation, with v0.6.0's date
corrected to 2026-09-04 and six things the release makes true; Limitations is grouped under four
headings and gains the reference-list cut (KI-59), pages holding the previous page's text (KI-57),
concepts versus extracted keywords, the definition grade, the wiki sample and the CLI provider
default (KI-4). Every count is from 2026-10-18: 104 documents, 3,341 pages, 44 reference links,
1,436 of 1,483 keywords in one document, 2,757 Python and 299 desktop tests (the full suite passed,
2,754, before the 3 guard tests were added). `ci.yml`: the 7 `uses:` lines pinned to commit SHAs,
each the commit its major tag pointed at that day; `.github/dependabot.yml` for `github-actions`
only, monthly, minor and patch grouped; `tests/unit/test_ci_actions_pinned.py` fails on a tag (seen
failing on `@v4`). Committed by the user with the checkpoint as `71b3903`.

**Why.** Plan session 5b: the README is the release's front page, and its Status still described
0.6.0 with the wrong date and 2,389 tests; two Limitations links pointed at gitignored files a
public reader cannot open. S-12 was decided at the checkpoint the same day: a tag can be moved to
other code, a SHA cannot.

**Rejected.** Writing Status as already released: the README is pushed before the release, so it
says "in preparation" and one line changes on release day. Bumping the actions' majors while
pinning: that is a behaviour change, and Dependabot proposes each as its own PR.

**What it opens.** Five Dependabot PRs, all new majors (`checkout` 4 → 7, `setup-node` 4 → 7,
`setup-buildx-action` 3 → 4, `setup-uv` 3 → 10, `build-push-action` 6 → 7): the user's to review.
Dependabot alerts read off (404). The CHANGELOG's Unreleased section stops before the definitions
work and the KI-62 to KI-65 fixes; release criterion 5 needs it to match the README.

## 2026-10-18 (2) — Release checkpoint: ADR-053 and ADR-041 accepted, 93a's suggestions stored, a docs revamp planned after 0.7.0

**What changed.** The user took the seven calls left before 0.7.0, one at a time. ADR-053 is
accepted, with expert vocabularies (93c) now built before model candidates (93b), which ride with
ROADMAP 25. ADR-041 is accepted as amended: option 6's buildable part is the gap list, so KL1b
closes and no separate per-concept status is built. `extract_definitions --apply` stored 20
suggestions for 11 of the 13 concepts (12 strong, 8 some), nothing chosen, after a `library.db`
backup. `docs/security.md` S-12 is decided: SHA-pinned actions and Dependabot for actions only,
before the release; alerts are the user's toggle. ROADMAP row 98 adds a docs revamp after the
release (row 84 folded in), and row 95 follows it.

**Why.** Both ADRs had been `proposed` for one and two months while what they decide was built
and used; the release describes them. Row 51's `--apply` had nothing to write: since ADR-054 the
proposer reads the 13 concepts, and none shares a head with another (the 27 candidates are between
terms). Row 53 (3)'s hand score leaves the release path: the merge default pairs 0 concepts, and
the review page's store was empty.

**Rejected.** Refreshing epistemics: a dry run marks 1,188 chunks (445 today), most `contested` —
the stance ADR-041 does not trust; markers stay off by default and the table is rebuilt with KL3.
Update PRs for pip, npm and cargo: weekly noise for one maintainer; upgrades stay in sessions, the
audit gate catches advisories.

**What it opens.** The README (the next session) must not imply `is_a` proposals exist on this
library. The docs revamp's evidence and order are in the local plan.

## 2026-10-18 — A wiki topic's prompt fits the model's context: a large topic is summarised from a fixed sample, and its note says so (KI-65)

**What changed.** `wiki.summarize_cluster` sends a bounded prompt. `material_budget` reserves the
answer (`max_tokens`), a chat frame and the prompt's fixed text out of `WIKI_CONTEXT_TOKENS` (new,
8,192); `select_material` fills the rest with members in `sample_order`, a hash of each
`doc_hash` (the same members give the same sample in any order, and no end of the library is
favoured), up to the first block that does not fit. Size is estimated without a tokenizer, as
UTF-8 bytes / 3 (`BYTES_PER_TOKEN`), above the true count for every script. Excerpts are read in
rounds of 16, so a topic reads the store for its sample, not for every member (48 reads for the
87-paper topic, not 87). A topic summarised from a sample says so: in the prompt's heading ("a
sample of 36 of 87"), in the note ("_Summary written from 36 of 87 papers_", `summary_from: 36` in
the frontmatter, *in summary* beside each sampled source) and in the CLI report. An empty topic,
or a window too small for the fixed text, makes no call. Tests: `tests/unit/test_wiki_budget.py`
at 0, 1, 20 and 10,000 documents and five windows (fake client, fake store, no network);
`tests/unit/test_llm.py` for `num_ctx` and `ollama_context_length`. `docs/knowledge-layer.md`
§6's wiki row no longer says the note is written from a cut prompt.

**`num_ctx`: passed, from the model.** `OllamaClient` takes `num_ctx`; the default `None` keeps
every other caller (reviewer, judge, taxonomy) on the server's default, unchanged.
`wiki.fit_window`, called by `build_wiki`, gives an Ollama client `num_ctx = min(WIKI_CONTEXT_TOKENS,
the model's trained length)`, read from `/api/show` by `llm.ollama_context_length`, and the budget
uses the same number. Keeping the budget below the default was rejected: the default depends on
the GPU (`vram-based default context`, 4,096 on this 12 GB card), and on overflow llama.cpp keeps
only half the window (the log's `limit=2050`), so "below the default" is a number the app cannot
read. Asking for the window makes the budget and the server agree on any card.

**Checked live, read-only** (Ollama forced; the library read through `sqlite3` `mode=ro`; notes
rendered to the session scratchpad, nothing under `data/`). The 87-paper topic, rebuilt from
`library.db` (cosine 0.90). Before: 15,232 real tokens on `llama3.1:8b` (15,245 in the
baseline), 15,775 on `qwen2.5:7b`, 15,766 on `qwen3.5:9b`. After, at a 8,192 window: 36 papers,
6,239 / 6,517 / 6,486 real tokens plus a 400-token answer, no `truncating input prompt` in the
server log, 5-11 s a topic. The estimate ran 14-17% above the real count (3.50-3.63 bytes per
token on this library). Two of the three summaries name both the machine-learning and the
neuroscience papers; `qwen3.5:9b`'s names machine-learning ones only. Their quality is not judged
here: that is RG-018's re-measure.

**Rejected.** Summarising in stages (map-reduce): a call per ~36 papers plus a merge, so a
10,000-paper topic costs hundreds of calls, and the merge summarises summaries, not papers. The
model's whole window (32k-262k tokens here): VRAM and time for one note, and the sample would
change with whichever model ran. A real tokenizer: an arbitrary Ollama model exposes none, and
the byte estimate held on all three. The "most central" papers as the sample: that needs a
centrality measure and its threshold; a hash sample is unbiased and needs neither.

**What it opens.** RG-018's re-measure with KI-65 fixed (baseline §4, condition (c)). KI-65 can
close in the local `KNOWN_ISSUES.md` once this lands. A large topic still leaves papers out (51 of
87); if a sample proves too thin, a second pass is the map-reduce above. `WIKI_CONTEXT_TOKENS`
is set by environment variable only; `build_wiki` has no flag for it.

## 2026-10-17 (4) — A write to the local API needs the token the app was started with (security S-6)

**What changed.** The Tauri shell mints 32 random bytes per launch (`getrandom` 0.3, already in
`Cargo.lock` through Tauri: one line in the lock, no new crate), passes them to the API sidecar as
`DOC_API_TOKEN`, and hands them to its window through a new `api_token` command. The API
(`apps/api/launch_token.py`, mounted inside CORS) answers 401 to any POST, PUT, PATCH or DELETE
without a matching `X-Doc-Api-Token`, compared in constant time, and logs `security_token_refused`
with the method and path, never a value. Reads stay open. The domain clients call `apiFetch`
(`core/api/_base.ts`), which adds the header to writes only (`core/launchToken.ts`, tested), so
reads cost no extra preflight. With `DOC_API_TOKEN` unset nothing is checked: `just api`, the test
suite (`conftest` removes a shell's token) and the headless image run as before.

**Why the shell accepts a preset token.** The RG-012 release check launches the installed app and
drives its API from PowerShell, outside the window; a token only the window knew would fail the
run as if the app were broken. The harness now sets a random `DOC_API_TOKEN` before launching and
sends it on its two writes (a test holds that). Whoever can set the shell's environment already
runs as the user.

**Checked.** 8 API tests (401 on `/add`, `/ingest`, `DELETE`; a wrong, empty or case-changed token
refused; the token reaches the route; reads and preflights pass; the 401 carries CORS headers;
unset means unchecked; the refusal log holds no token), 4 frontend tests, the harness test, `cargo
check`. Live in `tauri dev`: with no token set the app loads and a write succeeds; with one preset
on both sides, a POST without the header gets 401 from the window and from outside (`/add`,
`/ingest`, `DELETE`, `PATCH`), the app's own write succeeds, and `/api/health` stays open. **Not
checked:** the installed build (a walkthrough row) and the RG-012 sandbox run (the next release).

**Rejected.** Guarding reads too: CORS already stops a page reading answers and S-5 stops it
becoming same-origin, while every figure, health poll and page image would pay a preflight. A
`BaseHTTPMiddleware`: pure ASGI leaves the chat's event stream untouched. Wrapping `window.fetch`
globally: an explicit `apiFetch` can be read and tested. Failing closed when unset: the dev loop
and the suite have no shell to mint one.

**What it opens.** S-7 is the current step. S3's "arbitrary path" half: `/add` and `inspect` still
take any path, now only from the app's own window.

## 2026-10-17 (3) — KL4: neither wiki clustering path partitions the library, so the default stays; the trust table states the placement runs RG-015 already had (RG-018, RG-015)

**What changed.** No code. `tests/eval/baselines/wiki_clustering_2026-10-17.md` measures both wiki
paths on the working library (104 documents). Cosine union-find at 0.90, the default, makes 12
topics, one of them 87 documents; the concept-community path makes 80, 75 of them single-document,
because the 13 graph concepts reach 29 documents. An `--apply` on `llama3.1:8b` (Ollama forced,
notes written to a probe folder, `data/wiki/` untouched), read by hand: 23 of the community path's
29 grouped documents sit with their subject, against 8 of the cosine path's 95.
`docs/knowledge-layer.md` §6: the placement row said "unmeasured, never run"; it now states the
three runs in RIGOR_TODO RG-015 (the labelled one: `llama3.1:8b` 82% defensible / 70% exact against
a 60% majority baseline, `qwen3.5:9b` 87%), and a wiki row is added. §4's diagram and §5's table
said the wiki reads communities; it reads similarity edges, communities opt-in. RG-015 is restated
as blocked on ADR-019's domain-coverage detectors, which `gaps.py` does not define. ROADMAP KL4
done.

**Why the default stays.** Flipping trades one blob for 72% single-paper notes and scatters the six
political-science papers the cosine path groups correctly. A new threshold would be fitted to this
library: the blob breaks between 0.93 and 0.94. Re-measure when the graph's vocabulary covers most
of the library, when a relative primitive is designed, or when KI-65 is fixed.

**Found.** KI-65: the 87-document note was written from a prompt Ollama cut from 15,245 tokens to
2,050 (`OllamaClient` sets no `num_ctx`; the default here is 4,096), so it describes one of the
last dozen papers. Nothing in the app says so.

**Rejected.** The CLI's `--apply` into `data/wiki/`: it would overwrite the user's notes for a
measurement, so the probe calls the same `build_wiki` with `wiki_dir` set. Retuning
`WIKI_MIN_SIMILARITY` from the sweep. Scoring the 75 singletons for coherence: a one-paper topic is
on one subject by construction; 10 were sampled for faithfulness.

**What it opens.** KI-65's fix (a material budget per call, `num_ctx` from the model). A relative
clustering primitive is design work with an ADR. RG-015 waits on the coverage detectors.

## 2026-10-17 (2) — CI hung for three hours: the extraction pool forked a multi-threaded process; it spawns on every OS now

**What happened.** CI on `93edbe6` sat "in progress" for 2 h 59 min in
`tests/integration/ingest/test_ingest_rebuild_preserves_library.py`, then again on a rerun of the
same commit, at the same place, with two orphan pytest children each time. That test ingests two
documents, so `warm_extraction_cache` opens a pool of two (`light` = `min(2, cores)`). Linux's
default start method is fork; the caller was multi-threaded, and a forked child inherits any lock
another thread holds at that instant and waits on it forever. Every green run before had logged
Python's own warning for that test ("This process … is multi-threaded, use of fork() may lead to
deadlocks in the child", 8 times on 2026-10-15). Why `93edbe6` made it deterministic is not
established; the hazard was there before it.

**What changed.** `ingest/workers.py`: the pool uses `multiprocessing.get_context("spawn")` on every
OS. Windows always spawned, so the desktop's path is unchanged and Linux and the Docker image now
run what was tested; `_extract_one` was already written to be picklable for it. A unit test pins
the start method. The pytest step in `ci.yml` gets `timeout-minutes: 20` (3.5 min when green): a
hang showed as "in progress", which the session-start check reads as "not yet". KI-58 gains the
silent variant.

**Not verified here:** Linux. On this box the integration test passes (it always spawned); the fix
is proven on Linux by the first CI run that carries it.

**Rejected.** `forkserver` (Linux-only, and a third behaviour to reason about). Forcing one worker
in the test (hides the same hazard the API meets on Linux, where `/api/ingest` runs beside
uvicorn's threads). Finding the thread that held the lock: spawn removes the whole class.

**What it opens.** Nothing new; the extraction pool's frozen-build guard is unchanged.

## 2026-10-17 — The concept panel shows what the text says about a name: short and long forms, a cited author, and where an unused label occurs (ROADMAP 93)

**What changed.** The three signals the user's answers supported (`term_signals_2026-10-14.md` §5)
are on the Definition card in the Graph tab's concept panel, for a concept and a term alike:
- **Short and long forms**: "“DBS” is spelled out as “deep brain stimulation” · in 2 documents ·
  already one of its forms", with the sentence it came from, both forms marked, its paper and
  *Open passage*. The marks come from the rule that found the pair (`term_signals.offer_marks`, a
  re-read with positions), so they land on the short form as written and on word edges, never on
  the `co` of "communication", the review page's fault.
- **Cited as an author**: shown when "et al." follows half or more of the mentions
  (`TermSignals.cited`; the runner's `_majority` moved into the module as `majority`), with the
  first such sentence and the name marked.
- **Not in your library's sentences**: for a label no body sentence uses, where it occurs instead:
  tables, headings, a figure's caption or description, reference lists, lines that are not
  sentences (a list, a page header or footer, code), or sentences that write it another way
  (`where_it_occurs`). Each with a count and one marked example. Never the word "noise".
- Not shown: the word beside a label (not measurable as asked).

`GET /api/concepts/{id}/signals` (definitions router; 404 for an unknown id or a field) answers
from `term_signals.load_concept_signals`. The keyword index picks the documents through
`definitions.parents_mentioning`, a sibling of `chunks_mentioning` that lets a match in a figure's
text choose a document and returns the figure parents beside the prose ones. Nothing is written.
On the working library every label tried answers in under 0.25 s; `comb` and `cre`, whose index
prefix matches 86–90 documents, take about 1 s.

**Why the places use a looser match than the counts.** Whether a sentence uses the label is
decided by the product's matcher (a space, hyphen or dash joins words; case rule). Where a label
*comes from* is asked the keyword extractor's way, any non-alphanumeric run between its words,
because that is how the row was made. With the strict matcher, `comput vis` ("Comput. Vis."),
`resources neuroscience` (a page header across a paragraph break) and `koonce emerson` ("Koonce,
Emerson et al.", a running header) had no place; with the extractor's they have theirs. A hit
inside a body sentence is shown as *written another way*: no count includes it.

**Bounded read.** At most `SIGNAL_DOCUMENTS = 200` documents (about 15 ms each here, so a few
seconds at the bound), those with the most matching blocks; above it the card says "Read in the
200 documents that mention it most, of N". Every count is over the documents read. No label on
today's library reaches the bound (most: 90).

**Rejected.** Reading the whole vector store per panel open (minutes at 10,000 documents).
Importing `ingest.figures.figure_parent_text`'s separator: the `ingest` package loads torch, and
the first panel open took 30 s; the separator is a constant in `term_signals`, held to ingest's by
a unit test. An *Add as a form* button on an offer: a form's exact or broad mark is the user's
act in Manage keywords (ADR-054), so the card says where, and writes nothing.

**What it opens.** The smaller signal fixes stay queued (baton 2026-10-15): names cited alone
(`sporns`), "Fig." as a short form, a display form for damaged long forms (an offer shows the
extraction's bold markers as they are). The index's prefix match orders documents by blocks that
may be other words ("combination" for `comb`), so past the bound the documents read are not
always the ones that use the label most.

## 2026-10-14 (5) — The library re-extracted and the graph rebuilt: 38 documents get their ligature letters back

**What changed.** No code. At the user's request, after `87fbf34` (KI-63) and `6697446` (KI-64):
every store backed up (`data/*.bak-20261014-214357-preki64`), then "index the folder"
(`python -m doc_assistant.ingest`), the keyword index rebuilt, and the graph and its gaps rebuilt
with the Rebuild button's two calls. All 104 PDFs re-extracted; **38 changed, 66 identical, no
errors**. Each of the 38 changed only by restored ligature letters, 519 in all, one of them a "tt"
("paterns"): the measured 22 undercounted. The graph: version `58136820706f4cab` →
`dbf0e870aea6fed1`; the same 29 documents, 72 pairs and gap rows; one link, `re-ranking` — `BM25`,
49 → 50 shared chunks; two mention counts +2; written forms 366 → 370. The stored text holds no
ligature loss except the 3 OCR errors of the 1949 Hebb scan.

**Corrected.** `hyphen_matcher_2026-10-14.md` §1 summed every form, broad ones included, and
predicted `cre` 376 → 395; the graph counts exact forms only (ADR-054), `cre recombinase` is broad,
and `cre` did not move. The baseline says so beside the old figure.

**Not done, on purpose.** `scripts.compute_epistemics --apply`. The `chunk_epistemics` table dates
from 2026-07-25 (graph `c97cd22221d9cf81`), stale through every rebuild since and off by default
(`EPISTEMICS_MARKERS_ENABLED=false`, KI-33); refreshing it would take 445 rows to 1,188 marked chunks
of a signal the app hides. The user's call.

**What it opens.** The vocabulary row `fne-tune` now matches nothing in the text; whether it goes is
the user's (ADR-054: a term is never removed for the user).

## 2026-10-14 (4) — Table cells keep a ligature's second letter: "fne-tuning" is "fine-tuning" again (KI-64)

**What changed.** `extractors.widen_zero_width_chars` gives a character of zero width a box
0.01 pt wide at its own origin, and `_keep_ligature_letters_in_table_cells` hands both table-cell
functions (`pymupdf.table.extract_cells`, and `pymupdf4llm.helpers.utils.extract_cells` that this
app's layout path runs) characters measured that way. Neither library's code is copied or edited.
`extract_pdf_pymupdf` installs it, so the PDF extraction fingerprint moves and the other formats'
do not. Three tests: the box, both cell functions on the measured "fine" (each returns "fne" as
shipped and "fine" wrapped), and the two functions' parameters, which an upgrade must keep.

**Why.** Importing pymupdf4llm turns quad corrections off for the whole process; with them off the
second letter of a split ligature has a zero-width box, and both cell filters keep a character only
if more than half its box is inside the cell. 336 of the 339 damaged words in the stored text were
in table rows, in 19 documents; prose kept its letters.

**Measured** (`tests/eval/baselines/ligatures_2026-10-14.md`, the committed extractor against the
fix on the 19 PDFs and 3 controls): 416 insertions and no other edit; the damaged words 316 → 0;
"i ≠ j" in one eLife paper's tables gets its slash back; one control held "fy" and "Optifex", which
the word list missed. Extraction time unchanged.

**Rejected.** Turning quad corrections back on after the import: it would move every page of every
PDF, through pymupdf4llm's own layout code. Upgrading to 1.28.2: its changelog names no fix, and it
changes table detection. Repairing the stored text by dictionary (ADR-043). Wrapping only PyMuPDF's
function: tried first, it changed nothing, because the layout path has its own copy of the filter.

**What it opens.** The library changes when the user re-extracts: the next "index the folder"
re-extracts every PDF once, or a per-document re-ingest of the 19 (ADR-048), identity kept
(ADR-047). An upstream report is drafted (`.claude/probes/2026-10-14-ligatures/upstream_issue_draft.md`);
posting it is the user's call. The 3 OCR errors in the 1949 Hebb scan are a different matter.

## 2026-10-14 (3) — A form's words joined by a space, a hyphen or a dash are one form (KI-63)

**What changed.** `concept_skeleton.form_source` is the one definition of what a form matches: its
characters literally, except where a space, hyphen or dash joins two of its words, where any of
them matches, across a wrapped line but not a paragraph break. `compile_boundary_pattern`,
`form_matcher` (with `form_probe` as its prefilter and `spelling_key` for the case rule), the
written-form vote, the definition and naming patterns, the term signals and the keyword-index
query of "Look in my library" (`definitions._index_phrases`, every mix of spaces and hyphens)
build on it. A written form is voted on one line (`tidy_spelling`: "SSp- bfd" is `SSp-bfd`), a
written form that joins the label's words otherwise is a spelling of it, and two forms of one
concept that are now one (`cross-encoder`, `cross encoder`) count a mention once. Ten tests; with
the rule reverted, eight of them fail (the other two pin the word split and the index phrases).

**Measured** (`tests/eval/baselines/hyphen_matcher_2026-10-14.md`, the committed code against the
change on the same library, dry runs). The graph's 13 concepts: the same documents, links, gap
rows and graph version; four count a few more mentions. The 357 rows: 1,079 → 1,117
concept–document pairs, none lost (`cosine-similarity` 1 → 16 documents, `state-space` 2 → 11);
8 written forms change spelling, 4 are new; one definition candidate gained. Term signals: "never
in body prose" 31 → 28, which corrects the review baseline's §4 (3 of 31, not 2: `query
document` is written with an en dash).

**Rejected.** Any whitespace as a joiner: the first run joined `resources neuroscience` across
the blank line of an eLife page header. A dictionary of word pairs, or a join without a joiner
(`sspbfd`): no rule says when two words are one. Deduplicating forms in `surface_forms`: it keys
written forms and the user's form marks (ADR-054), so only presence's count dedupes.

**What it opens.** The stored graph and written forms change at the next rebuild (the user's).
The keyword-index path still misses one document on each of 4 rows, before and after: a term at
the end of a hyphenated word ("retriever" in "G-retriever") is a mention to the full read and not
to a prefix query. KI-64 (ligatures) is next of the two the review found.

## 2026-10-14 (2) — Term signals measured on the user's reading: the abbreviation offers hold, the neighbour signal cannot be judged as asked (RG-034, ROADMAP 93)

**What changed.** No code. The user answered the "Term Signal Review" cards, and the answers are a
baseline, `tests/eval/baselines/term_signals_2026-10-14.md`. Abbreviation offers: 44 of 44 right in
their sentence (Wilson 0.92–1.00), 10 of them without `initials`, so `initials` stays evidence and
never becomes a filter. Cited authors: 5 of 5. The neighbour signal ("followed by the same word"):
7 answers, then stopped; the cards showed a count and no text, and the question held two
questions. It is not shown. "Never in the text": not answered; each label was looked up instead,
and 2 of 31 are in the prose with a hyphen. RG-034 is done; ROADMAP row 93 says so. KI-63 and KI-64
are filed from what the lookups found.

**Why.** RG-034: nothing reaches the app until a reading says which signal holds.

**Rejected.** Asking the user to judge silent labels or neighbours without their text: the user
declined to answer what they could not check, and where a label occurs answered most of them
better. Counting the 7 neighbour answers: "no" meant "a term on its own as well", not "not a
fragment".

**What it opens.**
- **The panel** shows abbreviation offers and cited authors, each with its sentence and paper,
  marked as written (the review page marked case-blind and was caught on `CO` and `NA`); silent
  labels with the place they occur.
- **KI-63:** the shared matcher misses hyphenated spellings: 23 of 183 multi-word rows,
  33 concept–document links. **KI-64:** ligatures lost in 20 of 104 documents.
- **For the signal:** names cited alone (`sporns`, `morecraft`), "Fig." as a short form, a display
  form for damaged long forms.
- **The user's direction:** a glossary per paper before the library's, a field level only if two
  are not enough. A short form is spelled out its own way by its paper in 57% of uses, and `SC`
  means three things across papers. And for what needs judgement (which meaning, which phrase is a
  paper's term): an LLM with clear instructions, its prompt developed in the eval harness against
  these answers, a local model first.

## 2026-10-14 — cpc re-vendored 1.12.0 → 1.18.1, from the tag (T-002)

**What changed.** `cpc-init`, run with no `--profile` from a worktree of cpc's `v1.18.1` tag,
replaced the gitignored `tools/conventions/cpc/` (54 modules) and laid cpc's own launcher,
`tools/conventions/run.py` and `cpc.just`, beside this project's `rungate.py`, which stays. The
only tracked change is the version in the `scripts/conventions.toml` header; the local CONTEXT says
the same. T-002 is closed against this entry.

**Measured.** `cpc-verdict-diff --base v1.12.0 --head v1.18.1` over `docs_check`, `init_check`,
`sprint_check` and `integrity_check`: +0 new findings, −0 gone, and one new NOTE that changes no
exit code: the entry context is ~8,290 tokens and no `entry_max_tokens` cap is set. After the drop,
`run.py gates` (`docs_check --strict`, `integrity_check --strict`) 0/0, `init_check` 0/0,
`settings_doc --check-config` no findings. Nothing is listed under `[vendor] patched`, so nothing
was re-applied.

**Rejected.** `--profile standard`, which would lay every standard file the project left out. The
optional steps in cpc's migration notes are left for the owner; none changes a verdict.

**What it opens.**
- A token cap: `[budgets] entry_max_tokens = 8000` fails today at ~8,290, so history moves out of
  the local CONTEXT first.
- `[known_issues] archive_on_resolve = true`: no resolved section is live, so it can be switched on
  with nothing to move.
- `mod cpc 'tools/conventions/cpc.just'` in the `justfile`, and `run.py` in place of `rungate.py`
  once nothing calls the shim.
- Not proposed: `cpc-init --add hooks`, a `pre-push` hook. The push hooks were removed on
  2026-08-19 by the user's decision.
- `cpc-keypoint session-close` now fails while a cpc run that crashed or hit a usage error is
  unanswered (`friction note` or `friction mine`; the log is in `.git/`).

## 2026-10-09 (6) — Term signals: what the prose says about a term, read and offered, never applied (ROADMAP 93, ADR-053 decision 4)

**What changed.**
- **`knowledge/term_signals.py`** (new, pure, no model). Four things read off the body prose
  about a row of the vocabulary, each with the counts it rests on:
  - *spelled out / abbreviated* — a long form beside a short one, in either order, by Schwartz &
    Hearst's rule (2003). The long form is a substring of its sentence (ADR-043);
  - *the word beside it* — the content word most often joined to the label, before and after;
  - *a cited author* — mentions followed by "et al.";
  - *in prose* — mentions, and documents with one.
- **`scripts/term_signals.py`** — a runner that only reads. There is no `--apply`: a signal has no
  table and nothing applies one.
- **A review page**, "Term Signal Review": 109 cards on rows outside the 19 priority concepts,
  answered yes / no / can't tell. Private, not in the repository.

**Why.** Decision 4 of ADR-053: `dbs` and `pddl` should find their spelled-out forms from the
library itself, and a fragment or a cited name should say so with its reason. Restated against
ADR-054: what a signal finds is a form to *offer*; marking a form stays the user's act.

**On the working library** (104 documents, 46,558 prose sentences, 357 rows):
- The prose gives 1,080 short/long pairs, 80 of them in more than one document.
- 28 labels are spelled out and 13 are abbreviated: 51 offers, 47 of them forms the row does not
  have. `dbs` finds *deep brain stimulation* in two documents and `pddl` finds *planning domain
  definition language* in one; both are forms the user had already added.
- **Case is identity.** The library spells `CRE` out as *cAMP response element*. The concept is
  written `Cre`, so that long form is not offered to it; the same rule keeps *Character Error
  Rate* from `cer`, which the library writes `Cer` and follows with "et al." 13 times in 13.
- 18 one-word labels have the same word after them in half their mentions or more (`pose` then
  *estimation*, 313 of 480), 11 the same word before; 5 are followed by "et al." in half or more;
  31 are never used in body prose.

**Not known, and why nothing is shown in the app yet.** How often each signal is right. Schwartz
& Hearst accept any in-order spelling, so *deep learning approaches (DNNs)* is a pair. Each pair
carries `initials` (whether every capital of the short form lands on a word start or a capital of
the long one) as evidence; 10 of the 51 offers lack it. ADR-053 asks for a check outside the 19
before anything is shown, because the 19 are where the signal was noticed. That check is the
review page; the baseline and the panel follow the user's answers.

**Checked.** 23 unit tests, with sentences written in the library's shapes (a citation after the
short form, curly quotes round the long one): the pair is found in both orders, a cited name and
a remark in brackets are not pairs, the long form is verbatim, pairs are counted in documents, a
form the concept already has is marked known, a short form in another case is another word, the
inputs are untouched, an empty library gives every row its zeros. ruff, mypy, bandit and the full
suite pass (2,666). The page's store took a probe write and gave it back; the probe is deleted.
**Not checked:** the page saving one of the user's answers. That is read back after the first one.

**Rejected.**
- *Filtering pairs on `initials`.* It would drop *OrthoBackfill (OBF)* and *descending
  interneurons (dINs)*, which are right. Whether it helps is what the labels will say.
- *Offering `CRE`'s long form to `Cre` with a warning.* Decision 3 already says they are two
  words.
- *A table for signals.* Nothing chooses or dismisses one yet; they are recomputed from the prose.
- *Toggling an answer off on a second click* on the review page. The morning's Manage keywords
  lesson: a second click cleared a mark unseen. The page has a separate *clear*.

**Opens.** The user answers the cards; then the baseline, RG-032's second claim (do exact forms
find definition candidates the name misses), and the offers in the concept panel for the signals
the answers support.

## 2026-10-09 (5) — What the user's marks on the 22 forms did: four concepts lose 14 documents, 30 links become 25, the gap list keeps its 12 rows (RG-032, first claim)

**What changed.** No code. A baseline, `tests/eval/baselines/forms_marked_2026-10-09.md`, and the
pointers to it: an addendum to ADR-054 (its first ⚠ line now has figures), the ADR index, ROADMAP
row 94 and the presence row of `docs/knowledge-layer.md`.

**Why.** ADR-054 was accepted with one effect unknown "until the user classifies them". The user
classified the 22 forms on 2026-10-09 (10 broad, 12 exact, no rename) and rebuilt. Session 4's
signals propose forms into this classification, so its effect is recorded before they are built.

**What the marks did** (a rebuild with nothing marked against the graph the user rebuilt, one
reader):
- Four concepts lose documents, 43 → 29 between them: `knowledge distillation` 11 → 4,
  `contrastive learning` 13 → 10, `re-ranking` 11 → 9, `hard negatives` 8 → 6. Each of the 14
  pairs that went is reached through one named broad form and is listed beside its concept. Six
  documents carry no concept any more.
- Five of the ten broad marks move no document count.
- Links 30 → 25, and 17 of the 25 rest on fewer shared chunks. `passage retrieval` keeps its 7
  documents and loses the most: its one broad form, `passage ranking`, is written in 95 chunks of
  prose, 90 of them in one survey. Six links now rest on exactly the two shared chunks a link
  needs, four of them since the marks.
- The same 12 deterministic gap rows; three `unsourced_claim` rows rest on fewer answer claims.

**Checked.** Every figure in the baseline was recomputed from the two saved graph states before
the file was staged. That caught two errors in its first draft: "six of the ten broad marks" where
five are listed, and "four links on exactly two" where there are six.

**Limits.** One library, one reader's marks; nothing here says a mark is right. The marks were
applied together, so where both ends of a link have a broad form the loss is not split between
them. Both graphs were taken while figure chunks counted as text (entry (3)); the marked graph on
the prose reader has the same documents, links and gap rows, and the before/after was not redone
on it.

**Rejected.**
- *Editing the ⚠ line of ADR-054 in place.* The ADR is append-only; the addendum cites the
  baseline and the line stays as it was decided.
- *Redoing the comparison on the prose reader.* It needs every mark cleared on a copy of the
  library for three links that differ by one or two chunks.

**Opens.** RG-032's second claim (do exact forms find usable definition candidates the name
misses) is unmeasured and belongs with session 4's signals.

## 2026-10-09 (4) — The Graph tab's door to Manage keywords: one name, no change of tab, and a rebuild that does not wait to be offered

**What changed.**
- **One name.** The button added to the Graph tab that morning said *Manage concepts*; the
  empty graph's said *Choose the vocabulary*. Both open the view whose heading is *Manage
  keywords*, and both now say so. `GLOSSARY.md` U-001 pins it, with the rule behind it: a view
  has one name on its heading, on every control that opens it and in every sentence that sends
  the reader to it.
- **The view opens over the graph.** `curateVocabulary` (`App.svelte`) switched to the Library
  before opening the overlay, because entering the Library is what loads the two lists the view
  shows. It loads them itself now and leaves the tab alone; this covers the per-concept *Edit*
  too. Closing the view re-reads the graph when a write in it had dropped the graph's latch, so
  the notice naming what changed is there at once.
- **A standing *Rebuild*** beside it (`ConceptGraph.svelte`), left out while the stale notice is
  up, since that one says why.

**Why.** The user, on first use: the button should read *Manage keywords* — "I don't think it is
a good idea to use different names for the same thing, we need to be clear with this and have a
convention" — and clicking it "go[es] from graph to library". And after the reader fix of entry
(3) they were asked to rebuild and had nothing to click: the graph offers a rebuild when the
vocabulary has moved since it was built, and a change to how the library is *read* moves nothing
it watches. A standing rebuild had been an open question since row 94; this is the case that
answers it.

**Checked.** `svelte-check` 0 errors, 286 frontend tests pass. In the running dev app: from the
Graph tab the door opens the view with the tab still on Graph and returns to it on close; the
button reads *Manage keywords*; *Rebuild* is present with no notice up. The rebuild itself was
not clicked: it is the user's.

**Rejected.**
- *Keeping "Manage concepts" for the Graph tab's door because it opens on the Concepts section.*
  A name per entry point is two names for one view.
- *Telling the graph that the reader changed* (a reader version in the built graph, compared at
  read time). More honest than a button that is always there, and a backend change for a case
  that has occurred once. The standing control covers it.

**What it opens.** Nothing new. The rule in U-001 applies to the rest of the app and has not been
checked against it.

## 2026-10-09 (3) — The knowledge layer read figure chunks as the document's own text; it now reads prose (ROADMAP 97)

**What changed.**
- **`concept_skeleton.prose_parents`** (new, pure): child-row metadata → one entry per prose
  parent, a `chunk_type == "figure"` parent left out. `load_presence_inputs` returns what it
  keeps, so presence and co-occurrence, the written-form vote and the library-wide definitions
  scan all read prose.
- **`definitions.chunks_mentioning`** (the one-concept path): a figure's block in the keyword
  index neither picks a document nor comes back as one of its parents.
- **`epistemics.load_pc_parent_chunks`** is unchanged and now says why: it reads for what
  retrieval can return, and a retrieved figure needs its markers.
- Retrieval, both stores and the keyword index are untouched. No locked setting moved.
- `docs/knowledge-layer.md` (the presence row); ROADMAP 97, and row 95 restated on prose.

**Why.** A described figure is stored as a parent chunk of its own, after the document's prose:
its caption, the vision model's description, and a whole copy of the passage that cites it. That
is what retrieval wants. The knowledge layer's reader returned every parent, so it took those
chunks for text the document wrote, with three effects. A cited passage was counted twice, and a
link needs only two shared chunks. A model's wording could make a term present in a document that
never uses the word, where presence is meant to be decided by the text alone. And appended after
the prose, the figure text moved the References heading before the halfway mark the bibliography
cut wants, so the cut missed nine documents. Found while listing, for the user, how the library
uses the forms they were about to mark: one sentence showed twice in three documents, and the
extraction cache held it once.

**Measured** (`tests/eval/baselines/prose_parents_2026-10-09.md` — the committed code against the
change, dry runs on the working library with the user's marks in place):
- 615 figure parents in 83 of 104 documents, 6.6% of the text that was read; 181 prose parents
  stored more than once, one of them nine times.
- **The 13 concepts: the same documents, the same 25 links, the same 12 gap rows.** Three links
  rest on one or two fewer shared chunks, so the graph version changes.
- The 357 rows: 1,103 concept–document pairs → 1,079. The 24 are all terms, and each is a word
  found in a model's description and in no caption (`plateau` in 8 documents).
- Written forms: four move, all on terms.
- The bibliography cut fires on 85 documents, not 76. Section 5 of the baseline replaces section 7
  of `vocabulary_shape_2026-10-01.md`: what the cut wrongly removes after a reference list is 3.7%
  of the text in 32 documents, not 4.5% in 39.
- Definition candidates: 20 for the concepts and 110 over every row, the same sentences before
  and after.
- Five new tests; the two that go through the readers fail on the code before. Suite: 2,643
  passed.

**Rejected.**
- *Keep the caption, drop only the description and the copy.* The caption is already in the prose
  parent it was extracted with: no pair came from a caption alone.
- *Filter in each caller.* Presence, the vote and two definition paths are four places to forget;
  there is one reader.
- *A `chunk_type` column on the index's `parents` table.* It changes retrieval's index file for a
  rule the reader can apply from the block's own metadata.
- *Align the marker reader with this one.* A retrieved figure would lose its markers.
- *Take the copy out of the figure chunk.* It is what lets an answer read a figure inside the
  passage that argues from it.

**What it opens.** It applies at the next graph rebuild, which is the user's: the graph version
changes, three links lose one or two shared chunks, and the written forms are voted on prose.
ROADMAP 95 (a mention in a reference list stops counting) starts from the prose figures. Not
measured: retrieval returning a figure and the passage it cites together, so that an answer reads
that passage twice.

## 2026-10-09 (2) — The audit gates' second catch: four upgrades for advisories published since the last green run

**What changed.** Four versions, in two lock files, and nothing else.
- `uv.lock`: fsspec 2026.4.0 → 2026.6.0, multidict 6.7.1 → 6.9.1.
- `apps/desktop/package-lock.json`: source-map-js 1.2.1 → 1.2.2, dompurify 3.4.15 → 3.4.16.
  `package.json` is unchanged.

**Why.** CI on `b50941a` — a frontend change that touched no dependency — failed in both audit
steps (run 37933120201), on advisories published in the eight days since the last green run.
pip-audit reported two it had not seen: fsspec (CVE-2026-104851) and multidict
(CVE-2026-104874). `npm audit --audit-level=high` reported source-map-js at high
(GHSA-68fv-2mgg-jv7q) and listed two low ones on dompurify (GHSA-p98j-92pf-mc4p,
GHSA-6688-9rhm-gjv2). Lint, types and the tests passed; the secret scan, which runs after
pip-audit, was skipped.

**Where the four sit.** Nothing under `src/`, `apps/` or `scripts/` imports fsspec or multidict:
fsspec arrives through huggingface-hub and torch, multidict through aiohttp and yarl.
source-map-js is build-time only (vite → postcss). dompurify is the one direct dependency: it
sanitises rendered answers (`lib/chat/Markdown.svelte`), on a string, and both of its advisories
are about the `IN_PLACE` mode, which the app does not use.

**Read and measured.**
- `uv lock -P fsspec==2026.6.0 -P multidict==6.9.1`: each pinned to the smallest version that
  clears its advisory. 236 packages before and after, two version lines changed, no package
  added, removed or downgraded, no dependency edge added or removed, every changed artifact from
  `files.pythonhosted.org`.
- `npm audit fix`: two packages changed in the lock, six lines each way.
- The gates as CI runs them: `scripts.pip_audit_gate` — 196 audited, 5 reviewed, **0 unreviewed**;
  `npm audit --audit-level=high` — 0 vulnerabilities.
- The unit and integration suite passed on the upgraded packages: 2,643, in a tree that also held
  an unrelated change under `src/` with five new tests (CI runs this commit alone). Frontend: 286
  tests pass, `svelte-check` 0 errors. No trial freeze of the sidecar; the release sessions
  freeze anyway.

**Rejected.**
- *Reviewing the four into the ignore lists.* Each has a fix the lock takes without moving
  anything else.
- *An unpinned `uv lock -P <package>`.* On 2026-09-30 that pulled a major SDK version and three
  new packages for the same kind of fix.

**What it opens.** This is the second push in eight days that changed no dependency and turned
`main` red (the first was `a5bb634`, 2026-10-01). A blocking audit with nothing that looks between
pushes makes each push pay for what was published since the last one: the second data point for
S-12, the scheduled or alert-driven check.

## 2026-10-09 (1) — Manage keywords built 312,166 dropdown options on opening: a row now has one text field, a mark shows at once, and the Graph tab opens the view (ROADMAP 96)

**What changed.**
- **One field to add a form.** Each row of `LibraryManageKeywords.svelte` carried an
  "Add a form…" `<select>` listing every unassigned keyword. It is now a text field over one
  shared `<datalist>`: type to search the keywords, or type any form and press Enter.
  `formToAdd` (`library.ts`) refuses a blank, the row's own name and a form the row already has.
- **A mark shows when it is clicked.** The exact and broad buttons show the mark that is being
  saved (`shownBreadth`) and take no click until the server has answered. A save that fails
  leaves "not saved — click again" on the form. `setFamilyFormBreadth` (`App.svelte`) replaces
  the one row with the route's answer instead of fetching every family again, and resolves to
  whether the mark was saved.
- **The Graph tab opens the view.** A *Manage concepts* button beside the coverage sentence
  opens Manage keywords on its Concepts section (`ConceptGraph.svelte`, `openOnConcepts`).
- The × that removes a form sits a little apart from the two marks.

**Why.** The user marked the 22 forms of ADR-054 on 2026-10-09 and met three faults in an hour.
The view froze on opening. "Add a form" listed 1,229 uncurated keywords and could not add a
string that is not one of them, so a form removed by a slip (`contrastive`) had no way back
through the screen. And a mark looked unsaved for as long as the view took to redraw: a mark is
set by toggling, so the second click cleared what the first had set. Two attempts on one form
left no mark, and nothing said so. They also asked for a way into the view from the graph: "It
would be more logical to 'manage keywords' too in graph."

**Measured** — the dev app on the working library (104 documents, 254 rows shown), opened from
the Library's keyword filter with the family list already loaded:

| | Before | After |
|---|---:|---:|
| Elements in the page once the view is open | 322,887 | 11,958 |
| `<option>` elements | 312,166 (254 rows × 1,229) | 1,228 (one list) |
| Time the page is blocked by the click that opens it | 5.8 s | 0.1 s |

Measured in a browser pane that was not on screen, so the times are what the script cost, not
what a person waited. Opening makes no request of its own, before or after. 286 frontend tests
pass (two new), `svelte-check` reports 0 errors; at 375 px in the light theme and at desktop
width in the dark one, neither the new field nor the new button overflows.

**How the writes were checked.** With `fetch` replaced in the page so that no request other
than a GET could leave it: a slow save (the mark shows at once, both buttons are disabled, a
second click sends nothing), a failed save (the mark goes back and the message appears), a
retry (the message goes), and adding `DPR` to `dense retrieval` (one POST carrying that
string). The library file's write time did not move during the checks. That the route accepts a
string that is no keyword was shown the same day by a real call: `contrastive`, put back at the
user's word.

**Rejected.**
- *A lazily filled `<select>` per row.* Still a list of uncurated strings to scroll, and still
  unable to add a form that is not a keyword.
- *Marking forms inside the graph's concept panel.* A second place that writes the vocabulary,
  against ADR-017 A1. The graph gets a door to the one place.
- *A confirmation before a form is removed.* With any string addable again, a slip now costs one
  field and Enter. The removal itself is still silent.

**What it opens.** The fix for the figure chunks the knowledge layer reads as document text is
next. From the same hour of use and not done here: a wider left margin, a model picker in
Settings, and saying on the screen why *Add documents* is unavailable in a browser tab.

## 2026-10-01 (7) — The secret gate's first CI run was red, and right: on Windows the scanner had been skipping 47 files (S-14)

**What changed.**
- **The scanner reads every file as UTF-8, on both platforms.** `scripts/secret_scan_gate.py`
  starts the scanner with `python -X utf8`, and `.pre-commit-config.yaml` overrides the hook's
  `entry` to do the same (same repository, same rev).
- **`.secrets.baseline` records three more findings**, added by a scan run in UTF-8 mode: a fake
  id in `tests/unit/test_chat_controller.py` and two fake keys quoted in an archived DEVLOG entry.
  Its plugins, filters and existing entries are unchanged.
- The gate prints a finding's path with forward slashes, so a Windows run and CI name it alike.

**Why.** CI on `07f4b50` failed at the new secret-scan step with three unrecorded findings, while
the same gate passed here. `detect-secrets` opens a file in the platform's default encoding and
treats a decode error as "binary, skip". On Windows that encoding is cp1252, and a UTF-8 file
with one byte cp1252 does not define — a curly closing quote is enough — is skipped whole, with
no message. Linux read every file. So the hook that runs before every commit on this machine had
never looked inside those files, and the gate's "837 files scanned" was not true here.

**Measured.**
- 47 of 838 tracked files are UTF-8 text that cp1252 cannot decode: 20 under `apps/`, 16 under
  `docs/`, 5 tests, 3 under `src/`, 2 data files, 1 at the root; 28 are code or tests.
- The scanner's plugins match all three lines when handed them directly; its file scan reported
  none of them in the default encoding and all three with `-X utf8`. Line endings and path
  separators make no difference.
- After the change the gate gives the same verdict on Windows and in a Linux run (WSL) of this
  tree: 838 files, nothing unrecorded. The suite: 2,638 passed.
- The reconfigured hook fails on a scratch file holding a curly quote and the published example
  key, names the file and not the key, and passes over every tracked file.

**Rejected.**
- `# pragma: allowlist secret` on the two DEVLOG lines: the archive is kept verbatim.
- Setting `PYTHONUTF8=1` for the whole machine: it changes every Python program on it, and the
  repository's rule is an explicit encoding where the file is opened.
- Replacing the upstream hook with a local one in the project's environment: the `entry`
  override keeps the pinned rev and changes one thing.
- Leaving the local hook as it was because CI now catches it: CI runs after the push, and in a
  public repository that is after the key is out.

**What it opens.**
- Until this is pushed, `main` is red.
- The same default hides in any tool that opens a file without naming an encoding. The
  repository's own code is covered by its encoding rule; third-party tools are not, and the way
  this one was found was a second platform disagreeing with the first.
- A baseline entry is a hash at a line: the three new ones are fake values, read before they were
  recorded.

---

## 2026-10-01 (6) — The test suite runs on a data directory of its own (KI-60 closed)

**What changed.**
- **One redirect for everything.** `tests/conftest.py` sets `DOC_DATA_DIR` to a fresh, empty
  directory before `doc_assistant.config` is first imported: `.pytest-data/run-<pid>-<token>/`
  in the repository (gitignored), made for the run and removed when it ends. The library
  database, the vector stores, the extraction cache, the session logs, the settings file and the
  graph sidecars all resolve inside it, in the test process and in any subprocess a test starts.
- **It fails closed.** If `config` was imported before the conftest could set the variable, the
  run stops before any test and names what `config` resolved.
- **Cleaning up.** At the end the default database engine is disposed and every Chroma system the
  run opened is stopped, then the directory is removed. A directory left by a killed run is swept
  by a later run once it is a day old. `DOC_TESTS_KEEP_DATA_HOME=1` keeps the directory and prints
  its path, to see what the tests wrote.
- The session fixture of entry (5) is gone: the default engine is bound inside the run's
  directory by construction.
- `tests/unit/test_suite_data_home.py` (14) pins the mechanism: no path in `config` points into
  the repository's `data/`, each default store is inside the run's directory, a subprocess lands
  in the same one, a run whose `config` was imported first is refused, and only a stale run
  directory is swept.

**Why.** Entry (5) fixed the database and measured what was left. The same run-by-run debris had
been landing in the working library for months: 945 of the 946 folders under
`data/cache/referenced/` are named after test fixtures (`a.md`, `b.md`, `gone.md` …), and the
4,740 timestamped session logs in `data/exports/` include every test run's (about 37 each). None
of it failed a test, so nothing would have stopped the next thing a test wrote.

**Measured.**
- **Before the change, with the data directory pointed at an empty folder:** the suite passes
  unchanged (2,623 tests) and writes 76 files there — 38 extraction-cache files in 19 folders,
  37 session logs, and one vector store. `tests/unit/test_wiki.py` is the test that opens the
  default store, which creates it when it is absent.
- **After it:** every file under `data/` except `sources/` and the backups, compared by size and
  modification time before and after a full run — 7,815 files, none added, removed or changed;
  the working database and both vector stores keep their timestamps. 2,636 tests pass (coverage 92.7%), and `.pytest-data/` is
  gone when the run ends.
- **Chroma holds its file.** On chromadb 1.5.9 a directory with an open store cannot be removed
  on Windows, and dropping the client does not release it; stopping the cached systems does.

**Rejected.**
- The system temp directory. `config` moves the vector stores to `%PROGRAMDATA%` when the data
  path is not ASCII (KI-11), and the temp path of a Windows account with an accented name is not:
  the tests' stores would land in the machine-wide directory the installed app uses, outside
  anything a run removes. In the repository the path is ASCII whenever the checkout is.
- `os.environ.setdefault`: a developer who exports `DOC_DATA_DIR` to run the app against a library
  elsewhere would hand that library to the tests.
- Keeping the session engine fixture beside the redirect: two mechanisms for one guarantee.
- Failing the session when `data/` changed during it: it would fail whenever the app is used
  while the tests run.
- Changing `test_wiki.py` so it stops opening the default store: it is contained now.

**What it opens.**
- The debris earlier runs left in the working `data/` is still there; removing it is the user's
  call.
- ROADMAP 76's shared-fixture item is done, and with it KI-58's second open half: a test can no
  longer pass on this machine because the working library exists. The Windows CI job is open.
- `wiki.sample_chunks` creates an empty vector store when it reads a missing one — a read with a
  side effect, left as found.

---

## 2026-10-01 (5) — The test suite no longer reaches the working library's database (KI-60)

**What changed.** `tests/conftest.py` gains a session fixture that binds the database layer's
default engine to a throwaway file for the whole run. A test that reaches the database without
swapping the engine now lands on an empty database, as it does on a fresh checkout.

**Why.** `db.session` binds its engine to the configured `library.db` at import. Two route test
files (`test_reingest_routes.py`, `test_ingest_progress.py`) enter the app's lifespan without
redirecting it, and the lifespan's first act is the schema migration. On 2026-10-01 a full run
therefore added row 94's new column to the working library, before the app had ever been started
on that code. Read back afterwards: integrity check `ok`, no form marked, and 593 labels and 404
alias rows identical to the backup of 2026-09-01 — the column is the only change, and it is the
one the app would have made at its next start. Reproduced against an empty data directory:
before the fixture those two files create a `library.db` there, after it they do not.

**Rejected.**
- Stubbing the migration in the two files: it fixes today's two and leaves the next unredirected
  test free to do the same.
- Pointing the whole suite at a temporary data directory (`DOC_DATA_DIR`): the right end state,
  but two tracked files live under `data/` and which tests depend on that directory is unmeasured.
- A session-end check that the working database's modification time did not move: it would fail
  whenever the app is used while the tests run.

**Measured after the fix** (one full run, every file under `data/` compared before and after):
`library.db` untouched. The run still added 58 extraction-cache files for the tests' own documents
(`cache/referenced/`), two session logs under `exports/` and one row in the vector store's
write-lock table; the store's content is unchanged (18,011 embeddings).

**What it opens.** This is the database half of ROADMAP 76's shared-fixture item, and of KI-58's
second open half. The debris above is KI-60's open part: nothing the library holds changes, but it
accumulates, and a test that can open the working vector store could write to it. A temporary
data directory for the suite would close it, and belongs to row 76.

---

## 2026-10-01 (4) — A concept has a name and exact or broad forms, and the other rows are terms (ROADMAP 94, ADR-054); CI's secret scan can fail (S-13)

**What changed.**
- **A form is exact or broad.** One additive column, `concept_aliases.breadth`; unset reads as
  exact. `concept_skeleton.load_concepts()` — the loader that presence, edges and the gap list's
  claim attribution count through — leaves a broad form out. `load_broad_forms()` and
  `match_broad_presence()` match it on its own, and a node carries the documents only a broad form
  reaches (`broad_doc_ids`, `broad_forms`) beside `doc_ids`, never in them. The name is always
  exact: a broad mark on an alias that repeats the label is ignored.
- **Concepts and terms.** Stored definition candidates, merge suggestions, `is_a` proposals and
  field-placement proposals read concepts (rows with `graph_include`). In all four runners
  `--include-terms` reads every row and cannot be applied (for placement it replaces
  `--all-concepts`, kept as an alias). A term refuses a definition write (409) and returns the
  sentences found for it without storing them.
- **The name on every screen.** `written_forms.shown_labels` is the one helper. The vocabulary
  search, the gap list, the taxonomy view and Manage keywords carry `written`, as the graph did.
- **Manage keywords** lists concepts apart from terms. A concept's row has an exact | broad control
  per form. Delete asks first, says what goes with the row (forms, the chosen definition and the
  other options, field placements, gap decisions, the graph's documents) and offers "Take off the
  graph instead". The graph's Edit button, and a term's Manage keywords button, open the view on
  that row.
- **The graph says what it is behind on.** A build records a fingerprint of each concept's name
  and forms in `skeleton.meta["forms"]`; the view compares it with the live vocabulary and names
  the concepts that changed. A graph built before the record says it cannot tell, and offers the
  rebuild.
- **S-13.** CI's secret step is `python -m scripts.secret_scan_gate` (`just secret-scan`): the
  `detect-secrets` hook over every tracked file against a temporary copy of the baseline. It fails
  on a secret the baseline does not record, never rewrites the tracked baseline, makes no network
  call, prints a finding's type and place and never its value, and exits 2 when it could not look.
  It is stricter than the local hook in one way: the hook can ask a provider whether a candidate
  key is live and drop it if not, and the gate keeps it. It adds about 19 seconds to a CI run.

**Why.** ADR-054, accepted 2026-10-01: a count should say what it rests on, and until now it could
rest on a row nobody read or on an alias that means something wider. S-13: the step CI ran,
`detect-secrets scan --baseline`, is the command that writes a baseline — it recorded a new finding
and exited 0 (found 2026-09-30).

**Measured.**
- **Nothing moved.** `build_concept_skeleton(apply=False)` on the working library with the
  committed code and with this change: the same graph version (`cb99a7f9f36545da`), 13 concepts,
  30 edges, 86 concept–document pairs, equal field by field
  (`tests/eval/baselines/names_and_forms_2026-10-01.md`).
- **One mark, on a copy of the library.** `distillation` marked broad: `knowledge distillation`
  11 documents → 4, with 7 listed beside; no edge lost, six of its edges on fewer shared chunks;
  3 documents leave the graph's coverage. The graph named the concept until it was rebuilt. This
  is one mark to exercise the path. The user's reading of the forms has not happened.
- **Terms.** `viral` and `SPECTER` opened in the app on the copy: usage passages for both, two
  defining sentences for `SPECTER`, no control to choose one, and no row stored for either.
- **The count to classify is 22, not 31.** The 13 concepts hold 31 alias rows; 9 repeat the name.
- **S-13.** In a scratch repository the published example key fails the gate (exit 1) and passes
  the command CI ran before (exit 0, the key written into the baseline). The repository passes:
  834 files, about 19 seconds.
- Checked live on the copy in light and dark and at 375 px: no failed request after a clean load.
- The suite: 2,622 passed, coverage 92.8%. The gate's test file gained one test and had one
  rewritten after that run, and passes on its own (24). The desktop's 284 tests and its type
  check pass.

**Rejected.**
- Adding a broad form's documents to presence behind a flag: edges and gaps would count them.
- `graph_version` as the "forms changed" signal: it fingerprints the result, and the gap and
  epistemics sidecars key their own staleness on it.
- Calling a graph built before the record "changed": there is nothing to compare, so it says that.
- Deleting a row that has nothing attached without asking: one rule for every delete.
- For S-13, failing on `git diff --exit-code .secrets.baseline` after the old scan: it turns the
  scan's side effect into the signal, and a recorded line that only moved would fail it.
- Running the hook on the tracked baseline in CI: the hook rewrites the file it is given.
- Naming the `just` recipe `secrets`: the scanner reads that name and the command under it as a
  keyword with a value, so the commit hook failed on the recipe (and on a sample report line in
  the gate's own test). The recipe is `secret-scan`.

**What it opens.**
- The user's part: mark the 22 forms and name the concepts in Manage keywords, then rebuild.
  RG-032 is measured after that, before ADR-053 decision 4's signals (the next session).
- The working library's graph predates the record, so it will show the "cannot tell" notice until
  its next rebuild. That rebuild also applies the case-aware matching of 2026-09-30 (`cre` 7 → 6).
- The Graph tab offers Rebuild only when it reports itself behind. A standing control is a design
  choice left open.
- Found on the way: the test suite migrated the working library (the next entry, KI-60).
- `docs/specs/feature-tag-families.md` calls the hidden group "glossary-only"; the view says
  "unused" now, and the spec carries a dated note.

---

## 2026-10-01 (3) — ADR-054 accepted, with the user's amendment: a base vocabulary per field is a source of terms

**What changed.** No code.
- **ADR-054 is accepted** (user, 2026-10-01). It gains *Amendment 2026-10-01*, and its Status and
  its "to decide" line now say what was accepted. A curated base
  vocabulary for a field is a third source of rows beside the user's additions and the keyword
  extractor; its entries stay terms until the user takes them on. One sentence of the Decision is
  widened: a term is a candidate row nobody has taken on, whatever proposed it, because a base
  entry may not occur in the library at all.
- **ROADMAP KL2** names it as the concept-level half of "the taxonomy as the reference class for
  expected coverage"; row 94 (ADR-054's build) moves from *gate first* to *planned — next*; the
  index line in `docs/decisions.md` follows.

**Why.** The user, on reading ADR-054: a curated default vocabulary for a given topic "would go
hand in hand with the taxonomy"; otherwise "the text is clear". With the amendment in front of
them the user accepted the ADR: "Okay, let's go with this". The idea continues the user's
question of 2026-09-21 (experts have already listed and defined a field's terms), which ADR-053
answered for definitions only. It also supplies what `docs/knowledge-layer.md` §1 says a gap needs:
an expected structure to deviate from. The taxonomy says which fields exist; a base vocabulary says
what a field contains.

**Rejected.**
- Making a base list's entries concepts by default: that is the bulk promotion of 2026-07-05 with
  a better source, and ADR-054 exists because 344 rows became concepts without being read.
- Importing a vocabulary whole: ADR-028 decision 7 measured it as "a facet that partitions
  nothing" (30,000 MeSH descriptors for a small library).
- Rewriting ADR-054's Decision in place: the file is append-only; the amendment states the one
  widened sentence.
- Designing the feature here: which vocabularies, their licences and sizes, and whether a topic's
  list is taken on entry by entry or as a reviewed whole belong to their own ADR.

**What it opens.** The ADR-032 grill (KL2) decides what the base list is for and how it is taken
on; 93c's local copies are its data step. Known limit before any build: of the 19 priority
concepts, expert vocabularies define 8, gloss 6 and have nothing for 5
(`tests/eval/baselines/reference_vocabularies_2026-09-21.md`), so a base list is solid for the
established fields and thin for the youngest ones. The acceptance opens row 94: names and forms on
the 13 concepts. CI on `7357fcc`, the commit that carried the dependency fix and ADR-054 as
proposed, is green.

---

## 2026-10-01 (2) — The vocabulary is read against the library; concepts and terms, names and forms (ADR-054, proposed)

**What changed.** No code. Three documents:
- **`tests/eval/baselines/vocabulary_shape_2026-10-01.md`** (new): a read-only snapshot of the 357
  text-bearing concept rows against the library's text — where they came from, how far each
  spreads, what the words beside a one-word label are, what each alias of the 13 graph concepts
  contributes on its own, and what the bibliography cut removes.
- **`docs/decisions/ADR-054-concepts-and-terms-names-and-forms.md`** (new, proposed) and its index
  line: a *concept* is a row the user has taken on (`graph_include`, 13 today) and the other 344 are
  *terms*; a concept's label is its name, and each matched form is *exact* (counts as presence) or
  *broad* (counted beside it). One additive column, no second table. Not built.
- **`docs/ROADMAP.md`:** row 94 (ADR-054's build) and row 95 (the bibliography cut, body-text
  presence); row 93's abbreviation signal moves after 94.

**Why.** The user, before the abbreviation and fragment signals of ADR-053 decision 4 were built:
"we will need to think more about vocabulary". The open items — a label shorter than its concept, a
homograph, an alias that means something else, a label that is an author's surname — turned out to
share one cause: nothing in the data says whether a row is a meaning someone chose or a string an
extractor produced. The four choices were put to the user one at a time, each with a recommendation
and rows from the library, and the user took the recommended option each time.

**Measured** (104 documents, 8,861 parent chunks; the shipped matcher; no model):
- **Provenance.** 13 rows are hand-made and are the only ones on the graph. 344 were created in
  one promotion on 2026-07-05; 113 of them are a keyword of no document in today's extraction.
- **Spread.** 197 of the 344 occur in the prose of at most one document, 30 in none; 18 occur
  nowhere in the text (`comput vis`, `koonce emerson`).
- **Neighbours.** Of 161 one-word labels with at least 5 prose mentions, 22 are followed by the same
  word in at least half of them: 16 are the front of a longer term (`pose` → "estimation" 326 of
  498) and 5 are surnames followed by "et al.". The user's three examples do not resolve this way:
  "viral vector" is 7 of 103 uses of `viral`.
- **Forms.** `distillation` alone reaches 7 of the 9 documents counted for
  `knowledge distillation`. 89 of the 99 `passage ranking` mentions are a benchmark's name. "Cre
  recombinase" occurs once against 255 for `Cre`.
- **The cut.** 113 of 1,103 concept–document pairs exist only past the bibliography cut. The cut
  drops everything after the References heading: in 39 of 76 cut documents that includes content
  sections (appendices, methods, sections emitted late by the PDF extractor), about 555,000
  characters or 4.5% of the library — text the keyword extractor, the definition scan and the
  written-form vote never see.

**Decided by the user.** (1) The 13 are the concepts; the 344 are terms. (2) Name plus exact and
broad forms. (3) The names-and-forms build comes before decision 4's signals, which then propose
into it. (4) Reference-list mentions stop counting as presence after the next release, in one
measured change with the cut's fix; the release notes state the limit.

**Rejected.**
- Keeping all 357 as concepts and cleaning them with signals: 344 reviews to reach the state the 13
  already have, with the meaning features running over unread strings meanwhile.
- A second table for terms: ADR-018 reserved that for vocabularies that differ in shape, and the
  difference found is membership, which its flag already records.
- A display name with matching unchanged: `distillation` would keep counting in full.
- Fixing the cut before the release: it moves keywords, definition candidates and written forms
  at once.
- Writing the snapshot's numbers into the ADR alone: a decision file cites its evidence, so the
  counts went into a baseline a reader can check.

**What it opens.** Row 94 is gated on the user accepting ADR-054. The 31 aliases of the 13 need
the user's exact-or-broad call, and the effect on presence, edges and gaps is unmeasured until then.
The snapshot's counts are observations, not graded signals: whether a neighbour share or an "et al."
share separates a term from a concept needs the user's labels on a sample (decision 4). Found on the
way: Manage keywords deletes a concept row and its definitions with no confirmation — in row 94.

---

## 2026-10-01 — The pip-audit gate's first catch: three upgrades for eight advisories published overnight

**What changed.** `uv.lock`: sentence-transformers 5.5.1 → 5.6.0 · urllib3 2.7.0 → 2.8.0 ·
virtualenv 21.3.3 → 21.7.13, and python-discovery 1.3.1 → 1.6.1, which virtualenv 21.7.13
requires. Each named package is pinned to the smallest version that clears its advisories
(`uv lock -P <package>==<version>`). Read package by package: 228 packages before and after, none
added, removed or downgraded, every artifact from `files.pythonhosted.org`, and no new dependency
edge (python-discovery drops its `platformdirs` edge).

**Why.** CI on `a5bb634` (the written-case commit below) failed in the pip-audit step and nowhere
else: eight advisories were published after the gate went blocking the day before (2026-09-30 (2)).
One is in sentence-transformers (CVE-2026-68770: `trust_remote_code=False` is bypassed when the
model path exists on disk, so Python files inside a model directory run at load). Three are in
urllib3 (an HTTPS proxy's TLS settings mixed with the target server's; two ways a server can stall
or bloat a streamed response). Four are in virtualenv, which is here only through pre-commit (a
downloaded wheel was not verified; a prompt or a path could reach `pyvenv.cfg` and the activation
scripts unescaped). All eight have a fix release, so none went into `pip-audit-ignore.toml`.

**Measured.** The gate on the tree before the change: 8 unreviewed advisories, exit 1 — the list
CI printed. On the upgraded tree: 196 packages audited, the 5 reviewed ignores matched, 0
unreviewed, exit 0. **The embedder and the reranker give identical output on both versions:**
through the app's own factories (`embeddings.get_embeddings`, the `CrossEncoder` the pipeline
builds), on the CPU, 8 texts and 1 query embedded with bge-base and 8 query–text pairs scored with
bge-reranker-base differ by at most 0.0 between sentence-transformers 5.5.1 and 5.6.0. A second run
on 5.5.1 also differed by 0.0, so the comparison is repeatable. The unit + integration suite on the
dev venv: **2,561 passed, 0 failed**, coverage 92.7%.

**Not done.** No trial freeze. The sidecar spec collects sentence-transformers whole
(`collect_all`, `scripts/doc_assistant_api.spec`), the lock adds no package and no edge, and the
release runbook freezes and smoke-tests the bundle before any release. CI's CPU venv was not
rebuilt locally; the push's CI run is that check.

**Rejected.**
- Reviewing the eight into the ignore file: each has a fix the lock takes without moving anything
  else.
- An unpinned `uv lock --upgrade-package`: on 2026-09-30 it pulled anthropic 1.9, langsmith 0.14
  and three new packages for the same kind of fix.

**What it opens.** The gate reads a live advisory database, so `main` can turn red on a push that
changed no dependency — here one day after the gate became blocking. That is what S-8 asked for,
and the red run was seen only because the session looked (KI-58's shape). S-12 (Dependabot pull
requests, and the alerts toggle) is the planned control that would raise a new advisory before a
push finds it; this run is the first data point for that call.

---

## 2026-09-30 (3) — A label keeps its written case (ADR-053 decision 3); the API refuses a foreign Host (S-5)

**What changed.**
- **`knowledge/written_forms.py`** (new) + the derived table **`concept_written_forms`**: the full
  skeleton build votes, for every surface form of every text-bearing concept, how the library
  spells it — whole-word uses in body prose (`definitions.document_sentences`, now public), not at
  a sentence start; each document votes for its most-used spelling, most documents win. Stored
  beside `Concept.label`, never into it (ADR-043; a build test holds the label byte-identical), and
  on each `ConceptNode` as `written` (in `skeleton.json` only when set).
- **One matcher for all four callers** (`concept_skeleton.form_matcher`): presence, the gap list's
  claim attribution, definitions (passages and usage examples) and epistemics. A form written in
  lower case matches in any case, exactly as before; a form written with a capital matches only
  spellings that differ from it in **word-initial** letters (`word_case_key`). `surface_forms` is
  public now; substring mode, the RG-008 A/B lever, ignores written forms.
- **Display:** the graph payload carries `written`, and the Graph tab shows it on the nodes, the
  panel heading and the rail (`lib/graph/labels.ts`, tested).
- **S-5, the host guard:** `create_app` adds `TrustedHostMiddleware` last, so it runs first; hosts
  from `DOC_API_ALLOWED_HOSTS`, else `127.0.0.1,localhost`; a blank setting means loopback, never
  every host. `tests/conftest.py` allows the `TestClient`'s `testserver`; `docker-compose.yml` and
  `.env.example` name the setting.
- `GLOSSARY.md` C-013 *written form*; ADR-053 amendment 2026-09-30; `docs/knowledge-layer.md`
  (presence row); `docs/security.md` (S1, S-5, floor row 6); ROADMAP rows 60 and 93.

**Why.** The user's labels (2026-09-22): "din is not the same as dIN. Important of being
case-sensitive." Presence feeds `single_source`, the trust table's one trustworthy gap signal, so
two words sharing a lower-cased label were being counted as one concept. S-5: the API listens on
loopback, but DNS rebinding lets a web page reach it under a hostile name (S1, T2).

**Measured** (`tests/eval/baselines/written_forms_2026-09-30.md`, read-only, the working library):
367 written forms, 155 with a capital. Presence moves for **16 of 357 concepts, all losing
documents**: nine lose another word, a surname or OCR noise (`CRE` from `cre`, "vitamin Din" from
`din`, the numpy paper's author Colbert from `colbert`, "StS" initials from `sts`), seven lose the
same name lower-cased, mostly in bibliography titles (`gpt-4`, `deeplabcut`). On the graph only
`cre` moves, 7 → 6 documents. **The 24 labelled definition candidates are unchanged (11 usable)**;
the definitions layer's one change is that "vitamin Din" is no longer a usage example of `dIN`.
Nothing moves in the stored graph until the user rebuilds. S-5 checked live: the running API
answers a foreign `Host` with 400, both loopback names with 200, and the desktop dev app loads
through it. The Graph tab showed `Cre`, `DBS`, `Ntsr1` with the graph response rewritten in the
page, since the stored skeleton predates written forms.

**Rejected.**
- The most-used spelling matched exactly (R1, 27 concepts move): `assistant` fell from 10 documents
  to 4 and `plateau` from 12 to 2 — ordinary words capitalised just over half the time.
- Occurrences voting: one paper repeating `PERSONA` outvoted the three that say `persona`.
- Exact spellings for anything but a capitalised word (R2, 20 move): Title Case quoted from titles
  won the vote, and prose such as "a sentence encoder LSTM" stopped counting. The word-initial rule
  restores those four and keeps every correct split.
- Rewriting `Concept.label` to the written form: ADR-043 — derived data sits beside curated data.
- Deriving the written form inside each caller: epistemics sees one text at a time and the usage
  examples read six documents, so each would vote differently. One stored vote, one rule.
- Showing the written form everywhere in this change: vocabulary-search hits and the gap list read
  their label server-side from other payloads; left as a follow-up.

**What it opens.** The seven remaining losses mostly come from reference lists: presence reads the
whole text, the vote only body prose. Matching body text only would remove them, and would also
stop reference-only mentions counting — its own measurement. ADR-053 decision 4, the abbreviation
signal, now has its input (141 labels with a capital). The user's rebuild applies the 16 changes;
the baseline lists each one with what stopped counting.

---
