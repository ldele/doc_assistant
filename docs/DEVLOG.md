<!-- status: active · updated: 2026-10-20 · class: append-only -->

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
> **2026-09-30 (3) → 2026-10-09 (3)** in [`docs/archive/DEVLOG-archive-008.md`](archive/DEVLOG-archive-008.md)
> (rotated 2026-10-17: 11 entries) ·
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
