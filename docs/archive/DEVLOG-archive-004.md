<!-- status: archived · updated: 2026-08-28 · class: append-only -->

# DEVLOG — archive 004 (2026-08-05 (1) → 2026-08-07 (6))

Older entries, moved verbatim from `docs/DEVLOG.md` on 2026-08-28 so the working log stays about
recent work. Newest-first, same format, unedited. Rotated because the live log had reached
4,040 lines against the 4,000-line cap in `tests/unit/test_doc_sizes.py`, which fails before it
can grow further. Cut on a date boundary so no day is split across two files.

---
## 2026-08-07 (6) — an eval run now records the settings that produced it (RG-026's precondition)

**What changed.** New `eval/run_settings.py` — `run_defining_settings()`, a snapshot of the 13
config values that determine what a run measures (the six chunk sizes, the embedder, and
`use_parent_child` / `use_multi_query` / `top_k` / `candidate_k` / `bm25_weight` /
`rerank_candidate_cap`). `Store` takes a `settings_provider` and merges the snapshot **under** the
caller's `config`, so an explicit per-run override (`run_eval --bm25-weight`) still wins — the
recorded value is always the one that ran. New `Store.run_config(run_id)` so a past run can be
audited at all. `scripts/run_eval.py` wires it; 9 tests.

**Why.** KI-41: the 2026-06-06 chunking sweep swept one configuration six times and **nothing in
the run record could contradict its notes** — `config_json` held `embedding_model` / `n_cases` /
`scorers` and no chunk sizes. That is why it took two months and an unrelated investigation to
find. RG-026 makes recording the varied settings the precondition for re-running the sweep, and
this is that precondition.

**The design was wrong first, and the suite said so.** The obvious shape — merge the snapshot
inside `persist_run`, importing `run_defining_settings` at the top of `store.py` — makes it
impossible for a runner to forget, and I built that first. It fails
`test_eval_harness_isolation.py`: `doc_assistant.eval` must import **no** app wiring, because the
harness is designed to be lifted into a standalone repo (ADR-003 Decision 8), and `run_settings`
reaches into `config` by nature. Injection is the shape that satisfies both — the coupling sits at
the project's edge of the harness, and a lifted copy drops one file.

**The cost of that is real and is written down where it bites:** a new runner must remember the
argument, so the rule is in `scripts/CLAUDE.md` and in both docstrings. A guarantee enforced by
construction would have been better; the extractability contract is worth more, and it is
test-enforced while "remember the argument" is not.

**Rejected.**
- **Backfilling old rows** with today's config values. Their geometry is genuinely unknown —
  substituting a plausible value would convert "unrecorded" into a false record, which is worse
  than the gap. `run_config` returns exactly what is there, and its docstring says to read with
  `.get()` and report "not recorded".
- **Recording every config knob.** A record nobody trusts to be minimal is a record nobody reads.
  The membership rule is "changing it changes what the run measures", so worker counts, caches and
  the lazy reranker are deliberately absent.
- **Surfacing the settings in `format_run_summary`.** Worth doing when someone is comparing runs;
  it is a different change and would not have caught KI-41 (the sweep printed per-config notes
  that *looked* right).

**What it opens.** RG-026's precondition is met, so a chunking re-run would now be self-auditing —
still a GPU-day, still the user's call. Nothing reads the new fields yet: a `sweep`-side assertion
that the arms actually differ (fail loudly when two grid points record identical settings) is the
natural next guard, and it is the one that would have turned KI-41 into a first-run error.

---
## 2026-08-07 (5) — `.env` stops beating the environment (KI-38), and the chunking sweep turns out to have measured nothing (KI-41)

**What changed.** `config.load_dotenv(override=True)` → `config._load_env()`, whose rule is: **a
non-empty process environment variable wins; `.env` fills in the absent and the empty.** 7 tests in
`tests/unit/test_config_env_precedence.py`. Corrections to the record where it now reads false:
the locked-settings chunk-size row (`.claude/CONTEXT.md`), `evals/README.md` § Chunk sizes, and a
dated correction appended to the 2026-06-06 baseline.

**Why.** KI-38 recorded the override as a credit leak — `LLM_PROVIDER=ollama <cmd>` runs on
Anthropic and bills. That was true and it is the smaller half. The override applied to **all 19 keys
`.env` defines here**, and `.env.example` ships the chunk sizes uncommented, so
`scripts/sweep_chunking.py` — whose entire mechanism is passing `PARENT_CHUNK_SIZE` /
`CHILD_CHUNK_SIZE` to an ingest subprocess — had no effect on the thing it sweeps.

**The finding, and it is measured rather than argued.** `case_results.token_input` scales with the
evidence block, so a parent size of 1000 vs 3000 must move it. Across all **18** runs of the
2026-06-06 sweep it does not move at all: mean **4326.7**, min **3582**, max **5106**, in every
config — and **identical per case** between the control and parent 3000, all ten. The field is live
elsewhere in the same DB (4372.7, 4615.8). **The sweep that closed the "defaults never measured"
caveat compared one configuration with itself six times**, at a cost of ~6 full corpus re-embeds.
The chunk-size lock is back to *unmeasured* — not wrong, unsupported.

**One free result out of it.** Those six configs report `contains_all` 0.906–0.933 and `llm_judge`
3.793–3.951 **on identical inputs**, which makes that spread a direct reading of the harness's noise
floor at n=3 on the public 10. The baseline's own text called it "within the trial-to-trial noise
bands" — right, for a reason it could not have known.

**Rejected.**
- **The fix the KI proposed** — threading an explicit provider argument through the answer path. It
  addresses the provider symptom and would never have reached `PARENT_CHUNK_SIZE`; the defect is in
  config loading, so that is where it is fixed. One place, one rule, whole class.
- **Dropping `override=True` outright**, which the KI correctly ruled out: it re-opens the
  empty-`ANTHROPIC_API_KEY` shadowing the original comment describes. Narrowing the override to
  exactly the empty case keeps that protection — and it is what the comment already justified, so
  the code now does what its own reason says.
- **Changing `REVIEWER_PROVIDER_PINNED`.** The pin is ADR-011 U1c and deliberate. It stays; what
  changes is that `REVIEWER_PROVIDER=ollama` in the environment now *works*, so the residual partial
  leak has a cure instead of being unreachable.
- **Re-running the sweep now.** It is a GPU-day of re-embedding and a deliberate call, not a
  cleanup — and it should not run until it records what it ran.

**What it opens.** `runs.config_json` stores `embedding_model` / `n_cases` / `scorers` and **no
chunk sizes** — an experiment that does not record the setting it varies cannot be audited, which is
exactly why this survived two months. Recording the varied settings is the precondition for any
re-run. Unknown whether other env-driven experiments were affected: the two checked are clean —
`sweep_bm25_weight.py` passes weights in-process, and the 2026-06-04 embedder A/B demonstrably took
its `EMBEDDING_MODEL` override (its arms differ), which is consistent with it having run on the
retired CPU box under a different `.env`.

---
## 2026-08-07 (4) — the Windows encoding rules are written down as rules, not as one runbook's war story

**What changed.** One rule, four homes: `.claude/CONTEXT.md` non-negotiable **#9** (canonical text),
an `AGENTS.md` digest bullet (the only version a fresh clone gets — `.claude/` is local-only), a
`docs/setup.md` § *Windows: text encoding* table (contributor-facing), and a line in
`scripts/CLAUDE.md`, which is where the console rule actually bites. `docs/desktop-packaging.md`
§5 trap 1 now points at the general rule instead of standing alone.

**Why.** All three defaults were already known and each had cost a run, but each was recorded only
where it was discovered: the PowerShell-5.1-reads-BOM-less-UTF-8-as-ANSI trap lived inside the RG-012
sandbox runbook (so it read as a sandbox quirk rather than a Windows one), the `sys.stdout`
reconfigure existed as a copied header in **36** files (all 36 correctly `hasattr`-guarded, checked
while writing this) with the reason recorded nowhere, and the
`encoding="utf-8"` convention on file I/O was written down nowhere at all. A convention that lives
only in existing code is one refactor from being dropped as noise.

**The part worth carrying: none of this is gate-visible.** CI is Linux, pytest captures stdout
through its own UTF-8 buffer, and ruff's `PLW1514` is not in `select` — so all three failures are
green-suite failures. That sentence is now in the rule itself, because it is the reason the rule has
to exist at all rather than being replaced by a check.

**Rejected.**
- **A lint instead of a rule** (enabling `PLW1514`) — it would cover the file-I/O third only, says
  nothing about the console or PowerShell halves, and this repo's convention is that a locked
  behaviour gets its rule text first. Worth doing on its own merits; recorded as an open, not folded
  in here.
- **One home instead of four.** `.claude/` is local-only, so a canonical-only rule is invisible in a
  clone; a docs-only rule is invisible to an agent reading the entry file. The duplication is the
  digest pattern the repo already uses for the other eight non-negotiables.
- **Restating the rule in `src/doc_assistant/CLAUDE.md`** — the module files cap at 40 lines and are
  explicitly not for project-wide rules; the file-I/O rule is already visible there as `fsutil`.

**What it opens.** `PLW1514` is unselected, so nothing stops a new `open()` without `encoding=`.
Enabling it is cheap and would need a sweep of the existing call sites first.

---
## 2026-08-07 (3) — the extraction cache now knows which extractor wrote it (KI-40), so yesterday's fix can actually reach a user

**What changed.** `ingest/cache.py`: an `extraction_fingerprint()`, recorded beside every cached
`.md` as `<name>.md.fp` and compared in `is_cache_fresh`; a `write_cache()` that writes the pair;
`reason=` on the extraction log line. 9 new tests, 5 integration fixtures re-pointed.

**Why.** The text-layer fallback recovers three documents from ~0 to 46k / 89k / 778k characters —
**on a fresh ingest**. `is_cache_fresh` compared mtimes only, so on an existing library it changed
nothing, silently. **Shipping it would have been inert for every current user**, which is the worst
kind of release: the improvement is real, measured, and invisible to the people who have the
problem. KI-14 and KI-29 both changed extraction output and both had the same hole.

**Bump-free, copying the precedent already in this repo.** `sparse_index.fingerprint` hashes the
tokeniser's source specifically so a change invalidates *"without anyone remembering to bump a
constant"*. The same standard applies here, so the fingerprint hashes **every extractor function's
`co_code`** — plus three things bytecode cannot see:
- **module-level tunables**, referenced by *name* from a function so their values never reach
  `co_code`. `_TEXT_LAYER_KEPT_MIN` is exactly that knob, and it changes output;
- **`config.PDF_EXTRACTOR`**, which selects the extractor at all;
- **the PyMuPDF / PyMuPDF4LLM versions** — a dependency upgrade changes extraction output with no
  code change of ours.

Plus `_EXTRACTION_VERSION`, a manual escape hatch for the residue (a changed string literal alters
output without altering bytecode). **Bytecode, not source**, for two concrete reasons: PyInstaller
ships `.pyc` so `inspect.getsource` raises in the frozen build, and hashing source would re-extract
every library on a **comment** edit.

**The cost is surfaced before it is paid, and that came for free.** `plan_files` is stat-only, so it
reports the work without extracting: `--dry-run` now reads `would_reembed=97` **in 8 s**. Each
re-extraction logs `reason="extractor_changed"`, so a one-off slow ingest is explained rather than
mysterious.

**Rejected.**
- **Treating a fingerprint-less cache as fresh** (grandfathering existing libraries). It would have
  made the upgrade path work on paper and deliver nothing — the exact defect being fixed.
- **A hand-bumped version constant alone**, which is what the KI first proposed. The repo's own
  precedent rejects it, and a forgotten bump reproduces this bug in full silence. It survives only
  as the escape hatch.
- **A header inside the `.md`.** Those bytes *are* the document text and are hashed into
  `doc_hash`; anything added would change every document's identity.
- **Writing the fingerprint first.** It could then vouch for a truncated `.md`. Written last, a
  failed write costs one needless re-extraction and cannot lie.

**A drift the tests caught, and the fix that removes the class.** 13 integration tests failed
because five separate fixtures hand-wrote a cache entry as "a `.md`, newer than the source" — which
is no longer what a cache entry *is*. Rather than patch five call sites, "a cache entry is the pair"
now lives in one function (`write_cache`) that both the pipeline and the fixtures call. Five
hand-rolled definitions are how the meaning drifted in the first place.

**What it opens.** The first ingest after any extractor change re-extracts the library — inherent,
per-change rather than per-launch, and now visible in the plan. On a 10k corpus that is the ~41 h
figure from `performance.md`, so a future extractor change is a **release-note-worthy event**, not a
silent one. The API/UI ingest path shows the plan less prominently than the CLI's `--dry-run`;
worth a look when the Library UI is next touched.

---
## 2026-08-07 (2) — EX1: three of the four "scanned" documents never needed OCR. Retrieval recall 28/35 → **34/35**

**What changed.** A text-layer fallback inside `extract_pdf_pymupdf` (+ `_recover_lost_page`, a
`Protocol` pair to keep it testable, 9 guard tests), and **KI-40**. No OCR, no new dependency, no
system binary, no second extraction path.

**Why — the premise of ADR-039 was right as a category and wrong about which documents are in it.**
EX1 was scoped as "scanned PDFs have no text layer, so OCR them". Before installing anything, I
looked at the four degraded documents. **Three of them already carry a good text layer:**

| document | health | `page.get_text()` | what the extractor cached | word-like |
|---|---|---|---|---|
| `hebb_1949` | marginal | **776,162 chars** | 5,117 | 94% |
| `hodgkin_huxley_1952` | marginal | **88,754** | 3,219 | 72% |
| `hubel_wiesel_1959` | broken | **45,995** | 86 | 88% |
| `middleton-2001` | broken | **0** | 0 | — (a true scan) |

**Mechanism, reproduced on the shipped extractor** (so it was live, not a stale cache): PyMuPDF4LLM
sees a full-page image, emits `**==> picture [331 x 154] intentionally omitted <==**` and never
reaches the invisible text behind it; `strip_image_placeholders` (KI-14) then removes even that,
leaving `## ##`. Measured retention on those pages: **0.0%–3.2%**.

**The fallback is licensed by a measurement, not a guess.** The two populations do not overlap or
come close — 14 healthy PDFs over 28 pages kept **97.3%–108.9%** (over 100% because markdown *adds*
structure), the degraded ones **0.0%–3.2%**. A ~94-point gap with nothing in it, so a threshold in
the middle is **structural, not corpus-tuned** — which is what the robustness contract actually
demands. `_TEXT_LAYER_KEPT_MIN = 0.5`, and a test asserts it stays inside the measured gap.

**Result, end to end:**

| document | chunks before | chunks after | health |
|---|---|---|---|
| `hubel_wiesel_1959` | 1 | **61** | broken → **healthy** |
| `hodgkin_huxley_1952` | 7 | **125** | marginal → **healthy** |
| `hebb_1949` | 16 | **1,019** | marginal → **healthy** |
| `middleton-2001` | 0 | 0 | broken (unchanged — correctly) |

Corpus health **93/2/2 → 96 healthy / 0 marginal / 1 broken**. **Retrieval recall on the private
35-case set: 28/35 → 34/35.** The single remaining miss is `middleton_frontal_subcortical`, whose
document is the one genuine no-text-layer scan.

**So ADR-039's actual scope is now one document in 97**, not four — and that document is the case
the ADR describes exactly. The OCR work is still worth doing and still gated on RG-025; it is simply
much smaller, and it no longer blocks the recall gap it was thought to own.

**Rejected.**
- **Installing Tesseract + Ghostscript first.** Neither is present, both are system binaries, and
  three of the four documents turned out not to need them. Looking at the documents cost minutes and
  removed the dependency question from 75% of the problem.
- **Fixing this in the health scorer.** `health.py` was right: these documents *were* broken as
  ingested. The defect was upstream of the label.
- **Suppressing the placeholder differently / tuning KI-14's stripper.** The placeholder is a
  symptom; the text never reached the markdown in the first place.
- **A ratio derived from these four documents.** That would be corpus-tuned. The threshold is
  justified by the *gap between two populations*, and the test pins it to that gap rather than to a
  value.

**What it opens — and it is bigger than this fix (KI-40).** `is_cache_fresh` compares **mtimes
only**, so the cached `.md` is never invalidated when the *extractor* changes. **Shipping this fix
does nothing for any library that has already ingested** — including every existing user. KI-29 and
KI-14 both changed extraction output and both had the same hole. The pattern to copy is already in
this repo: `sparse_index` fingerprints its inputs and logs `sparse_index_stale`. Until then the cure
is manual, and it has a trap: re-extraction changes `doc_hash`, `_existing_document_id` matches on
`doc_hash`, so a `--files` re-ingest **mints a second row and skips orphan cleanup** — measured here
as 97 → 100 documents, each file listed twice with different health. A plain `ingest` reconciles it
(`cleanup_orphans_sqlite` handles "the pre-change hash of a document whose content changed"), which
is what was run: back to 97.

Also: the corpus grew by ~1,200 chunks, so the 2026-08-07 citation-coverage baseline was measured
against a *slightly different* corpus than the one now on disk. It compares before/after within
itself, so its conclusion stands, but a future re-run will not be exactly comparable.

---
## 2026-08-07 (1) — the prompt fix cured the citation FORMAT and moved coverage not at all (KI-36 re-measured on shipped code)

**What changed.** No code. One baseline —
`tests/eval/baselines/citation_coverage_2026-08-07.md` — plus KI-36.

**Why.** v0.4.1 ships a provider card publishing **36% / 14% / 81%** citation coverage, and those
numbers were measured on the **previous** prompt. The 2026-08-06 header change altered how sources
are presented to the model, so the app's own published claim about itself was unverified against
the code that shipped. That is exactly the kind of quiet staleness that turned a CHANGELOG limit
into a lie earlier this week.

**Measured** — same 27 healthy-document cases, same exclusions, same settings, one variable:

| provider / model | before | after (shipped) | Δ | citing nothing |
|---|---|---|---|---|
| `anthropic/claude-haiku-4.5` | 81.2% | **83.5%** | +2.4 pp | 0/27 → 0/27 |
| `ollama/llama3.1:8b` | 36.4% | **37.6%** | +1.2 pp | 11/27 → **12/27** |
| `ollama/qwen2.5:7b` | 13.5% | **18.0%** | +4.5 pp | 18/27 → 18/27 |

**The finding is the negative, and it is worth as much as a positive would have been.** Every delta
is same-signed but inside what a single repeat over 27 cases can resolve against ~3% case-level
retrieval noise, so **none of it is reportable as an improvement**. Paired by case, `llama3.1:8b`
had **no** answer that cited nothing start citing, and one that had cited stop.

**So: prompt engineering fixed what prompt engineering could fix, and did not touch the rest.** The
header change took header-copies 6 → 0 and RG-012 FAIL → PASS — a format defect, cured outright.
Coverage sat still. **That is a second, independent line of evidence for KI-36's capability-floor
conclusion:** the first was cross-provider (same prompt, 81% vs 36%), this is same-provider (changed
prompt, same coverage). Two different cuts, same answer.

**Rejected.**
- **Updating the shipped provider card to 38 / 18 / 84.** Every delta is within noise, and changing
  a shipped string means rebuilding and re-running RG-012 on a freshly tagged release — trading a
  real risk (an unverified rebuild) for a cosmetic gain. Refresh it at the next release that
  rebuilds anyway; the baseline records which column to use.
- **Calling +4.5 pp on qwen an improvement.** It is the largest delta and the noisiest arm (13.5 →
  18.0 with 18/27 answers still citing nothing). Reporting it would be exactly the "claim without a
  control" this project's rigor gate exists to stop.
- **Adding repeats now to settle the drift.** Worth doing before anyone acts on the drift; not
  worth blocking a release that does not depend on it.

**Correction to 2026-08-05 (2)** (append-only, so recorded here rather than edited there): that
entry's table says `qwen2.5:7b` had **19/27** answers citing nothing. Recounting the stored rows
gives **18/27**. The pooled figure (13.5%), the median (0.000) and every user-facing number derived
from them are unaffected; only the count was wrong. Corrected in KI-36.

---
## 2026-08-06 (4) — release tooling: a preflight that encodes every trap that actually bit, and CI finally runs the frontend

**What changed.** New `scripts/release_preflight.py` (+ `just preflight`), `docs/RELEASE.md`,
`tests/unit/test_release_preflight.py`, a **frontend job in CI**, and `npm audit fix` (postcss
→ 0 vulnerabilities).

**Why.** Today's release was cut by hand and nearly went wrong four separate ways. None of those
were interesting problems — they were all "did you actually do the thing" problems, which is what a
script is for. **Every check in the preflight is an incident, not a best practice:**

| Check | The incident |
|---|---|
| `versions` | v0.4.0 bumped four version strings and missed `uv.lock`. CI installs `--locked`, so the job died *before* the gates and `main` was ungated for days. |
| `artifact_fresh` | The whole of 2026-08-06. Source-green says nothing about a frozen binary (KI-34). If the installer predates the code, the thing tested is not the thing shipped. |
| `sidecar_size` | KI-34 is a size cliff — 1545.5 MiB broken vs 1562.1 MiB fixed. The cheapest possible check on a packaging bug that is invisible from source. |
| `rg012` | Matches the installer build timestamp the harness logged against the installer on disk. **A PASS from a previous build is worse than no PASS** — it reads as evidence. |
| `dev_commands` | KI-39: the app told users to run `just api`. |

**Two of the five checks were wrong on their first run, both in the "looks green" direction** — and
that is the part worth keeping:
1. `sidecar_size` was written in **decimal MB** while every recorded reference number is **MiB**,
   putting the floor at ~1478 MiB — *below both* the broken and fixed sizes. It would have passed
   the exact build it exists to reject, while printing `[ok]`. **A units bug in a safety check is
   worse than no check.**
2. `dev_commands` matched `just \w+` and flagged **"just now", "just a", "just the"**. `just` is an
   ordinary English word; only a real recipe name makes `just X` a command, so the names are now
   parsed out of the justfile — which also keeps the check honest as recipes come and go.

Both are pinned by tests, so neither can regress quietly.

**CI had no frontend job at all.** The desktop app is half the product, and `npm test` (78 tests)
plus `svelte-check` (189 files) ran on a developer's machine or not at all — every frontend fix
today, including KI-39 and the citation-contract pin, was guarded only by me remembering. Added as
its own job: `npm ci` (lockfile-exact, same discipline as `uv sync --locked`), `svelte-check`, then
tests. Seconds long, no Python.

**`npm audit fix`** (carried since 2026-08-05): postcss had a **high**-severity path-traversal
advisory and the lockfile pinned **8.5.15 while the local tree had 8.5.25** — so CI would have
installed the *vulnerable* one. Now 8.5.26, **0 vulnerabilities**, 78/78 + svelte-check still green.
Build-time only, so the v0.4.1 artifact is unaffected and needs no rebuild.

**Rejected.**
- **Automating the CHANGELOG check beyond "the section exists".** A script cannot tell whether a
  known limit is still true — and that is exactly the failure mode we hit (0.4.1 claimed the
  clean-machine install was unverified for three days after it was verified). `docs/RELEASE.md` §2
  makes it a judgment step instead of pretending.
- **Making the preflight a pre-commit hook.** It reads build artifacts and the RG-012 archive;
  most of it is meaningless mid-development and would train people to ignore it.
- **A `release` recipe that runs the whole thing end to end.** The sequence has two ~10-minute
  builds, a machine-level Ollama rebind and a VM in the middle. A checklist that a human drives
  beats a script that pretends those are atomic.

**What it opens.** `preflight` hard-codes the Windows harness path and the msvc triple — fine on the
one build box, wrong the day there is another. And it cannot check the thing that matters most:
whether the answers got worse.

---
## 2026-08-06 (3) — RG-012 FAILED on a citation form, and the fix was to stop showing the model a bracket to copy

**What changed.** `pipeline.format_docs_for_prompt` — the retrieved-passage header is no longer
bracketed: `[Source 3: paper.pdf, page 4]` → `Source 3 — paper.pdf, page 4`. `prompts.py` updated to
match, plus one new line: *"Square brackets appear nowhere in the sources — every `[n]` you write is
a citation."* Two guard tests.

**Why — a real RG-012 FAIL, and the genuine version of what KI-35 falsely claimed.** The re-run
against the rebuilt installer produced a meticulously attributed answer that cited
**`[Source 1: reranking_bert_nogueira_2019.pdf]`** six times — the *source header format*, copied
verbatim. `synthesis._CITATION_TOKEN_RE` accepts a label plus digits, not a filename, so it resolves
to nothing: **`valid=[]`, 6 malformed, 12/12 sentences uncited, all 12 claims badged `uncited`** on
an answer that names and quotes its source in every paragraph. `prompts.py:47` has warned against
exactly this since 2026-07-14 and the model did it anyway.

**Three runs, same model, same question, three different citation forms** — the instability
measured rather than asserted:

| run | form | outcome |
|---|---|---|
| 2026-08-05 17:31 | `[Source 1]` ×3 | resolved (the 2026-07-14 tolerance covers it) |
| 2026-08-06 14:13 | `[1]` ×5 | PASS |
| 2026-08-06 20:10 | `[Source 1: paper.pdf]` ×6 | **unresolvable → FAIL** |

**Chosen fix: remove the imitation target, not widen the reader.** Square brackets now appear
**nowhere** in the context, leaving `[3]` in the instructions as the only bracketed thing in the
entire prompt. The alternative — teaching the parser to swallow `[Source N: anything]` — was
explicitly rejected: KI-35 itself warned that *"a silently-widened parser hides model drift"*, and
that is precisely what it would have done here. The model would have gone on emitting a
non-canonical form and we would have stopped being able to see it.

**Measured after the change**, same question, 6 runs, shipped local default (`llama3.1:8b`, $0):
**header-copies 0/6**; 5/6 clean canonical `[n]` (3–6 citations each, all resolving); the sixth
flagged `[CLS]` — a BERT token quoted out of the source text, not a citation attempt, and the same
benign "audit cries wolf" class already noted for bracketed phrases. **Caveat: 6 runs on one model.**
The failure appeared in 1 of 3 runs before, so 6/6 is suggestive, not conclusive; RG-012 is the gate
that decides.

**Rejected.**
- **Widening the parser to accept `[Source N: …]`.** Unambiguous, and tempting — but it converts a
  visible defect into an invisible one. The header was the cause; fix the cause.
- **Arguing harder in the prompt.** Five citation rules and a parenthetical naming this exact
  confusion were already there and were ignored. A sixth rule is not a mechanism.
- **Keeping brackets but changing the inner text** (e.g. `[3]` alone as the header). Then the
  context contains bracketed numbers, and a model echoing one is *indistinguishable* from a real
  citation — worse than either alternative.

**RG-012 re-run after the full rebuild (sidecar re-frozen 646 s + installer re-bundled): PASS.**
Install 202 s → health ~30 s → 3 PDFs / 322 chunks → cited turn 17 s → **4 resolved citations, 4
canonical `[n]`, 0 labelled, 0 unresolvable.** Verified in that artifact: badges `uncited` ×6 /
`weakly grounded` ×2 (KI-37), `best source relevance` present and the old `Reranker scores` heading
gone, `citation_note_md` **empty** (audit clean — no malformed anything), and the answer's bracket
tokens are exactly `[3] [1] [8] [7]`.

**What it opens.** The audit still counts any bracketed token containing letters as a failed
citation attempt, so a passage quoting `[CLS]`, or a model wrapping a phrase in brackets, produces a
false "malformed citation" warning. Same disease as KI-35, one layer down; recorded in KI-36.

**Harness note — Windows Sandbox runs ONE instance, and a stale VM silently eats the run.** Two
launches produced a booted-but-idle sandbox whose `LogonCommand` never fired: `vmmemWindowsSandbox`
from the previous run was still alive, and killing `WindowsSandboxServer` /
`WindowsSandboxRemoteSession` does **not** take the VM down with it. Flat VM working-set (982 →
984 MB over 30 s) is the tell — an installing sandbox climbs. **Wait for every
`vmmemWindowsSandbox` to disappear before relaunching**, or the gate reports nothing and looks hung.

---
## 2026-08-06 (2) — the readiness gate no longer gives up, and stops telling users to run `just api` (KI-39)

**What changed.** New `lib/shell/startup.ts` (pure, tested) + the readiness `$effect` in
`App.svelte` + the status-bar copy. Frontend only — the sidecar is untouched, so the rebuild reused
today's freeze.

**Why.** Characterising the "first-launch dead-backend window" with the RG-012 numbers in hand
turned up three facts that are individually defensible and jointly a bad first five minutes:
1. **A 60 s budget** (60 polls × 1 s) against a PyInstaller **onefile** sidecar that extracts
   ~1.5 GB to `%TEMP%` before uvicorn binds.
2. **No retry.** The `$effect` reads no reactive state before its `await`, so it runs **once per
   mount**; after the loop fell through there was no timer and no control to trigger one. Only a
   relaunch recovered.
3. **The message was a developer instruction** — `backend unreachable. Run just api`. A task runner
   and a repo recipe that someone who installed an `.exe` does not have. **The app's only failure
   message asked for something that cannot exist on the machine showing it.**

The margin was thinner than it looked: RG-012 measured health at **~30 s — half the budget** — on an
idle VM, NVMe, file cache warm from the install that had just written those bytes. Defender scans
`%TEMP%`. Exceeding 60 s on a tester's machine is plausible, and that case was not "slow" but
terminal *and* misdirecting.

**The fix is the removal of a terminal state, not a bigger number.** Polling now backs off
(1 s → 2 s → 5 s, bounded) and **never stops**; `startupPhase` only changes what is *said*:
`connecting` → `slow` at 20 s ("a first launch can take a minute…") → `stalled` at 90 s, which shows
the fault colour **while still polling**, so a backend that turns up at minute three lands the app in
`ready` by itself.

**Verified live, on the exact scenario that used to be unrecoverable** — UI up with no backend at
all:

| elapsed | status bar | dot |
|---|---|---|
| 0 s | starting the engine… | wait |
| ~20 s | starting the engine — a first launch can take a minute… | wait |
| ~90 s | still starting — retrying. Restart the app if it never arrives. | **red** |
| backend started at ~170 s | 33,105 chunks · ollama/llama3.1:8b · bge-base | **ok** |

**and the page was never reloaded** (`performance.getEntriesByType('navigation')[0].type ===
'navigate'`). The old gate could not have done that from any elapsed time past 60 s.

**Rejected.**
- **Raising 60 → 120.** Trades one arbitrary cliff for another; the defect is that a terminal state
  exists at all, not its size.
- **Unbounded exponential backoff.** A late backend must still be noticed promptly; the delay caps
  at 5 s deliberately.
- **Dropping the red `down` state** so nothing ever looks broken. After 90 s something probably *is*
  wrong and saying so is honest — the fix is that saying so no longer means giving up.
- **A longer, fuller message.** The bar is `nowrap` + `text-overflow: ellipsis`, so a long sentence
  is simply cut. Both new strings were measured against the 375 px width (297 px and 307 px against
  335 px available); the fuller explanation is in a `title`.

**What it opens.** **RG-010 (cold start) has still never been properly recorded** — one measurement
on one fast VM is what we have, and that distribution is what decides whether the PyInstaller spec
should move onefile→onedir, which is the deeper fix. Filed in KI-39.

---
## 2026-08-06 (1) — **rebuilt the installer and RG-012 Tier-2 PASSED on it.** The release gate is closed

**What changed.** No source. A full artifact rebuild — sidecar re-frozen, installer re-bundled — and
the clean-machine gate re-run against it, because every fix since 2026-08-05 existed only in source
and the installer that last passed was built at 14:58 the previous day. **KI-34's whole lesson is
that a green source tree says nothing about a frozen binary**, so a release cannot rest on one.

**Build.** `uv sync --extra cpu --extra dev --extra packaging` (KI-3: `build_sidecar` refuses to
freeze a `+cu*` torch — the `cu130` wheel segfaults on a GPU-less box) → `just sidecar` (**847 s**)
→ `npx tauri build` (**601 s**). Venv restored to `--extra cu130` afterwards; `cuda_available True`
re-confirmed. **The sidecar came out at 1,562.1 MB — exactly the post-KI-34 reference** (1,545.5 MB
was the broken build), so the `pymupdf` data files are bundled; the size is the cheapest possible
regression check on that fix and it is worth keeping as one.

**RG-012 Tier-2 — PASS**, on a Windows Sandbox reporting `python on PATH? False`:

| step | result |
|---|---|
| silent install | **177 s** (330 s the previous day) |
| files laid down | `doc-assistant-desktop.exe` 10.7 MB · `doc-assistant-api.exe` **1,562.1 MB** |
| `/api/health` | **~30 s**, `chunk_count: 0`, `ollama/llama3.1:8b` |
| `/api/setup` | ollama reachable (9 models), `active_ready: true`, documents step correctly not-done |
| ingest 3 PDFs | **added=3, errors=0, 322 chunks**, ~36 s — KI-34 stays fixed |
| turn | **14 s**, 10 sources |
| citations | **5 resolved (5 canonical `[n]`, 0 labelled); 0 unresolvable** |

**Every 2026-08-05 fix is verified present in the frozen binary**, which is the point of the
exercise: `flagged_claims` badge as **`uncited` ×7 / `weakly grounded` ×3** (KI-37 — no
"unsupported" anywhere), and the card reading **`best source relevance` / `top-3 relevance span`**
with the old `**Reranker scores**` heading gone.

**The de-dup's fallback branch got verified for free, and by accident.** The card *did* print the
per-source list here — correctly: a fresh install has no concept graph, so
`_attach_source_evaluation` returns `None`, `source_eval` is null, no strip renders, and the card
is the only per-source surface. Between this run and the 2026-08-05 dev-app check, **both branches
are now exercised end to end** — strip present → card omits, strip absent → card keeps.

**The corrected gate also earned its keep immediately.** It reported
`5 canonical, 0 labelled, 0 unresolvable` instead of a bare count — and the model used canonical
`[n]` this time, on the same model and prompt that produced `[Source n]` the day before. **That is
the third independent confirmation that KI-35's premise was a one-off** (0/54 in the coverage runs,
plus this).

**Harness change.** `rg012-tier2.wsb` gained a `LogonCommand` so the gate runs unattended. The two
historical "LogonCommand never fires" failures were never Sandbox's fault — the script was
UTF-8-no-BOM and PowerShell 5.1 read it as ANSI. It is ASCII-only now and I byte-checked it (0
non-ASCII bytes) *before* trusting the command; the `.wsb` is byte-checked and XML-parsed too.

**Rejected.** Shipping the artifact that passed on 2026-08-05 — it predates every fix, so the thing
tested would not have been the thing shipped. Tier-1-only (skipping the answer engine) — it would
not exercise the cited-turn path, which is exactly what the fixes touched.

**⚠ Data loss, mine, recorded because the record must be honest.** Clearing `out\` for this run,
I issued the delete after an archive command had been **rejected as a whole** by a path-protection
guard — so its `New-Item`/`Copy-Item` never ran, and I did not check before deleting. The four
2026-08-05 artifacts are gone (`Remove-Item` does not use the Recycle Bin; no shadow copies). A
partial reconstruction recovered verbatim from the session that read them is at
`C:\rg012-host\out-2026-08-05-RECONSTRUCTED\README.md` — complete for the answer,
`flagged_claims`, `provenance_card_md` and `settings.json`; lines 3-26 of the log; fragments only
of the SSE stream. **No conclusion is at risk** (every decisive number was transcribed into KI-35/
36/37 and DEVLOG (2) beforehand, and the gate was re-scored against the real file while it
existed), but the raw provenance for a headline finding is not recoverable. **Copy, confirm the
copy, then delete — and never treat a failed command as a partially-applied one.**

**What it opens.** The gate now overwrites `out\` on every run; it should stamp its output into a
per-run folder instead. And `docs/desktop-packaging.md` still describes RG-012 as "paused" and
Tier 2 as blocked on the data-home decision — both stale, now twice over.

---
## 2026-08-05 (4) — pre-release UI pass: themed scrollbars, the answer column stops rendering under its own scrollbar, and "what is this 0.94?" answered

**What changed.** Three reported defects, all in the answer surface (user report with screenshot).

**1 · Scrollbars were unstyled — anywhere.** `grep scrollbar apps/desktop/src` returned **nothing**,
so every scrolling pane drew the raw OS bar: a bright grey slab against a dark reading surface,
and the loudest element on a page whose prose is deliberately quiet. Added `--scrollbar-thumb` /
`--scrollbar-thumb-hover` to `app.css` using the same `color-mix(in srgb, var(--fg) N%, transparent)`
trick `--graph-edge` already uses — **one declaration covers both themes**, because `var()` is
late-bound. Declared twice on purpose and they do not fight: `scrollbar-color` is the standard
property and wins in Chromium (WebView2, the Windows Tauri runtime) and Firefox — where it is set
to anything but `auto`, Chromium ignores `::-webkit-scrollbar` entirely — and the `::-webkit-` block
is the fallback for WKWebView, which has no `scrollbar-color`. Graceful degradation, not duplication.

**2 · The answer column rendered underneath its own scrollbar.** Measured, not guessed:
`section.conversation` had `padding: var(--space-2) 0` — **zero** horizontal — while scrolling, so
its 15 px vertical bar ate the content box (`width 790` → `clientWidth 775`). Every right-anchored
child sat exactly ON that boundary: `.usage` and the source-evaluation `.score` column both ended at
x=1206 = the content edge, gap **0 px**, which is why the screenshot reads "0 tokens · **loca**".
Fixed with a real gutter (`padding: var(--space-2) var(--space-3)`) plus `scrollbar-gutter: stable`,
so the space is reserved even when no bar shows and a growing turn no longer shifts the whole column
15 px left as it crosses the fold. Verified live: gap **0 → 11 px** for both elements, no body
overflow. *This was never a "cut window" — nothing was clipped horizontally; content was drawn under
an overlaying bar.*

**3 · "Is 0.94 a relevance score or a quality score?" — a fair question the UI invited.** It is
**retrieval relevance**: the cross-encoder reranker's score for that *chunk* against *this question*,
the same number the ranking used. **Nothing in this app scores a source's quality.** But the number
sat unlabelled, in a block headed *"Source evaluation"*, beside an epistemic assessment — and since
KI-33 withheld that assessment, the strip's only real content **is** a year and this score. So the
heading now promises an evaluation the row no longer delivers, and the bare decimal is the only
thing left to read as one. Fixed by naming it: a legend under the heading (*"relevance = how well
the retrieved passage matches your question (reranker score, 0–1). Not a judgement of the source."*)
and a small-caps `relevance` unit beside every number, so a lone decimal is unambiguous even when
the legend has scrolled away.

**And the repetition, which was real.** On a low-confidence answer the provenance card printed
`**Reranker scores** — [1] reranker 0.907 …`: the *same* measure, keyed the same way, one decimal
deeper, on the same answer as the strip. The card now omits it when the strip is on screen
(`_ProvenanceInputs.source_strip_rendered`, set from `source_eval is not None and bool(sources)` —
`Turn.svelte`'s own render condition). It **keeps** the aggregate signals (`best source relevance`,
`top-3 relevance span`), which appear nowhere else and are what the confidence verdict is derived
from. Renamed throughout: "reranker" is our word, not the reader's.

**Rejected.**
- **Dropping the per-source list unconditionally.** The strip needs a concept graph
  (`_attach_source_evaluation` returns `None` without one), so on a graph-less corpus the card is
  the *only* per-source surface. Pinned both ways by test, so the de-dup cannot become a data loss.
- **`overflow-x: hidden` on the conversation** to stop the clipping. It would hide the symptom and
  silently swallow any genuinely too-wide child (a table, a long code block) instead of letting it
  scroll. The defect was missing padding; fix the padding.
- **A tooltip alone for the score.** It already had one (`title="Reranker relevance score…"`) and
  the question still got asked — hover text is not documentation, and it does not exist on touch.
- **Renaming the strip's "Source evaluation" heading.** Tempting while the assessment is withheld,
  but the heading is right for what the strip is *for*; the honest fix is restoring the assessment
  (ADR-041), not renaming the feature around a temporary containment.

**What it opens.** The strip's heading over-promises while KI-33 holds — worth revisiting if the
containment outlives the release. And the **source explorer** the same report asked for (chunk →
parent → document) is filed in `docs/ui-checklist.md` §3, deliberately **not** built here: it is new
capability, not a defect, and the pre-release pass is for defects.

**Verified live**, real local turn (`ollama/llama3.1:8b`, $0, 12 s): score reads `relevance 0.87`,
legend renders, card shows `best source relevance 0.872` with **no** per-source list, 11 px gap to
the scrollbar, themed thumb applied. **Method note:** the first check appeared to *fail* — the card
still said "top reranker". The API sidecar had been started before the edits and uvicorn was not run
with `--reload`, so it was serving stale Python. **A frontend hot-reload proves nothing about the
backend**; restart the sidecar before believing a backend-rendered string.

---
## 2026-08-05 (3) — tell the user the free path cites less, where they choose it (KI-36 follow-through)

**What changed.** Four lines of copy in `apps/desktop/src/lib/settings/ProviderSetup.svelte`'s
Ollama card, beside the existing "needs ~5 GB of disk, happiest with a GPU" trade-off sentence:
the measured citation-coverage gap, with the run size and corpus named so it can be reproduced.

**Why.** Entry (2) measured it: on the same prompt and the same retrieval, `llama3.1:8b` cites 36%
of its sentences and `qwen2.5:7b` 14%, against 81% for Haiku. The integrity layer is right to flag
the difference — but a first-run tester on Ollama meets a wall of `uncited` badges with nothing
anywhere telling them the provider is the reason. **The measurement is useless to the user if it
only lives in the DEVLOG.** `ProviderSetup` is both the first-run surface and the ongoing switcher
(it is embedded in `Settings.svelte`), so one placement covers both moments of choice.

**Tone, deliberately.** Indicative and reproducible, not a verdict: it states *n*, the corpus size
and that prompt and retrieval were held constant, and it closes with what stays true —
*"Answers stay grounded in your documents; more claims will simply show as uncited."* Inform, don't
block: nothing is gated, the local option is not discouraged, and the number is the user's to weigh.

**Rejected.**
- **A computed field on `ProviderReadiness`** (backend → wire → `types.ts` → component). It is not
  a verdict the backend can recompute — it is a benchmark result — and three layers of plumbing for
  a static string is drift surface, not thinness. The card's existing trade-off copy sets the
  precedent, and a comment at the line points at the DEVLOG method so the numbers cannot rot
  silently.
- **Warning at answer time instead** ("many claims are uncited because you are on a local model").
  Better targeted, but it fires when the user can no longer act cheaply; the choice point is where
  the information changes a decision. Worth revisiting *in addition*, not instead.
- **Naming no model.** Vague hedging ("local models may cite less") is exactly the tone the project
  rejects — the numbers are measured, so print them.

**Verified live** (dev server, mocked nothing — this is static copy): renders in the Ollama card
between the trade-off line and the host line; identical colour treatment to its sibling paragraphs
in **dark and light**; at **375 px** its box is byte-identical to the two pre-existing `p.detail`
siblings (`left 481, width 284`) — the overlay's own pre-existing offset at that width, not this
change — and the body never scrolls horizontally. `svelte-check` 188 **0/0**, `npm test` **73/73**.

**What it opens.** The Settings overlay does not reflow at 375 px (pre-existing, seen while
checking this). And the frontend's health poll retries **21 times against a 500** while uvicorn
binds — the cold-start window the baton lists as item 2, now with a number.

---
## 2026-08-05 (2) — KI-35 was a **bug in the gate, not the app**: RG-012 Tier-2 had passed. The real defect is citation *coverage* (KI-36/37/38)

**What changed.** `.claude/KNOWN_ISSUES.md`: KI-35 rewritten as a corrected diagnosis; **KI-36**
(citation coverage), **KI-37** (the `unsupported` collision, fixed), **KI-38** (the `load_dotenv`
override hole) filed. The RG-012 gate at `C:\rg012-host\script\rg012-run.ps1` fixed (local-only
harness). Two code changes, both consequences of the investigation rather than of the original
report: the claim badge split (`helpers._claim_badge` + `ClaimReview.svelte`) and the citation
contract pinned across the wire (`apps/desktop/src/lib/chat/citations.ts` extracted from
`Markdown.svelte`, `tests/fixtures/citation_vectors.json` read by **both** suites).
**Nothing in the citation *parser* changed — it was already right.**

**Why.** KI-35 was the baton's #1 next action and it described a defect that does not exist. It
claimed `llama3.1:8b` cites `[Source 1]`, that neither parser resolves that, and that the integrity
layer therefore "inverts" on the shipped local path. Checked against the run's own archived
`result.json`, with the shipped code: `cited_source_numbers` → `[1, 5, 2, 1]`; `audit_citations` →
`valid=[1,2,5]`, `malformed=[]`, `clean=True`; the `[Source 5]` claim scored **`weakly grounded`**,
not `unsupported`. **Both parsers have tolerated `[Source n]` since 2026-07-14** —
`synthesis.py:35` and `Markdown.svelte:52`. The 13 flagged claims were the sentences that genuinely
carried no citation: the model cited **4 of 16**. The integrity layer reported the truth.

**The actual defect was one line of the gate.** `rg012-run.ps1` counted `'\[\d+\]'` — a *stricter*
contract than the app implements — logged `FAIL: answer produced but not cited`, and that verdict
was filed as an app bug. Re-scored with the app's own token: `resolved=4 canonical=0 labelled=4
unresolvable=0` → **PASS**. **RG-012 Tier-2 passed on 2026-08-05**; the release's packaging gate has
been green since KI-34 was fixed. The gate now separates three outcomes — resolved / unresolvable /
no-bracket-at-all — because they need completely different fixes.

**The rule, which is the reusable part: a verification gate must call the contract, never restate
it.** A restated contract drifts, and because a gate is trusted, its false verdict gets filed
against the code it was meant to protect. Same class as KI-34 one level up (there the gate tested
too little; here it tested something never promised).

**Then the measurement KI-35 should have been.** 35 private cases → the real `ChatController` on
the live 97-doc corpus, provider forced local, $0.
- **Retrieval first, to avoid confounding it:** recall 28/35. **All 7 misses are 4 documents the app
  already labels degraded** (`middleton-2001.pdf` 0 chunks, `hubel_wiesel_1959.pdf` 1, `hodgkin_
  huxley_1952.pdf` 7, `hebb_1949.pdf` 16). **Recall on healthy-document cases is 28/28.** That is
  ROADMAP **EX1** / ADR-039's OCR case, now quantified: degraded documents are the *entire*
  retrieval-recall gap, and they are all pre-1970 scans.
- **Citation coverage on the remaining 27** (`llama3.1:8b`): pooled **79/217 = 36.4%**, and
  **bimodal** — 11 answers cite nothing at all, 9 cite ≥85%, almost nothing between. 8 of the 11
  zero-citation answers are substantive uncited assertions; 3 are correct refusals.
- **The comparison that decides it** — same prompt, corpus, retrieval and cases:

  | provider / model | pooled coverage | median | answers citing **nothing** |
  |---|---|---|---|
  | `anthropic/claude-haiku-4.5` | **155/191 = 81.2%** | 0.875 | **0 / 27** |
  | `ollama/llama3.1:8b` (shipped local default) | 79/217 = 36.4% | 0.545 | 11 / 27 |
  | `ollama/qwen2.5:7b` | 40/296 = 13.5% | 0.000 | 19 / 27 |

  **So the prompt is not the defect** — a sixth citation rule is not what separates 81% from 36% —
  and `llama3.1:8b` is already the better of the two locals. **KI-36 is a local-model capability
  floor: an honest limitation of the free path to be *documented at the provider choice*, not a bug
  to fix before release.** The integrity layer is working; it correctly reports that a local model's
  answers are largely uncited. What a first-run Ollama tester lacks is any explanation of why.
- **`[Source n]` appeared 0/81 times** across all three models — KI-35's premise does not reproduce.
- Three *real* unresolvable forms showed, all rare and none of them the one filed: `Source 6
  [file.pdf]` (number outside the bracket, filename inside), `[2][11][17]` against 10 sources, and —
  on **Haiku** — claim text wrapped in brackets, `[a Bayesian non-parametric model …][7]`, the exact
  anti-pattern `prompts.py:53` forbids. That last one makes the audit **cry wolf**: the citations
  resolve (coverage 1.000) but the seven phrase-brackets are counted as failed citation attempts, so
  a fully-cited answer renders "⚠ 7 malformed citation(s)". A bracketed phrase immediately followed
  by a resolvable token is a style violation, not a citation attempt. Recorded in KI-36, not fixed.

**KI-37, found in passing and fixed.** The RG-012 card renders the reviewer's **"unsupported
claims: `0`"** directly above **"⚠ 13 claim(s) to review … *(unsupported)*"** — one word, two
meanings, same view. Worse, `claim_marker` labelled a *correct refusal* `unsupported` (3 answers,
16 badges), accusing the model exactly when it did the right thing. The badge now says what the
structural marker actually found — **`uncited`** (no citation token) or **`unresolved citation`**
(cites only numbers mapping to nothing) — the same three-way split the gate now makes. Presentation
only: `MARKER_UNSUPPORTED` and the persisted `AnswerClaim.marker` are untouched, so there is no
migration and `test_adjudication_persistence`'s marker triple still holds. `ClaimReview.svelte` now
tests for the one *benign* label and defaults the rest to `bad`, so a future severe badge cannot
silently render as mild.

**The contract is now pinned across the wire.** `[n]` had three implementations (Python, Svelte,
the gate) and a test on only one. The Svelte regex is extracted to a plain, dependency-free
`lib/chat/citations.ts` — the module kind `node:test` can actually run — and both suites now assert
the **same** `tests/fixtures/citation_vectors.json`. Verified the pin bites: deleting the
`[Source n]` tolerance from the TS side fails 3 frontend tests, naming the vector. (The component
imports it extensionless and the test imports it with `.ts`: `svelte-check` rejects the extension,
`node:test` requires it.)

**Rejected.**
- **Widening the parser** (KI-35's proposed mitigation): already implemented, and it changes nothing
  — the dominant failure is answers with no citation of any form, which no parser can reach.
- **Editing the prompt now.** It already carries five explicit citation rules and one of them names
  this exact confusion; a sixth on intuition is not a fix. Moving coverage is an eval-harness
  experiment with a control (rigor-gate), not a prompt tweak.
- **A refusal detector** for KI-37 — a heuristic wrong in both directions; renaming the badge to
  what it measures makes the refusal case merely true, with no detector to get wrong.
- **Archiving KI-35 as a resolved row.** The correction *is* the issue; compressing it to a row
  would drop the lesson and leave the original wrong story as the memorable one.
- **Filing the degraded-document finding as a new KI** — EX1/ADR-039 already own it; this run adds
  evidence, not a new issue.

**What it opens.** (a) **Tell the user why the free path cites less** — the measured 81/36/14 split
belongs where the provider is chosen (first-run setup / provider picker), in the "inform, don't
block" register. That is now the main open item from this work, and it is a UX change, not a
retrieval or prompt one. (b) **The audit should stop crying wolf** on a bracketed phrase that is
immediately followed by a resolvable citation (Haiku's one bad answer). (c) **EX1/OCR now has a
measured payoff** — 7 of 7 retrieval misses, all pre-1970 scans. (d) The gate still restates the
contract in PowerShell; the honest end-state is the API exposing the audit structurally so the gate
asserts on the app's own verdict instead of a third regex. (e) Unmeasured: run-to-run variance
(single repeat per model), and whether Sonnet differs from Haiku.

**Method note.** Two harness bugs of mine preceded the results and both would have corrupted them:
the first smoke run picked the two *worst* cases (both `hodgkin_huxley`, a `marginal` document) and
read 0% coverage as a citation defect — it was a retrieval miss; and a `Set-Location` for `npm test`
leaked into the next run's working directory. **Also: `config.py`'s `load_dotenv(override=True)`
means `LLM_PROVIDER=ollama <cmd>` silently runs on Anthropic and bills** — filed as KI-38, with the
seam that does work (`app_settings.get_llm_selection`, plus a separate line for the pinned reviewer).

---
## 2026-08-05 (1) — RG-012 Tier-2 finally ran, and the shipped installer **could not ingest a single PDF** (KI-34)

**What changed.** One line of `scripts/doc_assistant_api.spec` — `"pymupdf"` added beside `"fitz"` —
plus **KI-34**. The gate itself ran for the first time in the project's history.

**The finding.** Provenote 0.4.1 installed on a clean, Python-free Windows Sandbox. Install ✅,
launch ✅, backend ✅, `/api/setup` ✅ (Ollama detected reachable, 9 models, `active_ready:true`,
documents step correctly not-done). Then ingest of 3 PDFs: **`added=0, errors=3`, in 0.65 s** — far
too fast to have attempted extraction. Every file, identically:

```
[Errno 2] No such file or directory:
  ...\Temp\_MEI82922\pymupdf\layout/resources/onnx/layout_rf2.4.1+imf1.yaml
```

**Cause.** The spec collected **`"fitz"`** only. `fitz` is the *legacy import shim*; PyMuPDF's real
distribution directory is **`pymupdf/`**, carrying data read at extraction time. `collect_all("fitz")`
bundles the shim and none of it, so the frozen build **imports cleanly and then fails every PDF**.
Verified after the fix: `collect_all("pymupdf")` yields **129 data files**, including the exact
`layout_rf2.4.1+imf1.yaml` the error named.

**Why five layers of green missed it, which is the part to carry.**
1. **Invisible from source** — site-packages has the file, so the 1447-test suite, the eval harness
   and the desktop dev loop all pass.
2. **The v0.4.0 WSL clean-room run passed** for the same reason: it was a *source* install.
3. **The standalone sidecar smoke this session ran passed too** — `/api/health` → 33,105 chunks —
   because the missing file is read on the **extraction** path, not at import. **Booting a frozen
   binary proves nothing about whether its data files were bundled.** A packaging gate must push a
   real document end to end.

**Two more findings from the same run, neither fixed.** (a) **First launch has a long
dead-backend window** while 1.5 GB of onefile extracts before uvicorn binds — health lands in 10–20 s
warm, but minutes cold, and the UI does not signal it convincingly. A tester's first impression.
(b) The install folder is `Provenote/` while the executable is `doc-assistant-desktop.exe` — ADR-012's
product/code identity split working exactly as designed, but undocumented and it cost two diagnostic
rounds here.

**Method note.** Three of my own harness bugs preceded the real one and each was mine, not the
product's: a UTF-8-no-BOM script that Windows PowerShell 5.1 read as ANSI (em-dashes broke parsing —
and *that* was the true cause of the two "LogonCommand never fires" failures I had blamed on Sandbox);
`Select-Object -First 1` picking the **June 0.1.0** installer out of a folder holding both; and
filtering for a guessed `Provenote.exe`. **Scripts that drive a gate are ASCII-only and never select
an artifact incidentally** — both now enforced in the harness at `C:\rg012-host\`.

**Rejected.** Calling the earlier "backend unreachable" a product bug — measured twice at ~10 s and
~20 s to health, so it was extraction latency, not a failure. Declaring RG-012 Tier-2 passed on the
strength of install + launch + health: the gate's actual claim is **a cited turn**, and ingest fails,
so the answer is still unproven.

**What it opens.** Rebuild (sidecar + installer) and rerun in a **fresh** sandbox — the gate should
now reach the cited turn. Then: the first-launch extraction window deserves its own fix before any
tester sees it, and the packaging gate should grow a real-document step so this class cannot recur.
