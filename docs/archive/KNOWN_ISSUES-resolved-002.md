<!-- status: archived · updated: 2026-09-10 · class: disposable -->

# KNOWN ISSUES — resolved (archive 002)

Closed entries moved **verbatim** out of `.claude/KNOWN_ISSUES.md` on 2026-09-10 (the file's own shape rule: open issues in full, closed ones as one-line index rows). Same split as archive 001. Numbering is global and never reused.

## KI-52 — deleting a document from the Library left its registry row behind, so the app reported a file as *missing* that the user deleted through the app — **FIXED 2026-08-28**

**Found while cleaning up after the KI-51 work**, by noticing `source_files` held 102 rows against
98 files on disk. All four extras were documents deleted minutes earlier through
`DELETE /api/library/documents/{id}`, and `/api/sources` reported every one of them as `missing`.

**Cause.** `library/documents.py::delete_document` never touches `SourceFile` — the module contains
no reference to it at all. It removes the `Document` row, the chunks, the figure dir and the cached
markdown, and sends the source file to the Recycle Bin, but the registry row that named that file
survives. `registry.scan_sources` then derives `missing` for it, correctly and forever.

**Why it matters, and why it is small.** `missing` is meant to say *something happened to this file
behind the app's back* — a moved folder, an unplugged drive. Using it for a file the user deleted
**through the app** is the app misreporting its own action, and the row cannot be cleared from the
UI. It is not data loss and nothing is orphaned in the index: the document and its chunks are
genuinely gone, so retrieval is unaffected. It is cruft plus a misleading label.

**It is the mirror of KI-51 part 1**, which was the same seam in the other direction: undo removed
the registry row and left the document. That half is now fixed, and `library/add.py::undo_add`
shows the shape a fix takes — resolve the row through its own root, then remove both sides.

**Fixed 2026-08-28, on purpose and with its own tests**, alongside the ADR-046 delete dialog —
which is the change that had to touch this path anyway, so the deferral held exactly as long as it
should have. `documents._forget_source_row` drops the row, matched through `registry.pathkey` and
resolved through **its own root**, so a same-named file under another root is untouched (the
mistake that once deleted an unrelated document out of the library; a test pins it).

**The row goes only when the file goes.** With `delete_file=False` the file is still on disk and no
longer indexed, which is precisely what `derive_status` reports as `new` — the row is still true,
and keeping it is what lets the next scan offer the file again. The bug was never "the row exists";
it was "the row survives the file".

**Verified live:** deleting a document with *Also delete the file* left 0 rows mentioning it and
`/api/sources` back to 98 — where before it lingered as `missing` with no way to clear it.

## KI-51 — "Undo all" did not undo an indexed add: the document, its chunks and the referenced root all survived — **FIXED 2026-08-28** (all three parts; kept for the shape)

**Measured 2026-08-28 against the live 97-document library**, which was restored to baseline
afterwards and verified row-for-row against `data/library.db.bak-20260828-prewalkthrough`
(documents 97 · source_files 97 · source_roots 1 · figures 881). Three separate holes, one theme:
**undo withdraws the registry row and stops there, while the add reached further than that.**

**1 · Undo after index-now leaves the document behind.** Added `walkthrough-note.md` in `copy`
mode with index-now ticked: library 97 -> 98 documents, 39,131 -> 39,135 chunks. `POST
/api/documents/undo-add` then returned `{"undone": 1}`, deleted the file from `data/sources/` and
dropped the `source_files` row — and **left the `documents` row and its 4 chunks in place**.
`/api/library/documents` still listed it; the corpus still read 98 / 39,135. Removing it properly
via `DELETE /api/library/documents/{id}` reported `chunks_removed: 4`, which is what undo should
have done. So after the advertised undo the library holds a document whose file *the app itself
just deleted*: still listed, still retrievable, still citable, and impossible to open.
`library/add.undo_add` only ever touches `source_files` (plus the copied file); it never consults
`documents`. `AddDocuments.svelte:242` offers "Undo all" whenever `applied.added.length > 0`, with
no regard for whether `indexPaths` ran.

**2 · Undo in reference mode never withdraws the reference.** The same file added in `reference`
mode behaved correctly on the way in — nothing copied (the library folder stayed at 97 files), a
new `source_roots` row of kind `referenced`, `origin='referenced'` — and correctly on the way out:
the user's own file was **byte-identical** after undo (sha256 checked before and after), which is
the ADR-014 amendment holding. But the `source_roots` row **survived**. The next `/api/sources`
scan re-discovered the file under that still-registered root as `status: "new"` — so the folder is
still referenced, and the next "index all" would re-ingest precisely what the user undid, without
asking. Un-adding the only file from a referenced root does not un-reference the root.

**3 · The undo was offered while the indexing it would undo was still running — FIXED 2026-08-28.**
`POST /api/ingest` is `status_code=202` and `indexPaths` only checks `r.ok`, so `apply()` returned
as soon as the run was *accepted*, and "Added N" with a live **Undo all** appeared while the worker
was still reading the file.

**Corrected 2026-08-28 — this was never a data-loss bug, and the original filing said it was.**
`routers/sources.py::_running` predates this branch and already 409s both `/api/documents/add` and
`/api/documents/undo-add` while an ingest is in flight, and `status.state` is set to `running`
synchronously *before* the 202 returns, so there is no gap. Clicking Undo mid-run therefore
produced an error, not a deleted file. The real defect was smaller and still worth fixing: the
sheet presented a control that could only fail, and surfaced the refusal as a raw `applyError`.

The fix: the sheet records that *it* started a run (`startedIngest`), derives `indexing` from the
shared `ingestRun` watcher, says "Still indexing — progress is in the status bar", and disables
**Undo all** with that reason until the run ends — so the UI now agrees with the guard the API
already enforced. Verified live: the button greys out on apply and re-enables on completion.
**Done stays enabled throughout** — the sheet is closable; only the control that would fail waits.

*Note the gate is deliberately narrow:* it keys on this sheet's own run, not on any run, and the
API permits only one ingest at a time (409). A run started elsewhere does not disable this undo.

**Why nothing caught it.** The feature had never been driven — this is the first time. `svelte-check`
(205/0) and `node:test` (150) cannot reach a `.svelte` component at all (`apps/desktop/CLAUDE.md`),
and the Python suite exercises `undo_add` against `source_files` only: there is no test that
indexes and *then* undoes.

**Not a regression from the 2026-08-27 review.** All three guards that review added to `undo_add`
(library root · `origin='copied'` · the time window) held exactly as written, and reference mode
did not touch the user's file. This is missing **scope**, not broken correctness in what is there.

**Parts 1 and 2 fixed 2026-08-28, in `undo_add`** — which already resolved each row through its
own root, so it could also be told to finish the job:

* `_purge_document_for` removes the `Document` the add produced, via the extracted
  `library.documents.purge_document_record` (row + chunks + figure dir + cached markdown), so one
  definition of "remove a document" serves both undo and delete. **The guard is the point:** the
  document must resolve to the same path *and* have been added inside
  `UNDO_DELETE_WINDOW_SECONDS`, because a path can carry a document the user has had for months —
  under ADR-047 a replacement even inherits its id — and undo must not destroy that on their
  behalf. A decline is logged, never silent. Without a `chroma_db` handle the row is left alone
  rather than orphaning its chunks: a caller that cannot reach the index cannot finish the job, and
  half-removing is worse than not starting.
* `_drop_root_if_emptied` withdraws a `referenced` root once its last file goes. The library root
  is exempt — an empty library is a normal state, not a stale reference.

**Verified live against the real library**, both directions: an indexed add went 98 → 99 documents
/ 39,705 → 39,706 chunks and came back to **98 / 39,705** on undo, with no `documents` or
`source_files` row left; and a reference-mode add went 1 → 2 roots and back to **1**, with the
user's own file byte-identical (sha256 before and after) and the scan no longer offering it as
`new`. Six unit tests cover the guards, including the two that must *decline*.

**The ADR-014 amendment is untouched:** reference mode still never touches the file.

## KI-49 — a guarantee added at one layer can be defeated by an earlier pass, and the run still exits 0 — **FIXED 2026-08-25, kept for the shape**

**What happened.** ADR-047 gave `_existing_document_id` a source-path fallback so a document keeps
its identity across a re-extraction. It was verified directly against the live library — **97/97
documents kept their id** — and reported as such. The very next full re-ingest **destroyed 767 of
881 figure rows, 1,170 keywords and 381 epistemics rows**, and reported `exit=0` with no error, no
warning, and a plausible-looking log.

**Why the verification was worthless.** `ingest.main()` runs `cleanup_orphans_sqlite` **before**
the document loop. That pass was hash-keyed too: it re-extracts every source, finds the hashes no
`Document` row still matches, and **deletes those rows**, FK-cascading the sidecars. By the time
`_existing_document_id` was called there was nothing left for it to find. The fallback was correct
and unreachable.

**The generalisable trap, which is why this stays filed after the fix:**

1. **A component proved is not a system proved.** The probe called the resolver directly. Nothing
   exercised the *order* the pipeline actually runs in, and the order was the whole bug.
2. **`exit=0` is not a verification.** The destructive path is a `session.delete` inside a sweep
   that is *supposed* to delete things. Nothing distinguishes "removed what it should" from
   "removed everything" without comparing counts to a baseline.
3. **Cheap invariants beat careful reasoning.** One before/after count comparison caught in seconds
   what a reading of the resolver could not. Take a baseline before any bulk operation.

**What now guards it:** `tests/unit/test_cleanup_keeps_reextracted.py` pins gone-vs-stale in both
directions and `tests/unit/test_document_identity.py` pins the resolver — but neither would have
caught this alone. The ordering is covered by `tests/integration/ingest/test_ingest_orphan_cleanup.py`,
which runs `ingest.main()` end to end.

**Two traps found while verifying the fix, both worth knowing independently:**

- **Copying `data/` breaks identity matching.** `documents.source_original` still points at the
  original location, which is exactly what the fallback matches on, so a trial against a copy
  silently creates duplicates instead of updating. Rewrite `source_original`, `source_cache` and
  `figures.image_path` before trusting such a trial.
- **`_chroma_base()` relocates the vector store to `%PROGRAMDATA%` when `DATA_PATH` is non-ASCII**
  (`config.py`, guarding a chromadb persistence bug). A scratch dir under a home containing an
  accented character resolves non-ASCII, so `DOC_DATA_DIR` there yields an **empty** Chroma rather
  than the copied one — the first trial classified nothing and exercised none of the fix while
  appearing to pass.

**Do not** conclude from this that the sweep should never delete. A source file that is genuinely
gone must still end its document; that half is asserted too.

## KI-42 — Marker is unrunnable on this box, so the table-extraction runner is silently broken; `marker-pdf` was never pinned — **FIXED 2026-08-08**

**✅ FIXED same day, three changes.**
1. **Pinned** — `config.MARKER_VERSION = "1.10.2"` (the last 1.x), used as
   `uvx --from marker-pdf==<version>`. Verified end to end: the table runner found **7 tables** in
   `rag_lewis_2020.pdf` in 40 s. **Bump only after an end-to-end run** — the failure mode is a hard
   stop, not a quality regression, and 2.0.0 is the proof.
2. **A non-zero exit is now `MarkerUnavailableError`, and the runner refuses to swallow it.** The
   load-bearing distinction: *a document with no tables still exits 0*, so a non-zero exit is the
   tool failing, not the PDF being unusual — one fact about the machine, not 97 about the corpus.
   The run stops and names the cause instead of grinding out 96 more useless rows.
3. **The report stopped hiding it** — errors sort **first** (they used to sort below every success,
   since ranking was by table count) and error notes are **no longer clamped to 30 characters**,
   which had been cutting the cause off mid-word.

10 guard tests in `tests/unit/test_marker_availability.py`.

**The trap that must not be undone:** the old guard checked that `uvx` **exists**. It does exist —
the failure was two layers deeper, at model-spawn time. **A guard that tests the launcher instead of
the capability reports green on a broken path**, which is the same shape as the RG-012 gate that
restated a contract rather than calling it. If a cheap capability probe is ever wanted, it must
actually run Marker on a real page; `--help` would have passed on 2.0.0.

**The rule this earns:** *an out-of-process escape hatch is still a dependency. Unpinned, it is a
dependency that changes without a commit.*

---

**Original diagnosis, kept because the failure mode generalises:**

**Found:** 2026-08-08, attempting RG-025's Marker-vs-Tesseract comparison. `uvx --from marker-pdf
marker_single` now fails at the **layout** stage — before any OCR — because current `surya` routes
inference through a spawned backend:

- **`vllm`**, auto-selected whenever an NVIDIA GPU is present (so: this box), shells out to
  `docker run`. Docker Desktop is installed but not running, and it would also pull a container image.
- **`llamacpp`**, the fallback (`SURYA_INFERENCE_BACKEND=llamacpp`), needs a `llama-server` binary
  that is not installed; its documented install path covers only macOS and Linux.

**The blast radius is not the comparison — it is `scripts/extract_tables_marker.py`.** That runner
resolves Marker through the *same* command (`eval_marker_tables._marker_command`), so the
high-fidelity table-extraction enrichment path does not run here at all. Nothing surfaced this: the
runner's own guard only checks whether `uvx` **exists**, which it does — the failure is deeper, at
model-spawn time.

**Root cause: `uvx --from marker-pdf` is unpinned**, so every invocation fetches the latest upstream
release and the project's behaviour drifts with it. `scripts/eval_marker_tables.py:85` still
describes running "against the pinned marker-pdf version at build time" — **there is no pin anywhere
in the repo**, so that comment now documents a guarantee that does not exist.

**Fallback while it was broken:** `python -m scripts.extract_tables` (pdfplumber), which the runner
already names — lower fidelity, but it runs. No longer needed; see the fix above.

## KI-41 — the 2026-06-06 chunking sweep compared one configuration with itself six times — **RESOLVED 2026-08-08** (re-swept; lock now measured)

**✅ RESOLVED.** The cause was fixed (KI-38), the silent-failure mode was closed by the sweep's own
preflight (below), and **the experiment was re-run twice on 2026-08-08** — public 10 / paid Haiku
and 97 docs / 35 cases / local Ollama. Both self-audited: **6 distinct geometries recorded per run**
(was: 1 geometry with 6 notes), and `token_input` spans **2529 → 7044** where the void run read
4326.7 on every case of every config. **The defaults survived on evidence** — nothing beat
`2000/200 · 400/50` beyond variance — so the outcome matches the void run's claim by coincidence,
which is precisely why the audit trail rather than the verdict is the thing to keep.
Baselines: `tests/eval/baselines/chunking_sweep_{public,private}_2026-08-08.md`. RG-026 closed.
The account below is kept because the failure mode generalises.

---

**Found:** 2026-08-07, tracing KI-38's blast radius. `scripts/sweep_chunking.py` drives its grid by
passing `PARENT_CHUNK_SIZE` / `CHILD_CHUNK_SIZE` (+ overlaps) in the subprocess environment — and
`config.load_dotenv(override=True)` then replaced all four with `.env`'s values before the ingest
read them (KI-38). Every grid row therefore re-embedded and evaluated **the same corpus**.

**Proven from the stored runs, not inferred.** `case_results.token_input` scales with the size of
the evidence block, so a parent-size change from 1000 to 3000 must move it. Across all **18 runs**
(6 configs × 3 trials, `data/eval.duckdb`):

| | mean `token_input` | min | max |
|---|---:|---:|---:|
| every one of the 6 configs | **4326.7** | **3582** | **5106** |

Per case, the control (parent 2000) and parent 3000 are **identical on all 10 cases** — 3626 / 3984
/ 4816 / 4734 / 5106 / 3582 / 4625 / 4406 / 4540 / 3848, twice. The field is live, not a constant:
other experiments in the same DB read 4372.7 (`candk20`) and 4615.8 (the post-ADR-036 re-measure).

**Consequences.**
1. **`docs/decisions.md` / the locked-settings row overstate what is known.** "6-config sweep on
   public corpus — none beats it" is now "never measured"; the row is corrected to say so.
2. **The spread the baseline reports is the harness's noise floor, and that part is a free result
   worth keeping.** `contains_all` 0.906–0.933 and `llm_judge` 3.793–3.951 were produced by
   **identical inputs**, so at n=3 on the public 10 that band is noise, not signal. The baseline's
   own reading ("within the trial-to-trial noise bands") was right for a reason it did not know.
3. It cost ~6 full corpus re-embeds on the RTX box.

**Not evidence that the defaults are wrong** — only that nothing supports them. Re-running is now
meaningful (KI-38 is fixed); it is a GPU-day of re-embedding, so it is a deliberate call, not a
cleanup. **The "make the run record what it ran" precondition is done** (2026-08-07): every run now
records the 13 run-defining settings via `eval/run_settings.py`, readable with
`Store.run_config(run_id)`. Old rows are untouched and still record nothing — deliberately, since
substituting today's values would turn "unknown" into a false record.

**This can no longer happen silently (2026-08-08).** `sweep_chunking.preflight` resolves every arm's
settings in a subprocess under that arm's own environment — calling `run_defining_settings`, the
same function that writes the record, so the gate cannot disagree with the thing it guards — and
exits 1 before any ingest unless each arm gets the four values it asked for and no two arms resolve
identically. It runs in `--dry-run`, so the check costs ~4 s. **The trap that must not be undone:
the probe has to be a subprocess** (`config` resolves the environment once at import, and the
sweep's ingest and eval are subprocesses — an in-process read tests nothing), and there is
deliberately **no bypass flag**: a failure that reads as a normal result will be skipped on exactly
the run that needed the check.

**The general rule this earns:** *an experiment that does not record the setting it varies cannot be
audited — and a driver that passes its variable through a channel something else can overwrite will
fail silently, in the direction of "no effect".*

## KI-40 — the extraction cache was keyed on mtime alone, so an extractor fix never reached a library that already ingested — FIXED 2026-08-07

**✅ FIXED same day.** `is_cache_fresh` now compares an **extraction fingerprint** as well as the
mtime, and `write_cache` writes the pair (`<name>.md` + `<name>.md.fp`) so a half-written entry
cannot exist. A cache with no recorded fingerprint is **stale by definition** — which is the point:
those are exactly the entries holding output no current version would produce.

**Bump-free, following the `sparse_index` precedent** (which hashes the tokeniser's source so a
change invalidates "without anyone remembering to bump a constant"). The fingerprint covers:
- **every extractor function's `co_code`** — a logic change invalidates automatically. *Bytecode*,
  not source: PyInstaller ships `.pyc` so `inspect.getsource` would raise in the frozen build, and
  comments/docstrings — which cannot change output — must not re-extract an entire library;
- **the tunables bytecode cannot see** — a module-level constant is referenced by *name*, so its
  value never appears in `co_code`. `_TEXT_LAYER_KEPT_MIN` is exactly that, and it changes output;
- **`config.PDF_EXTRACTOR`**, and **the PyMuPDF / PyMuPDF4LLM versions** — a dependency upgrade
  changes extraction output with no code change of ours, which is a real cause, not a hypothetical;
- **`_EXTRACTION_VERSION`**, a manual escape hatch for the residue (a changed string literal alters
  output without altering bytecode).

**The cost is visible before it is paid.** The ingest plan is stat-only, so it reports the work
without extracting anything: measured here, `--dry-run` went to `would_reembed=97` **in 8 s**. Each
re-extraction also logs `reason="extractor_changed"`, so a one-off slow ingest is explained rather
than mysterious. Guarded by `tests/unit/test_extraction_cache_fingerprint.py` (9 cases, one per
component and per drift mode).

**One-time cost, by design.** The first ingest after an extractor change re-extracts the library —
that *is* the fix, and extraction remains the binding scale constraint (~41 h projected at 10k
documents), so it must never become a per-launch cost. It is per *change*, and only when the change
is real.

**Historical (kept — this is why the fingerprint exists):**

## KI-40 (original report) — the extraction cache is keyed on mtime alone — 2026-08-07

**Found:** 2026-08-07, shipping the text-layer fallback (EX1). The fix recovers three documents from
~0 to 46k / 89k / 778k characters — **on a fresh ingest**. On an existing library it changes
nothing, silently.

**Cause,** `ingest/cache.py`:

```python
def is_cache_fresh(original: Path, cached: Path) -> bool:
    if not cached.exists():
        return False
    return cached.stat().st_mtime >= original.stat().st_mtime
```

The cached `.md` is a derived artifact of **(source file, extractor, extractor config)**. Freshness
tracks only the first. So a user who upgrades keeps the *old* extraction of every document they
already have — for good — and the only cure is manually deleting `data/cache/*.md`, which is
undocumented and not exposed anywhere in the UI.

**This is what makes it more than a local inconvenience:** every extraction improvement this project
ships is invisible to exactly the people who have already used it. KI-29 (page markers reaching the
embeddings) and KI-14 (image placeholders) both changed extraction output, and both had the same
hole. The corpus that most needs a fix is the one that will never receive it.

**The pattern to copy already exists in this repo.** `sparse_index` fingerprints its inputs and
logs `sparse_index_stale: corpus or tokenizer changed; rebuilding`. The cache wants the same: a
fingerprint of the extractor identity + relevant config alongside each cached file, compared on
read. A version constant bumped by hand is enough — it does not need to be automatic, it needs to
*exist*.

**Do not fix this by dropping the cache.** Extraction is the binding scale constraint (~41 h
projected at 10,000 documents, `docs/performance.md`); re-extracting everything on every launch is
not an option. The point is to invalidate **when the extractor changed**, not always.

**Interim, for anyone with an affected library — and mind the second step:**

```bash
rm data/cache/<name>.md                              # force re-extraction
python -m doc_assistant.ingest                       # NOT --files
```

**Use a plain ingest, not `--files`.** Re-extraction changes the document's content hash, and
`_existing_document_id` matches on `doc_hash` — so the run mints a **new** `Document` row and leaves
the old one behind. `--files` *skips orphan cleanup* (documented, in its own `--help`), so the stale
row and its chunks survive: measured here as 97 → **100** documents, each affected file appearing
twice, once healthy and once broken. A plain ingest runs `cleanup_orphans_sqlite`, which is built
for exactly this — it removes "the pre-change hash of a document whose *content* changed" — and
reconciles it. Nothing is lost either way, but the intermediate state is a library that lists the
same paper twice with different health.

## KI-39 — the readiness gate gives up after 60 s, permanently, and then tells the user to run `just api` — **FIXED 2026-08-06** (entry corrected 2026-08-11)

**✅ FIXED same day it was filed** (DEVLOG `2026-08-06 (2)`), and this heading said OPEN until
**2026-08-11**, when a release-prep read of the app's failure text found the code already correct.
All three parts of the fix shape below landed: the readiness `$effect` in `App.svelte` polls
**without a deadline**, backs off between attempts (`backoffDelayMs` / `startupPhase` in the pure,
tested `lib/shell/startup.ts`), and never enters a terminal state — a backend that turns up at
minute three still takes the app to ready. `StatusBar.svelte` now renders *"starting the engine…"*
→ *"starting the engine — a first launch can take a minute…"* → *"still starting — retrying.
Restart the app if it never arrives."*, none of which names a developer command. The
`dev_commands` check in `release_preflight` guards the regression.

**What is genuinely still open is a different question, and it belongs to RG-010:** the cold-start
*distribution* on non-VM hardware has never been recorded (one measurement, one fast box, ~30 s
against a 60 s budget). That number decides whether the PyInstaller spec should move
onefile→onedir. Tracked as the RG-010 measurement, not as this defect.

**Found:** 2026-08-06, characterising the "first-launch dead-backend window" (baton item 2) with
the RG-012 numbers in hand. Three facts that are individually defensible and jointly a bad first
five minutes for a beta tester.

**1 · The budget is 60 s, against a first launch that unpacks 1.5 GB.** `App.svelte`'s readiness
`$effect` polls `/api/health` **60 times at 1 s** and then sets `shell.status = 'down'`. The frozen
sidecar is a PyInstaller **onefile** binary — it extracts ~1.5 GB to `%TEMP%\_MEI*` *before* uvicorn
binds, then loads the embedding + reranker models.

**2 · It never retries.** The `$effect` reads no reactive state before its `await`, so it runs
**once per mount**. After the loop falls through there is no timer, no retry, and no control to
trigger one. The only recovery is quitting and relaunching the app — which restarts the 1.5 GB
extraction, though the OS file cache usually makes the second attempt much faster.

**3 · The message is a developer instruction.** The status bar renders, verbatim:

> `backend unreachable. Run just api`

`just` is a task runner the tester does not have, `api` is a recipe in a repository they do not
have. **The app's only failure message asks for something that cannot exist on the machine it is
shown on** (`StatusBar.svelte:25`).

**How close is the margin?** Measured on the RG-012 Sandbox, 2026-08-06: **health at ~30 s — half
the budget** — on an idle VM with an NVMe disk and the file cache *warm from the install that had
just written those same bytes*. `docs/desktop-packaging.md` §5 already names 30 s as the threshold
at which the spec should move onefile→onedir. A tester on a spinning disk, or with an antivirus
scanning a 1.5 GB extraction to temp (Defender does scan `%TEMP%`), plausibly exceeds 60 s — and
that case is not slow, it is **terminal and misdirecting**.

**Not yet established:** the actual distribution of cold-start times on non-VM hardware — this is
one measurement on one fast box, and RG-010 (cold-start) has never been recorded properly. That
number is what decides whether the ceiling should be raised or the spec should go onedir.

**Fix shape (small, and none of it needs a measurement first).**
- **Never give up.** Keep polling with a backoff (1 s → 5 s) indefinitely; a backend that has not
  answered yet is not a backend that will never answer.
- **Say what is happening.** "Starting the engine — first launch unpacks a large model bundle and
  can take a minute" beats a silent dot, and after ~45 s it should say so explicitly.
- **Never print a dev command in a shipped build.** Replace with something a user can act on
  (retry, view the log, report), and keep `just api` behind a dev-only condition if it is kept.

**Do not "fix" this by raising 60 to 120.** That trades one arbitrary cliff for another; the defect
is the existence of a terminal state, not the size of the number.

## KI-38 — `load_dotenv(override=True)` made **every** env-var override silently ineffective — FIXED 2026-08-07

**✅ FIXED.** `config._load_env` replaces `load_dotenv(override=True)` with the narrow rule the
override was actually for: **a non-empty process environment variable wins; `.env` fills in the
absent and the empty.** The empty-`ANTHROPIC_API_KEY` shadowing that motivated the original override
still cannot happen — that half has its own test.

**It was wider than "the credit-leak hole" recorded below.** The override applied to every key
`.env` defines (19 on this box), including `PARENT_CHUNK_SIZE` / `CHILD_CHUNK_SIZE`, which
`.env.example` ships **uncommented** — so `scripts/sweep_chunking.py`, whose entire mechanism is
passing those in the subprocess environment, silently swept nothing. See **KI-41**: the sweep that
locked the chunk sizes compared one configuration with itself six times, proven from the stored runs.

**Reproduced before and after** (`CHILD_CHUNK_SIZE=999 PARENT_CHUNK_SIZE=111 LLM_PROVIDER=ollama`):

| | `LLM_PROVIDER` | `CHILD_CHUNK_SIZE` | `PARENT_CHUNK_SIZE` |
|---|---|---|---|
| before | `anthropic` | 400 | 2000 |
| after | `ollama` | 999 | 111 |

With no env override, `.env` still wins over the code defaults and the key still resolves — the
before/after of the *normal* path is unchanged. Guarded by
`tests/unit/test_config_env_precedence.py` (7 cases). **Non-vacuous:** restoring the old behaviour
fails exactly the two tests that encode this issue, and leaves the empty-key guards passing.

**⚠ Residual, by design, and worth knowing before a "local" run.** `.env` here sets
`REVIEWER_PROVIDER=anthropic`, so `REVIEWER_PROVIDER_PINNED` is true and `resolve_reviewer` refuses
to follow the generation provider (ADR-011 U1c, deliberate — it protects cross-run comparability).
Forcing only `LLM_PROVIDER=ollama` therefore still bills the **reviewer** on flagged answers: a
*silent partial* leak that a short smoke run never triggers. The fix makes the cure available where
it was not before — pass **`REVIEWER_PROVIDER=ollama` too**, and the environment now wins.

<details>
<summary>Original report (2026-08-05) — kept for the diagnosis</summary>


**Found:** 2026-08-05, setting up the KI-36 measurement. `config.py:14` calls
**`load_dotenv(override=True)`** — deliberately, and the reason at the line is sound (a host that
exports an *empty* `ANTHROPIC_API_KEY` must not shadow the real key). The consequence is not
recorded anywhere: **for every key `.env` defines, a real environment variable is ignored.**

`.env` on this box defines `LLM_PROVIDER=anthropic` and `REVIEWER_PROVIDER=anthropic`. So

```
LLM_PROVIDER=ollama python -m whatever      # silently runs on ANTHROPIC, and bills
```

The existing guard (`assert_provider_intent`, and the sidecars defaulting to Ollama *explicitly*
rather than inheriting `LLM_PROVIDER` — the KI-4 lesson) protects the **runners that take
`--provider`**. It does not protect anything that reaches for the answer path directly, which is
what an ad-hoc measurement, a notebook, or a new script naturally does.

**What actually works** — the same seam the desktop provider switch uses, which sits *above* config:

```python
from doc_assistant import app_settings, config
app_settings.get_llm_selection = lambda: ("ollama", "llama3.1:8b")   # generation
config.REVIEWER_PROVIDER, config.REVIEWER_PROVIDER_PINNED = "ollama", True   # and the reviewer
```

**The reviewer needs its own line.** `REVIEWER_PROVIDER` is set in `.env`, so
`REVIEWER_PROVIDER_PINNED` is true and `resolve_reviewer` refuses to follow the generation
provider — by design (ADR-011 U1c). Forcing generation local while the reviewer still bills is a
*silent partial* leak: it fires only on flagged answers, so a short smoke run never sees it.

**Do not "fix" this by dropping `override=True`** — that re-opens the empty-key shadowing the
comment describes. The fix, if one is wanted, is to make the answer path's provider resolution take
an explicit argument rather than read module state.

**Adjacent, same session:** a stale `OLLAMA_HOST=0.0.0.0` in the process environment (left from the
RG-012 Sandbox run) fails every local call with `WinError 10049 — The requested address is not
valid in its context`. `0.0.0.0` is a valid *bind* address and an invalid *connect* address; the
error names neither Ollama nor the variable. The app surfaces it correctly ("No Ollama server
answering at 0.0.0.0") and the readiness gate then blocks the composer, so the whole app looks
broken. `.env` cannot override it (`OLLAMA_HOST` is commented there, so the process env wins —
the same `override=True` asymmetry as above); `.claude/launch.json` has an `api-local-ollama` entry
that pins it via `uv run --env-file`.

**Same family, cost a false conclusion 2026-08-05:** the dev **sidecar does not hot-reload**. Vite
HMR updates the frontend instantly, so a backend-rendered string (the provenance card, any
markdown block built in `chat_controller`) keeps showing the OLD text after a Python edit and the
change reads as broken. `.claude/launch.json`'s api entries do not pass `--reload`. **Restart the
sidecar before believing a backend-rendered string** — a frontend hot-reload proves nothing about
the backend.

*(Note on the original's proposed fix: it ruled out "dropping `override=True`" — correctly, since
that re-opens the empty-key shadowing — and proposed threading an explicit provider argument through
the answer path instead. The fix taken is neither: the override was **narrowed to the empty case**,
which is what its own comment justified. That cures the whole class in one place, including the
chunk-size keys an answer-path argument would never have reached.)*

</details>

## KI-37 — "unsupported" named two different things in one card, and a correct refusal got the accusing one — FIXED 2026-08-05

**Found:** 2026-08-05, alongside KI-36. Two independent integrity signals render into the *same*
answer view using the same word for different things:

- `chat_controller/helpers.py:243` — the **LLM reviewer's** verdict: `unsupported claims: 0`,
  meaning *nothing in the answer contradicts or outruns the evidence*.
- `chat_controller/helpers.py:418` — the **deterministic marker layer's** badge: `unsupported`,
  meaning *this sentence carries no `[n]` that resolves to a retrieved source*.

The RG-012 turn showed both at once: **"faithfulness 5/5 · citation density 4/5 · unsupported
claims: 0"** directly above **"⚠ 13 claim(s) to review … *(unsupported)*"**. A user cannot
reconcile those, and there is no reading of the card that explains it.

**The second half is worse.** `claim_marker` returns `unsupported` for any sentence with no
resolving citation — including a *correct refusal*. Measured: 3 of 27 answers correctly declined
(*"the provided text does not mention …"*) and were rendered as **16 claims "to review
(unsupported)"**. The integrity layer accuses the model of ungrounded claims precisely when it did
the right thing, and the honest-hedging behaviour the prompt asks for is the one it punishes.

**✅ FIXED 2026-08-05 — renamed at the presentation boundary only.** `MARKER_UNSUPPORTED` and the
persisted `AnswerClaim.marker` are untouched (`test_adjudication_persistence` still pins the marker
triple), so no migration and no history rewrite. New `chat_controller.helpers._claim_badge` splits
the badge the same three ways the RG-012 gate now does, so the whole stack tells one story:

| what the marker layer found | badge |
|---|---|
| sentence carries no citation token at all | **`uncited`** |
| cites only numbers that map to no retrieved source | **`unresolved citation`** |
| cited a real source, weak reranker score | `weakly grounded` (unchanged) |

The split is `Claim.citations`, which is empty *only* when nothing was cited — no new state, no
inference. `ClaimReview.svelte` now tests for the one benign label (`weakly grounded` → `weak`) and
defaults everything else to `bad`, so a future severe badge cannot silently render as mild.
Guarded by `test_uncited_claim_is_flagged` and
`test_claim_citing_only_a_nonexistent_source_reads_unresolved_not_uncited`.

**Do not "improve" this with a refusal detector.** Classifying "is this sentence a refusal" is a
heuristic wrong in both directions on a real corpus; naming the badge accurately makes the refusal
case merely *true* instead of accusatory, with no detector to get wrong.

**Still open, deliberately:** the reviewer's own `unsupported claims: N` line keeps its wording —
it is the correct word for what *it* measures (a claim outrunning the evidence). The collision is
gone because the other side stopped using it, which is the honest direction: the structural layer
never had grounds to say "unsupported".

## KI-35 — the RG-012 gate re-implemented the citation contract more strictly than the app, and scored a PASSING turn as FAIL — FIXED 2026-08-05 (the filed diagnosis was wrong; corrected same day)

**Filed 2026-08-05** as *"`llama3.1:8b` cites as `[Source 1]`, not `[1]` — every claim then reads
UNSUPPORTED"*, with the impact *"the integrity layer inverts on the shipped default provider"*.
**Every causal claim in that filing was wrong.** The correction is the issue, so it is kept here in
full rather than archived as a resolved row.

**What was actually measured** (same day, against the run's own recorded `result.json`, using the
shipped code):

| Filed claim | Measured |
|---|---|
| "the per-claim marker layer and the citation links both parse `[n]`" | **False.** Both have tolerated `[Source n]` / `[Sources 2, 4]` / `[2, 4]` **since 2026-07-14** — `synthesis._CITATION_TOKEN_RE` (`synthesis.py:35`) and `Markdown.svelte`'s `CITE_BODY` (`:52`), added for exactly this. |
| "a `[Source 1]` citation matches nothing" | **False.** `cited_source_numbers(answer)` → `[1, 5, 2, 1]`; `audit_citations` → `valid=[1,2,5]`, `malformed=[]`, `out_of_range=[]`, `clean=True`. |
| "every claim is rendered `unsupported`" | **False.** All four *cited* sentences scored `ok`/`weak` — claim #4, which carries `[Source 5]`, is `weakly grounded`. The 13 flagged claims are the sentences that genuinely carry **no citation of any form**. |
| "the integrity layer inverts / reports the opposite of the truth" | **False.** It reported the truth: the model cited **4 of 16 sentences**. |
| "the prompt's anticipated confusion caused it" | Unsupported. 27 fresh turns on the same model produced **`[Source n]` 0/27 times** — the observed form was a one-off, not a behaviour. |

**The actual defect was in the gate, not the app.** `rg012-run.ps1` counted `'\[\d+\]'` only —
a *stricter* contract than the app deliberately implements — so it logged
`inline citation markers: 0` and `FAIL - answer produced but not cited`. That false verdict was
then filed as an app bug. **RG-012 Tier-2 had in fact passed.**

**Fix (2026-08-05).** The gate now mirrors the app's own token and reports the three outcomes
separately, because they need completely different fixes:
- `resolved > 0` → **PASS** (noting how many used the non-canonical `[Source n]` form);
- bracket-with-letters that no parser resolves → **FAIL (wrongly-formatted)** — a prompt/parser problem;
- no bracket of any form → **FAIL (uncited)** — a grounding problem.

Re-scored against the archived `result.json`: `resolved=4 canonical=0 labelled=4 unresolvable=0` →
**PASS**, exactly agreeing with the Python parser's `[1,5,2,1]`.

**The rule this cost a session to learn — a verification gate must call the contract, never restate
it.** A gate that re-implements the thing it checks will drift from it, and because a gate is
trusted, its false verdict is filed as a defect in the code it was supposed to protect. This is the
same class as KI-34 ("booting a frozen binary proves nothing about its data files") one level up:
there the gate tested too little, here it tested something the app never promised. **Do not
re-derive `[n]` anywhere.** The forms are pinned by `tests/unit/test_synthesis.py`
(`cited_source_numbers`) and the parser lives in exactly two places, deliberately kept identical.

**What the investigation found instead — see KI-36.** Format was never the problem; **citation
*coverage*** is. Pooled over 27 healthy-corpus cases on `llama3.1:8b`: **79/217 sentences cited
(36.4%)**, and starkly bimodal — 11 answers cite *nothing at all*, 9 cite ≥85%.

**Evidence:** ⚠ **the original artifacts were destroyed on 2026-08-06** — `out\` was cleared for the
re-run after an archive command was rejected as a whole by a path-protection guard (so its
`New-Item`/`Copy-Item` never ran) and the delete was issued without checking the copy existed.
`Remove-Item` does not use the Recycle Bin; no shadow copies. **A partial reconstruction, recovered
verbatim from the session that read them, is at
`C:\rg012-host\out-2026-08-05-RECONSTRUCTED\README.md`** — complete for `result.json`'s answer,
`flagged_claims` and `provenance_card_md`, complete for `settings.json` (it was echoed inline into
the log), lines 3-26 of `rg012.log`, fragments only of `chat-stream.txt`.
**The conclusion is not at risk:** every decisive number was transcribed into this file and the
DEVLOG *before* the loss, and the corrected gate was re-scored against the real `result.json` while
it still existed (`resolved=4 canonical=0 labelled=4 unresolvable=0` → PASS).
**The lesson is the same one this issue is about:** verify the thing you are relying on before
acting on it. Copy, *confirm the copy*, then delete.
Also: the corrected gate at `C:\rg012-host\script\rg012-run.ps1` · coverage run recorded in KI-36.

## KI-34 — the frozen build could not ingest ANY PDF: `collect_all("fitz")` misses `pymupdf`'s data files — FIXED 2026-08-05 (rebuild pending confirmation)

**Found:** 2026-08-05, running **RG-012 Tier-2** for the first time — Provenote 0.4.1 installed on a
clean, Python-free Windows Sandbox. Install fine, app launches, backend healthy, `/api/setup` fully
correct. Then ingest of 3 PDFs: **`added=0, errors=3`**, in **0.65 s** — far too fast to have
attempted extraction. Every file failed identically:

```
[Errno 2] No such file or directory:
  ...\Temp\_MEI82922\pymupdf\layout/resources/onnx/layout_rf2.4.1+imf1.yaml
```

**Cause.** `scripts/doc_assistant_api.spec` collected **`"fitz"`** only. `fitz` is the *legacy import
shim*; modern PyMuPDF's real distribution directory is **`pymupdf/`**, and it carries data files read
at extraction time — 7 of them under `layout/resources/onnx/` (~19 MB of `.onnx` plus the `.yaml`
the error names). `collect_all("fitz")` bundles the shim and none of that, so the frozen build
**imports cleanly and then fails every single PDF at runtime**.

**Fix.** Add `"pymupdf"` alongside `"fitz"` in the spec's `collect_all` list. Verified the data
exists to be collected: `pymupdf/layout/resources/onnx/` holds 7 files including
`layout_rf2.4.1+imf1.yaml` (7,406 bytes).

**Why nothing caught it — the part worth carrying.** It is **invisible from source**: site-packages
has the file, so the whole test suite, the eval harness, the desktop dev loop and even the v0.4.0
**WSL clean-room run** all pass. Only a *frozen* build on a box without the package can fail this
way. **The import-time smoke test is not enough either** — the standalone sidecar smoke this session
ran (`/api/health` → 33,105 chunks) passed happily, because the missing file is read on the
*extraction* path, not at import. **A packaging gate must exercise a real document end-to-end, not
just boot.**

**Class, not instance.** Any dependency with a runtime-read data directory can do this — the spec's
own header already says "add to `hiddenimports`/`datas`, repeat". The generalisation: when a package
has both a legacy alias and a real distribution name, **collect the real one**; and when it ships
non-`.py` resources, importing it proves nothing about whether they were bundled.

**Do not "simplify" the spec by dropping either name** — `fitz` is still what the code imports,
`pymupdf` is what carries the data.

## KI-28 — thinking models returned an EMPTY completion through `OllamaClient` — FIXED (2026-07-26)

**Found:** 2026-07-26, trying to run RG-015's precision pass on `qwen3.5:9b`. Every one of the
first 3 items logged `taxonomy_propose_unparseable`, 0 proposals — which reads as "this model
cannot do the task". It was not the model.

**The trap, and it will recur as local models move to hybrid-thinking by default.** A thinking
model emits its reasoning into Ollama's **separate `message.thinking` field**, and that reasoning
is drawn from the **same `num_predict` budget** as the answer. `OllamaClient.complete` sent
`num_predict=max_tokens` and read only `message.content`. With
`taxonomy_propose.DEFAULT_MAX_TOKENS = 256` the reasoning consumed the entire budget, so the call
came back `done_reason="length"` with `content=""`. Measured directly against `/api/chat`:

| `think` | `num_predict` | `done_reason` | `content` |
|---|---|---|---|
| default (unset) | 256 | `length` | `''` |
| `false` | 256 | `stop` | `{"choice": 3, "confidence": 1}` (14 tokens) |
| `true` | 2048 | `stop` | `{"choice": 3, "confidence": 1}` |

**Why it was expensive to diagnose:** `complete()` returns `str`, and *every* caller parses that
string. An empty string is therefore indistinguishable downstream from "the model answered
nonsense" — the failure surfaced at the caller as a model-quality problem, four layers from its
cause. It would have hit the reviewer, the eval judge, `gap_suggest` and `taxonomy_propose`
identically, and silently.

**Fix (`src/doc_assistant/llm.py`):** `OllamaClient(model, *, reasoning=False)` — `reasoning` maps
to Ollama's `think` and is **off by default**, because this adapter exists to return one short
JSON object. `reasoning=True`/`None` stay available for a caller that wants the trace (and must
raise `max_tokens` to match). Second half of the fix: an empty completion now logs
`ollama_empty_completion` with model + budget + reasoning flag, so this class of failure can never
again be misread as model incompetence. Guard tests in `tests/unit/test_llm.py` (default is
`False`; override reaches the client; warning fires when blank and stays silent when not).

**Verified live:** the full 97-document taxonomy pass through the *shipped* runner —
220 calls, 110 proposals, **0 unparseable, 0 empty completions** — reproducing a shim run
placement-for-placement.

**`think: false` is safe on non-thinking models** — verified against `llama3.1:8b` and
`qwen2.5:7b`, both answer normally with it set. (Ollama accepts the key regardless; older servers
rejected `think` on non-thinking models, so re-check if the Ollama version is ever pinned back.)

**STILL OPEN — the streaming answer path has the same exposure, differently.**
`pipeline.build_chat_model` constructs `OllamaLLM` against `/api/generate`, **not** this adapter:
no `num_predict` cap (so it cannot fail the silent-empty way) and no reasoning flag — meaning on a
thinking model the reasoning can **leak into the streamed answer text**. Not changed here because
it alters user-facing output and deserves its own call.

## KI-8 — PC→baseline marker mapping (PR-M1) is coarse at parent boundaries — RESOLVED (2026-07-21, E1.1)
- **Resolved (2026-07-21, E1.1 — re-projection, option 2):** `build_epistemics` now projects the
  marker sidecar onto **both** segmentations — baseline chunks and PC parents — via the same
  structural attribution, each row carrying an authoritative `chunk_key` (`{doc}:{idx}` or
  `{doc}:p{parent_index}`, a new additive column). `_chunk_key` returns the parent key for a PC
  parent, and `_attach_markers` joins **both** modes by a direct key lookup against
  `load_epistemics_index()` — the coarse `markers_for_parent` text-containment (and
  `load_marked_chunks`/`MarkedChunk`) is retired. The blanket-`except` also gained a WARNING log
  (silent failure + always-on strip = silently-lying UI). Guard: `test_reprojects_onto_pc_parents_
  keyed_by_parent_index` + the inverted `test_chunk_key_parent_child_chunk_uses_parent_key`. Live $0
  probe on a copy (one injected stance): 196 PC-parent keys join directly (was 0), and 28/196 marked
  parents (14% here) would have been missed by containment. **Do not** reintroduce containment
  mapping. (Ready to commit — not yet committed.)
- **Symptom:** In the default parent-child retrieval mode, the live 7d marker chip maps a marked baseline
  chunk onto a retrieved parent by **text containment** (`epistemics.markers_for_parent`): a parent gets a
  marker if it *contains* a marked chunk's text. The two collections are independent segmentations, so a
  parent spanning a marked chunk plus three clean ones is marked as a whole — over-attribution within the
  parent. A marked chunk straddling two parents marks both.
- **Why it's acceptable (for now):** markers are an **advisory chip, not a gate** (inform-don't-block), and
  over-attribution is fail-safe — it points the user at a real contested concept *in that passage*. The
  marker never changes synthesis, ranking, or the answer (byte-identical when absent).
- **Status:** chosen in PR-M1 ADR-1 over the heavier alternative (re-project `chunk_epistemics` onto PC
  parents — a second projection + migration + its own attribution-quality validation). That precise
  re-projection is the documented upgrade **if** containment proves too coarse on real data.
- **Compounding caveat:** marker *quality* upstream still comes from the superseded open-vocabulary graph
  (KI-7) — `contested` is local-model-noisy. M1 surfaces what the sidecar holds; it does not fix extraction.
- **Mostly moot in practice since 2026-07-02 (PR-R7):** the live chip is now default-OFF
  (`EPISTEMICS_MARKERS_ENABLED=false`, ADR-005, superseded by ADR-027), so this containment coarseness only bites when a user opts
  the markers back on. The precise re-projection upgrade rides with Node B, alongside KI-7 retirement.
- **Update (2026-07-16, docs review):** two bullets above are outdated. (a) The "compounding caveat" —
  the open-vocabulary graph is gone (KI-7 RESOLVED 2026-07-07, G1: `concept_graph.py` deleted); marker
  data now sources from the curated Node-A/B `concept_skeleton`, and attribution actually reaches chunks
  since the KI-15 label fix (G7). (b) The "mostly moot / default-OFF" bullet — G1 flipped
  `EPISTEMICS_MARKERS_ENABLED` back to **default-ON** (superseding ADR-005), so the containment
  coarseness is live again by default. The issue this entry tracks (PC-parent containment mapping is
  coarse) is unchanged and still OPEN; the re-projection upgrade remains the documented fix.
- **Update (2026-07-19, scale review — the direction claim above is arithmetically wrong):**
  "a marked chunk straddling two parents marks both" is unreachable — containment is a strict
  full-substring test (`knowledge/epistemics.py:234`) and a `BASELINE_CHUNK_SIZE=1000` chunk can
  never fit inside a `PARENT_CHUNK_OVERLAP=200` overlap, so a straddling chunk is contained in
  **neither** parent and its markers silently vanish. The real failure mode is systematic false
  *negatives* (order ~40% of marked chunks at these sizes), not fail-safe over-attribution — in
  the default-ON, default-PC configuration. The documented upgrade (re-projection, option 2)
  or overlap-based matching fixes it. See `docs/REVIEW_2026-07-19_scale-robustness.md` WE-7.
- **Pointer:** `docs/archive/pr-m1-epistemics-markers.md` ADR-1 (option 2 = the re-projection upgrade).

## KI-17 — stochastic gap rows outlive their concept → orphaned gaps served to the graph UI (2026-07-18, RESOLVED 2026-07-21, E0.2)
- **Resolved (2026-07-21, E0.2):** `gaps._reconcile_stochastic_gaps(live_ids)` deletes stochastic
  rows whose `concept_id` is not in the `graph_include`-filtered `load_concepts()`, **hoisted to run
  unconditionally on every `build_gaps --apply`** (the placement correction — it must reach a
  deterministic-only apply). A reconcile, not a blanket delete: a promotion on a *live* concept
  survives; only orphans are reaped. Guard test `test_orphaned_stochastic_gap_is_reconciled_away`
  (excluded-anchor gap gone, live-anchor promotion kept). Live $0 probe on a copy of the real DB:
  1 orphan reaped, 1 live promotion survived. **Do not** move the reconcile back inside the
  `suggest` branch. (Ready to commit — not yet committed.)
- **Symptom:** `load_graph_view()` serves **27** gaps against a **13**-node skeleton; **10** of them
  (all `kind="suggested_concept"`, all `determinism="stochastic"`) carry a `concept_id` that resolves
  to no node. The view's own report disagrees with the sidecar: `build_gaps --apply` printed
  "Total gaps: 15 · Rows written: 15", but the route returns 27. Surfaced by the ADR-018 rescope
  (357 → 13 graph concepts, 2026-07-18); the 10 orphans were generated on 2026-07-08 against the
  old 357-concept vocabulary.
- **Cause:** the two gap classes have different write disciplines (this is ADR-017's own finding,
  read from the other end). `gaps.py:257` **delete-and-replaces** deterministic rows, so those
  self-heal on every rebuild; `_write_stochastic_gap_rows` (`:273`) is a **status-preserving
  upsert**, which is correct for not losing a user's triage — but it has **no delete pass for rows
  whose concept left the vocabulary**. Nothing reconciles a stochastic row against the current
  vocabulary, so it is immortal. Deleting a `Concept` (or, now, excluding it) strands its gaps.
- **Impact:** PR-G2a's index badges gaps by looking the concept up, so an orphan renders no row —
  it inflates the gap *count* without being reachable. Worse for **PR-G2b**, which promotes gaps to
  a first-class destination with a per-row triage action: a row you cannot resolve to a concept is a
  row you cannot dismiss or promote.
- **Workaround:** none needed for correctness today (the orphans are invisible in the index, not
  wrong answers); read the `build_gaps` report — not `len(view.gaps)` — as the true gap count until
  fixed.
- **Candidate fix (PR-G2b territory, ADR-017 C1):** in `_write_stochastic_gap_rows`, delete
  stochastic rows whose `concept_id` is not in the current vocabulary before the upsert — a
  reconcile pass, not a blanket delete, so triage on a *live* concept still survives. Guard test:
  a stochastic gap on a concept that is then excluded (`set_graph_include(cid, False)`) →
  `build_gaps --apply` → the row is gone, while a stochastic gap on an included concept keeps its
  status. Decide alongside the C1 override sidecar, since both concern what a rebuild may destroy.
- **Placement correction (2026-07-19 review):** the reconcile as sketched sits inside
  `_write_stochastic_gap_rows`, which only executes under `suggest and apply` and early-returns on
  zero suggestions — a deterministic-only `build_gaps --apply` (this KI's own repro) would never
  reach it. Hoist it to run unconditionally on every `--apply`, keyed against the
  `graph_include`-filtered `load_concepts()` (excluded = removed; the unfiltered table would fail
  this KI's own guard test). See `docs/REVIEW_2026-07-19_scale-robustness.md` (GP/KI-17 check).

## KI-20 — concept curation hard-deletes vocabulary where ADR-018 mandates demote — RESOLVED (2026-07-21, E0.1)
- **Resolved (2026-07-21, E0.1):** artifact + `classify_noise` verdicts route through new
  `concept_curation.demote_concepts` (`graph_include=False` — keeps the row, its aliases, and its
  ADR-015 keyword family) via the new `apply_plan` seam the runner drives. `remove_concepts` stays
  as the reserved, separately-confirmed hard-delete primitive, no longer wired to the noise stages.
  Guard tests `test_noise_verdict_demotes_and_keeps_the_family` / `test_remove_concepts_is_the_
  reserved_hard_delete`. **Do not** re-point the noise stages at `remove_concepts`.
  (Ready to commit — not yet committed.)
- **Symptom:** `concept_curation.remove_concepts` (`knowledge/concept_curation.py:400`) deletes
  `Concept` + `ConceptAlias` rows outright; stages 1–3 (artifact filter, `classify_noise` LLM,
  near-dup merge) route into it. `classify_noise` is precisely the path that mislabels real
  specialist vocabulary (`cre`/`dbs`/`ntsr1`/`pddl` — the trap hit twice, 2026-07-17/18).
  Deleting a Concept also deletes its keyword family (ADR-015 shared table) and cascades into
  presence/edges/gaps.
- **Cause:** the module predates ADR-018's demote verb; stage-0 ranking was correctly migrated to
  read-only but the destructive stages were not revisited.
- **Impact:** contained today — dry-run default, `--apply`-gated, and stages 1–3 have never been
  applied on the real corpus; the contract violation is the risk, not a live loss.
- **Workaround:** never run `scripts/curate_concepts.py --apply` stages 1–3 until fixed; curate
  with `set_graph_include(cid, False)`.
- **Fix:** route noise/artifact verdicts through `set_graph_include(id, False)` (keep row +
  family); reserve deletion for an explicit, separately-confirmed path. Guard test: a
  `classify_noise`-flagged concept keeps its family after `--apply`.
- **Pointer:** REVIEW finding CS-5 (verified); ADR-018; `docs/specs/feature-concept-graph.md`
  Traps; KW-9 is the same verb error at the tokenizer (`KEYWORD_MIN_CHARS` deletes unmined).

## KI-21 — in-app graph rebuild refreshes the skeleton but not the gaps the view serves — RESOLVED (2026-07-21, E0.3)
- **Resolved (2026-07-21, E0.3):** `_default_rebuild_graph` (`apps/api/main.py`) now chains
  `build_gaps(apply=True, min_degree=derive_min_degree(result.skeleton))` after the skeleton build.
  `min_degree` is the runtime **Q1 of the rebuilt skeleton's connected-node degrees**
  (`gaps.derive_min_degree`) — no hardcoded literal (measured **3** on the real 26-node graph,
  matching the CLI baseline). With the E0.2 reconcile also chained in, the served gap set equals a
  fresh recompute (live probe: 11 == 11, a just-inserted stale gap dropped). The plain skeleton
  build preserves Node-B stance (E0.5b), so the rebuild does not darken epistemics. Guard test
  `test_rebuild_refreshes_gaps_and_drops_stale_ones`. (Ready to commit — not yet committed.)
- **Symptom:** the ADR-017 B1 rebuild route (`apps/api/main.py:232` `_default_rebuild_graph`)
  calls `build_concept_skeleton(apply=True)` only — `build_gaps` has no API caller — and
  `load_graph_view` serves all `GapRow`s with no `graph_version` cross-check
  (`knowledge/concept_graph_view.py:96`, `knowledge/gaps.py:355`). After an in-app rebuild the
  UI shows gaps computed from the previous skeleton (including the gap the user just closed)
  until the CLI runs. Distinct from KI-17 (rows outliving `build_gaps` itself): here `build_gaps`
  never runs at all on the app's only rebuild affordance.
- **Cause:** B1 shipped the skeleton half of the acquire loop ("gap → ingest → rebuild → gap
  closes"); the gaps half was left to the CLI.
- **Impact:** the loop the button exists to close does not close in-app; stale-gap confusion
  compounds KI-17's orphans.
- **Workaround:** run `python -m scripts.build_gaps --apply` after any in-app rebuild.
- **Fix:** chain `build_gaps(apply=True, min_degree=<runtime-derived>)` after the route's
  skeleton build (needs KI-19/GP-1's runtime Q1 so the route needs no hardcoded default), or
  stamp `graph_version` onto gap rows and filter mismatches in the view; land together with the
  KI-17 reconcile (both concern what a rebuild must refresh).
- **Pointer:** REVIEW finding GP-4 (verified); ADR-017 B1.

## KI-25 — the concept graph emptied itself the moment KI-23 was fixed (`graph_include` landed NULL) — RESOLVED (2026-07-20)
- **Symptom (user-reported):** the Graph view showed **nothing**. `GET /api/concepts/graph` returned
  **0 nodes / 0 edges / 0 communities**, and its own staleness block said `n_concepts_in_db: 0`
  while the `concepts` table held **26** rows.
- **Cause — the fix for one issue triggered another.** ADR-018 made the graph vocabulary
  **opt-in** via `concepts.graph_include`, and `load_concepts()` documents that "NULL (every row
  predating the migration) reads as excluded". That column had never reached this box (**KI-23**).
  Running `python -m doc_assistant.db.migrations` by hand on 2026-07-20 — while diagnosing KI-23 —
  finally added it, **NULL on all 26 rows**, so every concept became excluded at once and the
  vocabulary the graph builds from went to zero. The migration was correct; what was missing is
  that an additive column with an opt-in default needs its **backfill** run in the same breath.
- **Not detected by anything.** The graph route degrades honestly to an empty graph (it is the
  documented "empty vocabulary → empty graph" path), the suite stayed green, and no gate compares
  "concepts in the DB" against "concepts the graph can see".
- **Fix (2026-07-20):** `python -m scripts.backfill_graph_include --apply` — the runner that exists
  for exactly this, applying ADR-018's rule retroactively (`source == "manual"` opts in). All 26 are
  `source="manual"` (they were hand-inserted during the 2026-07-01 baseline run, KI-13's
  workaround), so all 26 opted back in. Then a skeleton rebuild:
  `build_concept_skeleton(apply=True)` — **Node A only, deterministic, zero-LLM, $0**.
  Result: **26 nodes / 70 edges / 3 communities / 14 gaps**, `stale: false`; the app's concept index
  and the ego view both render again (verified live, 9 circles + 11 edges for `Connectome`).
- **What is NOT restored:** `concept_edges` was already **empty** before the fix, so no Node-B
  stance annotations were lost *by this* — but none exist now either. Re-running Node B
  (`build_concept_skeleton --apply --enrich`) is an **LLM pass** and was deliberately not run;
  KI-4's rule applies (force `--provider ollama`, which lives on the other box).
- **The general trap, worth more than this instance:** *an additive column whose NULL default
  changes behaviour is not a safe additive migration.* `_ADDITIVE_COLUMNS` already carries the note
  for `graph_include` ("Lands NULL on every existing row, which reads as excluded;
  `scripts/backfill_graph_include.py` sets the policy") — the note was right and simply nobody was
  in a position to act on it, because the column had never landed. Any future opt-in column should
  pair its `_ADDITIVE_COLUMNS` entry with its backfill runner in the same change.
- **Pointer:** `src/doc_assistant/knowledge/concept_skeleton.py` (`load_concepts` — the filter and
  its NULL semantics) · `scripts/backfill_graph_include.py` · ADR-018 · KI-23 (the migration that
  triggered it) · KI-21 (the in-app rebuild's own gap).

## KI-27 — unpaged whole-collection Chroma reads fail hard past ~32.7k chunks — RESOLVED (2026-07-25)

- **Symptom:** every whole-store read raised
  `chromadb.errors.InternalError: … (code: 1) too many SQL variables`. It took down
  `compute_epistemics`, `build_concept_skeleton`, **and `RAGPipeline.__init__`** — the last one
  means the answer path could not construct, i.e. **chat was down**, not degraded.
- **Trigger:** the 2026-07-25 corpus transfer took the parent-child store from ~16k to **33,163**
  chunks (47 → 97 documents). SQLite's parameter ceiling is **32766**, and Chroma's SQLite backend
  binds **one parameter per returned row** — so the failure is a step function at that row count,
  not a slowdown. The baseline store (12,800 rows) was still under it, which is why ingest and
  `compute_doc_vectors` kept working and the failure looked module-specific at first.
- **Scale lesson, and the reason this is filed rather than just fixed:** 97 documents is **1% of
  the way** to the 10k-document robustness contract, and the C4 scale review (2026-07-19) had
  flagged "unpaginated whole-corpus loads" as a *performance* risk. It is not — it is a hard
  correctness cliff, and the cliff sits inside the corpus range this project targets.
- **Fix:** new `src/doc_assistant/chroma_read.py` → `get_all(collection, where=, include=)`, which
  pages with `limit`/`offset` (`PAGE_SIZE=5000`, a structural bound, not a tuned threshold) and
  concatenates per key so callers see exactly what an unpaged read would have returned. Applied to
  every whole-store read: `pipeline.py` (BM25 build + `chunk_count`), `epistemics.py` (both),
  `concept_skeleton.load_presence_inputs` (which also batches its `$in` document filter),
  `doc_vectors.py`, `ingest/store.py`, `ingest/cleanup.py` (×2).
- **Guard:** `tests/unit/test_chroma_read.py` (8 tests) drives a fake collection that **counts
  pages**, so "no single call exceeds the page size" is asserted, plus order-preservation, the
  empty-collection contract, and page-size-independence. **Do not** reintroduce a bare
  `coll.get(include=[...])` over a whole collection — that is the bug.
- **Verified live (2026-07-25):** API boot rebuilt the BM25 index over all 33,163 chunks and a real
  `$0` Ollama turn returned a cited answer (10 sources, reranker 0.98→0.86, `is_local: true`).
