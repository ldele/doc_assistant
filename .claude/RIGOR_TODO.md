<!-- status: active · updated: 2026-07-20 · class: living -->

# RIGOR_TODO — deferred rigor tracker

Log every time experimental or performance discipline is skipped or postponed.
Close an item by doing the work (`Status: done`) or by a dated waiver — never by deleting it.
The gate (`rigor_gate.py`) fails while any `blocks-ship` item is `open`.

Severity: blocks-ship | degrades | nice-to-have
Status:   open | done | waived

> ## ⚠ THIS FILE IS NOW SHARED VIA GIT (ADR-020, 2026-07-18) — AND IT IS INCOMPLETE
>
> **It was per-machine until 2026-07-18** (gitignored like `.claude/SESSION.md`), which let the two boxes
> accumulate **disjoint item sets** for ~3 weeks. ADR-020 added it to the `.gitignore` allowlist so it
> syncs like `CONTEXT.md` / `KNOWN_ISSUES.md`: it tracks **project validation debt**, not per-machine
> state, so there was never a reason for it to be local.
>
> **The damage this caused, as a concrete example:** **RG-014 has no entry in this file** — yet it is cited
> as authority in **ADR-017, ADR-018, ADR-019, `docs/specs/feature-concept-graph.md` and
> `docs/ui-checklist.md`** ("`single_source` is the strong, low-volume signal"). A week of design decisions
> leaned on an item whose text exists only on the other box. **Do not cite an RG item you cannot read.**
>
> ### Present in this copy
> RG-001 / 008 / 009 · RG-010 · RG-011 · RG-012 · RG-013 · RG-015.
>
> ### Known missing — canonical text lives ONLY on the work box
> **RG-014** (the concept-graph gap-signal verdict — most urgent; actively cited) · **RG-007** (a real
> canonical item; its `feature-7-concept-graph.md` link was repointed to ADR-004 on 2026-06-26,
> `docs/DEVLOG.md:2681`) · **RG-002 / RG-004** (believed *moot* under the concept-graph redesign — nodes
> curated, co-occurrence now in the skeleton; they close as "redesign re-founds this",
> `concept-graph-redesign.md:285` — **do not re-run them against the old graph**) · RG-003 / RG-005 /
> RG-006 (existence unverified from this box).
>
> ### ⚠ FIRST SYNC ON THE WORK BOX MUST BE A MERGE, NOT AN OVERWRITE
> That box still has this file as an **untracked, ignored** file. On `git pull`, git will refuse to
> clobber it ("untracked working tree file would be overwritten") — that refusal is the safety net.
> **Do not force past it.** Procedure:
> ```sh
> git stash list                                  # ensure nothing pending
> mv .claude/RIGOR_TODO.md .claude/RIGOR_TODO.workbox.md   # preserve that box's items
> git pull                                        # now lands the tracked copy
> # merge by hand: copy RG-002/003/004/005/006/007/014 out of RIGOR_TODO.workbox.md into RIGOR_TODO.md,
> # keeping RG-001/008/009/010/011/012/013/015 that came from git
> rm .claude/RIGOR_TODO.workbox.md                # only after the merge is verified
> ```
> Then delete this "Known missing" section and this procedure block — the file is whole at that point.
>
> **Gate not wired:** there is no `scripts/rigor_gate.py` in this repo, and neither pre-commit nor CI
> reference it — the "gate fails" line above is **aspirational**; this is a manual discipline doc.

---

## RG-001 / 008 / 009 — concept-graph skeleton edge-precision + presence-recall validation run

**Logged:** 2026-06-18 (redesign); reconstructed here 2026-07-01
**Severity:** blocks-ship (RG-008 gates a *usable* graph + the gap layer) · RG-009 degrades
**Status:** **open — validation run DONE 2026-07-01, gate NOT passed.** Node A is BUILT
(`src/doc_assistant/knowledge/concept_skeleton.py`, 2026-06-30); the `--apply` run ran on the real corpus (baseline
`tests/eval/baselines/rg001_concept_skeleton_2026-07-01.md`) and found the graph **near-complete (46% density
at the provisional K=2) → NOT usable**. Thresholds **not locked**; the gap layer (ADR-004) stays blocked.
**Area:** ML / retrieval (deterministic, zero-LLM skeleton)

**What's missing**
One human-in-the-loop spot-check `--apply` run of the deterministic concept skeleton on the real ~61-doc
corpus, to set two thresholds **from the data, not guessed** (`concept-graph-redesign.md` §Gate/DoD:281–349):
- **RG-008 (blocks-ship) — edge precision per provenance tier.** Spot-check a sample of `concept_edges` for
  correctness, split by provenance (co-occurrence / citation / similarity); set
  `CONCEPT_SKELETON_MIN_COOCCURRENCE` (provisional **2**, `config.py:438`) + the trust threshold from the run.
  Density blow-up is the headline risk — the lever is `min_cooccurrence`↑, **not** a design change
  (`concept-graph-redesign.md:294`).
- **RG-009 (degrades) — presence recall.** Against a few hand-labelled docs; bounded by alias coverage (the
  curation burden). Word-boundary matching is the documented recall/precision upgrade lever
  (`concept_skeleton.py:168–170`).

**Why it blocks**
The gap-detection layer (`docs/decisions/ADR-004-gap-detection-layer.md` / `docs/specs/feature-gap-detection.md`)
defines every Tier-1 signal (isolated / thin-bridge / under-connected) *relative to the edge set* — an over-
or under-connected skeleton makes every gap count meaningless. So this is a **correctness gate on
gap-detection, not optional rigor** (`docs/DEVLOG.md:2655–2658`). It also gates PR-B (Node B stance) and
marking the graph *usable*.

**Prerequisites (both bit Feature 6 — check first):**
1. Curated `Concept`/`ConceptAlias` rows seeded (`scripts/seed_concepts.py` promote flow) — **no vocabulary →
   empty graph** (a real curation step, not a formality).
2. `DocSimilarity` populated (`scripts/compute_doc_vectors.py --apply`) or similarity-provenance is silently absent.

**How it gets closed**
Free on local Ollama / host (KI-5 — enrichment runs on the host; this box has the corpus, `data/library.db` +
`data/chroma`). Seed vocab → `scripts/build_concept_skeleton.py --apply` → spot-check edges + presence → set
the two thresholds → record a baseline under `tests/eval/baselines/` via the rigor-gate discipline. Moot
siblings RG-002 / RG-004 close as "redesign re-founds this" (do not re-run against the old graph).

**Run 2026-07-01 — results (baseline `tests/eval/baselines/rg001_concept_skeleton_2026-07-01.md`):**
Built the missing prerequisites first (all free/regex): metadata backfill + citation extraction (→ 7 internal
citation links) and a **provisional** 30-concept vocabulary seeded *directly* (the `Keyword`→`--promote` seam
is dead — nothing writes `Keyword` rows; see KI-13). Findings:
- **RG-008 (still open):** graph is near-complete — 201 edges / **46% density** at K=2 (max C(30,2)=435).
  `min_cooccurrence`↑ alone is weak (still 27% at K=5); the real lever is **vocabulary breadth** — dropping
  the 9 broad hubs (≥7/10 docs: BERT, language model, transformer, …) cut edges **201→57** at K=2 (46%→27%).
  similarity-provenance annotates **100%** of edges, citation ~88% → non-discriminating (KI-7 same-domain
  doc saturation, re-confirmed; a corpus property, not a threshold/vocab lever).
- **RG-009 (still open, degrades):** 29/30 concepts surfaced; 1 miss (query expansion, alias gap); "BERT"
  presence is substring-inflated (matches SBERT/ColBERT/RoBERTa) → the word-boundary-matching upgrade lever
  is needed before precision is trustworthy.
- **Re-run (b) — keyword-grounded vocabulary (per-doc TF-IDF, KI-13 extractor):** 139 concepts, 146/148 df=1
  → **1250 edges / 13% density @K=2, 17 communities that map to PAPERS** (federation of per-paper cliques —
  clusters by document, not cross-cutting concept). Fragments where the manual-hub regime saturated.
- **Re-run (c) — corpus mid-DF band mode (built + run to test (b)'s hypothesis):** added a general `corpus_band`
  extractor mode (`min_df`/`max_df_frac`/`top_k`), ran once with defaults (2 / 0.7 / 60). **Hypothesis FAILED:**
  the df 6–7 band is dominated by generic academic vocabulary (`consider`, `introduce`, `benchmark`, `document`,
  `sebastian`…), not domain concepts → the **most saturated graph of all: 60 concepts / 1466 edges / 83% density
  @K=2**. On a same-domain corpus, concepts and boilerplate share a DF range — no band separates them.
- **Gate NOT passed — DEFINITIVE: the blocker is the CORPUS, not the vocabulary method or the code.** Four
  vocabulary regimes span 13%→83% density; every one fails because 10 same-domain papers can't support a
  cross-document concept graph (saturated doc-similarity → provenance non-discriminating; unstable N=10
  co-occurrence; statistical keyword selection can't separate concept from academic boilerplate without a
  larger multi-domain corpus). To close: (1) **re-run on the larger multi-domain corpus** (`cases.yaml`) — the
  gate; (2) a semantic/LLM or curated-domain-stopword vocabulary gate (deferred, NOT tuned to this corpus per
  the run brief); (3) word-boundary presence matching (RG-009). Only then lock
  `CONCEPT_SKELETON_MIN_COOCCURRENCE` + mark the graph usable + unblock the gap layer. On-disk (public): 60
  corpus-band concepts + skeleton @K=2 (`data/`, gitignored; reset: `DELETE FROM concept_aliases; DELETE FROM concepts;`).
- **Multi-domain re-check (2026-07-01, SEPARATE corpus — `tests/eval/baselines/rg001_concept_skeleton_multidomain_2026-07-01.md`):**
  24 open-access arXiv papers across 6 domains in a separate data home (`DOC_DATA_DIR=…/data_multidomain`,
  gitignored; public corpus untouched), current params. **Result: doc-similarity DID de-saturate (43% of pairs
  vs 100% public) — but the graph is still unusable (60% density, generic-vocab blobs).** Two corpus-INDEPENDENT
  blockers now proven: (a) **DF statistics can't identify domain concepts** — same-domain they share a DF band
  with boilerplate, multi-domain they're low-DF; a *semantic* gate is required, not a threshold; (b) **PDF
  extraction placeholder noise** (KI-14). So the corpus is a *necessary but not sufficient* fix. No tuning done
  (per the run brief).

---

## RG-010 — frozen-sidecar cold-start (launch → first `/api/health 200`)

**Logged:** 2026-06-22
**Severity:** degrades
**Status:** done (measured both boxes; degrades-only, no hard threshold)
**Area:** perf

**What's missing**
The frozen `dist/doc-assistant-api.exe` launch-to-warm time. No hard threshold — record + watch (if
>~30 s, switch the PyInstaller spec onefile → onedir).

**Measured (work box, 2026-06-23, warm HF cache):** ~35–40 s (39.7 s, 34.9 s) with `HF_HUB_OFFLINE=1`.
**Measured (RTX box, 2026-06-24, warm HF cache):** **46.2 s** (process spawn → `/api/health 200`), onefile
unpack of the 385 MB bundle + bge/reranker/Chroma load. Above the ~30 s soft guideline → **onefile→onedir
is the lever** if cold-start becomes a UX issue (skips the per-launch temp unpack).
**Measured (RTX box, 2026-06-24, KI-9 weights-bundled 1.6 GB build):** **30.9 s** — bundling the ~1.5 GB of
weights did **not** regress cold-start (within run-to-run/OS-cache variance; unpack dwarfed by model load on
NVMe). So KI-9's offline capability comes free here, and the first-run HF download (≈218 s, KI-9) is
**eliminated** — onedir optimization is no longer needed.
Baseline: `tests/eval/baselines/rg011_first_token_ollama_2026-06-24.md` (frozen follow-up).

**How it gets closed**
Closed as a recorded measurement (degrades, not blocks-ship). Revisit (onedir / KI-9 bundle-weights) only
if the cold-start UX matters at ship; owned by the M4 ship checklist (`docs/desktop-packaging.md` §4).

---

## RG-011 — SSE first-token latency (FastAPI/SSE boundary vs in-process)

**Logged:** 2026-06-22
**Severity:** blocks-ship
**Status:** **done** (2026-06-24) — freeze + SSE boundary add **no** first-token penalty, measured on the frozen artifact against a same-session control
**Area:** perf

**What's missing**
Does the Tauri/FastAPI desktop boundary (`apps/api`, HTTP/SSE over 127.0.0.1) + the PyInstaller freeze add
meaningful first-token latency vs the in-process `ChatController` the Chainlit/CLI renderers call? Originally
**blocked** on the work box: the corporate TLS-MITM proxy SSL-failed the Anthropic first-token call (KI-10),
so it could not be timed there.

**Measured (RTX box, 2026-06-24 — local Ollama, no external TLS):**
- *Source-server boundary (n=5):* in-process median 4.563 s vs HTTP/SSE 4.140 s → SSE hop adds no latency.
- *Frozen artifact `dist/doc-assistant-api.exe` (n=5, same-session control):* in-process median **5.859 s**
  (sd 0.035) vs frozen HTTP/SSE **5.312 s** (sd 0.520), Δ **−0.55 s** → **the freeze adds no first-token
  penalty. PASS.** (Absolute numbers differ from the source run by Ollama GPU warm-state, not the build —
  the valid comparison is same-session frozen-vs-control, done here.)
- Config: `ollama/llama3.1:8b` (GPU) + bge-base/reranker on CPU torch, public corpus 2455 chunks, fresh
  session/sample. Tool: `scripts/measure_latency.py --launch … --repeat 5` (+ `--in-process`).
- Baseline: `tests/eval/baselines/rg011_first_token_ollama_2026-06-24.md` (source + frozen follow-up).

**Closed because** the latency question is answered on the actual frozen artifact with a same-session
control: no penalty. Remaining non-latency caveat (not part of this gate): a **paid-provider** first-token
on the frozen build is blocked on the work box by KI-10 (cert trust) — but first-token *latency* is
provider-independent (the SSE/freeze overhead is plumbing), so the verdict stands. If a paid-path number is
ever wanted, fix KI-10 first.

---

## RG-012 — clean-machine smoke (Python-free box)

**Logged:** 2026-06-22
**Severity:** blocks-ship
**Status:** open
**Area:** repro

**What's missing**
Install the bundled installer on a box with **no Python / no toolchain** and drive one real turn.
- Tier 1 (proves the freeze): app + sidecar come up, `/api/health` green, no missing-module / DLL error.
  Work box (2026-06-22, Windows Sandbox): hit the KI-9 first-run HF download, then KI-10 cert failure —
  **both now fixed (2026-06-24): KI-9 weights bundled + verified offline; KI-10 truststore shipped.** So a
  fresh-box Tier-1 should now come up offline, no download, no cert error. Just needs a clean box to run on.
- Tier 2 (full smoke — a real cited turn): needs the **data-home decision built** (seeded corpus or a
  first-run ingest flow — the unbuilt product gap; the sandbox has `chunk_count: 0` otherwise). **Also
  required (found + fixed 2026-06-24, KI-11):** a fresh ingest under the per-user home `C:\Users\<user>\…`
  silently produced an unloadable Chroma index when the username is non-ASCII; fixed by relocating the
  Chroma store to an ASCII `%PROGRAMDATA%` path (`config._chroma_base`). Verified via the venv; **needs a
  re-freeze** to reach the shipped artifact.

**Status note (2026-06-24):** installer rebuilt this session carrying the KI-9/KI-10 sidecar
(`doc_assistant_0.1.0_x64_{en-US.msi, -setup.exe}`, ~1.9 GB, gitignored). The only remaining blockers are
**a clean box** (Windows Sandbox not enabled on the RTX box — `WindowsSandbox.exe` absent → admin + restart,
or a second Python-free box) and, for Tier-2, the **data-home flow**.

**Update (2026-07-01, from `.claude/CONTEXT.md` 2026-06-30):** the **data-home / first-run-ingest flow is now
built** (backend `77eb5f9`: `/api/settings` + `/api/ingest`; `apps/desktop` settings panel + empty-corpus
banner) — the Tier-2 product gap is closed **in source**. Tier-2 now pends only **(a) a re-freeze** bundling
the data-home flow + the KI-11 Chroma-ASCII fix into the shipped sidecar, and **(b) the clean-box run**.
Tier-1 (offline / cert-trust) is already unblocked (KI-9 + KI-10 shipped). RG-012 stays **open** (blocks-ship).

**How it gets closed**
Windows Sandbox (or a second clean box) + a re-freeze carrying the built data-home flow + KI-11 fix
(`docs/desktop-packaging.md` §5).

---

## RG-013 — M4 freeze must re-verify `structlog` is bundled

**Logged:** 2026-06-23
**Severity:** degrades
**Status:** **done** (2026-06-24)
**Area:** repro

**What's missing**
`structlog` became a base dep (ADR-003, KI-1). The PyInstaller freeze must re-verify it is bundled and the
frozen build still emits structured logs (no missing-import / console-silencing regression).

**Verified (RTX box, 2026-06-24):** the frozen `dist/doc-assistant-api.exe` startup console emits
structlog-rendered events (`…Z [info ] loading_embeddings [doc_assistant.pipeline] model=bge-base`); a scan
of the full startup log for `structlog|ModuleNotFound|ImportError|Traceback` returns **0**. structlog is
import-followed into the freeze (no explicit hiddenimport needed) and structured logging survives with no
regression. The spec (`scripts/doc_assistant_api.spec`) pulls it via `collect_submodules("doc_assistant")`.

**Closed:** confirmed on the frozen build. No spec change required.

---
## RG-015 — domain-coverage gap detectors: precision vs the five existing kinds (ADR-019)

**Logged:** 2026-07-18
**Severity:** degrades (a low-precision detector makes the gaps destination noisier, not wrong)
**Status:** open — the detectors are not built yet; this is the gate on *retiring* anything
**Area:** gap layer / concept taxonomy

**What's missing**
ADR-019's grill resolved branch 6 as **additive**: domain-coverage gap kinds land *alongside* the five
existing ones (`isolated`, `single_source`, `thin_bridge`, `under_connected`, `unsourced_claim`), and
**nothing is retired until measured.** The measurement itself is deferred — this item is it.

**Why it is deferred rather than skipped.** The obvious move is to retire the three degree-based kinds
(RG-014 called `under_connected` graph-degree noise). But RG-014 was measured on the **RTX box's 76-doc /
26-concept corpus**, and its headline verdict — `single_source` is the strong, low-volume signal — **did
not survive** the 2026-07-18 ADR-018 rescope on this box: at 357 concepts `single_source` was 224/302
gaps, and at 13 concepts the kinds are near-flat (`thin_bridge` 4 · `isolated` 3 · `single_source` 3 ·
`under_connected` 3 · `unsourced_claim` 2). Retiring three signals on a verdict that already failed to
transfer once is exactly the error this tracker exists to prevent.

**The work**
Once domain-coverage detection ships: hand-audit every gap of every kind on the real corpus and record
precision per kind (is this a finding a researcher would act on?). Then decide retirement **per kind**,
with the corpus size and vocabulary size stated as bounds on the verdict. Record in
`tests/eval/baselines/`, alongside the RG-014 baseline it supersedes.

**Reopens if:** the vocabulary or corpus changes size materially (this is now twice-demonstrated —
a verdict measured at one vocabulary size does not transfer to another).

## RG-016 — graph floors + gap-kind ranking must be re-derived at scale (they encode n≈26–76)

- **Severity:** degrades · **Status:** open · **Added:** 2026-07-19 (scale review, KI-19)
- **What was skipped:** three graph thresholds ship as absolutes measured once on tiny inputs:
  `CONCEPT_SKELETON_MIN_COOCCURRENCE=2` (validated at 76 docs only — ADR-008 is explicitly
  scope-bound), `_DEFAULT_MIN_DEGREE=3` (a frozen Q1 snapshot from a 26-concept graph whose
  docstring claims "corpus-derived"; already failed to transfer at 357 and at 13 concepts), and
  the UI gap-kind rank table (single_source first, on the authority of RG-014 — unreadable from
  this box, non-transferring per RG-015).
- **The experiment owed:** synthetic (or real) corpora at ~1k and ~10k docs; sweep the
  co-occurrence floor as a function of chunk volume (density curve vs the ADR-008 21.5%
  reference); implement min_degree as runtime Q1 and verify it reproduces the 26-concept
  behavior; re-derive kind ranking from per-kind precision at ≥3 vocabulary sizes.
- **Until then:** treat all three as display heuristics, not signals; extends RG-015.

## RG-017 — family Tier-2 embedding threshold (0.86) was never derived on the embedder it runs on

- **Severity:** degrades · **Status:** open · **Added:** 2026-07-19 (scale review, KI-19)
- **What was skipped:** `DEFAULT_EMBEDDING_THRESHOLD=0.86` gates Tier-2 family detection, but
  both call paths embed with **bge**, whose same-domain cosine ceiling this project already
  measured at ~0.77–0.82 (the concept-merge feature hit the identical wall and switched to
  specter2). The tier therefore under-fires structurally — `connectome`≈`connectomics` (its
  design case) cannot pass; the only observed proposal was a substring pair.
- **The experiment owed:** a labelled pair set (true families vs near-misses from the real
  vocabulary); ROC on bge vs specter2; pick embedder + threshold from the curve, record the
  baseline, then re-run `detect_family_candidates` on the real corpus and hand-check.
- **Until then:** Tier-2 silence is not evidence of no families.

## RG-018 — wiki clustering: the recorded "wrong primitive" is still the shipped default

- **Severity:** degrades · **Status:** open · **Added:** 2026-07-19 (scale review, KI-19; the
  deferred item itself dates to the monolith)
- **What was skipped:** the monolith records "absolute-cosine threshold is the wrong primitive
  (use relative / community clustering)"; the community path is BUILT
  (`WIKI_USE_CONCEPT_COMMUNITIES`) but default-false, validated only on the 10-paper public
  corpus, while the live default remains union-find at `WIKI_MIN_SIMILARITY=0.90` — tuned on
  that same corpus. `[[links]]` derive from cosine edges on both paths.
- **The experiment owed:** the flip validation on the current corpora (47 + 76 docs): communities
  vs cosine on cluster count/size distribution + a hand-read of note coherence; then flip the
  default, and derive links from inter-community skeleton edges (the recorded refinement).
- **Until then:** wiki topics on a same-domain corpus are one-blob-plus-outliers by construction.

## RG-019 — `contested` fires on one disputing doc; marker density is saturating unvalidated

- **Severity:** degrades · **Status:** open · **Added:** 2026-07-19 (scale review, KI-19)
- **What was skipped:** `coverage="contested"` triggers on `nc >= 1` (an implicit, unnamed
  threshold); 53.6% of real chunks carried a marker at 47 docs (recorded 2026-07-08, flagged
  "worth a wider spot-check", never done) and the trigger is monotone in corpus size.
  `agreement_ratio` is computed into the artifact and consulted by nothing.
- **The experiment owed:** the deferred density spot-check (sample marked chunks, measure
  precision of "contested" against a hand judgment), then derive a named floor
  (min disputing docs and/or an agreement-ratio band — the `MIN_DATED_DOCS_PER_SIDE` pattern)
  and re-measure density at both corpora sizes.
- **Until then:** do not lean on marker density as a UI signal (the 2026-07-08 caveat stands).

## RG-020 — folder-scoped retrieval (ADR-025 F2): filter latency at 10k docs + scoped-BM25 statistics

- **Severity:** degrades · **Status:** **partially discharged 2026-07-20** (real-corpus half
  measured at F2 build time — `tests/eval/baselines/rg020_scoped_retrieval_cost_2026-07-20.md`;
  the synthetic 10k half is still open)
- **What was deferred:** the query-time doc-hash filter ships on design reasoning, not measurement.
  Two bounds owed when F2 builds: (a) Chroma `where doc_hash $in [...]` latency with a
  thousands-of-hashes member set at the 10k-doc robustness contract (the "reverses if" trigger —
  per-folder precomputed indexes are the fallback); (b) the scoped BM25 arm scores against
  subset statistics (avgdl/IDF differ from global) — correct behavior, but record a before/after
  retrieval sanity check so a scoped-recall surprise isn't misread as a bug.
- **The experiment owed:** retrieve+rerank latency, unscoped vs scoped at small/medium/large
  folder sizes on a synthetic 10k corpus (the `measure_latency.py` harness pattern); record in
  `tests/eval/baselines/`. Assert the unscoped path stays byte-identical (guard test in F2).
- **Until then:** F2 may ship scoped retrieval for real corpora sizes (~10²); do not claim the
  10k contract holds for scoped turns.
- **Measured 2026-07-20 (76 docs / 30,882 chunks), part (b) + the guard done:** BM25 subset
  rebuild ≈20 µs/chunk (622 ms whole corpus · 248 ms for a 30-doc scope · 27 ms for 3 docs);
  Chroma `$in` 136 ms unscoped → 193/232/408 ms for 3/30/76 hashes, i.e. the cost tracks the
  **`$in` list length**, not the corpus share. Shipped with a single-slot scoped-ensemble memo
  keyed on the hash set (F2 spec S5) so a sticky scope pays the rebuild once. The
  **unscoped-byte-identical guard test exists** (`tests/unit/test_pipeline_scope.py::
  test_scope_none_uses_the_prebuilt_ensemble_and_builds_no_filter`). Scoped-BM25 subset
  statistics are recorded as expected-and-correct in the baseline.
- **Still owed (why this stays open):** part (a) at the **10k contract** — extrapolation puts a
  full-scope rebuild near ~80 s and a thousands-long `$in` list into Chroma, so the ADR-025
  fallback (per-folder precomputed indexes) is still live. Do not claim 10k for scoped turns.

## RG-021 — eval-harness index-composition fingerprint (benchmark runs on a polluted index are silent)

- **Severity:** degrades · **Status:** open · **Added:** 2026-07-20 (corpus-groups grill, ADR-025 fork 6; first flagged in the 2026-07-20 demo-collection DEVLOG "Opens")
- **What was deferred:** nothing mechanically detects an eval run over an index containing
  non-eval documents (e.g. the demo collection). BM25/IDF statistics and the vector neighborhood
  are corpus-global, so such numbers are not comparable to committed baselines even with perfect
  retrieval-side folder filtering — the guard must live in the harness, not the folders feature.
- **The experiment owed (small build, not a run):** `run_eval` records index composition (doc
  count + a digest of sorted doc_hashes) in every run record; warn on mismatch vs the baseline's
  recorded composition. Becomes urgent the first time a baseline is re-recorded after `--demo`
  has ever been ingested on the box.
- **Until then:** the manual rule stands — benchmark on the clean eval-10 index only
  (`evals/README.md` states it; the download-default guard test pins the fetch side).

## RG-022 — `RERANK_CANDIDATE_CAP` multiplier is a cost/recall tradeoff on the multi-query path, unvalidated

- **Severity:** degrades (multi-query path only) · **Status:** open · **Added:** 2026-07-23
  (retrieval-hygiene sprint)
- **What was deferred:** `RERANK_CANDIDATE_CAP` (default `CANDIDATE_K * 3` = 60) bounds the number of
  candidates fed to the cross-encoder in one turn. It **provably never bites the single-query default
  path** (single-query unions ≤ `2*CANDIDATE_K` and the cap is validated `>= 2*CANDIDATE_K` at import),
  so the shipped default is byte-identical. But when multi-query is on (opt-in via `USE_MULTI_QUERY`
  or the U1 per-turn override), the cap truncates the cross-phrasing candidate union — a real
  **cost vs recall** tradeoff whose default multiplier (`3`) is chosen for a bounded worst case
  (~4× → ~3×), **not measured**.
- **The experiment owed:** on the eval-10 index with multi-query forced on, sweep the cap
  (`2*CANDIDATE_K` … unbounded) and record answer-quality (the eval scorers) vs rerank latency;
  confirm the recall knee sits at or below the shipped default. Record in `tests/eval/baselines/`.
- **Why deferred, not assumed:** multi-query is **off by default**, so this changes nothing in the
  shipped path today; it becomes blocking only if MQ is ever promoted toward a default (which is its
  own eval-gated decision — the ADR-010 sandbox exposes MQ precisely to measure it first).
- **Until then:** the cap is a cost ceiling, not a tuned optimum — do not cite MQ retrieval quality
  under the cap as validated. Override `RERANK_CANDIDATE_CAP` (env) to reproduce the pre-cap
  unbounded behaviour for a sweep.

## RG-023 — epistemic-health detectors parked pending cost measurement + a heterogeneous corpus

- **Severity:** blocks a feature (not a shipped path) · **Status:** open · **Added:** 2026-07-23
  (ADR-019 taxonomy grill — the parked epistemic-health cluster)
- **What was deferred:** the epistemic-health detector layer (its own future **ADR-EH**), sequenced
  **after the concept graph / taxonomy is validated** (user decision, 2026-07-23). Design captured in
  `docs/PLAN_2026-07-23_concept-graph-taxonomy-epistemics.md` §4–6. The grill **could not lock** these
  because their deciding reasons are unmeasured numbers, not judgment:
  - **Per-concept contradiction / hedging / citation-density** — the answer-level versions exist
    (reviewer `citation_density`/`hedging_adequacy`, per-answer). The per-concept versions imply an
    LLM pass over each concept's evidence chunks; cost scales with concept × chunks and is **unmeasured**.
  - **Source-trust scoring** — needs the **heterogeneous corpus** it exists to protect (Wikipedia /
    web / personal notes); on an all-papers box there's nothing to differentiate, so the tiering can't
    be validated. Precondition for widening ingestion beyond peer-reviewed papers (the "quality list", B13).
  - **Dual + non-paper staleness** — the semantic `superseded_trend` is built; document-age staleness
    and its meaning for undated/non-paper content (SOPs, notes) is a new detector.
  - **Content-type degradation** — citation-dependent mechanics must degrade honestly on books/SOPs/notes
    (no abstract/references section); the 0-doc robustness contract generalized to content type.
  - **Degree-based-detector retirement** — gated by **RG-015** (coverage-detector precision).
- **The experiment owed (before ADR-EH can lock):** measure the per-concept LLM pass cost offline on
  messy real output + a local model (cost-discipline); build a small heterogeneous test corpus to
  validate source-tiering; then grill ADR-EH.
- **Until then:** the taxonomy (ADR-019 amendment) ships and is *used* (navigation, coverage math, gap
  surfacing) without the epistemic-health synthesis. Do not build per-concept detectors on the
  association-only graph (Node B stance NULL, KI-4) — measure first.
