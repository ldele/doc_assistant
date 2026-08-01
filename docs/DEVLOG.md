<!-- status: active · updated: 2026-08-01 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> Entries **2026-07-14 and earlier** live in [`docs/archive/DEVLOG-archive-001.md`](archive/DEVLOG-archive-001.md)
> (moved verbatim 2026-07-21). This file keeps 2026-07-15 onward.

---
## 2026-08-01 (6) — v0.4.0 verified from a clean clone on Linux; the Dockerfile's CPU-torch trick was silently defeated by `pip`

**What changed.** No source code. `.dockerignore.md` → **`.dockerignore`**, the Dockerfile switched
from `pip` to `uv`, and a header on `docker-compose.yml`.

**The clean-room run (WSL2, Ubuntu 26.04, system Python 3.14).** A fresh `git clone` of the v0.4.0
tag carries tracked files only — no `.env`, no `.venv` — which is how this box finally produced the
**keyless first-run state the 2026-07-28 session recorded as impossible here** (`load_dotenv()` walks
up from the module file, so the repo `.env` always loads on Windows no matter where the process
starts).

| Step | Result |
|---|---|
| `uv sync --extra cpu --extra dev` (verbatim from QUICKSTART) | exit 0, **113 s** |
| Interpreter chosen | **3.12.13**, fetched automatically on a 3.14 host |
| Cold start → `/api/health` 200 | **~30 s**, downloading **419 MB** to an empty HF cache |
| `/api/setup`, no key + no documents | `active_ready=False chunk_count=0 ready=False` — honest |
| Ollama probe | `ollama_probe_failed … Connection refused` — reported, not assumed |
| `/api/compare` at 0 documents | empty lists, no crash (the robustness contract) |
| `download_corpus` → `ingest` | 10 papers, **10 added / 0 errors**, 524 s (51.7 s/paper, CPU) |
| Warm start | **10 s** |
| Retrieval | **2,301 chunks**, `arm=sparse_index`, `keyword index: on_disk`, 10 sources, top 0.9967 |

The top documents match the Windows run for the same query (`hyde_gao_2022` · `bge_cpack_xiao_2023` ·
`rag_lewis_2020` · `dpr_karpukhin_2020`), and 2,301 chunks is the same count — so ingest and
retrieval are reproducible across platform, which nothing had checked before.

**Two things I asserted before running it, both wrong, both corrected by the run.**
1. *"`requires-python = ">=3.10"` means nothing enforces the 3.12 pin."* A **tracked
   `.python-version` (3.12)** enforces it, and uv honoured it on a 3.14 host. The residual gap is
   narrow: `.python-version` binds uv, not pip, so `pip install -e .` would still accept 3.10+.
2. *"`data/` in a fresh clone is unexpected."* It is correct — two tracked files, the ANZSRC CC-BY
   taxonomy that TX1 deliberately commits.

**A measurement that weakens a recorded argument.** KI-9 justified bundling ~1.5 GB of weights into
the installer with a **≈218 s** first-run HuggingFace download. Measured here: **~30 s / 419 MB**.
Not a contradiction — that figure was June, on the *frozen* build, in Windows Sandbox, and it
predates the **lazy reranker** (2026-07-28), which moved the reranker's weights off the launch path
entirely. So the download is now smaller and later than the number the bundling decision rests on.
The decision stands (the user parked the installer question); the *evidence* for it is weaker than
the record says, and that is worth knowing before anyone cites the 218 s again.

**What this is NOT.** Generation was never exercised there — the clean box has no key and no Ollama,
which is precisely why it could test the keyless state. So this closes the **source-install** gate,
not **RG-012 Tier-2**, which is about a frozen Windows binary on a Python-free box and remains open.

**The Docker bug, and it was load-bearing.** `pip install -e ".[cpu]"` cannot honour the CPU-torch
routing: `[tool.uv.sources]` maps torch to the `pytorch-cpu` index **only under uv**, so pip saw
nothing but `torch>=2.12` and resolved it from PyPI — whose Linux wheel bundles CUDA, pulling
several GB of `nvidia-*` into an image with no GPU. This is the exact trap `ci.yml` documents. Now
`uv sync --locked --extra cpu`, which also honours `uv.lock` (pip ignored it entirely). The base is
pinned to `python:3.12-slim` with the KI-2 reason attached, and the healthcheck `start-period` rose
to 300 s to cover the model download measured above.

**`.dockerignore.md` was inert for 173 commits.** Docker reads `.dockerignore` exactly, so every
build shipped `data/` (590 MB here), `.venv/` and `node_modules/` into the context.

**Untested, and stated as such: Docker is not installed on this box**, so none of the Dockerfile
changes have been built. The `ghcr.io/astral-sh/uv:0.12.1` pin was at least verified to resolve
(registry manifest, HTTP 200) rather than guessed. `docker compose build` is owed before trusting it.

---
## 2026-08-01 (5) — v0.4.0: walkthrough, version bump, release notes. **Source release; the installer stays stale on purpose**

**What changed.** Version 0.3.0 → **0.4.0** across all five strings (`pyproject.toml` ·
`package.json` · `tauri.conf.json` · `Cargo.toml` · `Cargo.lock`, the last matched on its own package
entry so the `dtor`/`urlpattern` crates at 0.3.0 are untouched), a CHANGELOG entry, and the README
status block. No source code.

**Why 0.4.0 and not 0.3.1.** ADR-036 changed retrieval ranking and ADR-038 **removed** two documented
environment switches (`DOC_SPARSE_INDEX`, `DOC_BM25_CACHE`). Removing an interface is not a patch.

**Walkthrough first, $0 on `ollama/llama3.1:8b`** — the release claim needed evidence, not the test
suite's word. One real turn returned a grounded answer with **10 inline citations**, 10 sources, the
source-evaluation strip (scores 0.98 → 0.47, sigmoid-bounded), the provenance card and `0 tokens ·
local`. Conversation history put the new turn on top; Library served 97 documents with titles and
years; the graph served 13 nodes / 19 edges / 6 communities / 18 gaps; Settings → Corpus read
`on_disk, 40.7 MB`. **0 console errors.**

**Two findings the walkthrough produced, both now in the CHANGELOG's Known limits.**
1. **`contested` has saturated, and it is measured rather than suspected: 53.3% of assessed chunks**
   (396 of 743) versus **3.9%** carrying `corroborated`. In the live turn, 8 of 10 sources read
   *contested* — on "what is dense passage retrieval vs BM25", which is not a controversy. This is
   exactly RG-019's prediction (53.6% at 47 documents, monotone in corpus size) holding at 97. The
   integrity strip is the product thesis and it is currently closer to noise than signal.
2. **The answer copies the source's voice** — "Our dense retriever outperforms…", first person, as
   if the app wrote the paper. An 8B artifact, but the first thing a tester will notice.

**Release shape: source only, deliberately.** The bundled installer is `doc_assistant_0.1.0` from
2026-06-24 — pre-rename, pre-icon, pre-ADR-034, and pre-everything in this release. Shipping it would
ship a different app, so the CHANGELOG says plainly that this is a source release. **RG-012 Tier-2
(blocks-ship) stays open**: no cited turn has ever run on a clean, Python-free box.

**The installer question, closed for now by the user** ("we don't need it for now"). It was
investigated rather than deferred blind: a *smaller* installer means downloading model weights from
HuggingFace at first run, which is the design KI-9 measured and rejected — **≈218 s** of first-run
download and an offline launch that never goes green, against a bundled build whose cold start
**did not regress** (30.9 s) for 385 MB → 1.6 GB. One thing has changed since: ADR-034 built a
first-run readiness surface, so weights could become a visible, resumable step rather than a silent
hang. That is a different proposal from the rejected one and would need its own ADR; not written,
because it is not wanted yet.

**Gates:** ruff + format clean · `mypy src` 86 files · **pytest 1446** · `svelte-check` 187 0/0 ·
**npm 68/68** · `docs_check --strict` 0/0 · all five version strings verified aligned at 0.4.0.

---
## 2026-08-01 (4) — Corrected the "retrieval is deterministic" claim; ADR-039 proposes an OCR sidecar (docs only)

**What changed.** No source behaviour. (a) The determinism claim is corrected where it was asserted:
`scripts/sweep_bm25_weight.py` (docstring, `--repeat` help, two console lines) and the ADR-036
baseline, which gains a dated correction rather than being re-recorded — its *aggregate* numbers were
unaffected, only the claim that a per-case list is reproducible. (b) New **ADR-039** (proposed) +
roadmap row **EX1** + **RG-024/RG-025**.

**The correction is specific, not a hedge.** The divergence is not in either retrieval arm: the
pre-rerank candidate list was byte-identical across runs and only post-rerank parents moved, so it is
the **cross-encoder breaking ties** — and it showed up on a case whose target document has 0 chunks,
where nothing is a real match and scores cluster. Aggregate recall was identical across runs, so a
single pass still represents the aggregate; it does not represent any individual case's document
list.

**ADR-039, and the reason it is not an LLM.** A PDF with no text layer extracts to nothing and the
document is silently unreachable — measured today: **93 healthy / 2 marginal / 2 broken** of 97, with
`middleton-2001.pdf` at `chunk_count=0`. The tempting move is to have a model investigate. **The
diagnosis is already deterministic and strictly better:** `health.py` classifies with reasons, and
KI-26 had already characterised this file (290 extracted characters, 15 of them page markers). What
is missing is *remediation*, and for pixels-that-should-be-text that is **OCR** — deterministic — not
inference.

**The load-bearing decision inside it: OCR emits a text layer, not markdown.** The artifact is a
searchable PDF at `<data>/ocr/<doc_hash>.pdf` which the extractor reads instead of the original, so
`extract_pdf_pymupdf` stays the **only** extraction path and a recovered document becomes an ordinary
one. Marker OCR (option 2) would have been cheaper to wire — it needs no system binary and already
runs out-of-process for tables — but it emits markdown, forking extraction into two shapes that must
stay compatible forever. This session deleted exactly that kind of second path (ADR-038), and KI-29
is the record of what a path nobody exercises does to an answer.

**A correction to something I said in this session:** Marker is **not** a project dependency. It is an
external tool `scripts/extract_tables_marker.py` shells out to (`uvx --from marker-pdf
marker_single`), and `extractors.py` raises for any `PDF_EXTRACTOR` other than `pymupdf`. That makes
option 2 more expensive than it first looked, and it is the precedent for ADR-039's absent-tolerance
rather than a free ride.

**Gated, not assumed.** RG-025 blocks enabling recovery by default, on an asymmetry worth keeping:
a document with no text layer is *honestly* absent, while one full of OCR garbage is retrievable,
rerankable and **citable**. RG-024 records that the ~2% broken rate is one corpus at n=97 — two
documents — and bounds nothing about a scan-heavy library.

**Gates:** ruff + format clean · `docs_check --strict` 0/0 · `sweep_bm25_weight --dry-run` smoke-run
(35 cases load, 34 scored).

---
## 2026-08-01 (3) — ADR-038: the in-RAM sparse arm is deleted. One keyword arm, and a failed build now **says** keyword search is off

**What changed.** `bm25_cache.py` (and its test file) deleted; `DOC_SPARSE_INDEX`, `DOC_BM25_CACHE`,
`_load_bm25_corpus`, `_split_parent_texts`, `_build_bm25`, `_bm25_docs`, `_parent_texts`,
`sparse_index.enabled()` and the in-RAM branch of `_ensemble_for` all gone. `sparse_index.py` is the
keyword arm. ADR-038 records it; ADR-035 is marked superseded outright, ADR-036 amended.

**Why now.** ADR-036 kept the old arm as a rollback because its A/B could only run on the public
10-case set, where both arms scored a saturated **1.0000**. The 2026-08-01 (2) run repeated it on the
private 35-case set — recall **0.70–0.83**, headroom everywhere, **post-rerank recall identical at
both k**, and the only movement (pre@10 +0.0147) favouring the on-disk arm.

**The part that needed designing rather than deleting: what a failed build now means.** Before, it
fell back to a slower in-RAM arm that still did keyword matching, and the user was never told. Now
there is nothing to fall back to — **retrieval runs on the vector arm alone** and an exact term the
embedder does not place nearby is missed. Removing a silent recovery means the state has to be
*said*:

- **`keyword_index_unavailable` is deliberately not `not sparse_index_active`.** The latter is also
  true for an empty library, which is a supported state and nothing to report. Conflating them would
  either cry wolf on every fresh install or stay silent on a real degradation. Pinned by a test that
  asserts both a failed build **and** an empty corpus.
- `corpus_stats` reports `mode="unavailable"` (replacing `in_memory`) and **withholds the index's
  size and build time** — a stale file may still be on disk, and its bytes would describe an index
  serving nothing.
- The panel says *"Keyword search is off — answers are using meaning-based search only, so exact
  terms may be missed"*, styled as a warning.

**An inversion worth noticing: `rebuild_sparse_index` used to raise when no index was live.** Correct
then (the fallback was serving, so a rebuild was meaningless); wrong now, because that state is
exactly what a rebuild fixes. It runs whether or not an index is live, and the route's only 409 left
is an empty corpus.

**A bug caught while writing it.** The first version read `self._corpus_empty` — a *construction-time*
snapshot. A fresh install launches against 0 documents, so after the user's first ingest the Rebuild
button would have refused with "the corpus is empty". Emptiness is now re-derived from the rebuild's
own fingerprint scan. Guard test:
`test_it_indexes_documents_ingested_after_an_empty_launch`.

**A vacuity trap found in the existing suite.** `test_pipeline_scope.py` selected the legacy arm with
`_sparse = None`. With that arm deleted the file still passed — because it was silently exercising
the *vector-only* path, and the vector fake honours the scope filter too, so
`test_scope_filters_the_bm25_arm` asserted nothing about a keyword arm at all. The rig now builds a
**real** on-disk index over the same four documents. **Verified non-vacuous by sabotage:** unscoping
the scoped `SparseRetriever` fails that test plus `test_a_scoped_turn_scopes_the_sparse_arm`, and
nothing else.

**Rejected.** (a) *Keep the flag, default it off* — a rollback nobody exercises is a second code path
that rots; git history is the rollback. (b) *Build a fresh in-RAM index on failure* — that is the arm
just deleted, minus the cache, at the moment the machine is already unhappy, and it restores the
silence. (c) *Raise on a failed build* — an unwritable data home would stop the app answering at all
(inform, don't block). (d) *Report `unavailable` with the stale file's size* — technically true,
actively misleading.

**Gates:** ruff + format clean · `mypy src` **86 files** (was 87) · **pytest 1445 passed** (1498
before; net −53 as the legacy arm's tests went with it, +4 new) · `docs_check --strict` 0/0.

**What it opens.** Answer-level equivalence is still unproven — 9 of 35 queries hand the model a
different evidence set, and recall@k does not score the other slots. And `sweep_bm25_weight`'s
docstring plus the ADR-036 baseline still assert "retrieval is deterministic", which 2026-08-01 (2)
falsified.

---
## 2026-08-01 (2) — Sparse-arm A/B repeated on the private 35-case set: shipped recall **identical**, but the arms return different evidence on 9 of 35 — and **retrieval is not deterministic**

**What changed.** No source code. One new baseline
(`tests/eval/baselines/sparse_arm_private35_2026-08-01.md`). ADR-036's A/B ran on the public 10-case
set where both arms scored a perfect 1.0000 on all four recall metrics — parity at the instrument's
ceiling, not proof — so the legacy in-RAM arm, `DOC_SPARSE_INDEX` and `bm25_cache.py` have shipped
alongside the default ever since, waiting for a better instrument.

**The blocker was not real, and that is the first finding.** Three baton entries recorded the private
35-case set as absent from this box. `tests/eval/cases.yaml` is **tracked in the repo** (35 cases,
dated 2026-05-28) and all 33 of its `expected_citations` fragments resolve against the live 97-document
library — **35/35 runnable**. The project's own top-priority item had been parked on an unverified
claim for four session entries. *Verify a "missing file" claim before inheriting it.*

**Result ($0, retrieval only, one process per arm, the arm asserted rather than assumed).** The
instrument discriminates this time — recall **0.70–0.83**, headroom in every direction:

| | on-disk | in-RAM | Δ |
|---|---:|---:|---:|
| pre@5 | 0.8186 | 0.8186 | 0.0000 |
| pre@10 | **0.8333** | 0.8186 | **+0.0147** |
| post@5 | 0.7010 | 0.7010 | 0.0000 |
| post@10 | 0.7598 | 0.7598 | 0.0000 |

**Post-rerank — what the user receives — is identical at both k.** The single movement is pre-rerank
and favours on-disk: `brain_network_hubs` surfaced `nihms-326467.pdf` at candidate rank 9 where the
control missed it entirely (pre@10 1.0 vs 0.5) — and **it does not reach the user**, because both arms
then return the identical three documents post-rerank. That is a second, independent instance of the
2026-07-03 weight-sweep finding: the cross-encoder re-scores the full union, so candidate-order
differences wash out. Arrived at by changing the ranking *function* this time, not the arm *weight*.

**⚠ The finding the recall table cannot express: 9 of 35 queries (26%) return a different evidence
set** — differing by *set*, not order, diverging as early as rank 2. Recall@k scores only whether the
*expected* document is present; it says nothing about the other 8–9 slots, which are exactly what the
LLM reads. Two arms can hand the model materially different evidence and score identically.

**⚠ Control result — retrieval is NOT deterministic, and that assumption was load-bearing.** ADR-036's
baseline says *"retrieval is deterministic on both arms, which is what `--repeat` buys elsewhere"*;
`sweep_bm25_weight`'s docstring says the same. Re-running the on-disk arm against the unchanged index
**disagreed on 1 of 35 cases**, so single-pass retrieval comparisons carry a ~3% case-level noise
floor. Measure it; do not assume it away.

**The noise is disjoint from the signal, which is what makes the 9/35 trustworthy** — and it is fully
localised. The noisy case's *pre-rerank candidate list is byte-identical across runs* (same ten files,
same order); only post-rerank parents moved, at ranks 5–6. So the flip is in the **cross-encoder**, not
either retrieval arm. And the case explains its own ties: `middleton_frontal_subcortical`'s only
expected citation is `middleton-2001.pdf`, which carries **`chunk_count=0, extraction_health='broken'`**
— the text-layer-less scan KI-29 exposed. Its target is not in the index, recall is 0.0 in both arms
and both runs, and with no true match the reranker scores cluster and the tail is decided by ties.

**Rejected.** (a) *`run_eval` in a loop* — it generates an answer per case, which is paid and
stochastic and which the arm does not affect; the weight-sweep's retrieval-only recall functions are
the right instrument and were reused rather than reimplemented. (b) *Trusting `DOC_SPARSE_INDEX`* —
the pipeline degrades silently to the in-RAM arm when the on-disk build fails, so the harness asserts
`sparse_index_active` and refuses to record on a mismatch; without it, a failed build would have
reported perfect parity for the worst possible reason.

**What it opens.** The bar the baton set is cleared and deleting the legacy arm is now defensible —
but it is a decision, not a measurement, and a separate increment. What is still unproven is *answer*
equivalence: closing that needs `contains_all`/`llm_judge` over the 35 cases (paid, and the case
file's own header warns its references are `author_verified: false` in places). Also opened: the
non-determinism claim in ADR-036 and `sweep_bm25_weight` should be corrected, and `middleton-2001.pdf`
is an unfixable 0-chunk case that will keep polluting any A/B run on this set.

---
## 2026-08-01 — Public eval re-measured after KI-29 + ADR-036: **no scorer moved beyond its variance**

**What changed.** No source code. One new baseline
(`tests/eval/baselines/public_eval_2026-08-01.md`) and the `evals/README.md` headline re-pointed at
it. The quality record had been describing an index that no longer existed: the locked reference is
**2026-06-01**, and two changes since then touched the answer path without ever being scored end to
end — **KI-29** (07-29) stripped `<!-- page:N -->` out of the LLM's evidence block, 49% of parent
texts on the live corpus, followed by a full re-embed; **ADR-036** (07-30) swapped the keyword arm's
ranking function, and FTS5's `bm25()` is not `rank_bm25`'s. ADR-036's own A/B measured only
*recall*, where both arms scored a saturated 1.0000.

**Result (n=5, `bge-base`, haiku generator + judge, `cu130`):** `citation_overlap` **1.000 ± 0.000**
· `contains_all` **0.932 ± 0.014** · `llm_judge` **3.694 ± 0.258**. Against the 06-04 reproduction
(1.000 / 0.927 / 3.738): **0.000 / +0.005 / −0.044**. Nothing moved.

**Why 06-04 and not the 06-01 locked reference.** 06-01 does not record its generator model, and the
DEVLOG shows `.env` was switched to `claude-haiku-4-5` on **2026-06-02** — after it was locked — while
fixing the `load_dotenv(override=False)` bug that had been shadowing the key. 06-04 ran after that
switch, so its generator is known to match today's. Comparing stochastic scorers across an unrecorded
generator change would have been the wrong diff.

**The isolation this needed, because the live corpus is 97 documents.** BM25/IDF statistics and the
vector neighbourhood are corpus-global, so a run against the live index is not comparable to any
committed baseline (RG-021). The run used a scratch `DOC_DATA_DIR` ingested from zero: **2,301 chunks
/ 10 documents**, **0** markers in children *and* parent texts, `sparse_index_active=True` with the
legacy in-RAM corpus empty. The check that actually proves it is after the fact — across all 50 turns
exactly **10 distinct documents** were cited, and they are the eval 10.

**Two things stated as limits rather than smoothed over.** (a) `citation_overlap` was *already*
1.000, so it demonstrates no regression **at the available resolution**, not ranking parity — the
same ceiling ADR-036 hit, and the reason `DOC_SPARSE_INDEX=0` and the legacy arm should stay until
the A/B is repeated on a discriminating case set. (b) This run's `llm_judge` trial-mean std is
**0.258**, roughly 3× June's, so it can only resolve judge changes larger than about ±0.5 — "no
regression" is a **weak** claim on that scorer and a strong one on the other two.

**The negative result worth keeping: KI-29 bought no measurable answer-quality gain here.** Removing
markers from the evidence block was expected to help and did not move anything beyond noise. Its
weakest link is stated in the baseline: marker density on the *public* corpus was never measured
(10%/49% are 97-corpus figures), so a null result on a barely-affected corpus proves little.

**A DB trap found while reading the results back.** `scores.value` is `DOUBLE NOT NULL` and a skipped
judge call is persisted as **`value = 0.0` with `scoreable = false`**. A first pass that averaged raw
`value` reported `sbert_motivation` at 2.933 and the overall judge mean at 3.620; filtering on
`scoreable` gives 3.667 and 3.694. Any aggregate read straight from `data/eval.duckdb` must filter —
the harness's own summary does.

**Also observed, not fixed.** `run_eval`'s `--db` default is a literal `PROJECT_ROOT / "data" /
"eval.duckdb"`, so it does **not** follow `DOC_DATA_DIR` like every other data artifact — these runs
landed in the main log despite an isolated corpus. Harmless and arguably useful (one log makes runs
comparable), but surprising enough to document. And `sbert_motivation`'s judge flake has now recurred
in all three runs (3/5, 3/5, 1/5 skipped): the 06-01 baseline called it a KNOWN_ISSUES candidate "if
it recurs", and it has, twice.

**What it opens.** The quality record is current again. Still owed: the sparse A/B on an instrument
that can discriminate (the private 35-case set is not on this box) before deleting the legacy arm.

---
## 2026-07-30 (4) — PF2 closed as **no knobs**: ADR-036 dissolved the premise, so the app ships corpus facts instead

**What changed.** New `corpus_stats.py` (documents, chunks, disk per artifact, which keyword-index
arm is live + its size and build time), `ChatController.corpus_stats()` / `rebuild_keyword_index()`,
`RAGPipeline.sparse_index_active` + `rebuild_sparse_index()`, `POST
/api/settings/reindex-keywords`, the `corpus` block on `GET /api/settings`, and the Settings →
**Corpus** section rendering it through a tested pure module (`settings/corpus.ts`). 20 guard tests
(8 backend, 3 route, 9 frontend). ADR-037 records the decision; ROADMAP PF2 closed.

**The finding that decided it.** PF2 was filed while the sparse arm still held the corpus in RAM.
Checking its proposed knob list against the code *after* ADR-036 — before asking the user anything —
found the premise mostly gone: `DOC_BM25_CACHE` now only affects the legacy arm; the
scoped-ensemble cache size lost the rebuild cost it traded against; `MARKER_MAX_WORKERS` /
`MAX_VLM_CALLS_PER_DOC` / `FIGURE_RENDER_DPI` are read **only by `scripts/`**, never by the in-app
ingest; and the one new switch, `DOC_SPARSE_INDEX`, is **not output-neutral**, which was the test
that made a knob safe to expose. What remained was a single minor toggle — while the need behind the
request ("a few hundred documents is not a lot") had already been met in engineering.

**So the question put to the user was the right one to ask**, and all three answers are theirs:
corpus facts in Settings rather than presets or documentation-only; **one** bounded action (rebuild
the keyword index, not a full re-index — that is hours at 10k documents with no progress yet); and
the memory line **states the shape, never a live number** (a live RSS figure needs a new dependency,
fluctuates, and is dominated by model weights a user would misread as their corpus's cost).

**The design rule worth keeping: the sentence is frontend copy, `keyword_index.mode` is the wire.**
The two arms say *opposite* things about memory — on-disk keeps it flat, the legacy in-RAM arm grows
at ~5.9 KB/chunk. A pre-rendered sentence from the backend would be one refactor away from
reassuring a user whose process is doing the reverse. For the same reason `corpus_stats` reports the
**live pipeline's** arm rather than whether an index file exists: a stale file beside the legacy arm
would otherwise answer the reassuring way. Both are pinned by tests.

**Rejected.** (a) *The three-tier preset proposal* in `performance.md` §5 — written pre-ADR-036, and
its Tier A is now empty; §5 is rewritten to record why. (b) *Reading `controller.rag._sparse` from
the router* — the shell would have owned logic (non-negotiable #3); the assembly moved to the
controller and the pipeline grew a public `sparse_index_active`. (c) *A `202` + status route for the
rebuild* — it is seconds here and minutes at 10k documents, bounded work with a definite end, so
polling machinery would buy nothing; the ADR names the condition that would change that. (d) *A
confirmation dialog* — the index is derived data the next launch would rebuild anyway; a confirm
would imply a risk that does not exist.

**The rebuild's real hazard, and the test for it.** Swapping a live index means three things move
together — the handle, the prebuilt ensemble whose `SparseRetriever` binds it, and the scoped-
ensemble LRU whose entries bind it too. Miss any one and the app serves results from the index the
user just replaced, with no error anywhere. `TestRebuild` pins all three plus the 409 when the
on-disk arm is not live.

**Gates:** ruff + format clean · `mypy src` **87 files** · **pytest 1498 passed** (+18) ·
`svelte-check` **187 files 0/0** · **npm test 67/67** (+9) · `docs_check --strict` 0/0.
**Live-verified** on the real corpus (97 documents / 33,105 chunks): the panel reads *97 documents ·
33,105 chunks · 589 MB · 6.1 MB per document · on disk, 39 MB, built 4 h ago*; clicking **Rebuild**
went `built 4 h ago` → `Rebuilding…` → `built just now` with no console errors, and retrieval after
the live swap still returned 10 sources led by `dpr_karpukhin_2020.pdf`. Checked at 1280 px and at
375 px dark (no overflow, label and button do not collide). ⚠ The pane reports
`visibilityState: hidden`, which freezes the fly transition and parks the drawer off-screen — the
trap recorded twice before; geometry was measured after neutralising the transform, and the section
matches its siblings' box exactly.

---
## 2026-07-30 (3) — ADR-036: the sparse arm moves to an on-disk SQLite/FTS5 index. **195 → 21 MB, no corpus in RAM** — and retrieval changes, so it was gated on an A/B

**What changed.** New `sparse_index.py`: one SQLite database beside the Chroma store holding chunk
text + metadata, a **contentless FTS5 index** over the same `keywords.tokenize` token stream, and the
parent blocks in their own table. `RAGPipeline` wires a `SparseRetriever` into the ensemble instead of
`BM25Retriever`, resolves parent text through the index, and scopes a folder turn with
`WHERE doc_hash IN (...)` inside the ranked query instead of rebuilding an index over a subset.
`chroma_read.iter_pages` (new, `get_all` now a thin accumulator over it) lets the build stream.
`DOC_SPARSE_INDEX=0` keeps the legacy in-RAM arm. **KI-32 is resolved and archived.**

**Measured on the live corpus (97 docs / 33,105 chunks), on-disk vs in-RAM, separate processes:**

| | in-RAM (control) | on-disk (shipped) | |
|---|---:|---:|---|
| Python heap after construction | 185 MB | **21 MB** | **8.8×**, and **0 chunks resident** |
| Construction | 4.53 s | **2.79 s** | 1.6× |
| Sparse arm per query | 66.6 ms | **27.4 ms** | 2.4× |
| `retrieve_with_scores` per turn | 336 ms | **279 ms** | 1.2× |
| On disk | 39.9 MB snapshot | 40.7 MB index | ~equal |

**Why FTS5 and not an exact-equivalence rewrite.** Option B — reimplementing Okapi over an SQLite
postings table — would have preserved ranking *exactly*, and was rejected on its slope: the in-RAM
arm's semantics are "score every document containing any query term", stopwords included, so
reproducing them on disk means reading most of the corpus per query, in Python. It would have
inherited the cost profile this work exists to remove. Tantivy was rejected for the packaging risk
(KI-9) with no gain over FTS5.

**So retrieval changes, and that is the honest cost.** FTS5's `bm25()` is k1=1.2 with no IDF floor;
`rank_bm25` was k1=1.5, epsilon=0.25. Spiked *before* committing to the design: **84% of top-20
candidates agree**. End to end, after the cross-encoder re-scores the union: **8 of 10 public eval
queries return a byte-identical final top-10**, 2 differ by one document, and recall against the
cases' expected citations is **identical at 1.0000** on pre@5/pre@10/post@5/post@10. ⚠ **Both arms
score a perfect 1.0000, so that instrument has a ceiling** — it demonstrates parity at the resolution
available and cannot see a small regression. Hence `DOC_SPARSE_INDEX=0` stays until the A/B is
repeated on a discriminating case set.

**Three things the probes corrected, each after being wrong first.**
1. **Chroma is not corpus-resident.** A stage-by-stage working-set probe first showed **+449 MB** on
   "the first vector query", which read as *the vector store is now the ceiling*. Isolating the embed
   step moved all 449 MB to **CUDA context initialisation**; the vector query itself costs **+1 MB**.
   A wrong conclusion about the next piece of work, avoided by one more measurement.
2. **The launch id scan was the last corpus-linear allocation.** Collecting chunk ids to fingerprint
   the store peaked at **3.1 MB (94 B/chunk ⇒ ~0.3 GB at the contract)**. The fingerprint now streams
   page by page and is order-independent by **summing** per-id digests rather than sorting them —
   sorting is what forced the list into memory. Same digest, proven by a test.
3. **The first memory-property test was vacuous.** It asserted `_bm25_docs == []` on a rig that had
   set that attribute itself, so it passed with the in-RAM load reintroduced. Rewritten to construct a
   **real** `RAGPipeline` with fakes only at its boundaries; it now fails when the guard is removed.

**Rejected.** (a) *Deleting the in-RAM arm now* — it is the rollback and it was the control for the
A/B; deleting it before the eval can discriminate would leave no way back from a quality regression
nobody can currently see. (b) *Keeping the scoped-ensemble LRU's rationale* — the ~20 µs/chunk
rebuild it existed for is gone; the memo stays because it still serves the fallback path, and the
comment now says so rather than implying a cost that no longer exists. (c) *An `AND` MATCH expression*
— FTS5's default for a bare term list, and it would have silently returned a fraction of the
candidates. (d) *Serving a stale index* — unlike `bm25_cache`, this **is** the arm, so a stale or
corrupt database is rebuilt, never used.

**Guard tests: 52 new** (`test_sparse_index.py` 36 + `test_pipeline_sparse_arm.py` 16; suite
1432 → 1484), split the way ADR-035's were: *semantics* (OR-not-AND, tokenizer parity
including `cross-encoder`, scope applied before `LIMIT`, FTS5 operators in user text treated as data),
*refusal* (stale ids, wrong collection, changed tokeniser, corrupt file, foreign database, failed
build leaving no half-index), and *the memory property* itself. Two existing rigs
(`test_pipeline_scope.py`, `test_pipeline_parent_texts.py`) now set `_sparse = None` explicitly —
they pin the fallback arm, which is still shipped.

**A defect the tests caught, worth keeping.** `fingerprint` originally imported `tokenize` at module
level, so monkeypatching `keywords.tokenize` could not invalidate the index — the separate-binding
trap `src/doc_assistant/CLAUDE.md` already records, one layer down. Now imported per call.

**Live-verified, $0** (Ollama `llama3.1:8b`): one real `ChatController` turn returned **10 sources**,
top `dpr_karpukhin_2020.pdf` at **0.9795**, scores **sigmoid-bounded in [0,1]** (the integrity layer
depends on that), no page markers in the evidence, clean citation note, `_bm25_docs` empty throughout.

**Gates:** ruff + format clean · `mypy src` **86 files** · **pytest 1484 passed** (+52) ·
`docs_check --strict` 0/0. `.gitignore` covers the new 41 MB artifact — the same trap ADR-035 hit,
caught this time before staging.

**What it opens.** Backend RAM is now **flat at ~2 GB**, so the binding constraint moves to the
**first ingest** (~41 h of single-threaded extraction at 10k documents) and then to disk (~60 GB).
Launch still scans every chunk id (~0.2 s at 33k, ~20 s projected); the successor for that is an
ingest-side version stamp, which would also retire chromadb's ~159 MB paged-read high-water mark.

---
## 2026-07-30 (2) — KI-32 step 1: `parent_text` deduplicated out of the in-RAM corpus. Predicted 3x, measured **1.36x** — and the shortfall is the finding

**What changed.** `pipeline._split_parent_texts` lifts `parent_text` out of every chunk's metadata
into one `(doc_hash, parent_index) -> text` entry, and `pipeline._parent_text_for` re-attaches it for
the parents a turn actually returns. `bm25_cache` payload **v2** carries the map instead of ~5.5
copies per parent (`_CACHE_VERSION` bumped, so an existing snapshot is rebuilt, never reinterpreted).
17 new guard tests (`test_pipeline_parent_texts.py` 14, `test_bm25_cache.py` 28 -> 31).

**Measured on the live corpus (97 docs / 33,105 chunks, 6,045 parents = 5.5 children each):**

| | before | after | |
|---|---:|---:|---|
| Python heap for the BM25 corpus | 265 MB | **195 MB** | **1.36x** (8,012 → 5,892 B/chunk) |
| Snapshot on disk | 85.2 MB | **39.9 MB** | **2.1x** |
| `RAGPipeline()` construction | 5.81 / 5.97 s | 5.62 / 5.82 s | unchanged within noise |
| Retrieval output | — | — | **identical**: 3 queries, 10 sources each, same documents, same scores |

**The prediction was 3x and it was wrong, so the interesting result is the attribution.** The estimate
came from arithmetic on text sizes (a 400-char child carrying a 2,000-char parent, ~5 copies). The
duplication factor was right (5.5) and the mechanism was right; the premise that text dominates was
not. `tracemalloc` by allocation site, per chunk: **~1.4 KB Chroma metadata dicts + strings · ~1.4 KB
BM25 token strings retained by the index · ~0.8 KB `BM25Okapi.doc_freqs` (one dict per chunk) ·
~0.6 KB pydantic `Document` overhead · ~0.4 KB child text · ~0.4 KB parent text.** **The chunk text is
~8% of the footprint.** So step 2 (index off the heap) is not an increment on step 1 — it removes the
token strings, the frequency dicts and the `Document` materialisation, ~85% of what remains. Every
downstream claim was corrected: RAM ≈ 2 GB + **2.0** MB/document (was 2.7), 16 GB wall ~**5,000**
documents (was ~3,700), contract **~22 GB** (was ~29).

**Equivalence, and how it was proved without a second build.** In one process: run the queries on the
shipped path, then restore `parent_text` into the same `_bm25_docs` metadata from the map, empty the
map, and re-run. That reproduces the pre-change form against the same store, so a diff is a real
difference and not a corpus or cache artefact. Identical on all three queries.

**Metadata first, map second — the order is load-bearing.** The vector arm returns documents straight
from Chroma, which still stores `parent_text` on every child, and a document ingested *while the app
is running* can only be expanded that way (the map is a construction-time snapshot). Preferring
metadata keeps that case working exactly as before and makes the change invisible to every existing
caller and test; the map is only what the BM25 arm now needs.

**Rejected.** (a) *Fetching the top-K parents from Chroma per turn* — the original KI-32 sketch. A
deduplicated in-memory map gets the same reduction with no per-turn I/O and no chance of the store and
RAM disagreeing; the fetch only becomes necessary in step 2, when the documents stop being resident at
all. (b) *Map-only lookup, dropping the metadata branch* — it would silently drop hits for
documents ingested after construction, and "silently drops hits" is the exact failure shape this
change had to avoid. (c) *Copying metadata instead of deleting the key* — holds both forms at once and
saves nothing; the in-place delete is what lets the duplicates be collected. (d) *Removing
`parent_text` from what Chroma stores* — that is a storage change forcing a re-embed, and the
enrichment layer reads it from the store (`epistemics`, `concept_skeleton`, `library/chunks`).

**Guard tests, and they are non-vacuous.** Reverting `_parent_text_for` to the old
`doc.metadata.get("parent_text")` fails **3 of 13** in the new file (measured before the 14th, the
folder-scoped one, was added), including the equivalence test.
One test asserts the *absence* of `parent_text` in the in-RAM metadata, i.e. it fails if someone
restores the duplicates; one asserts that with an empty map those candidates vanish, which is what
makes the "still expands" test mean something. Cache side: a payload missing `parents` is **refused**
rather than read as "no parents" (that would drop every BM25-only hit), and a v1-fingerprinted
snapshot is stale by construction.

**Gates:** ruff + format clean · `mypy src` **85 files** · **pytest 1432 passed** (+17) ·
`docs_check --strict` 0/0.

**What it opens.** PF3 / KI-32 step 2, now with attribution instead of a guess about where the memory
is. Also worth noting for it: the snapshot halved but construction did not get faster, so unpickling
was never the launch cost — the launch cost is building the index and materialising documents, which
is precisely what step 2 removes.

---
## 2026-07-30 — the cost/scale record gets one home, and the optimisation pass gets an honest ledger (KI-32 found by measuring)

**What changed.** New `docs/performance.md` (living): launch/query/ingest/sidecar cost, memory, disk,
an **optimisation trade-off ledger**, linear projections to 1k and 10k documents, the knob inventory,
a recommendation on which knobs should become user-facing, and a measurement-debt list. Pointers
added from `README.md` (doc table + the Benchmarks section + a corrected Limitations bullet),
`docs/setup.md` (hardware), `evals/README.md` (routing row), `AGENTS.md` (Reference line). ROADMAP
rows **PF1-PF3**. Two new measurements recorded in
`tests/eval/baselines/memory_and_lazy_reranker_2026-07-30.md`. **No source code changed.**

**Why a second file and not one benchmark page.** `evals/README.md` is the *quality* record, produced
by the eval harness over a fixed question set. Cost is a different instrument, a different corpus
condition (device, file cache) and a different audience question. Splitting on that axis keeps **one
home per number**; the alternative was a third copy of the quality table, which is the shape things
drift in. Both files now route to each other explicitly so neither reads as the whole story.

**Two figures measured because the ledger could not be written without them.**
1. **The answer path's memory, which had never been measured.** `tracemalloc` around
   `RAGPipeline()`: **265 MB of Python heap for 33,105 chunks = ~8.0 KB/chunk** (working set 1,821 MB
   before the reranker, 2,327 MB after). Cross-checked against the 85 MB snapshot of the same corpus
   (2.6 KB/chunk packed) — ~3x for live Python objects is what `str`/`dict` overhead predicts, so the
   two agree. Projected at the 10,000-document contract: **~27 GB**. Filed as **KI-32**.
2. **The lazy-reranker penalty on the current wheel.** Recorded as ~3.7 s from a CPU split; measured
   on `cu130` it is **~5.0-5.3 s** (weight load 4.84-5.03 s + 0.2-0.3 s CUDA warm-up, steady-state
   `predict` 0.01-0.02 s). **The trade got sharper on both sides with the GPU, not better**: shorter
   launch, longer first answer, much faster answers after. The 07-28 baseline is append-only and was
   not edited; the correction lives in the new baseline and the ledger.

**The conclusion the ledger forced, and it is not flattering.** ADR-035 made the launch of a design
that **cannot reach 10k documents** faster. It was still the right call (that is where the slope was,
and the cache is correct, fingerprinted and disableable), but it lowered a constant and left both the
O(corpus) shape and the RAM ceiling untouched — and by removing the symptom (a slow launch) it removed
the thing that would have surfaced the ceiling. The usable form of the finding is a formula, not a
verdict: **backend RAM ≈ 2 GB + ~2.7 MB per document**, so ~4.7 GB at 1,000 documents, a practical
wall **under 4,000 on a 16 GB box**, ~9,000 on 32 GB, and ~29 GB for the contract. A few hundred
documents is cheap; a few thousand works; past that it is a redesign (PF3), and no setting bridges it.

**Two things verification changed while the tables were being written, both worth keeping.**
1. **Where the 8 KB/chunk actually goes: not the chunk text.** Every child chunk carries its
   **parent's full text** in metadata (`ingest/chunking.py:194`), so ~5 children each hold a copy of
   the same 2,000-character parent. ~2.4 KB/chunk of text × ~3x for live Python objects ≈ the measured
   8.0 KB. That makes step 1 of KI-32 a **bounded** change rather than a redesign (drop `parent_text`
   from the in-RAM copy, look the top-K parents up instead) — but not a free one, since
   `retrieve_with_scores` reads `parent_text` off the winning candidate to build the parent it
   returns. ⚠ **The "~3x, wall moves to ~10,000 documents" predicted here was wrong** — it was built
   on this same text-size arithmetic. Implemented the same day: **1.36x**. See the entry above.
2. **Disk was understated: `ingest` writes BOTH vector stores** unconditionally
   (`ingest/__init__.py:272` baseline, `:300` parent-child), and only the parent-child one serves the
   default answer path. 120 MB of the 628 MB here, ~19% at any size. Not a bug (flat mode and some
   sidecars read the baseline store) but it belongs in a disk estimate, so the per-document figure is
   **6.5 MB**, not the 5.2 MB the parent-child store alone would suggest.

**On the knob question (PF2), the useful distinction turned out to be output-neutrality.** Every knob
this optimisation pass introduced (BM25 snapshot, reranker laziness, the scoped-ensemble LRU, ingest
workers) trades time/space/money and **cannot change what an answer says** — so exposing those does
not touch the locked-settings rule at all, while `TOP_K`/`CANDIDATE_K`/chunk sizes/embedder cannot be
exposed without either an eval experiment or making `evals/` unattributable. Recommendation recorded,
decision deliberately **not** made here: it wants `grill-me` and an ADR.

**Rejected.** (a) *Folding cost into `evals/README.md`* — see above. (b) *Editing the 07-28 baseline
with the corrected 5 s penalty* — those files are append-only for exactly this reason; a new dated
baseline plus a ledger row keeps the history readable. (c) *Filing the memory ceiling under KI-18* —
KI-18/19 are scoped to `knowledge/`, and this is the answer path, which is why nobody had found it
there. (d) *Adding a `--memory` mode to `scripts/profile_stages.py` now* — the right home, but this
increment is docs; recorded as debt with the method written out so it is reproducible meanwhile.
(e) *Quoting a new end-to-end launch figure* — the snapshot's 2.7x is a **stage** measurement, and
inventing a launch total from it would be exactly the arithmetic-dressed-as-measurement this file
exists to stop. Logged as debt instead.

**What it opens.** PF2 (the exposure ADR) · PF3 / KI-32 (the sparse arm off the heap, plus resumable
parallel extraction — ~41 h projected at 10k documents, and extraction is GPU-immune) · a synthetic
≥1k-document corpus, which is the only way to test the linear assumption every projection here rests
on (RG-016 already wants one).

---
## 2026-07-29 (5) — ADR-035: the BM25 arm launches from a persisted snapshot (5.36 s -> 1.99 s)

**What changed.** New `bm25_cache.py` + `RAGPipeline._load_bm25_corpus` / `_build_bm25`. The BM25
corpus is written once to a snapshot beside the Chroma store and reloaded on subsequent launches
instead of being re-read from Chroma and re-tokenised every time. ADR-035 carries the full
reasoning; `.gitignore` excludes the artifact (85 MB, pure derived data).

**Why this and not something else.** With the reranker lazy (07-28) and the GPU wheel in (07-29),
this is the **only startup component that scales with the corpus** — everything ahead of it (Python
imports, embedder weights) is a constant. At the 10,000-document contract the step alone projects to
minutes, so the 1.7-5 s today is not the problem; the slope is.

**The constraint that picked the design.** `self._bm25_docs` is retained for ADR-025 F2
folder-scoped retrieval, so the documents must be materialised regardless — "persist the index"
could never mean "skip loading the corpus". Whatever is persisted has to carry the documents too.

**Measured, cache off vs on, interleaved across 4 rounds** so machine-load drift hits both arms
equally: **5.359 s -> 1.990 s, 2.7x**. Retrieval verified **identical** on the live corpus — three
queries, 10 sources each, same documents and same scores with the cache off and on.

**What is stored, and what deliberately is not.** Only stdlib types: `(text, metadata)` tuples plus
token lists, with `Document` / `BM25Okapi` / `BM25Retriever` reconstructed at load. Pickling the live
retriever measured **0.354 s** (vs ~0.50 s) and was **rejected**: it makes the on-disk format an
implementation detail of a third-party class, where a langchain upgrade can deserialise into a
subtly different object instead of failing outright. A cache whose worst failure is "retrieval
quietly changes" is the wrong trade for 150 ms — especially in the same session that paid for KI-31.

**Two things measurement caught that reasoning had not.**
1. **The first fingerprint never hit — not once.** It keyed on `chroma.sqlite3`'s mtime, on the
   reasoning that ingest is the only writer. But **opening a `chromadb.PersistentClient` rewrites
   that mtime even for a pure read**, so every launch invalidated the snapshot it had just written.
   The fingerprint is now the **sorted chunk ids** (0.221 s for 33,105 ids, against 5.389 s for the
   full read it guards) plus the collection name and `inspect.getsource(tokenize)`. Both directions
   are pinned: replaced ids invalidate; touching the store file alone does not.
2. **The miss path tokenised the corpus twice** — once inside `BM25Retriever.from_documents` and
   again to build the payload — and did it *even with the cache disabled*, because argument
   expressions evaluate before the call. Now it tokenises once and hands the terms to both, and
   skips the work entirely when `DOC_BM25_CACHE=0`.

**Honesty note on the absolute numbers.** The same whole-store read measured **1.06 s** early in the
session and **5.39 s** later the same day on an unchanged store — OS file-cache state dominates.
Quote the interleaved **ratio**, not either absolute; the ADR says so too.

**Rejected.** (a) *Persisting only the index, not the documents* — impossible given F2 above.
(b) *Documents without pre-tokenised terms* — measured 0.921 s, leaves the tokenise pass on the
table. (c) *`collection.count()` as the staleness signal* — 0.011 s and tempting, but blind to a
`--rebuild`, which replaces every chunk while keeping the count identical.

**Costs, stated.** ~85 MB on disk here (~9 GB projected at 10k documents, comparable to the store
itself), and the first launch after an ingest pays the write. `DOC_BM25_CACHE=0` disables it without
a code change, so a suspected cache problem can be ruled out without deleting files.

**Guard tests:** `tests/unit/test_bm25_cache.py` (28), split into *equivalence* (identical scores and
identical returned documents vs a freshly built index) and *refusal* — stale, corrupt, truncated,
not-a-dict, mis-paired docs/tokens, changed tokeniser, changed collection, reordered ids, empty
corpus, unwritable destination, no leftover temp files. The refusal half is the point: a cache over a
retrieval input must fail to a slower launch, never to a different answer.

---
## 2026-07-29 (4) — KI-31: `get_all` dropped every page of embeddings after the first; document similarity ran on 39% of the corpus

**What changed.** `chroma_read.get_all` concatenated pages with `isinstance(value, list)`, but
chromadb returns **`embeddings` as a `numpy.ndarray`** — which failed that test and fell through to
"keep the first page, discard the rest". New `_is_array` (duck-typed on `shape` + `__len__`, so the
module keeps no numpy import) extends the concatenation branch. Separately,
`doc_vectors.load_chunk_embeddings_by_document` now raises on a length mismatch and zips
`strict=True`.

**Why / how it was found.** Re-running `compute_doc_vectors` after the KI-29 re-embed reported
**"Documents in library: 37"** on a 97-document corpus. A read of the 12,786-chunk baseline store
was returning **5,000 embeddings against 12,786 metadatas**, and the caller's
`zip(..., strict=False)` discarded the surplus without a word.

**Measured, before → after:** `get_all` 5,000/12,786 → **12,786/12,786**; `compute_doc_vectors`
**37 documents / 370 edges → 96 / 960**. Edges recomputed and persisted.

**Blast radius is exactly one caller.** `load_chunk_embeddings_by_document` is the only `get_all`
call that requests `embeddings`; the other nine ask for `documents`/`metadatas`/`ids`, all plain
lists, and paged correctly throughout. Damage was confined to `doc_similarities` — the
related-documents panel and the similarity graph. **Nothing on the answer path** (BM25 build,
retrieval, epistemics) was affected.

**The lesson worth keeping.** This arrived *with* the KI-27 paging fix on 2026-07-25 — the unpaged
read it replaced returned everything in one call, so the ndarray never needed concatenating. It was
invisible to that fix's own tests because the fake collection returned a **list for every key**: the
one type that breaks concatenation was the one type never exercised. **A silent truncation that
still produces well-formed output is the worst shape a bug can take** — edges were produced, scores
were plausible, the graph rendered, and it described 37 of 96 documents.

**Rejected.** Importing numpy into `chroma_read` for an `isinstance` check — the module pages a
store, it does not do numerics; duck-typing keeps the dependency out.

**Guard tests:** `FakeCollectionWithEmbeddings` (returns an ndarray, like the real thing) plus two
tests pinning cross-page concatenation and row/metadata alignment. **Verified non-vacuous:**
restoring the old check fails exactly those two.

---
## 2026-07-29 (3) — KI-29 closed: the parent-child chunker now strips page markers (option 2, + the re-embed)

**What changed.** `build_parent_child_chunks` applies `clean_chunk_text` to **both** the child
`page_content` and the `parent_text` metadata. Stripping happens at assembly, **not** on `text` up
front, so chunk boundaries and the table-caption binding in `_table_aware_parents` see exactly the
same input as before — only the stored text changes. Corpus re-embedded (`ingest --rebuild`, 4:01).

**Why option 2 and not the cheap patch.** The re-embed was the only argument for option 1, and the
GPU move cut it from ~17 min to ~2 min (entry 1 today). The user chose the correct place.

**Measured on the live corpus, before → after** (the whole store was scanned, not sampled):

| | before | after |
|---|---:|---:|
| child `page_content` with a marker | **3,312** / 33,163 (10.0%) | **0** |
| `parent_text` with a marker | **16,254** / 33,163 (49.0%) | **0** |
| baseline store (always did it right) | 0 | 0 |

**Nearly half of all parent texts** — the evidence block the LLM receives — carried a marker. KI-29's
original "2 of 10 excerpts" understated it. Live `retrieve_with_scores`: 10 sources, **0** markers.

**It uncovered a second bug, and that is the part worth remembering.** The re-embed failed on
`middleton-2001.pdf`: `Expected Embeddings to be non-empty list ... in upsert`. That document is a
**scan with no text layer** — its whole cached markdown is 290 characters of 15 page markers.
*Before* this change those markers were **embedded as if they were content**; after it the document
correctly yields zero chunks, and Chroma rejects an empty upsert. `ingest` now guards both
`add_documents` calls on a non-empty list, drops baseline chunks that clean to empty, and logs
`no_indexable_text` — so the document is recorded honestly as `chunk_count=0` / health `broken`
(the KI-24 sweep keeps its library row) instead of aborting the run. **0 chunks is a state to
report, not to crash on.** Re-run: **97 added, 0 errors**, and the document is now correctly flagged
`broken: only 0 chunk(s) produced`.

**Rejected.** (a) *Stripping `text` up front, before `_table_aware_parents`* — removing
`
<!-- page:N -->
` leaves a blank line, which `_split_trailing_paragraph` reads as a paragraph
boundary; that would quietly change table-caption absorption. Assembly-time stripping leaves the
splitters' input byte-identical. (b) *Fixing it in `library/chunks.py` or the frontend* — the reading
view's job is to show what the retriever stored; stripping there would make the one honest window
into the store lie about it. (c) *Renumbering `chunk_index` after dropping empties* — indices come
from `enumerate(raw_chunks)`, so survivors keep their original index and `chunk_epistemics`'
`(document_id, chunk_index)` / `{doc}:p{parent_index}` keys stay aligned across the re-embed.

**Guard tests:** `tests/integration/ingest/test_page_marker_stripping.py` (11), asserting on the
*chunker's output* rather than the stripper — the bug was always that a working stripper was not
called on the path that matters. **Verified non-vacuous:** reverting the change fails 5 of them.

**What it opens / carried.** The fix is in the chunker, so any existing corpus keeps its markers
until `ingest --rebuild` is run (~4 min here). `doc_similarities` was recomputed afterwards
(see entry 4); `compute_epistemics` was left alone — its keys survived the re-embed by construction.

---
## 2026-07-29 (2) — KI-30 closed: one shared `--doc` resolver behind all four sidecar runners

**What changed.** `library.documents.resolve_document_prefix` (+ `DocumentRef`,
`DocumentPrefixError`) is now the single entry point behind every runner's `--doc`. It tries an
**id prefix first**, falls back to a **doc_hash prefix**, escapes `LIKE` wildcards, and raises a
CLI-ready message naming the candidates when a prefix is ambiguous. All four runners
(`extract_citations`, `extract_doc_metadata`, `compute_doc_vectors`, `extract_keywords`) call it;
`compute_doc_vectors._resolve_doc_filter` — the previous best-behaved variant, and the model for
this one — was deleted in its favour.

**Why.** KI-30: the flag meant three different things across four runners, and the two that filter
on `doc_hash` rejected the very `Document.id` that the API, the graph, the library grid and
`list_documents()` all hand out. `extract_citations --doc <document id>` exited 1 with "No documents
matched." while its own help promised id support. That is the slowest sidecar (54 s whole-corpus),
so it is exactly where per-document scoping pays, and it was the one that could not be driven by the
id a caller actually has.

**Verified on the live corpus (97 docs, $0).** The exact command from the KI-30 symptom now returns
the document: **7.4 s scoped vs 54 s whole-corpus** (most of the 7.4 s is interpreter startup — the
*work* is genuinely scoped). Hash prefixes still resolve; an unknown prefix still exits **1**; `--doc
f` reports `3 documents share that id prefix — f027a3d4 (41111_2021_Article_191.pdf), fe739b78
(ai_usage_cards_2023.pdf), f9cea60a (elife-61909-v3.pdf)`.

**Help text now states what each runner actually scopes** — including `extract_keywords`, which says
out loud that scoping the write does **not** scope the corpus TF-IDF (~4% saving, KI-18). The flag
no longer pretends to be an incremental switch where the module is corpus-global by construction.

**Rejected.** (a) *Making all four runners scope the work* — `extract_keywords` and
`compute_doc_vectors` are corpus-global by construction; the honest fix is to say so, not to fake
it. (b) *Unifying `library.pins.find_document_by_short_id` too* — a third variant (id-only, returns
`None`), but its contract is what the chat pin flow needs and changing it would alter pin behaviour
for no current gain. Left deliberately, noted in the archived KI-30. (c) *A shared helper module
under `scripts/`* — this is business logic; `src/` owns it (non-negotiable #3).

**Guard tests:** `tests/unit/test_doc_prefix_resolver.py` (15). The first pins the regression
itself; others pin id-beats-hash precedence, ambiguity, blank input, the **0-document** corpus
(robustness contract), and that `_` cannot act as a `LIKE` wildcard. Two wiring tests assert each
runner imports the shared resolver and that neither hash-only runner still contains
`doc_hash.startswith` — **a correct resolver nobody calls fixes nothing**.

**What it opens.** Per-document enrichment is now drivable from an id, which is the precondition for
running citations/metadata incrementally on ingest instead of as a whole-corpus batch. Not wired to
ingest here.

---
## 2026-07-29 (1) — the GPU wheel: query 3.1x, re-embed 7.9x, launch unchanged; a wrong setup.md claim corrected

**What changed.** `uv sync --extra cu130 --extra dev` on this box (RTX 4070, `DOC_TORCH=cu130` was
already set machine-wide; the venv had drifted to the CPU wheel). **No code changed** — neither
`get_embeddings()` nor the `CrossEncoder` construction pins a device, so both auto-select CUDA once
the CUDA wheel is present. New baseline recorded at
`tests/eval/baselines/stage_profile_2026-07-29.md`; the 07-28 CPU baseline was **not** edited.

**Why.** Item 1 of the 07-28 baton: every transformer figure in that baseline was CPU-side on a box
with an idle GPU, which put a device caveat on the two conclusions that mattered (the rerank share
of the query budget, and the re-embed cost).

**Measured** (same instrument, same flags, same corpus, same sampled documents):

| Stage | CPU (07-28) | GPU (07-29) | Speedup |
|---|---:|---:|---:|
| `retrieve_with_scores` (retrieve + rerank + expand) | 907 ms | **296 ms** | **3.1x** |
| ⇒ cross-encoder rerank + expand, by difference | ~680 ms | **~134 ms** | ~5.1x |
| embed rate | 31.6 ms/chunk | **3.8 ms/chunk** | **8.3x** |
| ⇒ full re-embed of 33,163 chunks (projected) | ~17 min | **~2.1 min** | 7.9x |
| BM25 search / index build | 28 ms / 698 ms | 26.7 ms / 662 ms | 1.0x (pure CPU — expected) |
| cold PDF -> markdown extraction | 15.2 s/doc | **14.7 s/doc** | **1.0x — no gain** |
| cold launch, fresh process | 11.7 s | **12.10 s** | **none — marginally worse** |

**Three conclusions from the 07-28 baseline change, and one negative result matters most.**
(a) *"The reranker is three quarters of the retrieval budget, the one part worth optimizing"* — on
the GPU it is **~45%** of a sub-300 ms budget. Nothing in the query path is worth optimising now.
(b) **KI-29's cost argument collapses**: the re-embed that made option 1 (patch at retrieval) the
pragmatic choice is now ~2 min, so option 2 (fix `build_parent_child_chunks`, the correct place) is
affordable. Re-costed in KI-29; the user's call, recommendation recorded. (c) **Extraction is not a
GPU workload** — 14.7 s/doc vs 15.2 s, within noise. It is the most expensive per-document cost in
the system and any future ingest-throughput work has to target it, not embedding.

**Launch got slightly *worse*, and that is stated rather than averaged away:** CUDA context init
costs ~0.4 s and buys nothing at startup, which is dominated by Python imports + embedder weights.
The profiler takes a **single** cold sample, so the extra three fresh-process runs (12.58 / 12.10 /
12.14 / 11.90 s) were measured by hand — one sample cannot support a 0.4 s claim.

**A user-facing claim was wrong and is now corrected.** `docs/setup.md` promised **~70 ms**
retrieve+rerank on an RTX 4070. Measured on an RTX 4070, on this corpus: **296 ms** — ~4x
optimistic. (The 07-28 baseline attributed this claim to the README; it is `docs/setup.md:50`.) The
line now carries the measured figure, the CPU comparison, the corpus it was measured on, and a
pointer to the baseline — indicative, not a guarantee.

**Rejected.** Pinning `device="cuda"` anywhere in code — `sentence-transformers` already resolves
`cuda -> mps -> cpu`, and hardcoding it would break the CPU/CI path for zero gain.

**What it opens / carried.** Chunking-parameter sweeps and embedding-model swaps cost ~2 min per arm
instead of ~17, so those experiments become practical. **Carried:** the frozen-sidecar build
(`just build-sidecar`) expects a **CPU-synced** venv (KI-3), so a release build now needs
`uv sync --extra cpu --extra dev` first and a re-sync back afterwards.

---
## 2026-07-28 (5) — per-document cost estimates (mean/best/worst, named) + the reranker goes lazy: 16.1 s -> 11.7 s launch

**What changed.**

1. **`scripts/profile_stages.py` now samples per document** instead of profiling one median file:
   `--docs N` picks N documents **spread across the size distribution, always including both
   extremes** (so "best/worst" comes from the real tails, not from whichever file sorted first), and
   every stage reports **mean / best / worst with the document named**. `--extract` adds cold
   extraction per sampled document (off by default — it is the slowest per-document stage, so
   sampling it costs N times that). Embedding is measured over the document's **whole** chunk set,
   not a fixed 64-chunk batch, because batch effects are real at both ends of an 11x size range.
2. **`RAGPipeline.reranker` is now a lazily-loading property** (recommendation 2 from the baseline),
   with a setter so the existing `rag.reranker = fake` test idiom keeps working.

**Measured, 8 documents spanning 0.1 MB -> 32.4 MB** (full tables in
`tests/eval/baselines/stage_profile_2026-07-28.md` §5):

| Stage | Mean | Best | Worst |
|---|---:|---:|---:|
| read cached markdown | 0.4 ms | 0.2 ms | 0.8 ms |
| chunk parent-child | 7.5 ms | 1.0 ms | 19.3 ms |
| embed one document | **12.0 s** | 2.8 s (`rnn_regularization_zaremba_2014`) | **30.2 s** (`2203.07436v4`) |
| embed rate | 31.6 ms/chunk | 27.8 | 36.7 |
| extract (COLD) | **15.2 s** | 5.7 s | **36.7 s** |

**The estimator that falls out, and it is the useful part:** **~4.0 child chunks per 1,000
characters** (range 3.85-4.30 across a 14x span in chunk count) and **~32 ms/chunk** (±14%) ⇒
**embed ≈ 128 ms per 1,000 characters**, ±15%. So a document's ingest cost is predictable from its
extracted character count *before* embedding anything.

**And the trap in that:** **file size is the wrong predictor.** `31870-130793-1-PB.pdf` is 1.5 MB but
39k chars (165 chunks, 4.9 s) while `fpos-6-1305055.pdf` is 0.6 MB with 56k chars (216 chunks,
6.8 s) — MB tracks embedded images and scan quality, not text volume. Any progress bar or cost
estimate must be driven by extracted characters, which are available the moment extraction finishes.

**The lazy reranker, measured both sides.** Cold launch **16.1 s -> 11.7 s** (11.99/11.63/11.51 over
3 fresh processes): **~4.4 s saved**, better than the 3.7 s the component split predicted. **Stated
as the trade it is:** the launch is 4.4 s shorter and the *first* question of a session is ~3.7 s
longer (it absorbs the load). Right side of the trade for a desktop app — the readiness gate unblocks
the UI sooner and a first question is usually preceded by typing — but not a free win.

**Verified live, $0:** construction leaves `_reranker is None`; the first query logs
`loading_reranker`, returns 10 sources, and scores stay **sigmoid-bounded in [0, 1]** — which the
integrity layer depends on (`_sigmoid_activation_kwarg` exists precisely so a library upgrade cannot
silently switch to raw logits). Warm query top score 0.98. Full suite **1357 passed**, unchanged: the
setter is why every existing test that injects a fake reranker still works.

**A latent hole the change exposed.** Typing the property revealed that the `reranker.predict` call
site had **never been type-checked** — the eager attribute was inferred as untyped. mypy now checks
it, and reports that the sentence-transformers stub models only a *single* pair or a flat list, never
the batch-of-pairs form that is the documented way to score N candidates. The call keeps its shipped
runtime shape with a narrow, explained `# type: ignore[arg-type]`; the point worth keeping is that
**an opaque attribute silently disabled type checking on its every use**.

**Guard tests** (`tests/unit/test_pipeline_retrieval.py`): the property must not load on
construction, must load exactly once on first touch, and must cache; plus the setter must bypass the
real load entirely. Without these the 4.4 s reverts the first time someone "simplifies" the property
back to an attribute.

**Rejected alternatives.** Profiling one median document (hides an 11x spread and cannot answer
"worst case") · reporting only a mean (the worst document is the one that sets a progress bar's
honesty) · predicting cost from file size (measured wrong above) · a source-scanning test asserting
`__init__` contains no `CrossEncoder(` (brittle; the behavioural guards cover the regression that
matters) · loading the reranker in a background thread at startup (hides the cost rather than moving
it, and racing the first query for the GIL is worse than a visible one-off).

**What it opens (kept for tomorrow, in order).** (1) The remaining recommendations: install the CUDA
torch extra — every transformer figure here is CPU on a GPU box; persist the BM25 index (1.8 s now,
but it is the startup component that scales with the corpus). (2) **KI-30** — one shared `--doc`
resolver accepting id *or* hash prefix, with matching help and a guard test per runner; incremental
enrichment is unreachable until then, and `extract_citations` (54 s) is where the saving is. (3) The
KI-29 decision, now against a measured ~17 min re-embed.

**Gate.** ruff + format clean · `mypy src` 84 files · **pytest 1357 passed** · `docs_check --strict`
0/0.

---
## 2026-07-28 (4) — stage profile: where the time actually lives, so "must we re-embed?" stops being a guess

**What changed.** New `scripts/profile_stages.py` — times startup, query and ingest stage by stage on
the live corpus at **$0** (no provider constructed, no generation, the embedding measurement writes
nothing), plus `--sidecars`, which shells out to the real runners in dry-run mode to compare
**scoped vs whole-corpus** cost. Full table + method + the classification:
`tests/eval/baselines/stage_profile_2026-07-28.md`.

**Why, and why it is a script and not a one-off.** The ask was to separate what can run at runtime
from what is batch-only, so a re-scan/re-embed stops being routine. That is a measurement, and this
repo's discipline is that a measurement gets an instrument and a recorded baseline — the numbers move
with corpus size and torch device, so a re-run has to be one command.

**The five numbers that matter** (97 docs / 33,163 chunks, **CPU torch on a GPU box**):
- **Cold launch 16.1 s** (15.95/16.14/16.62), decomposed by cumulative subprocess measurement:
  imports **6.8 s** · embedder **2.4 s** · **reranker 3.7 s** · store read + BM25 build + chat model
  **~3.2 s**. Only the last row scales with the corpus.
- **Query 907 ms** for retrieve+rerank, of which the **cross-encoder is ~680 ms (75%)** — vector
  search 179 ms, BM25 28 ms, question embed 16 ms.
- **A re-scan is already free: 0.3 ms** to read a document's cached markdown (ingest dedupes by
  hash). What is expensive is **embedding: 30.5 ms/chunk**, i.e. **~13 s per document** and
  **~17 min for the whole corpus**.
- **Cold extraction is 24.2 s per document** — the real first-ingest cost (~40 min for 97 docs), and
  the reason the cache exists.
- **`extract_citations` is the slowest sidecar at 54.1 s** whole-corpus; everything else is 0.6-5.6 s.

**Two findings the profile turned up, both filed.**
- **KI-30:** `--doc` means three different things across four runners, and `extract_citations` /
  `extract_doc_metadata` filter on **`doc_hash`** while their help promises "doc_hash or id prefix" —
  so passing a real document id exits 1. Nothing in the enrichment layer is meaningfully incremental
  yet: `extract_keywords --doc` saves **4%** (3.97 s vs 4.14 s) because scoping the write does not
  scope the corpus TF-IDF (KI-18, now with a number), and `compute_doc_vectors --doc` filters only
  the report by its own admission.
- **KI-29's re-embed cost was a guess (~40 min) and is now measured (~17 min)** — corrected in place
  with a note that the measurement supersedes it. That is the number the KI-29 decision should be
  made against.

**A reconciliation, not a contradiction.** KI-18 quotes the epistemics projection at "~34 s @ 47
docs"; this profile measures **5.61 s @ 97 docs**. Both are right: the cost is
O(chunks x **vocabulary**), and ADR-018 cut the graph vocabulary 357 -> 13 (~27x). Exactly the trap
KI-19 warns about — do not cite these constants without the experiment attached.

**Honesty notes kept in the baseline.** (a) The first in-process embedder load of a session measured
**30.7 s** against 2.4 s marginal in a warm-cache subprocess — that is the OS file cache, so 16 s is
not the worst case and 30.7 s is not the load cost; both are stated rather than averaged away.
(b) Dry-run sidecar timings were **verified to do the full computation** (both `compute_epistemics`
and `extract_citations` print per-document results without `--apply`), so they are compute numbers
minus DB writes, not skipped work. (c) The full re-embed figure is extrapolated from a 64-chunk batch
and labelled as such.

**Rejected alternatives.** Micro-instrumenting each sidecar internally (timing the public entry
points answers the incrementality question and cannot drift from what a user runs) · a single
end-to-end "ingest one document" timing (it hides the extract/chunk/embed split, which is the whole
point) · averaging the cold-cache outlier into the median (it would have buried a real
first-launch-after-boot effect) · running the 17-minute re-embed to confirm the projection (the
per-chunk rate is the reusable number; the total is arithmetic).

**What it opens.** Ranked recommendations in the baseline: (1) install the CUDA torch extra — every
transformer number here is CPU on a GPU box, and it is the same two stages a GPU fixes; (2) defer the
eager reranker load in `RAGPipeline.__init__` for **3.7 s off every launch**; (3) persist the BM25
index — 1.8 s today, but it is the startup component that scales with the corpus and it is rebuilt
from scratch every launch; (4) fix KI-30 before optimizing the enrichment layer, since incremental
runs are unreachable until then.

**Gate.** ruff + format clean · `mypy src` 84 files · `docs_check --strict` 0/0.

---
## 2026-07-28 (3) — verification pass over the graph + library surfaces: one fix, one new known issue (KI-29)

**What changed.** `apps/desktop/src/lib/graph/ConceptGraph.svelte` — the "Appears in N documents"
rows printed the authors **twice**: `docTitle()` returned `docLabel(d)`, which appends `· first
author` for the breadcrumb/search case, while the row renders `docByline()` (`authors · year`)
immediately after it. Read live as *"A Primer on Motion Capture… · Alexander Mathis et al."* then
*"Alexander Mathis et al. · 2020"*. `docTitle` now returns `d.title || d.filename`; `docLabel` is no
longer imported there. Verified live over HMR: the row is now title / byline / mention count.
`library.ts` is untouched — `docLabel` is right for the breadcrumb, the folder picker and the
`aria-label` on a grid tile, none of which render a separate byline.

**Verified working, and cross-checked against the API rather than eyeballed** (97 docs / 33,163
chunks / ollama): concept graph 13 nodes / 19 edges / 6 communities, `stale: false`,
`n_concepts_in_db == n_concepts_in_skeleton`, Node-B relations present (`is combined with`, provenance
incl. `llm_relation`); **0 orphaned gaps** across all 18 (the KI-17 regression check) with every gap
label resolving; the ego view renders 8 circles / 14 lines / 6 labels for a degree-5 concept and its
presence panel matches the node's 10 `doc_ids`; library grid 97 tiles with the date buckets summing
to 97 and the folder count matching the API (Demo corpus 18); the reading view groups 92 parents /
478 children, summing exactly to `child_count`; taxonomy serves 236 fields / 23 roots / 344 unassigned
concepts (= 357 − 13 `graph_include`, as ADR-018 intends) with a sensible rollup; keyword families
357 with 0 blank or duplicate canonicals; connections serve related + external refs. 0 console errors,
no page-level overflow at 1280.

**Found, logged, NOT fixed — KI-29.** The reading view showed `<!-- page:N -->` as literal text. The
view is not the bug: it exists to show *what the retriever stored*. `build_parent_child_chunks`
(`ingest/chunking.py:160`) never applies `clean_chunk_text` to the child `page_content` or to
`parent_text`, while the baseline path does (`ingest/__init__.py:170`) — and parent-child is the
**default** retrieval mode, so the cleaned path is the one nothing uses at answer time. Measured, not
inferred: a real turn returned 10 sources of which **2 excerpts carried a marker**, and nothing strips
them in `pipeline.py`/`synthesis.py`/`chat_controller/` — so it is in the evidence block the LLM reads
and in the passage the user sees, and the child text was embedded with it.

**Why it is a decision and not a commit:** the correct fix (strip in the builder, matching the
baseline path) means the 33,163 stored children were embedded on different text than new ones would
be, so it needs a **full re-ingest** (~40 min, $0 local) to be coherent. Stripping at retrieval fixes
prompt + display with no re-ingest but leaves the store as-is. Both options, their costs and the
"do not fix it in the view" warning are in `.claude/KNOWN_ISSUES.md` KI-29. Same class as KI-26's
`_JOURNAL_HEADER`: a documented stripper that is never called on the path that matters.

**Gate.** svelte-check 186 files 0/0 · npm test 58/58 · docs_check --strict 0/0.

---
## 2026-07-28 (2) — detect-secrets blocked the release commit: two false positives + a baseline 18 findings stale

**What changed.** The `detect-secrets` pre-commit hook failed on the v0.3.0 commit. It was working
correctly — two things were wrong, and only one was mine:

1. **Two real false positives in the new tests.** `assert client._client.api_key == "sk-ant-from-app"`
   and `assert seen["init"]["api_key"] == "sk-ant-padded"` match the `Secret Keyword` detector
   (keyword `api_key` + a quoted literal). Marked with `# pragma: allowlist secret` — the documented
   mechanism, and self-documenting at the line, which a baseline entry is not. The other new
   fake-key literals do not match the detector's patterns and needed nothing.
2. **`.secrets.baseline` had been stale since 2026-07-04.** It carried **10** `Hex High Entropy
   String` findings for `tests/eval/corpus_manifest.yaml`; that file now has **28** — the demo corpus
   grew from 10 to ~30 papers (`0c777d8`) and nobody refreshed the baseline, because the hook scans
   **staged** files on a real commit and the manifest is rarely staged. So `pre-commit run
   --all-files` — the battery you would run before a release — had been failing on a file nobody had
   touched. Refreshed with `detect-secrets scan --baseline .secrets.baseline`; the diff is exactly
   +18 `Hex High Entropy String` rows in that one file (sha256 checksums of public papers), **no new
   files, no new finding types, plugins/filters unchanged** — verified by diffing the parsed JSON,
   not by reading the patch.

**Also learned:** the hook refuses to run against an **unstaged** baseline ("`git add
.secrets.baseline` to fix this") — so a baseline edit must be staged in the same breath, or the hook
reports nothing useful and looks broken.

**Deliberately NOT changed.** `pre-commit run --all-files` also rewrote three unrelated tracked
files, and I reverted all three: `data/anzsrc-2020-for-20210429.ttl` (28 trailing-whitespace strips
**inside `"""…"""@en` literals** — that is upstream CC-BY vocabulary data, and stripping a space
inside a literal changes the literal), plus two Android icon XMLs missing a final newline
(`icons/android/mipmap-anydpi-v26/ic_launcher.xml`, `icons/android/values/ic_launcher_background.xml`
— note `c8d17ce` fixed two *different* XMLs). None of them can block a commit that does not stage
them; sweeping them into a release commit would have been the wrong trade.

**Gate.** Whole hook battery on the staged set (what a real commit runs): ruff · ruff format · mypy ·
bandit · **detect-secrets** · hygiene — all **Passed**; `test_llm.py` 47 passed. 48 staged paths,
nothing unstaged.

---
## 2026-07-28 — first-run setup in the app (ADR-034) + v0.3.0 release prep: BYOK key entry, honest Ollama detection

**What changed.** The release-blocking half of "hand this to a testing user": setup no longer
requires editing a file or reading the source. Four defects found while building it, all real:

1. **A key entered at runtime could never have worked.** `pipeline.build_chat_model` read
   `ANTHROPIC_API_KEY` through a module-level `from config import …` binding — the *separate
   binding* trap this repo has now paid for three times. Every Anthropic call site
   (`AnthropicClient`, `build_chat_model`, the figure VLM client, `assert_provider_intent`) now
   resolves per construction via the new `credentials.resolve_key`. A `src/doc_assistant/CLAUDE.md`
   rule now says never to read the constant at a call site.
2. **`provider_available("ollama")` was unconditionally `True`** — "Ollama (local) needs nothing".
   Nothing checked whether a server was running or a model pulled, so the picker offered Ollama to
   a machine that had never installed it and the user found out from a transport error. New
   `llm.ollama_probe` answers reachability + installed models; `provider_available` keeps its
   local-state meaning (a credential is present) because a *stopped* local server must not
   invalidate a selection the user legitimately wants.
3. **The empty chat screen named only half the problem** ("No documents indexed yet") whether or not
   a provider could answer. It now renders the backend's outstanding-step list.
4. **Four planning ADRs (030–033) had files but no rows in `docs/decisions.md`.** Indexed.

**New modules.** `credentials.py` (one reader of key material: `<data home>/credentials.json`,
env-wins precedence, `key_source`/`key_hint` for display, never logged) · `readiness.py` (the
first-run picture: per-provider configured/reachable/models/action + a step list) ·
`apps/api/routers/setup.py` + `models/setup.py` (`GET /api/setup`, `POST`/`DELETE
/api/setup/anthropic-key` — the only route that accepts a secret) ·
`apps/desktop/.../settings/ProviderSetup.svelte` + `settings/setup.ts` (tested helpers) ·
`ChatController.refresh_chat_model()` (rebuild from current credentials without persisting a choice
the user did not make) · `library.count_documents()`.

**Why (the two decisions worth the words).** *Storage:* a data-home file, **not** the OS keychain
ADR-011 recorded as the north-star — that path carries an unvalidated PyInstaller bundling risk
("the exact class of freeze problem KI-9/KI-10 already cost this project"), and shipping it inside
the release whose purpose is a smooth first run inverts the cost/benefit. The cost is stated in the
UI and the README: plaintext on disk. *Precedence:* the environment wins over the stored key,
because the CLI reads the import-time constant and cannot see the store — the alternative is an app
and a CLI silently using different keys. The panel names the live source, so it can never show a key
it is not sending.

**Verification is free by construction.** A key is checked with `models.list()` — a metadata GET
that bills nothing — and the three outcomes are distinct: rejected (400, **nothing stored**),
unverifiable (stored + the reason: offline is not evidence a key is bad), ok. First-run setup can
never surprise a user with a charge (KI-4's discipline on a new path).

**Rejected alternatives.** App writes `.env` (co-authoring a file the user hand-edits; no `.env`
exists in a packaged install) · keychain via `keyring` (above) · session-only in-memory key (re-entry
every launch) · store-then-verify (a typo leaves a broken install looking configured) · probing
Anthropic on every settings read (rate limits, and an offline box waits on a timeout) · collapsing
"configured" and "reachable" into one flag (they have different fixes, and one would block a
selection the other should only warn about) · `len(list_documents())` for the count.

**Live verification, $0.** A second API instance on a temp data home (`DOC_DATA_DIR`, unreachable
`OLLAMA_HOST`) gave a genuine first-run install: `GET /api/setup` reported 0 chunks + "No Ollama
server answering", and the chat pane rendered "One step to go" unmocked. Against the real Anthropic
API a bogus key returned **400 and stored nothing**; with `ANTHROPIC_BASE_URL` pointed at a dead
port, a fake key stored with `verification: "unreachable"`, and `DELETE` removed it. On the real
install (97 docs / 33,163 chunks) the setup panel reported both providers ready — including the
9 models Ollama actually has — and one real `ollama/llama3.1:8b` turn returned 10 sources, top
reranker 0.9854 on `dpr_karpukhin_2020.pdf`, `is_local: true`. Temp data home + its ProgramData
Chroma namespace removed afterwards; the real data home never received a `credentials.json`.

**Two harness notes.** (a) With `visibilityState: hidden`, a Svelte state change lands one
round-trip later and a `transition:fly` panel stays off-screen — measure after a second call and
neutralize the transform, or you will file a working drawer as broken. (b) `load_dotenv()` walks up
from the **module file**, not the cwd, so the repo `.env` is found no matter where the process
starts; a keyless state cannot be produced by changing directory.

**Release prep (v0.3.0).** Version aligned across `pyproject.toml`, `package.json`,
`tauri.conf.json`, `Cargo.toml`/`Cargo.lock` (0.2.0/0.1.0 → **0.3.0**; nothing was ever tagged, so
this is the first release number that means anything). New `CHANGELOG.md` and `docs/QUICKSTART.md`;
README/setup/`.env.example` updated to say the app configures its own engine; `architecture.md`
gained the `setup` domain row.

**Gate.** ruff + format clean · `mypy src` 82 files · **pytest 1357 passed** (+42: 12 credentials,
12 readiness, 8 probes/key-resolution, 10 setup-route) · `svelte-check` 186 files 0/0 ·
`npm test` 58/58 (+8) · `docs_check --strict` 0/0 · live preview 0 console errors, no page-level
overflow at 1280.

**What it opens.** The keychain move is now a one-module change behind `credentials` (ADR-034 names
it as the reversal path). A second keyed provider is a row in `_KEYED_PROVIDERS`. Still owed for a
real installer release: a fresh PyInstaller sidecar + `tauri build` (the existing bundle is from
2026-06-24 and still named `doc_assistant_0.1.0`), and RG-012 Tier-2 on a clean box. `tests/conftest.py`
is this repo's first autouse fixture — it exists so a key saved in the app cannot change the suite's
verdict.

---
## 2026-07-27 — session-close conformance: the cpc gate caught five header errors and four warnings

**What changed.** Running `rungate.py keypoint session-close` before handoff (rather than at the
end of the *next* session) surfaced nine issues, all now fixed:

- **`status:` header vocabulary is closed.** `docs_check.HEADER_RE` accepts exactly
  `active|superseded|archived`. The four new ADRs carried `status: draft` and my
  `SESSION-archive-002.md` carried `status: archive` — none matched, so all five read as *missing*
  a header. The HTML comment tracks **document lifecycle**; an ADR's own `**Status:** proposed`
  line in the body is a different axis and stays untouched. Fixed to `active` / `archived`.
- **Living docs need their `updated:` bumped when edited** (rule 12): `architecture.md` said
  2026-07-26, `ROADMAP.md` said 2026-07-25, both committed today. Bumped — and the stale
  parenthetical on `architecture.md` ("repository layout moved here from the README") replaced with
  what actually changed.
- **Module `CLAUDE.md` files were over the 40-line cap** (41 and 42) after this pass added rules.
  Compressed rather than raising the cap; nothing dropped.

**Why it is worth a DEVLOG entry.** Two of these would have silently misled: a `status: draft`
header looks *more* careful than `active`, but the checker treats it as absent, so the file drops
out of conformance scanning entirely. And a `updated:` date that lags the last commit is exactly
the signal rule 12 exists to catch — a living doc that looks current but isn't.

**Verified.** `docs_check --strict` **0 errors, 0 warnings** (was 5/4) · `generate --check` OK ·
`ruff` clean over `src/`+`tests/`+`apps/` · `mypy src` 82 files · `svelte-check` 182 files 0/0 ·
`npm test` 50/50. Python suite unchanged since `598b570` (1,315 passed); nothing in this entry
touches runtime code.

---
## 2026-07-27 — Tracks 1–3 into the ROADMAP + three topbar/rail placement changes

**Roadmap.** ADR-030..033's twelve increments now sit in the PR table: `MM1–MM3` (document outline
layer → `doc_map` read model → `DocumentMap.svelte`), `T1–T5` (source-trust indicators; T5 parked
as the first network feature), `RP1–RP4` (generation presets). Flagged in the preamble that all four
ADRs are **stubs** — each says "needs `grill-me` before the Decision section is filled" — so those
rows are scope, not contract.

**One collision worth noting:** the plan numbers its Reports track `R1–R4`, but `R1–R4` are already
taken in the ROADMAP by the 2026-07 remediation rows (ingest hygiene, concept presence, keyword
termhood, skeleton provenance). Renumbered **`RP1–RP4`** here, with the mapping stated in the
preamble — the ROADMAP is a shared ID namespace and silently reusing an id would have made two
different things look like one.

**UI placement (user request).**
1. **Brand → right.** The `provenote` mark + wordmark move out of the left cluster into the right
   one, beside Settings. No CSS change needed: `.brand` is `flex: none` and position-agnostic, and
   the 780px rule that drops the wordmark (keeping the mark) still applies.
2. **Search → left.** The magnifier moves from the right cluster to between the sidebar toggle and
   back/forward. It opens a *navigation* overlay, so it now sits with the other navigation
   affordances rather than next to Settings.
3. **Taxonomy → pinned rail footer.** On the Graph rail it was the first row of the scrolling
   concept index; it is a destination, not an index entry, so it moved out of the `<nav>` into a
   `.railfoot` sibling. `.sidebar` is already a flex column with the list at `flex: 1;
   overflow-y: auto`, so a `flex: none` sibling pins to the bottom with no positioning hacks.

**Verified.** `svelte-check` 182 files 0/0, `npm test` 50/50, 0 console errors. Live at 1280px the
toolbar reads left→right **Menu · Collapse · Search · Back · Forward · tabs … brand · Settings**,
0 overflow; 375px dark keeps the mark, drops the wordmark, 0 overflow.

The footer's "unscrollable" claim was **proved, not assumed** — and the first attempt was a vacuous
test: `.graphrail` reports `scrollHeight == clientHeight` because `GraphIndex` renders its own
inner `.clist` scroller. Scrolling *that* element by 395px left the footer at exactly the same
`top` (348 → 348). A pinned-element test that never actually scrolled anything would have passed
either way.

---
## 2026-07-27 — `chat_controller.py` 1,423 → a package, and the re-export trap billing 66 tests

**What changed.** Five modules + a barrel: `session` (63, ADR-3 caller-owned state) · `views` (121,
the pure render payload `apps/api/models/chat.py` mirrors) · `events` (39, the `TurnEvent` union) ·
`helpers` (493, pure formatters + turn-knob resolution) · `controller` (770, `ChatController`).
Dependency direction is strictly **session/views → events/helpers → controller**.

**The cycles in the first dependency scan were fake.** A naive name-reference scan reported
`session→controller`, `helpers→controller`, `views→session`, i.e. no valid layering. Every one
turned out to be a **docstring mention** — `` ``ChatController`` `` in a class doc, a Sphinx
`:meth:` cross-reference. Only `events→views` (`result: TurnResult`) was a real code edge. Checking
each candidate line by hand instead of trusting the count is what made the split possible; the
regex would have said "don't do this".

**Then the re-export trap, at scale.** The `library` split predicted it and cost 2 test lines; here
it broke **66 tests**, because `test_chat_controller.py`/`test_turn_parity.py`/`test_retrieval_scope.py`
patch *module-level imported names* on `chat_controller` — `is_library_query` (39×),
`current_graph_version`, `record_answer`, `SYNTHESIS_MODE`, `adjudicate_claim`, `RAGPipeline`… Once
those bindings live in `controller.py`/`helpers.py`, `setattr(chat_controller, …)` writes to the
package and the real caller never sees it.

The fix needed a distinction worth writing down: **rebinding a name** must target the owning module
(`chat_controller.controller.is_library_query`, `chat_controller.helpers.SYNTHESIS_MODE` — note the
config constants belong to `helpers`, since `_resolve_turn_knobs` consumes them), whereas
**setting an attribute on a shared module object** (`chat_controller.app_settings.SETTINGS_PATH`)
works through any binding, so `app_settings` is simply re-exported. 57 call sites repointed.

**Three misses the mechanical generator made**, each caught by a different gate — worth knowing the
pattern: (1) `log` used without `structlog` in the body, so no logger was bound — *ruff*;
(2) `TurnEvent = Token | Step | Result` is a **module-level type alias**, invisible to a `def`/`class`
scanner, so `controller` lost its import and the barrel lost the export — *ruff*; (3) test patch
targets — *pytest*, 66 of them. Static analysis of a file tells you what is *defined*; it does not
tell you what the rest of the repo *reaches into*.

**Verified.** 34/34 declarations present, **zero body diffs**, and both module-level assignments
(`TurnEvent`, `_MARKER_LABELS`) confirmed present. `ruff` + `ruff format` clean over `src/` and
`tests/`, `mypy src` clean (**82 files**), full suite **1,315 passed** — the exact pre-split count.

---
## 2026-07-27 — `library.py` 1,528 → a `library/` package (the `src/` twin of the apps/ pass)

**What changed.** `src/doc_assistant/library.py` becomes a package of 8 sub-domain modules +
a re-exporting `__init__`, cut along its own banner sections and **named to match
`apps/api/routers/library/`**: `models` (67) · `documents` (319) · `pins` (195) · `folders` (199) ·
`keywords` (273) · `chunks` (135) · `citations` (174) · `similarity` (133).

**Scoped before cutting** (the discipline that changed the plan in step 4). The section dependency
graph is **acyclic with five leaves** — only `similarity→citations` (3), `query→models` (2),
`pins→{models,documents}`, `folders→{documents,keywords}` (1 each). 27 import sites across
apps/scripts/tests, all preserved by the barrel.

**The hazard scoping predicted, verified empirically.** `tests/integration/test_document_meta.py`
did `import doc_assistant.library as lib; monkeypatch.setattr(lib, "_reveal_in_file_manager", …)`.
After the split that name is a **re-exported binding on the package**, not the one
`reveal_document_source` resolves. Proven at the REPL, not assumed:

    patching the package changes what the caller resolves?  False
    patching the owning module changes it?                  True

So the patch would have silently missed and the test would have run the **real** function — opening
a file manager during the suite. Both sites now patch `doc_assistant.library.documents`, and the
rule is in `src/doc_assistant/CLAUDE.md`: **patch the module that owns a helper, never a package
that re-exports it.** It is the Python twin of the "patch it on its router module, not `main`" rule
already in `apps/api/CLAUDE.md` — a barrel buys compatibility and hides binding identity.

**Verified.** Declaration-level equivalence proved the same way as the `types.ts`/`api.ts` splits
(which caught two silent truncations): **67/67 declarations present**, bodies byte-identical after
normalising comments/whitespace, with the only two diffs being ruff's `UP037` unquoting
`list["SimilarDoc"]` / `list["FamilyProposal"]` — safe because every module carries
`from __future__ import annotations`. `ruff` clean, `ruff format` clean, **`mypy src` clean (77
files, up from 69)**, and the **full suite green: 1,315 passed**.

**Generation note.** The modules were generated mechanically (bodies + per-module import resolution
computed from actual name usage), which got the imports ~90% right and left exactly two classes of
miss that ruff caught: `log` used without `structlog` appearing in the body, and a `TYPE_CHECKING`
guard whose import lived in the shared header. Worth remembering: deriving imports from usage
misses names introduced by *convention* rather than by reference.

---
## 2026-07-26 — step 5 phase 2 complete: `ChatPane` + `chat.svelte.ts`; App.svelte 2,725 → 1,245

**What changed.** The last pane, and the one that blocked the rest: `chat/chat.svelte.ts` (state +
DOM mechanics) and `chat/ChatPane.svelte` (130 markup + 243 CSS). **`App.svelte` 1,659 → 1,245** —
901 script / ~75 markup / ~270 style. Across the whole pass: **2,725 → 1,245, −54%**.

**Why chat needed a module where Library did not.** `convoEl` and `taEl` are *bound DOM refs*, and
both sides need them: the pane binds them, but App's `newConversation`/`resumeConversation` still
call `chat.taEl?.focus()` after the pane mounted them. Props cannot express that. Putting them in
the rune module — `bind:this={chat.convoEl}` — is what made the extraction possible at all.

The autoscroll `$effect` moved with them as `useChatAutoscroll(viewing)`, taking `viewing` as a
**getter thunk** rather than a value: it is conversation-view state App owns, and the effect must
re-run when it changes (opening a past chat scrolls to the bottom too). `pinned` and `nextId`
stayed **non-reactive** module locals behind setters — `pinned` is written on every scroll event,
so making it `$state` would have made the autoscroll effect re-run on scroll, i.e. a feedback loop.

**Still in App, deliberately:** `send` (needs the folder scope + refreshes conversations),
`doCompare`, `doExport`, `newConversation`, `resumeConversation`, and the `activeSource` derivation
— every one spans chat × conversations × folders.

**Three shorthand traps, all caught by the compiler.** Rewriting `overrides` → `chat.overrides`
turned `bind:overrides` into `bind:chat.overrides` (invalid). This is the third instance of the
same class this pass — after `{conversations}` and `{mode}` — so it is now a known cost of
renaming a variable that appears in Svelte shorthand: **`{x}`, `x,` and `bind:x` all need
hand-editing.** The regex also had to exclude `<` from its lookbehind, or `<input`/`<textarea`
would have become `<chat.input`.

**One deliberate reversal.** The scope `<select>` first became `value=… onchange=…` because a
`const` prop cannot be bound — but `<option value={null}>` makes string round-tripping through the
DOM subtly wrong, and this is the ADR-025 F2 retrieval scope, where a silent mistake means answers
scoped to the wrong folder. Switched to `$bindable()` so `bind:value` is preserved exactly as
before. Faithful beats clever on the answer path.

**Verified.** `svelte-check` **182 files, 0 errors, 0 warnings**; `npm test` 50/50; 0 console
errors. Live at 1280px, every boundary crossing exercised: a sample chip fills the composer (App
writes `chat.input` → module → the pane's bound textarea, Send enables); **autogrow 62 → 160px**
through the module's DOM ref, and back to 60 on clear; the scope `$bindable` round-trips with its
`.scopepick.scoped` class; opening a past conversation renders the read-only replay **and
autoscrolls to the bottom** — the effect firing across the module boundary, the single riskiest
part of this change; a citation click still opens the source panel; Back-to-current restores the
composer. 375px dark: 0 overflow, 0 offending elements.

**Step 5 is done, and so is the plan.** All six steps complete. `App.svelte` is now a shell: the
cross-domain orchestration (nav history, readiness gate, `selectMode`, the chat-scope guard, the
conversation/chat lifecycle) plus overlay wiring — which is what an app shell should be.

---
## 2026-07-26 — step 5 phase 2 (second slice): `LibraryPane` out, and why it takes 29 props

**What changed.** The Library workspace leaves `App.svelte` with its styles:
`library/LibraryPane.svelte` (205 lines of markup + 251 of CSS). **App.svelte 2,081 → 1,659** —
948 script / ~140 markup / ~570 style. Since the pass started it is **2,725 → 1,659 (−39%)**.

**A finding that removed work:** the *graph* pane needed no extraction at all. Its branch is 19
lines and is already a single `<ConceptGraph …/>` invocation — the component *is* the pane. Only
library and chat were ever real.

**Why this one takes an explicit 29-prop contract, unlike `Topbar`'s 8.** `Topbar` could import
`shell` because that state is genuinely shell-owned and its module is a leaf. The Library pane's
derived pipeline (`facetList`, `keywordsOf`, `visibleDocs`) is not: it is computed from documents ×
**keyword families** × folders, and family state is the domain step 4 deliberately left in App
because a family write also re-points the live facet selection (PR-2.5 D5) — three domains in one
function. Extracting it just to shorten a prop list would break the boundary this whole pass has
been protecting.

So the pane is **presentational with a written-out dependency surface**. For a leaf view that is
arguably the better shape anyway: the contract is 29 lines at the top of the file stating exactly
what the Library needs, instead of an implicit reach into module state. It only reads `prefs.svelte`
(its own view/sort preferences) and the pure helpers from `library.ts` directly.

**Method note.** Rather than hand-derive the contract, the markup was extracted mechanically and
`svelte-check` was used as the oracle: every unbound name it reported became a prop or a callback.
That surfaced six I would have missed (`selectCollection`, `folderNames`, `openManageFolders`,
`onSetDocId`, `onClearSelection`, plus the `KeywordFacet` type living in `library.ts`, not
`core/types`). It also caught one prop I had invented and never used — `onSetQuery`, dropped, since
an unused entry in a contract that exists to document dependencies is worse than none.

**Verified.** `svelte-check` **180 files, 0 errors, 0 warnings** (0 warnings again the real signal:
251 lines of CSS moved scope and nothing was orphaned); `npm test` 50/50; 0 console errors. Live at
1280px: CSS intact after the move (`.library` flex:1, `.libnav`, `.crumb` 12.3px, `.viewtoggle`),
97 tiles; and every callback exercised — drill into a document and Back (97 tiles restored),
select mode in/out (Add to folder · Clear · Done, tile toggling), the keyword-filter overlay,
grid/list round-tripping through localStorage, and collection select (Demo corpus → 18 tiles with
a "Library › Demo corpus" breadcrumb → All documents → 97). 375px dark: 0 overflow, **0 offending
elements** measured against the viewport.

**Remaining.** Only the chat pane (~132 lines of markup). It is the awkward one: `convoEl` and
`taEl` are bound DOM refs that App's scroll-pinning `$effect`, `autogrow`, `resetComposer` and
`taEl?.focus()` all reach for, so extracting it means moving chat state into a rune module first —
the phase-1 pattern again, applied to the domain step 4 named as most coupled.

---
## 2026-07-26 — step 5 phase 2 (first slice): `Topbar` + `StatusBar` out of App.svelte

**What changed.** The two pieces of pure chrome leave `App.svelte` **with their styles**:
`shell/Topbar.svelte` (122 lines of markup + 188 of CSS) and `shell/StatusBar.svelte` (20 + 41).
`App.svelte` **2,439 → 2,081** lines; its `<style>` block drops **229 lines**.

**This is what phase 1 bought.** `Topbar` reads `shell` and `sidebarPrefs` straight from the leaf
rune modules, so it needs only **8 props** — the nav-history cursor (`canBack`/`canForward` +
their two callbacks) and four things App genuinely owns (`onSelectMode`, which lazy-loads four
domains; `onOpenSearch`; `exportDisabled`, which depends on live turn state; `onExport`). Before
phase 1 the same component would have taken about twenty. `StatusBar` takes **zero** props: it
renders `shell.status`/`shell.health`, which App's readiness gate writes.

**Verified.** `svelte-check` **179 files, 0 errors, 0 warnings** — the 0 *warnings* matters here,
since Svelte reports unused CSS selectors, so an orphaned rule left behind in App would have shown
up. `npm test` 50/50. Live at 1280px, 0 console errors: styles survived the scope move (topbar 47px
with its 1px rule, status dot the green 7px circle, corpus line intact); all 8 props exercised —
mode switch, **nav history round-trips across the new component boundary** (Library → Back → Chat
→ Forward → Library, with Forward correctly re-enabling), Export present-but-disabled with no
turns, search overlay opens. At 375px the moved media queries still fire exactly as their comment
promises: tab labels drop, then the wordmark, the mark stays, mobile/desktop buttons swap, 0
horizontal overflow.

**A false alarm worth recording, because it will recur.** First measurement said the topbar was
**21px wide with 363px of horizontal overflow** — alarming, and it looked like the CSS scope move
had broken the flex column. It had not: `innerWidth` was **0**. The Browser pane was not displaying
(`visibilityState: 'hidden'`), so the viewport had collapsed and every width was an intrinsic
min-content measurement. `resize_window` with a *preset* did not re-establish it; passing explicit
`width`/`height` did, and then `.app` and all three children measured 1280 with 0 overflow. Two
sessions running now, the harness has produced a convincing-looking layout "bug" that was purely a
measurement artifact (see the phase-1 entry's stalled CSS transition). **Check `innerWidth` and
`visibilityState` before believing a geometry regression.**

**Remaining for phase 2.** `App.svelte` is 2,081 lines: 949 script / ~560 markup / ~570 style. The
three mode panes (library / graph / chat) are the rest, and they are harder than the chrome — they
read genuinely App-owned domain state (documents, folders, keywords, turns, citations), so each
wants either its domain's rune module or a deliberate prop contract. Chrome first was the right
order precisely because it needed no such decision.

---
## 2026-07-26 — step 5 phase 1: `shell/shell.svelte.ts`, the leaf module the pane split needs

**What changed.** The last of `App.svelte`'s non-domain state moves to one rune module: `mode`,
`sidebarOpen`, `appMenuOpen`, `showShortcuts`, `showAbout`, `showSettings`, `searchOpen`,
`searchQuery`, `health`, `status`. Script 955 → **947**, 68 call sites rewritten.

**Why this is phase 1 and not the pane split itself.** Measuring first (again) changed the plan:
the markup region references **40+ App identifiers**. Extracting `Topbar`/`StatusBar`/`LibraryPane`
as prop-taking components would thread twenty-odd props into each — *less* reviewable than the
2,400-line file they came from, which defeats the point. `shell.svelte.ts` is deliberately a
**leaf** (it imports nothing from any sibling state module), so in phase 2 the panes can `import
{ shell }` instead. That is the whole reason this is a separate, boring commit.

**Still in App.svelte, unchanged:** the nav-history `$effect` (reads `shell.mode` *and* library
collection/docId *and* graph selection), the readiness gate (writes `shell.health`/`shell.status`
but also kicks conversations + folders), `selectMode` (lazy-loads four domains) and the chat-scope
guard. Orchestration stays where it is visible.

**Why this one was not a blanket find-and-replace.** `mode`, `status` and `health` collide with
things that must not change: `class="tb-mode"`, `class="status-dot"`, `class="statusbar"`,
`role="status"`, the `NavEntry` type's own `mode:` field, and `/api/health` in comments — a
word-boundary regex matches `status-dot` (the `-` is a boundary) and would have written
`class="shell.status-dot"`. So those three were rewritten **line-by-line against an explicit,
asserted list**; only the seven unambiguous identifiers got a global regex. Three shapes needed
hand-editing: the `{mode}` shorthand prop → `mode={shell.mode}`, the `mode,` object shorthand in
the nav entry → `mode: shell.mode,`, and `bind:query={searchQuery}`.

**Verified.** `svelte-check` **177 files, 0/0**; `npm test` **50/50**; the `<style>` block has zero
diff lines. Live, 0 console errors: the readiness gate drives the status bar through its real
states (`connecting` + `wait` dot → `ok` dot + "33,163 chunks · ollama/llama3.1:8b · bge-base");
mode tabs, app menu (3 items), shortcuts modal, Settings drawer, and Cmd-K search all round-trip
(typing "dense" filters, Cmd-K closes); 375px dark at 0 horizontal overflow.

**A measurement trap worth recording.** The mobile drawer *looked* broken — `.open` applied but
`transform` stayed at `translateX(-100%)`. It was not a regression: `document.visibilityState` is
**`hidden`** in this harness (the pane does not composite, which is also why `screenshot` times
out), so CSS transitions never advance — `getAnimations()` showed the animation stuck at
`currentTime: 0`. Setting `transition: none` and re-measuring gave `translateX(0)`: correct. When a
CSS-driven state looks stuck here, neutralise the transition before calling it a bug.

---
## 2026-07-26 — step 6: `core/types.ts` + `core/api.ts` split by domain, mirroring `apps/api/models/`

**What changed.** The two wire-boundary files became packages named for the same domains as
`apps/api/models/` and `apps/api/routers/`: `core/types/` (12 modules from 513 lines) and
`core/api/` (12 clients + `_base.ts` from 631). Each keeps an `index.ts` barrel, so every existing
`from '../core/types'` / `'../core/api'` still resolves — no consumer churn — while a wire change
is now a **one-file diff** you can line up against `models/<domain>.py`.

Two shared pieces were hoisted into `api/_base.ts`: `API_BASE` and `errorDetail` (the FastAPI
`detail` unwrapper, which handles both the plain-string and the structured-offenders shapes).
Cross-domain type coupling turned out to be a single edge — `conversations` imports `TurnScope`
from `chat` — which is a good sign the domain lines are real.

**Rejected.** Pointing all ~25 component imports at the domain modules in the same pass. The
barrel gives the whole reviewability benefit (the split is what makes the diff one-file); rewriting
every import site is churn that would bury it. The barrels say "prefer the domain module" for new
code instead.

**Two silent-truncation bugs caught by asserting, not by reading.** Splitting by parsing is exactly
where quiet data loss hides, so both files were split mechanically and then **proved equivalent**:
every declaration re-parsed from the originals and compared body-by-body after normalising
comments and whitespace. That caught (1) `export type GapKind` losing its entire 9-member union —
a brace-matching heuristic that returned `i+1` for a multi-line type alias — and (2) the private
`errorDetail` helper being swallowed into one module, breaking six others. The second surfaced as
24 `svelte-check` errors; **the first would have compiled if the next line had not happened to be
another `export`.** Final check: types 51/51 and api 54/54 declarations present and byte-identical,
modulo the two intended `export` additions.

**Verified.** `svelte-check` **176 files, 0 errors, 0 warnings**; `npm test` **50/50**. Live: after
clearing the Vite cache (a deleted module lingers in the dev server's graph — same restart dance as
the step-3 asset paths), all three modes render, 0 horizontal overflow. Proved the split is what
actually loads rather than trusting an absent error: `performance.getEntriesByType('resource')`
shows **all 14 `core/api/*` modules fetched**, the old flat `core/api.ts` **not fetched**, and zero
failed requests. Worth noting the browser console buffer is **cumulative across navigations** in
this harness — 16 stale HMR errors from before the restart persisted through a hard reload and
looked like a live failure; the server log and the resource timings were the honest signals.

---
## 2026-07-26 — App.svelte step 4: the domains that actually decouple → `.svelte.ts` rune modules

**What changed.** Five per-domain rune modules pulled out of `App.svelte`'s script, which drops
**1,233 → 955 lines**: `graph/graph.svelte.ts` (99) · `library/taxonomy.svelte.ts` (118) ·
`library/prefs.svelte.ts` (60) · `shell/prefs.svelte.ts` (67) · `chat/conversations.svelte.ts` (48).

**Why it is a *partial* step 4, on purpose.** The plan said "split the script into 8 per-domain
state modules". Inventorying the coupling first (the prerequisite this plan flagged) showed that
would be a mistake. Of the 5 `$effect` blocks, **3 are irreducibly cross-domain**: the nav-history
observer reads mode + library collection + library docId + graph selection; the readiness gate
writes health/status and kicks conversations + folders; the chat-scope guard reads folders and
writes chat state. `selectMode` orchestrates four domains and `refreshFamilies` writes the facet
selection as well as the family list.

Forcing those into modules buys nothing and costs the thing the whole pass is for: coupling that
is currently **visible in one file** would become **invisible**, spread across an implicit import
graph — and two of the three would need import cycles to express at all. So the rule applied was:
extract a domain only when it is genuinely self-contained; leave cross-domain orchestration in
`App.svelte`, which is exactly what an app shell is for.

**Not extracted, and why:** *keyword families* — every mutation re-runs `remapSelection` over the
facet selection against the document list (PR-2.5 D5: a family write changes what a facet *unit*
is, so a live selection must be re-pointed or the grid silently empties behind a chip that still
looks selectable). That is three domains in one function and it stays put. *conversations* was
extracted only as far as the list + pin/archive/rename; `openConversation`/`resumeConversation`/
soft-delete all write live chat state, so they stayed.

**Svelte 5 shape.** State is one exported `$state` **object** per module, not separate `let`s — an
imported binding cannot be reassigned across a module boundary, so `graph.selectedId = x` is the
only form that works. `$effect` cannot run at module top level (no effect context), so the one
intra-domain effect is exported as `useGraphHygiene()` and called from App during init.
`.svelte.ts` modules need the compiler and **cannot run under `node:test`** — so the pure, tested
modules (`library.ts`, `search.ts`, `gaps.ts`, `taxonomy.ts`) were left untouched beside them. The
extension is the marker: `taxonomy.ts` is pure and tested, `taxonomy.svelte.ts` is reactive state.

**Verified.** `svelte-check` **151 files, 0 errors, 0 warnings**; `npm test` **50/50**. Live, with
0 console errors: cross-module reactivity confirmed (clicking a GraphIndex concept updates
`graph.selectedId` and the ego panel re-renders); sidebar collapse and library view/sort each
toggle, apply, and round-trip through localStorage (all restored to their prior values); the
taxonomy modal opens on the full ANZSRC forest (357 concepts · 97 documents · 236 fields) and its
drill-in populates **both** pickers — 14 concepts read from `graph.data` across the module
boundary, 98 documents from App's lazy-load wrapper. 1280px light and the 375px path both at
0 horizontal overflow.

**What it opens.** Step 5 (markup + style → pane components) is now the bigger remaining win:
`App.svelte` is still 2,447 lines, of which **1,492 are markup + style**, not script.

---
## 2026-07-26 — `GapList.svelte` was a **binary file** to git: raw NUL byte → the `\0` escape

**What changed.** One byte. `apps/desktop/src/lib/graph/GapList.svelte:25` builds the `busy`-Set
dedup key as `` `${it.concept_id}<NUL>${it.kind}` `` — and the separator was a **literal 0x00 byte**
in the source, not an escape. `file` reported `application/octet-stream`, and **git rendered every
change to the file as `Binary files … differ`** while grep skipped its contents entirely. Replaced
the raw byte with the two-character escape `\0`.

**Why it matters.** A file git cannot diff is a file that cannot be reviewed — it silently opts out
of code review, and grep-based auditing never sees it. Found during the `apps/` reviewability pass
(entry below) and deliberately held back from that commit: folding a content fix into a move-only
diff is the exact thing that makes refactors unreviewable.

**Runtime-identical, proven not assumed.** In a JS/TS template literal `\0` *is* the NUL escape:
``node -e`` on `` `${a}\0${b}` `` vs `a + String.fromCharCode(0) + b` → `true`, codepoints
`120,0,121`. The composite key is byte-for-byte what it was. The one hazard — `\0` followed by a
digit is a legacy octal escape and a SyntaxError in strict mode — does not apply: the next
character is `$`. Asserted in the edit script rather than eyeballed.

**Rejected.** A distinct printable separator (``, `::`) — a behaviour change to a dedup key
for no benefit, and it would need its own reasoning about collision-safety against UUIDs and gap
kinds. The point was to make the file text, not to redesign the key.

**Gotcha worth remembering.** The first byte-level edit script **silently did nothing**: passed
through a `<<'PY'` heredoc, `b"\\0"` reached Python as `b"\x00"` (len 1), so the replace was
NUL→NUL. It only surfaced because the script asserted `len(new) == len(d) + 1`. Without that
assert the run would have reported success on an unchanged file. When editing bytes through a
shell heredoc, build the literal from explicit values (`bytes([0x5C, 0x30])`) and assert the size
delta — do not trust backslashes to survive the trip.

**Verified.** `file -bi` → `text/html; charset=utf-8` (was `application/octet-stream`); the staged
blob has **0 NUL bytes**; a simulated follow-up edit now renders as a **readable line-level diff**
instead of "Binary files differ". `svelte-check` 146 files 0/0 · `npm test` 50/50. Live: Graph →
Gaps renders 18 open gaps; **Promote round-trips 200 OK and changed exactly 1 of 18** (the busy-Set
key still targets a single item — no collision), then **all 18 were reset to `surfaced`, restoring
the pre-test baseline exactly**. 0 console errors, `$0` (ollama/llama3.1:8b).

---
## 2026-07-26 — `apps/` reviewability pass: one domain axis across both shells (steps 0–3 of 6)

**What changed.** Three structural moves, no behaviour change anywhere.

1. **`apps/api/models.py` (1,165 lines) → `apps/api/models/`** — 12 domain modules + `_common`,
   cut along the file's *own* pre-existing banner sections and named to match `routers/`:
   `chat` · `compare` · `conversations` · `library` · `connections` · `folders` · `keywords` ·
   `sources` · `concepts` · `taxonomy` · `settings`. The 7 routers now import from their domain
   module, so the import line names the domain; `__init__` re-exports flat for back-compat (the
   one remaining consumer is `tests/unit/test_api_models.py`).
2. **`apps/api/routers/library.py` (301 lines) → `routers/library/`** — it was the only router
   bundling three sub-domains (documents 7 routes, folders 6, keyword-families 7). Now one module
   each, composed in `__init__`, so `main.create_app` is untouched. The three path prefixes are
   disjoint, so include order carries no matching meaning; order *within* each module is
   load-bearing and was preserved verbatim.
3. **`apps/desktop/src/lib/` (43 flat files) → 6 domain folders** — `chat/` · `library/` ·
   `graph/` · `settings/` · `shell/` · `core/`. Pure `git mv` + import rewrites, no renames: git
   reports every file as `R`, so the diff reads as moves. Deliberately *not* combined with
   dropping the `Library` prefix — a move+rename diff is exactly what makes a refactor
   unreviewable, which would defeat the point.

**Why.** `apps/api` had already solved this with the APIRouter split and `apps/desktop` never
got it: the API was 8 domain routers with one 1,165-line outlier, the frontend was a flat bag of
30 components + 8 logic modules + 4 tests + the client + the wire types. One rule now holds on
both sides — *one domain per module, and the domain word is the same on both sides of the wire* —
so reviewing a feature end to end is reading one row of the table now in
`docs/architecture.md` § `apps/` — the domain spine.

**Two naming traps the map exists to prevent**, both found by reading rather than assuming:
`Sources.svelte` is **selective ingestion** (files on disk, matching `routers/sources.py`), *not*
the citation sources of an answer — those are `chat/Source*.svelte` / `models/chat.py`. And
`/api/library/*` is three sub-domains, not one.

**Rejected.** (a) Splitting `models.py` but leaving routers importing the flat `__init__` — keeps
the diff smaller but throws away the main benefit, since the import line no longer says which
domain a route serves. (b) Renaming `LibraryGrid.svelte` → `library/Grid.svelte` in the same
pass — correct end state, wrong moment (see above); left for a follow-up. (c) Rewriting the
`models.py` pointers in ADR-010 / ROADMAP TX2b / the feature specs — those are dated historical
records and stay as written; only the *live* pointers were repointed (`core/types.ts`'s 14
`Mirrors …::Payload` comments, `vite.config.ts`).

**Verified.** `ruff` clean; `mypy src` clean (69 files); **161 API tests pass** (every test that
imports `apps.api`). Frontend `svelte-check` **146 files, 0 errors, 0 warnings** and `npm test`
**50/50** — both identical to the pre-move baseline. Live preview: all three modes render
(chat shell + history, Library grid, Graph index/gaps), 0 console errors, About dialog opens with
`app-mark.png` loaded and both vendored fonts `loaded`; 375px dark = **0 horizontal overflow**.

**What the live preview caught that the gates did not.** `svelte-check` and `npm test` both passed
while the app was *broken*: `AboutDialog.svelte` imports `../assets/brand/app-mark.png` and
`fonts.css` has four `url('../assets/fonts/…')` — **asset** paths, invisible to a TS type-check
and to node:test, and Vite failed to resolve them. This is the same class as the
`function f(x?)` footgun already in `apps/desktop/CLAUDE.md`: the type gate is not the run gate.
A path-only refactor still needs the preview.

**What it opens.** Steps 4–6 of the plan, each its own session: `App.svelte` is still **2,725
lines** (70 `$state`, 19 `$derived`, 5 `$effect`, 83 functions, 22 child components) — split its
script into per-domain Svelte 5 `.svelte.ts` rune modules (the 5 `$effect` blocks are the
cross-domain coupling points and must be inventoried first), then its markup+style into pane
components, then mirror the `models/` split into `core/types.ts` + `core/api.ts`.
Note `.svelte.ts` modules cannot run under `node:test`, so that split must not absorb any of the
pure tested logic (`library.ts`, `search.ts`, `gaps.ts`, `taxonomy.ts`) — that line stays hard.

**Found in passing, not fixed** (own change, deliberately not folded into a move-only diff):
`graph/GapList.svelte` line 25 contains a **literal NUL byte** as a composite-key separator
(`` `${it.concept_id}\x00${it.kind}` ``). Git and grep classify the file as **binary**, so it
never renders a readable diff — an actively anti-reviewable file. Replacing the raw byte with the
escape `\0` is runtime-identical and makes it text.

---
## 2026-07-26 — KI-26 residual: a leading dash rode into the stored title (`- PASSAGE RE RANKING WITH BERT`)

**What changed.** `_clean_markdown` now strips a leading run of bullet/dash glyphs
(`-‐-―•·‣▪●`) from every title candidate. `reranking_bert_nogueira_2019.pdf` extracts as
`## - PASSAGE RE RANKING WITH BERT` — the hyphen of "RE-RANKING" landing at the front of the
heading — and that dash was stored, so the document sorted **first** in the library grid and was the
opening card of the freshly-recorded README GIF. Fixed at `_clean_markdown` rather than in the
title picker because every candidate path (headings, bold lines, the citation-block fallback,
`_is_skippable_heading`) already runs through it.

**Verified the same way KI-26 was.** Re-diffed the extractor against all 97 stored titles rather
than spot-checking the one: **2 differ, 1 changes, 0 lost.** The fix is
`- PASSAGE RE RANKING WITH BERT` → `PASSAGE RE RANKING WITH BERT`; the other difference is the
already-known stale `Disclaimer` row on `FPG_007_CriticalCareinNeurology_2012.pdf`, where the
extractor correctly returns `None` and the runner declines to overwrite a value with nothing. So
the blast radius is exactly one document, which is what a punctuation strip should be.

**Applied and re-derived.** `extract_doc_metadata --doc 06ea458a0072 --apply --force` (plain
`--apply` reported `Total field updates: 0` — the runner only fills *empty* fields, so correcting an
existing wrong value needs `--force`; worth knowing before assuming a re-run heals bad data). Then
the layer that consumed the old string was redone: the document's `proposed` placement was deleted
and re-proposed, landing on **Machine learning at 0.95** where the old run had made the same call
on the dashed string. That re-propose also served as live confirmation that the shipped runner now
resolves `qwen3.5:9b` through the KI-28-fixed adapter (2 calls, 0 abstentions, 0 unparseable).

**Tests:** +9 in `tests/unit/test_metadata_extractor.py` (43 total) — the real Nogueira shape, a
parametrized sweep over all seven glyphs, and a guard that *internal and trailing* dashes survive
(`Cross-encoder re-ranking for open-domain QA`), since the failure mode of an over-broad strip is
mangling legitimately hyphenated titles.

**Rejected:** stripping in `_title_candidates` only (would leave the citation-block fallback and
the skip-heading comparison seeing the dash); a general "strip all leading punctuation" (quotes and
parentheses do open real titles).

**What it opens.** Three KI-26 residuals still stand, all recoverable from cached markdown and all
listed in the baton: `mdl_tutorial_grunwald_2004` (plain unmarked first line), `ai_usage_cards_2023`
(`Preprint of the paper:` + self-citation), `nihms-326467` (`Published in final edited form as:`
hiding **Human Connectomics**). The `Disclaimer` row needs `--force` or the ADR-013 metadata editor.

---
## 2026-07-26 — README restructured 338→142 lines (setup/usage split out), demo GIF re-recorded on the current UI, em-dashes removed

**What changed.** (1) **README cut from 338 lines / 22.9 KB to 142 / 8.0 KB**, with nothing dropped,
only relocated. New [`docs/setup.md`](setup.md) takes install, torch extras, hardware guidance,
local-LLM specs, Docker and the Windows SSL fix; new [`docs/usage.md`](usage.md) takes ingest/launch,
the demo corpus, enrichment passes, library commands, `sync_sources` and the test matrix;
`docs/architecture.md` gains the repository-layout tree, which otherwise had no home. Kept in the
README: the mermaid diagram (renders natively on GitHub, unlike ASCII), the benchmark table with its
variance column, and Limitations. The ~140 lines of Setup + Usage were what pushed Benchmarks and
Limitations below the fold, which is the wrong order for the two sections that carry the evidence.

(2) **Three stale claims corrected while moving them.** `just api` **does not exist** as a recipe
(the justfile has `app`/`desktop`/`sidecar`/`test`/`typecheck`/`check`/`lint`, no `api`) — the README
had been handing readers a command that fails; replaced with the real invocation lifted from
`scripts/launch_app.ps1`: `uv run --no-sync uvicorn apps.api.main:app --host 127.0.0.1 --port 8001`.
Test count 1,015 → **1,306**. "mypy --strict clean" → "mypy (strict)", since AGENTS.md forbids that
flag (it thrashes the incremental cache) and `strict=true` is already in `pyproject.toml`. The
local-model limitation was rewritten from "~flat 0.8 gap ratings" to this session's measured numbers.

(3) **Em-dashes and AI-writing tells removed** from all three files. 28 flagged by
`ai-tell-detector`'s deterministic scanner (all em-dash asides; zero buzzword or construction hits),
plus code-comment and table-cell dashes the scanner does not reach. Final state: **0 em-dashes, 0
tells** across README/setup/usage. Retained `→`, `≥`, `≈` — technical notation, not punctuation.

(4) **Demo GIF re-recorded on the current UI** (`docs/assets/provenote-demo.gif`): 960×600,
**13 frames, 23.1 s, 0.83 MB — down from 1.73 MB**. Recorded against the live **97-document /
33,163-chunk** corpus on **`ollama/llama3.1:8b` ($0**, confirmed on `/api/health` before recording;
KI-4). Storyboard unchanged: empty state → sample chip → streamed cited answer → sources + citation
side panel → Library grid → concept-graph ego view. The new StatusBar means every frame now carries
`33,163 chunks · ollama/llama3.1:8b · bge-base`, which evidences the local-and-real claim for free,
and the Library grid shows the **KI-26-fixed titles** rather than the `OPEN ACCESS` banners the old
GIF captured.

**Why the recorder needed work.** The Topbar/StatusBar pass moved the mode switcher from
`.modes button.mode` to `.tb-modes button.tb-mode[role=tab]`, so the old storyboard would have
recorded three Chat frames and called it done. Rather than guess selectors from the Svelte source, a
`probe_dom.js` pass read them off the running app first; the recorder now also **logs what it
matched at each beat** (`sources: {...}`, `citelink: {...}`, `graph concept picked: ...`), so a
future stale selector surfaces as a named miss instead of a silently boring frame.

**Rejected:** committing the recorder into `scripts/` — still undeclared puppeteer-core/Pillow deps,
same call as 2026-07-19; the toolkit lives in the session scratchpad and survived retrieval from the
prior session's, which is evidence the arrangement is workable. Also rejected: clearing the chat
history rows that show in the GIF sidebar (real data; no DELETE endpoint) and re-running the ego
layout to un-clip the `hard negatives` node label (cosmetic, one node, not worth a re-record).

**What it opens.** The Library grid's first card is `- PASSAGE RE RANKING WITH BERT`, a leading-dash
title artifact — a fourth KI-26 residual, and the one most visible to a reader since it sorts first.
The concept-graph ego view clips node labels at the box edge on a 1280×800 viewport.

---
## 2026-07-26 — RG-015 labelled precision run (3 instruments, n=97) + KI-28: thinking models returned an empty completion through `OllamaClient`

**What changed.** `src/doc_assistant/llm.py`: `OllamaClient` gains
`reasoning: bool | None = False` (mapped to Ollama's `think`) and now logs
`ollama_empty_completion` instead of silently returning `""`. Four guard tests in
`tests/unit/test_llm.py`. Docs: RG-015 in `.claude/RIGOR_TODO.md` gains the measurement it has
owed since 2026-07-18; new KI-28.

**Why.** The task was RG-015's precision run on a current local model. `qwen3.5:9b` scored zero —
3/3 `taxonomy_propose_unparseable` — which looks exactly like an incapable model. It was the
adapter. A thinking model writes its reasoning to Ollama's separate `message.thinking` field,
drawn from the **same `num_predict` budget** as the answer; `OllamaClient` sent
`num_predict=max_tokens` (256, from `taxonomy_propose.DEFAULT_MAX_TOKENS`) and read only
`message.content`, so the response came back `done_reason="length"` with `content=""`. Probed
against `/api/chat`: think-on/256 → `''`; think-off/256 → `{"choice": 3, "confidence": 1}` in 14
tokens; think-on/2048 → correct but far slower. Reasoning-off is the right default for an adapter
whose entire job is one short JSON object. The empty-completion warning is the load-bearing half:
`complete()` returns `str` and every caller parses it, so `""` was indistinguishable downstream
from "the model answered nonsense" — the bug hid four layers from its cause, and would have hit
the reviewer, eval judge, `gap_suggest` and `taxonomy_propose` identically and silently.

**The measurement (division-level, ground truth hand-labelled from the real publications and
written to disk before any model's answers were opened).** Scored in two strata because
`_document_context` sends **only title + authors + year** — title quality is the entire input, not
merely correlated with output quality.

| Instrument | A (real title, n=89) | B (no/furniture title, n=8) | Overall | Abstentions |
|---|---|---|---|---|
| always-answer-majority baseline | — | — | 60% | — |
| `llama3.1:8b` (shipped default) | 82% | 88% | 82% | 0 |
| `qwen2.5:7b` | 72% | 50% | 70% | 2 |
| `qwen3.5:9b` | **88%** | 75% | **87%** | 0 |

Newer is not automatically better: `qwen2.5:7b` is a year old and **loses to the incumbent**, with
a different attractor rather than fewer errors. `qwen3.5:9b` beats it by 5 points overall. All
three clear the 60% majority baseline comfortably, so the binding constraint is input quality, not
model choice. **Confidence measured, not eyeballed:** `qwen2.5:7b` is *anti*-correlated
(mean 0.674 correct vs 0.700 wrong; all 27 errors carry 0.70, its maximum — a ≥0.7 auto-accept
admits 100% of them); `qwen3.5:9b` separates by only +0.042. The "never threshold on confidence"
rule is now a measurement.

**Rejected.** (1) *Raise `max_tokens` instead of disabling reasoning* — works (verified at 2048)
but pays reasoning tokens on every call of a pure classification task, and leaves the silent-empty
failure one budget-overrun away. (2) *Hardcode `think: false`* — the reviewer path may genuinely
want a trace; a parameter with a documented default costs nothing. (3) *Leave the fix in
scratchpad and report the defect* — rejected by the user; the measured configuration should be the
shipped one, and the full pass was re-run through the shipped runner to confirm it (220 calls, 0
unparseable, 0 empty completions, reproducing the shim run placement-for-placement). (4) *Score
against a single correct division* — 51 of 97 documents are genuinely dual-coded, so this charges
the model for the taxonomy's ambiguity; both a lenient and a strict number are reported instead.
(5) *Blame the model for `middleton-2001.pdf`* — it has **0 extracted body characters**; excluded
as an extractor fact, not a classifier failure.

**What it opens.** (a) **The streaming answer path is affected too — but not the way this entry
first claimed.** The claim here was that `pipeline.build_chat_model` (`OllamaLLM` on
`/api/generate`, no reasoning flag) would let a thinking model *leak its reasoning into the
answer*. That was inferred from the code and it is **wrong**: measured on the shipped constructor,
the answer text is clean and correctly cited — Ollama keeps reasoning out of the content on
`/api/generate` as well. What actually happens is that the reasoning is generated and streamed as
**empty content deltas**, invisible in the final string and highly visible to anyone watching
tokens arrive:

| model | time to first *visible* char | empty chunks before it |
|---|---|---|
| `qwen3.5:9b` | 11.99s · 15.17s · 12.79s | 236 · 713 · 539 |
| `llama3.1:8b` | 5.93s (cold) · 2.39s · 2.42s | 1 |

Warm-to-warm that is **~12.8s vs ~2.4s to first token (~5×)**, with the answer completing <0.2s
after it — so essentially the whole wall clock is dead UI, and thinking length is unstable
(236→713 empty chunks on three identical prompts). **No one hits this today**: `LLM_MODEL`
defaults to the Anthropic analysis model, or `llama3.1:8b` locally. It becomes reachable now that
`qwen3.5:9b` is on the box and is the taxonomy default. **Deliberately not fixed**, because unlike
KI-28 it is a real trade, not a free win: there, thinking output was unreadable by construction;
here the reasoning may improve a user-facing RAG answer, and no one has measured that side.
`OllamaLLM` does expose `reasoning`, so the symmetric one-liner is available — the alternative,
and the better fit for this project's "inform, don't block" rule, is to leave thinking on and
surface a *thinking…* state during the empty-delta phase, since the defect is the **silent** dead
air rather than the latency. (b) **Three more KI-26 title residuals**, all recoverable from cached
markdown: `mdl_tutorial_grunwald_2004` (title is the plain unmarked first line — the picker seems
to need a heading or bold), `ai_usage_cards_2023` (`Preprint of the paper:` + self-citation, same
family as the Frontiers fix), `nihms-326467` (`Published in final edited form as:` hiding **Human
Connectomics**). (c) The lexical failure mode is now legible and identical across models — one
salient token captures the placement; `qwen3.5:9b`'s sharpest case is *"political scienc\*"* →
**Chemical sciences** at 0.8 confidence, 4/4 political-science documents wrong, ruled out as an
index bug by replaying stage-1 and resolving the index by hand.

**Config switched (user decision, same session).** `TAXONOMY_PROPOSE_LLM_MODEL` default
`llama3.1:8b` → **`qwen3.5:9b`** (`config.py`), with the runner's `--help` and `.env.example`
updated to match — the first of the three confined-suggestion instruments to diverge from
`llama3.1:8b`, and it diverges on a measurement rather than a preference. Two consequences worth
stating plainly: the default now **requires a 6.6 GB `ollama pull qwen3.5:9b`** on a fresh box
(recorded in `.env.example`), and it is a **thinking** model, so it is only usable *because* of
the KI-28 fix above — on the previous adapter this default would have produced zero proposals and
looked like a broken model. The stored proposals in `library.db` are already qwen3.5:9b's, so
config and data now agree; `CONCEPT_SKELETON_LLM_MODEL` and `GAP_SUGGEST_LLM_MODEL` are untouched
(neither has been measured this way — do not assume the result transfers).

---
## 2026-07-25 — Slow commits diagnosed: `mypy --strict` was invalidating the incremental cache on every alternation (45s → 6.7s)

**Complaint (user):** commits on this machine are slow, expected to be "a lot of hooks or checks at
once". **Measured instead:** the hook battery is cheap and **one hook was paying a full cold mypy on
almost every commit**.

**The mechanism.** `[tool.mypy]` already sets `strict = true` *and* `warn_unused_ignores = false`.
Passing `--strict` on the CLI re-enables `warn_unused_ignores`, so it is a **different option set** —
and mypy keys its incremental cache on the options, so it discards the cache. The project's own
agent-facing docs prescribed `mypy --strict src`, while the pre-commit hook (and CI) run the bare
`mypy src`. Every alternation therefore cold-started the other form:

| command | timing on this box |
|---|---|
| `mypy src` (hook/CI form), warm | **2.4 s** |
| `mypy --strict src` after that | 40.9 s |
| `mypy src` again (now cold) | **40.5 s** ← every commit after a `--strict` run |
| `mypy src` once more (warm) | 2.3 s |

**Fix.** One canonical local command — `uv run --no-sync mypy src` — recorded where the habit comes
from: `.claude/CONTEXT.md` non-negotiable **§8** (with the numbers), the `AGENTS.md` digest,
`src/doc_assistant/CLAUDE.md`, and this session's spec's Gate line. New `justfile` recipes
`typecheck` (canonical) and `typecheck-strict`, the latter pinned to `--cache-dir
.mypy_cache-strict` so the divergent flag set **cannot** cold-start the canonical one (verified: two
strict runs in a row, bare form still 2.3 s afterwards); `.mypy_cache-strict/` gitignored.

**Rejected.** Dropping mypy from the hook (it is the pre-check that keeps CI green). Scoping
mypy/bandit to staged files (`pass_filenames: false` + whole-tree paths is deliberate CI parity — the
hook's own comment explains it, and whole-tree keeps the cost flat rather than diff-dependent).
Removing `strict=true` in favour of the flag (would make CI's bare invocation non-strict).

**Where the remaining ~6.7 s goes** (whole staged tree, warm, measured with `pre-commit run
--verbose`): mypy 2.3 · bandit 1.6 · detect-secrets 0.9 · hygiene hooks 0.5 · ruff + ruff-format 0.05
· pre-commit's own stash/restore + env checks ≈ 1.3. `git status` is 0.04 s, and the hook envs
(`~/.cache/pre-commit`, 8 envs / 203 MB) are stable — neither was implicated. Pre-push adds ~6.5 s
(cpc `test_api_check` 4.6 · `docs_check` 1.8 · `push_guard` 0.1).

**Also found while measuring:** the staged tree had one real `E501` (a docstring line in
`knowledge/taxonomy.py` added after the last lint run) — the hook was *failing*, not just slow, which
is its own contribution to "commits are slow" via retry cycles. Fixed. Worth remembering that
pre-commit checks the **staged** content (it stashes unstaged work first), so a fix that is not
`git add`-ed does not clear a hook failure.

**Not measured / left to the user:** whether a *post-boot* first commit is slower still because
Windows Defender real-time scanning is on with no exclusions for `.venv` (48 k files), the repo, or
the pre-commit cache. Reading or setting exclusions needs elevation, so it is the user's call —
excluding those three paths is the usual remedy if a cold commit still drags.

---
## 2026-07-25 — KI-26 fixed: six metadata-extraction failure shapes, measured on the corpus rather than guessed

Titles **14 → 6** unusable (5 of the 6 now honest `None`), banner titles **9 → 1**, years **75 → 89
of 97**. Method: diagnose each failing document against its *real* cached markdown, fix one shape at
a time, and re-diff the whole 97-document corpus after every change — so "fixed" is a count, not an
impression. The diff also gates regressions: **0 titles lost** at every step.

**The six shapes** (each has a guard test built from the real document, reduced):
1. **Page furniture wins the title pick.** `OPEN ACCESS`, `Disclaimer`, `Graphical abstract`,
   `ORIGINAL ARTICLE` are *headings above the title* in publisher front matter. Added to
   `_SKIP_HEADINGS` — exact-match only, so "Meta-analysis of X" is never touched.
2. **Frontiers titles are recoverable, not lost.** Their front matter carries a self-citation
   (`… (2024) <title>. _Front. Neuroanat._ 18:…`), so the title is **read back** rather than
   guessed. Recovered all 7. The first version of this regex required a specific terminator and
   worked on the real file but failed the reduced test — over-fitted to one sample; it now keys on
   the italic-journal boundary alone.
3. **The author line became the title** on two documents, because the title was a **bold line** and
   the authors were the **heading**, and the picker preferred headings. Now **position beats markup
   level**. Worth stating plainly: *no capitalisation heuristic can separate "Junyu Ren Lek-Heng
   Lim" from "Attention Is All You Need"* — order can, and does.
4. **The year came from someone else's paper.** With no publication keyword in DPR's header, the
   loose scan ran the whole 3k head and hit a citation year in the abstract → **2012** for a 2020
   paper. The loose tier is now bounded to the **front matter** (cut at abstract/introduction), so
   it is structurally unable to read a reference as a publication date.
5. **Tier order + reach.** `published`/`copyright`/`©` now outrank `received`/`accepted` (submission
   ≠ publication — that alone fixed a 2020-vs-2021); the keyword window may **cross a line break**
   (PMC manuscripts wrap, which had been costing the authoritative tier entirely); a **journal
   running header** (`… VOL. 51, NO. 7, JULY 2004`) is a tier; and a **filename year** fills in
   below all of them — but never overrides a date the document states about itself. The filename
   tier also refuses arXiv-shaped names, because `1904.01169v3.pdf` is not the year 1904.
6. **Dead code, found by removing the thing that hid it.** `_JOURNAL_HEADER` is anchored `^[A-Z]`
   but `_is_skippable_heading` lowercases first — so the journal-citation skip **never fired**. The
   H1-over-H2 preference had been compensating for it. Removing that preference surfaced it
   immediately as a failure of the repo's own `test_title_prefers_h1_over_h2`.

**Downstream redone, because the inputs changed** — the point of fixing extraction is what consumes
it: the 10 documents whose titles were corrected had their taxonomy proposals **deleted and
re-proposed** (they had been classified on `OPEN ACCESS` and on author names), and the re-run is
visibly better — `Cajal's legacy in the digital era`→Neurosciences 1.00, `In search of relevance…`→
Political science, `Low-dimensional topology of deep neural networks`→Pure mathematics. And
`compute_epistemics` was re-run because **18 document years moved** and G3/G6's `superseded_trend`
is a median-year-per-side rule.

**Rejected.** A capitalisation/name-shape heuristic for author-line titles (see 3 — it fails on real
titles). Clearing a stale title to `None` under `--force` (the runner declines to overwrite with
nothing; that is the safer default, so one stale `Disclaimer` row stays until re-ingest or a user
edit via the ADR-013 metadata editor). Trusting the filename year over the document (four
arXiv-vs-conference disagreements remain, and in each the document's published year is the better
answer). Chasing the 5 title-less old scans — they have no heading anywhere in the extracted
markdown, so `None` is the correct output, and their years now come from their filenames.

**Gate:** ruff/format · mypy 69 files · `tests/unit/test_metadata_extractor.py` **34 passed** (+11).

**What it opens.** RG-015's precision measurement is now unblocked — it was measuring the extractor
as much as the classifier while 14% of titles were garbage. And `confidence` in `DocMetadata` still
weights title/authors/year/DOI equally; now that the tiers differ sharply in trustworthiness, a
year's contribution arguably ought to depend on *which* tier produced it.

---
## 2026-07-25 — Corpus transfer: 47 → 97 documents, then the full enrichment chain re-run on the enlarged corpus

The retired machine's 50 remaining source PDFs (153 MB) arrived in its backup and were ingested
here. **Its `library.db` was NOT adopted** — the vector store could not come with it (KI-11 puts
Chroma under `%PROGRAMDATA%` on a non-ASCII data path, so no repo-folder backup contains it), and a
`library.db` whose chunk rows point at absent embeddings is worse than no DB. Re-ingest from
`data/sources/` was the only coherent path, and it is also self-verifying: hash-dedupe proved the
old 47 were untouched (`added=50, errors=0, skipped=47`).

**Chain, in dependency order, all $0 (local Ollama / CPU embeddings):**

| Stage | Before | After |
|---|---|---|
| documents | 47 | **97** |
| parent-child chunks | ~16k | **33,163** |
| `doc_similarities` | 470 edges / 47 docs | **970 / 97** (needed `--force`: the runner skips when edges exist, so new docs would have got none) |
| `citations` | 2,859 (**9** in-corpus) | **4,825 (25 in-corpus)** — the bigger corpus resolves more references into real edges |
| `keywords` | 688 | **1,376**, 1,376 distinct (checked — the exact doubling is coincidence, not duplication) |
| concept skeleton | 13 nodes / 19 edges | 13 / 19, **all 19 Node-B annotated** (relation + per-doc stance), 6 communities, presence in 30 docs |
| `chunk_epistemics` | — | **743** rows |
| gaps | 15 | 16 deterministic + 2 stochastic preserved, **0 orphans** (the KI-17/E0 reconcile holding at 97 docs) |
| taxonomy placement | 47/47 docs proposed | **97/97 — 0 unclassified** (+50 proposals, 100 calls, **0 abstentions**, all `origin="proposed"`) |

**The graph vocabulary did not grow, and that is the honest read:** 13 `graph_include` concepts over
97 documents. Doubling the corpus doubled the *keywords* (1,376) but the curated map is unchanged, so
concept presence covers 30 of 97 documents. `scripts/rank_candidates.py` now has twice the evidence
to rank promotion candidates from — the cheapest real improvement available, and a curation call
(ADR-018's opt-in boundary), not something to auto-apply.

**It broke the app mid-chain** — the parent-child store crossed SQLite's parameter ceiling and every
unpaged Chroma read started failing, including the BM25 build in `RAGPipeline.__init__`. Diagnosis,
fix and guard are the KI-27 entry above; the chain resumed after it.

**Verified live after the rebuild ($0):** a real SSE turn cited `dpr_karpukhin_2020.pdf` ×5 +
`rag_lewis_2020.pdf`, top reranker 0.9795, source-evaluation strip populated
(`coverage: contested`, `n_claims`, `year`), `is_local: true`.

**The taxonomy pass repeated its known failure shape, now with more evidence (RG-015).** Confident
and right where the title is real (`Attention Is All You Need`→Machine learning 1.00, `Deep Residual
Learning`→Computer vision 1.00, `hodgkin_huxley_1952`→Medical physiology 0.90,
`Automatic Classification of Heart…`→Cardiovascular medicine 0.90). Confident and meaningless where
the title is page furniture: **`Disclaimer`→Neurosciences 0.80**, `Graphical abstract`→Medical
biotechnology 0.80, and two author-line titles placed on the authors' names. Zero abstentions across
100 calls, again — **the model does not decline, so placement quality is bounded by title quality,
and confidence carries no information** (0.8/0.9/1.0 only). Coverage-based gap detectors must count
**curated** links; the read model's `n_*_proposed` split exists for exactly this.

**Found in that verification — a metadata defect that is *wrong*, not missing:**
`dpr_karpukhin_2020.pdf` is recorded as **`year=2012`** (it is 2020) and `rag_lewis_2020.pdf` has no
year at all. Year drives the G3/G6 median-per-side `superseded_trend` rule, so a mis-extracted year
can invert a direction verdict — and the strip presents it to the user as fact. Unusable titles are
now 13/97 (7 `OPEN ACCESS` banners + 1 `Disclaimer` + 5 null, all old scans). **KI-26 widened** from
banner titles to metadata extraction generally, with the year rule spelled out.

**Rejected.** Copying the retired box's `library.db` (embeddings absent → a DB that lies).
`--rebuild` (KI-24: it deletes and re-derives everything, discarding folders/tags/metadata overrides
— the dedupe made it unnecessary). Restoring the CUDA wheel first: the venv is on `2.12.0+cpu`
despite the box having a working GPU, but measured ingest was ~25 s/doc, so a multi-GB download to
save ~20 minutes was not worth the venv surgery mid-chain. (It *is* worth doing before the next
embedding-heavy run — noted in the baton.)

**What it opens.** `data/sources_manifest.yaml` still describes 77 entries for 97 files — the 50 new
ones have no recorded provenance, so `sync_sources` cannot re-fetch them. Node-B stance now exists
for all 19 edges, which makes the epistemics chips and the D3 strip non-trivial again (they were
association-only). And RG-015's precision question can finally be asked on a corpus that is not
tiny.

---
## 2026-07-25 — KI-27: unpaged Chroma reads are a correctness cliff, not a perf risk — the corpus transfer took chat down at 33k chunks

Found by ingesting the transferred corpus (entry below). **Filed separately because the lesson is
not about the corpus:** the C4 scale review (2026-07-19) listed "unpaginated whole-corpus loads"
under *performance*. It is not. Chroma's SQLite backend binds **one SQL parameter per returned
row**, so a whole-store `get()` **fails outright** — `too many SQL variables` — the moment a
collection passes SQLite's **32766** ceiling. A step function, not a slope.

**What broke.** The ingest took the parent-child store from ~16k to **33,163** chunks (47 → 97
docs). That is 397 rows past the ceiling, and it took out `compute_epistemics`,
`build_concept_skeleton`, and — the one that matters — **`RAGPipeline.__init__`**, whose BM25 index
build reads the whole store. The pipeline could not *construct*, so **chat was down, not degraded**.
The baseline store (12,800 rows) was still under the ceiling, which is why ingest and
`compute_doc_vectors` kept working and the first crash looked module-specific. It took a second
failure, in a different module, to show it was a class.

**Fix.** New `src/doc_assistant/chroma_read.py` → `get_all(collection, where=, include=)`: pages via
`limit`/`offset` and concatenates per key, so a caller sees exactly what an unpaged read would have
returned. `PAGE_SIZE = 5000` is a **structural** bound on parameters-per-statement, not a tuned
threshold — the parametrised test asserts page size 1, 7 and 100 all return the identical result.
Applied to every whole-store read: `pipeline.py` (BM25 build + `chunk_count`), `epistemics.py` (×2),
`concept_skeleton.load_presence_inputs` (which additionally batches its `$in` document filter — a
second, smaller ceiling), `doc_vectors.py`, `ingest/store.py`, `ingest/cleanup.py` (×2).

**Typing note.** `get_all`'s first parameter is `Any`, deliberately and with a comment: the callers
are a raw `chromadb.Collection` *and* LangChain's `Chroma` wrapper, whose `get` signatures differ in
the `include` literal type. A `Protocol` would type-check exactly one of them — a fiction.

**Guard.** `tests/unit/test_chroma_read.py` (+8) drives a fake collection that **counts pages**, so
"no single call exceeds the page size" is asserted rather than assumed, alongside order
preservation, the empty-collection contract (0 docs is legitimate), filter pass-through, and
page-size independence. Full suite **1290 passed**.

**Live proof ($0, ollama/llama3.1:8b):** API boot rebuilt the BM25 index over all 33,163 chunks
(`building_keyword_index` → `bm25_excludes count=0`), `/api/health` served `chunk_count: 33163`, and
a real SSE turn returned a cited answer — 10 sources, top reranker 0.9795, all citing
`dpr_karpukhin_2020.pdf`/`rag_lewis_2020.pdf`, `usage.is_local: true`, `cost_usd: null`.

**Rejected.** Raising the page size to "just above the corpus" (the same cliff, moved). Catching the
error and falling back (it is a query the code should not be issuing). Fixing only the module that
crashed first — the second failure is what proved it was systemic. Leaving `pipeline.py` for later:
it is the one that means *chat is down*.

**What it opens.** Two reads remain unpaged **by construction, not oversight** — `wiki.py` (one
document, `limit=per_doc`) and `library.py`/`ingest` hash lookups — all bounded filters. Worth a
sweep if a new whole-store reader appears. And the C4 review's other "corpus-linear hot paths"
deserve re-reading with this lens: which of them are actually cliffs?

---
## 2026-07-25 — ADR-029: the working state goes local — all of `.claude/` + PLAN/REVIEW docs + the UI checklist untracked

The user retired the second machine ("we won't be using the other pc anymore… we can conveniently
hide more stuff in .gitignore"), which **kills ADR-020's deciding premise**: it committed
`.claude/RIGOR_TODO.md` *specifically* to converge two boxes' disjoint copies ("git is the mechanism
the ritual was missing"). With one box git is not a sync mechanism for these files at all — only a
publication channel, on a repo that is public and read as a portfolio.

**What.** `.gitignore`: `.claude/*` + allowlist → plain `.claude/`; new `docs/PLAN_*.md`,
`docs/REVIEW_*.md`, `docs/ui-checklist.md`. `git rm --cached` on the 7 files (all still on disk):
`.claude/{CONTEXT,KNOWN_ISSUES,RIGOR_TODO}.md`, both `PLAN_2026-07-2x` docs,
`REVIEW_2026-07-19_scale-robustness.md`, `ui-checklist.md`. New
`docs/decisions/ADR-029-local-only-working-state.md` (supersedes ADR-020, amends ADR-001's allowlist
— now empty, narrows ADR-022's publish set); ADR-020 marked superseded in the index.

**Why (and the costs, which are real).** These files are addressed to the builder, not the reader.
But: **~40 committed documents link to them** (109 occurrences of `ui-checklist|PLAN_2026|REVIEW_2026`
— measured, not estimated), so a clone gets dangling links; and **git was the off-machine copy**,
which disappears exactly as the second machine does. Both are named in the ADR rather than glossed,
with three mitigations shipped: new committed **`docs/local-only.md`** (what's local + why + what the
public record *is*), an `AGENTS.md` banner over the coordination list (a clone has none of these; the
digest is what you get; back up `.claude/` yourself), and the recommendation that a private sibling
repo is the upgrade path if the backup gap bites.

**Rejected.** Status quo (publishes internal validation debt + half-finished punch lists for a sync
benefit that no longer exists). A private submodule *now* — the better end state, but a second repo to
run for a solo project; kept as the named reverse-if. Untracking `RIGOR_TODO.md` alone (leaves exactly
the planning docs the user objected to). Rewriting the 109 references — would churn append-only history
(DEVLOG, sprint archives) whose value is being verbatim.

**What it opens.** ADR-020's owed merge is now conditional on the pending data transfer: **RG-014**
(cited as authority in ADR-017/018/019 + `feature-concept-graph.md`), RG-007 and possibly
RG-003/005/006 exist only on the retired box's copy. If that file doesn't arrive, those citations are
permanently unverifiable and must be treated as such. Also newly unreachable: **G4**
(`SPRINT-004-ki10-frozen-os-trust`) needed a TLS-MITM box to verify — that was the retired machine.

---
## 2026-07-25 — Taxonomy increment 3 (ADR-028 D8): auto-propose placements, propose-only, $0 on local Ollama

The taxonomy could be filled by hand (increments 1/2a/2b) but nothing was placed, so ADR-028 D6's
coverage math read 0 everywhere. This is the first-pass filler: one quarantined local-LLM pass that
**proposes** an `in_field` parent per unplaced concept and a field per unclassified document, written
as `origin="proposed"` links the user accepts or deletes. Spec:
`docs/specs/feature-taxonomy-auto-propose.md` (written first, DoD 1–9).

**What.**
- **Schema:** `ConceptHierarchy.origin` (`curated`|`proposed`, non-null, default curated) + the
  `_ADDITIVE_COLUMNS` row with a **literal DEFAULT** so the ALTER backfills existing rows in the same
  change (KI-25) — every pre-increment edge is the seed or a user edit; a proposal cannot predate the
  pass that makes them.
- **Seam (`knowledge/taxonomy.py`):** `add_hierarchy_edge(..., origin=)` where a **curated write over
  a proposed row promotes it in place** — that *is* the accept primitive, so accepting needs no new
  endpoint — and a proposed write never demotes a curated row. New `unplaced_concepts(graph_only=)` /
  `unclassified_documents()`; `load_taxonomy` now carries `origin` on each edge.
- **Pass (`knowledge/taxonomy_propose.py`, new):** **two-stage narrowing** — division (~23 options)
  then group *within that division* (~10) — because a small local model chooses far better twice-small
  than once-from-236, and the intermediate answer is itself a valid placement target. **Abstention is
  first-class** at both stages: stage-1 "none" ⇒ no proposal (a wrong placement inflates the very
  coverage this layer exists to make trustworthy); stage-2 "none" ⇒ the **division placement stands**
  (`field_id == division_id` marks it). Confidence = the chain's **weakest present link**, and is
  `None` rather than fabricated when the model gives no usable number. `propose_placements` is DB-free
  and writes nothing; `run_propose` owns the session either side of it (the `gaps.build_gaps` shape).
- **Runner (`scripts/propose_taxonomy.py`):** dry run = scope + call budget with **zero LLM calls**;
  `--apply` runs and writes. `TAXONOMY_PROPOSE_LLM_{PROVIDER,MODEL}` default to Ollama explicitly
  (KI-4 — this is the highest-volume of the three confined passes, so an inherited paid default would
  be the worst footgun of the set) and `--apply` routes through `assert_provider_intent` before any
  client exists. `--limit`/`--all-concepts` always print what they dropped or skipped.
- **Wire:** `TaxonomyField.n_{concepts,documents}_proposed` + `FieldMember.origin` through
  `taxonomy_view` → payloads → `types.ts`; `FieldMember extends LabelledOption`, and the attach
  picker's vocabulary is now `LabelledOption` (an *offerable* concept has no link, so no origin —
  TX2b had reused `FieldMember` for it).

**Why zero calls on a dry run** (not "run the LLM, just don't write"): `assert_provider_intent`
deliberately no-ops when `apply=False`, so a calling dry run with `--provider anthropic` would bill
without ever tripping the guard. "No `--apply`" has to mean "no spend".

**Coverage counting decision.** `*_direct`/`*_rollup` stay **origin-inclusive** — a proposed link is
genuinely attached, pending review — with the proposed share broken out at the *direct* level so a
consumer can subtract. No rollup-level breakdown until something consumes it: that is the
coverage-based gap detectors, which stay gated behind **RG-015** and should require curated links,
because auto-propose precision is unmeasured. This ADR-028 "must revisit" (the display rule) is now
decided for the read model; the UI half is TX3b.

**Rejected.** A separate proposals table (the distinction needed is the provenance of *one link*, and
`DocumentField.origin` already established the vocabulary; two stores would drift). One-shot over all
236 fields (weakest form of the question for an 8B model). Forcing a placement when the model is
unsure. Clamping an out-of-range index into range (it would manufacture a placement the model never
chose — an out-of-range answer is a parse failure). `is_a` proposals (a different judgment, no seeded
candidates).

**Gate.** ruff/format clean · `mypy --strict src` 69 files · bandit 0/0 · **pytest 1282 passed** ·
`svelte-check` 0/0 (146) · `npm test` 50 pass · `docs_check --strict` 0/0 · `integrity_check` 0/0
(518). New/extended tests: 15 in `tests/unit/test_taxonomy_propose.py` (scripted call-**counting**
fake client — the "zero items/no fields/dry run ⇒ zero calls" contracts are asserted, not assumed),
6 in `test_taxonomy.py`, 1 new API integration test + detach assertions on the existing attach test.

**A hole this increment exposed, closed in it:** `attach_document_field` had no counterpart, so a
*proposed* document classification could never be rejected — "the user accepts or deletes" was not a
property the product had. Added `taxonomy.detach_document_field` + `DELETE
/api/taxonomy/documents/{id}/fields/{id}` + `api.ts detachDocumentField` (origin-agnostic: it also
undoes a curated attach, mirroring `DELETE /hierarchy`).

**Live on this box ($0, ollama/llama3.1:8b).** Its DB **predated increment 1 entirely** — no `kind`
column, no taxonomy tables (the 07-23/24 taxonomy work was verified on the now-retired machine). So
the run was: back up `library.db` → `init_db()` added `concepts.kind` + created both tables →
`seed_taxonomy --apply` (236 domains + 213 edges) → dry run (13 in-scope concepts + 47 documents = 60
items, ≤120 calls, **344 out-of-graph concepts reported as skipped**, zero LLM calls) → `--apply`:
**120 calls, 60 proposals, 0 abstentions, 0 division-only, 13 + 47 rows written.** Then live over the
running API: 236 fields / 23 roots intact · unassigned 357→344 · **rollup crosses group→division with
proposals included** (Information and computing sciences 10c/15d rolled up, 0 direct) · every member
serves `origin:"proposed"` · **accept = the plain curated POST promoted in place** (direct stayed 9,
proposed 9→8) · new DELETE removes 1 then 0. Box restored (the accept test's edge was deleted and
re-proposed — same placement at temperature 0).

**Quality, read honestly (this is RG-015's first evidence, logged there).** 13/13 concepts plausible —
`cre`→Genetics, `dbs`/`ntsr1`→Neurosciences, `pddl`→Artificial intelligence, a good counter to the
twice-made "unfamiliar short label = junk" error — but **every** IR concept landed in *Machine
learning* although ANZSRC has 4605 Information retrieval: expect coarse, not wild. **~9 of 47
documents are clearly wrong**, each from one salient token hijacking the choice (`SciRepEval`→
Astronomical sciences, `Res2Net`→Mechanical engineering, `Leroy: Library Learning…`→Library studies).
**7 more were unplaceable for a reason that is not the model's:** their title is the literal string
`OPEN ACCESS` — Frontiers PDFs whose banner became the title, filed as **KI-26** (it hits the Library
list and citations too, not just this pass). And **confidence is not a signal** — every value was
0.8/0.9/1.0, the same flat-rating pathology `gap_suggest` showed on llama3.1:8b; do not rank or
auto-accept on it. The proposals were left in place (origin-marked, rejectable).

**What it opens.** **TX3b:** the proposal badge + accept/reject in `LibraryTaxonomy.svelte` (accept =
the existing curated POST, reject = the existing DELETE) and an unplaced queue. **RG-015** should now
measure placement precision on a sample before any detector trusts it. The `--all-concepts` run over
this box's 344 keyword-family concepts is available but deliberately not the default (ADR-018's
boundary).

---
## 2026-07-24 — UI cleanup pass 3b: corpus/model info → bottom status bar (toolbar keeps a small mark)

Small follow-up to pass 3. The corpus/model subtitle (`N chunks · model · embedding`) moved out of the
toolbar brand into a new full-width **bottom status bar** — it's ambient status, not navigation, so the
status-bar footer is its idiomatic home; the toolbar keeps just the small mark + wordmark as the
identity anchor (user's call: "keep a small mark top-left"). **Frontend only, $0.** `svelte-check` 0/0 ·
`npm test` **50 pass** · live-verified (no overflow any mode, mobile, dark, 0 console errors).

**What.**
- `App.svelte` — brand in the toolbar is now mark + wordmark only. New `.statusbar` as the last child
  of `.app` (flex column: toolbar · below · statusbar): a connection dot (green ready / amber
  connecting / red down) + the corpus/model text, thin (~26px), full width under sidebar + content,
  `role="status"`. Same three health branches the toolbar meta used.
- `Sidebar.svelte` — **bugfix surfaced by this change**: `.sidebar` was `height: 100vh`, which now
  overshot by the toolbar height (the rail sits inside `.below`, between toolbar and status bar, not
  the full window) → 47px of phantom vertical scroll. Changed to `height: 100%`. The mobile drawer
  (absolute in `.below`) inherits the fix and now ends exactly at the status bar.

**Why.** Ambient status belongs in a status bar (IDE/editor convention), not in the action bar; moving
it also de-crowds the toolbar. Brand stays top-left where identity anchors are conventionally expected.

**Rejected.** Moving the whole brand (logo + wordmark) into the status bar — a headless top bar reads as
off; kept the mark up top. A chat-only status bar — the bar spans all modes for consistency (thin +
quiet so it never competes with the composer sitting just above it in chat).

**What it opens.** The status bar is now a natural home for other ambient signals later (scope, sync,
indexing progress). `--ok` (green) has no token yet — used an inline `#2e9e5b` fallback; worth adding a
success token to the palette if this recurs.

---
## 2026-07-24 — UI cleanup pass 3: unified top toolbar (browser-chrome shell) + back/forward view history + ☰ app menu

Reshaped the whole shell to the browser-chrome pattern the user kept referencing. **Frontend only,
$0, zero-LLM — nothing in `src/` or `apps/api/` touched.** `svelte-check` 0/0 · `npm test` **50 pass**.
Live-verified on :1421 with the api backend up (30,882 chunks): full nav-history retrace, all menus,
collapse, mobile, dark — 0 console errors.

**Why.** The user was lukewarm on the split-chrome shell (mode pills in the sidebar + brand/actions in
a header bar) and asked for the two remaining browser-toolbar pieces: **← → back/forward** ("go back
window", full view history) and a **☰ app menu** (shortcuts/settings/more). Both only make sense once
the shell commits to a single top toolbar, so we unparked and did the restructure (user chose "unified
top toolbar" + "every view you visit" for history).

**What.**
- **Unified top toolbar** (`App.svelte`): `.app` is now a column — a full-width `.topbar` over a
  `.below` row (sidebar │ resizer │ content). The bar carries `☰ · ⊟ · ← → · brand · [Chat|Library|
  Graph] · 🔍 · ⚙`. The mode switch **moved out of the sidebar** into the toolbar; the sidebar is now
  purely the contextual list. The old in-`main` `<header>` is gone (that was pass-2's full-width
  banner; now the toolbar spans the actual window, above the sidebar too).
- **Collapse simplified**: with the mode tabs + search always in the toolbar, collapsing just hides
  the rail (+ resizer) — the pass-1 **mini-rail is removed** (redundant). `sidebarCollapsed` persists.
- **Back/forward view history** (`App.svelte`): a passive `$effect` observes the navigable snapshot
  `{mode, libraryCollection, libraryDocId, graphSelectedId}` and records an entry on any change (cap
  50, `untrack` so the stack write doesn't self-trigger). `navBack`/`navForward` replay an entry
  through the real navigation paths; setting `navIndex` first makes the observer see the restored
  state already matches and not re-record. A new navigation after going back truncates the forward
  tail. Chat is tracked at mode granularity (opening a past conversation stays an in-rail affordance).
  Verified: Chat→Library→doc→Graph→concept records 5 steps; back retraces to Chat (arrow disables at
  the ends); forward re-walks; a fresh nav kills the forward tail.
- **☰ app menu**: Settings, Keyboard shortcuts, Export transcript (chat-only), About Provenote.
  Two small new modals — **`ShortcutsDialog.svelte`** (Ctrl/⌘K, Enter, Shift+Enter, Esc; ⌘ vs Ctrl by
  platform) and **`AboutDialog.svelte`** (product blurb + live corpus stats + the standing **ANZSRC
  CC-BY 4.0** attribution the taxonomy seed owes, ADR-028). Settings keeps its own ⚙ fast path too.
- **Sidebar** (`Sidebar.svelte`): dropped the `.top`/mode-pills/collapse/search cluster and the
  mini-rail + their props (`onSelectMode`, `collapsed`, `onToggleCollapse`, `onOpenSearch`) — the bar
  owns all of that now. On mobile the drawer re-anchors to `.below` (`position: absolute`) so it
  slides in **under** the toolbar, which stays visible/usable (was overlapping it).
- New icons: `arrow-right`, `keyboard`, `info`.

**Responsive.** Toolbar sheds weight as it narrows: the model/chunk subtitle hides < 1080px, then the
tab labels go icon-only + the wordmark hides < 780px. Mobile keeps ☰/drawer-⊟/←→/search; no overflow
at 375px.

**Rejected.** Bolting ← → and ☰ onto the old split shell (kept the very split the user disliked). A
full app-menu-only settings (kept ⚙ as the fast path). Tracking chat conversation-opens as history
steps (noisy; the sidebar already navigates those). A full-height mobile drawer over the toolbar
(re-anchored below it instead).

**What it opens.** This unparks the demote-Graph fork by making the mode switch a toolbar concern —
Graph now lives as a toolbar tab like the others; where it *belongs* is still an open product question,
but the shell no longer forces it into the sidebar. History is in-memory per session (not persisted).
The two new modals are the first non-Confirm info dialogs — a shared modal shell is now clearly worth
extracting (six hand-rolled scrim+card copies and counting).

---
## 2026-07-24 — UI cleanup pass 2: full-width banner, top-left collapse·search cluster, slim New-chat, folder-picker filters, Library multi-select → add-to-folder

Second play-feedback pass — cross-mode layout homogeneity + folder ergonomics. **Frontend only, $0,
zero-LLM — nothing in `src/` or `apps/api/` touched.** `svelte-check` 0/0 · `npm test` **50 pass**.
Live-verified on :1421 (read_page + computed styles; light+dark, desktop+375px, 0 console errors).

**What.**
- **Banner spans the full content width in all three modes.** `<header>` was hoisted out of `<main>`
  into `.content` (new `.viewport` wrapper centers each mode's `<main>` below it). Previously the
  header lived inside `main`, whose 820px chat cap vs 1500px library/graph cap made the banner jump
  width/centering between modes. Header now carries its own `1rem` side padding (it no longer inherits
  main's gutter). Verified: header x/width identical across Chat/Library/Graph (== `.content`).
- **Top-left cluster = collapse · search** (the conventional chat-app shell): the global-search
  trigger moved from the header `.actions` into the sidebar's top row beside the collapse button, and
  onto the collapsed mini-rail. Header `.actions` now holds only chat-Export + Settings. Ctrl/⌘-K
  unchanged; search stays visible on mobile (the drawer is its home there), collapse is desktop-only.
- **"New chat" slimmed** from the chunky full-width filled button to a tree-style row (`+ New chat`)
  at the top of the chat list — matching Library's "All documents" and Graph's "Taxonomy" row idiom,
  so every rail opens the same way. New `plus` icon.
- **Folder picker (Manage folders) made findable:** search now matches title/**author**/**filename**
  (via the grid's `filterDocs`, was display-label-only — a doc was unfindable by filename or 2nd
  author); added quick **type chips** (only when the corpus has ≥2 formats) + an **Unfiled** chip
  (docs in no folder yet, the common case when building one) + an "N shown" count.
- **Library multi-select → add to folder** (new). A **Select** toggle beside the view switch turns
  tiles/rows into checkboxes; a slim action bar shows "N selected · Add to folder… · Clear · Done".
  The folder menu lists folders (or routes to Manage folders when there are none). Selection is
  App-owned (`libSelected`); LibraryGrid stays a dumb renderer (`selectMode`/`selectedIds`/
  `onToggleSelect` props, ⋯ menu hidden in select mode). New `square-check-big` icon.

**Why.** Play feedback: the banner felt non-standard because it resized between modes; the button
cluster didn't match the conventional menu·collapse·search top-left; New chat was oversized; adding
many documents to a folder one-⋯-menu-at-a-time was tedious and the picker couldn't find docs by
filename/author.

**Folders are organization + opt-in scope, not RAG isolation** (restated for the record, unchanged):
default retrieval is whole-corpus; a folder only scopes a turn when explicitly picked beside the
composer (ADR-025 F2/F4), with a visible scope chip + provenance. Multi-select is add-to-folder only.

**Rejected.** **Bulk delete in the select bar** — the recorded position is "multi-select bulk delete
deferred (needs ADR)" (ui-checklist), so selection stays add-to-folder-only this pass. Keeping search
in the header (breaks the conventional top-left cluster). A 4th ⋯ menu item for multi-select (the
menu geometry is hardcoded for 3 items, `LibraryGrid.svelte`; a toolbar toggle sidesteps that).

**What it opens.** Bulk delete now has a natural home (the select bar) once its ADR lands. The
hoisted-header structure is the frame a future back/forward nav cluster would slot into (deferred —
it needs a real navigation history). `openMenu`'s hardcoded height is still a footgun for future menu
items. Folder-picker type chips are dormant on this single-format corpus (all-PDF).

---
## 2026-07-24 — UI cleanup: collapse control into the sidebar, graph rail into the shared Sidebar, chat-only Export, filter/search differentiation

Four post-play irritations fixed in one pass (grilled 2026-07-24; all four forks user-resolved to the
recommended option). **Frontend only, $0, zero-LLM — nothing in `src/` or `apps/api/` touched.** The
parked demote-Graph fork (ADR-025 fork 5) is untouched: Graph keeps its top-level nav slot — this fills
its empty rail, it does not move the destination. `svelte-check` 0/0 · `npm test` **50 pass** (+5).
Live-verified on :1421 (read_page + computed styles; light+dark, desktop+375px, 0 console errors).

**What.**
- **Collapse control** moved from the header (where it read as part of the brand block) into the
  sidebar's own top row; collapsed now shows a **48px mini-rail** (expand button + icon-only
  Chat/Library/Graph, still switchable). CSS-gated to ≥721px — a persisted collapsed flag cannot leak
  into the mobile drawer, which always renders the full sidebar. `.app.collapsed` now only hides the
  resizer; the sidebar hides itself via its `collapsed` prop.
- **Graph rail**: the concept index (Concepts/Gaps tabs, filter, lenses, list) moved out of
  ConceptGraph's in-view `aside.index` into the shared Sidebar — one rail, not an empty shared rail
  beside a private one. New **`lib/GraphIndex.svelte`** (dumb renderer), composed by App and injected
  via a **snippet prop** (`graphRail`) so Sidebar doesn't grow ~8 graph props. Selection
  (`graphSelectedId`) + the under-connected lens lifted to App — the rail badges and the ego panel's
  gap notes/dots must agree; a hygiene `$effect` clears selection if a rebuild drops the concept.
  ConceptGraph is now the full-width ego/detail panel; presence loads in an `$effect` on the
  `selectedId` prop with a cancellation flag (rapid-click out-of-order guard). GapList moved untouched
  (built relocatable, E5). Index filtering/sorting extracted to pure `conceptIndexRows` /
  `visibleConceptGaps` in **`lib/gaps.ts`** (+5 node:test cases).
- **Taxonomy entry** moved from the Library rail to the Graph rail (it's concept-space; same modal,
  same graph-node Place deep-link).
- **Export** renders only in chat mode — it exports the conversation transcript and was a permanently
  greyed button on Library/Graph.
- **Sidebar search boxes** are now visibly *filters*: "Filter chats… / Filter library… / Filter
  concepts…" placeholders, transparent fill, smaller type — the Ctrl/⌘-K overlay stays the one
  global search (spec A1 integrity boundary unchanged).

**Why.** Play feedback: the collapse button read as brand chrome; Graph showed a dead shared rail next
to its real one; Export was inert on 2 of 3 views; two same-styled search boxes implied redundancy.

**Rejected.** Keeping the collapse toggle in the header far-left (still header clutter; the control
belongs to what it collapses); a taxonomy field-tree as the graph rail (keeps the double-rail);
threading graph data through Sidebar props (snippet keeps Sidebar dumb); removing the sidebar filters
outright (in-place narrowing of the library grid is real utility).

**What it opens.** The graph rail now lives in the sidebar frame the demote-Graph fork will eventually
reshape — whatever that grill decides, the index is already a self-contained component. GraphIndex's
rail-local filter state resets on mode switch (accepted; lift to App if it grates). The duplicated
sort-menu markup (App library vs Sidebar chat) and duplicated `relTime()` (Sidebar/GlobalSearch) remain
known cleanup candidates.

---
## 2026-07-24 — Replace native `window.confirm()` for conversation delete with an in-app dialog

Swapped the OS-chrome confirm ("localhost:1420 says…") shown when deleting a conversation for a
standard in-app modal. **Frontend only, $0, zero-LLM — nothing in `src/` or `apps/api/` touched.**
`svelte-check` 0/0 · `npm test` **45 pass**. Live-verified: dialog opens from the conversation
options menu, centers with a scrim, Delete is danger-red with a trash icon, Cancel/Esc/scrim-click
back out, and cancelling leaves the conversation intact (no console errors).

**What.**
- **`lib/ConfirmDialog.svelte`** (new) — a generic scrim+card confirm dialog modeled on
  `LibraryDeleteConfirm`'s house style (Esc / scrim-click / Cancel close; `tone: 'danger' | 'default'`,
  danger carries the trash icon + red styling). Reusable, not conversation-specific.
- **`App.svelte`** — `deleteConversation(sid)` no longer calls `window.confirm()`; it opens the dialog
  via `pendingDeleteConvId` state, with `confirmDeleteConversation()` doing the soft-delete
  (`updateConversationMeta deleted:true`) and a `deleteConvBusy` guard. On error the dialog stays open
  to retry. Sidebar's `onDelete` contract is unchanged.

**Why.** The native `confirm()` broke the app's look and reads as untrustworthy OS chrome; the app
already had a matching modal pattern (`LibraryDeleteConfirm`) but hardcoded to documents.

**Rejected.** A one-off `ConversationDeleteConfirm` (would proliferate near-identical modals);
refactoring `LibraryDeleteConfirm` to consume the new generic (out of scope — its body is doc-specific
with chunk counts, left untouched).

**What it opens.** Other native `confirm()`/`alert()` sites (if any) can now migrate to `ConfirmDialog`;
`LibraryDeleteConfirm` could later be re-expressed on top of it.

---
## 2026-07-24 — Taxonomy increment 2b (ADR-028): the Svelte taxonomy view (placement modal)

Built the **renderer** for the taxonomy, increment 2b to the spec `docs/specs/feature-taxonomy-view.md`
(placement grill 2026-07-24). 2a served the field forest + coverage read model; this is the modal that
renders it and *places* concepts/documents onto it. **Frontend only, $0, zero-LLM — nothing in `src/` or
`apps/api/` touched.** `svelte-check` 0/0 · `npm test` **45 pass** (+4) · `docs_check`/`integrity_check` 0/0.

**What.**
- **`lib/taxonomy.ts`** — pure `buildForest(view) -> TaxonomyRow[]`: flattens the field DAG into ordered
  indented rows (root then `child_ids`), guarded by the **ancestor set of the current path** (not a global
  visited set — that would suppress a poly-parented subtree's second expansion). Diamonds terminate
  guard-free; the guard only fires on a corrupt-DB cycle. `lib/taxonomy.test.ts` — 4 node:test cases
  (order+depth, poly-parent expands under both, cycle truncates, empty→[]).
- **`lib/LibraryTaxonomy.svelte`** — a dedicated modal (grill: not a tab in Manage-keywords), reusing the
  scrim+dialog+Esc shell. Left: the 236-field forest with rollup badges + a label search that flattens on
  query. Right: the selected field's detail — removable concept chips, read-only doc rows (2a has no
  doc-detach route), and **attach-concept / attach-document** pickers. Honest zero-state header ("N not
  yet placed"); `focusConceptId` shows a "Placing: <label>" banner + preselects the attach picker.
- **`lib/api.ts`** — 5 self-contained client fns (`getTaxonomy`/`getFieldDetail`/`addHierarchyEdge`/
  `removeHierarchyEdge`/`attachDocumentField`); `removeHierarchyEdge` is the client's first DELETE with a
  JSON body. Error `detail` surfaced via the existing `errorDetail` helper (so the 409 cycle message shows).
- **`App.svelte`** — owns the data (`taxonomyView`/`FieldDetail`/`Concepts`/`FocusConceptId`/loading/error),
  `openTaxonomy`/`closeTaxonomy`, write-then-refetch mutation handlers. Concept-picker vocabulary =
  `getConceptGraph().nodes → {id,label}` (ledger #7). **Entry points:** a **Sidebar** "Taxonomy" rail
  action (beside Manage-folders) + a **graph node "Place"** sibling to Edit (`ConceptGraph.svelte`, new
  `onPlaceConcept`) that opens the modal with the concept preselected (D11). The modal is a top-level
  overlay — opens from any mode, so demoting Graph later can't strand it (grill: fully decoupled).

**Why.** The grill locked 2b as a Library-reached modal, decoupled from the parked demote-Graph nav fork.
The renderer needed live-app verification (screenshots time out on this box — `read_page`+`javascript_tool`).

**⚠ Gotcha (cost me the mount, worth carrying).** An **optional parameter** `focusConceptId?: string` in a
`<script lang="ts">` function compiles to invalid JS: the project's TS-strip drops the `: string` type but
**leaves the `?`**, emitting `function openTaxonomy(focusConceptId?)` → `SyntaxError: Unexpected token '?'`,
which silently fails the whole app mount (blank `#app`, no console error the preview tool surfaced).
`svelte-check` type-checks the *source* and passes it. **Rule: default optional params in svelte `lang=ts`
(`x: T | null = null`), never `x?: T`.** Recorded in `apps/desktop/CLAUDE.md`.

**Live-verified ($0, real 76-doc / 30,882-chunk corpus, DB restored to baseline after):** forest 236
fields / 23 roots, header **26 concepts / 76 docs / 26 not-placed**, all rollups 0 (honest zero-state).
Attaching BM25 to a group ticked that group to **1·0** *and* its division's rollup to **1·0** (set-semantics
crosses the group→division edge); removing reverted both + returned "26 not placed". The 409 cycle guard,
exercised at the API level (a division→group edge closing the seeded loop), threw the backend's cycle
message with **no partial write**. Graph "Place" opened the modal over Graph mode with the concept
preselected. Dark + 375px: single-column, **0 overflow, 0 console errors**.

**Rejected.** Field re-parenting UI (ledger #6 — ANZSRC structure rarely moves; the only cycle-capable
edit, deferred with its 409-UI test). A global-visited forest walk (breaks render-under-both). Wiring the
Library Collections rail (ledger #4 — a later increment).

**What it opens.** Increment 3 — auto-propose `in_field` parents for the 26 unassigned concepts
($0/Ollama, KI-4, RTX box). Later: Collections-rail population; a document-detach route; the field
re-parenting control (+ its cycle-409 UI surface).

---
## 2026-07-24 — Taxonomy 2b spec review: narrowed to placement-only + concept-picker source (docs only)

Code-grounded review of `docs/specs/feature-taxonomy-view.md` (increment 2b) against the shipped 2a
backend + the existing frontend; amendments applied to the spec. **Docs only — no code touched.**

**What.** Four findings, all folded in (grill-ledger rows #6–7 added):
- **Narrowed 2b to placement-only (#6).** The overview promised "edits the hierarchy" but T3 specified
  no field→field re-parenting control — and that is the *only* edit that can trip the 409 cycle guard
  (concepts are never edge targets). Resolution: 2b ships concept→field attach/detach + document→field
  attach; re-parenting stays API-only; DoD 5's cycle-409 check moved to the API level (`javascript_tool`
  → api client), since no UI path can form a cycle.
- **Named the concept-picker source (#7).** DoD required attaching a concept via the UI, but no prop or
  endpoint supplied the attachable vocabulary (2a serves only counts; `FieldDetail` only already-attached
  members). Resolution: feed from `getConceptGraph().nodes` → `{ id, label }`; accepted limitation:
  demoted (`graph_include=false`) concepts aren't offered — reopener recorded. T3 gains
  `concepts`/`focusConceptId` props; T4 gains `taxonomyConcepts`/`taxonomyFocusConceptId` + the graph
  deep-link becomes a `manageConcept` **sibling** calling `openTaxonomy(conceptId)`.
- **Pinned the tree guard as ancestor-path, not global-visited.** T2's tests conflicted otherwise: a
  global visited set terminates but stops a poly-parented subtree expanding under its second parent.
  Test wording fixed too (termination case is a *cycle*; a diamond terminates guard-free).
- **Truth-fix in T1:** `api.ts` has no shared `request`/`fetchJson` helper — fns are self-contained
  `fetch` + throw + typed cast; noted `removeHierarchyEdge` is the client's first DELETE-with-body.

**Why.** DoD 5/6 were unbuildable as written (no cycle-capable UI path; no concept list to pick from);
the guard ambiguity would have produced a test conflict mid-build.

**Rejected.** Building the re-parent control now (ANZSRC structure rarely moves; keeps the modal
simple); a new vocabulary-list endpoint (2a untouched — the graph client fn already serves the 26).

**Opens.** Ledger #6/#7 reopeners: a re-parent control when curation needs it (giving the 409 guard a
UI test); a vocabulary-list endpoint if demoted concepts ever need placement.

---
## 2026-07-23 — Taxonomy increment 2a (ADR-028): the curation backend (read model + read/write API)

Built the **serve + edit backend** for the taxonomy, increment 2a of
[ADR-028](decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md) to the spec
`docs/specs/feature-taxonomy-curation.md`. Increment 1 seeded a disconnected field forest (236 fields,
0 members); this is the read model a UI renders + the HTTP endpoints that attach concepts/documents and
edit the hierarchy. **Backend only, $0, zero-LLM. Staged.** `ruff` · `mypy --strict src` (67 files) ·
`bandit` clean · `svelte-check` 0/0 · `docs_check`/`integrity_check` 0/0.

**What.**
- **`knowledge/taxonomy_view.py`** — the read model: `load_taxonomy_view()` assembles the field forest
  (each `kind="domain"` node with parents/children + direct + rolled-up concept/doc counts) and
  corpus totals; `load_field_detail(id)` drills into one field's direct members. **Rollup is
  set-semantics (ADR-028 D6)** — a field's coverage = the *distinct* set of concepts/docs for which it
  is an ancestor (via `nx.ancestors` over the hierarchy DAG), deduped by id, so a concept under two
  groups of one division counts **once** at the division. Honest zero-state (everything 0 until members
  attach). Pure read, no LLM/network.
- **`apps/api/routers/taxonomy.py`** — thin shell over the `taxonomy.py` seam: `GET /api/taxonomy`
  (forest + coverage; 200 with 0s on an empty forest, not 404 — the trunk is bundled data),
  `GET /api/taxonomy/fields/{id}` (404 for a non-domain id), `POST/DELETE /api/taxonomy/hierarchy`
  (add/remove an edge — **attaching a concept to a field is just an `in_field` edge**, same endpoint;
  409 on a cycle, 404 on a missing id, 422 on a bad type via the Literal), `POST
  /api/taxonomy/documents/{doc}/fields/{field}` (400 `NotADomainError`, 404 no-such-doc). Registered in
  `main.py`; payloads in `models.py`; mirrored in `types.ts` (consumed by 2b).
- **12 new tests** — view-model units in `test_taxonomy.py` (rollup crosses group→division; polyhierarchy
  dedup; document rollup; `field_detail` None-vs-empty) + route integration in
  `tests/integration/test_api_taxonomy.py` (status-code mapping via `create_app(controller=…)`).

**Live on the seeded corpus.** `GET /api/taxonomy` → 200: **236 fields, 23 roots, 26 concepts (26
unassigned — the honest zero-state), 76 documents total**; Machine learning → 1 parent (Information and
computing sciences), 0 rollup concepts. Attach-then-read (in a test) shows a concept on a group rolling
up to its division.

**Why this slice.** The graph shipped its read model first (PR-G1) before any renderer; same order. 2a is
fork-free and verifiable on this box. The **frontend taxonomy view is 2b** — it needs live-app
verification and is where ADR-019 D11's "dedicated view, deep-linked like Manage-keywords" placement lands
(and where the app-shell nav track is settled).

**Rejected.** (a) Building the Svelte view now — bundles a forky, live-verify-heavy frontend into a
backend PR (build protocol: one PR per session). (b) A separate `attach-concept` endpoint — a concept→field
attachment *is* an `in_field` edge, so `POST /hierarchy` already covers it; a second endpoint would be two
ways to write one fact. (c) Deriving coverage from concept mentions — the taxonomy is explicit attachment
(ADR-028 D6), and the 25/47 concept-less docs need `document_field`, which a derived rule can't see.

**What it opens.** 2b (the Svelte taxonomy view); increment 3 (auto-propose `in_field` parents where a
concept is unassigned — all 26 are, today — $0/Ollama, KI-4); coverage-based gap detectors (RG-015). The
CC-BY attribution UI (T4) is still required before user-facing ship.

---
## 2026-07-23 — Taxonomy increment 1 (ADR-028): schema + write seam + full ANZSRC seed + consumer guard

Built the concept-taxonomy substrate, the first increment of
[ADR-028](decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md) to the design-locked spec
`docs/specs/feature-taxonomy-seed-schema.md`, **complete with the full ANZSRC 2020 FoR trunk and
seeded into the live DB**. **$0, zero-LLM, backend + data only. Staged, not committed** (cpc §13).
Full suite **1238 passed** (+17); `ruff` · `ruff format` · `mypy --strict src` (66 files) · `bandit`
clean; `docs_check`/`integrity_check` 0/0.

**What.**
- **T1a `kind` column on `concepts`** (`db/models.py` + `db/migrations.py` `_ADDITIVE_COLUMNS`):
  `"concept"` (text-bearing) | `"domain"` (abstract field node). The migration DDL
  `VARCHAR NOT NULL DEFAULT 'concept'` **backfills every existing row in the same ALTER** — the KI-25
  discipline made explicit (an additive column whose absent value would change behaviour ships its
  backfill in the same change). Indexed in the model *and* the migration so fresh and migrated DBs match.
- **T1b/c two new tables** (`create_all`, additive — no migration): `concept_hierarchy` (the curated
  `is_a`/`in_field` DAG, polyhierarchy-native) and `document_field` (doc→field m2m, `origin`
  curated|proposed). Both live **beside** `Concept`, so a `build_concept_skeleton` rebuild — which drops
  the derived `concept_edges` — cannot wipe them (the KI-17/KI-20 class; the load-bearing guard is test 6).
- **T2 `knowledge/taxonomy.py`** — the sole sanctioned write seam. `add_hierarchy_edge` enforces the
  acyclicity invariant (ADR-028 D3) via a whole-graph `nx.is_directed_acyclic_graph` check, idempotent on
  the unique key; `attach_document_field` validates the target is a `kind="domain"` node; `presence_nodes`
  is the single canonical `kind="concept"` accessor (ADR-028 D4 — the domain-exclusion written once, not
  scattered); `load_taxonomy` returns the read-only `DiGraph`.
- **T3 seeder** — `scripts/seed_taxonomy.py` (dry-run default, `--apply`, prints the CC-BY attribution
  every run) over `data/anzsrc_2020_for.json`. Field nodes key on a **stable UUID5** derived from the
  ANZSRC code (idempotent across runs/machines); group→division links go through `add_hierarchy_edge`.
- **16 guard tests** (`tests/unit/test_taxonomy.py`, the 9 DoD + 5 seam-coverage + 2 consumer-guard),
  each fails against the pre-increment code.

**The full ANZSRC data (23 divisions + 213 groups) — sourced authoritatively, not from memory.** The
user supplied the official ANZSRC 2020 FoR **SKOS/Turtle** (`anzsrc-2020-for-20210429.ttl`,
linked.data.gov.au). A stdlib transform (no rdflib) extracted `skos:notation`/`prefLabel`/`broader` per
concept → the 23 two-digit divisions + 213 four-digit groups (6-digit fields skipped — grafted on demand,
ADR-028 D7), division labels normalised from the source's uppercase to sentence case. **Hard-verified**
before writing: exactly 23 + 213, every group rolls to a real division (`code[:2]`), anchors present
(4602 Artificial intelligence, 4611 Machine learning, 3209 Neurosciences). This replaced the memory/fetch
route — deliberately declined earlier as unverifiable data at scale (the `WebFetch` endpoint 404'd and its
summarizing answer can't guarantee verbatim completeness).

**Consumer guard (ADR-028 D4) — the prerequisite for seeding safely.** Several `select(Concept)` consumers
assumed every row is text-bearing; seeding 236 abstract domains would flood them. Wired the `kind="concept"`
filter into `library.list_keyword_families` (the always-on families UI — would have shown 236 field names),
`concept_skeleton.load_glossary`, `list_keyword_candidates`, `concept_curation.load_concepts` +
`rank_keyword_candidates` (a near-dup merge must never fold a domain into a concept), and the family-rename
clash check. `load_concepts` (graph vocabulary) needed no change — its `graph_include` filter already
excludes domains. Two guard tests pin families/glossary domain-exclusion.

**Seeded into the live DB, verified invisible.** `init_db` (applying the new `kind` migration, backfilling
the 26 existing rows to `concept`) → `seed_taxonomy --apply`: **236 domains** (`source=anzsrc`,
`graph_include=False`) + **213 `in_field` edges**. API-level check on the real corpus: `list_keyword_families`
26, `load_glossary` 26, graph vocabulary 26, `presence_nodes` 26 — all unchanged; `load_taxonomy` sees 262
nodes / 213 edges. A pre-seed backup sits at `data/library.db.bak-pretaxonomy` (gitignored).

**Why now.** ADR-028 accepted, nothing built; the schema is the foundation every later increment (curation
UI, auto-propose, coverage math) writes against.

**Rejected.** (a) Hand-encoding the 213 groups from memory / a summarizing fetch — fails the rigor +
CC-BY-provenance bar; superseded by the user's official TTL. (b) Storing the hierarchy in `concept_edges` —
a rebuild would wipe it (the whole point of the separate table). (c) Seeding before wiring the consumer
guard — would have flooded the families UI and risked curation merge-corruption.

**What it opens.** Increment 2 (curation UI to edit the DAG), increment 3 (auto-propose `in_field` parents
where NULL, $0/Ollama, KI-4), and coverage math (`load_taxonomy` is its substrate). The About/Settings
CC-BY attribution UI is required before the taxonomy ships to users (spec T4). Committing the ~1 MB source
`.ttl` for provenance is the user's call (currently untracked in `data/`).

---
## 2026-07-23 — Concept-system docs consolidation (no code change)

Brought the concept-graph / keyword / taxonomy documentation into a "clean state" after a long,
meandering implementation. **Docs only — zero code, zero data mutation.** Both doc gates green
(`docs_check --strict` / `integrity_check` **0/0**).

**What changed.**
- **New canonical map** in `docs/architecture.md` → *Concept & knowledge system*: the one-page picture
  the feature never had — one `Concept` table with four hats (keyword candidate → concept → keyword
  family → taxonomy node), the **two distinct graph layers** (derived `concept_edges`, rebuilt every
  run, vs curated `concept_hierarchy`, survives a rebuild — the KI-17/KI-20 load-bearing rule), what
  reads the graph (epistemics / gaps / wiki), and the current build state.
- **`GLOSSARY.md`:** fixed six stale `Authoritative in:` paths that still pointed at the pre-ADR-023
  flat layout (`src/doc_assistant/*.py` → `.../knowledge/*.py`); added **C-011 domain/`kind`** and
  **C-012 concept hierarchy** for the ADR-028 taxonomy terms (both flagged *decided, unbuilt*).
- **`docs/specs/feature-concept-graph.md`:** replaced the strikethrough-laden header with a build-state
  table (PR-G2b **delivered by E5** — KI-17 resolved, triage shipped; routes moved to
  `apps/api/routers/concepts.py`); corrected the stale "TWO BOXES" note (this box is now 76 docs / 26
  concepts, not 47/357; labels resolve again; Node-B stance NULL); pointed the spec at the new
  architecture map.
- **`.claude/CONTEXT.md`:** ADR-018 "staged" → committed (G8 done); the "47 docs / 688 kw / 357 concepts"
  open-question bullet corrected to the box's actual **76 / 60 / 26** clean state.
- **`.env.example`:** the wiki-communities comment pointed at the **deleted** `scripts.build_concept_graph`
  and the empty **decoy** `data/graph/graph.json` → corrected to `build_concept_skeleton` /
  `data/skeleton/skeleton.json`.
- **Hygiene:** the two disposable PLAN notes kept `active` (both are still cited as `plan:` provenance
  by specs / ADR-028 / RG-023 — archiving would break those pointers) with completion banners; bumped
  four stale `updated:` headers (decisions.md, ROADMAP, ui-checklist, RIGOR_TODO) the prior commit left
  behind, clearing `docs_check --strict`.

**Why.** The concept system spans ADR-006/008/015/017/018/019/023/028 plus ~6 specs and two PLAN notes;
truth had drifted (deleted-module pointers, a decoy artifact, box-state numbers, pre-split module paths)
and there was no single map. Read-only data inventory (this box, `data/library.db`) confirmed the
vocabulary is **clean** — 26 curated concepts, zero junk to demote.

**Rejected.** (a) A full rewrite of `feature-concept-graph.md` — its verdict/traps/grill-ledger are a
valuable append-only record; targeted truth-fixes + a status table beat a risky rewrite. (b) Archiving
the two superseded PLAN notes per the cpc lifecycle rule — they are live `plan:` provenance for canonical
specs/ADRs; moving them breaks references, so `active` + a banner is the honest state.

**What it opens.** Data-curation decisions surfaced for the user (not actioned): `chunk_epistemics` empty
(markers/E2 strip dark — `compute_epistemics` is $0), Node-B stance regen (LLM cost call, RTX box / KI-4),
`gap_triage` table absent on this DB (predates the E0 model — `init_db` creates it), and the stale
`data/graph/graph.json` decoy (safe to delete). The taxonomy build (ADR-028 increment 1,
`docs/specs/feature-taxonomy-seed-schema.md`) remains the next code sprint.

---
## 2026-07-23 — Retrieval hygiene: scoped-ensemble LRU + reranker-input cap under multi-query

Two cost fixes from the plan's post-E5 "retrieval hygiene" note (`docs/PLAN_2026-07-21_exploration-epistemics.md`),
both in `pipeline.py`. **Staged, not committed** (cpc §13). Full suite **1221 passed / 1 skipped**
(+3 net); ruff · `ruff format` · `mypy --strict src` · bandit. Backend-only, frontend untouched.

**(1) Scoped-ensemble memo: single slot → small LRU.** The folder-scoped ensemble (ADR-025 F2 /
RG-020) was memoised in **one** slot keyed on the scope's hash set, so alternating between two
folders rebuilt the BM25 arm (~20 µs/chunk, measured RG-020) on *every* turn. Replaced with an
`OrderedDict` LRU (`_SCOPED_ENSEMBLE_CACHE_SIZE = 4`): `get`+`move_to_end` on hit, `popitem(last=False)`
past capacity. **Provably non-degrading** — the cached object is byte-for-byte the ensemble a rebuild
would produce; only *when* a rebuild happens changes. `scope=None` (the whole-corpus default path)
never enters the cache. Guard tests: the alternating-scopes-stay-warm case (single slot made it 5
rebuilds; LRU makes it 2) and an eviction case at a monkeypatched size of 2.

**(2) Cap the cross-encoder input under multi-query.** Multi-query (opt-in — `USE_MULTI_QUERY`
defaults false / U1 per-turn override) unions candidates across up to 4 query phrasings, growing the
rerank input — and thus the CPU cross-encoder cost — ~4× unbounded. New `RERANK_CANDIDATE_CAP`
(config, env-overridable, default `CANDIDATE_K * 3` = 60) truncates `all_candidates` before the
`reranker.predict` call. **The single-query default path is byte-identical**: a single query unions at
most `2*CANDIDATE_K` (= 40) across the two ensemble arms (the EnsembleRetriever-returns-full-union fact
already documented at `config.py` BM25_WEIGHT), and the cap is validated `>= 2*CANDIDATE_K` at import,
so it provably never bites there. Candidates accumulate original-query-first with first-seen dedup, so
the truncated tail is the lowest-priority cross-variation hits, never the primary query's. Guard tests:
the default cap leaves a full `2*CANDIDATE_K` single-query pool untouched; a capped multi-query union
keeps exactly the original query's candidates.

**Why / rejected.** *LRU size 4, a named structural constant not a tunable:* it trades a little
memory (each entry holds a BM25 index over its subset) for latency and never affects output, so it is
not eval-gated. *The reranker cap's default multiplier IS a cost/recall tradeoff on the multi-query
path* — filed as **RG-022** (eval-gated), not asserted as an optimum; it ships now only because
multi-query is off by default, so nothing in the shipped default path changes. Rejected: capping
per-query contribution (more code, same effect since dedup already orders original-first); making the
LRU size env-configurable (memory/latency knob, not a quality one — a named constant is honest and
simpler). **Opens:** RG-022 (validate the cap multiplier on the MQ path once a reason to run MQ by
default exists); the scoped-BM25 subset-statistics question (RG-020 part a) is unchanged by this.

---
## 2026-07-22 — APIRouter split of `apps/api/main.py` (pure refactor, behavior-identical)

The refactor the plan deferred through E4 and E5 (a behavior-preserving move doesn't belong inside a
feature diff), now its own increment. `apps/api/main.py` had grown to **1009 lines / 42 routes** in
one file; split into per-domain `APIRouter` modules. **Staged, not committed** (cpc §13). Full suite
**1218 passed** (unchanged from pre-split — the behavior-preservation proof); ruff · `ruff format` ·
`mypy --strict src` (65) · bandit. Frontend untouched.

**Structure.** `main.py` **1009 → 159 lines** — now only `create_app`: the lifespan (schema migration
+ controller), `app.state` wiring + the `ingest_fn`/`rebuild_graph_fn`/`controller_factory` test seams,
CORS, and seven `include_router` calls. Routes moved to `apps/api/routers/{health,chat,conversations,
library,concepts,settings,sources}.py` — one `APIRouter` per domain. Cross-router glue (the `app.state`
status dataclasses + their `202+poll` serializers, the settings read view, the lazy default job
runners) moved to `apps/api/services.py`. **Dependency direction is one-way:** `main` and routers import
from `services`; routers never import from `main` (no cycle). `chat`-only helpers (`_sse`,
`_event_stream`) live in the chat router.

**Behavior-preserving contract.** Every handler already read state via `request.app.state.*` + module
helpers, never a `create_app` local — so the move is `@app.get` → `@router.get`, nothing more. Route
declaration order preserved within each router (the load-bearing one: `/api/concepts/gaps` before the
parameterised `/api/concepts/{concept_id}/…`). `create_app`'s signature is byte-identical (the public
seam — 49 test call-sites unchanged). `_settings_view` + `_default_rebuild_graph` are re-exported from
`main` (via `__all__`) so the two tests importing them by name still resolve; `init_db` stays imported
in `main` so the startup-migration monkeypatch target holds.

**Test churn (2, both legitimate consequences of a moved symbol).** `test_figure_served_and_missing`
patched `apps.api.main.load_figure_image_paths` → repointed to `apps.api.routers.chat` (where the
figure route now imports it). No other monkeypatch targeted a moved symbol (grep-verified).

**Live $0 smoke.** Booted the real app; one endpoint per router returned correctly (health 200, a
provenance-source 404-as-expected, all domain reads 200); real data flows (47 docs / 15 gaps / 16 039
chunks via the split app); 0 server errors. Route enumeration: 42 API routes, same set as pre-split.

**Rejected.** Distributing the status dataclasses across their routers and having `main` import from 4
routers (a shared `services.py` is simpler to reason about + keeps the import graph acyclic); a
top-level `routers/` re-export shim (the explicit `include_router` list in `create_app` is the clearest
manifest). **Opens.** `library.py` (the backend service module, ~1.7k lines) is the other half of the
plan's "split by domain" note — a separate, larger refactor, not bundled here. `apps/` isn't in the
`mypy --strict` gate (only `src`); a `ServerSentEvent` attr-export stub would be needed to add it —
noted, not done.

---
## 2026-07-22 — E5: first-class gap list + triage (ADR-004 / ADR-017 C1) — panel in the Graph view

ROADMAP row E5 (last E-track row). Gaps were computed (ADR-004) but reachable only *embedded* in the
graph payload, joined to nodes — no first-class list, no triage. Now: a triageable gap list in the
Graph view with a durable dismiss/promote override. **Staged, not committed** (cpc §13). Full suite
**1218 passed** (+10 backend); ruff · `ruff format` · `mypy --strict src` (65) · bandit · **svelte-check
0/0 (140)** · **npm test 41/41** (+7 gaps.ts) · docs 0/0 · integrity 0/0.

**The load-bearing decision (ADR-017 C1): triage is a user override in its OWN sidecar.** New
`GapTriage` table keyed on `(concept_id, kind)` — *not* `GapRow.status`. Deterministic `gaps` rows are
delete-and-replaced on every `build_gaps` run (regenerable, ADR-004), and a rebuild is part of the
acquire loop the surface exists to close, so a dismissal written onto the row would not survive it.
`load_gaps` now resolves the **effective** status = `override ?? row.status` (one source of truth — the
graph node-badge lens and the list agree; a dismissal removes the gap from both). Mirrors `DocumentMeta`
(ADR-013). Additive table via `create_all`, no migration. `set_gap_status(surfaced)` = reset = delete the
override. Stochastic rows keep their own persisted status when un-overridden (C1's "don't double-write").

**Backend/API.** `load_gap_overrides` / `set_gap_status` (gaps.py); `load_gap_list` +
`GapListItem` (concept_graph_view.py) resolves the concept UUID → label (the flat list needs it; the
graph carries labels only on nodes, KI-15) with a fallback to the id itself for stochastic candidates.
`GET /api/concepts/gaps` (label + effective status; empty when unbuilt) + `POST /api/concepts/gaps/triage`
(`{concept_id, kind, status}`; `surfaced` resets; the `Literal` enum 422s a bad status).

**Frontend.** Extracted the gap taxonomy to a **pure, node-tested** `lib/gaps.ts` (GAP_META + rank/tone
+ `orderGaps`/`gapVisible`) — shared by ConceptGraph (which now imports it instead of an inline copy) and
the new `GapList.svelte`. ConceptGraph gains a rail-mode toggle **Concepts | Gaps**; `visibleGaps` now
drops `dismissed` gaps from the node lens too. `GapList` is **self-contained** (fetches its own
effective-status data, owns its triage writes) so it can move out of the graph without coupling when the
Graph-destination fork settles — the recorded iteration gate. RG-014 presentation preserved: strong
list-shaped kinds first, `under_connected` opt-in; dismissed hidden (recoverable via a "Show dismissed"
toggle); Promote/Dismiss/Reset per row; `onSelectConcept` jumps the ego view.

**Live $0 verify (real corpus, 16k chunks).** API: 15 gaps served with labels + effective status.
Triaged a real UUID-keyed `single_source` gap → dismissed → **triggered a real graph rebuild (202+poll,
deletes+rebuilds the deterministic rows) → the dismissal SURVIVED** (C1's whole point, proven end-to-end).
Reset deletes the override. `bogus` status → 422. UI: rail toggle renders; Dismiss dropped the row from
the worklist (15→14 open) and surfaced a "Show dismissed" toggle; the toggle reveals it with a dismissed
tag + Promote/Reset; Reset restores it; a concept link jumps the ego view to that concept (SVG rendered).
0 console errors; 375px 0-overflow; dark tokens resolve. Box left clean (0 overrides).

**Rejected.** Storing triage on `GapRow.status` (dies on the rebuild — the exact B14 trap the grill
originally got wrong); a second corpus-wide graph surface (RG-014: the gap payload is list-shaped, and the
Graph-destination fork is parked); the plan's `APIRouter` split of `api/main.py` "with E5" — a
behavior-preserving ~200-line refactor doesn't belong inside a feature diff; owed as its own increment
before the next route-heavy work.

**Opens.** The `APIRouter` split (own commit). Triage currently has no bulk action; the gap→acquisition
loop (B13, own ADR) attaches to `promoted` later. `suggested_concept` volume swings with `gap_suggest`
runs — the list shows them but leads with the deterministic strong kinds.

---
## 2026-07-22 — E4: document-connections panel (ADR-027 D1) — the exploration surface, per-doc

ROADMAP row E4. The plan's headline gap closed: `doc_similarities` (470 edges, all 47 docs) and
the citation graph (2,859 extracted / 9 resolved in-corpus) were computed and reachable by **no
endpoint** — dead to the UI. Now: one endpoint + one panel in the Library document view.
**Staged, not committed** (cpc §13). Full suite **1208 passed** (+9); ruff · `ruff format` ·
`mypy --strict src` (65) · bandit · **svelte-check 0/0 (138)** · **npm test 34/34**.

**Shape (user decision, 2026-07-22): per-doc panel, NOT a top-level network mode.** At 9 resolved
citation edges a corpus-wide network view renders near-empty (the exact "empty Graph reads as
broken" complaint), and it would entangle the parked Graph-destination fork (ADR-025 fork 5).
**The graph/navigation treatment stays an OPEN GATE, per the user** — v1 is deliberately
list-shaped, and a depth-1 ego graph is exactly `cites` + `cited_by`, so a later iteration
(SVG ego view, corpus-level view, depth param) reads the same bundle without an API break.
Recorded in ui-checklist.

**Backend.** `library.document_connections(doc_id, related_limit, external_cap, embedding_model)`
→ `DocConnections | None` — assembles the *existing* read models (`similar_docs` + `cites_out`
split internal/external + `cited_by` deduped by source doc with `n_citations`); None = unknown doc
(404). The similarity read is scoped to the active embedder (edges from another model describe a
different geometry — never mixed). `external_refs` = the titled slice of unresolved citations,
capped at `EXTERNAL_REFS_CAP=50` (a wire-size bound, not corpus-tuned) with `external_total`
alongside — no silent truncation. Empty sidecars → empty lists (0-doc contract).

**Wire + frontend.** `GET /api/library/documents/{id}/connections` → `DocConnectionsPayload`
(4 sub-payloads); `types.ts` mirrors. New `DocConnections.svelte` in `LibraryBrowser` under the
doc header: Related papers (cosine chip) · Cites / Cited by (in-library links, ×n badge) ·
collapsed "References (N extracted, not in your library)" with a showing-N-of-M cap note.
Advisory: load failure degrades to one quiet line; all-empty bundle renders one honest muted
line. Click-through = `onOpenDocument` threaded from App.svelte — the D1 hop (doc → related doc,
panel reloads) is the feature.

**Live $0 verify (real corpus).** DPR doc: all 4 sections (10 related · cites → dpr/rag/sbert
set · cited-by · "References (33 extracted)"); clicked the top related link → doc view swapped to
the RAG paper, panel reloaded (13 links) — the exploration loop works. b2a1754a via API: 10
related (top specter2@0.952), 3 in-corpus cites, 50-of-60 external cap, 404 probe clean. 0 console
errors; 375px 0-overflow (also with the refs list open); dark-theme tokens resolve.

**Rejected.** Top-level citation-network mode (near-empty at 9 edges + entangles the parked
Graph-destination fork); the APIRouter split of `api/main.py` before adding the route (the plan
suggests it "before E4/E5" — one ~15-line route doesn't justify a ~200-line refactor riding the
same diff; owed before E5's larger route surface, see Opens); returning a separate `graph` payload
(depth-1 ego ≡ cites+cited_by — derivable, YAGNI).

**Opens.** The graph/navigation iteration gate (user, explicit). E5 (gap list) next — do the
`APIRouter` split with it. Citation resolution is thin (9 edges): re-running
`scripts/extract_citations.py --apply` after corpus growth (or improving the resolver) would
enrich the panel for free. External refs later feed the B13 gap→acquisition loop (own ADR).

---
## 2026-07-22 — E3: persisted epistemics answer-layer toggle (ADR-027 D2) — full-stack

ROADMAP row E3 (ADR-027 D2 — the last unbuilt half of the surfacing split). Whether epistemics
*influences* the answer layer (the marker chips) is now a persisted user setting, layered as a
**three-layer resolution**: U1b per-turn override > persisted setting > `EPISTEMICS_MARKERS_ENABLED`
env default. The effective value is snapshotted per turn into `AnswerRecord` (ADR-011 instrument
discipline). **Staged, not committed** (cpc §13). Full suite **1199 passed** (+17); ruff ·
`ruff format` · `mypy --strict src` (65) · bandit · **svelte-check 0/0 (137)** · **npm test 34/34**
· docs_check 0/0 · integrity_check 0/0 (492).

**Backend.** `app_settings.get/set_markers_enabled` + `effective_markers_enabled()` (the
`effective_llm` pattern — persisted if set, else config; re-read each turn so a toggle applies
next-turn, no restart). `_resolve_turn_knobs` baselines the markers knob on the persisted-effective
default; `_overrides_note` now takes `markers_default=` and compares against **it**, not the env
constant — a persisted choice is the user's default, and stamping it "Session override (this answer
only)" on every turn would be a provenance lie (the note fires only on a genuine per-turn U1b diff).
New additive nullable `answer_records.epistemics_markers_enabled` (`_ADDITIVE_COLUMNS`; NULL = pre-E3
row = honestly "unknown"), recorded on **both** result paths (AI + human-mode) via
`_ProvenanceInputs.markers_enabled` / `record_answer(epistemics_markers_enabled=)`, read back in
`AnswerProvenance`. Deliberately NOT folded into `prompt_version` (same rule as the F2 scope: it
never reaches the prompt).

**Wire + frontend.** `SettingsUpdate += epistemics_markers_enabled` (validator: the toggle alone is a
valid body; empty body still 422). `_settings_view` serves the **effective** value (the raw constant
would go stale on the first toggle — the provider/model rule), which also makes the U1b sandbox
baseline correct for free. New "Answer epistemics" Settings section (persisted toggle, snaps back on
error) between Provider and RAG sandbox; `api.ts setMarkersEnabled`; `types.ts` meaning-shift note.

**Tests (+17).** Precedence unit tests (persisted beats config both directions; non-bool stored value
degrades to "never set"); knob-layering tests incl. the two provenance-honesty pins (persisted-off →
`overrides_note == ""`; override-vs-persisted diff reads `(default False)`); provenance round-trip
(False recorded / legacy row → None); API round-trip (POST persists → GET serves effective; toggle-only
body valid); the D3 boundary from the persisted side (persisted-off still attaches the evaluation
strip); + an AnswerRecord snapshot assertion on a real (fake-RAG) turn. **Test-infra hardening:** 4
files gained an autouse `SETTINGS_PATH` isolation fixture — `_resolve_turn_knobs`/`_settings_view` now
read the persisted setting, so an unpatched suite would read the dev box's real `settings.json` (the
KI-22 class of environmental mislabel, preempted). Existing tests patching the dead
`chat_controller.EPISTEMICS_MARKERS_ENABLED` namespace copy were repointed at `config.` (the layer the
resolution actually reads).

**Live $0 verify (real corpus, ollama/llama3.1:8b — KI-4 honored, provider checked first).** Settings
UI: new section renders (light, 375px → 0 overflow, 0 console errors); toggle off → `settings.json`
gains `epistemics_markers_enabled: false` → GET serves False → the sandbox baseline follows; sandbox
override flips session-only (file untouched — the boundary held live). One real SSE turn (step → 79
tokens → result → done) recorded `epistemics_markers_enabled = 0` on the newest `answer_records` row.
Boot migration verified live: `schema_migrated_at_startup` added the new column (plus, notably,
`retrieval_scope_json` + `chunk_key` — this box's real DB had never received the E1.1/F2 columns; the
KI-23 in-app migration caught all three). State restored after verify (key removed → "never set");
side effect: one `e3-verify` conversation in this box's history.

**Fix in passing.** `.env.example`'s `EPISTEMICS_MARKERS_ENABLED` block claimed "default off (R7)" —
stale since the KI-7 retirement flipped the default to true; rewritten to state the real default + the
three-layer resolution + the D2/D3 boundary. ROADMAP E0–E2 status cells trued up to their commit
hashes (they read "staged" but the user committed them 2026-07-21).

**Rejected.** Folding the flag into `prompt_version` (pollutes eval joins; never reaches the prompt);
an unset/revert-to-default API affordance (YAGNI — the UI only sets true/false, and "never set" is
recoverable by deleting the key); gating the D3 strip (ADR-027's boundary is explicit).

**Opens.** E4 (exploration surfaces) / E5 (gap list) per the plan's sequence; RG-019 still deferred
(measurement-gated, moot at 0 stance); Node-B stance regen on the RTX box to make the chips + strip
non-trivial on real data.

---
## 2026-07-21 — E2: always-on source-evaluation strip (ADR-027 D3) — full-stack

Spec: `docs/specs/feature-e2-source-evaluation-strip.md` (ROADMAP row E2 · ADR-027 D3). A per-source
evaluation strip below every chat answer — always-on, $0 (sidecar lookup joined against TOP_K, no
LLM). Honest to build now: E1.1 made the marker join trustworthy, E1.2 gave the source path clean
seams. **Staged, not committed** (cpc §13). Full suite **1182 passed / 1 skipped** (+6); ruff ·
`ruff format` · `mypy --strict src` · bandit · **svelte-check 0/0** · **npm test 34/34** · docs 0/0 ·
integrity 0/0.

**The boundary (ADR-027).** D3 (this strip) is **always-on** assessment; D2/E3 (the answer-influence
toggle over `eff_markers_enabled`) governs the *answer-surface* marker chips only and **never hides
the strip**. So the strip's per-source `evaluation` attaches unconditionally; the existing `markers`
field stays gated by the toggle — both derived from **one** scoped sidecar read.

**Backend (E2a/E2b).** `epistemics.load_source_evaluations(chunk_keys)` — a scoped, indexed read
(unlike the full-scan marker index; KI-18) returning per-key `ChunkEval(coverage, superseded,
n_claims)` + the sidecar `graph_version`; `coverage` = contested > corroborated > unique.
`current_graph_version()` (a 1-row `concept_presence` read) drives the freshness compare;
`library.document_years(ids)` a scoped year join. `_attach_markers` → `_attach_source_evaluation`
(always-on): sets `sv.evaluation` + `sv.reranker_score` for every source, sets `sv.markers` only when
`markers_enabled`, returns `SourceEvalSummary(graph_version, stale)`; returns `None` (no strip) when
no concept graph is built (0-doc/fresh). WARNING-logged on failure (never a silent lying UI).
`SourceView` gains `evaluation`/`reranker_score`; `TurnResult` gains `source_eval`.

**Wire (E2c).** `SourceViewPayload += evaluation (SourceEpistemicsPayload) + reranker_score`;
`TurnResultPayload += source_eval (SourceEvalSummaryPayload)`; `types.ts` mirrors both. Replay
(`ConversationSource`) stays degraded (no strip).

**Frontend (E2d).** New `SourceEvaluation.svelte` below the answer: a per-source row — a colour-coded
coverage chip (contested=warn, corroborated=ok, single-source=neutral, none=muted "not assessed"), a
`superseded` badge, doc year, rerank score — and a footer "assessed as of `{graph_version}`" with a
**stale** warning. Renders nothing when `source_eval` is null (honest degrade). Wired into `Turn.svelte`.

**Tests (E2e).** The marker-attach path changed (`load_epistemics_index` → `load_source_evaluations`),
so the marker tests were rebuilt around a `_stub_source_eval` helper (ChunkEval fixtures). New D3
guards: `test_d3_strip_always_on_even_when_markers_disabled` (the boundary — evaluation attached while
markers gated off), coverage-derivation + "not assessed", freshness-stale. Turn-parity byte-identical
preserved (strip no-ops with no graph). Note: turn tests **without** `temp_db` must now stub the strip
reads (D3 reads always) — else they'd hit the real DB via `current_graph_version()`.

**Live $0 verify** (real API for init endpoints + a `window.fetch` `/api/chat` SSE mock — fake
sources, no paid turn). The strip rendered below the answer: `[1] contested·2019·0.91`, `[2]
corroborated·2023·0.88`, `[3] single-source·superseded·2011·0.85`, `[4] not assessed·0.70`, with a
**stale** badge + "assessed from an earlier graph (b59a4aa6)" footer. **0 console errors**; coverage
chips resolve to distinct tokens in **light + dark** (contested amber, corroborated green, superseded
red); **375px → 0 horizontal overflow**.

**Opens.** RG-019 (a `contested` denominator) still deferred — measurement-gated, and moot at 0
Node-B stance on this box (the strip is honest-uniform, not saturated). **E3 / D2** (the persisted
answer-influence toggle) is the next ADR-027 row. Node-B stance regen (to make the strip's assessment
non-trivial on real data) still needs the RTX box (KI-4).

---
## 2026-07-21 — E1.2: extract `_handle_rag` into named seams (pure refactor, no behavior change)

ROADMAP row E1.2 (the code-health half of E1, deferred from E1.1). The AI-turn generator
`chat_controller._handle_rag` had grown to **~278 lines / ~14 responsibilities**; the plan calls for
breaking it up *before* E2/E3 wire the always-on epistemics strip into it. **Pure refactor — byte-identical
behavior.** **Staged, not committed** (cpc §13). Full suite **1179 passed / 1 skipped** (+3 unit tests);
ruff · `ruff format` · `mypy --strict src` · bandit · docs_check 0/0 · integrity_check 0/0.

**Approach.** `_handle_rag` is a *generator* (yields `Step`/`Token`/`Result`), so the yield-bearing
flow stays in the orchestrator; only the **non-yielding computation** is extracted. Three seams, each
a verbatim lift:
- `_resolve_turn_knobs(overrides) -> _TurnKnobs` — the ADR-010 effective-knob resolution (top_k /
  synthesis_mode / multi_query / markers_enabled / reviewer_evidence_chars + `overrides_note`). Preserves
  the subtlety that retrieval passes the **raw** `overrides.use_multi_query`, while the *effective*
  `multi_query` is for the provenance note only.
- `_capture_provenance_and_review(_ProvenanceInputs) -> _ProvenanceOutcome` — the 88-line
  provenance-record + confidence-signals + conditional LLM-reviewer + card-format block, try/except-bounded
  exactly as before (a failure still collapses to a "Provenance capture failed" card + empty `record_id`).
  Inputs bundled in a frozen `_ProvenanceInputs` so the seam is single-argument; the `overrides_note`/
  `scope_note` suffix stays in the caller (it owns the turn knobs).
- `_build_claims_block(record_id, full_answer, retrieved_chunks)` — the Chunk-2a segment→persist→render
  block; the caller keeps its `if record_id is not None` guard.

**Result.** `_handle_rag` **278 → 198 lines**; the three heaviest concerns are now named, testable seams
(and E2's source-evaluation strip has clean spots to slot into — `_build_source_views` + the TurnResult
assembly). Stopped here deliberately: extracting the yield-bearing export/TurnResult tail would need a
~20-field parameter bundle that hurts readability more than it helps.

**Verification.** The safety net is the existing `test_turn_parity` (byte-identical when markers absent)
+ the full `test_chat_controller` suite — all green, unchanged. Added 3 focused unit tests pinning
`_resolve_turn_knobs` (defaults / all-None-overrides ≡ defaults / applies + notes the diff). No KI, no
live probe (no data path touched). **Opens:** E2 (ADR-027 D3 always-on source strip) — now honest to
build on (E1.1) *and* has clean seams to wire into (E1.2).

---
## 2026-07-21 — E1.1: marker-join trustworthiness — KI-8 re-projection (correctness core)

Spec: `docs/specs/feature-e1-marker-join.md` (ROADMAP row E1). The honesty prerequisite for ADR-027's
always-on source-evaluation strip (E2): the 7d marker chip — the join E2 renders — silently
under-reported by **~40%** in the default parent-child retrieval mode. **Correctness core only**; the
`_handle_rag` extraction (E1.2) is a separate refactor, deferred. **Staged, not committed** (cpc §13).
Full suite **1176 passed / 1 skipped**; ruff · `ruff format` · `mypy --strict src` · bandit ·
docs_check 0/0 · integrity_check 0/0.

**The defect (KI-8).** `chunk_epistemics` was keyed on the **baseline** segmentation
(`{doc}:{chunk_index}`). In default PC mode retrieval returns **parents** (`parent_index`, never
`chunk_index`), so `_chunk_key` returned `None` and `_attach_markers` fell back to
`markers_for_parent` — a strict **substring-containment** test. A 1000-char baseline chunk cannot be a
substring of a parent it only partially overlaps (200-char overlap), so a marked chunk straddling a
parent boundary was contained in *neither* parent and its markers vanished (review WE-7 — systematic
false *negatives*, not fail-safe over-attribution).

**The fix (KI-8 option 2 — re-projection).** Project the node weights **directly onto the PC parent
segmentation** with the same structural attribution the baseline projection uses, keyed
`{doc}:p{parent_index}` (the ADR-4 composite `concept_skeleton.load_presence_inputs` already builds).
The PC join is now a **direct key lookup** — the coarse containment path is retired.

**What.**
- **E1.1a — schema.** Additive nullable `chunk_key` VARCHAR on `chunk_epistemics` (+ `_ADDITIVE_COLUMNS`,
  indexed): the authoritative, segmentation-agnostic join key. The regenerable table fills it on the
  next `compute_epistemics --apply`; `load_epistemics_index` falls back to `{doc}:{chunk_index}` when
  it is NULL, so a migrated-but-not-recomputed DB still joins flat rows (parent rows arrive on
  recompute) — a clean transition, no hard backfill.
- **E1.1b — projection.** `ChunkEpistemics.chunk_key` is now a stored field (was a derived property);
  `project_chunk`/`project_chunk_weights` carry it. `load_doc_chunks` yields baseline keys; new
  `load_pc_parent_chunks` yields `{doc}:p{parent_index}` (mirrors `load_presence_inputs`).
  `build_epistemics` projects `load_doc_chunks() + load_pc_parent_chunks()` — both segmentations, one
  attribution rule. Retired `markers_for_parent` / `load_marked_chunks` / `MarkedChunk` /
  `_load_baseline_texts` (no remaining consumer).
- **E1.1c — controller.** `_chunk_key` returns `{doc}:p{parent_index}` for a PC parent; `_attach_markers`
  joins **both** modes on `sv.chunk_key` against a single index (loaded once), dropping the containment
  branch and the now-unused `scored` arg. The blanket `except` gains a **WARNING log**
  (`attach_markers_failed`) — advisory markers still never break a turn, but under an always-on strip a
  silent failure is a silently-lying UI, so it must be observable.

**Guard tests (each fails against pre-fix code).** `test_reprojects_onto_pc_parents_keyed_by_parent_index`
(a parent's `{doc}:p{idx}` key carries the marker — fails today: nothing projected onto parents);
`test_chunk_key_parent_child_chunk_uses_parent_key` (inverts the old `…_is_none`);
`test_markers_pc_join_via_chunk_key` (direct join, no containment); `test_marker_load_failure_…_but_warns`
(the WARNING fires — asserted via a fake logger, not `capture_logs`/`caplog`, which both hinge on the
global structlog→stdlib bridge being already configured and so flake across the suite);
`test_project_chunk_carries_pc_parent_key`. Updated `test_compute_epistemics` (stub `load_pc_parent_chunks`,
4-tuple chunks, + the re-projection case), `test_turn_parity`, `test_epistemics` (retired containment
tests + new signatures), and the E0.4 empty-input test (stub the new loader).

**Live $0 probe** (isolated copies of the real 76-doc DB + skeleton; real Chroma read only; originals
byte-unchanged). This box has no Node-B stance, so one contested stance was injected to make a marker
exist, then the **real** re-projection ran: 11961 baseline + 5617 PC parents loaded → the marker index
now carries **196 PC-parent keys (was 0)**; a retrieved parent's `_chunk_key` resolves directly against
it; and **28 of 196 marked parents (14% on this single-concept sample) would have been left unmarked by
the retired containment** — the systematic false-negative direction KI-8 describes, on real data.

**Opens.** E1.2 (`_handle_rag` extraction, ~287 lines) before E2/E3 wire into it. Marker *quality*
(RG-019 `contested` denominator; Node-B stance regen on the RTX box) is unchanged — E1 fixes the
*join*, not the *data*. The old PR-M1 ADR-1 (containment) and the `feature-7d` "deferred live surfacing"
note are now superseded by the direct-key join.

---
## 2026-07-21 — E0 correctness batch: five P0 fixes before the always-on epistemics surfaces

Spec: `docs/specs/feature-e0-correctness-batch.md` (from `docs/PLAN_2026-07-21_exploration-epistemics.md`
§E0 + the C4 review's P0 list). ADR-027 D3 makes the epistemics **assessment** always-on — *an
always-on strip must not show false data* — so this closes the "a rebuild/curation silently destroys
curated state" class (three of these are the same shape as KI-25) plus the boot + zero-doc footguns,
before E1/E2 wire the surfaces. Backend-only, deterministic, **zero LLM / zero eval ceremony** (no
locked setting touched). Every item ships a **guard test that fails against the pre-fix code**; the
full suite is green (**1178 passed / 1 skipped**, +14), `ruff`/`ruff format`/`mypy --strict src`/
`bandit`/docs_check/integrity_check all clean. **Staged, not committed** (cpc §13). Build order was
E0.4 (safety net) → E0.1 → E0.2+E0.3 → E0.5a → E0.5b.

**E0.4 — zero-doc honesty, pinned by a test (WE-1/WE-9/GP-7).**
- *What.* `wiki.load_doc_graph` catches `OperationalError` → `([], [])` + a hint; the `epistemics`
  build guards its sidecar write (a never-migrated DB has no `chunk_epistemics` table, so the
  delete-all in `_write_rows` tripped `OperationalError`) → honest empty result + hint, `applied`
  reflects it. Missing-skeleton still raises `FileNotFoundError` (kept — the CLI + existing test rely
  on it). New parametrized `tests/integration/test_empty_input_honesty.py` over all four build paths
  (wiki/epistemics/skeleton/gaps).
- *Why.* The `.claude/CONTEXT.md` "degrade honestly at 0 documents" contract survived by habit; two
  build paths crashed and nothing gated it. Non-vacuous: wiki/epistemics raise on a never-migrated DB
  today (proven by a raw `OperationalError` repro).
- *Rejected.* Making missing-skeleton return empty too (breaks the CLI contract + `test_missing_
  skeleton_raises`); skip-on-empty for the epistemics write (would stop clearing stale rows on a real
  corpus) — instead the `OperationalError` catch degrades only the genuinely-unmigrated case.

**E0.1 — curation demotes, never deletes (KI-20 / CS-5).**
- *What.* New `concept_curation.demote_concepts(ids)` (`graph_include=False`, keeps row + aliases +
  ADR-015 keyword family) + `apply_plan(plan)` seam the runner now drives. Artifact + `classify_noise`
  verdicts route through demote; `remove_concepts` is kept as the **reserved** explicit-deletion
  primitive, no longer wired to the noise classifier. `CurationPlan.remove_ids` → `demote_ids`.
- *Why.* `classify_noise` is exactly the stage that mislabels specialist vocab (`cre`/`dbs`/`ntsr1`/
  `pddl`), and a delete cascades the keyword family + presence/edges/gaps — irrecoverable. ADR-018's
  demote verb, applied. Guard `test_noise_verdict_demotes_and_keeps_the_family`; `remove_concepts`'
  delete is covered too so the distinction is tested, not commented.
- *Rejected.* Looping `set_graph_include` per id (N sessions) — a single bulk update instead;
  changing the near-dup *merge* to demote (a merge folds aliases into the survivor first, no
  vocabulary lost — left as-is, out of the DoD).

**E0.2 — rebuild reconciles orphaned stochastic gaps (KI-17 / GP).**
- *What.* `gaps._reconcile_stochastic_gaps(live_ids)` deletes stochastic `GapRow`s whose anchor
  `concept_id` left the `graph_include`-filtered vocabulary, **hoisted to run unconditionally on
  every `build_gaps --apply`** (the review's placement correction — inside the suggest branch it never
  reached a deterministic-only apply). `GapsResult.n_reconciled` + CLI report line for transparency.
- *Why.* Stochastic rows were status-preserving upserts with no delete pass → immortal orphans (the
  live 27-gaps-over-13-nodes symptom). A reconcile, not a blanket delete: a promotion on a *live*
  concept survives. `suggest_for_thin` always anchors on an existing concept (target lives in
  `evidence`), so a live suggestion is never a false orphan. Guard `test_orphaned_stochastic_gap_is_
  reconciled_away`; **updated** `test_stochastic_rows_survive_a_deterministic_rebuild` to anchor on a
  live concept (its old synthetic non-vocab anchor is precisely the orphan the reconcile now reaps).
- *Rejected.* `notin_([])` bulk SQL delete (empty-IN edge cases) — load + filter in Python (gaps are
  tens of rows).

**E0.3 — in-app rebuild refreshes gaps, not just the skeleton (KI-21 / GP-4).**
- *What.* `_default_rebuild_graph` chains `build_gaps(apply=True, min_degree=derive_min_degree(skeleton))`
  after the skeleton build. `gaps.derive_min_degree` = runtime **Q1 of the rebuilt skeleton's
  connected-node degrees** (no literal; fails safe to 1 on a tiny graph). Guard `test_rebuild_
  refreshes_gaps_and_drops_stale_ones` (route composition: served gaps == fresh recompute, stale gap
  gone).
- *Why.* The acquire loop the button exists to close (gap → ingest → rebuild → gap closes) never
  closed in-app — the view served gaps from the previous skeleton, incl. the one just closed.
  Derived `min_degree` measured **3** on the real 26-node graph, matching the CLI's validated Q1
  baseline — the "no corpus-tuned literal" contract, satisfied by derivation.
- *Rejected.* A hardcoded default (a corpus-tuned magic number, `.claude/CONTEXT.md`); stamping
  `graph_version` on gap rows + filtering in the view (heavier; the refresh already makes the served
  set correct).

**E0.5a — a failed startup migration fails the boot.** The lifespan (`apps/api/main.py`) re-raises on
`init_db()` failure with a clear message instead of swallowing and serving a half-migrated schema —
KI-23 moved `init_db` here precisely because a stale **answer-path** column 500s every turn, a worse
and later failure than refusing to start. Deliberately reverses the old "never let a migration
problem stop the app" comment (documented in-code). Fixed the stale `apps/api/CLAUDE.md` line ("the
API does not `init_db()` on startup"). Guards in `test_api_startup_migration.py` (boot fails on a
broken migration; boots clean on a good one).

**E0.5b — a plain rebuild preserves Node-B stance.** `build_concept_skeleton(apply=True)` without
`--enrich` recomputes structure only, so it now re-attaches existing Node-B `stance_by_doc`/`relation`
(+ the `llm_relation` provenance token and the weight that follows) to edges whose concept-pair still
exists (`_reattach_stance` / `_load_existing_stance`, `stance_loader` DI seam) instead of wiping it —
the G6-run footgun that darkens corpus-wide epistemics on every in-app rebuild. Transparent to
`--enrich` (which re-derives every edge's stance, setting `()` on the ones it skips), so it only
protects the plain path. Updated the `scripts/CLAUDE.md` footgun note. Guard `test_plain_rebuild_
preserves_node_b_stance`. *Known bound (documented):* a stance entry for a since-removed document
lingers until a real `--enrich`; preserving stale-but-mostly-right stance beats wiping all of it.

**Live $0 probe (isolated copies of the real 76-doc library + 26-node skeleton; real Chroma read
only; originals verified byte-unchanged).** This box's graph is currently clean (0 stochastic gaps,
0 stance), so the probe injected the exact condition each fix protects, then exercised the real data
paths: **E0.5b** — injected stance survived a plain rebuild (26 nodes/70 edges) in both `skeleton.json`
and `concept_edges`, while a `stance_loader`-empty run (pre-fix simulation) wiped it; **E0.2** —
derived `min_degree=3`, 1 orphan reaped, 1 live-anchored promotion kept; **E0.3** — served
deterministic gaps (11) == fresh recompute (11), an injected stale gap dropped.

**Deferred / opens.** E1 (KI-8 marker re-projection + `_handle_rag` extraction) is the next sprint;
KI-18 scale hot-paths + KI-19 tuned constants stay measurement-gated (RG-016..019) — not touched here.
The near-dup merge still hard-deletes the folded row (correct: aliases move first). ADR-017 C1's
gap-triage override sidecar can inherit the E0.2 reconcile seam when PR-G2b lands.

---
## 2026-07-21 — App-shell polish: global search overlay + collapsible sidebar (chat-first shell, a+b)

The two **shovel-ready** sub-items of the "App shell → chat-first layout" backlog row
(`docs/ui-checklist.md`, 2026-07-20). Frontend-only, no backend, no wire-type change. The row's
third clause — *demote Graph out of the top-level nav* — is **deliberately not built**: where Graph
goes is an open design fork (empty-Graph → per-folder-concepts, ADR-025 fork 5) the baton says to
`grill-me` first. Spec: `docs/specs/feature-app-shell-search-collapse.md`.

**What.** (a) A **global-search overlay** — scrim + centred dialog (reuses the LibraryKeywordFilter
modal shell), opened from a new header button **and Cmd/Ctrl-K**. It searches conversation titles +
document title/filename/authors/keywords, groups the hits (Chats, then Documents, each capped at 8
with an honest "+N more"), and jumps to the chosen one. Empty query shows up to 6 recent chats.
Keyboard-first: autofocus, ↑/↓/Enter/Esc, mouse hover shares the same highlight. (b) A **collapse
toggle** in the header that hides the rail on desktop and brings it back at its persisted width.

**Why (the two decisions worth naming).**
1. **It is a *navigation* search, not a corpus search** (spec A1). The composer *is* the corpus
   surface; a second box that looked like retrieval but wasn't would be exactly the integrity lie the
   product avoids — and message bodies / chunk text aren't client-side anyway, so searching them
   needs a backend this row excludes. Placeholder + empty state say "chats and documents" so the
   scope reads honestly.
2. **The trigger lives in the header, not the sidebar** (A2). The common sidebar-search convention
   would put it in the rail — but the rail can now be collapsed (b), which would hide the search
   entry point exactly when you collapse. Header + Cmd/Ctrl-K is always reachable.

**Design notes.** Match logic is a pure, tested module `lib/search.ts` (`searchEverything`) —
`npm test` gate, house pattern since PR-2.5; the overlay is a dumb renderer, App owns the data +
navigation (reuses the existing `openConversation`/`openDocument` entry points, so no new nav logic).
`search.ts` is deliberately **self-contained** (no runtime import of `library.docLabel`): node's
test runner strips the type-only `./types` import but can't resolve an extensionless *value* import,
which is the same constraint that kept `library.ts` value-import-free — a real gotcha for the next
tested module. Collapse is a client-only pref in `localStorage` (theme/width precedent), desktop-only
under a `min-width:721px` guard so the mobile off-canvas drawer is untouched; `.app.collapsed
:global(.sidebar)` reaches the child component's root.

**Rejected.** Searching chat/chunk content (needs a backend — out of scope + blurs into the
composer). A sidebar-hosted trigger (collapsing hides it). Driving collapse via `--sidebar-width: 0`
(leaves the border + a live resizer; `display:none` is cleaner).

**Verified.** `svelte-check` 0/0; `npm test` **34/34** (23 existing + **11 new** in `search.test.ts`).
**Live on the real 76-doc corpus ($0, no LLM turn fired):** header button + Ctrl-K both open the
overlay (Ctrl-K toggles closed, autofocuses); empty → 6 recent chats; "retrieval" → 2 chats + 4 docs
(matched on title/keywords, author·year bylines); Enter opened a chat read-only (Chat mode), a doc
result opened the DPR paper (Library mode) — documents lazy-loaded by the overlay from a cold Chat
session; no-match → honest-empty line; collapse hid the rail + resizer, persisted across a reload,
expanded back to 260px; dark tokens resolve (surface/border/scrim-via-color-mix/fg-2); 375px → collapse
toggle hidden + hamburger shown + 0 px horizontal overflow; **0 console errors** throughout.

**What it opens.** The row's clause (c) — Graph nav placement + the empty-Graph → per-folder-concepts
fork — is still open and wants `grill-me` before any code. `apps/desktop/CLAUDE.md`'s "Tests: none"
line is stale (the `npm test` runner has existed since PR-2.5) — corrected in this change.

---
## 2026-07-20 — PR-2.7: the Manage view at scale (F1–F4) + KI-25, the graph emptied by KI-23's fix

Two things, logged together because the second was found while verifying the first in the running app.

### PR-2.7 — F1–F4

**What.** Four presentation fixes over the keyword overlay and the Manage view, all frontend-only,
all resting on new **pure** helpers in `lib/library.ts` (`unitDocCounts`, `splitRareFacets`,
`splitInheritedFamilies`, `filterByQuery`) so the rules are unit-tested rather than eyeballed.

**F4 is the substantive one.** A facet exists to *partition* a set; a keyword on one document
partitions nothing — selecting it yields that document, which search already does better. So the
1-doc tail is collapsed behind "Show rare (N)". That single principled threshold sweeps up the ugly
strings (`mathrm`, `102ff`, `fne-tune`) **and** the real specialist vocabulary (`va1v`, `avpv`)
without having to classify them — which is the point, because they are not distinguishable by
inspection. Nothing is destroyed: search bypasses the split entirely, and a **selected** facet is
never demoted.

**Two guards worth naming.** *Honest-empty*: when every facet is rare (a small collection), nothing
is demoted — collapsing the whole list would look broken rather than informative. *Stable rarity*:
the counts come from a `unitDocCounts` map over the pre-facet pool, not from `KeywordFacet.count`
(which is relative to the *faceted* pool and would make the rare set shift under the user as they
filter).

**F1 was already satisfied.** `.kwlistfoot` is a flex sibling of the scrolling `.kwlist`, so
"Manage keywords…" is already a pinned footer. Verified live and recorded rather than "fixed" —
PR-1 landed the same day the feedback was taken.

**The spec's F3 grounding was wrong; the live data corrected it.** It expected "only ~6 are real
families; the rest are 0-member concepts". Measured: **12** have ≥1 alias (real collapses), **10**
have 0 aliases but **>0 docs** (`ImageNet` 10, `Tractography` 10 — not collapses, but they *do*
partition the grid), **4** are inert (0 aliases, 0 docs). Only the 4 are hidden; the heading now
carries the split. Hiding all 22 would have removed working facets to satisfy a mis-estimate.

**A trap found in this PR's own rule.** A family created with no members starts at 0 aliases /
0 docs — exactly the shape the glossary-only group hides — so creating one would look like it
silently failed. `submitCreate` now reveals that group when, and only when, the new family has no
members.

**Rejected:** *deleting the rare tail* — mostly real vocabulary, and delete is not
reversible-by-search; *hiding every 0-alias concept* — 10 of them filter real documents; *a
corpus-tuned "only demote if the list is long" rule* — the 1-doc principle is scale-free and the
project forbids corpus-tuned constants.

**Verified:** 8 new frontend tests (23 total); suite **1164 passed / 1 skipped** · ruff ·
`mypy --strict src` · `svelte-check` 0/0 · docs+integrity 0/0. **Live, $0:** overlay 55 facets →
**25 shown / 30 demoted** (the spec predicted 30, exactly); toggle round-trips 25 ↔ 55; searching
`mathrm` finds a demoted keyword; Manage pool 38 → 12; families 22 ↔ 26; F2 shows "Go to family" +
a warning on an exact match and suggests `Brain connectivity`/`Connectome` on `conn`. Dark at
375 px: 0 px overflow, 0 console errors.

### KI-25 — the graph emptied itself when KI-23 was fixed

**Symptom (user-reported):** the Graph view showed nothing; `/api/concepts/graph` returned **0
nodes** while `concepts` held **26** rows.

**Cause.** ADR-018 made the graph vocabulary opt-in via `concepts.graph_include`, and
`load_concepts()` documents that NULL "reads as excluded". That column had never reached this box
(**KI-23**); running the migration by hand on 2026-07-20 — *while diagnosing KI-23* — finally added
it, **NULL on all 26 rows**, excluding every concept at once. The migration was correct. What was
missing is that **an additive column whose NULL default changes behaviour is not a safe additive
migration** — it needs its backfill in the same breath. `_ADDITIVE_COLUMNS` even carries that note
for this exact column; nobody was in a position to act on it, because the column had never landed.

**Fix.** `backfill_graph_include --apply` (ADR-018's rule retroactively: `source == "manual"` opts
in — all 26 qualify) then a skeleton rebuild, **Node A only, deterministic, $0**. Result: **26
nodes / 70 edges / 3 communities / 14 gaps**, `stale: false`; the concept index and the ego view
both render again (9 circles + 11 edges for `Connectome`).

**Not restored, deliberately:** `concept_edges` was already empty, so nothing was lost *by this* —
but Node-B stance annotations do not exist now either, and regenerating them is an **LLM pass**
(`--apply --enrich`, KI-4: force `--provider ollama`, which lives on the other box). Not run; the
user decides.

**Why nothing caught it:** the graph route degrades honestly to an empty graph (the documented
"empty vocabulary → empty graph" path), so the suite stayed green and no gate compares "concepts in
the DB" to "concepts the graph can see".

---
## 2026-07-20 — PR-2.6: family-aware grid tiles (D6 — a family selection highlighted nothing)

**What.** `LibraryGrid` learns about tag families through one optional prop, `keywordsOf`,
defaulting to `(d) => d.keywords`. `App` passes the accessor it already derives for the facet
overlay, so tiles and facets finally agree on what a *unit* is. Ordering moved into a pure
`orderedUnits(units, active)` in `lib/library.ts`; the `+N` overflow count now counts units.

**Why it was broken.** `LibraryGrid` matched `activeKeywords.includes(rawKeyword)`. With a family
selected, `activeKeywords` holds the **canonical** (`Pretrained model`) while tiles held the raw
member forms (`pretrained`/`huggingface`) — the match could never fire. `orderedKeywords` broke
identically, so the matching form was not floated to the front and could hide behind `+N`. One
root cause, one file: the grid never learned about families, which is why D6 was carved **with**
PR-2.6 rather than into PR-2.5 (splitting would have touched the same file twice).

**Why the default is the raw list.** It is the whole no-families guarantee: with `canonicalOf`
empty, `familyUnitsOf` already returns the raw keywords, and the default prop means a caller that
knows nothing about families renders exactly what it rendered before. Verified live with a plain
keyword control.

**Why the `+N` count had to change too.** It read `d.keywords.length`, so a tile holding `llm`
and `llms` would claim one more chip than it renders once the family collapses them — and the
collapse would not actually free the tile's scarce chip budget, which is half the point of showing
a family atomically.

**Rejected:** **passing the family list into the grid** and mapping there — that would put the
grouping rule in two places (the overlay already owns it) and give the component a data dependency
it does not need; **keeping ordering inside the component** — it is the half of D6 easiest to get
wrong, so it belongs where it can be tested; **rendering both the canonical and its members** —
contradicts the overlay's atomic-entry rule and spends the chip budget on duplicate forms of one
concept.

**Verified:** 5 new frontend tests (15 total, `npm test`); full suite **1164 passed / 1 skipped** ·
ruff · `mypy --strict src` · `svelte-check` 0/0 · docs+integrity 0/0. **Live on the real 76-doc
corpus ($0, no LLM), driven in the running app:** with the probe family selected, **22 of 22**
tiles highlighted and in **all 22** the active chip was floated first — the spec measured **0 of
25** before. Control (plain keyword `cajal`): 9 of 9 highlighted and floated, so the default path
is unchanged. Dark theme at 375 px: 0 px horizontal overflow, active chip visually distinct, 0
console errors. The probe family was built from two previously un-familied keywords
(`pretrained` + `huggingface`) precisely so deleting it restored the vocabulary exactly — verified
at 26 concepts / 17 aliases before and after, both keywords unclaimed again.

**Opens:** PR-2.7 (Manage view at scale) is next in the carve. Svelte-5 gotcha worth knowing:
`{@const}` must be the immediate child of a block (`{#each}`), not of the element where it reads
most naturally.

---
## 2026-07-20 — PR-2.5: hardening the tag-family write paths (D1–D5, all five defect-driven)

**What.** The five defects the post-commit review of `0c3b0d4`+`0af43db` found in the tag-family
**write** paths — none of which the 977 tests passing at the time caught. Each repro is now a
regression test written **first**, so it fails against the shipped code. Read path untouched.

| # | Defect | Fix |
|---|--------|-----|
| D1 | Rename onto an existing canonical created duplicate `Concept.label` rows → `add_concept`'s get-or-create then raised `MultipleResultsFound` for that label **forever**, 500ing the create route and breaking `promote_keyword` repo-wide | `rename_keyword_family` raises `KeywordFamilyExists`; the API shell maps it to **409** |
| D2 | Rename silently dropped the family's own canonical keyword — it is only an *implicit* member, so re-pointing the label let the original keyword fall out and reappear as the standalone chip the feature exists to remove; `doc_count` fell with it | Rename carries the old label into the alias set first |
| D3 | "New family" took its canonical as unchecked free text, so a keyword already claimed elsewhere ended up in **two** families and `familyCanonicalMap` resolved it order-dependently — three different numbers for one keyword | `create_keyword_family` routes the canonical through `add_family_member`, reusing the move-on-reassign guard |
| D4 | The sibilant `-es` rule always stripped two characters, so every plural whose singular ends in `e` (`database`, `size`, `cache`, `response`) **never matched** — a silent false negative that degraded a `confidence=1.0` structural pair to a threshold-dependent fuzzy one, or to nothing | `_stem` → `_stem_candidates`; Tier 1 groups on a non-empty intersection, union-find because a name can now bridge buckets |
| D5 | A live keyword selection was not re-pointed when families changed → the grid emptied behind a chip that still looked selectable. The Manage view is opened *from* the overlay, i.e. exactly where a selection is live | New pure `remapSelection` in `lib/library.ts`, called after every family write |

**Why D2 was fixed by rename rather than by create.** The spec offered both. Seeding the canonical
as a real alias on create would have needed a **migration for the 26 pre-existing concepts** on this
box; carrying the old label at rename time changes nothing about existing rows, because the label
stays the implicit member `_build_family` already treats it as. Smaller blast radius for the same
invariant.

**Why D4 emits two candidate stems instead of a better single rule.** There isn't one. `boxes`→`box`
and `databases`→`database` are structurally identical — both stems end in a sibilant — so no
lexicon-free rule can separate them. Emitting both trades an implausible false **positive** (needs a
real keyword equal to an over-stripped stem: `cas` beside `cases`) for a silent false **negative**,
which is the worse of the two because a proposal is reviewed before it is applied and a
non-proposal is not reviewable at all.

**Why the frontend got a test runner and no dependency.** The spec's DoD asks for the first tests of
`familyCanonicalMap`/`familyUnitsOf`/`facetFilter` — but the frontend had **no test runner at all**,
a prerequisite the spec never states. Node 24's built-in `node:test` runs the real `.ts` module with
native type stripping, so `npm test` works with **zero new dependencies**. Test files are excluded
from `tsconfig.json` so the app config doesn't have to carry `@types/node` +
`allowImportingTsExtensions` for test-only imports; they are run, not type-checked.

**Rejected:** **adding vitest** (or `@types/node`) — a dependency and a new gate to maintain, for
something the runtime already does; **enforcing uniqueness in the Manage view** (D1) — the invariant
belongs at the library boundary, the view is one of several callers; **exact-case collision
matching** — the client lowercases its canonical map, so two families differing only by case would
collide there anyway; **dropping `_stem` while keeping a "primary" stem for readability** — dead code.

**Verified:** 5 new integration tests + 6 new/updated unit tests + **10 new frontend tests**; full
suite **1164 passed / 1 skipped**; ruff · `ruff format` · `mypy --strict src` · bandit ·
`svelte-check` 0/0 · `npm test` 10/10 · docs+integrity 0/0. **Live, $0, on a copy of the real
76-doc library** (the original verified untouched at 26 concepts / 17 aliases before and after):
Detect reproduced `pvpo`≈`avpv pvpo` @ 0.77 with the real bge embedder → Accept → **Rename**, which
kept `pvpo` as a member and held `doc_count` at 1 (D2), a colliding rename returned **HTTP 409**
(D1), and re-creating that same label still returned **200** — the repo-wide poisoning is closed.
Measured honestly: on this corpus the fixed stemmer finds **exactly the same three pairs** as the
old one (`llm`/`llms`, `connectome`/`connectomes`, `keypoint`/`keypoints`) — no regression, and no
new find either, because these 60 keywords contain no e-final plural. D4 is proven by unit test, not
by this corpus.

**Not done, deliberately:** D5 was not driven in the browser. Exercising it live would write to the
user's curated vocabulary, and move-on-reassign is **not undoable** — detaching a keyword deletes
that `ConceptAlias` row, and deleting the new family does not restore it. It is covered by 4 unit
tests on the extracted pure function plus `svelte-check`.

**Opens:** that non-undoable move is ADR-015's stated semantics, not a defect — but D3 now extends it
to the *canonical*, so naming a new family after a keyword claimed elsewhere silently strips it from
the other family. Worth knowing before bulk curation; recorded in the spec. **PR-2.6** (family-aware
grid tiles, carrying D6) is next in the carve.

---
## 2026-07-20 — KNOWN_ISSUES split: open issues in the working file, closed ones archived verbatim

**What.** `.claude/KNOWN_ISSUES.md` went from **738 lines to 237**. The 14 resolved entries moved
**verbatim** to `docs/archive/KNOWN_ISSUES-resolved-001.md` (544 lines); the working file keeps the
10 open issues in full plus a new **Resolved — index** table. `AGENTS.md`'s coordination-file list
points at both.

**Why.** 526 of the 738 lines — 71% — were closed issues. The file is read at session start to find
what might bite *today*, and four fifths of it was history. Same shape ADR-022 already applied to
the decisions monolith: living index in the working file, canonical detail frozen in `docs/archive/`.

**What each retained row keeps, and why those two things.** `| KI | What it was | What keeps it
fixed — do not undo | Resolved |`. A closed issue still carries exactly two live risks: **the trap**
(so it isn't re-diagnosed from scratch — e.g. "never `cu130` on a GPU-less box") and **the load-bearing
fix** (so nobody deletes it not knowing what it holds up — e.g. the API lifespan's `init_db()` call
is the *only* migration trigger the app has). Everything else — reproduction steps, the diagnosis
narrative, rejected alternatives, verification detail — is history, and history belongs in the
archive.

**Rejected:** **summarising on the way into the archive** — the archive is the canonical account,
and a summary of a summary is how detail quietly dies; **deleting resolved entries outright** — the
KI-15 and KI-22 write-ups are the record of *how a class of bug was caught*, which is worth more
than the bug; **splitting by date rather than by state** — "resolved" is the property that makes an
entry stop being operational, a date boundary is arbitrary; **per-heading anchor links** into the
archive — they rot on the first heading edit, and the KI number is trivially findable.

**Verified:** a script asserts every resolved body appears **byte-identical** in the archive, every
open body byte-identical in the working file, every resolved KI has an index row, and the header is
preserved — all four clean. `docs_check --strict` 0/0.

**Opens:** numbering stays global and never reused (the KI-23-was-KI-20 note travelled with its
entry into the archive). Next rotation is `KNOWN_ISSUES-resolved-002.md`; no cap is enforced by a
gate — `session_max_entries` covers the baton only.

---
## 2026-07-20 — `document_meta` gets its missing foreign key; rebuild migrations exist now (ADR-026)

**What.** `document_meta.document_id` is now a real FK to `documents.id` with `ON DELETE CASCADE`.
Getting there needed a new migration mechanism: SQLite cannot `ALTER TABLE … ADD CONSTRAINT`, and
`db/migrations.py` was additive-only by design. `_rebuild_table` implements SQLite's documented
rebuild dance — FKs off, one transaction, create the model's shape under a temp name, copy the rows
worth keeping, drop, rename, `foreign_key_check`, commit — and `_rebuild_document_meta_fk` is the
first (and so far only) caller. **ADR-026** records both the fix and the policy around the
mechanism.

**Why it mattered more than "a missing constraint".** Correctness was being held up by convention:
`delete_document` deletes the override by hand and its docstring says "no FK — explicit". Every
*bulk* path forgot. The pre-KI-24 `--rebuild` forgot, and — the part KI-24 left open —
`cleanup_orphans_sqlite` **still** forgot, on every incremental ingest that finds a gone or
content-changed source. So orphaned overrides were still being produced today: unreadable (every
read path resolves through a live document), never cleaned, accumulating.

**Why the new shape is rendered from the model, not hand-written DDL.** A hand-written
`CREATE TABLE` in a migration is a second definition of the table that silently drifts from
`create_all`. `_rebuild_table` compiles the SQLAlchemy model's own table under a temp name, and
copies only columns present in *both* the live table and the model — so a rebuild is safe against a
schema that predates an additive column.

**Why rebuilds are named functions, not a `_TABLE_REBUILDS` list.** A data-driven registry mirroring
`_ADDITIVE_COLUMNS` would be a framework for n=1 and would make rebuilding a table feel as routine
as adding a column. It is not: it rewrites the table. Named function, idempotent, ADR-justified.

**Orphans are dropped, and logged in full first.** They cannot be carried (they are what the
constraint forbids) and cannot be rescued (an override records no filename or hash, only a dead
id). `init_db` returns the orphan count with the change description, so the KI-23 startup log
states it. On the real corpus there were **zero**.

**Rejected:** **compensating in application code** — add the override delete to the orphan sweep
and `_sweep_rebuild_rows`; that is the bet that already failed twice, and a third caller would
forget too. **A periodic orphan-cleanup pass** — sweeping up after a defect instead of making it
impossible. **Alembic** — real tooling for one non-additive change on a single-file local SQLite
app is a dependency plus a workflow; the reopener is explicit (a second or third rebuild → adopt
it). **Leaving the orphans** after adding the FK — the constraint would be a claim the data does
not support. **Fixing `ConversationMeta` the same way** — it *cannot* have an FK: conversations are
derived by grouping `AnswerRecord` rows, there is no table to point at. Its bare `session_id` is
correct, not the same defect.

**Verified:** 8 new tests (`tests/integration/test_document_meta_fk_migration.py`) driving `init_db`
over a genuinely pre-migration table — FK added, live overrides kept, orphans dropped and counted,
idempotent, delete cascades, the orphan sweep no longer leaves a row behind, an unknown-document
override is rejected outright, and **a failed rebuild leaves the old table exactly as it was**
(no temp table, no data loss — the migration refuses rather than inventing a value). Full suite
**1143 passed / 1 skipped** · ruff · `ruff format` · `mypy --strict src` · bandit · docs+integrity
0/0. **Verified non-vacuous:** with the migration disabled, 7 of the 8 fail — the survivor is the
fixture's own assertion that the legacy table really has no FK. **Live, on a copy of the real
`data/library.db`** (the original deliberately untouched — the app will migrate it on next start):
FK added with CASCADE, the one real override row preserved with its `authors`/`year` values,
`PRAGMA foreign_key_check` clean, 76 documents intact, no leftover temp table, 0 orphans dropped.

**Opens:** the reopener above (a second rebuild → Alembic). `delete_document` keeps its now
redundant explicit override delete, retained so the ADR-014 path still reads as the complete story.

---
## 2026-07-20 — KI-24 fixed: `ingest --rebuild` rebuilds the index instead of resetting the library

**What.** The rebuild branch no longer runs `delete(DBDocument)`. It wipes both Chroma stores and
re-embeds, as its CLI help always claimed ("Wipe the vector store and re-embed everything"); the
rows it does **not** reproduce are swept afterwards by the new `ingest._sweep_rebuild_rows`,
classified gone/stale exactly the way `cleanup_orphans_sqlite` classifies them.

**Why the delete had to go, rather than be compensated for.** KI-24 proposed snapshotting
`document_folders` by `doc_hash` and restoring it after the loop. Auditing every FK to
`documents.id` before building that showed the blast radius was much wider than folders —
`document_tags`, `document_keywords`, `citations`, `doc_similarities`, `document_parts`,
`chunk_epistemics`, `concept_presence`, `ingestion_events` all cascaded; `is_archived` and `notes`
were reset by the re-insert; **`document_meta`** (the ADR-013 metadata overrides) has *no* FK, so
its rows were **orphaned** rather than deleted, silently inert against ids that no longer existed.
And because `figures` is keyed by document id, `figure_units()` found none mid-rebuild, so the
reindexed corpus carried **no figure chunks at all** until the paid VLM describe pass was re-run —
a silent retrieval-quality regression riding along with the data loss. Snapshotting all of that is
a re-keying layer; **keeping the rows** makes `_existing_document_id` resolve to the same id and
every association simply stays attached. One less mechanism, strictly more preserved.

**Why the sweep runs after the loop, not before.** `cleanup_orphans_sqlite` reads its candidate set
from the Chroma metadata — which this branch has just deleted — so it cannot run here. After the
loop the rebuild has already told us what every present source produces, so gone/stale falls out
with no re-hashing. The sweep keys on `indexed - indexed_before` ("what this run produced"), never
on "the store was empty": an rmtree that silently fails must not be able to delete a library.

**One deliberate behaviour change beyond restoring the invariant.** A document whose file is still
on disk but which produced nothing this run (extraction error, empty extract) is now **kept and
reported** (`rebuild_kept_unreproduced_rows`). The bulk delete removed it unconditionally, so a
transient extraction failure used to cost the user their folders and metadata for that document.

**Rejected:** the **snapshot-and-restore** KI-24 originally proposed — it preserves folders and
tags but cannot preserve figures (the rebuild reads them *during* the loop, before any restore
could run) and leaves the `document_meta` orphaning untouched; **a new `--reset-library` flag** to
keep the old nuke available — nothing asked for it, and a destructive escape hatch nobody
requested is how the original silent loss got shipped; **writing the snapshot to disk first** so a
crashed rebuild could recover it — that only exists as a problem if you snapshot at all.

**Retires ADR-025 F3 spec M3/M9.** M3 called a rebuild "the one honest exception" where a demo
removal is re-fought; M9 recorded the loss as warned-about-not-fixed. Both are amended in
`docs/specs/feature-corpus-folders-demo.md`: membership is preserved, so the demo hook sees no new
rows on a rebuild and **nothing is re-fought anywhere**. The `rebuild_clears_folder_membership`
warning F3 added is gone with the behaviour it described.

**Verified:** 6 new tests (`tests/integration/ingest/test_ingest_rebuild_preserves_library.py`);
full suite **1135 passed / 1 skipped** · ruff · `ruff format` · `mypy --strict src` · bandit ·
docs+integrity 0/0. **Non-vacuous:** restoring the bulk delete fails exactly the three
preservation tests and no others. **Live ($0, isolated `DOC_DATA_DIR`, real embedder + real
Chroma, rebuild run as its own process; the real 76-doc library verified untouched before and
after):** two documents re-embedded with **identical ids**, "My reading" (2) and "Demo corpus" (1)
intact, metadata override and notes preserved; then deleting a source and rebuilding logged
`rebuild_removing_rows gone=1` and dropped exactly that row.

**Opens:** the derived sidecars now survive a rebuild *because the ids are stable*, but they are
not re-derived by it — re-run their runners when the chunking changes. `document_meta` still has
no FK; nothing creates new orphans, but it is not referentially enforced. A quirk found on the
way, production-irrelevant but worth knowing: chromadb caches one system per persist path, so an
**in-process** rebuild reattaches to the store `rmtree` was meant to remove — `--rebuild` is a CLI
entrypoint in a fresh process and no API route exposes it, and the sweep is written not to care.

---
## 2026-07-20 — ADR-025 F3: demo corpus auto-assigns into a folder at ingest + a one-time backfill

Closes the ADR-025 carve (F1 folders → F2 retrieval scoping → **F3 demo auto-assign**). Contract
written first: `docs/specs/feature-corpus-folders-demo.md` (M1–M11).

**What.** New sidecar `src/doc_assistant/demo_corpus.py`: load the `collection: demo` pins from
`tests/eval/corpus_manifest.yaml`, decide whether a file *is* one of them by **bytes** (size
fast-path, then SHA-256 — so a renamed demo PDF still counts), resolve the demo folder, and assign.
`ingest.main()` gained a two-line seam — `get_document_row_hashes()` diffed around the processing
loop — and hands the newly-created rows to the hook. New runner
`scripts/backfill_demo_folder.py` (dry-run default, `--apply`, `--force`) covers documents that
were already in the library. `app_settings` gained `demo_folder_id` + `demo_backfill_done`.
`process_one_document` is **untouched**.

**Why the trigger is "the row is new", not `process_one_document`'s `"added"`.** `"added"` is also
returned for *re*-ingests — the inverse-orphan repair, a `--path` rerun — so keying on it would
re-add a document the user had removed from the folder by hand, every run. The ADR's own words are
"ingest of a **new** document"; the row-set difference is literally that, and it keeps the locked
ingest hot path free of a new parameter (M1/M2).

**Why the folder is resolved by a persisted id, not by name.** ADR-025 promises an ordinary,
renamable folder. A name-keyed lookup would silently create a *second* "Demo corpus" the first
time someone renamed theirs. The id lives in `settings.json` because it is a per-install
**pointer**, not document data — no schema change (M5).

**Why the backfill refuses to run twice.** A second pass re-adds exactly the papers the user
removed. `--force` exists and says so loudly (M8). A run that assigns nothing does **not** burn the
flag, so back-filling before ingesting doesn't lock out the real backfill.

**Rejected:** a **`folders.origin` additive column** to mark the demo folder — `settings.json`
already fits a per-install pointer, and one auto-managed folder doesn't earn schema surface
(reopener recorded). A **tombstone so a deleted demo folder never returns** — that would couple the
generic `delete_folder` to demo semantics; per-*document* removals are what ADR-013 protects, and
those are never re-fought (M6). **Bundling the manifest into the PyInstaller build** — the demo
corpus is a repo-clone flow end to end, so a packaged install has no demo files to assign; the
missing manifest is a quiet no-op by design (M10). A **demo badge / demo-specific UI** — ADR-025
fork 1 is one organizing concept, one write surface (M11).

**Found while specifying, logged not fixed — KI-24.** `ingest --rebuild` runs
`delete(DBDocument)`, and `document_folders` cascades on the document side: **every folder is
silently emptied** while still appearing in the rail. F3 adds a warning naming the count
(`rebuild_clears_folder_membership memberships=4` in the live probe) and logs the issue; the real
fix (snapshot membership by `doc_hash`, restore after) is its own change. The demo folder is the
one that self-heals, because a rebuild makes every document look newly ingested (M3/M9).

**Also corrected: a duplicate KI number.** The schema-migration issue filed yesterday as **KI-20**
collided with the existing KI-20 (concept curation hard-deletes vocabulary). Renumbered to
**KI-23** in the living `KNOWN_ISSUES.md` + code/spec/test references; the append-only DEVLOG and
baton entries above still read "KI-20" and were **not** rewritten — KI-23 carries a pointer note.

**Verified:** 27 new tests (13 unit + 14 integration); full suite **1129 passed / 1 skipped** ·
ruff · `ruff format` · `mypy --strict src` · bandit · docs_check 0/0 · integrity_check 0/0.
**Live, end-to-end, $0** (real ingest, real Chroma, local embedder, isolated `DOC_DATA_DIR` — the
real 76-doc library never touched, verified 76 docs / 0 folders / no settings file before and
after): a **renamed** demo PDF + a private PDF ingested → only the demo one joined "Demo corpus";
re-ingest after a manual removal put **nothing** back; the folder renamed to "Sutskever reading
list" kept receiving new demo papers with **no** second folder created; `--rebuild` logged the
membership warning, left a hand-made folder empty, and refilled the demo folder. The live backfill
dry run against the real corpus found 18 demo files on disk, none ingested — and **caught a
reporting bug** on the way: the summary counted never-ingested files as "already members". Fixed
and covered by a test.

**Opens:** KI-24 (the real rebuild fix, and `document_tags` has the identical exposure) · the M8
reopener (a per-document "was auto-assigned" marker would make backfill re-runs safe without a
flag) · an "Eval corpus" folder for the other collection is deliberately **not** built (not in
ADR-025) · `--remove-demo` still leaves the emptied folder behind, by choice.

---
## 2026-07-20 — KI-20 resolved (schema migrates on API start) + A/B compare honours the scope

Two decisions the user took after reviewing F2 (`0e45dd3`), built together because both are
about the same thing: a surface that quietly describes something other than what it did.

**What (1) — KI-20, RESOLVED.** `init_db()` was called from **exactly one place in the running
app**: `ingest/__init__.py:405`. So a user who pulled an update and only chatted never received
new additive columns, and the **packaged build never migrated at all**. Evidence it had already
bitten: this box was missing `concepts.graph_include` (added 2026-07-07) for ~2 weeks, silently.
Fix — the API lifespan now calls `init_db()`, and `init_db`/`_apply_additive_columns` **return the
columns they added** so the lifespan logs `schema_migrated_at_startup columns=[...]` at WARNING
(`schema_current` otherwise). A migration error is caught and logged, never a startup crash.

**What (2) — the A/B compare scopes both sides.** `compare_retrieval(..., scope_folder_id)`
threads the resolved scope into **both** arms; `CompareResult.scope_label` drives a card line
("Both sides searched X only"). Wire: `CompareRequest.scope_folder_id`,
`CompareResultPayload.scope_label`, `types.ts`, `compareRetrieval(..., scopeFolderId)`.

**Why:** (1) F2 put an additive column on the **answer path** (`answer_records.
retrieval_scope_json`), so the long-tolerated migration gap stopped being a sidecar problem and
became "every turn fails to record". (2) With a folder selected, an unscoped diff describes
retrieval the next answer will not perform — the same quiet mismatch F2 exists to remove. Holding
the document set constant across A and B is also what makes the comparison *about the knob*.

**Rejected:** for KI-20, a **startup schema check that only warns** — it diagnoses without fixing,
so the answer path still breaks until the user acts; and **wrapping the provenance write in
`suppress`** — that hides a schema fault by silently dropping provenance, i.e. buys uptime with
the integrity layer. For the compare, **leaving it unscoped but labelled** — the label removes the
lie but keeps showing a comparison the user can't act on.

**Verified:** 3 new tests (13 total in `test_retrieval_scope.py`); full suite **1102 passed / 1
skipped** · ruff · `mypy --strict src` · bandit · `svelte-check` 0/0 · docs+integrity 0/0. The
KI-20 guard test builds a genuinely stale schema (drops the column), starts the app, and asserts
the column is back — **verified non-vacuous**: it fails when the lifespan call is removed. Live
startup logged `schema_current` on this (already-migrated) box. **Live A/B through the real API
and real pipeline ($0, retrieval only, no generation):** unscoped compare reached `bge_cpack`,
`dpr_karpukhin`, `rag_lewis` — **all outside** the probe folder; scoped compare kept **both**
sides entirely inside it; `scope_label` null unscoped, `"__ab_probe__ (3 documents)"` scoped.
Probe folder deleted; DB left at 76 docs / 0 folders.

**Opens:** F3 (demo sha-match auto-assign) untouched. Multi-folder scopes, persisted
per-conversation scope, and scoping the enrichment sidecars stay parked (ADR-025). RG-020's
synthetic 10k measurement still owed.

---
## 2026-07-20 — F2: query-time folder retrieval scoping + the honesty contract (ADR-025 carve step 2)

**What:** built **F2** — a folder can now scope one chat turn's retrieval. Contract first:
`docs/specs/feature-corpus-folders-scope.md` (S1–S10). `library.folder_doc_hashes` resolves a
folder to its non-archived members' `doc_hash`. `pipeline.retrieve_with_scores(..., scope=)`
scopes **both arms before scoring** — vector via Chroma `$and[keep_for_retrieval≠False,
doc_hash $in [...]]`, BM25 by rebuilding over the subset of a now-retained `self._bm25_docs` —
memoised in one slot keyed on the hash set. `chat_controller` gains `ScopeView`,
`_resolve_scope`, `_scope_note`, a `scope_folder_id` parameter threaded like `overrides`, and
`TurnResult.scope`. Provenance gains an additive `answer_records.retrieval_scope_json` column.
API: `ChatRequest.scope_folder_id` + `TurnResultPayload.scope`. Desktop: a composer scope
selector (in-memory only) and a scope chip on the answer — plus the same chip on reopened
conversations, replayed from the record via `ConversationTurn.retrieval_scope`.

**Why:** F1 shipped a Library filter and said in-product that chat still searched everything.
F2 makes that false — and the same honesty rule then points the other way, which is the whole
content of this increment: a scoped answer that doesn't say it was scoped is indistinguishable
from a whole-library one, i.e. the `is_archived` failure with a nicer UI.

**Rejected:** (1) **putting the scope inside `RagOverrides`** — the plumbing is identical, but
`RagOverrides` is ADR-010's governance channel for *locked quality knobs*; a scope is a *content
filter*, and filing it there would blur exactly the distinction that keeps it out of the eval
gate (and would render it through `_overrides_note`'s "🧪 Session override", framing a content
choice as an experiment). (2) **Sending a doc-hash list from the client** — it goes stale between
a Library edit and the next turn, and would let a caller retrieve an arbitrary set that no folder
ever contained, which the provenance record would then attest to as "this folder". The id travels;
the backend resolves. (3) **Falling back to unscoped when the folder is unknown/deleted/empty** —
the single most important rejection: "I couldn't honour your scope" must never collapse into "I
searched everything", so an unresolvable scope is a distinct empty `frozenset` all the way down
and the turn answers honestly with zero sources. (4) **Persisting the selector** (localStorage /
server-side) — that is ADR-025's rejected global scope: a scope you forgot you set silently
narrows every future answer. (5) **Folding the scope into `prompt_version_hash`** — it would mint
a prompt version per folder and pollute every eval join keyed on it.

**Measured before deciding** (`tests/eval/baselines/rg020_scoped_retrieval_cost_2026-07-20.md`,
live 76-doc / 30,882-chunk index): BM25 subset rebuild ≈20 µs/chunk (622 ms whole corpus · 248 ms
for 30 docs · 27 ms for 3); Chroma `$in` 136 ms unscoped → 193/232/408 ms for 3/30/76 hashes —
the cost tracks the **`$in` list length**, not the corpus share. That is what bought the S5 cache.
**RG-020 partially discharged**; the 10k half stays open and is explicitly *not* claimed.

**Verified:** 17 new tests (`tests/unit/test_pipeline_scope.py` 7 + `tests/integration/
test_retrieval_scope.py` 10), including the S4 byte-identical guard, the S3 never-widen behaviour
on deleted/empty/unknown folders, both synthesis paths, scope isolation between turns, the API
round-trip and the additive-column migration. Full suite **1099 passed / 1 skipped** · ruff ·
`mypy --strict src` · bandit · `svelte-check` 0/0. **Live on the real corpus ($0, no LLM):**
unscoped retrieval hit 3 documents of which **2 lay outside** the probe folder; scoped retrieval
hit only in-folder documents; an empty scope returned 0 sources in 0.2 ms; unscoped retrieval
still worked afterwards. UI verified through the `window.fetch` SSE mock (**no paid turn**):
request body carries `scope_folder_id`, chip reads "Searched Retrieval demo only — 3 documents,
not the whole library", the deleted-folder variant reads "no documents were searched", an
unscoped turn adds no chip, selector tints only when scoped, light+dark, 375px no overflow, 0
console errors. Probe folders deleted; DB left at 76 docs / 0 folders.

**Opens:** ⚠ the live DB was **missing this column and `concepts.graph_include`** until
`init_db()` was run by hand — the API never migrates on startup (`apps/api/CLAUDE.md`), and F2
moves that gap onto the **answer path**, where it would 500 every turn. Logged as **KI-20**.
F3 (demo sha-match auto-assign) untouched. `compare_retrieval` (A/B) stays unscoped — stated,
not built. Multi-folder scopes, persisted per-conversation scope, and scoping the enrichment
sidecars remain parked (ADR-025).

---
## 2026-07-20 — F1: folders end-to-end (CRUD + membership + Library rail), ADR-025 carve step 1

**What:** built **F1** of the ADR-025 carve over the previously dormant `Folder`/`document_folders`
schema (0 rows). Contract first: `docs/specs/feature-corpus-folders.md` (D1–D9). Backend —
`library.py` gains `FolderSummary` + `list_folders`/`get_folder`/`create_folder`/`rename_folder`/
`delete_folder`/`add_documents_to_folder`/`remove_documents_from_folder`/`folder_document_ids`,
mirroring the shipped keyword-families surface (None = unknown, `ValueError` = blank/collision,
idempotent create, refreshed entity returned). `DocumentSummary` gains `folder_ids`;
`list_documents(folder=<name>)` becomes `list_documents(folder_id=<id>)`. API — six routes under
`/api/library/folders` + `LibraryFolderPayload`/`FolderCreate`/`FolderRename`/`FolderMembers`;
`types.ts` mirrored. Frontend — new `LibraryManageFolders.svelte` (create / inline rename /
confirm-delete / searchable bulk document picker, reusing the ManageKeywords modal shell), rail
section "Collections" → **"Folders"** rendering the API list with counts + a "Manage…" entry point,
`docsFor` matching `folder_ids`, and an **"Add to folder…"** item in the grid tile's ⋯ menu that
opens the view pre-filtered to that document.

**Why:** F1 is the demoable standalone step of the carve, and it is the piece that has to exist
before F2 can scope retrieval to anything. Reconciliation with the L4 Library-redesign spec is the
real judgment here — see "Rejected" below.

**Rejected:** (1) **the baton's "compose both auto-assign rules" instruction** — L4's own
2026-07-15 section already SHELVED source-dir subfolder mirroring when the user confirmed the
reopen condition (`source_dir` is flat by design), and named **manual assignment** as the only
honest path, gated on an ADR. ADR-025 is that ADR, so F1 builds manual assignment and mirroring
stays shelved; F3's sha-match is a separate rule, not a second mirror. (2) **Nesting** — the
schema is hierarchical but v1 creates every folder at the root (D1): ADR-025 flags nesting as the
reopener for the whole folders-are-groups identity, and it would force F2 to invent an answer to
"does scoping a parent include its children?". (3) **Name-keyed filtering** — `uq_folder_name_parent`
never fires for root folders (SQLite treats NULL parents as distinct), so uniqueness moved into
`library.py` and every filter keys on id (D2/D4). (4) **Deriving the rail from document payloads**
(`folderGroups`, now retired) — a folder derived that way cannot appear while empty, and an
invisible empty folder cannot be filled (D3). (5) **Deleting L4's write-trap test** — narrowed
instead: read routes still write nothing (D7).

**Honesty note (D8):** F1 ships the Library filter but *not* retrieval scoping, so the Manage view
states in-product that chat still searches every document. Without it, narrowing the Library reads
as narrowing the answer — the exact `is_archived` failure ADR-025 exists to prevent. F2 deletes
the line by making it false.

**Verified:** 15 new integration tests (`tests/integration/test_library_folders.py`) covering
case-insensitive idempotent create, blank/collision `ValueError`, unknown-id `None`/`False`,
idempotent membership, m2m overlap, archived excluded from counts (D5), id-based filtering, the
D6 "delete never touches documents" guard and the D7 read-path write trap. Full suite **1082
passed / 1 skipped** · ruff · `ruff format --check` · `mypy --strict src` · bandit ·
`svelte-check` **0/0**. **Live on the real 76-doc corpus ($0/offline):** created "Demo corpus",
bulk-added 3 documents → rail count 3 → grid filtered to 3 tiles with the breadcrumb resolving the
id to the name; a rename onto an existing name surfaced `a folder named 'demo CORPUS' already
exists` inline without blocking; deleting both folders left **76/76 documents intact** and reset
the active collection to All; light + dark tokens resolve; 375px no overflow; 0 console errors.
DB left with 0 folders.

**Opens:** F2 (retrieval scoping + per-turn selector + provenance/answer chip; carries RG-020) and
F3 (demo sha-match auto-assign + backfill) are untouched. `Tag` CRUD is the same shape and stays
dormant, deliberately not bundled. `DocumentDetails.folders` still ships names only (the drill-down
does not filter). Nesting, drag-and-drop assignment, and per-folder enrichment remain parked.

---
## 2026-07-20 — Docs: corpus groups grilled → design-locked as "Folders with retrieval scope" (ADR-025)

**What:** ran `grill-me` on the corpus-groups question the demo collection raised (demo corpus vs
personal papers in one store). 6 forks → all resolved or parked; ledger in the session baton.
**ADR-025** written (accepted, unbuilt): corpus groups ARE folders (reuse the dormant
`Folder`/`document_folders` schema — the ADR-015 reuse pattern); demo membership auto-assigned at
ingest by manifest sha-match + one-time backfill, user edits win (ADR-013 pattern); scoping = a
**query-time doc-hash filter on both retrieval arms** (no chunk-store writes; unscoped path
byte-identical); scope is per-turn request-scoped (ADR-010 pattern), sticky in UI only, and **the
provenance record + an answer chip always state the scope** (integrity, non-negotiable);
enrichment stays corpus-global in v1. Carve **F1 folders → F2 scoping → F3 demo auto-assign**,
spec at build time. Routed: RG-020 (scoped-retrieval bounds: Chroma `$in` latency at the 10k
contract + scoped-BM25 statistics + an unscoped-byte-identical guard test) and RG-021 (the eval
index-composition fingerprint, promoted from the 2026-07-20 demo-collection entry's "Opens");
ui-checklist §3 row (design-locked); decisions.md row.

**Why:** the user's fork — groups inside the main store vs a fully separable corpus — plus the
requirement that demo files stay easily deletable. The deciding constraint is the `is_archived`
precedent: doc-level flags scope every library-side read but NOT retrieval (chunks carry only
`doc_hash`), so any grouping that scopes the grid alone lies in chat. That makes retrieval
scoping the feature's core, not its garnish.

**Rejected (full list in ADR-025):** separate database (complexity, not storage — every read path
is corpus-global; env-level data home already gives coarse isolation); a new group object beside
folders; a partition column; chunk-metadata stamping (mutates the chunk store per edit);
post-rerank filtering (recall collapses on small scopes); persistent/global scope (a forgotten
scope silently narrows every answer).

**Opens:** F1's spec must reconcile ADR-025 with the L4 Library-redesign spec's Phase-B locks
(2026-07-14: "folders = mirror source subfolders at ingest + backfill") — two auto-assign rules
compose, they don't compete. Per-folder enrichment parked with a named reopener (facet clutter →
PR-2.7 demotion first). RG-020/021 carry the deferred measurements.

---
## 2026-07-20 — Demo-corpus removal: `download_corpus --remove-demo` (content-hash matched, ADR-014 safe-delete, dry-run default)

**What:** the demo collection is now cleanly removable. Core in `src/doc_assistant/library.py`
(scripts stay thin per the module contract): `SourcePin`/`SourceMatch`/`SourceRemoval` +
`match_pinned_sources()` — finds files under the sources dir by **content** (size fast-path so a
big corpus costs stats not reads, then SHA-256; rename-proof) and links each to its library row by
filename (content can't bridge that hop: `doc_hash` hashes extracted text, not file bytes; >1 row
sharing a name → flagged ambiguous, never auto-deleted) — and `remove_pinned_sources()` — ingested
matches go through `delete_document` (ADR-014: Recycle Bin first, then row/chunks/sidecars)
against the live index, the same doc's chunks are swept from the secondary Chroma store too (the
API delete only cleans the live one), never-ingested files go straight to the Recycle Bin; a
refused trash (locked file) fails that one match and the batch continues. Script:
`--remove-demo` (plan) / `--remove-demo --apply` (execute) per the dry-run-default polarity;
`_chunk_stores()` opens both Chroma stores **without loading the embedder** (get/delete never
embed) so cleanup works model-cache-free. **7 new integration tests**
(`tests/integration/test_demo_corpus_removal.py`) on the ADR-014 test harness. Verified live:
dry-run against the real `data/sources/` found **exactly the 18 just-downloaded demo files**
(all correctly triaged "file only — never ingested"), removed nothing. Full suite **1067 passed /
1 skipped** (pre-existing); ruff · `mypy --strict` · bandit(src) clean; `docs_check --strict` 0/0.
README demo note + corpus README gained the removal line.

**Why:** the corpus-groups discussion (2026-07-20): whichever way grouping lands later, "someone
wanting to use the app should be able to delete those demo files easily" stands alone — and it
rides entirely on shipped machinery (manifest pins + ADR-014), so it ships now while corpus
groups waits for its grill + ADR.

**Rejected:** matching library rows by `doc_hash` (it hashes extracted markdown, not file bytes —
no bridge from a PDF's sha256); hard-deleting anything (ADR-014's whole point — Recycle Bin +
re-download keeps every step reversible); a separate `scripts/remove_demo_corpus.py` (removal is
the download's inverse; one manifest-owning script, one `--dest`); auto-deleting on ambiguous
filename collisions (deleting the wrong user document to save a demo-cleanup click is the worst
trade available).

**Opens:** the bandit B310 `urlopen` advisory in `download_corpus.py` is pre-existing and outside
the gate (bandit runs on `src/` only) — fine, but worth a `# nosec` + comment if scripts ever
enter the gate. A file renamed *after* ingest removes as file-only and leaves its stale row to
the ingest orphan cleanup (documented in the docstring). The secondary-store sweep exists because
API deletes clean only the live index — if that ever changes in `delete_document` itself, drop
the sweep here.

---
## 2026-07-20 — Public corpus: 18-paper demo collection (Sutskever→Carmack list) + `download_corpus --demo`; verified-10 regime pinned by a guard test

**What:** `tests/eval/corpus_manifest.yaml` gains a **`collection: demo`** section — the 18
arXiv-pinnable papers of the rumoured Sutskever→Carmack reading list (30papers.com): ResNet +
identity mappings, dilated convs, RNN regularization, Deep Speech 2, Order Matters, Bahdanau
attention, Pointer Networks, the Transformer, NTM, relation nets ×2, MPNN, scaling laws, GPipe,
the coffee automaton, VLAE, and the Grünwald MDL tutorial (old-style id `math/0406077`). Every
entry pinned the honest way: ids + latest versions verified against the arXiv API, then **each
pinned-version PDF actually downloaded** (scratchpad, stdlib urllib + truststore through the
corporate proxy — the KI-10-addendum transport; 3 s spacing) and SHA-256 + byte-size recorded;
**18/18 re-verified through the script's own `--verify-only` path** (0 mismatches).
`scripts/download_corpus.py`: new pure `_selected()` + **`--demo` flag** (default selection
unchanged = the eval 10), a 3 s politeness sleep between real fetches, and an inform-don't-block
summary line when demo entries are excluded. **New guard
`tests/unit/test_download_corpus_selection.py` (5 tests)** pins the default selection to exactly
the verified-10 and every demo entry to `referenced_by_eval: false`. Docs: README Usage gains the
"try it on a ready-made corpus" note; `evals/README.md` + `tests/eval/corpus/README.md` state the
demo collection is never part of the benchmark regime. Unit suite **841 passed**; ruff clean;
`docs_check --strict` 0/0.

**Why:** the 2026-07-20 evals-split session scoped this (ADR-024 "Opens"): the app demos better
on a bigger, famous corpus (concept graph, wiki, gaps), but the verified-10 benchmark regime must
stay closed — extra corpus documents are retrieval distractors, so demo papers must be opt-in and
excluded from every published number. The list itself: zero overlap with the eval 10 (RAG methods
vs DL classics), all freely downloadable, nothing re-hosted.

**Rejected:** `tier: demo` (the chip spec's literal wording) — in the real schema `tier` is
*source provenance* (`arxiv` vs the forward-compat `committed`), so membership got its own
explicit `collection` field with absent = eval (pre-demo entries untouched, byte-identical);
reusing `referenced_by_eval` as the selector (conflates "a case cites it" with "in the eval
corpus" — a deliberate distractor paper would break the equivalence); the 9 non-arXiv items
(AlexNet, Hinton MDL, Cover–Thomas chapter, Legg thesis, CS231n, blog posts) — noted in the
manifest as ingest-as-HTML candidates, not silently fudged in.

**Opens:** running `--demo` then a benchmark run on the same index produces non-comparable
numbers — the docs say so, but nothing *mechanically* stops an eval run over a demo-polluted
index (would need an index-composition fingerprint in the eval harness; noted, not built). The
HTML items (Karpathy/Olah/Aaronson posts, CS231n) would exercise the HTML ingest path if ever
wanted. arXiv re-renders make SHA mismatches warnings by design — if one fires later, re-pin and
note it here.

---
## 2026-07-20 — Docs: benchmarks split out of README into a top-level `evals/` folder (ADR-024)

**What:** New top-level `evals/README.md` now holds the full benchmark write-ups — the headline
public benchmark (table + interpretation + the sbert_motivation judge-flakiness caveat), the
`bge-base` vs `specter2` embedder comparison, the chunk-size sweep, the BM25/vector-weight sweep,
and both reproduction guides — moved verbatim from the README's ~95-line Benchmarks section
(links re-based `../`), plus a "where the eval pieces live" map and the public-10 vs private-35
question-set split. The README's Benchmarks section shrinks to the headline 3-scorer table + one
interpretation paragraph + links (anchor `#benchmarks` kept — both in-README references still
resolve); layout tree gains the `evals/` line; the Status embedder note and the Running-tests
comment now point into `evals/`. `docs/architecture.md` gains a one-sentence pointer.
Decision recorded as `docs/decisions/ADR-024-evals-results-folder.md` + index row.

**Why:** the README is the door (readme-writer), and ~95 of its 413 lines were archive-depth
benchmark detail; the eval story also had no front door — harness in `src/doc_assistant/eval/`,
strategy/cases/baselines in `tests/eval/`, narrative only in the README. User directive named the
split and questioned the folder name; `evals/` over `benchmarks/` because "benchmarks" reads as
performance and the repo's own vocabulary is *eval* everywhere.

**Rejected:** `benchmarks/` as the name (vocabulary, above); moving `tests/eval/` wholesale into
the new folder (61 files reference those paths — script defaults, the CI ignore, code comments,
frozen append-only records; all churn, no gain); `docs/evals/` (the folder is audience-facing —
top-level GitHub visibility is the point). Full ledger: ADR-024.

**Opens:** `evals/` should accumulate future result write-ups (baseline data still goes to
`tests/eval/baselines/` per the locked-settings rule — narrative vs data). Separately scoped, not
built: a `tier: demo` extension of the public corpus from the 30papers.com list (the rumoured
Sutskever→Carmack 27) — ~17–18 are arXiv-pinnable; must stay OUT of the verified-10 benchmark
regime (extra distractor docs change retrieval difficulty and would invalidate every committed
baseline), so it needs a downloader flag + manifest tier before any papers are added.

---
## 2026-07-19 — Verify-the-app pass: root-caused the "6 pre-existing send2trash failures" → a live 500 bug (KI-22) + a dependency-presence guard test

**What:** Ran a full app-verification pass (all gates + a live $0 Ollama chat turn on the real
47-doc/16,039-chunk corpus). The turn worked end-to-end — correct SSE shape (step → 296 token →
result → done), 10 cited sources, 9 flagged claims, epistemics markers firing, `is_local:true`
`cost_usd:null` — and the concept graph served the ADR-018 numbers (13 nodes/19 edges/6 communities,
27 gaps = KI-17 reproducing). But the "6 pre-existing send2trash failures" the baton had carried as
*"venv drift, unrelated"* turned out to be a **real shipped-feature break**: `DELETE
/api/library/documents/{id}` 500s on every call because the declared base dep `send2trash>=2.1.0`
(`pyproject.toml:84`) was absent from the venv, imported lazily inside `library.delete_document`
(`library.py:330`) so it fails at call time, and the route catches only `RuntimeError` so it escapes
as a 500. Verified live with a nonexistent-id probe (deletes nothing) → 500 before, 404 after.
**Fix:** `uv pip install "send2trash>=2.1.0"` (venv-local, per-machine); suite went **1015 → 1021
passed, 0 failed** — first fully-green run in several sessions. **Added
`tests/unit/test_declared_dependencies.py`** (+35 tests): asserts every `[project].dependencies`
entry resolves via `importlib.metadata.version`, failing **by package name**, plus a pin on the exact
`from send2trash import send2trash` form. Recorded KI-22; the committed change is the guard test +
KNOWN_ISSUES/DEVLOG (the venv fix is gitignored `.venv` state).

**Why:** the failing tests were the suite correctly reporting a broken feature, but the cryptic
`ModuleNotFoundError`-from-monkeypatch shape made "test-infra noise" look plausible, so the misread
survived multiple sessions. A guard that fails by package name — "declared runtime dependency 'X' is
not installed … missing-dependency drift, not a test-infra flake" — makes the next such gap
unmissable and un-mislabellable.

**Rejected:** `uv sync` to restore the dep (would pull the multi-GB cu130 torch wheel, KI-3) —
installed the one pure-Python package instead; broadening the route's `except` to swallow the missing
dep (a missing hard dependency is a broken install, not a runtime condition to handle — the guard
test is the right layer); moving the lazy import below the unknown-id early return (papers over a
missing *required* dep without fixing it).

**Opens:** the guard only covers base `[project].dependencies`, not the `cpu`/`cu130`/`dev`/`packaging`
extras (an absent extra is expected on a lean install, so asserting it would false-positive); revisit
if an extras-drift bug ever bites. The baton's habit of labelling red tests "environmental" is worth a
cross-project atlas lesson (proposed, awaiting say-so).

---
## 2026-07-19 — Public docs refresh: README demo GIF + status/limitations truth-up, DEMO.md touch

**What:** (1) **Recorded a real demo GIF** and embedded it at the top of the README
(`docs/assets/provenote-demo.gif`, 23 frames, 1.73 MB, 960px): empty state → sample chip →
a genuinely streamed cited answer → the source side panel (with the per-claim review) → the
library grid → the concept-graph ego view. Recorded against the real 47-doc corpus on
**`ollama/llama3.1:8b` ($0 — provider switched via `/api/settings` and verified on
`/api/health` before any turn; KI-4)** by driving the dev app (API :8001 + Vite :1420) with
puppeteer-core + installed Chrome (the Browser pane's screenshot capture times out on this box —
known quirk, 2026-07-15 baton), frames assembled with Pillow. Recording tooling stays in the
session scratchpad (would add undeclared npm/Pillow deps if committed); pipeline documented in
agent memory. Side effects: 3 real 1-turn conversations now sit in this box's history (no DELETE
endpoint; harmless), and the provider switch surfaced a gitignore gap — the app's persisted
`data/settings.json` (U1c) was untracked-but-not-ignored and would have ridden into a public
commit; now gitignored as per-machine runtime state. (2) **README truth-up:** Status was frozen at 2026-07-02 ("concept graph
not yet usable", "gap detection blocked on RG-001", "712 tests") — now reflects the shipped
graph/gaps/markers/library/provider-switch stack, **1,015 tests**, the ADR-021/022/023
restructure, and links the scale review. New **Limitations** section (validated at ~50–100 docs
with the review's scale caveat, local-model ceilings, the KI-8/WE-7 marker-loss truth,
single-user design, Windows-first testing). "What it does" gains the concept-graph/markers/
library bullets; Project layout shows the db/ingest/knowledge/eval subpackages; decisions.md
references point at the index + archived monolith. (3) **DEMO.md** gains `just app`, the
Library/Graph walkthrough beat, and the GIF pointer.

**Why:** the README is the public face; it under-sold three shipped phases and over-claimed
nothing — but its status text was five iterations stale, and the user asked for a UI GIF.

**Rejected:** committing the GIF recorder into `scripts/` (undeclared puppeteer-core/Pillow
deps; revisit if the GIF needs regular regeneration); re-recording to purge the history rows
(real data, not worth touching `library.db`).

**Opens:** GIF re-record wanted after the next visual-identity pass; consider a
`docs/assets/` dark/light pair if the README ever needs theme-aware media.

---
## 2026-07-19 — C4 scale-robustness review: knowledge layer vs specs/ADRs at 0 docs and 10k docs (docs-only)

**What:** ran the user-directed in-depth review of the whole `knowledge/` layer against its own
specs/ADRs under four lenses (zero-doc, scale 0→10k, corpus-tuned constants, conformance) — four
independent read-only review passes (one per cluster), every finding required to quote the code
line it stands on; the seven highest-stakes claims re-verified by hand before publication (all
seven held). **Output: `docs/REVIEW_2026-07-19_scale-robustness.md`** — 36 findings + a
corpus-tuned-constants inventory + a P0/P1/P2 fix plan. Headlines: (a) zero-doc discipline
largely HELD (honest empty states everywhere; 2 crash edges in wiki/epistemics builders; the
contract is unpinned by tests); (b) every cluster has ≥1 corpus-linear-or-worse hot path
(unpaginated whole-corpus loads, a per-edge doc×doc Cartesian provenance product, O(chunks ×
concepts) full-recompute projection with a 512-pattern regex-cache cliff, O(n²) family cosine,
three unbounded LLM loops); (c) the over-optimize-on-current-corpus complaint is CONFIRMED and
localized — frozen Q1 `min_degree=3` whose docstring claims "corpus-derived", family threshold
0.86 **above bge's measured ceiling**, `contested` on `nc>=1` already marking 53.6% of chunks,
the monolith's recorded-wrong absolute-cosine 0.90 still the wiki default; (d) three conformance
breaks: curation hard-deletes vs ADR-018's demote (KI-20), the in-app rebuild never runs
`build_gaps` so the view serves stale gaps (KI-21), and KI-8's containment rationale is
arithmetically backwards — straddling chunks lose markers, they don't double-mark.

**Why:** the product contract (works at 0 docs, scales to 10k) had never been a review lens;
sessions kept tuning to the 47/76-doc corpora.

**Routed:** KNOWN_ISSUES **KI-18** (scale cliffs) / **KI-19** (tuned constants + LLM budgets) /
**KI-20** (delete-vs-demote) / **KI-21** (rebuild half-refresh) + KI-8 direction correction +
KI-17 fix-placement correction; RIGOR_TODO **RG-016..019** (each constant's owed experiment);
ROADMAP C4 done. **No code changed by this review** — fixes are follow-up sessions per the plan;
P2 constants are measurement-gated (never hand-tune).

**Rejected:** fixing "obvious" P0s inline this session (the session already carries the ADR-021/
022/023 restructure; review-then-fix in one diff would bury both); treating the review passes'
findings as publishable without an independent verification step (all 36 numbered findings were
consolidated; the 7 highest-stakes were re-verified line-by-line before anything was routed —
one of the 36, KW-8, is a positive no-defect trace, kept because it documents the 0-doc contract).

**Opens:** the P0 list is the natural next session (small, no eval needed); the LLM-budget policy
wants one ADR covering Node B / gap_suggest / wiki caps together.

---
## 2026-07-19 — ADR-023: knowledge/ subpackage — 11 corpus-derived modules out of the flat package

**What:** created `src/doc_assistant/knowledge/` and `git mv`'d the Phase-7 feature cluster into
it: `concept_curation`, `concept_graph_view`, `concept_semantics`, `concept_skeleton`,
`concept_skeleton_enrich`, `epistemics`, `gap_suggest`, `gaps`, `keyword_families`, `keywords`,
`wiki` (histories preserved). Package docstring states the layer's contract (Enrichment-Layer
sidecars; the answer path reads it, never depends on it). **49 files** rewritten to
`doc_assistant.knowledge.<mod>` (script-driven, word-boundary-safe, `--verify` pass shows 0
old-path references; covered `from doc_assistant.X import`, `from doc_assistant import X as y`,
`import doc_assistant.X`, and docstring prose; no monkeypatch-string forms existed). Living
docs/specs path-updated (KNOWN_ISSUES, RIGOR_TODO, ui-checklist, feature-concept-graph/-gap-
detection/-7d specs); `docs/architecture.md` module map + Mermaid gain the knowledge/ node;
`src/doc_assistant/CLAUDE.md` layout updated. Append-only records keep historical paths. Kept at
top level deliberately: `synthesis.py` (answer-path Chunk 2a), `tracking.py` (token infra),
`doc_vectors.py` (Phase-4 similarity input), the whole RAG path, and the existing `db/` /
`ingest/` / `eval/` — per the directive. ADR: `docs/decisions/ADR-023-knowledge-subpackage.md`.

**Why:** 63 modules, 40+ flat — "the concept graph" had no boundary to stay inside; the flat
listing stopped communicating the architecture (cpc §12: a real subsystem boundary earns its layer).

**Rejected:** naming it `features/` (generic, collides with `docs/features/`); compatibility
re-export shims at old paths (nothing external imports the package — cpc §12 no-speculative-
abstraction); touching `scripts/archive/` (frozen, unmaintained).

**Gate:** ruff ✓ (3 E501s fixed — two rewrite-lengthened docstrings + one pre-existing 100-char
line in `apps/api/main.py` that apps/-scoped habits had missed) · format ✓ · `mypy --strict src`
64 files ✓ · bandit 0 HIGH/MED ✓ · **pytest 1015 passed / 6 failed — byte-identical to the
pre-restructure failure set** (the known send2trash venv drift, `pyproject.toml` declares it,
`.venv` lacks it; `uv sync` fixes but pulls the multi-GB cu130 wheel — deliberately left).

**Opens:** none new; the Phase-D scale review (C4) now has a named review surface.

---
## 2026-07-19 — ADR-022: docs-system rationalization — index over monolith, DEVLOG fully inverted, per-artifact verdicts

**What:** decided which doc layers earn their place at scale and executed (ADR-022). (1)
**`docs/decisions.md` monolith (1578 lines) → frozen verbatim** at
`docs/archive/decisions-monolith.md` (`git mv`, header → archived/append-only, provenance note);
the path is now a **living ADR index** (one line per ADR-001..023 + the going-forward rule:
every decision = one ADR file + an index line). Re-scopes ADR-001 Step 4 — the planned ~50-file
split would have produced mostly-dead micro-ADRs duplicating ADR-002..021 and ROADMAP status. (2)
**DEVLOG ordering fixed once (completes ADR-001 Step 5):** the 103-entry oldest-first
`## Session:` historical block (2026-05-21 → 2026-07-04, lines 3323+) is now inverted below the
78-entry newest-first block — whole file strictly newest-first; entry **bodies verified
byte-identical** (script check), only `## Session: ` prefixes stripped; this entry is the logged
reformat note. (3) **Per-artifact verdicts recorded** (ADR-022 table): ADRs canonical · specs =
code-level contract layer · sprints = the delegated-execution mechanism (kept, roadmap_sync flow)
· `features/` = why-it-works layer, adopt for *frontier* features only, no backfill · loose
explainers stay · archive unchanged. (4) References updated: CONTEXT (×2), ROADMAP (header, table
intro, not-to-do, +C1–C4 rows, date bump), AGENTS.md (index + "top entries").

**Why:** the half-migrated state (living-labeled frozen monolith, hybrid DEVLOG order, unused
features/ layer) was context every session loaded and misread; scaling multiplies that cost.

**Rejected:** executing ADR-001 Step 4 as written (50 micro-ADRs — see ADR-022 option 1); deleting
`docs/features/` (cpc-init would re-lay it; the layer genuinely covers the hypothesis→outcome
record specs don't); bulk-materializing sprint contracts for planned rows (done per-row at
plan-start).

**Opens:** `docs/decisions.md` index is hand-maintained — a `cpc-generate` candidate if it rots;
FEATURE files start with the next frontier feature; the monolith's "Deferred Improvements" items
(wiki clustering threshold, coverage floor, SPECTER2 …) remain live backlog references into the
archive. Gate: docs_check 0 errors (2 pre-existing rule-12 warns clear with Phase-D edits).

---
## 2026-07-19 — ADR-021: cpc big-project layout — AGENTS.md entry + module CLAUDE.md files + vendored gates (this box)

**What:** adopted the cpc big-project variant (user request; ends ADR-001's conscious deferral of
cpc ADR-014). (1) **Entry layer:** new root `AGENTS.md` (canonical, tool-neutral; content ported
from the old `CLAUDE.md` + sub-module map + cpc ADR-020 keypoints table); `CLAUDE.md` is now a bare
`@AGENTS.md` stub, gate-enforced via `[entry] enforce_stub = true`. (2) **Module files** (≤40 lines
each): `src/doc_assistant/CLAUDE.md`, `apps/desktop/CLAUDE.md`, `apps/api/CLAUDE.md`,
`scripts/CLAUDE.md` — local traps only (wire-type drift, `--apply --enrich`, KI-4 provider guard),
globals by code. (3) **Conventions tooling separated from scripts:** cpc **1.2.3** vendored at
`tools/conventions/cpc/` via `cpc-init --profile standard` (run from the local cpc checkout at the
release tag; this box previously had NO vendored copy) + new `rungate.py` shim;
`.pre-commit-config.cpc.yaml` rewritten from pip-installing a pinned SHA of the **private remote**
to running the vendored copy (no network at hook time); justfile gains facade recipes
(`just check`/`lint`/`keypoint`). (4) `GLOSSARY.md` laid + **filled** (11 entries pinning the
Concept/Keyword/family/skeleton/gap vocabulary — the 2026-07-17/18 junk-labels trap is a
vocabulary-drift failure); `scripts/conventions.toml` refreshed to the 1.2.3 key set (project
values kept). ADR: `docs/decisions/ADR-021-adopt-cpc-big-project-layout.md`.

**Why:** 60+ backend modules across 4 real boundaries passed cpc §9's threshold; module-local traps
kept biting sessions that loaded only root context; this box's gate wiring was stale (pre-vendoring,
remote-dependent) and skewed vs the work box.

**Rejected:** staying CLAUDE.md-canonical (non-portable, permanent init-check deferral); a
CLAUDE.md→AGENTS.md symlink (degrades to plaintext on Windows clones — cpc ADR-014's own analysis);
moving `scripts/conventions.toml` out of `scripts/` (cpc `_config.py` resolves that exact path —
it is gate config, not a script; labeled instead).

**Opens:** `cpc-init-check` passes for the first time (kept on-call, not wired);
`src/doc_assistant/CLAUDE.md` must be updated when the knowledge-layer subpackage lands (same
session, ADR-023); Cowork project settings should be re-pointed at `AGENTS.md` (settings action,
outside the repo). Gate: `docs_check --strict` 0 errors; 3 pre-existing rule-12 date-bump warnings
(ROADMAP / KNOWN_ISSUES / RIGOR_TODO) left to clear with this session's later edits to those files.

---
## 2026-07-18 — ADR-020: share `RIGOR_TODO.md` via git (the two boxes held disjoint rigor trackers)

**What:** added `!.claude/RIGOR_TODO.md` to the `.gitignore` allowlist beside `CONTEXT.md` and
`KNOWN_ISSUES.md`, amending **ADR-001**'s `.claude/` contract. New
`docs/decisions/ADR-020-share-rigor-todo-via-git.md`; `CLAUDE.md`'s tracking line updated; the tracker's
own header rewritten from a "per-machine note" into a shared-file header carrying an item inventory and a
first-sync merge procedure. `SESSION.md` stays local — it genuinely *is* per-machine state.

**Why:** ADR-001 grouped the rigor tracker with the `SESSION.md` baton, but they are different kinds of
file. The baton records *"who worked last on this box"*; the rigor tracker records *validation debt of the
codebase*, which is true whichever machine you are sitting at. Grouping them let the two boxes accumulate
**disjoint item sets** for ~3 weeks.

**The failure is not hypothetical, and that is why this got fixed rather than noted.** **RG-014 has no
entry on this box** — while being cited as authority in **ADR-017, ADR-018, ADR-019,
`docs/specs/feature-concept-graph.md` and `docs/ui-checklist.md`** for "`single_source` is the strong,
low-volume gap signal". A week of design decisions rested on an item nobody working here could read — and
on 2026-07-18 that same verdict was found **not to transfer** across the ADR-018 vocabulary rescope, which
is exactly the bound a reader would have checked had the text been reachable. This copy holds
RG-001/008/009/010/011/012/013/015; RG-014, RG-007 and possibly RG-003/005/006 live only on the work box.

**Publication surface checked first — this repo is public.** Scanned for absolute user paths,
credentials, tokens and hostnames: **none**. Content is engineering measurements and box nicknames; the
one sensitive-sounding detail (a corporate TLS-MITM proxy) is **already public** in the committed
`KNOWN_ISSUES.md` KI-10. The publish decision stays the user's — staged, not committed.

**Rejected:** *keep it local + add a reconciliation ritual* — this **is** the status quo; the file has
carried a "still to reconcile against the work box" instruction since 2026-07-01 and it never happened, so
adding a second reminder supplies no mechanism (git is the mechanism). *Fold rigor items into
`KNOWN_ISSUES.md`* — conflates defects with validation-debt-on-work-believed-correct, and the
`rigor-gate` skill addresses `RIGOR_TODO.md` by name. *Move it to `docs/`* — breaks every existing
reference for nothing the allowlist entry does not already give.

**The hazard this change creates, and how it is contained:** the work box still holds the file as
**untracked and ignored**. On `git pull` git will refuse to clobber it — **that refusal is the safety
net and must not be forced past**. The tracker's header now carries the procedure (rename local copy
aside → pull → hand-merge the missing RG items → delete the temp) plus an explicit present-vs-missing
inventory, so the first sync is a **merge, not an overwrite**.

**Opens:** the merge itself, which can only be done on the work box — until then the shared copy is
incomplete **and says so in its own header**. Also surfaced, logged not fixed: the file's line *"The gate
(`rigor_gate.py`) fails while any `blocks-ship` item is `open`"* is **aspirational** — there is no
`scripts/rigor_gate.py` in this repo and neither pre-commit nor CI reference it. Sharing the file does not
make it enforcing; wiring a real gate is unticketed. **Staged; nothing committed (cpc §13).**

---
## 2026-07-18 — Stage-0 candidate ranking: triage mined keywords before promotion (read-only)

**What:** a **stage 0** for vocabulary curation in `concept_curation.py` — `rank_candidates()` (pure) +
`harvest_name_bigrams()` (pure) + `rank_keyword_candidates()` (impure wrapper), behind a read-only runner
`scripts/rank_candidates.py`. Orders mined keywords by **document reach** and reports three signals per
candidate: `docs` (distinct documents), `artifact` (reuses the existing deterministic `is_artifact`), and
an advisory `author?`. +13 tests. **Read-only — it ranks, never promotes, excludes, or writes.**

**Why:** the module's existing three stages prune a vocabulary that was *already* promoted; nothing ran
before promotion, which is why `--promote-all` was so destructive. The measurement that frames it:
**672 of 688 keywords (97.7%) appear in exactly one document** — the keyword extractor scores per-document
salience, not cross-document vocabulary, and a singleton keyword can never form a co-occurrence edge, so it
enters the skeleton as a permanently isolated node. Ranking by reach cuts the review set **688 → 16 (2.3%)**
without classifying anything.

**Ranking, not filtering — and that is a deliberate reaction to the same day's mistake.** Nothing is
auto-excluded: `pddl` is a legitimate 1-document concept, and the correction entry below records what
auto-exclusion produces on a multi-domain corpus. Signals order a human's review; they do not act.

**The author signal is reported honestly as weak.** `documents.authors` is free text that often holds a
*whole citation* — `"Omar Khatab and Matei Zaharia. 2020. ColBERT: Efficient…"` — so the field contains
paper **titles** as well as people. Measured live: 290 name bigrams harvested, **3 keywords flagged, only
1 a real author name** (`ziyang wang`); the others (`usage cards`, `responsibly reporting`) are title
fragments. **~1/3 precision → advisory only, never an auto-exclude.** Two guards keep it cheap rather than
catastrophic: only **capitalised** word pairs are harvested, and only **multi-token** candidates are ever
matched. That second guard is load-bearing and has its own test: **`bert` appears in 4 authors strings and
`colbert` in 1**, so a substring rule would silently drop two of the most important concepts in an IR
corpus. Noise classification stays with the existing `classify_noise()` LLM seam, whose prompt already
names author names as noise — free on Ollama, and not worth reimplementing deterministically at 1/3
precision.

**Rejected:** a hard `>=2 docs` gate (kills `pddl` and every legitimate single-source concept — and
`single_source` is the gap layer's *strongest* signal, so gating on reach would suppress the findings
ADR-004 exists to produce); substring matching against `documents.authors` (drops `bert`/`colbert` —
measured, not hypothesised); a new module (this is vocabulary quality, which `concept_curation.py` owns,
and it already had `is_artifact` to reuse); auto-promoting the top N (redesign **Decision 1** — the vocabulary
is curated by the user, never auto-extended).

**Live output on the real corpus (read-only, $0):** 688 candidates → 16 with reach >=2 → **8 unpromoted**,
of which 5 are real concepts the graph is currently missing — **`medical image segmentation` (3 docs)**,
`cajal`, `dice score`, `mamba`, `rag` — and 3 are correctly flagged artifacts (`18653 v1`, `10 18653 v1`,
`mrr 10`). It also confirms the morphological duplicates predicted from the pool: **`passage`/`passages`
and `mrr`/`mrr 10` are all promoted concepts** — `dedup_pairs()` (stage 3, already built, never run) is the
tool for those. Gate: ruff ✓ · format ✓ · `mypy --strict src` (63) ✓ · bandit 0 HIGH/MED ✓ ·
**1015 passed** (+13).

**Opens:** the ranker surfaces candidates but **promotion is still one-at-a-time via `seed_concepts
--promote`** — a batch review UI (or an `--llm` judged pass over just the top-ranked slice) is the natural
follow-up, now that the review set is 16 rather than 688. **25 of 47 documents still have zero concept
presence** — reach-ranking cannot fix that, because those documents' keywords are singletons *by
construction*; closing it needs per-domain seeding from document titles/abstracts, which is a different
instrument. `dedup_pairs()`/`classify_noise()` both remain built and unrun.
**Staged; nothing committed (cpc §13).**

---
## 2026-07-18 — CORRECTION to the ADR-018 entry: the 4 "junk" concepts are real specialist vocabulary

**What:** retracts one claim in the ADR-018 entry below — that `cre`, `dbs`, `ntsr1`, `pddl` are "4 junk
manual entries … worth curating out". **They are correctly curated domain concepts.** Nothing was removed.
The `set_graph_include(cid, False)` action item that rode on that claim is withdrawn.

**The evidence (traced to source, which is what the original claim skipped):**

| concept | alias | home document(s) | mentions |
|---|---|---|---|
| `cre` | Cre recombinase | mouse axonal-projection paper · "Neuroanatomy goes viral!" | **203** |
| `dbs` | deep brain stimulation | hypothalamic stimulation · dopamine/beta-oscillations | **134** |
| `ntsr1` | neurotensin receptor 1 | mouse whisker-cortex paper | 30 |
| `pddl` | planning domain definition language | hierarchical-planning paper | 46 |

Every one carries a correct expansion alias and real textual presence. **`cre` has more mentions than
`BM25`** (203 vs 137).

**Why the error happened, precisely:** the corpus spans at least four domains (IR/RAG · systems neuroscience ·
viral tracing/mouse genetics · AI planning), but the vocabulary was judged against the *one* domain the
session had been reading specs about. Lowercase acronyms sitting beside `BM25`/`dense retrieval` were
pattern-matched as extraction noise **without opening the documents they came from.**

**The supporting argument was also inverted.** "3 single-concept communities and 6 of the 15 gaps" was cited
as evidence of junkiness. That is the gap layer working: `isolated`/`single_source` on `pddl` means *"you own
exactly one AI-planning paper"* — a true coverage finding, which is the whole point of ADR-004. A correct
signal was read as a defect.

**This is a REPEAT.** The 2026-07-17 entry ("Manage view at scale scoped (PR-2.7)") reached the identical
conclusion one day earlier, in the same words — *"they are **mostly real specialist vocabulary, not junk**"*
(`16p11` = 16p11.2 truncated at the dot; `c57bl` = C57BL/6 across 7 docs; `va1v`/`dl5`/`osns` = Drosophila
glomeruli) — and set the rule **"demote, not delete: deleting real vocabulary isn't reversible-by-search."**
That rule was not consulted. Recorded as a trap in `docs/specs/feature-concept-graph.md` so the third
occurrence is cheaper than the second.

**What actually is wrong** (the finding the false one was hiding): **25 of 47 documents have zero concept
presence** — the vocabulary is too *small* and too *IR-skewed*, not too impure. Root cause of the poor
candidate pool: **672 of 688 keywords (97.7%) appear in exactly one document** — the extractor produces
per-document salience, not cross-document vocabulary, so `--promote-all` imported 672 document-specific
strings. Next PR ranks candidates instead of promoting them wholesale.

**Opens:** nothing removed, so no rebuild was needed; the 13-concept graph stands as measured.

---
## 2026-07-18 — ADR-018: scope the graph vocabulary with an opt-in `graph_include` flag (357 → 13 nodes)

**What:** added a nullable `graph_include` flag to `Concept` and filtered `concept_skeleton.load_concepts()`
on it (**ADR-018**). `library.list_keyword_families()` stays **unfiltered** — that asymmetry is the whole
decision. Creation paths follow one rule: `add_concept()` opts **in** (new `graph_include: bool = True`
param — the deliberate glossary path), `promote_keyword()` and `library.create_keyword_family()` opt
**out**. New `set_graph_include()` write surface + `backfill_graph_include()` (dry-run by default, touches
only `IS NULL` rows) behind a thin runner, `scripts/backfill_graph_include.py`. Migration is one append to
`db/migrations.py` `_ADDITIVE_COLUMNS` (+ an index — the filter runs on every build). +14 guard tests.

**Why:** **ADR-015's named "boundary risk" materialized.** Tag families and graph nodes are the same
`Concept` rows, and the two features want opposite things from that table — families want breadth, the
graph wants a small curated map. Measured on this box: **all 344** `source="keyword"` concepts share one
`created_at` (**2026-07-05**) — a single `seed_concepts.py --promote-all` run, against `promote_keyword`'s
own documented contract that a Keyword is *"a candidate only — never auto-written"*. The graph was 357
nodes of `'speckles'`/`'hyaline'`/`'13 intentionally omitted'`, and `single_source` was **224 of 302**
gaps — the exact signal RG-014 found strong *because* it was low-volume at 26 concepts.

**Polarity is the load-bearing half:** opt-in, so NULL reads as excluded and a new row never enters the
graph unbidden. Opt-out would let the identical regression recur on the next bulk operation; opt-in makes
re-flooding structurally impossible. A test asserts exactly that (`test_bulk_promotion_cannot_reflood`).

**Rejected:** filtering on the existing `source='manual'` (overloads a *provenance* field as a *curation*
control — they diverge the moment a graph-worthy concept arrives via `promote_keyword`, and the only fix
would be lying about its provenance); **deleting the 344** (destructive, cascades into
`concept_presence`/`concept_edges`/`gaps`, removes 344 keyword families from a shipped view to fix a
different feature — and being a data fix, the next `--promote-all` re-floods); splitting into two tables
(a real migration across four consumers to buy what a nullable column buys; revisit only if the two
vocabularies diverge in *shape*, not just membership).

**Applied + measured on the real corpus ($0, local Ollama, this box — 47 docs / 688 keywords):**
migration added the column (357 rows NULL) → backfill split **13 include / 344 exclude** → rebuild
`--apply --enrich --provider ollama` (the `--apply`-alone footgun avoided). **Graph 357 → 13 nodes**,
1534 → **19 edges**, 40 → **6 communities**, over 22 documents with presence. **Node B: 9 calls, 19/19
edges annotated, 63 stance assertions, 7 contested edges** — and the result reads as a real map
(`dense retrieval —[contrasts with]→ BM25`, `contrastive learning —[uses]→ hard negatives`) where 357
nodes read as noise. Directions: **7 contested / 6 stable / 0 superseded_trend**. **Gaps 302 → 15**
(isolated 3 · single_source 3 · thin_bridge 4 · under_connected 3 · unsourced_claim 2). Both ADR
guarantees verified live: graph vocabulary **13**, keyword families still **357**.
Gate: ruff ✓ · format ✓ · `mypy --strict src` (63) ✓ · bandit 0 HIGH/MED ✓ · **1002 passed** (+14; the
6 `send2trash` failures are pre-existing venv drift, unrelated to this diff — the dep is declared in
`pyproject.toml:84` but absent from `.venv`).

**Found in passing, logged not fixed — `.claude/KNOWN_ISSUES.md` KI-17:** the rescope stranded **10**
stochastic `suggested_concept` gap rows whose concept left the vocabulary. `build_gaps` delete-and-
replaces *deterministic* rows but **status-preserving-upserts** stochastic ones with no reconcile pass,
so they are immortal — `load_graph_view()` serves **27** gaps against 13 nodes while the runner reports
15. Invisible in PR-G2a's index (it resolves by concept), but it breaks **PR-G2b**, where every gap needs
a triage action. Fix belongs with the ADR-017 C1 override sidecar.

**Opens:** **the gap distribution must be re-derived before PR-G2b** — its "strong kinds first" ordering
rests on RG-014's verdict at 26 concepts on the *other* box's 76-doc corpus; at 13 concepts here the
kinds are nearly flat (4/3/3/3/2), so `single_source` is no longer self-evidently the headline. **No
curation UI:** opting a concept in is CLI-only until a follow-up adds the toggle to Manage-keywords (its
natural home — keeps ADR-017 A1 intact, the graph still never writes the vocabulary). The 13 included
concepts contain 4 junk manual entries (`cre`, `dbs`, `ntsr1`, `pddl`, added 2026-07-05) that now form
three single-concept communities and account for 6 of the 15 gaps — worth curating out. `--promote-all`
is now harmless to the graph but still violates `promote_keyword`'s candidate-only contract.
**Staged; nothing committed (cpc §13).**

---
## 2026-07-17 — Concept graph PR-G2a: the view — concept index + gap lens + ego graph + chunk nav (frontend)

**What:** built PR-G2a of `feature-concept-graph.md` — the third top-level view that renders the PR-G1 read
model. A **destination, not a modal** (`mode` union widened `'chat'|'library'` → `+'graph'` in the 4 measured
places + a third rail tab). New `lib/ConceptGraph.svelte`: a searchable **concept index** on the left (label ·
gap badge · doc count) with a **"Gaps only" lens** and an **"Include under-connected"** opt-in; selecting a
concept opens a depth-1 **ego graph** (hand-rolled SVG, no dependency) + a details panel that navigates concept
→ document → the chunks it appears in. New `lib/forceLayout.ts`: pure, seeded (mulberry32 + phyllotaxis init +
Fruchterman–Reingold to convergence, then fit-to-viewBox) — deterministic and epsilon-guarded so no coordinate
can be NaN. `types.ts`/`api.ts` mirror the 7 PR-G1 payloads + 4 client fns (404 → `null`, the normal first
run). `app.css` gains a **12-hue categorical community palette** (both themes) cycled by `community % 12`, plus
`--graph-edge`/`--graph-node-stroke` derived once from `--fg` via `color-mix` (late-bound, tracks both themes).
`Icon.svelte` gains the Lucide `waypoints` glyph. Deep-links only: node → `openDocument()` (Library), "Edit"
→ Manage-keywords (**ADR-017 A1 — the graph never writes the vocabulary**). Staleness banner + empty state
share one **Rebuild** affordance (202 + poll, ADR-017 B1). Read-only; `$0`.

**Why:** ADR-004's north star is gap detection; RG-014 found the strong signal (`single_source`) is
**list-shaped**, so the index — not the graph — is the home, and the graph earns its place as the navigation
surface. Ego-first (B3) bounds the hairball (`Embeddings` touches 80% of the graph) to one neighbourhood.

**Ordering honours the verdict:** `single_source` leads (danger tone); `under_connected` is **off by default**
behind a toggle (it is graph-degree noise at n=26). Gaps badge **nodes**, stance will colour **edges** (B9) →
no collision when Node B lands (PR-G4).

**Verified live ($0/offline, real corpus, via read_page + javascript_tool — the SVG DOM is the only assertable
surface; screenshots time out on this box):** index shows **26 concepts**, the 3 `single_source` true positives
(PHATE/Res2Net/SBERT) lead in danger tone, gap lens = **8** (under_connected hidden), → **10** when opted in.
Res2Net ego = 3 nodes/3 edges, Embeddings (degree 20) ego = **21 nodes** — **no NaN** in any cx/cy/x1/y1, all
in-viewBox, no collapse. **Determinism: identical positions across a re-render.** 3 distinct community fills;
theme flip changes fill + the `color-mix` edge stroke. Zoom clamps **0.4↔3.0**. Res2Net → its 1 document →
"Mentioned in 25 chunks" → **Open in Library** switches mode + opens the doc; **Edit** → Manage-keywords.
375px: **no horizontal overflow** (index + ego). **Gate:** `svelte-check` 0/0 (133 files); `vite build` clean
(157 modules); **still one runtime dep (`marked`)** — the layout is hand-rolled.

**Rejected:** a graph library (cytoscape/d3 — 4× the bundle, an eval-using lib breaks only in the packaged
Tauri CSP, and with zero frontend tests the SVG DOM must stay assertable); a modal (6 hand-rolled scrim dialogs
exist, all capped transient tasks — a graph is a destination); `weight`-as-thickness (range 2.377–2.949, flat);
a provenance legend (one state); contested/superseded colour (renders nothing until PR-G4); animating the
simulation (run off the render path, draw statically — determinism is the safety net); persisting pan (it is
position-specific and resets on re-centre; only zoom, a real preference, persists); a `color-mix` hue wheel for
communities (a rotated hue can land on low-contrast yellow in light theme — a fixed per-theme ramp controls
contrast, and colour is a positional grouping hint so cycling past 12 is harmless).

**Opens:** PR-G2b (gaps as a first-class destination + triage via the ADR-017 C1 override sidecar — `status` is
still the raw row value on the wire); the `unsourced_claim` count stays approximate until the claim-segmenter
heading bug is fixed (surface it, don't present it as precise); PR-G4 (Node B on the RTX box) unblocks the
reserved edge-stance encoding; community palette cycles (not collides) past 12 — fine for now, revisit if a
real corpus shows >12 communities. **PR-G1 is still staged/uncommitted — this builds on it; both await review.**
**Staged; nothing committed (cpc §13).**

---
## 2026-07-17 — Concept graph PR-G1: serve the read model (load_skeleton + load_gaps + 4 routes)

**What:** built PR-G1 of `feature-concept-graph.md` — the backend read model the graph view consumes.
**Staged, NOT committed.** New `src/doc_assistant/concept_graph_view.py` (`GraphView`, `GraphStaleness`,
`load_graph_view`, `load_concept_presence`); `concept_skeleton.load_skeleton()`; `gaps.load_gaps()`; six
payloads in `apps/api/models.py`; four thin routes in `apps/api/main.py`; 16 tests in
`tests/integration/test_concept_graph_api.py`.

**Why the pieces landed where they did.** `load_skeleton` is the **read half of `write_skeleton`**, so it sits
beside it in `concept_skeleton.py`; `load_gaps` likewise mirrors the row writers inside `gaps.py`, because
that module owns the gap domain. The *assembly* (skeleton + gaps + staleness) got its **own** module rather
than joining `library.py`: `library.py` serves documents/keywords, and the graph is a distinct top-level view
— putting it there would have grown an already-large module with an unrelated concern. `apps/` stays a shell:
every route is a pass-through, and the loader/assembly/staleness reasoning all lives in `src/`.

**Two decisions worth the record.** (1) **One wire id space — concept UUIDs everywhere** (node ids, edge
endpoints, gap anchors, community members), with `label` **only** on the node. The spec demanded a choice
because mixing ids and labels across a boundary is *exactly* what made KI-15 match nothing silently; the live
check asserts it (70/70 edges, 14/14 gaps resolve). (2) **Presence is served per-concept, not bulk** — a
**deliberate deviation** from the spec's "graph → skeleton + gaps + presence". Ego-first (B3) renders one
neighbourhood at a time, and bulk-shipping 1781 chunk keys for a 26-node graph is waste that scales badly to
357. `doc_ids` already rides each node, so only chunk-level navigation needs the extra call.

**Distinguishing "never built" from "broken" was the load-bearing detail.** `skeleton.json` is gitignored and
regenerable, so **absent is the NORMAL first run** → `load_skeleton` returns `None` and the route answers
**404 with a rebuild hint**, not a 500 and not a fake empty graph. But a file that *exists and won't parse* is
a corrupt artifact — a different state — so that **raises** (`raise RuntimeError(...) from e`); returning
`None` there would invite a rebuild that masks the real problem.

**Verified live on the real corpus ($0/offline).** `GET /api/concepts/graph` → **26 nodes / 70 edges / 3
communities / 14 gaps**, `graph_version b59a4aa6afa77978`, `stale:false`, every `relation` `null` (Node B
never run). Presence: `Embeddings` → **32 docs / 283 chunk keys**; unknown → `[]`. Rebuild: **202** → poll →
`done`, returning the **identical graph_version** — determinism proven end-to-end through the API, not just in
a unit test. Empty state: skeleton moved aside → **404** → restored → 200. Gates: ruff, ruff format,
`mypy --strict src`, bandit all clean; **full suite 994 passed** (was 977; **+16 new, 0 regressions**).

**Rejected:** (a) *the assembly in `library.py`* — see above. (b) *bulk presence* — see above. (c) *a blocking
rebuild route* — 7.1s on the event loop, when `POST /api/ingest`'s 202+poll is the repo's established shape for
exactly this (ADR-017 B1). (d) *`None` on a corrupt skeleton* — conflates two states. (e) *seeding the FK
referents in one `session_scope`* — these models carry no `relationship()`, so a single flush does not reliably
order the parent insert before the child and the FK trips; the test seeder commits the referent separately.
(f) *"fixing" `apps/api/main.py:734`'s pre-existing E501* — **CI lints `src/` + `tests/` only** (confirmed in
`.pre-commit-config.yaml`: "Scope = src/ + tests/ ONLY, to mirror CI exactly"), so `apps/` is deliberately out
of ruff's scope and that line is neither a gate failure nor mine.

**Two self-inflicted bugs caught before they shipped, both by measuring rather than reasoning:** my first
staleness diff was `sk_ids.symmetric_difference(db_ids) & db_ids`, a convoluted spelling of `db_ids - sk_ids`
— simplified. And the `Write` tool captured a literal `</content>` tag into four files (the module, the ADR,
the spec, the test); caught by a `SyntaxError` on the first import, stripped from all four.

**Opens:** PR-G2a (the view) is next and needs the **palette** decision (3 non-semantic hues, 3 communities —
luck, not headroom). The **`GET /api/concepts/graph` payload can't paginate** — fine at 44 KB / 26 nodes,
~600 KB at 357; state a position before the vocabulary grows. **RG-014 stays open** (the gap payload is ~50%
precise; the spec's "don't lead with `under_connected`" is the mitigation, and PR-G2a must honour it). ADR-017
C1's **gap-triage override sidecar is NOT built** — that is PR-G2b, and `GapPayload.status` is currently the
raw row value, not the effective one.

---
## 2026-07-17 — Wrote ADR-017 (concept-graph UI boundaries) + docs/specs/feature-concept-graph.md (docs-only)

**What:** authored the two artifacts the concept-graph grill routed to. **`docs/decisions/ADR-017-concept-graph-
ui-boundaries.md`** (new, accepted) and **`docs/specs/feature-concept-graph.md`** (new, design-locked, not
built). Docs-only; no code. `docs_check --strict` 0/0.

**Why an ADR at all:** the graph is the first UI over the curated vocabulary that is **not** curation, and it
crosses three boundaries that each had no owner — places where a plausible design quietly breaks a shipped
guarantee. ADR-017 decides **only** the boundaries; the spec owns the contract (ADR-019's "if an ADR owns the
*why*, the spec is the *how* and must not re-litigate it").

**The three decisions.** **A1 — read-only for the vocabulary + deep-link to Manage-keywords:** you edit the
*source* and regenerate, never the derived artifact; the "a keyword belongs to at most one family" invariant
keeps **one** home while **PR-2.5 is still repairing it** (rename → duplicate `Concept` labels → corrupts
`promote_keyword` repo-wide), and a second writer onto known-broken paths isn't worth a convenience. **B1 —
in-app Rebuild (202 + poll), CLI runner stays canonical:** 7.1 s is a button, and `POST /api/ingest` + a status
poll is the established pattern for this repo's *largest* derived build. The reasoning that mattered: **the
Enrichment-Layer Pattern constrains *what* derived data is (regenerable, sidecar, never mutates source) — not
*who is allowed to press go*.** **C1 — gap triage as a user-override sidecar keyed on `(concept_id, kind)`:** a
dismissal is a **user judgment**, not derived data, so it must not live in a table that is deleted and rebuilt
from the skeleton; `GapRow.status` becomes **effective = override ?? "surfaced"**, which is exactly ADR-013 A2's
shape (auto on the record, override in a sidecar). This keeps ADR-004's regenerable guarantee intact.

**C1 exists because RG-014 disproved the grill's own premise.** B14 had resolved "dismiss/promote from the
view" partly on *"`build_gaps` deliberately persists status across rebuilds"*. It does not — for deterministic
gaps (`gaps.py:257` deletes and replaces them; only the stochastic path is status-preserving), and **all 14
live gaps are deterministic**. The ADR records the corrected reasoning rather than the comfortable one.

**The spec's load-bearing constraint is the RG-014 verdict, not the design.** The gap payload is **~50%
precise**, and — the finding that shapes every screen — **the strong kinds are LIST-shaped (`single_source`,
`unsourced_claim`) while the weak kinds are GRAPH-shaped (`under_connected`, `thin_bridge`)**. So the spec
**leads with the index + gap list, defaults `under_connected` OFF**, and positions the graph as the *navigation
and context* surface rather than the dashboard. A spec that led with the pretty part would have shipped the
noise: `under_connected` is the **largest** kind and flags `Tractography` (10 docs) and `Motor control` (13
docs) — among the best-sourced concepts — as gaps.

**Rejected (ADR):** *graph writes in place* — every write instantly stales the view you're reading, so it lies
until rebuilt; *the graph replacing Manage-keywords* — a force layout is a poor bulk-alias editor and it
discards a view mid-hardening; *CLI-only rebuild* — breaks the acquire loop by ejecting the user to a terminal
mid-research; *auto-rebuild on staleness* — spends 7.1 s unasked **and destroys the seeded-determinism property
that is the feature's only verification surface**; *making the deterministic write path status-preserving* —
`gaps` would become a hybrid of derived + user data, forfeiting the regenerable property ADR-004 relies on, and
it needs a reconcile rule for gaps that stop firing; *no triage for deterministic gaps* — 0 stochastic gaps
exist, so that means no triage at all and a permanent nag.

**Rejected (spec):** *the cpc `SPEC-000` executor-brief template* — the house shape for feature contracts is
`feature-*.md` (Status/Owner → decision → grounding → carve → parked → ledger), per `feature-tag-families.md`;
SPEC-NNN is for delegated sprint briefs. *Leading with the graph* — see the verdict. *Shipping
contested/superseded encoding* — dead until PR-G4. *`weight` as edge thickness* (2.377–2.949, flat) and *a
provenance legend* (one state).

**Opens:** the spec is **design-locked, not built** — PR-G1 (serve; write the missing `load_skeleton()`) is
next. **RG-014 stays open** until the spec's narrowed claim is exercised and the two live defects land: the
**claim-segmenter heading bug** (12 of 61 `unsupported` claims are markdown headings — and they feed the
failure-tag gates driving the self-improvement loop) and the **`under_connected` corpus-vs-vocabulary guard**.
PR-G4 (Node B on the RTX box, $0) still gates every epistemic encoding. B13 (the acquire loop) needs its own
ADR when picked up.

---
## 2026-07-17 — RG-014: ran build_gaps --apply on a fresh skeleton; VERDICT ~50% precision. B1 narrows; 3 defects found

**What:** closed out RG-014's procedure at the user's request — *"run build_gaps --apply and check the 14
findings are real"*. **Data-only writes (both gitignored): `data/skeleton/skeleton.json` + the `concept_edges`
/ `concept_presence` / `gaps` sidecars. No source changed.** All $0/offline.

**Procedure.** Verified first that **0 edges carried Node-B annotation** (so a rebuild loses nothing) →
`build_concept_skeleton --apply` → fresh **`b59a4aa6afa77978`** replacing the stale `055312c8c15a7e69` →
`build_gaps` dry on the fresh skeleton: **still 14** (*the findings are stable across the rebuild — they were
not artifacts of staleness*) → `--apply`: **14 rows persisted** → re-run: **14 rows, no duplication —
idempotence confirmed.**

**VERDICT: 8 of 14 defensible, 6 of 14 noise/duplicate/misleading. B1 survives but NARROWS.**
- ✅ **`single_source` (3) — TRUE POSITIVE and the whole product thesis.** `Res2Net` appears **only** in the
  Res2Net paper; `SBERT` **only** in the Sentence-BERT paper; `PHATE` **only** in one neurodevelopment paper.
  Each is a method known *solely from its originating source* — no independent evaluation or replication in the
  corpus. This is exactly the user's *"technically, having a single source is not good"*, and it is directly
  actionable via B13 (acquire corroboration).
- ⚠️ **`unsourced_claim` (4) — real signal, ~33% contaminated input.** Aggregation is sound (`RAG` 15
  unsupported claims, `BM25` 5) and sampled prose is genuinely uncited. **But 20 of the 61 underlying
  `unsupported` claims are not prose claims at all — 12 are MARKDOWN HEADINGS** (`"# Dense Passage Retrieval
  (DPR) and Its Advantages Over BM25"`) **+ 8 fragments.** A heading can never cite, so it is *structurally*
  always unsupported. (43/61 also predate the 2026-07-14 parser fix.)
- ❌ **`under_connected` (5) — mostly noise at n=26, and it's the LARGEST kind.** It measures **graph degree**,
  which at a 26-concept vocabulary is dominated by **vocabulary sparsity, not corpus coverage**. **`Tractography`
  (10 docs, degree 2)** and **`Motor control` (13 docs, degree 2)** are among the corpus's **best-sourced**
  concepts and are flagged as gaps. **It conflates "my corpus is thin on X" with "my vocabulary is too small
  for X to co-occur with anything."** 2 of 5 duplicate `single_source`; only `MedSAM` is defensible.
- ⚠️ **`thin_bridge` (2) — redundant + half-misleading.** Both derive from **one edge** (`MedSAM` ↔
  `Embeddings`) and it flags **both endpoints**, so **`Embeddings` — the most-connected node in the graph
  (degree 20/25, 32 docs) — is reported as a "thin bridge" gap.**

**The finding that most affects the spec: the STRONG kinds are LIST-shaped (`single_source`,
`unsourced_claim`); the WEAK kinds are GRAPH-shaped (`under_connected`, `thin_bridge`).** B1 does **not**
reverse — the corroboration job stands on the two strong kinds, and the graph's *navigation* value (B7,
concept→doc→chunk, 1781/1781 verified) is independent — **but the graph is not the primary renderer for the gap
payload; a list is. The spec must not lead with `under_connected`.**

**⚠️ B14 REOPENS — the grill's reasoning was wrong.** B14 resolved "dismiss/promote from the view" partly
because *"`build_gaps` deliberately persists status across rebuilds"*. **It does not — for deterministic gaps.**
`gaps.py:257`: `session.execute(delete(GapRow).where(GapRow.determinism == "deterministic"))` — deterministic
rows are **delete-and-replace**; only `_write_stochastic_gap_rows` (`:273`) does the *"status-preserving"*
upsert that "never deletes". **Verified live: a `dismissed` deterministic gap reset to `surfaced` on the next
run.** **All 14 of today's gaps are deterministic** (stochastic = 0) → **dismissing any of them is futile, and
rebuild is part of the acquire loop.** I over-read the docstring's "stochastic rows persist their status" as
"rows persist their status". **Lesson: when a docstring qualifies a noun, the qualifier is load-bearing.**

**Rejected:** (a) *running `build_gaps --apply` against the stale skeleton* (the literal request) — it would
stamp findings with a dead `graph_version` and answer a question about an out-of-date graph; rebuilt first
(safe: writes are gitignored, idempotent, and 0 annotated edges existed). (b) *closing RG-014* — the run
answered "do they persist / are they real" (yes / ~half), but the spec hasn't absorbed the verdict and two live
defects remain. (c) *reversing B1* — `single_source` + `unsourced_claim` carry the job; the weak kinds are a
detector-tuning problem, not a premise failure. (d) *treating `under_connected` as permanently broken* — it is
noise **at n=26**; it should improve as the vocabulary grows (`--promote-all` → ~86).

**Opens:** **(a) BUG — the claim segmenter counts markdown headings as claims** → 12 permanent false
`unsupported` markers → false gaps **and they feed the failure-tag gates that drive the self-improvement loop**
(new checklist row). **(b) deterministic gap triage needs a durable store** keyed on `(concept_id, kind)`,
mirroring the stochastic upsert — **decide in ADR-017.** **(c) `under_connected` needs a corpus-vs-vocabulary
guard** (gate on doc-count, or defer the kind until the vocabulary is larger). Live state now: skeleton
`b59a4aa6afa77978`, `gaps` = **14 rows**, all `surfaced`, all deterministic.

---
## 2026-07-17 — GRILLED the concept graph (grill-me): 12 branches, 11 resolved / 1 parked; the root question was overturned by the repo (docs-only)

**What:** ran `grill-me` on the concept-graph view before its spec. **12 branches: 11 RESOLVED, 1 PARKED, 0
open.** Full ledger in the session baton; the durable half is in `docs/ui-checklist.md` §3. **Docs-only; no
code; nothing run but free read-only dry-runs.** **Routed, not authored** (the grill doesn't write artifacts):
a **new ADR-017 `concept-graph-ui-boundaries`** (B5/B6 read-only-for-the-vocabulary + deep-link; B8 an API
caller of an enrichment runner; B14 gap-status writes) cross-referencing ADR-015 (which reserved this track) +
ADR-004 (which owns gaps); a **new `feature-concept-graph` spec**; **B13 → the External literature discovery
row**; **RG-014 → `.claude/RIGOR_TODO.md`**.

**Why it mattered: the grill overturned its own root question by reading the repo instead of asking the user.**
I opened B1 ("what job does this do?") expecting to argue against a pretty-but-useless Obsidian clone.
**ADR-004 answered first:** *"Phase 7's stated purpose is **gap detection** — and the project's north-star reason
for it is to surface what the user (and the LLM) cannot see: concepts the corpus under-supports, claims it
cannot source, and directions for exploration the user did not think to look."* And the gap layer is **BUILT +
Ollama-validated with a recorded baseline** (Tier-1 + Tier-2a floor SPRINT-002/G2; ceiling SPRINT-005/G5) —
`gaps.py`, `gap_suggest.py`, `scripts/build_gaps.py` — **with 0 rows and ZERO UI**. A **free dry-run found 14
real gaps** on the live corpus (`under_connected` 5 · `unsourced_claim` 4 · `single_source` 3 · `thin_bridge`
2). The user's own framing independently matched the detector — *"technically, having a single source is not
good"* — which is exactly what `single_source` measures. **So the graph has a payload today; it is not
decoration.** **Lesson: read the archived spec before asking the user what a feature is for.**

**Measurements that decided branches (all free/offline):** **B3** — the hairball is real **today**: 22%
density, mean degree 5.4, but **`Embeddings` has degree 20/25 = touches 80% of the graph** (depth-1 ego = 81%
of all nodes, on 32/76 docs) while the **median ego is 6 nodes** → **ego-first, depth-1**. *(Side finding: a
degree-20 node on 32 docs means `Embeddings` is too generic to be a good concept — the graph reveals
vocabulary quality for free.)* **B7** — **1781/1781 (100%)** of `concept_presence.chunk_keys` resolve against
the live index (ADR-4 key `{document_id}:p{parent_index}`) → concept→doc→chunk navigation is real. *(My first
attempt reported 0/1781 — I'd built `doc:idx` instead of `doc:pN`; caught it rather than shipping it.)* **B8**
— **rebuild = 7.1 s, zero-LLM, deterministic** → a button, not a batch job; and it **confirmed the artifact is
stale**: `graph_version` `055312c8c15a7e69` → **`b59a4aa6afa77978`**, with **`doc_years` now present** (so
`superseded_trend` becomes possible once stance lands). **B9** — **every `Gap` is anchored to `concept_id`**
(even `thin_bridge`, whose endpoints live in `evidence.fact_ids`) while stance is an **edge** property ⇒ **gaps
encode on nodes, stance on edges, no palette collision** — and B3 (≤21 rendered nodes) defused the 3-hue
community gap entirely. **B14** — `GapStatus = surfaced|promoted|dismissed` **already exists** and `build_gaps`
**deliberately persists status across rebuilds**: the sidecar was *designed* for triage.

**Rejected:** (a) *a global map* (B3) — hub-dominated at n=26 already, and `--promote-all` is one command from
~86 nodes. (b) *top-N-by-degree capping* — it hides exactly the low-degree nodes the gap layer flags as
`under_connected`, i.e. it hides the findings. (c) *graph writes to the vocabulary* (B5) — you edit the source
and regenerate, never the derived artifact; and a second writer onto rows whose invariants **PR-2.5 is
currently repairing** (D1 rename → duplicate labels → corrupts `promote_keyword`) is reckless. (d) *the graph
replacing Manage-keywords* — a force layout is a poor bulk-alias editor. (e) *auto-rebuild on staleness* — it
spends 7.1 s unasked **and breaks the seeded-determinism verification story, which is the only test surface**.
(f) *CLI-only rebuild* — it breaks the acquire loop by dumping the user into a terminal mid-research; the app
**already** triggers its biggest derived build via `POST /api/ingest` **202 + status-poll**, so the precedent
exists and the CLI runner stays canonical (Enrichment-Layer intact). (g) *read-only gaps* — a view that can't
say "I know, that's fine" becomes a permanent nag.

**Opens / parked.** **B13 (parked, needs its own ADR):** the **gap → acquisition loop** — *"download and find
more information to complete the graph… we will need a provider list, and a quality list"*. It closes ADR-004's
loop and merges with the **External literature discovery** row; **transport is already spiked** (stdlib urllib →
Crossref, 25/25). **CONSTRAINT ON THE SPEC: model a gap as an object with an ACTION SLOT** (`GapStatus.promoted`
is that slot) so it attaches without rework. **RG-014 (blocks-ship): the 14 gaps are a DRY-RUN claim —
`build_gaps --apply` has NEVER been run and nobody has read the findings.** Run it on a *fresh* skeleton,
confirm the sidecar persists + status survives a re-run, and judge signal-vs-noise **before** the spec — **B1
reverses if it's noise.** (Note **RG-001** is still `open`/`blocks-ship` and its close instructions name
`build_concept_graph --apply`, **a command that no longer exists** — the module was retired in KI-7/SPRINT-001;
that entry needs re-pointing at the skeleton.) Node B on the RTX box (PR-G4) still gates every epistemic
encoding.

---
## 2026-07-17 — Planned the concept graph (PR-G1/G2/G4); RAG blast-radius + chat-mode boundary measured; 2 self-corrections (docs-only)

**What:** planned the **concept-graph view** in depth (3 explorers + a design pass, all grounded on live code +
the real corpus) and answered two user challenges with measurements. **Docs-only; no code; nothing run but free
read-only dry-runs.** Plan detail lives in a **local, uncommitted** plan file; everything load-bearing is
duplicated into `docs/ui-checklist.md` §3.

**Two self-corrections to my own uncommitted entry below — both were load-bearing and both are now fixed in
place** (the entry never landed, so correcting beats shipping a known-false claim + an erratum): **(1) Node B is
BUILT, not "unbuilt", and $0, not "paid"** — see the corrected text below; **(2) "citations 3918" is the WRONG
table for citation-highlighting** — those are bibliography refs with **0 resolved** (`target_document_id` → 0);
answer-citations are `answer_claims` 168. **Root cause of (1): I grepped `scripts/` for a FILE named like a
stance runner and concluded the feature didn't exist. The runner is a FLAG (`--enrich`) on the existing Node-A
script. Lesson: absence of a filename is not absence of a feature — grep the call graph, not the directory.**

**Concept graph — decisions (user).** (1) **Hand-rolled SVG + a seeded force layout, NO dependency.** The
reasons compound: 26 nodes (500 worst case) needs no library; the Tauri CSP is `default-src 'self'` with **no
`unsafe-eval`**, so an eval-using lib breaks **only in the packaged build** (dev-mode Vite won't catch it); the
frontend is a deliberate **1-runtime-dep** artifact (`marked`) with `dist` at **101 KB** (cytoscape ~400 KB =
4× the whole bundle) and a stated no-CDN/vendored ethos; and decisively — **zero frontend tests exist and
screenshots time out on this box, so SVG's DOM is the only verification surface**; canvas/WebGL would render
the feature literally unassertable. (2) **Association-only** — measured: **all 26 nodes are `unique`/`stable`,
`contested_edges()` → `[]`**, so contested colouring is dead UI; reserve `--danger`/`--warn-fg` for stance and
ship nothing that renders empty. (3) **One vocabulary — a family IS a concept**; don't filter by `source`.
**The real defect isn't the shared rows (ADR-015 chose that deliberately) — it's that the skeleton is a build
artifact, so the graph silently lags user edits.** Surface staleness + a Rebuild action instead (inform, don't
block). **Resolved:** full view, **not a modal** (there is no reusable modal shell — it's hand-rolled in **6**
components, all `min(84vh,620px)`-capped *transient tasks*; a graph is a *destination*); the shell change is
small (the `'chat'|'library'` union appears in exactly **4** places).

**Why the graph is ready when the rest of the phase isn't:** `concepts` 26 · `concept_edges` **70** ·
`concept_presence` 222 · 3 communities · `skeleton.json` carries a **complete render model with layout signal
precomputed** (`community`→colour, `degree`→radius, `doc_ids`→click-through). ADR-015 **reserved this track by
name** and PR-1/PR-2 made `Concept` a first-class UI citizen. It is greenfield — `concept_graph.py` was retired
(KI-7) leaving no dead code.

**Rejected:** (a) *d3-force/cytoscape/sigma* — see (1); a dep would also want its own ADR. (b) *canvas/WebGL* —
unassertable on this box, which is the binding constraint, not perf. (c) *filtering the graph by
`source='manual'`* — ADR-015 earmarked the column, but promotion is how the vocabulary is *meant* to grow, so
filtering it out is backwards. (d) *shipping contested/superseded colouring now* — 0 signal. (e) *leaning on
`weight` for edge thickness* — range is **2.377–2.949**, nearly flat. (f) *a provenance legend* — all 70 edges
are `{cooccurrence, similarity}` and **0 citations resolve**, so it would have exactly one state. (g)
*persisting anything against a community id* — they're **positional, not identity** (add a concept → they
renumber). (h) *animating the force sim* — run seeded to convergence off the render path; **determinism makes
the assertions non-flaky**, which is the point.

**User challenge 1 — "I don't see the problem with changing top-k; the issue is the embedding." MEASURED: they
are right, and it exposed a real gap.** `TOP_K` is **already** per-session, non-persistent **and bounded**
(`ge=1, le=CANDIDATE_K` → [1,20], **422, never a silent clamp**) — there was never a problem. The embedding *is*
the only catastrophic knob, but **not by the assumed mechanism**: collections are **namespaced per model**, so
switching reads an **empty** collection → `warning("empty_index")` → "the sources don't contain the answer". The
real footgun is narrower: `_LEGACY_COLLECTION="langchain"` is bound to **whatever `DEFAULT_MODEL` is** and both
registered models are **768-dim**, so changing `DEFAULT_MODEL` would **silently inherit bge-base's vectors** —
Chroma accepts the dimension and nothing detects it. **THE GAP: `EMBEDDING_MODEL` is in NEITHER ADR-010's split
NOR CONTEXT's locked table — the most dangerous knob is documented as "swappable" and governed by nothing,
while harmless `TOP_K` is locked.** Why: **ADR-010 sorts by *cost to change*; the user sorts by *blast radius*.
Those axes disagree, and blast-radius is unmapped.** Per-doc ingestion tuning: the **vector space genuinely
survives** (same model ⇒ same space; reranker scores pairs independently) — but **BM25's `avgdl` is
corpus-global**, so mixed chunk sizes **systematically penalize the longer-chunked doc on the sparse arm**;
health thresholds are absolute; **`TOP_K` counts parents, not tokens** (a per-doc `PARENT_CHUNK_SIZE=8000`
quadruples context+cost, **no guard**); and the splitters are **import-time singletons**, so runtime config
mutation does **nothing** on the hot path. **Contained, not free.** Captured as its own row.

**User challenge 2 — "chat modes = swapping some system prompts." MEASURED: good idea, one hard boundary.**
**`ANSWER_PROMPT` is not a prompt — it's the wire format `synthesis.py` parses** (`_CITATION_RE` is commented as
matching markers *"produced by ANSWER_PROMPT"*). **Proof the coupling already broke:** a 2026-07-14 fix records
the model emitting `[Source 2]`/`[2, 4]`, the parser dropping them, and **claims that DID cite reading as
uncited**. A user edit is that failure with the safety off: every claim → `MARKER_UNSUPPORTED` → **false
`unsupported` rows persisted to the adjudication log** → the failure-tag gates count them → **the
self-improvement loop learns from corrupted data**, silently. **Carve: the citing block is NOT user-editable;
the persona/task framing above it is.** Also: `CitationAudit.clean` is `True` for an answer with **zero**
citations (use `n_uncited_sentences`), and `chat_controller.py:516` caches `_answer_template_hash` **at
construction** — swappable prompts would make **every provenance record lie** unless it moves per-turn.

**Opens:** the graph's **palette gap** — only **3 non-semantic hues** for exactly **3 communities** is luck, not
headroom (`--promote-all` → ~86 nodes; KI-15 documents a real **357**-concept corpus) → **design for 300–500**.
`skeleton.json` is **gitignored + stale** (predates G3 → no `doc_years`; a rebuild changes `graph_version` —
never hard-code it) and a fresh clone has **none** (the empty state is the normal first-run path).
**`data/graph/graph.json` is a stale EMPTY decoy** (retired `concept_graph.py` residue) — reading it renders an
empty graph that looks like a layout bug. **No `load_skeleton()` and zero graph API routes** — the read model is
all to build; node ids are **UUIDs with labels only on nodes**, the exact id-space mismatch that caused KI-15.
Node B on the RTX box (PR-G4) unblocks **every** epistemic UI at $0.

---
## 2026-07-17 — Captured the UI phase's remaining features (7 rows); DIAGNOSED epistemics as blocked on Node B never having been RUN (docs-only)

**What:** the user named the rest of this UI phase — concept graph (Obsidian-like) + epistemics, ingestion
workflow, settings screen, user-tunable RAG pipeline, chunk screen, citation/source highlighting, figures +
tables. Measured the **data reality** behind each against the live corpus, then captured **6 new
`docs/ui-checklist.md` §3 rows** + **rewrote the epistemics row** (its guidance was wrong). **Docs-only; no
code; nothing run against the corpus except free read-only dry-runs.**

**The headline: three of these are not UI work — they're data problems, and they block differently.**

**1. Epistemics — the checklist's own fix was WRONG, and I nearly repeated it.** The row said *"the enrichment
run hasn't been applied here. First step is running the epistemics build ($0/local) and confirming the sidecar
populates."* I offered the user exactly that; they said **investigate first**. They were right — **running it
writes 0 rows.** Free read-only dry-run (`python -m scripts.compute_epistemics`): `Concept nodes weighted: 26`
· **`Contested nodes: 0`** · **`Superseded-trend nodes: 0`** · `Chunks with a claim: **1835**` · **`Chunks
marked: 0`**. **The pipeline is healthy — KI-15's fix works** (1835 chunks project fine); the **input signal**
is absent. `node_weights_for_epistemics` aggregates edges' `stance_by_doc` and documents that *"a stance-less
node → unique/stable"*; **all 70 `concept_edges` rows carry `stance_json = None`** (verified) → no node is ever
`contested` → no chunk is ever marked. Stance comes from **Node B**, the LLM relation/stance pass.
**⚠ SELF-CORRECTION (same session, caught before this entry was ever committed): I first wrote here that Node B
is "explicitly deferred and UNBUILT (no stance/relation runner exists in `scripts/`)" and that epistemics is
"blocked on an unbuilt, PAID LLM feature". BOTH CLAIMS ARE FALSE.** Node B is **code-complete and committed** —
`src/doc_assistant/concept_skeleton_enrich.py` (pure core; idempotent, re-deriving each edge's annotation from
scratch; **never creates a node or edge** — Node A owns those) — and it is **runnable today** via **`python -m
scripts.build_concept_skeleton --enrich`** (`scripts/build_concept_skeleton.py:150` `if args.enrich: return
_run_node_b(...)`, report formatter at `:59`). **It has simply never been run.** **And it is $0, not paid:**
`CONCEPT_SKELETON_LLM_PROVIDER` defaults to **local Ollama** (`llama3.1:8b`, `config.py:445-446`) — *not*
`LLM_PROVIDER` — with `llm.assert_provider_intent` as the KI-4 credit-leak guard; one call **per document**.
**The only real blocker: Ollama isn't on this dev box** (verified on the separate RTX machine) → **running Node
B there is the single prerequisite for every epistemic UI.** *How the error happened: I grepped `scripts/` for a
**file** named like a stance runner and concluded the feature didn't exist. The runner is a **flag** on the
existing Node-A script.* **Lesson: absence of a filename is not absence of a feature — grep the call graph, not
the directory listing.** (Also latent: `superseded_trend` can't fire even *with* stance unless the skeleton
carries publication years — the on-disk artifact predates G3 (2026-07-08), so its `meta` has no `doc_years`; a
rebuild adds them (66/76 docs have a year) **and changes `graph_version`** — never hard-code
`055312c8c15a7e69`.) **Lesson that survives the correction: a builder existing and being free does not mean
running it produces anything. Dry-run first.**

**2. Figures — blocked, and expensively.** `figures` = **0 rows**, and **0 chunks carry `chunk_type` in either
Chroma index** (`data/chroma` 11 965; `data/chroma_pc` **30 882** = the live one). The `chunk_type='figure'`
ingest path (`ingest/__init__.py:220-271`) has **never produced a chunk here**. Unblocking needs
`describe_figures`, *by its own docstring* **"the project's only paid, API-only enrichment"** (VLM, gated by
`MAX_VLM_CALLS_PER_DOC`) → a deliberate cost decision under the repo's discipline, not a build to just run.

**3. Tables — needs diagnosis, not a UI plan.** Extraction code **exists** (`ingest/tables.py`,
`scripts/extract_tables*.py`, `eval_marker_tables.py`) — I initially and wrongly said it didn't — but **no
`tables` table exists and no table chunk is indexed** (`document_parts` is 0 too). Where were they meant to
land, and why didn't they? Answer that before planning any "surface the tables" UI.

**Contrast — what IS ready.** **The concept graph is the phase's biggest unbuilt feature and nothing blocks
it:** `concepts` **26** · `concept_aliases` 17 · `concept_edges` **70** · `concept_presence` **222** ·
`doc_similarities` **760** · `skeleton.json` present. **ADR-015 explicitly reserved this track**, and PR-1/PR-2
just made `Concept` a first-class UI citizen — the read model, write path and vocabulary all exist, and Node A
is zero-LLM. **⚠ Second self-correction: I first cited "citations 3918" as citation-highlighting's data — WRONG table.**
Those 3918 are **bibliography references** (paper→paper) and **0 are resolved to an in-corpus document**
(`target_document_id IS NOT NULL` → 0, verified) — that's the *citation-graph* feature. **Answer**-citation
highlighting rests on `answer_claims` **168** / `answer_records` 26 + per-turn `result.sources`, and inherits
`synthesis.py`'s `[n]` parser (`_CITATION_RE`, `:23-25`) — whose `:30-38` records a 2026-07-14 fix where the
model emitted `[Source 2]`/`[2, 4]` and **claims that DID cite read as uncited**. **Ingestion** has its model + most of its read surface already (`source_files` 77,
`ingestion_events` 76, derived status, the `excluded` toggle).

**4. "User-tunable RAG pipeline" is a governance request wearing a UI costume.** ADR-010 **considered and
rejected persistent editable settings "on governance grounds"** — non-persistence **is** the wall that keeps a
restart returning to the eval-gated baseline, and CONTEXT's non-negotiable says a locked setting changes **only
via an eval-harness experiment**: *"a sandbox override is fine; changing a default is not a UI PR."* Captured
with the collision stated, **grill + ADR before any build** (user's call), so nobody ships it as "just a
settings screen" and quietly invalidates every baseline.

**Rejected:** (a) *running `compute_epistemics --apply` because it's free* — it writes 0 rows; free ≠ useful.
(b) *treating "epistemics" and "figures" as one blocked row* — they block on **different** things (an unbuilt
LLM pass vs a paid VLM run) and must be decided separately. (c) *planning the figures/tables UI now* — the data
question is unanswered for both halves. (d) *folding "improve the settings screen" and "user-tunable RAG" into
one row* — one is layout, the other reopens an ADR; bundling them is how the governance change would sneak in.
(e) *treating the concept graph as part of tag families* — ADR-015 deliberately separated them.

**Opens:** Node B (the LLM stance pass) is the single prerequisite for **all** epistemic UI — and it's paid, so
it lands under "prove on Ollama first" (KI-4). The two Chroma dirs (11 965 vs 30 882) should be reconciled or
documented before anything counts chunks. `folders`/`tags`/`document_tags`/`gaps`/`document_parts` are all **0
rows** — the Library redesign's Phase B (folders) has no data either. The concept graph must **not** imply
epistemic stance it doesn't have: its 70 edges are association-only.

---
## 2026-07-17 — Planned the 3 remaining UI features into 8 PRs; corrected 3 false checklist claims (docs-only)

**What:** explored + planned the other three UI features the user picked — **evidence-only chat mode**,
**missing-source + library-only delete**, **extended metadata + web autocomplete** — against the live code and
the real 76-doc corpus. Carved into **8 PRs**, sequenced **after** tag-families PR-2.5/2.6/2.7. **Docs-only; no
code changed; coding deferred to a later session by the user.** Full per-PR detail (files, seams, decisions,
tests, verification, risks) is in a **local, uncommitted** plan file; the load-bearing measurements and every
corrected claim are duplicated into `docs/ui-checklist.md` §3 + this entry so nothing is lost with it.

**Why it needed measuring: three claims in our own docs were FALSE.**
1. **"The web-fetch is the first outbound network call beyond the LLM APIs"** (ui-checklist) — **false.**
   `sources_manifest.py:278-285` `_http_get` already does `urllib.request.urlopen` (scheme guard + UA +
   timeout + `# nosec B310` + checksum verify); `scripts/download_corpus.py:72-73` too. The defensible claim
   is *"first outbound call from the API **serving path**"* (`apps/api/` has zero such imports). The
   correction **helps**: `_http_get` is a ready-made precedent to copy, not a novelty to argue against.
2. **"`article_type` is already parsed and thrown away"** (my own claim, this session) — **false, 0/76
   measured.** `_is_skippable_heading` (`metadata_extractor.py:125-138`) returns a **bool skip predicate**,
   and `_SKIP_HEADINGS` mixes document types ("research article") with **section names** ("abstract",
   "methods", "main") — it never fires as a taxonomy on this corpus. Reasoning from code *shape* was wrong;
   the measurement is why we caught it.
3. **"Evidence-only: ~90% exists, in Settings"** — broadly true, details wrong. `synthesis_mode` is already a
   **request-scoped per-turn override** (not a global setting), and it renders as a **concatenated markdown
   string** with an emoji + em-dash prefix (`chat_controller.py:1055-1058`) that the de-tell pass missed
   because it lives in `src/`. `result.mode` is **live on the wire and dead on the frontend** (zero
   consumers) — a free field.

**Measurements that drove the carve (live, this box):** `doi` **25/76** and 100% invisible in `apps/` (2 of the
25 are `type: component` — a figure DOI, wrong for the paper) · `notes` **0/76 with no writer** → dropped from
the DOI PR · local-text yield for the new fields is hopeless (journal 5/76 · url 1/76 · article_type **0/76**)
while **Crossref covers 21/25/25** → Crossref, not extraction · **0/76 sources missing** → the badge has
nothing to show live (synthesize a case) · existence check ×76 = **1.0 ms** → derive-live is sound ·
`USE_MULTI_QUERY` default **false** → the *rewrite* is the real $0 leak, not multi-query.

**Two findings that changed a decision, not just a detail.** (a) **Evidence-only is a better idea than it
looked:** human mode writes `"(human mode: evidence only)"` into history (`:1053`), so in a pure-human chat the
rewrite LLM reads a **zero-information stub** — it is *structurally incapable* of resolving a pronoun, i.e. it
costs money to accomplish nothing. And `_human_result` **already hardcodes `UsageView(0, 0, …)`** (`:1063`)
while the rewrite really spends tokens — **the turn already lies; forcing makes the lie true.** The honest
residual is *mixed*-mode chats. (b) **`is_archived` is a trap:** ~12 read paths filter it and nothing sets it,
but **the retrieval pipeline has no `is_archived` filter and Chroma chunks carry only `doc_hash`** — an
"archived" doc would vanish from the Library while its chunks kept being retrieved and cited. Archiving
without removing chunks is incoherent; removing them kills the reversibility that was the only reason to
archive.

**Transport spike (real outbound calls to the public Crossref API, disclosed to the user).** stdlib `urllib`
reaches `api.crossref.org` from this proxy box **with and without** `truststore.inject_into_ssl()` (~0.7–0.8 s;
25/25 DOIs, 0 failures) → **transport decided: stdlib urllib behind a named seam, not httpx.** Recorded as a
**KI-10 addendum** — the failure is **httpx-specific** (it pins certifi and ignores both the process-global
patch and `SSL_CERT_FILE`), which is why the KI-10 fix had to be branch B. **⚠ Scoped honestly: dev
interpreter (`sys.frozen is False`), one box/day/proxy state — it does NOT prove the frozen build**, which is
KI-10's actual subject.

**Rejected:** (a) *`is_archived` for library-only delete* — the retrieval-filter evidence above. (b) *reusing
`scan_sources` for the badge* — it **WRITES** (upserts rows, refreshes `last_seen`); a read-only list endpoint
must not mutate the registry. (c) *a 4th context-menu item for library-only delete* — `openMenu` hardcodes
geometry for 3 items (`LibraryGrid.svelte:68-69`) and two adjacent red Delete items on the most destructive
path is a footgun; a checkbox in the existing dialog (default = today's behaviour) is the gate already. (d)
*httpx for Crossref* — re-solves KI-10 for a second client; `os_trust_http_client()` returns `None` when not
frozen and is anthropic-typed (a pattern, not a component). (e) *the `cpc: allow-live-api` pragma* — the gate
tripping on `urllib.request` in a test file is **correct pressure**: the call must sit behind a named
monkeypatchable seam (precedent: `test_sync_sources.py`). (f) *a counter-based zero-LLM test* —
`expand_query` (`pipeline.py:302`) takes **no counter**, so it would silently miss the multi-query leak; spy
the pipeline instead. (g) *shipping the metadata migration before the ADR* — Crossref coverage is what decides
which columns earn one; PR-B first would be a migration written against a guess. (h) *`notes` in
`DocumentMeta`* — it has no auto default, so `_dedup_override` would mark it "customized" and corrupt the
`.editmark` dot's meaning.

**Opens:** `expand_query` takes no counter (real bug, its own PR) · **no network guard in the test suite** (the
`api` marker is declared but unused; **no `conftest.py` exists anywhere**) → a missed monkeypatch would
silently hit the network in CI · the `year_override` blanking trap (`library.py:244` — no way to override to
empty) that every new nullable field inherits · the frozen-build urllib check above · human-mode resume stores
a stub answer, not the evidence.

---
## 2026-07-17 — Manage view at scale scoped (PR-2.7) + 2 non-UI rows, from live user feedback (docs-only)

**What:** user reviewed the shipped Manage-keywords view + filter overlay and raised three things — the
"Manage keywords…" trigger sits under the scrollbar; nothing scales to ~100 families; and "102ff, 16p11 wtf
are that?". Scoped **PR-2.7 — Manage view at scale** (frontend-only) into `feature-tag-families.md`, plus two
**non-UI** backlog rows the third point exposed. **Docs-only; no code changed.**

**Why the "wtf" answer changed the design — traced every odd keyword to its source doc rather than assuming:**
they are **mostly real specialist vocabulary, not junk.** `16p11` is **16p11.2** (autism CNV) truncated at the
dot; `c57bl` is **C57BL/6** (mouse strain, **7 docs**) truncated at the slash; `va1v`/`dl5`/`osns`/`upns`/
`mgns` are Drosophila antennal-lobe glomeruli + neuron classes, all from **one** olfactory paper;
`avpv`/`pvpo`/`mpoa` are hypothalamic nuclei from one mating paper; `rabv` = rabies-virus tracing. The truly
broken ones are a small **clustered** minority: `mathrm` (LaTeX leak), `professium`, `outflux` — **all three
from the same 1952 scanned Hodgkin-Huxley paper** — plus `102ff` ("p. 102ff"), `fne-tune` (OCR of
"fine-tune"), `neurosc`. So the verb is **demote, not delete**: deleting real vocabulary isn't
reversible-by-search.

**The measurement that made it principled (live corpus, not assumed):** 60 keywords surface; **30 — exactly
50% — sit on a single document**, and every ugly string the user spotted is in that tail. **A facet exists to
partition a set; a 1-doc facet doesn't partition** (selecting it yields one doc, which search does better), so
rare keywords have near-zero *filtering* value **whether they're junk or gold**. That threshold sweeps up the
noise **without having to classify it** — and the overlay's existing search box is already the escape hatch.
Also measured: `FAMILIES (26)` is misleading — only ~6 are real families, the rest 0-member concepts inherited
from concept-graph seeding, several with 0 docs (`BERT`).

**Convergence worth not losing:** autocompleting the "New family" canonical against existing families is
simultaneously the user's navigation ask **and** the fix for defect **D3**. Carve guards against building it
twice: **PR-2.5 owns the `library.py` boundary invariant; PR-2.7 owns the control that stops the user reaching
it.**

**Rejected:** (a) *deleting the rare tail* — mostly real vocabulary; delete isn't reversible-by-search
(demotion is). (b) *modelling suppression as a "hidden" family* (a `Concept` whose aliases are the junk) —
abuses ADR-015, which defines a family as **synonym collapse**; overloading those rows corrupts every
consumer that reads them, including the concept-graph track that owns them later. (c) *folding PR-2.7's
autocomplete into PR-2.5* — the invariant belongs at the library boundary regardless; a view control is a
convenience on top, not a substitute (a second client bypasses it). (d) *a code-level stoplist for
suppression* (`VENUE_STOPWORDS` precedent) — wrong shape for **user-editable** data; the user-override
precedent is `DocumentMeta`/ADR-013 (a sidecar), and note the Enrichment-Layer Pattern does **not** govern it
(it's a user override, not derived data). (e) *treating the doc-count threshold as a locked setting* — it
governs **presentation**, not retrieval, so no eval-harness gate.

**Opens:** two **non-UI** rows now on the checklist. (1) **Extractor truncation** — `_TOKEN_RE`
(`keywords.py:36`) allows `-`/`+` as internal joiners but not `.`/`/`; fixing it touches the **ingest path**
(re-run extraction over 76 docs, re-check the `VENUE_STOPWORDS`/repeated-token interaction) and **must
preserve the ASCII-lowercase `Keyword.name` invariant** that the tag-families review leaned on (it's why
SQLite `lower()` == Python `.casefold()` there). (2) **Keyword suppression** — needs a decision on where it
lives (sidecar table + migration + UI write surface) → wants an ADR or a spec section; may pair with the
extractor fix (fix what you can, suppress what you can't).

---
## 2026-07-17 — Post-commit review of tag families PR-1/PR-2; PR-2.5 + PR-2.6 scoped (docs-only)

**What:** reviewed the two shipped tag-families commits (`0c3b0d4` PR-1, `0af43db` PR-2) — an agent review
of the diff plus a live drive of the running app on the real 76-doc corpus ($0/offline). **No code changed
this session; docs only.** Flipped the post-commit paperwork the commits left behind
(`feature-tag-families.md` + `ui-checklist.md` still said "staged, not committed"; now ✅ SHIPPED with SHAs),
and scoped two defect-driven follow-ups into the spec: **PR-2.5** (hardening the write paths, D1–D5) and
**PR-2.6** (family-aware grid tiles, D6). Full repros per defect live in the spec's carve table.

**Verified working live (so the review is not theoretical):** the overlay collapses families into atomic
entries (`Large language model` 3 forms, `Connectome` 3 forms, `Embeddings` 4 forms); selecting the LLM
family returns **14 documents**, all genuinely LLM papers (union correct) and every other chip recounts for
AND (`chatgpt` 7, `Embeddings` 7, non-overlapping → 0, greyed); Detect reproduced the DEVLOG's claim exactly
(`pvpo`≈`avpv pvpo` @ 0.77, tier MEANING) — **left unaccepted, real corpus unchanged**; 0 console errors.
Unplanned bonus confirmed on screen: the curated vocabulary supplies **display names**, so `bm25` renders as
`BM25` and `imagenet` as `ImageNet`.

**Why:** 977 tests passed and every gate was clean, yet **6 real defects** survived — all in the
**under-guarded write paths**. The read path reviewed clean (facet math, union-find determinism + the
`(a,b)` key ordering, thin-shell discipline, and the no-families default path being genuinely
byte-identical). Two defects **escape the feature**: D1 (rename onto an existing canonical → duplicate
`Concept` labels → create route 500s forever *and* `promote_keyword` throws `MultipleResultsFound`,
breaking `scripts/seed_concepts.py`) and D2 (rename silently drops the family's own canonical keyword,
re-creating the duplicate chip the feature exists to remove). Both sit on the **natural** post-PR-2 flow:
`_canonical_and_members` always proposes an existing keyword as canonical (`llm`), so Detect → Accept →
**Rename** is the obvious next click. D1 is precisely the boundary risk ADR-015 named in its Consequences,
now realized — the tag-families UI can corrupt vocabulary a different feature reads.

**Root cause of the test blind spot (worth keeping):** `test_rename_family_canonical` asserts only
`renamed.canonical` — never `aliases` or `doc_count` — and it even uses the exact
`create_keyword_family("llm", [])`→rename shape that triggers D2. The stemmer tests (D4) pick `boxes`/
`taxonomies`, the two inputs the sibilant/`-ies` rules happen to get right; verified against the real
`_stem`, `database`/`databases`, `size`/`sizes`, `cache`/`caches`, `response`/`responses` and
`analysis`/`analyses` all **MISS**, and `detect_family_proposals(["database","databases"])` returns `[]`.
The frontend grouping layer (`familyCanonicalMap`/`familyUnitsOf`) has **no tests at all**. So PR-2.5's DoD
puts the five repros in as regression tests *first* — they all pass today, which is the point.

**Rejected:** (a) *folding D6 into PR-2.5* — carved into its own PR-2.6 instead: D6 and the family-aware
tile display are the same root cause in the same file (`LibraryGrid` never learned about families), so
bundling them into a backend-correctness PR would both violate "never bundle" and touch that file twice.
(b) *fixing D3's free-text canonical in the Svelte view* — the "a keyword belongs to at most one family"
invariant (ADR-015) belongs at the `library.py` boundary; the view is a thin shell and a second client
would bypass a view-level guard. Same reasoning routes D1's collision check to
`library.rename_keyword_family` (→ 409) rather than to `commitRename`. (c) *logging the six to
KNOWN_ISSUES and building new features first* (user call — hardening was chosen instead; leaving Rename as
a data-corruption trap on the happy path was not acceptable). (d) *adding a DB unique constraint on
`Concept.label` for D1* — deferred to the PR: it's the right long-term shape but it's a migration touching
a table two other features read, so it needs its own decision rather than riding a UI fix.

**Opens:** PR-2.5 must **pick one** D2 fix (seed the canonical as a real alias on create, aligning
`create_keyword_family` with `promote_keyword`, vs. make rename carry the old label into the alias set) and
**note the migration for the 26 pre-existing curated concepts on this box** — they predate the feature and
carry no seeded alias. PR-3 (LLM confirm) stays parked behind Ollama proof (KI-4). The `Concept.label`
uniqueness question (rejected (d)) is now on the record for whoever touches the schema next.

---
## 2026-07-17 — Tag families PR-2: detection (feature-tag-families.md, ADR-015)

**What:** built PR-2 of the tag-families spec — a deterministic, zero-LLM detection pass that
proposes family groupings for un-familied keywords, for the user to review and accept (nothing
auto-applies). **Pure core:** new `src/doc_assistant/keyword_families.py`. Tier 1 (`_stem`,
morphological — a conservative plural/suffix stripper: `llms`→`llm`, `connectomes`→`connectome`),
Tier 2 (`_tier2_embedding`, bge cosine clustering via union-find so a chain of near-duplicates
proposes one group, not overlapping pairs — catches meaning-close/spelling-different pairs a stem
can't, e.g. `connectome`≈`connectomics`) with a hand-rolled Levenshtein `_edit_similarity` blended
into Tier 2's confidence as a *supporting* signal, never a gate. `embed_fn` is injected (no model
load inside this module) — `detect_family_proposals(names, *, embed_fn=None, embedding_threshold)`.
**Impure boundary:** `library.py` gains `detect_family_candidates(embed_fn=None,
embedding_threshold=None)` — loads every `Keyword` name, subtracts anything already a family's
canonical/alias (case-insensitive), and hands the rest to the pure core. **API:** `POST
/api/library/keyword-families/detect` (`apps/api/main.py`/`models.py`) reuses the controller's
already-loaded embedder (`controller.rag.embeddings.embed_documents` — no second model load), wraps
it to match the pure core's `embed_fn` contract, returns `KeywordFamilyProposalPayload[]`.
**CLI:** `scripts/detect_keyword_families.py` — report-only (no `--apply`, matching the DoD's
"nothing auto-applies"; `--no-embeddings` for an instant Tier-1-only pass, `--threshold`/`--model`
for Tier 2), loads a fresh embedder via the existing `concept_semantics.embed_texts` (acceptable on
a host CLI, unlike the API route). **Frontend:** `types.ts`/`api.ts` (`KeywordFamilyProposal`,
`detectKeywordFamilies`); `LibraryManageKeywords.svelte` gains a "Detect proposals" section (Detect
button → tier badge + canonical + members + confidence % + Accept/Dismiss per row; Accept routes
through PR-1's existing `onCreate`, Dismiss is a pure client-side filter — nothing was ever
written); `App.svelte` owns `detectProposals`/`detecting`/`detectError` state, cleared when the
Manage view closes (proposals are cheap to regenerate; staleness after other edits is harmless
since accepting still goes through idempotent CRUD).

**Why:** PR-2 was next per the design-locked spec's foundation-first carve (T8) — PR-1 shipped the
manual mechanism; PR-2 removes the toil of *finding* what to group by hand.

**Verified:** 14 new pure-core unit tests (`tests/unit/test_keyword_families.py` — stem edge cases,
Tier 1/2 grouping, the transitive-chain union-find case, Tier-1-consumed names excluded from Tier 2,
toy-vector Tier 2 with no real bge load) + 2 new integration tests (`/detect` route, 200 with both
tiers + familied-exclusion, empty-corpus 200/[]) — full suite **977 passed / 1 skipped** (pre-
existing, unrelated). ruff/ruff format/`mypy --strict src`/bandit clean; `svelte-check` **0/0**
(131 files). **Live on the real 76-doc corpus, $0/offline:** `--no-embeddings` found 0 proposals
(every simple-plural pair on this corpus is already familied or non-existent — a legitimate
negative, not a bug); the full run (real bge, offline-cached) found one genuine Tier-2 proposal —
`pvpo` ≈ `avpv pvpo` (confidence 0.77) — both via the CLI and via the app's Detect button (identical
result, confirming the API route's embed_fn wiring matches the CLI's). Accepted the proposal live
(created the family, refreshed the list, proposal cleared) then deleted it to leave the real corpus
unchanged (verification, not a curation decision for the user's data). Dark theme, mobile 375px,
0 console errors, 0 API server errors.

**Rejected:** returning Tier 2 as flat pairs (mirroring `concept_semantics.ConceptPair`) instead of
transitively-grouped proposals — the spec calls for "grouped candidates," and a chain of 3+
near-duplicates as one proposal is a better review unit than N overlapping pairs; adding a
fuzzy-match dependency (`rapidfuzz`/`python-Levenshtein`) for the edit-distance signal — ~15 lines
hand-rolled matches this repo's existing bias (cf. `keywords.py`'s `weirdness`/`c_value_scores`)
against a new dependency for one small deterministic computation; gating Tier 2 on edit-distance —
would defeat the tier's purpose (spelling-different, meaning-close pairs are exactly its job); a
`--apply` flag on the CLI script — the DoD is explicit that nothing auto-applies, so there is
nothing for `--apply` to do that the app's Accept button doesn't already own.

**Opens:** PR-3 (an optional, gated LLM confirm/merge pass — parked, prove on Ollama first per
KI-4) is the last carved piece; not scheduled. The Detect flow has no rate-limiting/debounce (a
user could click Detect repeatedly; each call re-embeds the current candidate pool — cheap at this
corpus's ~40-keyword scale, would want caching if the un-familied pool grows large). Proposals
aren't persisted across a Manage-view close/reopen (by design — see above) — if that reads as
friction once PR-2 sees real use, revisit.

**Staged, nothing committed (cpc §13).**

---
## 2026-07-17 — Tag families PR-1: families end-to-end, manual (feature-tag-families.md, ADR-015)

**What:** built PR-1 of the design-locked tag-families spec — a family = a curated `Concept` whose
`ConceptAlias` rows carry member `Keyword` names (ADR-015), reusing the existing vocabulary tables
(no schema change). **Backend:** `concept_skeleton.py` gains the missing mutation primitives
(`remove_alias`, `delete_concept`, `rename_concept` — matching `add_concept`'s style); `library.py`
gains `KeywordFamily` + `list_keyword_families`/`get_keyword_family`/`create_keyword_family`/
`rename_keyword_family`/`add_family_member`/`remove_family_member`/`delete_keyword_family` (thin
shells; `doc_count` = a case-insensitive union query over `document_keywords`/`Keyword`; a keyword
belongs to at most one family — `add_family_member` moves it off any other family's alias set).
Six new FastAPI routes under `/api/library/keyword-families` (`apps/api/main.py` + `models.py`),
mirroring the safe-delete/metadata-edit route conventions (404 on unknown family, 400 on a blank
canonical). **Frontend:** `types.ts`/`api.ts` wire contract; `library.ts` gains `familyCanonicalMap`/
`familyByCanonical`/`familyUnitsOf` (a pure pre-facet grouping step) plus an optional `keywordsOf`
accessor on `facetFilter`/`keywordFacets` (default = raw keywords, so the no-families path is
byte-identical to pre-PR-1 behavior); new `LibraryManageKeywords.svelte` (create/rename/add-remove-
member/delete, opened via a new "Manage keywords…" link in `LibraryKeywordFilter.svelte`, which also
now shows a family facet's "N forms" subtitle + hover listing its members); `App.svelte` loads
families alongside documents and threads `keywordsOf` through the facet/filter pipeline.

**Why:** PR-1 was next per the design-locked spec (ADR-015, grilled 2026-07-16) — the overlay built
last session ships raw per-keyword facets, so near-duplicates (`llm`/`llms`, `connectome`/
`connectomics`) still count as separate filters. Foundation-first carve (T8): manual CRUD before
detection (PR-2) or an LLM pass (PR-3, parked).

**Verified:** 17 new integration tests (`test_keyword_families.py` — CRUD, the move-on-reassign
invariant, union `doc_count`, route 200/404/400) + full suite **961 passed / 1 skipped** (pre-existing,
unrelated); ruff/ruff format/`mypy --strict src`/bandit clean; `svelte-check` **0/0** (131 files).
**Live on the real 76-doc corpus, $0/offline:** the box already carries 26 curated `Concept` rows from
earlier concept-graph work (e.g. `Large language model` ← `llm`/`llms`) — confirmed this is intended
reuse of the vocabulary, not pollution (ADR-015's "take advantage of," not the graph UI). The overlay
correctly collapsed `llm`/`llms` into one `Large language model` facet ("3 forms", hover lists the
aliases, count = union = 14 docs); toggling it filtered the grid to the 14-doc union and the strip
chip showed the canonical name. Full CRUD round-trip in Manage keywords (create a test family from an
un-familied keyword → add a second member → rename → remove a member → delete) verified live and
cleaned up (DB back to 26 concepts, no test residue). Dark theme flips via CSS vars; mobile 375px no
horizontal overflow; 0 console errors; 0 API server errors.

**Rejected:** listing only `source="keyword"`-promoted concepts as families (would hide manually
curated glossary entries like `RAG`/`BM25` that are equally valid single-keyword families; ADR-015
treats the whole vocabulary as reusable); mutating `facetFilter`/`keywordFacets` to hard-require a
`KeywordFamily[]` param (an optional `keywordsOf` accessor keeps the default path untouched, matching
the DoD's byte-identical requirement); collapsing `d.keywords` in place on `LibraryDocument` objects
(would also silently change what the grid tiles' own keyword chips display, out of PR-1's scope — the
collapse is confined to the facet/filter computation only).

**Opens:** PR-2 (detection: tiered morphological + `bge` embedding clustering, no auto-apply) and PR-3
(LLM confirm pass, parked — prove on Ollama first, KI-4) are next per the spec's carve. The Manage
view's per-family "add a keyword" `<select>` lists every un-familied keyword with no search/filter —
fine at the current ~40-keyword scale, would want a search box if the vocabulary grows a lot before
PR-2's detection reduces the un-familied pool. No UI surfaces a family's `source` field (`"manual"`
here) — not needed yet, but PR-2's detected proposals will likely want to distinguish themselves before
acceptance.

**Staged, nothing committed (cpc §13).**

---
## 2026-07-16 — UI: keyword filtering as a two-pane overlay (folds the inline-bar cut below)

**What:** grilled the just-built inline facet bar (entry below) — it doesn't scale past a few dozen
keywords — and redesigned the presentation into an on-demand **two-pane overlay**
(`LibraryKeywordFilter.svelte`, reusing the `LibraryMetaEditor` modal shell): left = a searchable keyword
list (Zotero mechanics — AND, grey-out unavailable, most-used-on-top); right = a live preview of the
matching documents (`title · author · year` + "N documents"). **Live commit, no Apply.** The inline
`LibraryFacetBar` is replaced by a slim `LibraryFilterStrip.svelte` — a "Filter by keyword" trigger + the
*selected* keywords as removable chips + a result count + Clear. `App.svelte` gains `keywordFilterOpen`
state; the overlay + strip share `facetList` / `visibleDocs` / the existing `toggle`/`clear` handlers.
**The pure `facetFilter`/`keywordFacets` logic is unchanged** — a presentation swap. Design lock + grill
ledger: `docs/specs/feature-keyword-filter-overlay.md`.
**Why:** user feedback in a `grill-me` session — an always-on chip bar is fine at 60 keywords, not at
600; an overlay (searchable, with a doc preview) is the standard escape hatch (command palette, Zotero's
tag selector, GitHub's label filter). 9 branches resolved, 3 parked.
**Verified ($0/offline, preview harness, real 76-doc corpus):** `svelte-check` **0/0** (130 files); strip
trigger → overlay opens (60 keyword rows sorted by count + "76 documents" preview); search "conn" → list
filters to `connectome`/`connectomics`/`connectomes`; toggle `embeddings` → preview "22 documents" + 26
rows greyed + grid behind 76→22 + strip chip + "Keywords · 1" + "22 docs"; **Esc closes and the selection
persists**; Clear → 76; dark theme adapts via CSS vars (scrim/surface/border/text); mobile 375px → panes
**stack** (single column), no horizontal overflow; **0 console errors** (verified on a fresh tab — the
mid-session HMR "Failed to reload LibraryFacetBar" errors were stale buffer ghosts from the delete).
**Rejected:** Zotero's *container* (a permanent docked tag panel would crowd the rail's Collections); a
fully overlay-only strip-less view (hidden filter state); draft-selection + Apply (the live preview makes
it redundant); user-pinned favorites + a general "Filters" hub + Cmd-K (all parked — see the spec).
**Opens:** user-pinned favorites ride tag-families (promote-to-`Concept`); the general Filters-hub reopens
when extended-metadata lands article-type/year/journal as filters; no JS unit tests (no vitest in the repo).

## 2026-07-16 — UI: faceted keyword filtering in the Library (Phase 8, frontend-only, staged)

**What:** the Library grid gains a **multi-select keyword facet bar** (new `LibraryFacetBar.svelte`).
Clicking a keyword chip toggles it into an **AND** filter; a chip greys out + disables when adding it
would empty the grid, and each available chip shows its live co-occurrence count. Retired the
single-select `{kind:'keyword'}` `LibraryCollection` (user pick — "pure facet"): keywords are no
longer a nav collection but an orthogonal facet on the collection → search → **facet (AND)** → sort
pipeline. Pure helpers `facetFilter` + `keywordFacets` (deterministic, ties break alphabetically) +
`KeywordFacet` in `lib/library.ts`; removed the now-dead `keywordGroups` and the Sidebar "Keywords"
nav group (+ its dead CSS). `LibraryGrid` `activeKeyword: string|null` → `activeKeywords: string[]`
(selected facets surface first + highlight on every tile). App: non-persistent `libraryKeywords` +
`toggle/clearKeywordFacets`; the filtered empty-state offers "Clear keyword filters". **Backend
untouched** — `LibraryDocument.keywords` already ships client-side; thin-shell preserved.
**Why:** user feedback — "click a keyword to filter; multi-select greys out the unavailable ones."
The lavender chips were display-only; this makes them a working facet. Live data also proves the
next backlog row's premise: `llms`/`llm` and `connectome`/`connectomics` show as separate chips
(tag families).
**Verified ($0/offline, preview harness, real 76-doc corpus):** `svelte-check` **0/0** (129 files);
live — 60 keywords (24 shown + "Show all (60)"), select `embeddings` → **76→22 tiles** + `pretrained`
19→14 / `llms` 13→7 recount + **26 chips greyed** (disabled, opacity 0.5, `cursor:not-allowed`,
`aria-pressed=false`); `llms`+`pretrained` → **5-tile AND intersection** with both surfaced +
highlighted on the tile; **Clear** restores 76; dark theme adapts via CSS vars (selected = ink on the
lightened `--accent`); mobile 375px no horizontal overflow; **0 console errors**.
**Rejected:** OR semantics (nothing is ever "unavailable" under OR → grey-out meaningless); an
orthogonal facet keeping the keyword-collection (two redundant keyword mechanisms); a backend
facet-count endpoint (76 docs + keywords already client-side → pure frontend); making tile chips
click-to-toggle (a chip lives inside the tile's body `<button>` — nested buttons are invalid HTML;
the facet bar is the v1 toggle home, tile-chip toggling deferred to a tile restructure).
**Opens:** `Tag` (user labels) not yet faceted (no producer/data — the natural follow-on); tag
families (`llms`/`llm` normalization) is the next backlog row (reuse `Concept`/`ConceptAlias`, needs
an ADR); tile chips as toggles (needs the nested-button restructure); no JS unit tests for the pure
helpers (no vitest in the repo — verified via svelte-check + the live harness).

## 2026-07-16 — cpc re-vendored 1.2.1 → 1.2.2; KI-16 RESOLVED

**What:** third cpc step today — the KI-16 fix landed upstream (cpc `bda91a5`, released as
**v1.2.2** same day) and this repo re-vendored via `cpc-init` re-run (`tools/conventions/cpc/`
`_VERSION` 1.2.2; the three deliberately-diverged lays — `AGENTS.md`/`GLOSSARY.md`/
`.claude/.gitignore` — pruned again, same as the 1.2.1 step). `docs_check` now skips embedded
checkouts structurally (`.venv`/`node_modules`/`.git` parts + any dir carrying its own `.git`),
so a live background-task worktree under `.claude/worktrees/` no longer produces phantom errors.
KI-16 flipped → RESOLVED (resolution bullet in the entry); CONTEXT wiring line → 1.2.2.
**Verified:** `docs_check --strict` **0/0 unfiltered** on the 1.2.2 vendor (upstream: 144/144
tests + the live repro on this repo's own worktree, 70 → 0, before the tag).
**Why:** closes the loop the same-day review opened — gate noise during background tasks was the
one environmental red herring left.
**Rejected:** waiting for the next natural touch to re-vendor (the fix specifically de-flakes THIS
repo's gate; adopt while the context is warm).
**Opens:** none.

## 2026-07-16 — Docs-staleness fix batch (applies the same-day review's findings)

**What:** applied the fixes from the docs review (entry below), docs-only. **ROADMAP:** flipped
**10 stale status cells** to committed with verified SHAs — S1 `2893544` · S2 `7224f10` · V2
`4fd772c` · V3a `181046c` · V3b `487f2df` · L4-A `9f597df` · **G3 `d7528ab` · G4 `5fc5964` · G6
`cb166d4` · G7 `1e1e7eb`** (the G-rows still said "staged — awaiting review" from 2026-07-08; SHAs
re-derived from `git log`/`-S`, not taken on faith — the review agent had mis-mapped two);
added rows **L5** (metadata enrichment + keyword de-noising, `8f31fe3`) / **L6** (metadata editing,
ADR-013, `e549254`) / **L7** (safe-delete, ADR-014, `95817fc`); fixed the stale Phase-7 bullet
("redesign not yet built" → built+validated, `concept_graph.py` deleted) and the 7d paragraph's
"host apply pending"; repointed all 17 `docs/sprints/SPRINT-*` references to `docs/archive/sprints/`
(every numbered contract is archived). **architecture.md:** `library.py` contract corrected
(read-only → + ADR-013/014 write paths); `ingest` row gains `registry` (S1); SQL-store node gains
`DocumentMeta`/`SourceFile` (+ sidecar note); the false "`concept_graph` … not replacing it yet"
claim replaced with the real state (deleted 2026-07-07, skeleton is the layer); `metadata_enrich` +
`concept_skeleton_enrich` named. **Specs:** 8 shipped specs advanced from design-locked/"NOT built"
to ✅ SHIPPED with SHAs (rag-sandbox, provider-switch, library-redesign Phase A, selective-ingestion,
visual-identity complete, ab-compare, library-browser, conversation-history) — design locks retained
as the design record. **ADR-002:** status corrected proposed → accepted/implemented (M0–M5 shipped
2026-06-25). **decisions.md:** 3 dangling `docs/doc-assistant-roadmap.md` routes → `docs/archive/…`.
**ui-checklist:** S1/S2 + L4-A + enrichment + metadata-edit + safe-delete added to §1, their §3 rows
flipped `[x]` with committed SHAs, visual-identity row → COMPLETE. **KNOWN_ISSUES:** KI-8 dated
correction (KI-7 citation outdated; markers default-ON again since G1 — the "mostly moot" bullet no
longer holds).
**Why:** the 2026-07-16 review found the paperwork lagging the code by up to 8 days; a doc that says
"staged" for committed work actively misleads the next session.
**Rejected:** stubbing the `docs/decisions.md` monolith (ADR-001 step 4) and deleting `HANDOFF.md` —
both are user decisions, left open; backfilling cpc headers onto the 19 specs (specs are
`[headers]`-exempt by config — optional consistency work, not staleness).
**Opens:** decisions.md stub vs hybrid (user call); HANDOFF.md retire/refresh (user call); the
enrichment row's local-LLM leftover pass; live end-to-end delete smoke (user's run).

## 2026-07-16 — cpc re-vendored 1.1.0 → 1.2.1 + documentation review (gates + judgment sweep)

**What:** re-ran `cpc-init` from the cpc checkout at release tag **`v1.2.1`** (via a temp git
worktree; not the unreleased 1.2.2 HEAD — `_VERSION` records releases): `tools/conventions/cpc/`
refreshed (19 modules; new `keypoint.py` = the ADR-020 workflow-boundary runner; the cpc LICENSE now
travels with the drop), and the two new 1.2.0 templates laid —
`docs/features/FEATURE-000-template.md` + `docs/specs/SPEC-000-template.md` (the per-feature
rationale layer + the ADR-019 executor brief). Pruned the three files `cpc-init` lays that this repo
deliberately diverges on: `AGENTS.md` (`init_check` stays unwired — ADR-014 entry-file adoption
consciously deferred, per `.pre-commit-config.cpc.yaml`), `GLOSSARY.md`, `.claude/.gitignore` (root
`.gitignore` already covers `.claude/*`). Gate battery on 1.2.1: `test_api_check` clean;
`sprint_check` green after flipping the **overdue SPRINT-019 → archived** (V3b shipped `487f2df`,
post-commit flip never done); `docs_check --strict` clean on the real docs — its 70 remaining errors
are phantom hits on the live `.claude/worktrees/` background-task worktree, logged as **KI-16**
(upstream one-line-class fix in cpc's `docs_check.py` identified; no `conventions.toml` workaround
exists). Docs corrected this session: `.claude/CONTEXT.md` (cpc version bump; wiring text no longer
claims `init_check` runs at pre-push; the new on-call `keypoint` command documented; the missing
**Provenote** product-identity fact added, ADR-012), `docs/ROADMAP.md` `updated:` bump (rule-12 WARN).
**Why:** adopt cpc 1.2.x; keep ADR-007's canonical wiring text honest; user-requested docs review.
**Rejected:** vendoring HEAD (1.2.2-unreleased); adopting AGENTS.md/GLOSSARY.md wholesale just
because `cpc-init` lays them (deliberate divergences stay deliberate); a `[headers] exempt` glob for
the worktree noise (`Path.match` is right-anchored — cannot left-anchor a recursive glob; KI-16).
**Opens:** the judgment sweep found real staleness beyond this session's fixes, deferred to its own
fix session: ROADMAP rows S1/S2/V2/V3a/V3b/L4-A still "staged, not committed" though committed +
no rows for metadata-edit (`e549254`)/safe-delete (`8f31fe3`); `architecture.md` stale (`library.py`
described "read-only" but now carries ADR-013/014 write paths; no `SourceFile`/`DocumentMeta`;
deleted `concept_graph` still described as present); specs `feature-rag-sandbox.md` +
`feature-provider-switch.md` say "NOT built" for shipped U1/U1c (`09afd0c`), 6 more shipped specs
never advanced past design-locked; ADR-002 still `Status: proposed` for the shipped desktop shell;
`docs/decisions.md` monolith never stubbed (ADR-001 step 4) — dual ADR home + 3 dangling
`docs/doc-assistant-roadmap.md` routes inside it; root `HANDOFF.md` (2026-05-26, self-described
transient "delete after pickup") contradicts the current phase map; 19/21 specs lack the line-1 cpc
header; `docs/ui-checklist.md` lags today's shipped work (boxes unflipped).

## 2026-07-16 — Fix: `POST /api/ingest` no-body scope resolves to the canonical path (Windows) + Python 3.12 pin

**What:** `apps/api/main.py` `ingest_start` now reads `app_settings.get_source_dir().resolve()` once at
the top, so the whole endpoint speaks one canonical path — the `scope=` ingest arg, `status.source_dir`,
and the selection pass. The no-body branch previously passed the *un-resolved* `str(source)` as `scope`,
diverging from the `files=` branch (already resolved via `registry.resolve_selection` →
`source_dir.resolve()`) and from the registry's universal `.resolve()` (`scan_sources` / `view_for`).
**Why:** fixes `test_selective_ingest.py::test_api_ingest_no_body_still_works`, a pre-existing failure
pulled in with S1 that tripped on Windows only — the un-resolved path kept the env-derived
`pytest-of-LDELEZ` casing while `src.resolve()` canonicalizes to the on-disk `pytest-of-ldelez` (the
same class of mismatch bites 8.3 short paths and symlinked source dirs). Idempotent in production
(`get_source_dir` already resolves in the env-override and stored-path cases).
**Rejected:** resolving only at the `scope=` call site (leaves `status.source_dir` non-canonical);
"fixing" the test's expectation (`str(src.resolve())` encodes the intended contract, matched everywhere
else). **Opens:** nothing.
**Also (build):** added a tracked `.python-version` = `3.12`. With no pin, `uv run` in a fresh worktree
selected Python 3.14 (`requires-python >= 3.10`) and built a broken venv — the project targets 3.12
(KI-2: native deps not 3.14-stable). Rebuilt this worktree's `.venv` on the official python.org 3.12.10
(`pythoncore-3.12-64`, matching the main venv; the uv-managed standalone hits the OpenSSL-applink crash)
from uv cache (`uv sync --extra cpu --extra dev --offline`, $0 / no network).
**Staged code + docs; `.venv` is gitignored (local only). Nothing committed without review (cpc §13).**

---
## 2026-07-16 — Document safe-delete: source file → Recycle Bin + confirmation (ADR-014)

**What:** single-document delete from the `⋯` menu — the source file goes to the **OS Recycle Bin**
(recoverable) and the doc leaves the library + search index, behind a confirmation dialog.
- **Backend (`library.delete_document(doc_id, chroma_db)`):** recycles the source file **first**
  (`send2trash`, resolved via `resolve_source_path`) — a trash failure raises and **aborts** the delete
  (never orphan a still-indexed file); a file already gone skips trashing. Then drops the `Document` row
  (FK-cascades citations/parts/similarities), the `DocumentMeta` override (no FK — explicit), the doc's
  chunks from the live Chroma store (counted), its figure dir (reuses `cleanup_orphan_figures`) and cached
  `.md`. Returns `DeleteResult(filename, trashed_file, chunks_removed)`. New `DELETE
  /api/library/documents/{id}` → 200 / 404 (unknown) / **409** (couldn't recycle the file).
- **Dependency:** `send2trash>=2.1.0` (base dep, pure-Python cross-platform trash; `uv add --native-tls`
  through the proxy; mypy override added).
- **Frontend:** `LibraryGrid` ⋯ menu gains a red **Delete…** item; new `LibraryDeleteConfirm.svelte`
  (scrim + card, Esc/Cancel/scrim-close) states "source file → **Recycle Bin** (recoverable) … removing
  its N chunks from the search index"; a red-tinted Delete button (busy state). `App.svelte` owns
  `deletingDocId`, drops back to the grid if the open doc was deleted, then re-fetches.

**Why:** user request for a "safe-delete" (file + DB) with a confirmation, and multi-select later.
**Decisions (ADR-014):** Recycle Bin over soft-delete/permanent (user pick); trash-first for consistency;
single-doc first, **multi-select (bulk delete / move-to-collection) deferred**.

**Verified:** `svelte-check` 0/0 (128 files); ruff/format + `mypy --strict src` (61) + bandit clean;
**pytest 15** (`test_document_delete` ×7: unknown→None, file trashed + row + chunks removed, file-already-
gone, **trash-failure aborts + row survives**, DELETE route 200/404/409 — all with a monkeypatched
`send2trash`, no real file touched; + `test_document_meta` ×8). UI live-verified up to the confirmation
(⋯ → red Delete… → dialog with the right target + chunk count + Cancel). **The live end-to-end delete
recycles a real file, so it is NOT run in automation** — the user's to try; the logic is covered by tests.

**Opens:** recovery is via the OS Recycle Bin (no in-app trash/restore view); **multi-select** (bulk +
move-to-collection) is next. The `send2trash` add re-resolved torch off `+cpu` on the proxy box; restored
via `uv sync --extra cpu --extra dev --native-tls`.

**Staged; nothing committed (cpc §13). New dep `send2trash` (pyproject + lock).**

## 2026-07-16 — Library polish: normalized tiles + sort control + active-keyword highlight (user feedback)

**What:** three frontend refinements to the L4 grid (no backend).
- **Normalized tiles** — every tile is now a **uniform height** (161px measured, all identical) because each
  row is a reserved fixed height: title (2 lines, clamped) → **byline** (author · publication year, reserved
  even when empty) → meta (`N pages · N chunks · **Added** <date>` — the ingestion date, now labelled to
  distinguish it from the publication year in the byline) → **keywords (always two reserved lines**, clipped
  beyond). Fixed the **author bug**: `authorLabel` split only on `,;and`, so space-separated author strings
  showed in full — now up to **3 authors show fully** (books/small collabs), **4+ collapse to "First et al."**;
  un-splittable space-only strings ellipsis-truncate (user can fix in the edit modal).
- **Sort control** (`libSort`, persisted) — a `↑↓` button + dropdown in the library toolbar (next to the
  grid/list toggle), options **Title A–Z · Author A–Z · Publication date (newest) · Added date (newest)**,
  applied client-side over the filtered collection (`sortDocs`). Default Title A–Z.
- **Active-keyword highlight** — selecting a keyword collection now surfaces that keyword **first + filled**
  (`.kw.active`) in every tile (`orderedKeywords`), so the reason a doc is listed is always visible even past
  the `+N` cap; the rail chip was already active-styled.

**Why:** user asked for a cleaner, normalized library — uniform boxes, "First et al." + date instead of all
authors, a clear ingestion-vs-publication date, reserved keyword space; a chat-style sort with more keys; and
the selected keyword highlighted/first in the boxes.

**Verified ($0, frontend-only):** `svelte-check` 0/0 (127 files); live on the real corpus — all 12 sampled
tiles measured an **identical 161px**; bylines "Reza Shadmehr, John W. Krakauer · 2008" (2 shown), "Laura E
Suarez et al. · 2022" (4+ → et al.), "2017" (year only); meta shows "Added 7/2/2026"; sort menu lists the 4
keys and Author A–Z reordered correctly; selecting `connectome` → all 18 tiles show it as the **first,
highlighted** chip. Test state reset (sort→default, collection→All).

**Opens:** the residual bad title/author extractions (hyphenation artifacts, a sentence as an author) are now
user-fixable via the edit modal — not a layout bug. Sort is one direction per key (reverse = easy follow-up).
**Deferred (user-requested, next):** document **safe-delete** (file + DB + index, with confirmation), then
**multi-select** (bulk delete / move-to-collection).

**Staged; nothing committed (cpc §13).**

## 2026-07-16 — Document metadata editing + reveal-in-explorer + author on tiles (ADR-013)

**What:** the first browse-time **write path** — a per-document `⋯` menu (Edit metadata / Reveal in
file explorer) mirroring the conversation ⋯ menu, plus the author on its own line on each tile.
- **Data model (`DocumentMeta` sidecar, ADR-013):** new table keyed by `document_id` with
  `title/authors/year_override`. Auto-extracted values stay the *default* on `Document`; **effective =
  override ?? default**; `customized` flags any override. `set_document_meta` **replaces** the small
  override set and **dedups each field against its auto default** (re-saving an untouched field creates
  no override), so a re-run of `enrich_metadata` never clobbers a user edit. Reset = delete the row.
- **Backend (`library.py` + 3 routes):** `set_document_meta`/`clear_document_meta` + `list_documents`
  now merges overrides (batch-loaded once) and carries `year`/`customized`; `resolve_source_path` +
  `reveal_document_source` (`_reveal_in_file_manager`: `explorer /select` / `open -R` / `xdg-open`,
  list-form, no shell). `PATCH /api/library/documents/{id}`, `POST …/reset-metadata`, `POST …/reveal`
  (404 on unknown doc / missing file). `LibraryDocumentPayload` gains `year`/`customized`.
- **Frontend:** `LibraryGrid.svelte` restructured tile/row into a container + body-button + hover-`⋯`
  (a `<button>` can't nest a `<button>`), single floating menu mirroring Sidebar; **author on its own
  muted line** (new `authorLabel`); a small accent "edited" dot when `customized`. New
  `LibraryMetaEditor.svelte` modal (Title/Authors/Year, Save/Reset/Cancel). `App.svelte` owns
  `editingDocId`, wires save/reset/reveal → API → `refreshDocuments` (re-fetch, like the chat actions).

**Why:** user request — correct the ~3% wrong titles + ~19 blank authors the extractor leaves, "like the
chats", with reset-to-default and a "must-have" reveal-in-explorer; author visible on the snippet.
**User decisions:** editable = Title/Authors/Year; author on its own line (a future "choose which columns
show" increment is out of scope but the override model supports it).

**Rejected:** override columns on `documents` (4 additive migrations + mixes user writes into the
extraction registry — the sidecar isolates them); a Tauri command + `shell:allow-open` for reveal (the
app's first Tauri command, untestable in preview — the API is always local, so a backend reveal fits the
100%-API-driven frontend). See ADR-013.

**Verified:** `svelte-check` 0/0 (127 files); ruff/format + `mypy --strict src` (61) + bandit clean;
**pytest 28** (new `test_document_meta`: dedup/blank-revert/reset/effective + PATCH/reset/reveal 200+404
with a monkeypatched reveal; + library regression). Live ($0): applied the new-table migration
(`python -m doc_assistant.db.migrations` — the API doesn't `create_all` on startup, same as
`conversation_meta`); API E2E PATCH→effective→reset→404 PASS; UI — tile shows title + author line +
`⋯`; menu → Edit metadata → change title → Save → tile updates + edit-dot; Reset → reverts (test edit
cleaned up). Reveal opens a real OS window on the host (mocked in tests; live-checked path = user's box).

**Opens:** the residual bad-author/OCR tail is now user-fixable. `document_meta` must be migrated on any
box that predates it (as `conversation_meta` was). Deferred: editing DOI/notes/tags/folders, per-field
reset, bulk edit; user-selectable library columns.

**Staged; nothing committed (cpc §13). `document_meta` table created in the live DB (empty).**

## 2026-07-16 — Keyword de-noising: venue/publisher/ID denylist + repeated-token filter

**What:** the library keyword chips (and the rail's Keywords nav) were dominated by scholarly-metadata
artifacts — `elife` (25 docs), `biorxiv`, `neuroimage`, `jneurosci`, `neurobiol`, `fnana`, `frontiersin`,
`zenodo`, `pmid`, `7554 elife` (the eLife DOI registrant). Two filters in `keywords.py::candidate_terms`
(the single choke-point every mode feeds through): (1) a curated **`VENUE_STOPWORDS`** frozenset (preprint
servers / repositories / publishers / journal abbreviations / ID labels) — a candidate is dropped if ANY
of its tokens is a venue token, so `elife` and the bigram `7554 elife` both go. **Deliberately excludes
words that double as domain concepts** (`cell`/`neuron`/`nature`/`science`) so real keywords survive.
(2) a **repeated-token n-gram** reject (`outflux outflux outflux` — an OCR artifact weirdness scored highly,
the exact case RG-001/R3 flagged). Regenerated the corpus vocabulary (`extract_keywords --mode contrastive
--force --apply`), sweeping the now-orphaned venue rows.

**Why:** user feedback — the chips were venue noise, not topics; "better metadata → better tags/keywords for
navigation." This is the follow-up lever R3 explicitly parked ("STOPWORDS/metadata strip for publisher
artifacts; collapse repeated-token grams").

**Rejected:** filtering author surnames (`sporns`/`cajal`) — not deterministically separable from topics
(`cajal` is both a person and a body of work), left as the manual-curation tail; denying `cell`/`neuron`/
`nature` (real neuroscience concepts that happen to be journal names) — would strip genuine keywords;
re-tuning the contrastive weirdness/C-value knobs (locked settings — a denylist is the surgical fix, no
eval needed since it only removes provably-non-topical tokens).

**Verified ($0/offline, deterministic):** dry-run reviewed before applying — venue tokens **0 remaining**,
the 60-term vocabulary now reads `embeddings`/`connectome`/`deeplabcut`/`bm25`/`cebra`/`res2net`/
`parcellation`/`tractography`/`markerless`/`keypoints`; ruff + `mypy --strict` clean; **pytest 53**
(3 new `candidate_terms` cases — venue+ID drop, keeps venue-homonym domain words, repeated-token drop — plus
the concept-skeleton suite that shares `candidate_terms`, unregressed). Applied to the live `data/library.db`
(**268 links, 14 orphan rows swept** across two passes); reloaded — tiles show clean domain chips, the
political-science paper honestly shows none (no domain keyword in a neuro/ML vocabulary).

**Opens:** residual non-venue proper-noun/OCR tail (`sporns`, `cajal`, `huggingface`, `mathrm`, `neurosc`)
— the **manual keyword/tag edit UI** (needs an ADR, first browse-time write path) is the fix. Concept
skeleton unaffected (built from the 26 curated concepts, not raw keywords). Re-runnable any time.

**Staged; nothing committed (cpc §13). Keyword rows regenerated in the live DB (reversible — re-extract).**

## 2026-07-16 — Metadata enrichment: wire the (unwired) extractor onto Document (real titles on tiles)

**What:** populated the empty `Document.title`/`authors`/`year`/`doi` columns (0/76 → title 76/76,
authors 57/76, year 66/76, doi 25/76) so the library grid shows real titles instead of filenames.
(1) **`metadata_enrich.py`** (new) — the runner: reads each doc's cached markdown (reuses
`keywords.load_document_texts`), runs the existing `metadata_extractor.extract_metadata`, and writes the
four columns. **Idempotent per column** — only fills a NULL unless `force` (so a later manual edit is never
clobbered); `apply=False` is a $0 dry run. Enrichment-Layer discipline: writes only those columns, never the
chunk store. (2) **`scripts/enrich_metadata.py`** (new) — thin CLI mirroring `compute_doc_vectors.py`
(`--apply`/`--force`/`--doc`, dry-run report). (3) **`metadata_extractor.py`** — the extractor was **fully
built but never called anywhere** (dead since Phase 4); wiring it revealed three false-positives on the real
corpus, fixed: skip publisher copyright/licence headings (Springer's "The Author(s), under exclusive
licence…" was hijacking a title), strip markdown hard-break backslashes (`WIESEL\`), and reject author
candidates that open with a discourse/section lead (`However,` / `Additional Key Words and Phrases:` — never
a name list; honest-empty beats a wrong author, so authors fell 61→57).

**Why:** user feedback on the new grid — filenames are unreadable; "if we improve the metadata we can make
better tags/keywords for navigation." Chose a **deterministic sidecar over editing source files** (never
mutate the user's PDFs — enrichment-layer + provenance) and **auto-fill first, manual-edit later** (the
manual-edit UI is the first browse-time write path → its own ADR, deferred).

**Rejected:** rewriting/renaming source PDFs (destructive, breaks re-ingest + provenance); an LLM extraction
pass (unnecessary — the deterministic heuristics already hit 100% titles on this corpus, $0/offline, and the
cost-discipline rule says prove the deterministic path first); storing `authors` as JSON (the frontend
`docLabel` splits a delimited string — kept the extractor's string form).

**Verified ($0/offline, deterministic):** dry-run on the real 76-doc corpus reviewed **before** applying
(quality gate); ruff/format clean; `mypy --strict` (2 files) clean; **pytest 27** (`test_metadata_extractor`
+ new `test_metadata_enrich`: apply-fills-NULL / dry-run-writes-nothing / idempotent-keeps-title /
force-overwrites). Applied to the live `data/library.db` (**224 columns written**); reloaded the grid —
tiles now read "A Primer on Motion Capture with Deep Learning…", "Res2Net: A New Multi-scale Backbone
Architecture · Shang-Hua…" etc., filename preserved as the hover `title`.

**Opens:** ~19 docs have no confident author (honest-blank) + a few noisy year picks — the **manual-edit UI**
(needs an ADR) is the fix for stragglers. **Keyword de-noising** is the natural next step (chips still show
venue/ID artifacts — `elife`/`biorxiv`/`zenodo`/`7554 elife` — from `keywords.py`, a separate change).
DOI 25/76. Re-runnable any time (`--force` to refresh, only-NULL by default).

**Staged; nothing committed (cpc §13). Metadata applied to the live DB (reversible — only-NULL fills).**

## 2026-07-16 — Library grid: mode-aware width + fixed-footprint tiles (user feedback)

**What:** two frontend fixes to the L4 grid from live-review feedback. (1) **Mode-aware main width**
(`App.svelte`) — `<main>` was hard-capped at `max-width: 820px` (the chat reading measure, ~68ch) and
centered, so in fullscreen the **library grid floated in a centered 820px column** with wide empty margins
and stayed stuck at 4 columns. Added `main.wide` (`max-width: 1500px`), bound `class:wide={mode ===
'library'}`; Chat keeps its 820px reading column. Measured at 1456px viewport: main 820→1122px, grid
775→1077px, **4→5 columns**, right-side whitespace 217→0. (2) **Fixed-footprint tiles** (`LibraryGrid.svelte`,
best-practice list from the user) — `.tile` gets `min-height: 128px`; `.name` clamps 3→**2 lines** with a
reserved `min-height: 2.7em` so a long filename (`2021.04.30.442096v1.full.pdf`) no longer reflows its row
(all tiles a uniform 140px); keyword chips cap at 3 + a **"+N" overflow chip** (`title` = the hidden ones)
so tags never wrap unpredictably; grid `minmax(150px→200px, 1fr)` for a comfortable tile; `aria-label={
docLabel(d)}` on tile+row buttons gives a clean accessible name instead of the whole-card text mash.

**Why:** the user tested the pulled-in Library Grid (9f597df) fullscreen and flagged the "weird white space"
+ long titles reflowing the grid; supplied a card-grid best-practices list.

**Rejected (from the list, not applicable here):** whole-card-as-`<a>` + CSS overlay (this is an in-app
open action, not URL navigation — a `<button>` with a clean `aria-label` is the correct semantic);
`flex-wrap: nowrap` clipping on the tag row (risks invisible tags — the 3+"+N" cap already bounds it);
skeleton loading states (deferred — docs load once from the local API, no async card-populate jank to mask).

**Verified ($0, frontend-only):** `svelte-check` **0/0** (126 files); live at 1456px — 5 uniform 140px tiles
per row filling the pane, 0 right-whitespace, "+8" overflow chip rendered, long filename clamped to 2 lines
(33px); Chat mode still `main` 820px / no `.wide`. Screenshot captured.

**Opens:** the 1500px cap centers the grid on ultrawide (>~1760px) — a readable-width choice, revisit to
`none` if full-bleed is wanted. Sidebar width is a separate user-resizable pref (default 260, persisted).
Next: **metadata enrichment** (real titles on tiles + de-noised keywords — the higher-leverage follow-up).

**Staged; nothing committed (cpc §13).**

## 2026-07-15 — Selective ingestion S2: Sources panel (scan · exclude · ingest-selected) in Settings

**What:** the S2 frontend over the S1 endpoints. (1) **`types.ts`** `SourceFile` (mirrors
`SourceFilePayload`) + **`api.ts`** `getSources()` / `patchSource(rel_path, excluded)` and
`startIngest(paths?)` extended to POST `{paths}` when a selection is given (no-arg keeps whole-dir);
`errorDetail` now renders the selective-ingest 400's `{error, offenders}` object, not `[object
Object]`. (2) **New `Sources.svelte`** — scans on mount (`GET /api/sources`, $0 stat-only) + a Rescan
button; per-file row = select checkbox + rel_path + status chip (new/changed/**indexed**/missing) +
an Exclude/Excluded toggle (`PATCH`); a **"Select new + changed"** quick action; **"Ingest selected
(N)"** → `startIngest([paths])` → the same tolerant poll as the whole-folder index → on done
`onCorpusChanged()` + clear selection + rescan. Excluded rows dim and drop from the selection;
`missing` rows can't be selected. (3) **Settings.svelte** mounts it as a new **"Manage files"**
section under "Your documents" (the parked S2-shape fork — resolved with the user to **fold into
Settings**, not a 3rd sidebar mode: simplest V1, ingestion stays in one place).

**Why:** S2 of `feature-selective-ingestion.md`. The M4 flow does whole-folder "index everything";
this adds per-file visibility + exclude + subset-ingest for a mixed/flat corpus, on the same locked
core (nothing in `src/` changed — pure renderer over the S1 API).

**Rejected:** (a) a dedicated **Sources sidebar mode** (Chat/Library/Sources) — more wiring + a
permanent tab for an occasional task; the user chose the Settings section for a simpler V1 (a mode is
a clean future upgrade if the file list outgrows the drawer). (b) a `doc_type` control — the column
is dormant (S1 lock), so no UI. (c) a CLI-style exclude in the panel — exclusion is one PATCH toggle.

**Verified ($0/offline, real 47-doc corpus):** `svelte-check` 0/0 (126 files). Live via the preview
harness: **`GET /api/sources` → 47 files all correctly derived `ingested`** (the `has_document`
path-match + cache-freshness both work on the real DB); the panel renders 47 rows with "indexed"
chips + exclude toggles; **exclude → `PATCH` persists** (fresh server GET confirms `excluded:true`,
row dims, summary "1 excluded") and **include reverts** it; **select a file → "Ingest selected (1)"
→ `POST {paths}` → `resolve_selection` → `main(files=[…])` → dedup-skip → "indexed 0 new, 1 unchanged,
0 errors"** ($0), then the UI shows done + clears the selection + rescans; dark theme resolves (muted
chips, dimmed excluded rows); mobile 375px the 92vw drawer rows fit with no overflow; **0 console
errors**. Test state restored (no excluded files).

**One-time migration surfaced (not an S2 bug):** this box's `data/library.db` was missing tables
(`conversation_meta` *and* the new `source_files`) — the API relies on ingest's `init_db()`
`create_all` to migrate, and it hadn't run since S1. Ran `init_db()` once → both created (23 tables),
`/api/sources` then worked. Same pattern as every prior additive table (Figure, gaps): an upgrading
user creates `source_files` on their next ingest. **Opens:** the API not migrating on startup is a
pre-existing latent gap (a stale DB 500s `/api/conversations` too) — a systemic "init_db in lifespan"
fix is its own change, out of S2 scope.

**Opens:** S2 could grow into a dedicated mode if the flat 47-file list ever needs full width. PR 17
(Zotero/Calibre) will surface as extra `SourceFile` producers. Touch note: the exclude toggle + row
checkbox are tap targets; no drag affordances involved.

**Staged; nothing committed (cpc §13).**

## 2026-07-15 — Selective ingestion S1: SourceFile registry + selection-scoped ingest (backend + CLI + API)

**What:** the S1 backend of `docs/specs/feature-selective-ingestion.md` (grilled + LOCKED same day).
Ingest is no longer all-or-nothing over one folder. Five parts.
(1) **`SourceFile` table** (`db/models.py`) — one row per discovered file: `rel_path` (unique key,
POSIX), `format`/`size`/`mtime`/`first_seen`/`last_seen` (identity), `excluded` (the one persisted
user intent), and a **dormant nullable `doc_type`** (ships now, wired to nothing — `create_all` can't
ALTER a column later, so a dormant column makes doc_type's future return a behaviour-only add). Rides
the additive `init_db()` `create_all`, like `Figure`. (2) **New `ingest/registry.py`** — pure core
(`derive_status` 8-combo truth table → new/changed/ingested/missing; `validate_selection` →
normalized rel_paths or `InvalidSelection` listing every offender by reason) + impure boundary
(`scan_sources` stat-only walk that upserts rows + derives status with **no content reads**;
`set_source_meta` PATCH seam; `resolve_selection` → explicit paths, explicit picks override
`excluded`; `plan_files` dry-run classifier; `view_for` single-row echo). (3) **`ingest.main(files=,
dry_run=)`** — `files` is a validated explicit list (mutually exclusive with `--path`/`--rebuild`,
skips orphan cleanup); an implicit walk now subtracts standing exclusions; `dry_run` reports
`would_add`/`would_reembed`/`skip_unchanged`/`excluded` **without loading embeddings or opening
Chroma**. (4) **CLI** `--files P…` / `--dry-run` with the exclusivity rules. (5) **API** — `GET
/api/sources` (scan + list), `PATCH /api/sources` (`excluded` only; 404 unknown), `POST /api/ingest`
gains an optional `{paths?}` body resolved + validated up front (bad path → 400 before anything
starts; no body = whole dir minus exclusions). Wire models `IngestRequest`/`SourcePatch`/
`SourceFilePayload` (named `SourceFile*` to avoid the citation-`SourceView` collision).

**Why:** the "in-app ingestion" pivot after L4. A flat mixed corpus needs user-controlled selection —
batch (pick a subset) + on-need (by status). The M4 "index everything" flow shipped; this adds the
registry + selection layer on the same locked ingest core (its six stages are untouched — this only
changes *which files enter*).

**Rejected / deferred (grill lock 2026-07-15, ledger in the spec):** (a) **`doc_type` behavior** —
deferred (all-PDF corpus → manual busywork, no consumer yet; it's not a chunk/embed lever); only the
dormant column ships. (b) **stateless computed listing** (no table) — rejected: persistent `excluded`
has nowhere else to live (no `Document` row pre-ingest), and the table is the status-listing source +
the PR-17 adapter seam. (c) **`default_doc_type` seeding fn** — omitted (no dead code; lands with the
column's activation). (d) **S2 UI shape** — parked to S2 kickoff. (e) a CLI exclude command — not
needed; `excluded` is set via the API/UI, honored by every surface.

**Verified (offline, $0, no real embeddings — cpc §13):** full gate green — ruff / ruff format /
`mypy src/` (60 files) / bandit (no issues) / **920 pytest passed** (+27: 19 unit truth-table +
validation, 8 integration) / coverage 84% (registry.py 96%). Integration proof on tmp dirs (no
corpus, no network): scan lifecycle new→ingested→changed→missing; resolve_selection excludes then an
explicit pick overrides; bad paths raise `InvalidSelection`; `main(dry_run=True)` returns the plan
with a `get_embeddings` **trap that never fires**; the API GET→PATCH→POST`{paths}` flow drives a fake
`ingest_fn` that receives `files=[b.pdf]` (scope None), unknown PATCH → 404, traversal path → 400,
no-body POST still uses the scope path. Zero regressions in the 125 existing ingest tests.

**Opens:** **S2** (Tauri Sources panel — status chips, exclude toggle, ingest-selected) — UI shape
(dedicated sidebar mode vs fold into Settings) decided at kickoff. PR 17 (Zotero/Calibre) writes this
registry through the same public fns (ADR-3 seam). `doc_type` reactivation when a 2nd format or a
routing eval arrives. Latent, pre-existing: a `source_dir` outside `config.DOCS_PATH` has no
resolvable markdown cache (`get_cache_path` is DOCS_PATH-relative) — `scan_sources` degrades such a
file to not-fresh rather than crash; unchanged by this work.

**Staged; nothing committed (cpc §13).**

## 2026-07-15 — Library redesign L4 Phase A: nav-tree rail + inventory grid + drill-down

**What:** the Library space is rebuilt from a flat doc list into a file-browser-style navigation
(spec `docs/specs/feature-library-redesign.md`, design-locked 2026-07-14). Three parts.
(1) **New `lib/library.ts`** — a client-side collection model: `LibraryCollection` (`all`/`type`/
`date`/`folder`/`keyword`), `docsFor()` to filter the cached document list, `dateBucket()`
(Today/This week/This month/Earlier relative to now), the `typeGroups`/`dateGroups`/`folderGroups`/
`keywordGroups` counters, and `docLabel`/`filterDocs` (moved out of `Sidebar.svelte` — the grid,
breadcrumb, and search all need them now). (2) **New `LibraryGrid.svelte`** — the "inventory" tile
grid (`repeat(auto-fill, minmax(150px, 1fr))`): each tile shows a format chip, the title/filename
(3-line clamp), page·chunk·date meta, and up to 3 keyword chips; a `list` view renders the old
stacked-row idiom instead. Dumb component — no filtering, emits `onOpenDocument(id)`. (3) **`Sidebar`
library mode** is now the nav tree: **All documents** → **Collections** (Phase-A empty-state until
folders populate) → **Types** (by `format`) → **Added** (date buckets) → **Keywords** (chips); Types
and Added render only with **≥2 entries** (a one-option filter is noise). (4) **`App.svelte`** owns
the drill-down: `libraryCollection` + `libraryDocId` + `libraryQuery`, a breadcrumb `Library ›
Collection › Doc` with a Back control (doc→grid, then collection→all), and the grid⇄list toggle
persisted in `localStorage` (`libraryView`). Library search now scopes to the **active collection**
(bindable `libraryQuery`) with a one-click "Search all N documents" escape on 0 matches. +7 Lucide
glyphs (`layout-grid`, `list`, `folder`, `file-text`, `calendar`, `tag`, `chevron-right`).

**Why:** the next Library increment after L1 (which parked search/filter/sort + folder navigation).
The user chose drill-down-with-Back over two-pane / detail-drawer from a clickable prototype; the
persistent rail means changing collection never needs "back" — only the doc→chunks step drills.

**Rejected:** (a) two-pane (chunks always in a right pane) and detail-drawer navigation — the user
preferred the file-browser feel (drawer noted as a possible later toggle reusing `SourcePanel`); (b)
a backend folder-tree endpoint now — Phase A filters entirely client-side because the
`LibraryDocument` payload already ships `format`/`added_at`/`keywords`/`folders`; folders (the one
empty axis on the current flat-ingested corpus) are Phase B (mirror source-dir subfolders at ingest
+ a backfill); (c) always-visible Types/Added sections — hidden below 2 entries so a single-format
corpus doesn't show a dead filter.

**Verified ($0, frontend-only, no backend/LLM change):** `svelte-check` 0/0 (125 files). Live on the
real corpus via the preview harness: nav tree renders (Collections empty-stated, **Types correctly
absent** — all-PDF corpus <2 types, **Added shows 2 buckets** — This month + Earlier, keyword chips);
inventory grid = 47 tiles at 4-col auto-fill; clicking the "medical image segmentation" keyword →
exactly 3 docs + breadcrumb `Library › medical image segmentation` + Back appears; opening a tile
drills to the `LibraryBrowser` chunk view (breadcrumb gains the doc crumb, view toggle hides); Back
returns to the keyword grid; grid⇄list toggle flips (3 rows / 0 tiles) and persists `libraryView`;
searching "colbert" inside the keyword collection → 0-match empty-state → "Search all 47 documents"
widens to All documents keeping the query → the single ColBERT match; dark theme resolves (`--bg`
`#1b1813`, dark `--lavender` `#b0a4ff`, tile surface `#23201a`), no body horizontal overflow; mobile
375px → off-canvas drawer (fixed, `translateX(-100%)` closed, `.open` + scrim on hamburger) + 2-col
grid; 0 console errors. Test residue restored (`libraryView` cleared, viewport/scheme reset).

**Opens:** Phase B — folder population (source-dir mirror at ingest + a backfill runner) + `GET
/api/library/folders` + server-side `folder`/`format`/`tag` filters, which lights up the Collections
section. Still parked: manual folder/tag editing (first browse-time write path — ADR trigger),
title/author metadata backfill (tiles show filenames until then), the detail-drawer variant,
virtualization for very large collections.

**Staged; nothing committed (cpc §13).**


---

*Earlier entries (2026-07-14 back to 2026-05-21) archived verbatim to [docs/archive/DEVLOG-archive-001.md](archive/DEVLOG-archive-001.md) on 2026-07-21.*
