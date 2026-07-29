<!-- status: active · updated: 2026-07-28 · class: append-only -->

# Stage profile — where the time lives (2026-07-28)

**Question.** Which parts of the pipeline are affordable at runtime, which are interactive-but-slow,
and which are batch-only — so that "do we have to re-scan / re-embed?" becomes a decision against
numbers instead of intuition.

**Instrument.** `scripts/profile_stages.py` (new). Times each stage in isolation on the **live
corpus**, `$0` (no LLM provider constructed, no generation). The embedding measurement calls the
embedder on real chunk text and **writes nothing**. Sidecars are timed by shelling out to their real
runners in **dry-run** mode — verified to do the full computation and skip only the DB write (both
`compute_epistemics` and `extract_citations` print per-document results on a dry run).

**Box / corpus (read every number below against this).**

| | |
|---|---|
| Corpus | **97 documents · 33,163 parent-child chunks** |
| torch device | **`cpu`, torch 2.12.0+cpu** — on a machine with a working NVIDIA GPU (a carried item, see Recommendations) |
| Embedder / reranker | `bge-base-en-v1.5` / `BAAI/bge-reranker-base` |
| Retrieval | parent-child ON, `CANDIDATE_K=20`, `TOP_K=10` |
| Python / OS | 3.12.3 / Windows 11 |
| Repeats | 3 per stage (1 for one-shot cold measurements); min-max range reported |

---

## 1. Startup — paid once per backend launch

Cold launch, measured in a fresh interpreter, 3 samples: **15.95 / 16.14 / 16.62 s** (median
**16.1 s**). Decomposed by cumulative subprocess measurement:

| Component | Cumulative | **Marginal** | Scales with |
|---|---:|---:|---|
| Python imports (torch + langchain + sentence-transformers) | 6.83 s | **6.8 s** | nothing |
| + embedder weights | 9.20 s | **2.4 s** | nothing |
| + reranker weights | 12.88 s | **3.7 s** | nothing |
| + Chroma open · whole-store read · BM25 build · chat model | 16.1 s | **~3.2 s** | **corpus** |

In-process component detail for that last row:

| Stage | Median | Range | Note |
|---|---:|---|---|
| open Chroma handle | 10.9 ms | 0.010-0.338 | |
| read all 33,163 chunks (paged, KI-27) | **1.13 s** | 1.108-3.684 | scales with corpus |
| build BM25 index over 33,163 chunks | **698 ms** | 0.650-0.705 | **in memory only, never persisted — recomputed every launch** |

⚠ **One outlier worth naming:** the first in-process `get_embeddings()` of the session measured
**30.7 s**, against 2.4 s marginal in a warm-file-cache subprocess. That is the OS file cache, not
the code — the *first* launch after a boot can cost roughly double the 16 s figure. Do not quote
30.7 s as the load cost; do not quote 16 s as the worst case either.

## 2. Query — paid per turn (retrieval only; generation excluded)

| Stage | Median | Range | Share of the 907 ms |
|---|---:|---|---:|
| embed the question | 16.5 ms | 0.016-0.017 | 2% |
| vector search (k=20) | 179 ms | 0.178-0.217 | 20% |
| BM25 search (k=20) | 28 ms | 0.027-0.031 | 3% |
| ensemble (both arms + fusion) | 228 ms | 0.227-0.236 | 25% |
| **`retrieve_with_scores` (retrieve + cross-encoder rerank + parent expand)** | **907 ms** | 0.904-0.932 | 100% |
| ⇒ **cross-encoder rerank + expand, by difference** | **~680 ms** | — | **75%** |

**The reranker is three quarters of the retrieval budget**, on CPU, for 20 candidate passages.
Generation is not in here — it is provider-bound and measured separately by
`scripts/measure_latency.py` (RG-011).

## 3. Ingest — paid per document, re-paid in full when the embedded text changes

Median-sized document of the corpus (`nihms133032.pdf`, 1.9 MB source, 427 child chunks):

| Stage | Median | Scale |
|---|---:|---|
| **read cached markdown** | **0.3 ms** | per document — *this is what a re-scan of an unchanged document costs* |
| extract PDF → markdown, **cold** (cache bypassed) | **24.2 s** | per document |
| chunk parent-child (→ 427 children) | 9.2 ms | per document |
| embed 64 chunks | 1.95 s | **30.5 ms/chunk** |
| ⇒ embed one document (427 chunks, projected) | **~13 s** | per document |
| ⇒ **full re-embed of 33,163 chunks (projected)** | **~1012 s ≈ 17 min** | extrapolated from the batch, not run end to end |

**A re-scan is already free; a re-embed is not.** Extraction is content-cached and ingest dedupes by
hash, so re-running ingest over unchanged documents is sub-millisecond per document plus the hash
check. The expensive, unavoidable-on-change work is **embedding**, and the number that governs every
"should we change stored text?" decision is **30.5 ms/chunk** — i.e. **~17 minutes of CPU for the
whole corpus** (this supersedes the ~40 min guess recorded in KI-29 before it was measured).

## 4. Sidecars — full-corpus dry runs, and how little `--doc` actually saves

| Runner | Whole corpus | One document | Saving | Verdict |
|---|---:|---:|---:|---|
| `extract_doc_metadata` | 0.58 s | *n/a* | — | cheap either way |
| `build_gaps` | 0.81 s | no flag | — | full recompute only, but cheap |
| `compute_doc_vectors` | 2.88 s | 2.30 s | 20% | `--doc` filters the **report**; its own help says "computation is always global" |
| `extract_keywords` | 4.14 s | **3.97 s** | **4%** | `--doc` scopes the *write*, not the work — the corpus TF-IDF loads everything (**KI-18, now with a number**) |
| `compute_epistemics` | 5.61 s | no flag | — | full recompute only |
| `extract_citations` | **54.1 s** | **failed** | — | the slowest sidecar, and its `--doc` **cannot be driven by a document id** (below) |

**Two findings in that table.**

1. **`--doc` means three different things across four runners**, and only `extract_keywords` matches
   on the document **id** that every other surface (API, graph, library) hands you.
   `extract_citations` and `extract_doc_metadata` match on **`doc_hash` prefix** while their help
   says *"one doc_hash or id prefix"* — so passing a real id exits 1 with `No documents matched.`
   Filed as **KI-30**.
2. **Nothing in the enrichment layer is meaningfully incremental yet.** The one runner that could
   save the most (`extract_citations`, 54 s) is the one whose scoping is unusable; the two that
   accept scoping recompute globally anyway.

**A reconciliation worth recording:** KI-18 quotes the epistemics projection at *"~34 s @ 47 docs"*,
which looks incompatible with **5.61 s @ 97 docs** here. It is not — the cost is
O(chunks × **vocabulary**), and the ADR-018 rescope cut the graph vocabulary from 357 concepts to 13
(`graph_include`), ~27x. Both numbers are consistent once vocabulary is held in view, which is
precisely why KI-19 warns against citing these constants without the experiment.

---

## Classification — the answer to "what can run at runtime?"

| Budget | Stages | Design consequence |
|---|---|---|
| **Runtime, per keystroke** (<50 ms) | question embed (16 ms), BM25 search (28 ms), chunking (9 ms), cached-markdown read (0.3 ms) | free to call on demand |
| **Runtime, per turn** (100 ms-1 s) | vector search (179 ms), ensemble (228 ms), rerank (~680 ms) | fine per question; the rerank is the only part worth optimizing, and it is GPU-bound |
| **Interactive-but-slow** (1-60 s) | app launch (16 s), whole-store read + BM25 build (1.8 s), every sidecar except citations (0.6-5.6 s), one document's embedding (~13 s) | acceptable as a one-off or a background job with progress; **not** in a request handler |
| **Batch only** (minutes) | cold PDF extraction (24 s **per document** ⇒ ~40 min for 97), full re-embed (~17 min), `extract_citations` full corpus (54 s and superlinear in reference count) | must be a job with progress + resumability; never blocking |

**What invalidates what** (the practical form of "do I have to re-run everything?"):

| Change | Forces | Cost on this corpus |
|---|---|---|
| New/changed source document | extract + chunk + embed **that document only** | ~24 s + ~13 s |
| Re-run ingest with nothing changed | hash check only | seconds total for 97 docs |
| **Stored chunk text changes** (e.g. the KI-29 page-marker strip) | **full re-embed** | **~17 min** |
| Chunk size / overlap change | full re-embed | ~17 min |
| Embedding model change | fresh collection + full re-embed | ~17 min |
| Concept vocabulary change (`graph_include`) | skeleton + gaps + epistemics | ~7 s |
| Anything at all | BM25 rebuild at next launch | 0.7 s (today) |

## Recommendations, in leverage order

1. **Install the CUDA torch extra on this box** (`uv sync --extra cu130 --extra dev`). Every number
   above that involves a transformer is a CPU number on a machine with a usable GPU — already a
   carried item in the baton. The repo's own README claims **~70 ms** retrieve+rerank on an RTX 4070
   against the **907 ms** measured here; that is a repo claim, not measured today, but the rerank
   (75% of the query budget) and the embed rate (30.5 ms/chunk, which sets the 17-minute re-embed)
   are exactly the two things a GPU addresses. Re-run this profile after switching to replace every
   figure above.
2. **Defer the reranker load.** It is constructed eagerly in `RAGPipeline.__init__`
   (`pipeline.py:188`) but not needed until a question has already been retrieved for — **3.7 s off
   every launch**, for a load the first query would absorb.
3. **Persist the BM25 index.** 1.13 s read + 0.70 s build is tolerable at 33k chunks and is the
   startup component that **scales with the corpus** — at the 10k-document contract it is the
   dominant launch cost, and it is recomputed from scratch every time.
4. **Fix `--doc` before optimizing anything else in the enrichment layer** (KI-30). Incremental
   enrichment is unreachable while the flag is inconsistent, and `extract_citations` (54 s) is where
   the saving would be.
5. **Treat cold extraction as the batch cost it is** (24 s/document ⇒ ~40 min for this corpus). It is
   already cached, which is why re-scans are free; what is missing is progress/resumability on the
   first pass over a large folder.
6. **Decide KI-29 with the 17-minute number in hand**, not the 40-minute guess.

---

## 5. Per-document estimates (added 2026-07-28, `--docs 8 --extract`)

Eight documents sampled across the **size distribution** (0.1 MB → 32.4 MB), always including both
extremes, so "best/worst" comes from the real tails rather than whichever file sorted first. Every
extreme names its document, because "worst case 30 s" is trivia and "worst case 30 s on
`2203.07436v4.pdf`" is a reproducible starting point.

| Stage | Mean | Best | Worst | Worst/best |
|---|---:|---:|---:|---:|
| read cached markdown | 0.4 ms | 0.2 ms (`fpos-6-1305055.pdf`) | 0.8 ms (`2203.07436v4.pdf`) | 4x |
| chunk parent-child | 7.5 ms | 1.0 ms (`rnn_regularization_zaremba_2014.pdf`) | 19.3 ms (`2203.07436v4.pdf`) | 19x |
| **embed one document** | **12.0 s** | **2.8 s** (`rnn_regularization_zaremba_2014.pdf`) | **30.2 s** (`2203.07436v4.pdf`) | 11x |
| **embed rate** | **31.6 ms/chunk** | 27.8 ms/chunk (`elife-103977-v1.pdf`) | 36.7 ms/chunk (`specter2_scirepeval_singh_2022.pdf`) | **1.3x** |
| **extract PDF → markdown (COLD)** | **15.2 s** | **5.7 s** (`rnn_regularization_zaremba_2014.pdf`) | **36.7 s** (`2203.07436v4.pdf`) | 6.4x |

Per-document detail:

| Document | Size | Chars | Chunks | Embed | Rate |
|---|---:|---:|---:|---:|---:|
| `rnn_regularization_zaremba_2014.pdf` | 0.1 MB | 23k | 99 | 2.8 s | 28 ms |
| `fpos-6-1305055.pdf` | 0.6 MB | 56k | 216 | 6.8 s | 32 ms |
| `specter2_scirepeval_singh_2022.pdf` | 0.9 MB | 95k | 366 | 13.4 s | 37 ms |
| `31870-130793-1-PB.pdf` | 1.5 MB | 39k | 165 | 4.9 s | 29 ms |
| `scaling_laws_kaplan_2020.pdf` | 2.5 MB | 94k | 376 | 13.2 s | 35 ms |
| `elife-78635-v2.pdf` | 5.8 MB | 103k | 406 | 11.7 s | 29 ms |
| `elife-103977-v1.pdf` | 9.6 MB | 114k | 462 | 12.8 s | 28 ms |
| `2203.07436v4.pdf` | 32.4 MB | 224k | 867 | 30.2 s | 35 ms |

### The estimator worth remembering

Two stable ratios fall out, and they are what make a per-document estimate possible **without
running anything**:

* **~4.0 child chunks per 1,000 characters** (measured range 3.85-4.30 across the eight — remarkably
  tight given a 14x span in chunk count).
* **~32 ms per chunk to embed** (range 27.8-36.7, i.e. **±14%** — the rate barely moves with document
  size, so per-document embed cost is essentially *linear in chunk count*).

⇒ **embed ≈ 128 ms per 1,000 characters of extracted text**, ±15%.

**Size in MB is the wrong predictor; character count is the right one.** `31870-130793-1-PB.pdf` is
1.5 MB but only 39k chars (165 chunks, 4.9 s), while `fpos-6-1305055.pdf` is 0.6 MB with 56k chars
(216 chunks, 6.8 s) — MB tracks embedded images and scan quality, not text volume. Any progress bar
or cost estimate should be driven by extracted characters (available straight after extraction), not
file size.

**Full re-embed of the live corpus: 1047 s ≈ 17.4 min** (15-20 min using the best/worst per-chunk
rates). This is the second independent measurement of that figure and it agrees with §3.

---

## 6. First improvement landed: the reranker is now lazy (2026-07-28)

Recommendation 2 above, implemented and measured. `RAGPipeline.reranker` became a lazily-loading
property instead of an eager `__init__` attribute.

| | Before | After |
|---|---:|---:|
| Cold launch (fresh process, 3 samples) | 16.1 s (15.95/16.14/16.62) | **11.7 s** (11.99/11.63/11.51) |
| First query | — | pays the ~3.7 s load once |
| Later queries | unchanged | unchanged |

**Measured saving: ~4.4 s off every launch**, slightly better than the 3.7 s predicted from the
component split. **State the trade honestly:** the launch is 4.4 s shorter and the *first* question
of a session is ~3.7 s longer. For a desktop app that is the right side of the trade — the readiness
gate unblocks the UI sooner, and the first question is typically preceded by typing — but it is a
trade, not a free win.

Verified live, `$0`: construction no longer loads the weights (`rag._reranker is None`), the first
query logs `loading_reranker` and returns 10 sources, scores stay **sigmoid-bounded in [0, 1]** (the
integrity layer depends on that — see `_sigmoid_activation_kwarg`), top score 0.98 on the warm query.

A side effect worth recording: typing the property surfaced that the `predict` call site had **never
been type-checked** — the eager attribute was inferred as untyped. The sentence-transformers stub
models only a single pair or a flat list, never the batch-of-pairs form that is the documented way to
score N candidates, so the call now carries a narrow, explained `# type: ignore[arg-type]`. Runtime
shape unchanged.

## Reproduce

```bash
uv run --no-sync python -m scripts.profile_stages -r 3 --ingest --docs 8 --extract
uv run --no-sync python -m scripts.profile_stages -r 1 --sidecars --sidecar-timeout 900
```

Raw samples ride in the `--json` output. Numbers move with corpus size and torch device; re-record
this file rather than editing it (append-only), and state both in the header.
