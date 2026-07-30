<!-- status: active · updated: 2026-07-30 (§3 appended: KI-32 step 1 measured — 1.36x, not the predicted 3x) · class: append-only -->

# Memory footprint + the lazy-reranker penalty, re-measured on the CUDA wheel (2026-07-30)

**Question.** Two figures the optimisation pass of 2026-07-28/29 left unmeasured, both of which the
trade-off ledger in `docs/performance.md` depends on:

1. **How much RAM does the answer path hold?** The BM25 arm materialises every chunk as a
   `Document` and keeps it for the whole process (`pipeline.py:166`, retained for ADR-025 F2
   folder-scoped retrieval). ADR-035 persisted that corpus to disk to cut launch time, but the
   *memory* it occupies was never measured, so "scales with the corpus" had no number attached and
   the 10,000-document robustness contract could not be argued about.
2. **What does the lazy reranker actually cost the first question?** The 2026-07-28 baseline records
   **~3.7 s**, derived from a CPU subprocess split. This box is now on `cu130`, and a GPU changes the
   *load* as well as the scoring.

**Box / corpus.**

| | |
|---|---|
| Corpus | **97 documents · 33,105 parent-child chunks** (post KI-29 re-embed, hence 33,105 not 33,163) |
| torch | 2.12.0+cu130, `cuda_available=True` (RTX 4070, 12 GB) |
| Embedder / reranker | `bge-base-en-v1.5` / `BAAI/bge-reranker-base` |
| BM25 snapshot | present and hitting (`bm25_cache_hit chunks=33105`) |
| Python / OS | 3.12.3 / Windows 11 |

**Instrument.** Ad-hoc, not `scripts/profile_stages.py` (which has no memory mode — folding it in is
recorded as debt in `docs/performance.md`). Reproducible in a few lines:

- **Python heap:** `tracemalloc.start()`, snapshot `get_traced_memory()[0]` after importing
  `doc_assistant.pipeline`, then again after `RAGPipeline()` + `gc.collect()`. The delta is the
  corpus-linear part by construction: `Document` objects, their metadata dicts, the token lists and
  `BM25Okapi`'s dicts are all pure Python, while torch weights are native allocations that
  tracemalloc does not see. That separation is the reason for using it rather than RSS alone.
- **Process working set:** `K32GetProcessMemoryInfo` via `ctypes` (`GetCurrentProcess` needs
  `restype = c_void_p`, or the call silently fails and reports 0).
- **Timings** were taken in a *separate* run with tracemalloc off, since tracing distorts timing.

$0: no generation, no API call (constructing a chat model issues no request).

---

## 1. Memory — one sample, one box

| Point | Working set | Traced Python heap |
|---|---:|---:|
| interpreter start | 15.9 MB | 0 MB |
| after `import doc_assistant.pipeline` | 964 MB | 244 MB |
| **after `RAGPipeline()`, reranker not loaded** | **1,821 MB** | **510 MB** |
| after the first `rag.reranker` access | **2,327 MB** | 550 MB |

**The corpus-linear figure, which is the one that matters:**

> **265 MB of Python heap for 33,105 chunks ⇒ ~8.0 KB per chunk.**

Cross-checked against the on-disk snapshot of the same corpus: `data/bm25_index.pkl` is
**85.2 MB**, i.e. **2.6 KB/chunk** packed (text + metadata + tokens, stdlib types only). ~3x
inflation from packed bytes to live Python objects is what `str`/`dict`/`list` overhead predicts, so
the two measurements agree.

**Reading it honestly.**
- **One sample, one corpus, one box.** Bytes per chunk depends on chunk text length and how much
  metadata each carries (`parent_text` is denormalised into child metadata, which is why the figure
  is this large); a corpus of shorter documents would land elsewhere.
- The working-set column includes torch, CUDA context and tracemalloc's own overhead. Do not read
  1,821 MB as "the app needs 1.8 GB" on a CPU box or without tracing.
- The reranker adds **~506 MB** of working set but only 40 MB of traced heap: it is native weights,
  and it is a **constant** (it does not scale with the corpus).

**Projection at the 10,000-document contract** (linear, from ~340 chunks/document on this corpus, so
~3.4M chunks): **~27 GB of Python heap for the BM25 corpus alone.** Filed as **KI-32**. This is not
a measurement; it is a linear extrapolation from a single point, and it is the reason the cache in
ADR-035 does not settle the scale question.

## 2. The lazy-reranker penalty on `cu130`

Two fresh processes, no tracing:

| | run 1 | run 2 |
|---|---:|---:|
| imports | 12.08 s | 7.24 s ¹ |
| `RAGPipeline()` (reranker **not** loaded) | 5.97 s | 5.81 s |
| **first `rag.reranker` access (weight load)** | **5.03 s** | **4.84 s** |
| first `predict()` (CUDA warm-up) | 0.27 s | 0.19 s |
| second `predict()` (steady state) | 0.01 s | 0.02 s |
| **⇒ first-question penalty** | **~5.3 s** | **~5.0 s** |

¹ Import time swings with the OS file cache exactly as the 07-28/07-29 baselines warn. These two
runs are *not* a cold-launch measurement and must not be quoted as one; the launch figure of record
stays **12.10 s** from `stage_profile_2026-07-29.md`, measured by the profiler over fresh processes.

**The correction this produces.** The recorded first-question penalty was **~3.7 s** (CPU, 07-28).
On the CUDA wheel it is **~5.0-5.3 s**: loading the cross-encoder onto the GPU is *slower* than
loading it for CPU inference, while scoring is ~5x faster once warm. So the lazy-reranker trade did
not get better with the GPU, it got sharper on both sides: a shorter launch, a longer first answer,
and much faster answers after that. `docs/performance.md` carries this as the ledger row; the
07-28 baseline is append-only and is not edited.

## 3. KI-32 step 1, implemented and measured the same day: **1.36x, not the predicted 3x**

§1 reasoned that since a 400-character child carries a ~2,000-character parent and ~5 children share
each parent, deduplicating the parent text should remove ~70% of the footprint. It was implemented
(`_split_parent_texts` + `_parent_text_for`, payload v2) and measured on the same corpus:

| | before | after | |
|---|---:|---:|---|
| Python heap for the BM25 corpus | 265 MB | **195 MB** | **1.36x** (8,012 -> 5,892 B/chunk) |
| Snapshot on disk (`bm25_index.pkl`) | 85.2 MB | **39.9 MB** | **2.1x** |
| `RAGPipeline()` construction | 5.81 / 5.97 s | 5.62 / 5.82 s | **unchanged within noise** |
| Retrieval output | — | — | **identical** (3 queries, 10 sources each, same documents, same scores) |

Parents: **6,045 across 33,105 chunks = 5.5 children per parent**, so the duplication factor was as
expected. The *prediction* was wrong, not the mechanism: the text simply was not most of the memory.

**Where the heap actually goes** (tracemalloc `statistics("lineno")` at steady state, phase deltas
from a probe that rebuilds the arm step by step, 33,105 chunks):

| Component | Per chunk | Share |
|---|---:|---:|
| Chroma metadata dicts + their strings | ~1.4 KB | ~28% |
| BM25 token strings, retained by the index's per-document frequency dicts | ~1.4 KB | ~28% |
| `BM25Okapi.doc_freqs` (one dict per chunk) | ~0.8 KB | ~17% |
| `Document` (pydantic) object overhead, ~4 blocks each | ~0.6 KB | ~12% |
| **the child chunk text itself** | **~0.4 KB** | **~8%** |
| parent text (deduplicated by step 1; was ~2.4 KB/chunk before) | ~0.4 KB | ~8% |

Direct phase measurement of the dedup: **-80.0 MB** at the moment `_split_parent_texts` runs.

**What this means for step 2, and it is the useful half of the result.** The chunk text is 8% of the
problem; **the sparse index's own structures are ~45%** and generic Python object overhead on 33k
`Document`s and metadata dicts is another ~40%. So moving the index on-disk (SQLite FTS5, tantivy,
mmap) is not an incremental gain over step 1 — it removes the token strings, the frequency dicts and
the need to materialise `Document`s at all, which is ~85% of what remains. Step 1 was worth doing
(it is also the 2.1x on disk), but it does not move the ceiling much on its own.

**Revised, from the measured 5,892 B/chunk × ~340 chunks/document:**
**backend RAM ≈ 2 GB + ~2.0 MB per document** (was 2.7), so ~4 GB at 1,000 documents, a practical
wall near **5,000 on a 16 GB box** (was ~3,700), **~22 GB for the 10k contract**, and ~13,000
documents on a 32 GB box — the contract becomes reachable on generous hardware, not on typical.

## Reproduce

No committed runner yet (that is the debt item). The method above is complete: `tracemalloc` around
`RAGPipeline()` for the heap, `time.perf_counter()` around the property access in a separate
untraced process for the penalty, and for §3 a phase-by-phase rebuild of the BM25 arm with
`tracemalloc.take_snapshot().statistics("lineno")` at the end. Equivalence was checked in-process:
run the queries, then restore `parent_text` into `_bm25_docs` metadata from the map, empty the map,
and re-run — that reproduces the pre-change form on the same store, in the same process. Re-measure
per box and per corpus; every figure moves with device, corpus size and chunk-metadata size.
