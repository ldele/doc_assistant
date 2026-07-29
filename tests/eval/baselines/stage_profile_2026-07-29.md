<!-- status: active · updated: 2026-07-29 · class: append-only -->

# Stage profile — the same box on the GPU (2026-07-29)

**Question.** `stage_profile_2026-07-28.md` measured every transformer stage on **CPU torch**, on a
box with an idle RTX 4070. This re-records the *same* profile after
`uv sync --extra cu130 --extra dev`, so the question "which stages does the GPU actually fix?" is
answered with a measurement instead of an assumption.

**Read this file as a delta.** The method, the caveats and the full per-document tables live in the
2026-07-28 baseline and are **not** restated here. Same instrument, same flags, same corpus, same
sampled documents (`-r 3 --ingest --docs 8 --extract`), so the two are directly comparable.

**Box / corpus (the only line that changed).**

| | |
|---|---|
| Corpus | **97 documents · 33,163 parent-child chunks** (unchanged) |
| torch device | **`cuda` (NVIDIA GeForce RTX 4070, 12 GB), torch 2.12.0+cu130** ← *was `cpu`, torch 2.12.0+cpu* |
| Embedder / reranker | `bge-base-en-v1.5` / `BAAI/bge-reranker-base` (unchanged) |
| Retrieval | parent-child ON, `CANDIDATE_K=20`, `TOP_K=10` (unchanged) |
| Python / OS | 3.12.3 / Windows 11 (unchanged) |
| Repeats | 3 per stage; cold launch re-measured over **4** fresh processes (see §1) |

**No code changed for this.** Neither `get_embeddings()` nor the `CrossEncoder` construction pins a
device, so both auto-select CUDA once the CUDA wheel is installed. The speedup below is a wheel
swap, not an optimisation.

---

## 1. Startup — the GPU does *not* help here

| Stage | CPU (07-28) | **GPU (07-29)** | Verdict |
|---|---:|---:|---|
| Cold launch, fresh process (median) | **11.7 s** ¹ | **12.10 s** ² | **no gain — marginally worse** |
| open Chroma handle | 10.9 ms | 10.0 ms | unchanged |
| read all 33,163 chunks (paged, KI-27) | 1.13 s | 1.89 s ³ | unchanged (I/O) |
| build BM25 index over 33,163 chunks | 698 ms | 662 ms | unchanged (pure CPU) |

¹ post-lazy-reranker figure from §6 of the 07-28 baseline, 3 fresh processes.
² 4 fresh processes: 12.58 / 12.10 / 12.14 / 11.90 s. The profiler's own cold measurement takes a
single sample, so the extra three were run by hand — one sample was not enough to call a ~0.4 s
difference.
³ median of 1.09–3.59 s; the CPU run's range was 1.11–3.68 s. The medians differ, the ranges do not
— this is disk-cache noise, not a device effect. Do not read a regression into it.

**State the trade honestly: CUDA costs ~0.4 s at launch and buys nothing there.** The launch is
dominated by Python imports + the embedder weights, neither of which the GPU accelerates, and CUDA
context initialisation is new work. The lazy-reranker change (07-28) remains the only thing that has
actually moved launch time.

## 2. Query — this is where the GPU pays

| Stage | CPU (07-28) | **GPU (07-29)** | Speedup |
|---|---:|---:|---:|
| embed the question | 16.5 ms | **5.8 ms** | **2.8×** |
| vector search (k=20) | 179 ms | **135 ms** | 1.3× |
| BM25 search (k=20) | 28 ms | **26.7 ms** | 1.0× (pure CPU — expected) |
| ensemble (both arms + fusion) | 228 ms | **162 ms** | 1.4× |
| **`retrieve_with_scores` (retrieve + rerank + parent expand)** | **907 ms** | **296 ms** | **3.1×** |
| ⇒ **cross-encoder rerank + expand, by difference** | ~680 ms | **~134 ms** | **~5.1×** |

**The headline finding, and it changes a design conclusion.** The 07-28 baseline concluded *"the
reranker is three quarters of the retrieval budget"* and named it the one part worth optimising. On
the GPU it is **~45%** (134 of 296 ms) and the retrieval budget as a whole has dropped below the
~300 ms mark where a per-turn cost stops being worth engineering against. **The rerank is no longer
the bottleneck; nothing in the query path is.**

⚠ **A user-facing claim was wrong, and is now corrected.** `docs/setup.md` promised **~70 ms**
retrieve+rerank on an RTX 4070 (the 07-28 baseline mis-attributed this to the README and flagged it
as "a repo claim, not measured"). Measured on an RTX 4070 today, on this corpus: **296 ms** — the
claim was ~4× optimistic. `docs/setup.md:50` now carries the measured figure, the CPU comparison,
the corpus it was measured on, and a pointer here.

## 3. Ingest — the 8× that matters

| Stage | CPU (07-28) | **GPU (07-29)** | Speedup |
|---|---:|---:|---:|
| read cached markdown (mean of 8) | 0.3 ms | 0.4 ms | unchanged (a re-scan is still free) |
| chunk parent-child (mean of 8) | 9 ms | 4.7 ms | unchanged in kind (CPU, noise-dominated) |
| **embed rate** | **31.6 ms/chunk** | **3.8 ms/chunk** | **8.3×** |
| embed one document (mean of 8) | ~13 s | **1.36 s** | ~9× |
| **⇒ full re-embed of 33,163 chunks (projected)** | **~1012 s ≈ 17 min** | **128 s ≈ 2.1 min** | **7.9×** |
| extract PDF → markdown (COLD, mean of 8) | 15.2 s | **14.7 s** | **1.0× — no gain** |

Per-chunk embed rate is now flat across the size spread (**3.4–4.4 ms/chunk** over documents from
23k to 224k characters, vs 27.8–36.7 ms on CPU), so the 07-28 estimator still holds and gets
simpler:

> **~4.0 child chunks per 1,000 characters × ~3.8 ms/chunk ⇒ embed ≈ 15 ms per 1,000 characters**
> (was ~128 ms/1,000 chars on CPU).

**Extraction is untouched, and that is the important negative result.** Cold PDF→markdown is
**14.7 s per document** on the GPU against 15.2 s on CPU — within noise. It is the single most
expensive per-document cost in the system and it is **not** a GPU workload; the same document
(`2203.07436v4.pdf`, 36.8 s) is still the worst case. Any future work on ingest throughput has to
target extraction, not embedding.

## 4. What this changes downstream

| Decision | Was (CPU) | **Is now (GPU)** |
|---|---|---|
| **KI-29** — strip `<!-- page:N -->` at the *builder*, which forces a re-embed | ~17 min re-embed; the reason to prefer the retrieval-side patch | **~2 min re-embed** — the cost argument for the workaround is gone |
| Chunk size / overlap experiments | ~17 min per arm — effectively locked | ~2 min per arm — an eval-harness sweep is now practical |
| Embedding-model swap (`specter2`) | ~17 min per collection | ~2 min per collection |
| Rerank optimisation | "the one part worth optimizing" | **not worth it** — 134 ms of a 296 ms budget |
| BM25 persistence | 0.7 s at 33k chunks, scales with corpus | **unchanged** — still the only launch component that scales, still the right call at 10k docs |

## 5. Caveats that survive the device change

- **The first embedder load of a session still measures ~35 s** (35.14 s here, 30.7 s on CPU) against
  ~2.4 s warm. That is the OS file cache, not the device. Do not quote it as a load cost, and do not
  read the CPU→GPU difference in it as meaningful.
- **Every ingest figure is `--docs 8`**, the same size-spread sample as 07-28 (both extremes always
  included), not the full 97.
- **The re-embed figure is extrapolated** from the measured per-chunk rate, not run end to end —
  same method, same caveat as 07-28.
- **This box is now on the `cu130` wheel.** The frozen-sidecar build (`just build-sidecar`) expects a
  **CPU-synced** venv (KI-3 / justfile), so a release build needs `uv sync --extra cpu --extra dev`
  first and a re-sync back afterwards. Carried item, not a blocker today.

## 6. Reproduce

```bash
uv sync --extra cu130 --extra dev
uv run --no-sync python -m scripts.profile_stages -r 3 --ingest --docs 8 --extract
```

Numbers move with corpus size and torch device — both are printed in the profiler header. Re-record
a new dated file rather than editing this one.
