<!-- status: active · updated: 2026-08-01 (ADR-038: the in-RAM arm is gone) · class: living -->

# Performance, cost and scale — the measured record

**What this file is.** One home for what the app *costs* to run: launch time, per-turn latency,
ingest throughput, memory, disk, and the trade each optimisation made to get there. It also carries
the knob inventory — what a user can change today, and what changing it would cost.

**What it is not.** It is not the *quality* record. Retrieval and answer quality live in
[`evals/README.md`](../evals/README.md) and are measured by a different instrument (the eval harness
over a fixed question set). The split is deliberate: one home per number, so nothing drifts between
two copies. Quality answers "is the answer right"; this file answers "what did it cost, and what
happens at 10x".

| Looking for | Go to |
|---|---|
| Retrieval / answer quality, scorers, the public benchmark | [`evals/README.md`](../evals/README.md) |
| Raw dated measurement runs (append-only, never edited) | [`tests/eval/baselines/`](../tests/eval/baselines/) |
| The enrichment layer's scale audit (hot paths, tuned constants) | [`REVIEW_2026-07-19_scale-robustness.md`](REVIEW_2026-07-19_scale-robustness.md) |
| Why a given design was chosen | [`decisions.md`](decisions.md) (ADR index) |
| Open weaknesses and measurement debt | `.claude/KNOWN_ISSUES.md` · `.claude/RIGOR_TODO.md` (local-only) |

**Three rules for every number below.**

1. **Every figure carries its corpus and its device.** The current corpus is **97 documents /
   33,105 parent-child chunks** on an **RTX 4070 (`cu130` wheel)**, Windows 11, Python 3.12.3.
   A CPU box is a different machine with different numbers, and so is a 1,000-document corpus.
2. **Quote ratios, not absolutes, for anything I/O-bound.** The same whole-store read measured
   **1.06 s** and **5.39 s** on the same unchanged store hours apart: the OS file cache dominates
   (ADR-035). Interleaved A/B ratios survive that; single absolutes do not.
3. **Projections are labelled as projections.** Everything past 97 documents in this file is a
   linear extrapolation from one corpus on one box. No scale run has ever been performed.

---

## 1. What it costs today

Sources: [`stage_profile_2026-07-28.md`](../tests/eval/baselines/stage_profile_2026-07-28.md) (CPU),
[`stage_profile_2026-07-29.md`](../tests/eval/baselines/stage_profile_2026-07-29.md) (same box, GPU),
[`memory_and_lazy_reranker_2026-07-30.md`](../tests/eval/baselines/memory_and_lazy_reranker_2026-07-30.md),
[`sparse_index_2026-07-30.md`](../tests/eval/baselines/sparse_index_2026-07-30.md) (the on-disk arm).

### Launch (paid once per backend start)

| | Cost | Scales with |
|---|---:|---|
| Python imports (torch, langchain, sentence-transformers) | ~6.8 s | nothing |
| Embedder weights | ~2.4 s | nothing |
| Chroma open | ~10 ms | nothing |
| Sparse arm: scan chunk ids + open the on-disk index | **~0.2 s + ~0** (ADR-036) | the id scan does |
| ⇒ what it replaced: whole-store read + tokenise + build | 1.67 s live · ~0.7 s from the snapshot ¹ | the corpus |
| Reranker weights | **not at launch any more** (lazy) | nothing |
| **Cold launch, fresh process** | **12.10 s** (GPU) · 11.7 s (CPU) · 16.1 s before the lazy reranker | now essentially all constants |

Pipeline construction alone (the part after imports) measured **4.53 s → 2.79 s** when the sparse
arm moved on disk.

¹ The snapshot figure is **derived**, not measured as one number: the chunk-id read that fingerprints
the cache (0.221 s) plus the cached load (~0.50 s), both measured in ADR-035's option table. What was
measured end to end is the **ratio**, cache off vs on, interleaved over 4 rounds: **2.7x**.

The first launch after a boot can cost roughly double: the first `get_embeddings()` of a session has
measured ~35 s against ~2.4 s warm. That is the file cache, not the code.

### Per turn (retrieval only; generation is provider-bound and measured separately)

| Stage | GPU | CPU |
|---|---:|---:|
| Embed the question | 5.8 ms | 16.5 ms |
| Vector search (k=20) | 135 ms | 179 ms |
| Sparse search (k=20) | **27.4 ms** on disk · 66.6 ms in RAM | 28 ms (in RAM) |
| Ensemble (both arms + fusion) | 162 ms | 228 ms |
| **`retrieve_with_scores` (retrieve + rerank + parent expand)** | **279 ms** on disk · 336 ms in RAM | 907 ms (in RAM) |
| ⇒ cross-encoder rerank + expand, by difference | ~134 ms (45%) | ~680 ms (75%) |
| **First question of a session pays, on top** | **~5.0-5.3 s** | ~3.7 s |

### Ingest (per document)

| Stage | Cost | Notes |
|---|---:|---|
| Re-scan an unchanged document | **0.4 ms** | content-cached + hash-deduped; a re-ingest is effectively free |
| Extract PDF → markdown, cold | **14.7 s mean** (5.7-36.7 s) | **not a GPU workload**; the dominant per-document cost |
| Chunk parent-child | ~5-8 ms | |
| Embed | **3.8 ms/chunk** (GPU) · 31.6 ms/chunk (CPU) | ~4.0 child chunks per 1,000 characters |
| ⇒ **embed ≈ 15 ms per 1,000 characters of text** | | file size in MB is the **wrong** predictor; characters is the right one |
| Full re-embed of the corpus | **~2.1 min** (GPU) · ~17 min (CPU) | forced by any change to stored chunk text |

### What each stage costs, as a share — the portable version

**Read this table for the shape, not the seconds.** Absolute times depend on the machine, the
document and whether OCR fires; the *ratios* between stages have held across every measurement so
far and are what should inform a design decision.

The reference unit is **one 30-page digital PDF** — a typical journal article or preprint, which is
what this corpus mostly holds (97 documents, 2,859 pages, ~30 pages each).

| Stage | Share of a first ingest | Order of magnitude | Scales with |
|---|---:|---|---|
| **PDF → markdown** | **~90%** | tens of seconds | **pages** (r=+0.73), not bytes (r=+0.37) |
| Embedding | ~5% | seconds | characters of text (~4 child chunks per 1,000 chars) |
| Figure detection | ~1% | sub-second | pages (renders each at `FIGURE_RENDER_DPI`) |
| Citation extraction | ~1% | sub-second | references per document |
| Epistemics · keywords · doc-vectors · gaps | <1% each | milliseconds | corpus for keywords/doc-vectors (see below) |
| Chunking | ~0% | milliseconds | characters |
| **Metadata** | **~0%** | milliseconds | nothing much — it is the cheapest stage by three orders of magnitude |
| *Figure VLM description* | *separate* | *seconds per figure, and money* | *figures; a paid pass, run on demand* |

**Four things this table is for.**

1. **Extraction is the only stage worth optimising.** Everything else combined is ~3 s against its
   ~34 s. A change that halves metadata cost saves nothing anyone can perceive; a change that
   halves extraction halves ingest.
2. **Estimate with pages, never megabytes.** Measured: a 15.4 MB / 20-page paper took 20.5 s while
   a 4.8 MB / 22-page one took 37.4 s. File size is close to useless as a predictor.
3. **OCR roughly doubles extraction.** A/B on this corpus: 1.4x–2.8x slower for +0% to +8% text —
   on one arXiv preprint, **11 extra seconds for 3 extra characters**. It fires far more than
   expected (87 of 97 documents, 1,128 of 2,859 pages) because `pymupdf4llm` finds a `tesseract`
   binary on PATH (KI-47).
4. **Two passes are corpus-global by construction** — `extract_keywords` and `compute_doc_vectors`
   recompute over the whole library, so adding one document to a 10,000-document corpus still pays
   a whole-corpus pass. That is the scaling problem, and it is not solved by making them faster.

**Indicative absolutes on the reference machine** (Windows, 28 logical cores, RTX + `cu130`, warm
model): a 30-page digital PDF costs **~15 s without OCR, ~35 s with**. A 300-page scanned book is
minutes, dominated entirely by OCR. A `.txt` or `.md` file is milliseconds — it skips extraction
altogether.

**A re-ingest is the worst case and should be rare.** An unchanged document re-scans in ~0.4 ms
because the extraction cache answers first; the per-format fingerprint (KI-48) exists so that an
unrelated change cannot invalidate that cache for every format at once.

#### Parallel extraction — measured, and more modest than it looks

Extraction is per-document and independent, so it is warmed in parallel before the serial loop
(`ingest.workers`). **16 documents, 28 logical cores, cold cache each round:**

| workers | wall | speedup |
|---:|---:|---:|
| 1 (serial) | 367.7 s | 1.00x |
| **2** | 250.1 s | **1.47x** |
| 4 | 218.8 s | 1.68x |
| 7 | 212.1 s | 1.73x |
| 14 | 211.1 s | 1.74x |

**Two workers buy 85% of the entire achievable gain, for 2 cores instead of 14** — which is why
the default budget is `light` and why the polite setting costs almost nothing. It does **not**
scale with cores, and the reason is not established: the long-tail hypothesis fails (the slowest
document here is 50 s, well under the 211 s floor) and so does OpenMP contention
(`OMP_THREAD_LIMIT=1` measured marginally *worse*, 225 s vs 217 s). Recorded as measured-but-
unexplained. Anyone projecting a large ingest should use **~1.5x**, not the core count.

### Enrichment sidecars (CPU-bound by nature; figures pre-date the GPU wheel)

| Runner | Whole corpus | Scoped to one document |
|---|---:|---:|
| `extract_doc_metadata` | 0.58 s | per-document by nature |
| `build_gaps` | 0.81 s | no flag (full recompute) |
| `compute_doc_vectors` | 2.88 s | filters the report only |
| `extract_keywords` | 4.14 s | **saves 4%** (corpus TF-IDF is global by construction) |
| `compute_epistemics` | 5.61 s | no flag (full recompute) |
| `extract_citations` | **54.1 s** | **7.4 s** (since KI-30) |

### Memory and disk

| | Measured at 97 documents |
|---|---:|
| **Python heap held by the sparse arm** | **21 MB — no corpus resident** (was 195 MB on the heap before ADR-036, 265 MB before KI-32 step 1) |
| Process working set after construction (reranker not loaded) | ~1.08 GB |
| Process working set, answering (reranker + CUDA loaded) | ~2.0 GB |
| Sparse index (`data/sparse_index.sqlite3`) | 41 MB |
| Parent-child vector store (`data/chroma_pc`, the retrieval default) | 416 MB |
| Baseline vector store (`data/chroma`) | 120 MB |
| Document store (`data/library.db`) | 7.4 MB |
| **⇒ on disk, all in** | **~584 MB ≈ 6.0 MB per document** |

**Both vector stores are always written.** `ingest` writes the baseline store *and* the parent-child
store for every document (`ingest/__init__.py:272`/`:300`), and only the parent-child one serves
retrieval by default. So disk cost carries a second copy of every embedding that the default answer
path never reads. Not a bug (the baseline store is what the flat-mode path and some sidecars use), but
it is 120 MB of the 584 MB here and ~20% of the projected disk at any size.

**Where the memory used to go, and why the answer mattered** (historical since 2026-07-30 — the
sparse arm no longer holds any of this, see §2). The first explanation was wrong and is worth
keeping as a warning: the obvious mechanism (every child chunk carries its parent's full text,
`ingest/chunking.py:194`, so ~5 copies of every parent) predicted that deduplicating it would remove
~70%. Implemented and measured, it removed **26%**. Attributing the heap by allocation site gave the
real picture, and that picture is what made "move the index off the heap" the only fix that reaches
the contract:

| Component | Per chunk | Share |
|---|---:|---:|
| Chroma metadata dicts + their strings | ~1.4 KB | ~28% |
| BM25 token strings, retained by the index's per-document frequency dicts | ~1.4 KB | ~28% |
| `BM25Okapi.doc_freqs`, one dict per chunk | ~0.8 KB | ~17% |
| `Document` (pydantic) object overhead | ~0.6 KB | ~12% |
| **the child chunk text itself** | **~0.4 KB** | **~8%** |
| parent text (deduplicated since 2026-07-30; was ~2.4 KB/chunk) | ~0.4 KB | ~8% |

**The text was 8% of the problem. The sparse index's own structures were ~45%, and generic Python
object overhead another ~40%** — which is why the on-disk index removed effectively all of it rather
than a slice. Method and per-site numbers:
[`memory_and_lazy_reranker_2026-07-30.md`](../tests/eval/baselines/memory_and_lazy_reranker_2026-07-30.md) §3.

---

## 2. The optimisation ledger — what each change bought and what it cost

Every row is a trade, not a free win. This is the section to read before "improving" any of them.

| Optimisation | Bought | Cost / trade | Shape at scale | Off switch |
|---|---|---|---|---|
| **Lazy reranker** (2026-07-28) | **4.4 s off every launch** | The **first question is ~5.0-5.3 s slower** (re-measured today on `cu130`; the recorded 3.7 s was a CPU figure — loading onto the GPU is slower, scoring is ~5x faster once warm). The +506 MB also arrives at first question, not at launch | Neutral: model weights are a constant | None. Hardcoded property; a knob candidate |
| **CUDA wheel** (`cu130`, 2026-07-29) | Query **3.1x**, embed **8.3x**, re-embed 17 min → 2.1 min | **Launch ~0.4 s worse** (CUDA context buys nothing there). Per-machine venv discipline: the frozen sidecar build needs a **CPU** sync first (KI-3), so **the shipped binary is CPU** and a tester sees the CPU column, not this one | Helps the per-turn constant and the embed slope. **Does nothing for extraction**, the dominant ingest cost | `uv sync --extra cpu` |
| **BM25 snapshot** (ADR-035, 2026-07-29) — **retired 2026-08-01, ADR-038** | Launch BM25 stage **2.7x** (interleaved A/B) | **40 MB on disk** (85 MB before KI-32 step 1) duplicating chunk text already in Chroma; the first launch after an ingest pays the write (~0.3 s); a `pickle` file in the data home | **Lowered the constant, kept the O(corpus) slope.** Deleted with the arm it cached | — (gone) |
| **Scoped-ensemble LRU** (4 entries, RH1) | Alternating between folders stops rebuilding BM25 every turn | Up to 4 subset indexes held in RAM. Scoped BM25 scores against **subset statistics**, so scoped and unscoped scores are not comparable (RG-020) | Each entry is subset-sized, so it inherits the same memory shape | None (structural constant) |
| **Extraction cache + hash dedupe** | A re-ingest of unchanged documents is **0.4 ms/document** | Disk for the markdown cache | **The good news of the ledger:** it is what makes re-running ingest over a large folder survivable | — |
| **Builder-side page-marker strip** (KI-29) | Correct stored text; 49% of parent texts were carrying markers into the LLM's evidence | **A full re-embed** (~2.1 min here). Existing installs keep markers until `ingest --rebuild` | Any change to *stored text* is an O(corpus) re-embed: ~3.6 h projected at 10k documents | — |
| **Shared `--doc` resolver** (KI-30) | `extract_citations` **54 s → 7.4 s** scoped | None | `extract_keywords` / `compute_doc_vectors` stay corpus-global **by construction**, so adding one document to a 10k corpus still triggers whole-corpus passes | — |
| **`RERANK_CANDIDATE_CAP`** (= `CANDIDATE_K*3`, RH1) | Bounds cross-encoder cost under multi-query | Drops the lowest-priority cross-variation tail. Provably inert on the single-query default path | Bounded by construction | `RERANK_CANDIDATE_CAP` env |
| **Deduplicated parent text** (KI-32 step 1, 2026-07-30) | Heap **265 → 195 MB (1.36x)**, snapshot **85 → 40 MB (2.1x)**, retrieval **identical** | A per-turn lookup instead of a carried string, and a second place the parent text can come from (metadata first, then the map). Snapshot payload v2, so an existing snapshot is rebuilt once | Cuts the per-document term ~26%: the wall moves ~3,700 → ~5,000 documents on 16 GB. **Does not change the shape** | None (the map *is* the corpus now) |
| **Sparse arm on disk** (ADR-036 / KI-32 step 2, 2026-07-30; sole arm since ADR-038) | Heap **195 → 21 MB (8.8x, no corpus resident)**, construction **4.53 → 2.79 s**, sparse query **66.6 → 27.4 ms**, per turn **336 → 279 ms** | **Retrieval changes**: FTS5's BM25 (k1=1.2, no IDF floor) is not `rank_bm25`'s. Re-measured on the private 35-case set (2026-08-01): **post-rerank recall identical**, pre-rerank +0.0147 to the on-disk arm — but **9 of 35 queries return a different evidence set**. Plus a 41 MB on-disk artifact | **Changes the shape.** Memory stops scaling with the corpus altogether; launch keeps an O(corpus) id scan | — (no fallback since ADR-038; a failed build is vector-only and reported) |

**The meta-trade worth naming.** The first four of these lowered a *constant* on the launch path.
That is real, and it also removed the symptom that would have made the *slope* visible: launch no
longer felt slow, so nothing prompted the question "what does this cost at 3.4M chunks?". Asking it
anyway is what produced KI-32 and, eventually, the only change in this table that altered the shape
rather than the constant. The projections in §3 exist so the question keeps being answered on paper
rather than by a user with 2,000 documents.

---

## 3. Scale: what the pass changed, and what it did not

**Linear projections** from 97 documents / 33,105 chunks (~340 chunks per document on this corpus's
mix of 20-40 page papers). **Not measured.** Superlinear paths are marked.

| | 97 docs (measured) | ~1,000 docs | ~10,000 docs (the contract) |
|---|---:|---:|---:|
| Chunks | 33,105 | ~340k | ~3.4M |
| **Backend RAM** | **~2.0 GB** | **~2.0 GB** | **~2.0 GB** (constants only, since ADR-036) |
| Sparse index on disk | 41 MB | ~0.4 GB | ~4.2 GB |
| Vector stores on disk (both) | 536 MB | ~5.5 GB | ~55 GB |
| Launch: scan chunk ids + open the index | ~0.2 s | ~2 s | ~20 s |
| First ingest: extraction (single-threaded) | ~24 min | ~4.1 h | **~41 h** |
| First ingest: embedding | ~2.1 min | ~22 min | ~3.6 h |
| Any stored-text change ⇒ full re-embed | ~2.1 min | ~22 min | ~3.6 h |
| `extract_citations`, whole corpus | 54 s | ~9 min | ≥1.5 h (superlinear in references) |
| Per-turn retrieval | 279 ms | vector search grows sublinearly (HNSW); sparse is an indexed lookup; rerank is constant | same |

**Memory is no longer what breaks first — that changed on 2026-07-30.** Every component of the
answer path was measured with a working-set probe, stage by stage, and none of them holds the
corpus: opening the sparse index costs **+1 MB**, a sparse query **+0 MB**, and a vector query
**+1 MB** (Chroma keeps its index on disk; an earlier probe that appeared to show +449 MB there was
measuring the CUDA context, which the same call initialises). What remains is constants — embedder
weights ~800 MB, CUDA context ~450 MB, reranker ~500 MB in use — so:

> **backend RAM ≈ 2 GB, flat**, where before ADR-036 it was 2 GB + ~2.0 MB per document

That is why the ceiling table that used to sit here is gone: the practical wall it described (~5,000
documents on a 16 GB machine) was a property of the in-RAM sparse arm, and the arm is on disk now.
**One corpus-linear cost survives at launch**, and it is time rather than memory: the fingerprint
scans every chunk id (~0.2 s at 33k, ~20 s projected at the contract), plus a ~159 MB working-set
high-water mark inside chromadb's paged read. The successor for both is an ingest-side version
stamp, so launch reads one row instead of every id. Not built; recorded in §6.

**How it was fixed, in two steps, and what each actually bought** (KI-32, roadmap PF2a/PF3):

1. **Deduplicate `parent_text` out of the in-RAM corpus** — predicted ~3x, **measured 1.36x** (265 →
   195 MB heap, 85 → 40 MB on disk), retrieval identical. Worth having, not the fix. **The lesson is
   in §1's table:** the prediction came from reasoning about text sizes, and text is 8% of the
   footprint.
2. **Move the sparse arm off the heap** ([ADR-036](decisions/ADR-036-sparse-index-on-disk.md)): an
   SQLite/FTS5 index beside the vector store, `Document` objects built for the returned rows only,
   folder scoping as a `WHERE` clause instead of a subset rebuild. **195 → 21 MB, no corpus
   resident**, and faster on every axis measured. The cost is that **retrieval changes** — FTS5's
   BM25 (k1=1.2, no IDF floor) is not `rank_bm25`'s — so it was gated on a quality A/B rather than
   asserted.
3. **Retire the in-RAM arm** ([ADR-038](decisions/ADR-038-retire-the-in-ram-sparse-arm.md),
   2026-08-01), once that A/B was repeated where the instrument could discriminate: on the private
   35-case set, **post-rerank recall is identical** (0.7010 @5 / 0.7598 @10) and the only movement —
   pre-rerank +0.0147 — favours the on-disk arm. There is now **one** keyword arm; a failed build
   degrades to vector-only and is reported rather than absorbed.

**A note on ADR-035, which ADR-038 supersedes outright.** Persisting the BM25 snapshot made the
launch of a design that could not reach 10k documents faster. It was the right call at the time (it
was where the slope was, the cache was correct and disableable, and it bought a measured 2.7x), but
it lowered a constant while the shape stayed — and it cached a structure that no longer exists.

**So what breaks first now?** On the evidence above, in order: **the first ingest** (~41 h of
single-threaded extraction at 10k documents), then **disk** (~60 GB), then the enrichment layer's own
hot paths (KI-18/KI-19). None of those is a redesign; all three are known work.

**On the first ingest, which is now the binding constraint.** ~41 h of extraction at 10k documents,
single-threaded, and extraction is the one stage a GPU does not touch. It needs parallelism, progress
and resumability before anyone points the app at a big folder. The enrichment layer's own hot paths
and corpus-tuned constants are catalogued separately in
[`REVIEW_2026-07-19_scale-robustness.md`](REVIEW_2026-07-19_scale-robustness.md) (KI-18/KI-19).

---

## 4. The knobs — what can be changed, and what it costs

**Output-neutral** means changing it cannot change what an answer says: it trades time, space or
money only. That distinction is what decides whether a knob is safe to expose (§5).

| Knob | Default | Trades | Changeable today | Output-neutral |
|---|---|---|---|---|
| torch extra (`cu130` / `cpu`) | per machine | ~3x query, ~8x embed | install time (`uv sync --extra …`) | yes (numerically near-identical, not bit-identical) |
| reranker laziness | lazy | 4.4 s launch vs ~5 s first answer | **not exposed** (hardcoded) | **yes** |
| `_SCOPED_ENSEMBLE_CACHE_SIZE` | 4 | folder-switch latency vs RAM | **not exposed** (structural constant) | **yes** |
| `MARKER_MAX_WORKERS` | 2 | table-extraction throughput vs CPU/RAM | env var | yes |
| `MAX_VLM_CALLS_PER_DOC` | 30 | figure descriptions vs **API spend** | env var | no (changes derived data) |
| `FIGURE_RENDER_DPI` | 150 | figure fidelity vs time/disk | env var | no |
| `TOP_K` | 10 | evidence breadth vs prompt cost | env + **per-turn sandbox** (ADR-010) | no |
| `USE_MULTI_QUERY` | off | recall vs ~4x rerank input + LLM calls | env + per-turn sandbox | no |
| `SYNTHESIS_MODE` | `ai` | `human` skips the interpretation LLM call | env + per-turn sandbox | no |
| `EPISTEMICS_MARKERS_ENABLED` | on | marker chips on the answer layer | env + **persisted setting** + per-turn | no |
| `REVIEWER_EVIDENCE_CHARS` | 1500 | reviewer accuracy vs judge tokens | env + per-turn sandbox (200-6000) | no |
| LLM provider / model | `.env`, else in-app | quality vs cost vs offline | **in-app Settings** (persisted) | no |
| `CANDIDATE_K` | 20 | recall headroom vs rerank cost | env only, at construction | no — **locked** (verdict unvalidated) |
| `BM25_WEIGHT` | 0.4 | pre-rerank ordering only (measured inert post-rerank) | env only | no — **locked** |
| Chunk sizes (`2000/200`, `400/50`) | locked | precision vs context; **forces a re-embed** | env only | no — **locked** |
| `EMBEDDING_MODEL` | `bge-base` | quality; **new collection + full re-embed** | env only | no — **locked** |

"Locked" means the project rule in `.claude/CONTEXT.md`: change only via an eval-harness experiment
that beats the control beyond its variance.

## 5. Should the cost knobs be user-facing? **Decided 2026-07-30 — no** (ADR-037)

The question was filed as PF2 while the sparse arm still held the corpus in RAM. **ADR-036 dissolved
most of its premise**: of the knobs §4 proposed exposing, the BM25 snapshot became legacy-only, the
scoped-cache size lost the cost it traded against, the ingest knobs turned out to be read only by
`scripts/` (CLI runners with their own flags), and the one new switch — `DOC_SPARSE_INDEX` — was
**not** output-neutral, which was the test that made a knob safe to expose. What remained was a
single minor toggle, while the need behind the request (*will this hold a big library?*) had been
answered in engineering. (ADR-038 has since deleted both env switches outright, so the table above
no longer lists them: the question is now moot rather than merely decided.)

**So the app ships the answer, not the controls**
([ADR-037](decisions/ADR-037-corpus-facts-not-performance-knobs.md)). Settings → **Corpus** reports
documents, chunks, disk total and per document, which keyword-index implementation is serving, its
size and when it was built, and one sentence about memory — plus one bounded action, *Rebuild*, for
the keyword index only. Three sub-decisions worth carrying:

- **No performance presets, and no restart semantics.** Every knob in §4 is decided at pipeline
  construction and there is no live rebuild path, so exposing one would mean either building that
  path or asking users to restart for a second of launch time.
- **The rebuild button is the index, never the corpus.** A keyword-index rebuild is derived data the
  next launch would regenerate anyway (2.8 s here, minutes at 10k documents). A *full* re-index is
  hours at that size with no progress or resumability — a button for it would be the worst reading
  of inform-don't-block.
- **The memory line states the shape, never a live number.** "Memory does not grow with your
  library" is the decision-relevant fact and is now true by construction; a live RSS figure needs a
  new dependency, fluctuates, and is dominated by model weights a user would misread as their
  corpus's cost.

**The knobs in §4 stay exactly where they are:** environment variables, documented here, a developer
rollback surface. ADR-010's per-turn sandbox still owns the *quality* knobs, and the eval-locked
settings stay locked.

## 6. Measurement debt

- **Launch was never re-measured end to end after the BM25 snapshot landed**, nor after the on-disk
  arm replaced it. Construction alone is measured (4.53 → 2.79 s); the whole-launch figure of record
  is still 12.10 s from before both.
- **Answer-level equivalence of the two sparse arms was never measured** (the A/B closed the
  retrieval question — post-rerank recall identical on the private 35-case set, 2026-08-01 — but
  **9 of 35 queries returned a different evidence set**, and recall@k does not score the other slots
  the LLM reads). Closing it means `contains_all`/`llm_judge` over those 35 cases: paid, and their
  references are `author_verified: false` in places.
- **Retrieval is not deterministic** — a same-arm re-run disagreed on 1 of 35 cases (~3% case-level
  noise floor), traced to the cross-encoder breaking ties on a case whose target document has 0
  chunks. `sweep_bm25_weight`'s docstring and the ADR-036 baseline both still assert determinism.
- **The launch id scan is still O(corpus)** (~0.2 s at 33k, ~20 s projected at 10k documents), plus
  ~159 MB of working-set high-water inside chromadb's paged read. The fix is an ingest-side version
  stamp; not built.
- **No committed memory runner.** Today's figures came from an ad-hoc `tracemalloc` + working-set
  probe (method recorded in the 2026-07-30 baseline). It belongs in `scripts/profile_stages.py`.
- **No scale run has ever happened.** Every number past 97 documents in this file is arithmetic. A
  synthetic ≥1,000-document corpus is the only way to test the linear assumption (RG-016 already asks
  for one).
- **Sidecar figures are pre-GPU** (CPU-bound by nature, so unlikely to move, but not re-measured).
- **Extraction parallelism is unmeasured** — the ~41 h projection assumes the current single-threaded
  path.
- `CANDIDATE_K=20` remains an unvalidated verdict (`.claude/CONTEXT.md` open questions).

## 7. Reproduce

```bash
uv run --no-sync python -m scripts.profile_stages -r 3 --ingest --docs 8 --extract
uv run --no-sync python -m scripts.profile_stages -r 1 --sidecars --sidecar-timeout 900
```

The profiler prints corpus size and torch device in its header; both belong in any quoted figure.
Re-record a new dated file in `tests/eval/baselines/` rather than editing an existing one, then
update the tables here.
