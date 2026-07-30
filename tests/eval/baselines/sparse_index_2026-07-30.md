<!-- status: active · updated: 2026-07-30 · class: append-only -->

# Sparse arm A/B — in-RAM BM25 vs on-disk SQLite/FTS5 (2026-07-30, ADR-036 / KI-32 step 2)

**Question.** Moving the keyword arm off the Python heap removes the memory ceiling KI-32 recorded,
but FTS5 ranks with a different BM25 formula (k1=1.2, no IDF floor) than `rank_bm25.BM25Okapi`
(k1=1.5, epsilon=0.25). So: **what does it cost, and what does it change?** Both halves measured,
because a memory win that quietly degrades answers is not a win.

**Box / corpus.**

| | |
|---|---|
| Corpus | **97 documents · 33,105 parent-child chunks** |
| torch | 2.12.0+cu130 (RTX 4070) |
| Retrieval | parent-child ON, `CANDIDATE_K=20`, `TOP_K=10`, `BM25_WEIGHT=0.4`, multi-query OFF |
| Cases | `tests/eval/cases.public.yaml` (the 10 public arXiv questions) |
| Python / OS | 3.12.3 / Windows 11 |

**Method.** Two separate processes, one per arm (`DOC_SPARSE_INDEX=0` selects the control) — a
single process would report the sum of both arms' memory and prove nothing about either. Retrieval
only, **$0**, no generation. The recall instrument is the one
`scripts/sweep_bm25_weight.py` already uses (`_recall_at_k` over each case's `expected_citations`,
bidirectional substring), so this is the project's existing measurement pointed at a new variable.
Retrieval is deterministic on both arms, which is what `--repeat` buys elsewhere.

---

## 1. Cost — the reason for the change

| | in-RAM (control) | on-disk (shipped) | |
|---|---:|---:|---|
| Python heap held after construction | 185.2 MB | **21.1 MB** | **8.8×** |
| Corpus resident | 33,105 chunks | **0** | — |
| Pipeline construction | 4.53 s | **2.79 s** | 1.6× |
| Sparse arm, mean per query | 66.6 ms | **27.4 ms** | 2.4× |
| `retrieve_with_scores`, mean per turn | 336.3 ms | **279.4 ms** | 1.2× |
| On-disk artifact | 39.9 MB (`bm25_index.pkl`) | 40.7 MB (`sparse_index.sqlite3`) | ~equal |

**Where the remaining process memory goes** (working-set probe, stage by stage, sparse arm active):

| Stage | Working set | Delta |
|---|---:|---:|
| imports (torch, langchain, sentence-transformers) | 87 MB | +74 MB |
| **embedder weights** | 887 MB | **+800 MB** |
| open Chroma handle | 906 MB | +18 MB |
| read 33,105 chunk ids (the fingerprint) | 1,065 MB | **+159 MB** |
| open the sparse index | 1,066 MB | **+1 MB** |
| 3 sparse queries | 1,066 MB | **+0 MB** |
| **first `embed_query` (CUDA context)** | 1,523 MB | **+457 MB** |
| first vector query (Chroma's index) | 1,524 MB | **+1 MB** |
| 3 more vector queries | 1,524 MB | +0 MB |
| reranker weights | 1,552 MB | +28 MB (≈500 MB once actually scoring) |

**Two findings in that table, and both were surprises.**

1. **Chroma is not corpus-resident.** A vector query adds ~1 MB, not hundreds. An earlier probe
   attributed +449 MB to "the first vector query" and was **wrong**: that call also initialises the
   CUDA context. Isolating the embed step moved the whole 449 MB to CUDA and left the vector search
   at 1 MB. Nothing about the vector store scales into the process.
2. **The only corpus-linear cost left at launch is the fingerprint's id scan** (+159 MB of
   working-set high-water at 33k chunks, mostly chromadb's paged `get()` machinery rather than our
   own list). The Python-side share was measured separately and fixed: collecting the ids peaked at
   **3.1 MB (94 B/chunk ⇒ ~0.3 GB at 3.4M chunks)**; hashing them **streaming, page by page**, peaks
   at **0.9 MB** and produces the identical digest. The chromadb share remains and needs an
   ingest-side version stamp to remove.

## 2. Quality — what actually changes for the user

**Recall against the public cases' expected citations, identical on both arms:**

| | in-RAM | on-disk |
|---|---:|---:|
| pre-rerank recall@5 | 1.0000 | 1.0000 |
| pre-rerank recall@10 | 1.0000 | 1.0000 |
| post-rerank recall@5 | 1.0000 | 1.0000 |
| post-rerank recall@10 | 1.0000 | 1.0000 |

**Final top-10 (post-rerank, what the user receives), per query:**

| Query | Final top-10 | Sparse-arm agreement |
|---|---|---:|
| `rag_two_formulations` | differs by 1 of 10 | 35% |
| `dpr_vs_bm25` | **identical** | 20% |
| `sbert_motivation` | **identical** | 40% |
| `cpack_resources` | differs by 1 of 10 | 5% |
| `scirepeval_multi_embedding` | **identical** | 20% |
| `bert_passage_reranking` | **identical** | 25% |
| `colbert_late_interaction` | **identical** | 10% |
| `hyde_zero_shot` | **identical** | 20% |
| `llm_judge_biases` | **identical** | 35% |
| `ai_usage_cards_dimensions` | **identical** | 5% |

**8 of 10 byte-identical, order included; 2 differ by one document.** The arms themselves disagree
far more than that (84% of top-20 candidates overlap in an earlier isolated comparison; the
file-level overlap above is lower because a query's 20 sparse candidates cluster into few files).
The cross-encoder re-scoring the union is what absorbs the difference — which also means this result
depends on the reranker staying in the pipeline.

**⚠ The ceiling in this instrument.** Both arms score **1.0000 everywhere**, so the public 10-case
set cannot detect a small regression: it proves no regression *it can see*. The private 35-case set
(`tests/eval/cases.yaml`) that could discriminate is not on this machine. Treat this as *parity
demonstrated at the resolution available*, not as *quality proven equal* — and keep
`DOC_SPARSE_INDEX=0` until the A/B is repeated where the instrument has room to move.

## 3. Live end-to-end check ($0, Ollama `llama3.1:8b`)

One real turn through `ChatController` on the shipped arm: **10 sources**, top source
`dpr_karpukhin_2020.pdf` at **0.9795** for a DPR question, reranker scores **sigmoid-bounded in
[0, 1]** (the integrity layer depends on that), no page markers in the evidence, citation note
clean, `_bm25_docs` empty throughout.

## Reproduce

```bash
uv run --no-sync python -m scripts.sweep_bm25_weight --cases tests/eval/cases.public.yaml --grid 0.4
DOC_SPARSE_INDEX=0 uv run --no-sync python -m scripts.sweep_bm25_weight --cases tests/eval/cases.public.yaml --grid 0.4
```

That reproduces the recall half with the committed instrument (the memory and per-arm timings came
from an ad-hoc probe — `tracemalloc` plus `K32GetProcessMemoryInfo` around construction — which is
the same measurement debt `memory_and_lazy_reranker_2026-07-30.md` records). Delete
`data/sparse_index.sqlite3` to force a rebuild. Numbers move with corpus size and device; re-record
a new dated file rather than editing this one.
