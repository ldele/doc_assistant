# BGE vs SPECTER2 — public corpus, symmetric (2026-06-04)

Re-checks the Phase 5 / Feature 3 embedder comparison (bge-base ahead of specter2,
originally measured on a separate corpus) on the **public verified-10 corpus**,
symmetrically at `--repeat 5` with the pinned judge.

**Setup**
- Corpus: identical for both arms — all 61 library docs, **27168 child chunks each**
  (specter2 was stale, missing exactly the 10 public papers; an incremental
  `EMBEDDING_MODEL=specter2 ingest --skip-cleanup` embedded just those 10, bringing
  both collections to the same 61 docs / 27168 chunks).
- Cases: `tests/eval/cases.public.yaml` (10 cases).
- Pipeline defaults (TOP_K=10, BM25 0.4 / vector 0.6, parent-child, bge-reranker-base).
  Only the **embedder** differs: `bge-base` (BAAI/bge-base-en-v1.5) vs `specter2`
  (allenai/specter2_base, adapter-less, as in the registry).
- Judge: `claude-haiku-4-5` (reference-only, temp 0). 5 trials. CPU torch (CPU dev box).

**Results (n=5, mean ± trial-mean std)**

| Scorer | bge-base | specter2 | Δ (bge − specter2) |
|---|---:|---:|---:|
| `citation_overlap` | **1.000** ± 0.000 | 0.900 ± 0.000 | +0.100 |
| `contains_all` | **0.927** ± 0.027 | 0.800 ± 0.031 | +0.127 |
| `llm_judge` | **3.738** ± 0.093 | 3.447 ± 0.090 | +0.291 |

**Reading**
- **bge-base scored higher on every scorer here**, by margins larger than the
  run-to-run (trial-mean std) bands. Small-N caveat applies (10 cases discriminate
  but are not statistically significant; read effect sizes, not a hard ranking).
- specter2 `citation_overlap` = 0.900 with **zero** trial-mean std: one of the 10
  cases deterministically fails retrieval under the adapter-less specter2 base
  (per-score std 0.303 = one case at 0 every trial). bge retrieves all 10 correctly
  every trial.
- The locked headline numbers (bge, 2026-06-01/2026-06-04) and this comparison agree:
  **bge-base stays the default embedder.**

**Run ids (`data/eval.duckdb`)**
- bge (n=5): e217759f, 6b4cbb1f, c8b534f8, 723eb410, ebc4a9a0
- specter2 (n=5): 5c39a703, 3d7ccc0b, b2d126ab, 4461503b, f506ad92
