# Chunking sweep — public corpus (2026-06-06)

Measures the parent/child chunk-size grid through the **real** pipeline to decide
whether the current locked defaults (`parent 2000/200 · child 400/50`) can be beaten.
Closes the long-standing "defaults never measured" caveat. Run on the **RTX/GPU box**
(`scripts/sweep_chunking.py`), the slow re-embed-per-config step done on CUDA.

**Setup**
- Corpus: `data/sources` = the public verified-10 arXiv papers (`corpus_manifest.yaml`).
- Cases: `tests/eval/cases.public.yaml` (10 cases).
- Each config: `ingest --rebuild` under that config's `*_CHUNK_SIZE` env vars, then
  `run_eval --repeat 3 --with-llm-judge` (3 trials), tagged with the config note.
- Pipeline otherwise at locked defaults (TOP_K=10, BM25 0.4 / vector 0.6, parent-child,
  bge-base embedder, bge-reranker-base).
- Judge: `claude-haiku-4-5` (reference-only, temp 0). torch `2.12.0+cu130` (RTX 4070, GPU).

**Results (n=3, mean ± trial-mean std)** — grid order; config 1 is the control (current default).

| # | parent/child | `citation_overlap` | `contains_all` | `llm_judge` | judge n_skipped |
|---|---|---:|---:|---:|---:|
| **1 — control** | 2000/200 · 400/50 | 1.000 ± 0.000 | **0.933** ± 0.046 | **3.951** ± 0.140 | 3 |
| 2 | 2000/200 · 256/32 | 1.000 ± 0.000 | **0.933** ± 0.014 | 3.911 ± 0.084 | 0 |
| 3 | 2000/200 · 600/75 | 1.000 ± 0.000 | 0.906 ± 0.013 | 3.917 ± 0.260 | 2 |
| 4 | 1500/150 · 400/50 | 1.000 ± 0.000 | 0.925 ± 0.014 | 3.815 ± 0.196 | 3 |
| 5 | 3000/300 · 400/50 | 1.000 ± 0.000 | 0.914 ± 0.005 | 3.793 ± 0.125 | 1 |
| 6 | 1000/100 · 256/32 | 1.000 ± 0.000 | 0.919 ± 0.019 | 3.920 ± 0.094 | 1 |

**Reading**
- **Defaults confirmed — no config beats the control `2000/200 · 400/50`.** It is tied-best
  on `contains_all` (0.933) and best on `llm_judge` (3.951).
- `citation_overlap` saturates at **1.000** for every config (retrieval cites the right paper
  in all 10 cases, every trial) — it cannot discriminate chunk sizes. The discriminators are
  `contains_all` and `llm_judge`, and the spread across configs (~0.03 on contains_all, ~0.04
  on judge between the control and the pack) is within the trial-to-trial noise bands. So this
  is "no reason to change the defaults", not "the control is decisively superior".
- **Larger parent hurts slightly:** config 5 (3000/300) is the *worst* on judge (3.793) and
  below the control on `contains_all`.
- **Closest alternative:** config 2 (smaller `256/32` child) matches the control on
  `contains_all` (0.933) with tighter variance (±0.014) and 0 skipped judge calls — the most
  defensible alternative, but it does not exceed the control on quality, so the default stands.
- Small-N caveat: 10 cases give effect sizes, not statistical significance — read trends, not
  a hard ranking. The flaky `sbert_motivation` judge call (timeout / JSON-parse) accounts for
  most of the skipped-judge counts, as on the headline public eval.

**Decision:** keep the locked defaults `parent 2000/200 · child 400/50`. The CLAUDE.md
Locked-settings chunk-sizes row is updated from "defaults never measured" to measured/confirmed.

**Provenance note:** configs 1–4 ran in one background sweep; that process was terminated
mid-config-5 (no Windows crash/Event-ID-1000 signature — a transient external/OOM kill, not a
config-specific failure). Configs 5–6 were re-run identically on the same box/session/env; the
re-run reproduced config 5 past the prior stop point with no issue. All six configs are one
machine, one torch build, one judge model.

**Run ids (`data/eval.duckdb`, 3 trials each)**
- `2000/200 · 400/50` (control): d5a337a6, 7f1ebdbc, 6a969961
- `2000/200 · 256/32`: b27adab0, 8f689796, ddab6d54
- `2000/200 · 600/75`: 883d9ffe, 3ce8afbc, ed27a69b
- `1500/150 · 400/50`: 89ea2c64, b095e3d8, 48533c79
- `3000/300 · 400/50`: ea296de3, 80af0554, c9fd9a11
- `1000/100 · 256/32`: 2d15d156, 15f10b79, 7e688a34
