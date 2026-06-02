# Public demo eval — reference baseline (2026-06-01)

Committed, human-readable reference for the public corpus eval. The live run
log (`data/eval.duckdb`) is gitignored and rewritten on every run; this file is
the stable thing to diff a fresh run against.

**Setup**
- Corpus: 10 arXiv papers behind the project's methods (`tests/eval/corpus_manifest.yaml`, download-only via `scripts.download_corpus`).
- Cases: `tests/eval/cases.public.yaml` (10 cases, strict substrings, abstract-grounded references).
- Embedder: `bge-base`. Pipeline defaults (TOP_K=10, BM25 0.4 / vector 0.6, parent-child, bge-reranker-base).
- Judge: `claude-haiku-4-5` (reference-only, temp 0).
- 5 trials (`--repeat 5`); reported as mean ± trial-mean std.

**Results (n=5)**

| Scorer | Mean | Trial-mean std | n_scored |
|---|---:|---:|---:|
| `citation_overlap` (0-1) | 1.000 | 0.000 | 50 / 50 |
| `contains_all` (0-1) | 0.927 | 0.034 | 50 / 50 |
| `llm_judge` (1-5) | 3.894 | 0.075 | 47 / 50 |

Deterministic-only n=5 (no judge) agrees: `citation_overlap` 1.000 ± 0.000,
`contains_all` 0.927 ± 0.014.

**Reading**
- `citation_overlap` = 1.000 ± 0.000: retrieval cites the correct paper for all 10 cases, every trial. Depends only on retrieval, so it has no run-to-run variance.
- `contains_all` = 0.927: scores the stochastic generated answer, so individual runs range ~0.88–0.98; the mean is stable and the std is small.
- `llm_judge` = 3.894/5: answers are genuinely good; the `contains_all` shortfall is wording, not correctness. The score is high because these references are abstract-grounded (not best-effort), which the reference-only judge credits — not a sign the pipeline is better on these papers than others.

**Known caveat**
- Flaky judge call on `sbert_motivation`: skipped in 3 of 5 trials (API timeout / JSON parse failure on that prompt). `llm_judge` mean is over 47 of 50 scores. Re-run to confirm; investigate if it persists.

**Run ids (data/eval.duckdb):**
- With judge (n=5): 06bacac7, 2e1d3194, 4f5368c4, b1bf7a76, 480090e5
- Deterministic (n=5): 02e31c7c, bf1676d2, 0a99e974, 89300c11, 47238288

Cases are deliberately strict — not tuned to score 1.0.
