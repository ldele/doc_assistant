# CANDIDATE_K split — public corpus A/B (2026-06-13)

Measures the 2026-06-07 retrieval-K split (`CANDIDATE_K`, commit `09115c8`) through the
**real** pipeline on the public corpus, to decide whether the shipped default
`CANDIDATE_K=20` should stay or revert to the pre-split `CANDIDATE_K=10`. Closes the
*public* half of the "provisional / not locked" caveat; the private (neuroscience)
arm is still pending (see Reading).

**The knob.** `CANDIDATE_K` = candidates fetched from *each* retriever (vector + BM25)
before reranking; `TOP_K` (=10) = the final post-rerank cut passed to the LLM. Pre-split
the pool was hardcoded to `10 == TOP_K`, giving the cross-encoder no headroom to reorder.
`CANDIDATE_K=10` reproduces the exact pre-split behaviour. Changing `CANDIDATE_K` does
**not** invalidate the embedding cache (query-time only) — no re-ingest between arms.

**Setup**
- Corpus: `data/sources` = the public verified-10 arXiv papers (`corpus_manifest.yaml`), ingested (PC store: 10 docs / 2349 child chunks).
- Cases: `tests/eval/cases.public.yaml` (10 cases).
- Pipeline at locked defaults (`TOP_K=10`, BM25 0.4 / vector 0.6, parent-child, bge-base, bge-reranker-base).
- Judge: `claude-haiku-4-5-20251001` (reference-only, temp 0). `--repeat 3`, mean ± trial-mean std.
- **CPU box** (no GPU needed — `CANDIDATE_K` is query-time retrieval depth, not embedding).

**Results (n=3, mean ± trial-mean std)**

| Scorer | A — `CANDIDATE_K=10` (control) | B — `CANDIDATE_K=20` (default) | Δ (B − A) |
|---|---:|---:|---:|
| `citation_overlap` (0-1) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 |
| `contains_all` (0-1) | 0.931 ± 0.019 | 0.933 ± 0.014 | +0.002 |
| `llm_judge` (1-5) | 3.833 ± 0.173 | 3.736 ± 0.193 | −0.097 |
| judge n_scored | 28 / 30 | 29 / 30 | — |

**Reading**
- **Statistical tie — `CANDIDATE_K=20` does not regress on the public corpus.** No scorer
  moves beyond its trial-mean std: `contains_all` is identical (+0.002), and the `llm_judge`
  gap (−0.097) is smaller than either arm's std (±0.17–0.19) and is partly an artifact of the
  flaky `sbert_motivation` judge call (skipped 2/3 in A, 1/3 in B — a single scored/skipped
  case shifts the mean by more than 0.097).
- **`citation_overlap` saturates at 1.000** for both arms — the public set is one-paper-per-topic,
  so retrieval cites the right paper in all 10 cases regardless of pool depth. This metric cannot
  discriminate pool size here, and a wider pool *structurally cannot* surface additional relevant
  papers on this corpus.
- **Validity check:** arm A (`CANDIDATE_K=10`) reproduces the locked pre-split public baseline
  (`public_eval_baseline_2026-06-01.md`: citation 1.000, contains_all 0.927–0.933, judge 3.74–3.89).

**⚠ Low power — verdict is UNVALIDATED, retest needed.** 10 cases over a 10-doc,
one-paper-per-topic corpus is too small to reliably rank `CANDIDATE_K`. A "tie" here means
*this corpus can't tell the two settings apart*, not that 20 is proven equal-or-better.
**Retest on a larger, multi-paper corpus** (private neuroscience `cases.yaml` and/or an
expanded public set) before drawing any firm conclusion or locking the value.

**Decision (provisional):** **keep `CANDIDATE_K=20`** as the default. It is *safe* (no regression on the public
corpus) and architecturally motivated (a wider pool gives the reranker headroom to reorder —
standard practice). It is **not yet a demonstrated win**: the cross-paper "crowding" benefit it
targets requires a **multi-paper-per-topic** corpus, which the public set is not.

**Still pending (RTX / private box):** re-run this A/B on the private neuroscience corpus
(`cases.yaml`, multi-paper-per-topic) before calling `CANDIDATE_K=20` a measured win rather than a
no-regression default. `CANDIDATE_K=10` reproduces the pre-split behaviour exactly if a revert is
ever wanted.

**Run ids (`data/eval.duckdb`, 3 trials each)**
- `CANDIDATE_K=10` (control): b2405e72, b48d5b85, e8b9f938
- `CANDIDATE_K=20` (treatment / default): c00871be, 95098e56, 90e2120f
