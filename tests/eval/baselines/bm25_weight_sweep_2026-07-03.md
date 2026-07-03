<!-- status: active · updated: 2026-07-03 · class: baseline -->

# BM25/vector ensemble-weight sweep — retrieval eval, 2026-07-03

The hybrid-retrieval split `BM25 0.4 / vector 0.6` is a **locked setting** that was never
measured ("vibes-locked" — CONTEXT Open questions / locked-settings table). R6 un-handicapped the
BM25 arm (`preprocess_func=keywords.tokenize`; `tests/eval/baselines/bm25_preprocess_2026-07-02.md`),
which was the stated prerequisite for sweeping the weight. This is that sweep. Eval-gated
(rigor-gate): change the default **only** if a weight beats the control beyond its variance.

**Verdict: KEEP 0.4/0.6.** Negative result on the shipped metric — no weight in `[0.0, 1.0]` beats
(or differs from) the control on post-rerank recall. The reason is structural, and the instrument is
shown to discriminate (below), so this is a real null, not a dead measurement.

## Method

- **Instrument:** retrieval-only **recall@K** over the 35-case benchmark (`tests/eval/cases.yaml`,
  34 with `expected_citations` → scored), retrieved filenames vs `expected_citations` by
  **bidirectional substring** (`hodgkin_huxley_1952` matches `hodgkin_huxley_1952.pdf`) — the same
  instrument as the R6 baseline. **$0**: `USE_MULTI_QUERY=false`, so `retrieve()` makes **no** LLM
  call and the whole sweep is fully offline (`HF_HUB_OFFLINE=1`). Driver:
  `scripts/sweep_bm25_weight.py`.
- **Two metrics per weight** (this is what makes the null *explanatory*):
  - **post-rerank** recall@K — the **shipped** path, `pipeline.retrieve(top_k=k)` (cross-encoder
    rerank → parent-dedup → top-k). This is the metric the gate is on.
  - **pre-rerank** recall@K — recall over the ensemble's fused candidate order *before* the
    cross-encoder (`ensemble.invoke`). This is where the weight actually acts.
- **Weight knob:** `BM25_WEIGHT` (config; `--bm25-weight` on `scripts/run_eval.py`). It is the
  BM25-arm ensemble weight; the vector arm takes the complement `1 - w`
  (`pipeline.resolve_ensemble_weights`). Grid `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`, control = `0.4`.
- **Corpus / config:** the current `data/` store (76 docs; PC mode `USE_PARENT_CHILD=true`),
  `CANDIDATE_K=20`, `TOP_K=10`, `bge-base` embedder + `bge-reranker-base`. The pipeline is loaded
  **once**; only the `EnsembleRetriever` is rebuilt per weight (a weight change is retrieval-time —
  no re-embed, no store mutation). Retrieval is **deterministic → ~zero variance**, so one pass per
  arm is representative; the `--repeat` intent is satisfied by determinism (`--repeat N` re-runs and
  asserts agreement).
- **Scope note:** as with R6, the public corpus (`cases.public.yaml`) is download-only from arXiv and
  absent on this box; `cases.yaml` (the private benchmark) aligns with the current store and is the
  available instrument.

## Result

| BM25 weight | pre@5 | post@5 | pre@10 | post@10 |
|---|---:|---:|---:|---:|
| 0.0 (vector-only) | 0.9363 | 0.8775 | 0.9363 | 0.9069 |
| 0.2 | 0.9363 | 0.8775 | 0.9363 | 0.9069 |
| **0.4 (control)** | **0.9363** | **0.8775** | **0.9363** | **0.9069** |
| 0.6 | 0.8824 | 0.8775 | 0.9363 | 0.9069 |
| 0.8 | 0.8824 | 0.8775 | 0.9363 | 0.9069 |
| 1.0 (BM25-only weight) | 0.8824 | 0.8775 | 0.9363 | 0.9069 |

- **post-rerank is FLAT across the entire weight range** — `post@5 = 0.8775`, `post@10 = 0.9069` for
  every weight, **identical to the R6 baseline** (recall@5 0.8775 / @10 0.9069), which cross-validates
  the instrument. No weight beats the control; nothing to change.
- **pre-rerank recall@5 MOVES**: `0.9363` at `w ≤ 0.4` → `0.8824` at `w ≥ 0.6`. The weight *does*
  reorder the candidate pool — it just doesn't survive reranking.

## Interpretation (honest)

**Why post-rerank cannot move (structural).** LangChain's `EnsembleRetriever.weighted_reciprocal_rank`
returns the **full union** of both arms' `CANDIDATE_K=20` docs (it re-orders by weighted RRF but never
truncates, and even a zero-weighted arm still contributes its docs to the union). `retrieve()` then
feeds that **entire union** to the cross-encoder, which re-scores every candidate and takes the top-k
by *reranker* score. So the ensemble weight only permutes the pre-rerank list, and the reranker
discards that order. The candidate **set** is weight-independent → the reranked top-k is
weight-independent. This is not "the benchmark under-exercises BM25" (R6's framing); on the current
pipeline the weight is **inert on the final output by construction**.

**The instrument discriminates (rigor / planted-signal check).** A flat post-rerank table is only
trustworthy if the measurement *can* move. It can: `pre@5` shifts by ~0.054 across the grid. So the
harness detects the ranking change the weight produces, and the flat post-rerank is a genuine
reranker-dominance result, not a broken or constant scorer.

**Directional read (not a gate).** The control (0.4) sits in the *better* pre-rerank regime — the
vector-leaning side (`w ≤ 0.4`) gives the higher candidate-pool recall@5 (0.9363 vs 0.8824) on these
natural-language queries, and 0.4 is the boundary of it. So `0.4/0.6` is, if anything, on the correct
side; pushing BM25 higher only *degrades* the candidate pool (before the reranker erases the
difference). This is a weak, single-corpus signal — it justifies keeping the split, not lowering it.

Read as **indicative + reproducible**, not a definitive verdict. This is one private, same-domain,
natural-language benchmark; a punctuation/case-heavy or keyword-style query set would exercise the
sparse arm harder, and the structural invariance above would still cap any post-rerank effect at zero
until the pipeline changes.

## When the weight would become live (follow-ups, not done here)

The weight is a no-op on post-rerank recall **only because** the reranker sees the full candidate
union. It would start to matter if any of these changed — each is its own experiment:

- **Truncate the fused candidate pool before reranking** (e.g. rerank only the top-N of the fused
  list instead of the whole union). Then which docs survive to the reranker depends on the weight.
- **Ablate / disable the cross-encoder** (measure ensemble output directly). `pre@5` above is that
  measurement — and there the weight moves recall.
- **Split `CANDIDATE_K` per arm**, so the arms contribute unequal pool sizes.

Absent one of those, re-running this sweep will reproduce the flat post-rerank table.

## Reproduce

```
# retrieval-only, $0, deterministic (this box, private benchmark):
HF_HUB_OFFLINE=1 USE_MULTI_QUERY=false \
  uv run --no-sync python -m scripts.sweep_bm25_weight

# custom grid / determinism check:
uv run --no-sync python -m scripts.sweep_bm25_weight --grid 0.0,0.3,0.4,0.5,1.0 --repeat 3

# the flag also exists on the general eval CLI (adds the free/paid scorers):
uv run --no-sync python -m scripts.run_eval --bm25-weight 0.5
```
