<!-- status: active · updated: 2026-11-15 (H4: reranker on/off, CANDIDATE_K 20 vs 10, RG-035's first re-run of KI-36's citation coverage) · class: baseline -->

# H4's other three measurements: the reranker, `CANDIDATE_K`, and citation coverage on the fixed build

**On the private set, neither the reranker nor `CANDIDATE_K=20` makes a difference this n can
detect. Citation coverage on the local models is far higher than the August figures:
`llama3.1:8b` cites 89% of its sentences (was 37.6%) and `qwen2.5:7b` 70% (was 18.0%), on the
same 27 cases, with Ollama's log showing no prompt cut.** The August figures were taken while
KI-66 cut every long prompt to its last ~2,050 tokens. They described the cut, not the models.

These are 34 and 18 cases on one machine, so the intervals are wide. "No difference at this n"
means this library cannot tell the two settings apart; it does not mean they are equal.

## Retrieval: the reranker on and off, `CANDIDATE_K` 20 and 10

**Method.** A retrieval-only script, read-only: each case's query through
`RAGPipeline.retrieve_with_scores(query, trace=...)` at locked settings, with nothing generated.
Rank scores come from `doc_assistant.eval.scorers.RankScorer`. "Reranker off" is the fused
candidate order before the cross-encoder (`trace["pre_rerank"]`), cut to the same 10 as the
reranked list: what the app would pass on with the rerank step removed. `CANDIDATE_K` was set per
process (`CANDIDATE_K=10` or `20`), with no re-ingest (it is query-time only). One trial per arm:
retrieval is deterministic, and a second K=20 run on the private set matched the first case for
case. The intervals are 95% paired bootstraps over cases (`eval.paired.bootstrap_interval`, seed
20261113, 10,000 resamples).

- **Private:** the 35-case private set on the working library (105 documents). 34 cases name an
  expected document; the one that does not is left out.
- **Public:** the 18 single-turn cases on the 28 demo papers (`download_corpus --demo`).

| Set | Comparison | Score | A | B | B − A [95% interval] | cases up / down / same |
|---|---|---|---|---|---|---|
| private (34) | reranker off → on (K=20) | rank_mrr | 0.888 | 0.880 | −0.008 [−0.049, +0.040] | 2 / 5 / 27 |
| | | rank_ndcg@10 | 0.883 | 0.874 | −0.010 [−0.044, +0.027] | 2 / 6 / 26 |
| | | recall@10 | 0.990 | 0.975 | −0.015 [−0.044, 0.000] | 0 / 1 / 33 |
| private (34) | K 10 → 20 | rank_mrr | 0.865 | 0.880 | +0.015 [−0.015, +0.064] | 1 / 3 / 30 |
| | | rank_ndcg@10 | 0.857 | 0.874 | +0.017 [−0.010, +0.058] | 2 / 3 / 29 |
| | | recall@10 | 0.966 | 0.975 | +0.010 [0.000, +0.029] | 1 / 0 / 33 |
| public (18) | reranker off → on (K=20) | rank_mrr | 0.972 | 0.958 | −0.014 [−0.125, +0.083] | 1 / 1 / 16 |
| | | recall@10 | 1.000 | 1.000 | 0.000 | 0 / 0 / 18 |
| public (18) | K 10 → 20 | every rank score | same | same | 0.000 | 0 / 0 / 18 |

**Reading.**
- **The reranker:** every interval contains zero on both sets. On the private set it moves a
  case down more often than up (5 against 2 on MRR), and it drops one case's document out of the
  top 10. That is a lead to look at, not a finding: one case and seven moves are what this n can
  show. The cross-encoder stays. Removing it would need an experiment on its own, with a larger
  set, and the integrity layer reads its scores (`pipeline.py`, sigmoid-bounded scores).
- **`CANDIDATE_K`:** on the private set, the multi-paper-per-topic set the 2026-06-13 record asked
  for, 20 brings one document into the top 10 that 10 misses, and loses none. The rank scores
  move both ways. The verdict stays **keep 20, a no-regression default, not a measured win**: the
  public half (2026-06-13) and this private half both find no difference at their n.
- **What it cannot show:** whether either holds on a library of thousands of documents, where the
  candidate pool is crowded; answer quality (retrieval only).

## Citation coverage on the fixed build (RG-035, measurement 1)

**Method.** `scripts.run_eval --cases tests/eval/cases.yaml --repeat 3`, a two-column panel
through the pipeline adapter: `ollama / llama3.1:8b` and `ollama / qwen2.5:7b`, on Ollama's
native endpoint with `num_ctx` 8192. The reviewer was forced to the local model and no judge ran.
Coverage is the share of sentences carrying a citation the app resolves, pooled over answers as
in `citation_coverage_2026-08-07.md`. It is reported on that record's 27 cases (the 8 whose
document is degraded or absent left out, by the same id prefixes) and on all 35.

**Ollama's log shows `truncated = 0` on all 210 calls** (35 cases × 3 trials × 2 models). The
longest prompt was 6,536 tokens, which the pre-fix window of ~4,096 would have cut.

| Model | 27 cases, pooled, per trial | 27 cases, pooled, all trials | 27-case answers citing nothing | 2026-08-07 (one run, cut prompt) | 35 cases, mean per answer |
|---|---|---|---|---|---|
| `llama3.1:8b` | 93.8%, 85.2%, 88.8% | 416/466 = **89.3%** | 0 of 81 | 77/205 = 37.6% (12 of 27 cited nothing) | 0.865 (trial std 0.036) |
| `qwen2.5:7b` | 74.9%, 68.8%, 65.5% | 392/560 = **70.0%** | 1 of 81 | 43/239 = 18.0% (18 of 27 cited nothing) | 0.715 (trial std 0.026) |

Run ids (store `eval_h4.duckdb`, outside the repository): `61054b78, e5554f28, 4e5d2f30` (llama),
`b8ea51ef, 5f237fe7, 7f70e1d3` (qwen). About 40 minutes per trial: 15:55 to 19:48.

**Reading.**
- **KI-36's local-model floor was mostly the cut.** The August answers that cited nothing at all,
  12 of 27 on llama, are gone: the prompt's instruction to cite was the part the cut removed. The
  gap to Claude Haiku 4.5 (83.5% on 2026-08-07, not re-run; a paid model never had the cut) is no
  longer a gap for `llama3.1:8b` at this n. `qwen2.5:7b` still cites less.
- **Not the same path as August.** That record used the `ChatController`; this one used the
  pipeline adapter, which builds the same answer prompt and runs the same citation audit
  (`synthesis.audit_citations`) without the controller's routing. On the public set the two paths
  gave close figures for llama (0.841 pipeline, 0.811 controller, H4 baseline), but they were not
  compared on this set.
- **What it cannot show:** whether the citations are right (coverage counts a resolving
  citation, not a supporting one); any paid model on the fixed build; the private library's
  degraded documents (8 cases, shown separately in the store).

## What this closes and opens

- **RG-035, row (1):** re-run. `citation_coverage_2026-08-07.md` and KI-36 carry a dated line.
  Rows (2) and (3), the Node-B positions and the private chunk sweep's answer metrics, stay open
  for later sessions (the release plan places them after H4).
- **ROADMAP row 69:** the private retest is done, and the verdict is unchanged (keep 20, not a
  measured win).
- **The reranker ablation:** run, with no difference at this n.
