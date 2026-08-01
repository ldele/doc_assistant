<!-- status: active · updated: 2026-08-01 · class: append-only -->

# Sparse arm A/B, repeated on a discriminating instrument (2026-08-01, private 35-case set)

**Question.** ADR-036 moved the keyword arm to an on-disk SQLite/FTS5 index, changing the ranking
function (FTS5 `bm25()`: k1=1.2, no IDF floor · `rank_bm25.BM25Okapi`: k1=1.5, eps=0.25). Its A/B ran
on the public 10-case set, where **both arms scored a perfect 1.0000 on all four recall metrics** —
parity at the instrument's ceiling, not proof. The legacy in-RAM arm, `DOC_SPARSE_INDEX` and
`bm25_cache.py` have shipped alongside the default ever since, waiting for this measurement.

**The blocker was not real.** Three baton entries (07-30, 07-31) recorded the private 35-case set as
absent from this box. `tests/eval/cases.yaml` is tracked in the repo (35 cases, 2026-05-28) and **all
33 of its `expected_citations` fragments resolve against the live 97-document library — 35/35 cases
runnable.** Verify a "missing file" claim before inheriting it.

## Setup

| | |
|---|---|
| Corpus | **97 documents · 33,105 parent-child chunks** (the live library) |
| Cases | `tests/eval/cases.yaml` — 35 private cases, 34 scored + 1 negative control |
| Retrieval | parent-child ON, `CANDIDATE_K=20`, `TOP_K=10`, `BM25_WEIGHT=0.4`, multi-query OFF |
| torch / box | 2.12.0+cu130, RTX 4070, Windows 11, Python 3.12 |
| Cost | **$0** — retrieval only, no generation, no judge |

**Method.** One arm per process (`DOC_SPARSE_INDEX=1` / `=0`), as ADR-036 did. The recall instrument
is the project's existing one — `scripts/sweep_bm25_weight._recall_at_k` / `_filenames`,
bidirectional substring over `expected_citations` — so this is comparable to ADR-036 and to the R6 and
weight-sweep numbers. The harness **asserts the arm it actually built** (`sparse_index_active`) and
refuses to record on a mismatch: the pipeline degrades silently to the in-RAM arm if the on-disk build
fails, and an A/B that quietly compared an arm to itself would report perfect parity for the worst
possible reason.

## 1. The instrument discriminates — which is the whole point

| Metric | on-disk (shipped) | in-RAM (control) | Δ | public 10-case (ADR-036) |
|---|---:|---:|---:|---:|
| pre-rerank recall@5 | 0.8186 | 0.8186 | 0.0000 | 1.0000 both |
| pre-rerank recall@10 | **0.8333** | 0.8186 | **+0.0147** | 1.0000 both |
| post-rerank recall@5 | 0.7010 | 0.7010 | 0.0000 | 1.0000 both |
| post-rerank recall@10 | 0.7598 | 0.7598 | 0.0000 | 1.0000 both |

Recall sits at **0.70–0.83**, not 1.0000. There is headroom in every direction, so a difference
between the arms had somewhere to show up.

## 2. Result — the shipped metric is identical; the one movement favours on-disk

**Post-rerank recall — what the user actually receives — is identical at both k.**

The only metric that moved is **pre-rerank recall@10, +0.0147, in the on-disk arm's favour**, and it
comes from exactly one case:

- **`brain_network_hubs`** ("What is the rich-club organisation of the brain?") expects two
  documents. The on-disk arm surfaced `nihms-326467.pdf` at candidate rank 9; the in-RAM arm did not
  surface it in the top 10 at all → pre@10 **1.0 vs 0.5**.
- **It does not reach the user.** Post-rerank, both arms return the *identical* three documents and
  both score post@10 = 0.5. The cross-encoder dropped the extra candidate.

That is the same structural finding the BM25-weight sweep recorded in 2026-07-03: the cross-encoder
re-scores the full candidate union, so candidate-order differences wash out by the time the answer is
assembled. This A/B is a second, independent instance of it — arrived at by changing the ranking
*function* rather than the arm *weight*.

## 3. But the arms are not equivalent — 9 of 35 queries return a different evidence set

| | |
|---|---:|
| Shipped top-10 identical in order | **26 / 35** |
| Shipped top-10 identical as a set | **26 / 35** |
| Differing cases | **9 / 35 (26%)** |

All nine differ by *set*, not merely order — different documents, not a reshuffle. Divergence starts
as early as rank 2. Examples: `hubel_wiesel_simple_complex` gains `elife-33281-v2.pdf` +
`fnana-09-00080.pdf`; `suarez_mammal_taxonomy` loses `cajal-lecture.pdf` + `nihms133032.pdf`;
`language_neuroanatomy` swaps one document for two.

**This is the finding the recall table cannot express.** Recall@k over `expected_citations` scores
only whether the *expected* document is present. It says nothing about the other 8–9 slots — which are
precisely what the LLM reads. Two arms can hand the model materially different evidence and score
identically.

## 4. Control — retrieval is NOT deterministic, and that assumption was load-bearing

ADR-036's baseline states *"Retrieval is deterministic on both arms, which is what `--repeat` buys
elsewhere"*; `sweep_bm25_weight`'s docstring says the same. **Re-running the on-disk arm against the
unchanged index disagreed on 1 of 35 cases.** So single-pass retrieval comparisons in this project
carry a case-level noise floor of roughly **3%**, and any future A/B should measure it rather than
assume it away.

**The noise is disjoint from the signal**, which is what makes §3 trustworthy:

| | |
|---|---:|
| Cross-arm differing cases | 9 / 35 |
| Same-arm re-run differing cases (noise floor) | 1 / 35 — `middleton_frontal_subcortical` |
| **Cross-arm diffs not explained by that noise** | **9 / 35** |

**Where the non-determinism lives, and why it is confined.** The noisy case's *pre-rerank candidate
list is byte-identical across both runs* — same ten files, same order. Only the post-rerank parents
moved (at ranks 5–6). So the flip is in the **cross-encoder**, not in either retrieval arm.

And the case itself explains the ties: `middleton_frontal_subcortical`'s only expected citation is
`middleton-2001.pdf`, which carries **`chunk_count=0`, `extraction_health='broken'`** — the
text-layer-less scan KI-29 exposed. Its target is not in the index at all; recall is 0.0 on both arms
in both runs and cannot be anything else. With no true match, reranker scores cluster and the tail
order is decided by ties. That is a benign mechanism, but it is **not** "retrieval is deterministic",
and the general claim should be retired.

## 5. Cost, as observed (not a controlled latency measurement)

| | on-disk | in-RAM |
|---|---:|---:|
| `RAGPipeline()` construction | **1.98 s** | 2.71 s |
| Mean ensemble `invoke` over 35 queries | 258.7 ms | **212.8 ms** |

**Do not quote the second row as a regression.** It times the *whole* ensemble call (sparse + vector +
fusion), not the sparse arm in isolation, so it is not comparable to ADR-036's per-arm 66.6 → 27.4 ms;
it is a single un-interleaved pass, which ADR-035's own rule says not to quote as an absolute for
anything I/O-bound; and the in-RAM arm was served from a warm `bm25_index.pkl`. It is recorded because
it was measured, and flagged because it is not evidence.

The memory result is unchanged and is the reason the change was made: 185 → 21 MB, corpus not
resident, backend RAM flat instead of growing ~2 MB/document (ADR-036 §1).

## Verdict

**The bar the baton set is cleared.** The A/B has now been repeated on an instrument that
demonstrably discriminates (recall 0.70–0.83, headroom everywhere, one metric actually moved), with a
measured noise floor, and **the shipped metric is identical on 35 cases at two values of k**. Where
anything moved at all, the on-disk arm was better.

**What this still does not prove.** Nine of 35 queries hand the LLM a different evidence set, and no
instrument here scores whether that changes an answer. Closing that gap means `contains_all` /
`llm_judge` over the 35 cases — a paid run, and one whose references carry `author_verified: false` in
places (the case file's own header warns against leaning on judge numbers from it). Retrieval parity
on the shipped metric is what is demonstrated; answer equivalence is not.

**Recommendation.** Deleting the legacy arm (`bm25_cache.py`, the `DOC_SPARSE_INDEX` branch,
`data/bm25_index.pkl`) is defensible on this evidence and is a decision, not a measurement — it is the
user's call, and it is a separate increment.

## Reproducing

```bash
DOC_SPARSE_INDEX=1 python ab_sparse_arm.py --expect-arm on_disk --out on_disk.json
DOC_SPARSE_INDEX=0 python ab_sparse_arm.py --expect-arm in_ram  --out in_ram.json
```

Harness: `scratchpad/ab_sparse_arm.py` + `ab_compare.py` + `determinism_probe.py` (session-local;
they wrap `scripts.sweep_bm25_weight`'s recall functions rather than reimplementing them). Promote
them into `scripts/` if this A/B is ever run a third time.
