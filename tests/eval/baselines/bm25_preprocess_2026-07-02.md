<!-- status: active · updated: 2026-07-02 · class: baseline -->

# BM25 preprocess_func (R6 fix 1) — retrieval eval, 2026-07-02

Remediation-plan §R6. `BM25Retriever.from_documents` used LangChain's default `preprocess_func`
(a bare `text.split()` — case-sensitive, punctuation attached, so `BM25?` never matches `bm25`).
R6 passes `keywords.tokenize` (casefold + tech-token). Eval-gated: ship only if it **beats or
matches** the control beyond variance.

## Method

- **Instrument:** retrieval-only **recall@K** — `rag.retrieve()` over the 35-case benchmark
  (`tests/eval/cases.yaml`), retrieved filenames vs each case's `expected_citations`
  (bidirectional substring). **$0** — no answer generation, no judge (`USE_MULTI_QUERY=false`, so
  `retrieve()` is local-only). Retrieval is **deterministic → ~zero variance**, so one pass per arm
  is representative (the `--repeat` intent is satisfied by determinism, not repetition).
- **Corpus:** the current `data/` store (76 docs, ~30.9k baseline chunks). BM25 is rebuilt in
  **memory** at pipeline init from the store, so control vs treatment is just the code change — no
  re-ingest, no store mutation.
- **Scope note:** the plan names the *public* corpus (`cases.public.yaml`), but it is download-only
  from arXiv and not present on this box; `cases.yaml` (the private benchmark) aligns with the
  current store and is the available instrument. The public run is the reproducible re-run below.

## Result

| Arm | BM25 `preprocess_func` | recall@5 | recall@10 |
|---|---|---:|---:|
| **Control** | default `text.split()` | 0.8775 (29/34 perfect) | 0.9069 (30/34 perfect) |
| **Treatment** | `keywords.tokenize` | 0.8775 (29/34 perfect) | 0.9069 (30/34 perfect) |

**Identical — zero regression.** Ship per the "matches control" gate.

## Interpretation (honest)

The benchmark is **reranker-dominated**: the cross-encoder + the 0.6-weighted vector arm already
surface the right chunks on these natural-language questions, so a better-tokenized BM25 candidate
pool doesn't change the final top-K here. The fix is nonetheless correct and worth landing:
- It **un-handicaps** the sparse arm — proven deterministically in
  `tests/unit/test_pipeline_retrieval.py::test_bm25_preprocess_func_matches_case_where_default_split_misses`
  (default `split()` ranks the wrong doc when the query is lowercased; `tokenize` ranks the right one).
- It is the **prerequisite for the 0.4/0.6 weight sweep** (a separate follow-up experiment): fixing
  the tokenizer moves the weights' optimum, so the sweep must run over a correctly-tokenized BM25.

Read as **indicative + reproducible**, not a definitive verdict — the private benchmark is one
corpus; a punctuation/case-heavy or keyword-style query set would exercise the sparse arm more.

## Reproduce

```
# private benchmark (this box), retrieval-only, $0:
HF_HUB_OFFLINE=1 USE_MULTI_QUERY=false python <retrieval recall@K over cases.yaml>

# public corpus (shareable):
uv run python -m scripts.download_corpus          # 10 arXiv papers
uv run python -m doc_assistant.ingest
uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml   # free scorers
```

## Rider fixes (correctness nits, ship regardless — R6 fixes 2–4)

- **Candidate dedup** now keys on a full-content SHA-256, not a 50-char prefix (distinct chunks
  sharing a header prefix no longer collapse). Guard: `test_dedup_keeps_distinct_chunks_sharing_a_50char_prefix`.
- **`expand_query`** on valid-but-non-list JSON now yields `[]` (was `[query]`, which prepended the
  query twice → the ensemble ran it twice). Guards: the three `test_expand_query_*` cases.
- **parent_text invariant** probed: every parent-child chunk carries `parent_text` (else it is
  unreturnable in PC mode). Guard: `test_every_parent_child_chunk_carries_parent_text`.
