<!-- status: active · updated: 2026-07-20 · class: baseline -->

# RG-020 (part 1) — scoped-retrieval cost on the live index

Measured **before** building ADR-025 F2, so the caching decision (spec S5) is a decision rather
than a guess. This discharges the *real-corpus* half of RG-020; the synthetic 10k half stays
open — see "What is still owed".

## Setup

- Box: the Windows dev box (CPU). Index: the live parent-child Chroma store,
  **76 documents / 30,882 chunks**, `bge` 768-dim embeddings, `keep_for_retrieval` unset on
  every chunk (so nothing is excluded).
- BM25: `BM25Retriever.from_documents(subset, preprocess_func=keywords.tokenize)`, `k = CANDIDATE_K`.
- Vector: `collection.query(query_embeddings=[random 768-vec], n_results=CANDIDATE_K, where=…)`,
  median of 5 runs, seeded RNG (the query vector is arbitrary — the filter is what is being timed).
- Scopes are the first *N* `doc_hash` values in sorted order; no folder rows were involved, so
  this measures the retrieval cost only, not membership resolution (a single indexed SQLite read).

## BM25 — subset filter + index rebuild

| scope | docs | chunks | subset filter | index build | query |
|---|---|---|---|---|---|
| whole corpus | 76 | 30,882 | 4.1 ms | **622 ms** | 52 ms |
| 40 % | 30 | 9,331 | 3.2 ms | **248 ms** | 11 ms |
| 5 % | 3 | 806 | 3.2 ms | **27 ms** | 1.7 ms |

Build is ~linear in chunk count: **≈20 µs/chunk**.

## Vector — Chroma `where` filter

| filter | median |
|---|---|
| unscoped — `{keep_for_retrieval: {$ne: false}}` (today's) | 136 ms |
| `+ {doc_hash: {$in: [3 hashes]}}` | 193 ms |
| `+ {doc_hash: {$in: [30 hashes]}}` | 232 ms |
| `+ {doc_hash: {$in: [76 hashes]}}` | 408 ms |

The cost tracks the **length of the `$in` list**, not the scope's share of the corpus.

## What this decided

1. **Ship it at real corpus sizes.** Both costs are dominated by LLM latency (seconds) at 10²
   documents. RG-020's own "Until then" clause authorises exactly this.
2. **Cache the scoped ensemble (spec S5).** A 248 ms rebuild on every turn of a 30-document
   folder is avoidable: the UI scope is sticky, so a single memo slot keyed on the hash set makes
   every turn after the first free, and a membership edit changes the key so the slot
   self-invalidates.
3. **Never rebuild for the unscoped path.** The 622 ms whole-corpus figure is the cost the
   prebuilt `self.ensemble` already pays once at construction; the unscoped path must keep using
   it (spec S4, guarded by `tests/unit/test_pipeline_scope.py`).

## What is still owed (RG-020 stays open)

- **The 10k-document contract is NOT demonstrated for scoped turns.** Extrapolating: ~4 M chunks
  would put a full-scope BM25 rebuild near ~80 s, and a folder holding thousands of documents
  would push a very long `$in` list through Chroma. Both worst cases are the *large* scope — which
  is also the least useful one, since scoping to nearly everything is what "no scope" already
  expresses for free. Still, until measured on a synthetic 10k corpus, **do not claim the 10k
  robustness contract holds for scoped turns.**
- The ADR-025 fallback (per-folder precomputed indexes) remains the live "reverses if" branch.
- **Scoped BM25 statistics differ from global** (avgdl/IDF are computed over the subset). That is
  correct behaviour — the arm is asked to rank *within* the folder — but it means scoped and
  unscoped reranker inputs are not directly comparable, so a scoped-recall difference is expected,
  not a bug. No before/after retrieval-quality comparison was run; scoping is a content filter and
  is deliberately outside the eval gate (ADR-025 fork 4).

## Reproducing

The numbers above came from throwaway probes against the live store (Chroma opened
embedder-free + `BM25Retriever` over the pulled chunk texts). Re-run by timing
`RAGPipeline._ensemble_for(scope)` construction and a `collection.query(..., where=…)` with the
`$and` filter the pipeline builds, on the corpus you care about.
