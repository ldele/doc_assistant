<!-- status: active · updated: 2026-08-01 · class: append-only -->

# ADR-038 — Retire the in-RAM BM25 arm; the on-disk index is the only keyword arm

- **Status:** accepted (built 2026-08-01)
- **Date:** 2026-08-01
- **Deciders:** user + Claude Code
- **Amends:** [ADR-036](ADR-036-sparse-index-on-disk.md), which shipped the on-disk arm but kept the
  in-RAM one as a rollback pending a measurement
- **Supersedes:** [ADR-035](ADR-035-bm25-index-persistence.md) entirely — its snapshot cached a
  structure that no longer exists
- **Measurement that unblocked it:**
  [`tests/eval/baselines/sparse_arm_private35_2026-08-01.md`](../../tests/eval/baselines/sparse_arm_private35_2026-08-01.md)

## Context

ADR-036 moved the keyword arm to an on-disk SQLite/FTS5 index and made it the default, but FTS5's
`bm25()` is not `rank_bm25`'s (k1=1.2 vs 1.5, no IDF floor), so **retrieval ranking genuinely
changed**. Its A/B ran on the public 10-case set, where both arms scored a perfect **1.0000 on all
four recall metrics** — parity demonstrated at the instrument's ceiling, not proven. The honest
response at the time was to keep the old arm behind `DOC_SPARSE_INDEX=0` and say so.

That left two retrieval implementations, `bm25_cache.py`, a second scoped-retrieval path, and a
39.9 MB snapshot artifact shipping for three weeks against a measurement nobody could run — the
private 35-case set was recorded in the baton as absent from this machine.

**It was not absent.** `tests/eval/cases.yaml` is tracked in the repo and all 33 of its
`expected_citations` fragments resolve against the live 97-document library: 35/35 cases runnable.
The claim had propagated unverified through roughly four session entries.

## The measurement

Retrieval-only, $0, one process per arm, with the arm **asserted** rather than assumed (the pipeline
degrades silently to the fallback if the on-disk build fails, so an unchecked harness could have
compared an arm to itself and reported perfect parity for the worst possible reason).

| Metric | on-disk | in-RAM | Δ |
|---|---:|---:|---:|
| pre-rerank recall@5 | 0.8186 | 0.8186 | 0.0000 |
| pre-rerank recall@10 | **0.8333** | 0.8186 | **+0.0147** |
| post-rerank recall@5 | 0.7010 | 0.7010 | 0.0000 |
| post-rerank recall@10 | 0.7598 | 0.7598 | 0.0000 |

**The instrument discriminates** — 0.70–0.83, not 1.0000, with headroom in every direction. The
shipped metric (post-rerank) is **identical on 35 cases at two values of k**, and the one metric
that moved favours the on-disk arm. Two facts are recorded alongside it and neither is decorative:
**9 of 35 queries return a different evidence set** (recall@k scores only whether the *expected*
document is present, not the other 8–9 slots the LLM reads), and **retrieval is not deterministic** —
a same-arm re-run disagreed on 1 of 35 cases, so there is a ~3% case-level noise floor. The noise is
disjoint from the 9, which is what makes them trustworthy.

## Decision

**Delete the in-RAM arm.** `bm25_cache.py`, `DOC_SPARSE_INDEX`, `DOC_BM25_CACHE`,
`_load_bm25_corpus`, `_split_parent_texts`, `_build_bm25`, `_bm25_docs`, `_parent_texts` and the
in-RAM branch of `_ensemble_for` are gone. `sparse_index.py` is the keyword arm.

**A failed index build now degrades to vector-only, and says so.** This is the real cost of the
decision and the part that needed designing rather than deleting.

## Consequences

### The one that is worse

Before, a failed build fell back to a slower, memory-hungry arm that still did keyword matching, and
the user was never told. Now there is nothing to fall back to: **retrieval runs on the vector arm
alone**, and an exact term the embedder does not place nearby will be missed.

Removing a silent recovery means the state has to be **said** rather than absorbed. Three changes
carry that:

1. `RAGPipeline.keyword_index_unavailable` — deliberately **not** the same as
   `not sparse_index_active`. The latter is also true for an empty library, which is a supported
   state and nothing to report. Conflating them would either cry wolf on every fresh install or stay
   silent on a real degradation.
2. `corpus_stats` reports `mode="unavailable"` (replacing `in_memory`) and **withholds the index's
   size and build time** — a stale file may well still be on disk, and its bytes would describe an
   index that is serving nothing.
3. The Settings panel says *"Keyword search is off — answers are using meaning-based search only, so
   exact terms may be missed"*, styled as a warning, and offers **Rebuild**.

### The inversion worth noticing

`rebuild_sparse_index` used to **raise** when no index was live — correct then, because such a
pipeline was serving keyword results from the fallback and a rebuild was meaningless. Now that state
is exactly the one a rebuild fixes, so it runs whether or not an index is live, and the route's only
409 is an empty corpus. A recovery button that refuses to run in the state it exists for would be the
wrong way round.

Emptiness is re-derived from the rebuild's own fingerprint scan rather than read from the
construction-time `_corpus_empty` snapshot: this method exists to be called after an ingest, and a
pipeline that launched against an empty library must not refuse to index what the user just added.

### The ones that are better

- One retrieval path instead of two — and the second one was the *untested-in-production* one.
- ~600 lines and two modules deleted; `pipeline.py` loses its largest branch.
- The 39.9 MB `data/bm25_index.pkl` is dead. It stays in `.gitignore` so an older checkout's leftover
  can never be committed by accident.
- Memory is flat by construction rather than by configuration — there is no longer a switch that
  reintroduces the KI-32 ceiling.

## What this does not claim

**Answer equivalence is not proven.** Nine of 35 queries hand the model a different evidence set and
no instrument here scores whether that changes an answer. Closing that gap means `contains_all` /
`llm_judge` over the 35 cases — paid, and the case file's own header warns its references are
`author_verified: false` in places. What is demonstrated is **retrieval parity on the shipped metric,
on an instrument that can discriminate**. That was the bar this decision was gated on; it is not the
same as the stronger claim.

## Rejected

- **Keep the flag but default it off** — a rollback nobody exercises is not a rollback; it is a
  second code path that rots. Git history is the rollback.
- **Fall back to a freshly-built in-RAM index on failure** — that is the arm we just deleted, minus
  the cache, at the moment the machine is already unhappy. It also restores the silence.
- **Raise on a failed build** — an unwritable data home would stop the app answering at all. Inform,
  don't block: half a retrieval with a warning beats no retrieval.
- **Report `unavailable` with the stale file's size** — technically true, actively misleading.
- **Keep `_split_parent_texts`** — its job (deduplicating parent text out of the in-RAM corpus) was
  subsumed the moment the corpus stopped being in RAM.
