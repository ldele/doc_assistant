<!-- status: active · updated: 2026-07-30 · class: append-only -->

# ADR-036 — Move the sparse retrieval arm off the Python heap into an SQLite/FTS5 index

- **Status:** accepted (built 2026-07-30); **amended by
  [ADR-038](ADR-038-retire-the-in-ram-sparse-arm.md) (2026-08-01)** — the A/B this ADR asked for was
  repeated on the private 35-case set, the shipped metric came back identical, and the in-RAM
  fallback + `DOC_SPARSE_INDEX` are now deleted
- **Date:** 2026-07-30
- **Deciders:** user + Claude Code
- **Supersedes in practice:** ADR-035's snapshot, which stays only as the fallback path's cache
- **Measurement:** `tests/eval/baselines/sparse_index_2026-07-30.md` (the A/B on the live corpus and
  the public eval cases) · `tests/eval/baselines/memory_and_lazy_reranker_2026-07-30.md` §3 (the
  attribution that set the target)

## Context

`RAGPipeline` held the entire corpus in memory to serve keyword retrieval: a `Document` per chunk, a
token list per chunk, and a `BM25Okapi` frequency dict per chunk. Measured at 97 documents / 33,105
chunks that was **195 MB of Python heap** after KI-32 step 1 (265 MB before it), linear in corpus
size, against a **10,000-document robustness contract** — roughly **20 GB** projected, i.e. a design
that could not reach its own contract on any consumer machine.

Step 1 (deduplicating the parent text) removed 26% and, more usefully, produced the attribution that
made this decision obvious. Per chunk: **~1.4 KB Chroma metadata dicts + strings · ~1.4 KB BM25 token
strings retained by the index · ~0.8 KB `BM25Okapi.doc_freqs` · ~0.6 KB pydantic `Document` overhead
· ~0.4 KB child text · ~0.4 KB parent text.** The chunk text is ~8% of the footprint. **There is no
version of "hold the corpus in Python, but smaller" that reaches the contract** — the index's own
structures and per-object overhead dominate, and both only disappear if the index leaves the heap.

## Options

| | Ranking | Memory at 33k chunks | New dependency | Verdict |
|---|---|---:|---|---|
| **A** — keep the in-RAM `BM25Okapi` (today) | the control | 195 MB, linear | — | the problem |
| **B** — reimplement Okapi over an SQLite postings table | **identical by construction** | O(postings touched) | none | rejected, below |
| **C** — **SQLite FTS5** (`bm25()` ranking) | different formula | **~0**, O(k) per query | none (stdlib `sqlite3`) | **chosen** |
| **D** — tantivy / a Rust index | different formula | ~0 | a native wheel in a PyInstaller bundle | rejected: KI-9's packaging risk for no gain over C |

**Why B was rejected even though it preserves ranking exactly.** The in-RAM arm's semantics are
"score every document containing *any* query term", and a query carries stopwords (`the`, `are`,
`of`) whose postings are most of the corpus. Reproducing those semantics on disk means reading
postings for every query term — at the contract, millions of rows per query, in Python. It would
preserve the ranking and inherit the very cost profile this ADR exists to remove: the current arm
already measures **66.6 ms/query at 33k chunks**, and that is the fast, in-memory version of the
same algorithm. Exact equivalence at the price of a worse slope is not a trade this project wants
twice.

## Decision

**Option C: one SQLite database beside the Chroma store** — `chunks` (text + metadata + `doc_hash`),
`parents` (one row per parent block, KI-32 step 1's map), and a **contentless FTS5 index** over the
token stream. `Document` objects are built for the returned rows only.

**Term parity is enforced, not hoped for.** The FTS5 table is declared
`tokenize="unicode61 tokenchars '-+' remove_diacritics 0"` and fed exactly what
`keywords.tokenize` emits (the token regex is `[a-z0-9]+(?:[-+][a-z0-9]+)*`, already casefolded), so
the vocabulary is the one the in-RAM arm had, `cross-encoder` included.

**Queries OR their terms.** A bare FTS5 term list means **AND**, which would silently return a small
fraction of the candidates — the same shape of failure as a stale cache, and invisible. Each term is
also double-quoted, so FTS5 operators appearing in a user's own question (`NOT`, `OR`, `*`) are data
rather than syntax.

**Scoping is a `WHERE doc_hash IN (...)` inside the ranked query**, before `LIMIT`, so ADR-025 F2's
folder scope returns the folder's own top-k. This retires the reason the scoped-ensemble LRU existed
(rebuilding BM25 over a subset at ~20 µs/chunk); the memo is kept because it still serves the
fallback path and costs nothing.

**Staleness reuses ADR-035's fingerprint** — chunk ids + collection + tokenizer source + schema
version — with one change forced by measurement: it is computed **streaming, page by page**, by
summing per-id digests (order-independent by construction rather than by sorting). Collecting the id
list first measured a **3.1 MB peak at 33k chunks, 94 B/chunk ⇒ ~0.3 GB at the contract**, and it
was the last corpus-linear allocation left at launch.

**The legacy arm stays, switched by `DOC_SPARSE_INDEX=0`.** It is the rollback, and it was the
control for the A/B below. It should be deleted once the comparison has been repeated on a corpus
where the eval instrument can actually discriminate (see Consequences).

## Consequences

**Measured on the live corpus (97 documents / 33,105 chunks, RTX 4070), on-disk vs in-RAM:**

| | in-RAM (control) | on-disk (shipped) | |
|---|---:|---:|---|
| Python heap held after construction | 185 MB | **21 MB** | **8.8×** |
| Corpus held in RAM | 33,105 chunks | **0** | — |
| Pipeline construction | 4.53 s | **2.79 s** | 1.6× |
| Sparse arm, per query | 66.6 ms | **27.4 ms** | 2.4× |
| `retrieve_with_scores`, per turn | 336 ms | **279 ms** | 1.2× |
| On disk | 39.9 MB (snapshot) | 40.7 MB (index) | ~equal |

**Retrieval changes, and the size of the change is measured rather than asserted.** The two arms
agree on **84% of the top-20 sparse candidates**. After the cross-encoder re-scores the union of
both arms, **8 of 10 public eval queries return a byte-identical final top-10** and the other two
differ by one document out of ten. Recall against the public cases' expected citations is
**identical: 1.0000 at pre@5, pre@10, post@5 and post@10**.

**The honest caveat on that eval.** Both arms score a perfect 1.0000, so the instrument has a
ceiling here: it proves no regression it is capable of seeing, and it is not capable of seeing a
small one. The private 35-case set that could discriminate is not on this machine. This is why the
change is justified on **cost**, with quality held at parity, and why the legacy arm and its switch
stay until the A/B has been repeated on a discriminating case set (recorded as debt in
`docs/performance.md`).

**What this does and does not fix.** After the swap, no component of the answer path holds the
corpus: the index adds ~1 MB at open, a query adds ~0, and — measured with a working-set probe —
Chroma's vector search adds **~1 MB**, so the vector store is *not* corpus-resident either. The
remaining process memory is constants: embedder weights ~800 MB, CUDA context ~450 MB, reranker
~500 MB in use. **The 10k-document memory ceiling KI-32 recorded is gone.** What is *not* fixed:
launch still performs an O(corpus) scan of chunk ids to fingerprint the store (~0.2 s at 33k,
projected ~20 s at the contract), and chromadb's own paged read has a ~159 MB working-set high-water
mark at 33k chunks that this ADR does not touch. The successor for both is an ingest-side version
stamp, so launch reads one row instead of every id.

**Costs, stated plainly.**
- **A second on-disk artifact** (~41 MB here, ~4 GB projected at the contract) duplicating chunk
  text already in Chroma, on top of the vector stores. Gitignored, rebuilt on demand.
- **Two retrieval code paths** until the legacy arm is deleted — the cost of keeping a rollback for
  a change that alters retrieval.
- **A first launch after an ingest pays the build** (2.8 s here for 33k chunks), streamed from
  Chroma page by page so the build never materialises the corpus either.
- **Ranking is now FTS5's.** `BM25_WEIGHT` still weights the arm, and the locked value is untouched,
  but the arm's internal ranking is no longer `rank_bm25`'s. Anything comparing against pre-2026-07-30
  sparse scores is comparing different scales.

**Explicitly not changed.** No locked setting moves: `CANDIDATE_K`, `TOP_K`, `BM25_WEIGHT`, chunk
sizes and the embedder are all exactly what they were, and the ensemble still fuses by weighted
reciprocal rank, so the sparse arm's score *scale* never mattered and still does not.
