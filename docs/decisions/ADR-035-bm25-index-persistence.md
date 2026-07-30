<!-- status: active · updated: 2026-07-29 · class: append-only -->

# ADR-035 — Persist the BM25 arm as a pure-data snapshot, fingerprinted against the chunk store

- **Status:** accepted (built 2026-07-29)
- **Date:** 2026-07-29
- **Deciders:** user + Claude Code
- **Measurement:** `tests/eval/baselines/stage_profile_2026-07-29.md` (the GPU baseline that
  re-ranked the startup costs) and the option table below, measured on the live corpus the same day.

## Context

`RAGPipeline.__init__` builds the BM25 arm from scratch on every launch: it reads **every chunk**
out of Chroma (`get_all(include=["documents", "metadatas"])`), materialises a `Document` per chunk,
then tokenises and indexes all of them. Measured on the live corpus (97 documents / 33,105
parent-child chunks, GPU box):

| Step | Cost |
|---|---:|
| `get_all` — whole-store read from Chroma | **1.058 s** |
| `BM25Retriever.from_documents` (tokenise + index + wrapper) | **0.615 s** |
| **Total, every launch** | **1.673 s** |

**Why this and not something else.** After the reranker went lazy (2026-07-28) and the GPU wheel
landed (2026-07-29), this is **the only startup component that scales with the corpus** — everything
else ahead of it (Python imports ~6.8 s, embedder weights ~2.4 s) is a constant. At the project's
**10,000-document robustness contract** (`.claude/CONTEXT.md`) the corpus is ~103× today's, which
puts this single step at roughly **170 s** — it stops being a launch cost and becomes a reason the
app cannot open. The 1.7 s today is not the problem; the slope is.

**The constraint that shapes every option.** `self._bm25_docs = all_docs` is retained for ADR-025 F2
folder-scoped retrieval, which rebuilds the BM25 arm over a subset. **The documents must be
materialised in memory regardless**, so "persist the index" cannot mean "skip loading the corpus" —
whatever we persist has to carry the documents too, or the 1.058 s read stays.

## Options

Measured end to end on the live corpus, not estimated:

| | What is persisted | Load cost | On disk | vs 1.673 s |
|---|---|---:|---:|---|
| **A** | nothing (today) | 1.673 s | 0 | — |
| **B** | documents only; rebuild BM25 at load | 0.921 s | 74.5 MB | 1.8× |
| **C** | documents **+ pre-tokenised terms**; rebuild the index at load | **~0.50 s** | 85 MB | **3.3×** |
| **D** | the live `BM25Retriever` object, pickled whole | 0.354 s | 88.8 MB | 4.7× |

Component split behind those numbers: tokenising all texts **0.200 s**, `BM25Okapi` index build
**0.206 s**, the rest being `Document`/wrapper construction. Tokenisation is *not* dominant — which
is why B leaves so much on the table and why C exists at all.

## Decision

**Option C — persist one snapshot containing only stdlib types** (`(text, metadata)` tuples plus the
token lists), and reconstruct `Document`, `BM25Okapi` and `BM25Retriever` from it at load.

**D is 0.15 s faster and was rejected anyway.** It pickles a live `langchain_community`
`BM25Retriever`, which makes the on-disk format an *implementation detail of a third-party class*: a
langchain upgrade then either fails to unpickle (acceptable) or unpickles into a subtly different
object (not acceptable, and not detectable). A cache whose worst failure mode is "retrieval quietly
changes" is the wrong trade for 150 ms — and this repo has just paid for one silent-truncation bug
this session (KI-31). Option C's blob **cannot** contain a foreign class, so that failure mode does
not exist.

**Invalidation — the part that actually has to be right.** The snapshot is keyed by a fingerprint of
everything that would change the index:

- the **chunk ids**, sorted and hashed. Any add, removal or replacement moves the set, and a
  `--rebuild` mints fresh UUIDs — so it catches the case a bare *count* cannot: an edit that
  replaced as many chunks as it removed. Measured **0.221 s** for 33,105 ids, against **5.389 s**
  for the full documents+metadata read it guards; sorted so the hash cannot depend on Chroma's page
  ordering;
- the **collection name** — an embedding-model change points retrieval at a different collection;
- the **tokeniser's source**, hashed via `inspect.getsource(tokenize)` — so changing the tokeniser
  self-invalidates the cache instead of relying on someone remembering to bump a constant;
- an explicit **`_CACHE_VERSION`** for changes those cannot see (e.g. the payload shape).

**Rejected: the store file's mtime**, which was the first implementation and is the obvious cheap
signal. It does not work: **opening a `chromadb.PersistentClient` rewrites `chroma.sqlite3`'s mtime
even for a pure read**, so the fingerprint invalidated the cache on the very next launch and the hit
rate was zero. Caught by measuring the thing rather than trusting it — the tests now pin both
directions (replaced ids invalidate; touching the store file alone does not).

Any mismatch, any read error, any corrupt file ⇒ **fall back to the live build**. The cache is an
accelerator, never a source of truth: there is no code path where a bad cache produces a wrong
index rather than a slower launch.

**Escape hatch.** `DOC_BM25_CACHE=0` disables it entirely, so a suspected cache problem can be ruled
out without a code change or deleting files.

## Consequences

**Good.** The BM25 startup stage drops **5.36 s → 1.99 s (2.7×)** on this corpus, measured
end-to-end with the cache off and on, **interleaved** across 4 rounds so machine-load drift hits
both arms equally. The saving grows linearly with the corpus, which is the point of the exercise.
Nothing about retrieval changes: the reconstructed index is built from the same texts with the same
tokeniser, pinned by guard tests asserting that cached and freshly built retrievers return
**identical documents and identical scores** for the same query.

> **On the absolute numbers.** The same whole-store read measured **1.06 s** early in the session
> and **5.39 s** later the same day, on an unchanged store — OS file-cache state dominates, and the
> option table above was taken in the warm condition. Quote the **ratio**, which was measured
> interleaved and holds in both conditions; do not quote either absolute figure as *the* cost. The
> `--json` output of `scripts/profile_stages.py` is the reproducible record.

**One consequence worth stating: the miss path now tokenises once, not twice.** The terms computed
for the snapshot are handed straight to the index builder, instead of `BM25Retriever.from_documents`
re-running `tokenize` over every chunk. The write is also skipped wholesale when the cache is
disabled — argument expressions evaluate before the call, so an unguarded `save(...)` would have
tokenised the whole corpus and thrown the result away on exactly the configuration that asked for no
cache.

**Costs, stated plainly.**
- **~85 MB on disk** for this corpus, duplicating chunk text already held in Chroma. At 10k
  documents that is ~9 GB, comparable to what the store itself holds. This is a real cost and the
  reason the cache is disableable; if it ever becomes the binding constraint, the successor is a
  memory-mapped or columnar format, not a return to rebuilding on every launch.
- **The first launch after an ingest is ~0.3 s slower** — it pays the write. One-off, synchronous,
  and deliberately not backgrounded: a thread here would race the first query for no user-visible
  gain.
- **`pickle` is used on a file in the user's own data home.** Same trust domain as `library.db` and
  the Chroma store sitting beside it, both equally writable; the blob holds no foreign classes, and
  the fingerprint is verified before the payload is trusted. It is *not* a transport format and must
  never be read from anywhere but the data home.

**Explicitly not changed.** No locked setting moves — `CANDIDATE_K` is applied to the retriever
*after* construction and is not baked into the snapshot, so this ADR needs no eval-harness
experiment. The BM25 weights, tokeniser and candidate count are all exactly what they were.

**Follow-on.** The whole-store read this avoids is still paid by anything else that needs every
chunk (`compute_epistemics`, the concept skeleton). Those are batch jobs, out of scope here; if the
same snapshot turns out to serve them, that is a later ADR.
