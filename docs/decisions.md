# Design decisions

This document contains the *why* behind my choices, and the planned evolution of the project.

---

## About This Project

This project implements and evaluates established RAG techniques on a personal 
document library. It does not introduce new algorithms or methods.

- Hybrid retrieval (BM25 + dense vectors): standard practice since 2023.
- Cross-encoder reranking: standard practice, originally popularized by Microsoft 
  research and Sentence-Transformers.
- Parent-child retrieval: a LangChain pattern, predates this project.
- Section-aware chunking: a common refinement, not unique here.
- Query rewriting / HyDE: documented techniques from the literature.

What this project does contribute is:
- An end-to-end working system over real research documents.
- A measurement harness that lets us compare these techniques empirically.
- Documented design decisions for a specific domain (personal academic libraries).
- Practical engineering: caching, streaming ingest, metadata cleanup, regression 
  tracking.

---

## Core Decisions

### Local-first by default

Designed to run locally on a personal machine. Cloud LLMs are available as a practical option. Embeddings, vector store, reranker are all local.

### Markdown as the universal intermediate format

Instead of having each extractor produce its own data structure, every format converts to markdown first. This means:

- One chunking strategy works for all formats.
- The cache is human-inspectable (you can open the `.md` files and see what got extracted).
- Adding a new format is one extractor function, no downstream changes.

The trade-off: some structural information is flattened. I considered it worth it for the simplification.

### Why not LlamaIndex or a managed RAG service

Wanted to practice using LangChain. Using LlamaIndex would have hidden some of the parts I wanted to learn.

### Streaming ingest over batch

A batch ingest holds the entire corpus in memory before storing. For most cases, that's fine ; for someone having 500+ research PDF papers, it isn't. 
Memory stays flat regardless of corpus size, and a crash on document 200 doesn't lose documents 1-199.

### Two-tier cache

Extraction is far slower than embedding. Caching only embeddings means re-extracting PDFs every time chunking strategy changes. Caching markdown separately decouples extraction from chunking — change the splitter, re-embed, but don't re-extract.

### Hybrid search instead of pure vector

Vector search loses on exact terms (author surnames, equation labels, library function names). BM25 loses on paraphrased questions. Ensemble retrieval at weights `[0.4, 0.6]` (keyword:vector) gave better hits during testing on technical documents.

**Methodology rigor: ⚠ vibes-locked, no numbers behind the weights.** "Gave better hits during testing" was a qualitative impression from before the eval harness existed. The *architectural* choice (hybrid retrieval) is well-justified by the asymmetric failure modes of pure-vector vs pure-BM25; the *specific weights* (0.4/0.6) are essentially arbitrary. Re-measure via `scripts/run_eval.py` with a weight sweep when the `--bm25-weight` flag exists (small follow-up). Citation_overlap alone (deterministic, free) is enough for this; no answer-LLM needed.

### Cross-encoder reranking on top of retrieval

Embedding-based retrieval is fast but not perfect. The reranker is slower per item 
but only sees ~10 candidates, so total cost is small.

### Document hierarchy with orthogonal topic tags
Folder (user-defined) → Document → Part (section/chapter) → Chunk
↑
(topic tags)

A chunk has one structural parent (its part within a document) but can carry 
many topic tags. This separates *structural* hierarchy (the document's 
inherent organization) from *semantic* membership (what concepts a chunk is 
about). Both should be queryable independently.

Folders are user-defined groupings that don't have to align with topics or 
documents — useful for projects, reading lists, course materials.

### Metadata as soft filter, not hard gate

The RAG remains the core of the project while using hybrid search + 
reranking. The addition of a metadata layer would act as an optional scope ("only search in folder X" or 
"only methodology sections"). 
This should preserve the baseline behavior while adding precision when explicitly requested.

### Two-step metadata cleanup

Current:
    Cleanup scripts use a dry-run + apply pattern:
    1. Dry run reports what would change, makes no modifications
    2. User reviews, then re-runs with `--apply`

Reasoning: the goal is to improve retrieval quality while not always re-running ingest which is costly. 
It only needs to be done once for each document, so it is extra worth it to run a cleanup on the files.

### Parent-child retrieval is the default

According to literature and testing, this choice should improve performance.

Measurement with strict judge (temperature=0, tight rubric):

|---|---|---|
| Retrieval recall | 0.90 | 0.95 |
| Correctness | 3.48 / 5 | 4.10 / 5 |
| Faithfulness | 3.90 / 5 | 4.35 / 5 |
| Methodology correctness | 3.00 | 4.40 |
| Latency | 5.24s | 4.52s |

A +0.62 correctness improvement with a strict judge is a substantial real
effect. Methodology questions in particular benefit (+1.40), because they
require procedural context that small chunks can't convey.

The mechanism: child chunks (~400 chars) are embedded and used for precise 
retrieval. When a child is retrieved, its containing parent passage 
(~2000 chars) is passed to the LLM instead. This gives the LLM richer 
context per retrieved item without sacrificing retrieval precision.

The baseline (single-chunk) retrieval remains available via 
`USE_PARENT_CHILD=false` in the environment.

Note: This setting is for what I currently have in my library.
What matters is that I can quickly check the performance of the settings.

**Methodology rigor: ⚠ moderate. Point estimates, no error bars.** The original measurement predates the eval harness; sample size (N), variance, and the case set are all unrecorded. The +1.40 lift on methodology questions is too large to be pure noise even at small N, so direction is trustworthy; magnitude is not. To re-validate, toggle `USE_PARENT_CHILD` and run `uv run python -m scripts.run_eval --with-llm-judge --repeat 3` against the current 35-case set — would give a defensible mean ± std comparison.

### TOP_K tuning — adopted at TOP_K=10

Hypothesis: increasing the number of retrieved chunks would help when the 
correct content was being found but not all of it made the top-5 cutoff.

Measurement swept TOP_K over [3, 5, 8, 10, 12, 15] to find optimal value
(strict judge, parent-child enabled):

| K | Correctness | Faithfulness | Latency | Notes |
|---|---|---|---|---|
| 3 | 3.57 | 3.95 | 4.3s | Too little context |
| 5 | 3.81 | 4.05 | 4.8s | Previous default, undertuned |
| 8 | 4.42 | 4.72 | 5.9s | Previously considered optimum |
| 10 | **4.55** | **4.74** | 10.9s | Peak on both metrics |
| 12 | 4.43 | 4.55 | 11.7s | Past peak, faithfulness drops |
| 15 | 4.33 | 4.45 | 13.2s | Clearly past optimum |

Decision: K=10 adopted. Both correctness and faithfulness peak there, with
clear declines on both sides. The +0.13 correctness over K=8 comes at +5s
latency, accepted as worthwhile for a research tool.

**Update (2026-06-07, commit `09115c8`): retrieval K was split.** `TOP_K` (=10) is
now only the *final* rerank cut — the parents passed to the LLM. A new `CANDIDATE_K`
(=20) is the candidate pool fetched from *each* retriever (vector + BM25) before
reranking; previously this was hardcoded to `10 == TOP_K`, leaving the cross-encoder
no headroom to reorder. The sweep table above predates the split (it ran with pool ==
cut == 10). **`CANDIDATE_K=20`: public-confirmed (no regression); private arm pending.** It changes
retrieval output, so it went through the eval harness. **Public-corpus A/B (2026-06-13,
`--repeat 3` + judge): a tie — no regression vs the pre-split `CANDIDATE_K=10`**
(`contains_all` 0.933 vs 0.931, `llm_judge` 3.74 vs 3.83 — both within trial-mean std;
`citation_overlap` saturates at 1.000 so it can't discriminate). Record:
[`tests/eval/baselines/candidate_k_public_2026-06-13.md`](../tests/eval/baselines/candidate_k_public_2026-06-13.md).
The public set is one-paper-per-topic, so it cannot exercise the cross-paper *crowding* the
wider pool targets — that benefit still wants a re-run on the private neuroscience corpus
(`cases.yaml`, multi-paper-per-topic) before `CANDIDATE_K=20` is a *measured win* rather than a
safe default. `CANDIDATE_K=10` reproduces the exact pre-split behaviour; a guard requires
`CANDIDATE_K >= TOP_K`. See `config.py:107-119`, `pipeline.py:82,88`, and
`tests/unit/test_retrieval_config.py`.

> ⚠ **RETEST NEEDED (TODO).** The 2026-06-13 A/B is a *weak* signal — 10 cases over a
> 10-doc, one-paper-per-topic corpus is too small to reliably rank `CANDIDATE_K`. Treat
> the "no regression / keep 20" verdict as **unvalidated**, not settled. Re-run on a
> larger, multi-paper corpus (the private neuroscience `cases.yaml`, and/or an expanded
> public set) before drawing any firm conclusion or locking `CANDIDATE_K`.

Outcome: substantial improvement across all measures except latency.
Five of six known regressions resolved, including all three historical PDF
failures that previous experiments couldn't fix (scanned documents).

Mechanism: the reranker's 6th, 7th, and 8th choices are still highly 
relevant (cutting at 5 was removing relevant context).

Trade-offs:
- Going below K=10 loses retrieval breadth on distributed-information questions
- Going above K=10 dilutes the LLM's context with marginally-relevant chunks
- The faithfulness drop past K=10 confirms this is the inflection point, not noise

Note: This setting is for what I currently have in my library.
What matters is that I can quickly check the performance of the settings.

**Methodology rigor: ⚠ moderate. Strong shape, weak error bars.** The 3 → 5 → 8 → 10 → 12 → 15 sweep with a clear peak on *both* correctness and faithfulness is hard to attribute to noise (an accidental peak would not be coherent across two independent metrics). Sample size, variance, and the case set are unrecorded. To re-validate, run a script-driven sweep over `TOP_K` (and now `CANDIDATE_K`) values, persisting per-K runs in DuckDB and aggregating via `Store.aggregate_runs`. The old `scripts/run_topk_sweep.py` was a pre-harness artefact and was deleted in the 2026-06-07 dead-code audit (commit `09115c8`); a new sweep would reuse the eval harness (cf. `scripts/sweep_chunking.py`).

### Multi-Query expansion — tested twice, not adopted

Hypothesis: generating 3 variations of each query and combining retrieval 
results would improve recall on ambiguous or distributed-information questions.

Two prompt variants tested:
- Variant A: paraphrase + narrower + broader
- Variant B: paraphrases only (no scope shifts)

Both regressed against the no-MQ baseline:

| Metric | TOP_K=8 alone | TOP_K=8 + MQ-A | TOP_K=8 + MQ-B |
|---|---|---|---|
| Correctness | 4.29 | 3.62 | 4.00 |
| Faithfulness | 4.55 | 4.25 | 4.30 |
| Regressions resolved | 5/6 | 2/6 | 2/6 |

Mechanism of failure: query variations multiply the number of candidates 
the reranker has to discriminate among. Even when variations stay tightly 
scoped (paraphrase-only), they dilute the reranker's selection of the 
top-K positions. The highest-relevance chunks survive, but the marginal 
positions (6th-8th) get displaced by slightly-less-relevant candidates 
from variations.

Note: This setting is for what I currently have in my library.
What matters is that I can quickly check the performance of the settings.
As of phase 2, the cost outweighs the benefits. 

The MQ code remains available but disabled by default.

**Methodology rigor: ✓ acceptable.** Two independent prompt designs both regressed against the same baseline — convergent evidence that the regression is mechanistic, not noise. Sample size is still unrecorded, but the qualitative consistency carries the conclusion. Re-validation low priority; if revisited, the bar is "show that MQ wins on *some* sub-population of questions" (e.g., genuinely ambiguous queries), which would imply a different design — query-type-conditional MQ rather than always-on.

### Document health tracking

Health classification implemented as a pure-Python scoring model in 
`src/doc_assistant/health.py`. Each document is classified as healthy, 
marginal, or broken based on observable signals at ingest time:

- Chunk count and chunks-per-page ratio
- Average chunk length
- Section detection rate
- PDF page marker presence
- Reference-flagged chunk ratio

Score ≥75 = healthy, ≥40 = marginal, <40 = broken.

The classification is informational, not blocking: broken documents are 
still retrievable, but visually flagged for the user's attention.
Broken document content might still be valuable. 

Known limitations:
- Some documents might be classified as marginal. (Issues with some pdf formats)
  Not blocking; accepted as a known issue.
- Section detection regex now strips markdown formatting from heading 
  text to prevent future occurrences. 
  (Might have to build something more robust after further testing).

To resolve some edge cases, a dedup script (`scripts/dedupe_documents.py`) existed 
that formerly cleared duplicate Document rows from the path+content hash drift
(likely an artifact of the chroma→sqlite migration). It was deleted in the
2026-06-07 dead-code audit once content-only hashing removed the root cause — see
the Content-only hashing section below.

### Library browser via chat commands

Implementation pattern: `/library` command renders all documents as a 
structured Chainlit message. Chosen for implementation simplicity and 
reliability within Chainlit's native message API.

Known scaling limit: this pattern works well for libraries up to ~100 
documents. Beyond that, rendering hundreds of cards in a single message 
becomes unwieldy and slow.

Future migration path when needed:
1. Add pagination (Phase 8): `/library page:2` for 20-doc pages
2. Or migrate to a real sidebar/separate UI (Phase 8+) if Chainlit gets 
   better layout support, or migrate the whole app to Streamlit/Reflex 
   with proper navigation
3. The data layer in `library.py` is independent of the UI — switching 
   UI frameworks doesn't require redesigning queries

The library.py abstraction is specifically designed so that any UI 
change is a UI change, not a data layer rewrite.

### Enrichment-Layer Pattern (post-ingest, idempotent, sidecar by default)

Established by Phase 4 (`citations.py`, `metadata_extractor.py`) and adopted
as the standing pattern for all further data enrichment.

Primary ingest (Phase 1–3) is locked: extract → markdown → chunk → embed →
store. Any new derived data — citations, figures, tables, doc-level
vectors, future enrichments — ships as a **separate module + CLI runner**
that reads from the markdown cache (and the PDF source where needed) and
writes to its own sidecar table or manifest.

Rules:

- One module per enrichment kind (`citations.py`, `metadata_extractor.py`, 
  `figures.py`, `tables.py`, `doc_vectors.py`, ...).
- One CLI runner per module under `scripts/extract_*.py` with 
  `--dry-run` / `--apply` / `--force` / `--doc <hash>` flags.
- Idempotent: re-running on the same input produces the same output. Skip 
  docs that already have results unless `--force`.
- Sidecar by default. Splicing back into the markdown is only allowed when 
  the enrichment is *text-shaped* (e.g., tables). Binary artifacts (figures, 
  embeddings) live outside the markdown.
- No mutation of the primary chunk store from an enrichment module.

Trade-offs:

- Re-runnability is the goal. As extractors improve, every enrichment can 
  be re-run without touching primary ingest.
- Cost: small duplication of scaffolding (one CLI per module). Accepted.
- Risk: enrichment results can drift from the underlying markdown if the 
  cache is regenerated. Mitigated by content-only hashing — re-extraction 
  produces the same hash if the content is unchanged.

### Enrichment provider-intent guard — no `--apply` CLI can *silently* spend (added 2026-06-15)

**Context — the footgun.** The `.env` here is all-Anthropic (`LLM_MODE=api` →
`LLM_PROVIDER=anthropic`, with `ANTHROPIC_API_KEY` set), and the generation
providers default to inheriting it (`WIKI_LLM_PROVIDER`/`REVIEWER_PROVIDER`/
`JUDGE_PROVIDER = LLM_PROVIDER`). So an enrichment run the user believed was
"local" — Ollama is installed and GPU-ready — silently billed the API. This
caused a real credit burn (2026-06-15). PR 16 (Feature 7) fixed it *for the
concept graph only* by defaulting `CONCEPT_GRAPH_LLM_PROVIDER` to `"ollama"`
explicitly and gating an Anthropic run behind a warning + abort window. This ADR
**generalises that fix** so the next enrichment CLI can't reintroduce the footgun.

**Decision.** Two complementary mechanisms, kept deliberately small:

1. **One shared guard — `llm.assert_provider_intent(provider, *, operation, apply,
   model, scope, abort_seconds=3)`.** A no-op for dry runs (`apply=False` never
   bills) and for local/free providers. For a **paid** provider (the single source
   of truth is the declarative `config.PAID_PROVIDERS = {"anthropic"}`) it makes
   the spend impossible to miss: it raises `ProviderCostError` if the credential
   is missing (so `--apply` fails up front, not deep in the SDK mid-batch), else
   prints a bordered cost banner to **stderr** (operation / provider / model /
   scope) and sleeps a Ctrl-C abort window. `DOC_ASSUME_YES=1` skips only the
   pause (automation/CI); the banner still prints — it is never silent. Every
   `--apply` enrichment CLI routes through this one helper (`build_wiki`,
   `describe_figures`, and `build_concept_graph`, whose hand-rolled inline guard
   was refactored onto it). ASCII-only output — Windows stderr may be cp1252.
2. **Generator-role enrichment defaults to LOCAL, not `LLM_PROVIDER`.** Following
   the concept-graph precedent, `WIKI_LLM_PROVIDER`/`WIKI_LLM_MODEL` now default to
   `ollama`/`llama3` **explicitly** instead of inheriting the analysis provider —
   the wiki summariser is a per-cluster batch generator with the same silent-spend
   profile as concept extraction. So `build_wiki --apply` is free by default and
   `--provider anthropic` is the opt-in (which then hits the banner above).

**Why the guard lives in `llm.py`, not `config.py`** (despite the helper being
imagined as `config.assert_provider_intent`): `config.py` is pure declarative
data with no I/O; provider *behaviour* already lives in `llm.py` next to
`make_client` / `reviewer_available`. So the **policy** (which providers bill) is
a config constant; the **behaviour** that reads it is in `llm.py`.

**Scope — pinned instruments left unchanged.** The reviewer and eval judge
(`REVIEWER_*` / `JUDGE_*`) keep inheriting `LLM_PROVIDER` *by design* — they are
pinned measurement instruments (cross-run comparability), and their CLIs
(`reviewer_report --anchor`, `run_eval --with-llm-judge`) are already explicit,
key-gated, documented-as-paid opt-ins — i.e. not *silent*. Routing their warning
through the same guard is a one-line extension if desired, but flipping their
*defaults* would break comparability, so it is out of scope here. This supersedes
the "deliberate departure from the wiki's inherit-the-default convention" note in
the Feature 7 ADRs above: the wiki now shares the concept-graph's default-local
stance, and both share one guard.

**Rejected alternatives.** (a) Warning-only with no default flip — leaves
`build_wiki`'s default paid, so the common path still surprises. (b) Default-flip
only — a stray `WIKI_LLM_PROVIDER=anthropic` in the env would still spend with no
warning; the guard is the backstop that fires regardless of *how* the provider was
chosen. (c) An interactive `input()` confirm — hangs in non-interactive/CI runs;
a printed banner + timed Ctrl-C window (mirroring the shipped concept-graph
pattern) is automation-safe.

### Research Integrity Layer

A cross-cutting layer that records how each answer was produced and 
separates *evidence* from *AI interpretation* in synthesis-mode answers. 
Ships in three chunks across Phases 5, 6, and 9.

**Sources (influences, not specs):**

- AI Usage Cards (arXiv 2303.03886) — provenance card schema.
- PRISMA-trAIce (PMC12694947) — Phase 9 export target for AI-assisted 
  literature reviews.
- BE WISE framework (Frontiers, April 2026) — influence on the `human` 
  synthesis mode. Treated as a vendor framework, not adopted as a spec.
- Nature Methods — disclosure norm; satisfied as a byproduct of trAIce 
  export.

**Chunk 1 — Provenance card (Phase 5).** New `answer_records` table stores
query, retrieved chunk IDs + scores, reranker scores, model name, prompt 
version, token cost, latency, timestamp. Rendered as a collapsible card 
under each answer. CLI export `/export-record <answer_id>` → JSON. The 
eval harness consumes the same record schema.

**Chunk 2a — Dual interpretation layer (Phase 6).** Synthesis-mode answers
return two layers: *evidence* (retrieved passages, faithfulness-scored, no
synthesis) and *interpretation* (LLM synthesis, clearly labeled, with 
retrieval-derived uncertainty markers). Per-claim accept/reject/edit, 
persisted on the answer record. Pre-interpretation checkpoint (coverage + 
citation density + faithfulness) warns the user before generating.

**Chunk 2b — Reviewer agent (Phase 6).** ✅ Done (2026-05-28).
`src/doc_assistant/reviewer.py` + `answer_reviews` sidecar table.
Reviewer defaults to Haiku, runs ONLY when PR 5.1's heuristic 
confidence flags fire (cost discipline). 4-dim rubric: 
faithfulness, citation_density, hedging_adequacy (1-5 each) + 
unsupported_claims_count. **Reference-FREE**: judges answer vs 
retrieved evidence, not vs a ground-truth reference — distinct from 
the eval harness's `LLMJudgeScorer` which compares against expected 
answers. `temperature=0`, single-turn, no system prompt — same 
isolation contract as the eval judge. JSON-in-prompt (no tool-use; 
keeps consistency with eval judge). No auto-retry. `/review <id>` 
slash command for manual re-review. 16 unit tests (+ 3 isolation tests in `test_reviewer_isolation.py`). Schema is 
reviewer-kind-agnostic so a future human-review path slots into the 
same table.
*Deferred (idea, 2026-06-02):* the trigger policy is currently fixed —
auto-run on heuristically-flagged answers + manual `/review`. A later
option could expose the trigger as a user setting (e.g. `REVIEW_MODE =
flagged | always | manual`). Current behaviour is the wanted default;
this is an opt-in knob, not a change to it.

**Chunk 2c — Reviewer aggregation & self-improvement loop (Phase 6).** Counts
a categorical `failure_tag` enum across reviewer verdicts to find *systematic*
failure modes, anchored against the verified eval set to separate reviewer bias
from real system fault. **Minimum-N gate (locked).** The reviewer is a biased
sampler (runs only on flagged answers) over a small corpus, so a raw count is
noise with a label. A `failure_tag` is **not reported as actionable** until it
clears `MIN_FAILURE_TAG_COUNT` occurrences across `MIN_FAILURE_TAG_DOCS` distinct
documents (config-driven and user-tunable, with conservative defaults ~10/5
tuned on the first real distribution). Below the gate the aggregation reports
"insufficient evidence," never a suggested fix. Counts are always shown with
their denominator ("4 / 7 flagged"), never bare. v1 is instrumentation, not
auto-action; the action layer waits until accumulated answer records clear the
gate. Read-only aggregation, Enrichment-Layer Pattern.

**Judge/reviewer are pinned instruments.** The eval `LLMJudgeScorer` and the
Chunk 2b reviewer are measurement instruments, not generators: their model +
version are pinned and recorded per run, and swapping is an explicit logged
event. Chunk 2c's bias-vs-fault anchor is only falsifiable if the instrument
does not drift. The full evaluation strategy, scorer limitations, verified-10
headline rule, and the local-judge calibration gate live in
[`tests/eval/TESTING.md`](../tests/eval/TESTING.md).

**Chunk 3 — PRISMA-trAIce export (Phase 9).** When the literature review 
generator produces a review, it also produces a structured trAIce-aligned 
disclosure (JSON + markdown). Pulls from `answer_records` and the 
adjudication log; no new instrumentation needed.

**Mode flag.** `SYNTHESIS_MODE = human | ai` (default `ai`).

- `human` — AI returns evidence only; interpretation is the user's. 
  BE WISE-influenced path.
- `ai` — dual-layer with reviewer scoring.

Both modes share the same pipeline; the flag selects the output template.

**Why retrieval-derived uncertainty markers, not self-reported confidence.**
LLMs are systematically overconfident and the mapping between 
self-reported scores and actual correctness is non-linear and 
model-dependent. Building a calibration model (Platt scaling or similar) 
requires a labeled dataset and ongoing maintenance — out of scope for a 
single-user tool. Retrieval-derived markers (citation density, source 
diversity, reranker score spread) are observable, not self-reported, and 
map to instrumentation that already exists.

### Feature 7 — cross-document concept graph: design decisions (PR 16)

Concept/entity graph over the library (nodes = concepts, edges = relations),
clustered into communities with high-degree "god nodes" surfaced. Post-ingest
**Enrichment-Layer** sidecar (`data/graph/graph.json` + `.manifest.json`), never
mutates the chunk store. Full code-level contract:
[`docs/specs/feature-7-concept-graph.md`](specs/feature-7-concept-graph.md);
resolutions + reopens-if in the `.claude/SESSION.md` baton (2026-06-15). Scope =
7a (extraction) + 7b (communities + god-nodes) + 7c (gap signals) in one PR; **7d
(knowledge-currency) is out** — it consumes this graph.

**ADR-1 — Community detection: NetworkX Louvain, not Leiden/graspologic.** The
roadmap named "Leiden (graspologic)"; both were verified unusable against the
live venv (networkx 3.6.1 / numpy 2.4.6). `networkx.leiden_communities` **raises
`NotImplementedError`** — it is a backend-dispatch stub with no bundled backend.
`graspologic` hard-pins `numpy<2.0` (+ a Rust wheel, gensim, POT, umap-learn,
statsmodels, Python<3.13), so installing it — even as an extra — forces a numpy
1.x downgrade that breaks the locked torch/sentence-transformers/chromadb stack
on both machines and corrupts `uv.lock`. **Decision:** use
`networkx.louvain_communities(weight="weight", resolution=…, seed=42)` — same
modularity family, ships in the installed networkx, **bit-deterministic with a
fixed seed** (verified). Keep a `detect_communities(algorithm="louvain")` seam
for a future numpy-2-compatible Leiden (`leidenalg`/igraph). Only new declared
dep: `networkx>=3.4` (pure-Python, already transitively present; floor is >=3.4
not >=3.6 because `requires-python>=3.10` and networkx ≥3.6 drops 3.10 — uv
resolves 3.6.x on the 3.12 runtime via markers). *Reopens if a
numpy-2 Leiden backend lands.*

**ADR-2 — Extraction provider: API-on-this-box, local-first stays the goal.** The
primary dev box has no Ollama (verified only on the RTX box —
[[rtx-machine-ollama-testing]]). `GRAPH_LLM_PROVIDER`/`MODEL` cascade from
`LLM_*` like `WIKI_LLM_PROVIDER`; validation here runs on `anthropic`/Haiku
(cheapest, matches the reviewer/VLM convention), free on Ollama on the RTX box.
**Consequence:** 7a is a modest *paid* run on this box;
`GRAPH_MAX_LLM_CALLS_PER_DOC` doubles as a cost cap. Local-first is held where a
local model exists, not abandoned. *Reopens if a local model lands on the dev
box.*

**ADR-3 — Node canonicalization: deterministic string + cosine over the existing
BGE embedder.** Feature 6 proved this corpus saturates (same-domain BGE vectors
~0.88–0.96); a noisy extractor fragments concepts ("Transformer" vs "the
transformer"), degrading community quality. **Decision:** `canonical_key`
(casefold/NFKC/strip) then merge near-duplicate *labels* by cosine over the
existing `embeddings.py` BGE factory (no new model/API, deterministic), above
`GRAPH_NODE_MERGE_SIMILARITY`. Rejected: string-only (ships a fragmented graph);
LLM-assisted dedup (cost + nondeterminism breaks idempotency). The merge
threshold is **provisional — tune on the first real run**; it is the top quality
risk.

**ADR-4 — Back-pointers use a deterministic composite chunk key, not a Chroma
id.** 7d needs `source_chunk_ids` so each claim is one hop from its evidence, but
`ingest.py` calls `add_documents()` with no `ids=` — Chroma generates UUIDs
**regenerated on every re-ingest**. There is no stable chunk id to reference (the
roadmap's "stable chunk ids" assumption was wrong). **Decision:** back-pointers
are the deterministic composite `{document_id}:p{parent_index}` (all
content-deterministic, survives re-ingest); the field keeps 7d's name, only the
value scheme is fixed. No chunk-store schema change. *Reopens if ingest later
stamps a stable persisted chunk id.*

**Deferred to 7d (persisted nullable now for forward-compat):** edge `polarity ∈
{supports,contradicts,refines,supersedes}` and `year` — no PR 16 feature reads
edge labels, and stance extraction is noisier on a small model and needs the
relative-year ordering 7d owns. PR 16 extracts a plain concept-`relation` verb
only. **JSON-only, no SQLite graph table** — 7d's `compute_node_weights(graph)`
takes the graph object and 7d owns its own `chunk_epistemics` table.

### Pipeline assembly via a serializable profile (added 2026-06-16)

A single `PipelineProfile` dataclass becomes the one object that selects every
swappable stage of a run — embedder, collection, LLM provider/model,
parent-child, reviewer, synthesis mode. `RAGPipeline.__init__` takes a profile
instead of reading module-level globals. Env vars become *the default profile*,
not the only configuration path.

**Context.** The four "stages" the UI will eventually expose — embedder,
classifier (`query_router.py`), analysis LLM, reviewer — are already
independently swappable, but selection is scattered across env vars read at
import time (`EMBEDDING_MODEL`, `LLM_PROVIDER`, `LLM_MODEL`, `USE_PARENT_CHILD`,
`USE_MULTI_QUERY`). There is no single object that says "*this* run uses
bge-base + folder `X`'s collection + Anthropic + reviewer-on." This blocks two
things: the parked **projects** feature (a project = a `Folder`; selecting one
must select its embedder + collection + scope in one move) and the eventual
**Chainlit replacement** (a new UI should configure *one profile object*, not
re-derive five globals).

The trigger was looking at flow/orchestration tools (Langflow). **Rejected**:
Langflow is a visual authoring *platform* (React Flow canvas, component
registry, its own API/MCP server). Adopting it means replacing the whole
front-of-house to gain one concept — formalized stage wiring — that is a single
dataclass here. LangGraph and Haystack pipelines were rejected for the same
reason: a runtime dependency to solve a wiring problem. The one idea worth
taking from that class of tool is the *principle*: **a pipeline is declarative
config, not imperative construction** (Langflow flows are JSON). We take the
serialization insight, not the software.

**Decision.**

- Introduce `PipelineProfile` (frozen dataclass) carrying: `embedder` (registry
  key → `embeddings.py`), `collection` (the resolved Chroma collection name),
  `llm_provider`, `llm_model`, `use_parent_child`, `use_reviewer`,
  `synthesis_mode`. It is plain data — serializable to/from dict/JSON.
- `RAGPipeline.__init__(profile: PipelineProfile)`. A
  `PipelineProfile.from_env()` constructor reproduces today's behaviour exactly,
  so the refactor is contract-preserving (existing callers get the env profile
  by default).
- **Do not invent a collection key.** The model×project axis is *already locked*
  by Feature 1b as `{folder_id}__{model_name}` (see Phase 6 → Feature 1b, and
  `embeddings.get_collection_name` for the model axis the default already
  proves). The profile's `collection` field *holds the resolved name*; it does
  not redefine the scheme. When projects ship, "select project" =
  `Folder` row → `(folder_id, folder.embedding_model)` → existing key →
  profile.
- **Granularity is the one genuinely new axis.** "User selects Documents or
  Concepts" means the collection key gains a third dimension only if Concepts
  become independently retrievable. Today the concept graph (`concept_graph.py`,
  PR 16) is a sidecar, not a retrieval store. Treat Concepts-as-retrievable as a
  *future* profile field (`granularity: "documents" | "concepts"`) gated on that
  store existing — not built now, but the profile is where it lands.

**Sequencing (hard dependency).** This sits *on top of* the LLM provider
protocol (`docs/specs/llm-provider-isolation.md`), which must land first.
`pipeline._build_llm()` (streaming, LangChain) and `llm.LLMClient` (one-shot
`complete()`) are two separate construction doors today (intentional — different
contracts). Formalizing assembly before unifying provider construction would
bake a two-door wiring into the profile. Order: **provider protocol → profile
assembler → (later) projects UI over profiles**.

**Consequences.**

- The parked projects feature becomes a thin mapping (`Folder` → profile), not a
  pipeline rewrite. Aligns with the UI-agnostic data layer already mandated for
  the Chainlit swap (Open Questions → UI ceiling): the new UI becomes a view
  over profiles.
- Eval gains a clean handle — a run is a profile; per-project eval (Feature 1b
  trade-off) is "iterate profiles," not new harness code.
- Cost: one dataclass + one constructor + threading `profile` through
  `RAGPipeline`. No new dependency. Module-level flag reads in `pipeline.py`
  move into `from_env()`.
- Risk: profile and the locked-settings table (`.claude/CONTEXT.md`) could
  drift. Mitigation: `from_env()` is the *only* place env→profile mapping
  lives, so the table stays the single source for defaults.

**Status:** design recorded; not built. Blocked on the provider protocol PR.
Not a standalone PR yet — fold the `PipelineProfile` introduction into the
provider-isolation work or the first projects PR, whichever lands first. No
code-level spec in `docs/specs/` until the collection-key granularity question
(Concepts-as-retrievable) is decided, which is itself gated on PR 16 maturing
past sidecar.

---

## Roadmap

Phases renumbered to absorb the doc-assistant-roadmap.md additions. See 
that document for the source of intent; this section records the locked 
phase plan.

---

## Production Engineering Standards

These apply across all phases. They are not deferred to Phase 6.

### CI/CD (GitHub Actions)

Every push and PR runs: ruff → mypy → pytest (coverage ≥40%) → bandit → pip-audit.
Merging on a red pipeline is never allowed. Branch protection enforced on `main`.

Rationale: mechanical checks before human review. Catching lint/type/security issues in CI is 10x cheaper than finding them in code review or production.

### Cross-machine toolchain — torch backend auto-detect, interpreter left to the box (added 2026-06-10)

The project must run unchanged on a GPU box (RTX, CUDA) and a CPU-only box. Two
toolchain hazards surfaced; one is fixed in-repo, the other is deliberately left
to the machine.

**Torch backend — `torch-backend = "auto"`, not a platform pin.** Earlier the uv
lock pinned `torch==2.12.0+cu130` for *all* `sys_platform == 'win32'`. That forced
the CUDA wheel onto CPU-only Windows boxes, where the transformer forward pass
segfaults (exit 139). Replaced with `[tool.uv] torch-backend = "auto"` (drop the
explicit `pytorch-cu130` index + the `[tool.uv.sources] torch` marker): uv probes
each machine's accelerator at sync time and resolves the CUDA wheel on the GPU box,
the CPU wheel everywhere else (incl. CI). One `uv sync` works on every machine; no
per-box torch swap. Rejected: (a) keeping the win32 pin + a documented manual CPU
swap (the prior workaround — recurring friction, easy to forget); (b) a GPU-opt-in
extra (`uv sync --extra gpu`) — explicit and deterministic, but adds a flag the GPU
box must remember; auto-detect needs nothing. Validated CPU-side (lock resolved the
`+cpu` wheel, full suite green); GPU-side wants one confirming `uv sync` on the RTX box.

> ⚠ **Update (2026-06-13) — `auto` does NOT deliver per-machine CUDA; superseded by a spec.**
> Verified on the RTX box + uv docs: `torch-backend` / `UV_TORCH_BACKEND` is honored **only by
> `uv pip`** — a no-op for `uv lock` / `uv sync` / `uv run`. So the committed lock pins a single
> wheel (`+cpu`) and the GPU box **silently runs CPU torch** (`cuda.is_available() == False`)
> despite the 4070; the "GPU box resolves the CUDA wheel" claim above is wrong. Rejected option
> (b) above (the `--extra gpu` pattern) is in fact the *correct* mechanism. The durable fix —
> mutually-exclusive `cpu` / `cu130` extras bound to explicit PyTorch indexes (`uv sync --extra
> cu130` on the GPU box, `--extra cpu` on CPU/CI) — is specced in
> [`docs/specs/torch-backend-per-machine.md`](specs/torch-backend-per-machine.md) and **✅
> implemented 2026-06-13**: `pyproject.toml` now has the `cpu`/`cu130` extras, two explicit
> `[[tool.uv.index]]` entries, `[tool.uv.sources] torch`, and `[tool.uv] conflicts`; `uv.lock`
> carries both wheels; CI pins `uv sync --locked --extra cpu --extra dev` + `UV_NO_SYNC=1`.
> Verified on the RTX box: `--extra cu130` → `cuda True`, `--extra cpu` → `+cpu`, and
> `--extra cpu --extra cu130` errors (conflict guard). Per-machine: `uv sync --extra cu130`
> (GPU) / `--extra cpu` (everywhere else, the safe default); add `--extra dev` for the
> toolchain, and prefix run commands on the GPU box (`uv run --extra cu130 …`).

**Interpreter OpenSSL — not pinned in-repo (deliberate).** Separately, the
uv-managed **Astral python-build-standalone** interpreter ships an OpenSSL that
hard-crashes (`no OPENSSL_Applink`) on *some* Windows boxes the moment any real SSL
client is built (`ssl.create_default_context`) — taking down every HTTPS call
(Anthropic, Ollama) and the SSL-touching tests. An official python.org CPython is
unaffected; the fix is to rebuild the venv on one. We did **not** encode this as a
repo-level `python-preference = "system"` + `.python-version`, because the GPU box
runs fine on the managed build and forcing system-Python resolution could break it.
Decision: keep it a documented remedy (`.claude/KNOWN_ISSUES.md` + README setup
note), not an enforced toolchain change — revisit only if more boxes hit it.

### Security tooling

- `bandit`: SAST on `src/`. HIGH findings block merge. MEDIUM acknowledged in PR.
- `pip-audit`: dependency CVE scan on every push. CVEs require fix or documented exception.
- `detect-secrets`: baseline committed. Pre-commit hook diffs against it. No secrets ever enter git history.

### Pre-commit (mandatory)

Hooks: ruff (lint + format), mypy, bandit, detect-secrets, file hygiene. Not optional.
`no-commit-to-branch` on `main` forces all changes through PRs.

### Structured logging

`structlog` with JSON output in staging/production. No `print()` in `src/`.
Every log entry carries: level, timestamp, module, event, and operation context.
Secrets and PII are never logged. Context binding per-request for the web UI.

### Exception hierarchy

`DocAssistantError` as base. Typed subclasses: `ExtractionError`, `IngestError`,
`PipelineError`, `StorageError`, `ExternalServiceError`. Exceptions chain with
`raise X from e` to preserve tracebacks. User-facing messages translated at UI boundary.

### Content-only hashing — implemented

Previous path+content hashing caused duplicate Document rows whenever a file
was moved, renamed, or re-extracted with a different extractor. This was a
data integrity issue blocking Phase 4 (citation graph depends on stable
document identity).

Change: `doc_hash(text, source)` → `doc_hash(text)`. SHA-256 of the extracted
markdown content only, truncated to 16 hex chars. Path is no longer part of
the identity.

Migration script (archived after the one-time run): `scripts/archive/migrate_to_content_hash.py` (dry-run + --apply).
Recomputes hashes in SQLite and both Chroma stores. Handles dedup collisions
when two paths had identical content by merging into the row with the highest
chunk count.

This also obsoletes `scripts/dedupe_documents.py` — the root cause (path
changes creating new hashes) no longer exists.

---

### ✅ Phase 1: Core RAG (complete)

Goal: working end-to-end RAG over personal documents.

- Multi-format ingestion (PDF, EPUB, HTML, DOCX, Markdown, plain text)
- Hybrid retrieval (BM25 + vector ensemble)
- Cross-encoder reranking
- Two-tier caching (markdown + embeddings)
- Streaming ingest
- Hash-based deduplication
- Conversation memory with question rewriting
- Inline citations with page numbers and section headings
- Source previews in the UI
- Token tracking with cost estimation
- Pluggable LLM (Anthropic API or local Ollama)
- Chainlit web UI + CLI fallback
- Docker support

### ✅ Phase 2: Quality Foundation (in progress)

Goal: make the existing Q&A measurably smarter. No quality improvement is 
worth claiming without a metric.

- 2.0 Section metadata cleanup (references, garbage, title-as-section)
- 2.1 Evaluation set: 20+ question/answer pairs spanning factual recall, 
  methodology, synthesis, state-of-the-art, and negative-answer cases
- 2.2 Metrics: retrieval recall@K, answer faithfulness, answer correctness,
  latency, token cost
- 2.3 Baseline measurement
- 2.4 Experiment: chunk sizes — **✅ DONE 2026-06-06 (RTX/GPU).** Sizes are
  config-driven (`PARENT/CHILD/BASELINE_CHUNK_SIZE`, `config.py:160-169`) and were
  swept over 6 parent/child configs via `scripts/sweep_chunking.py` through the eval
  harness on the public corpus. Result: defaults `2000/200 · 400/50` confirmed best —
  no config beats the control. Record:
  [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](../tests/eval/baselines/chunking_sweep_public_2026-06-06.md).
- 2.5 Experiment: parent-child retrieval
- 2.6 Experiment: section-aware retrieval
- 2.7 Experiment: query rewriting / HyDE
- 2.8 Lock in best combination
- 2.9 Tech debt: content-only document hashing

Deliverable: a measurable improvement in answer quality, with numbers.

### ✅ Phase 3: Document Store + Library UI (complete)

Goal: treat the library as a first-class object. Browse, upload, organize, 
inspect, and control how documents are processed.

Core components:
- SQLite document store implementing the Folder → Document → Part → Chunk 
  hierarchy
- Library browser in the Chainlit UI
- Drag-and-drop upload with debounced ingest
- Per-document, per-folder, and per-part metadata editing
- Optional metadata-scoped retrieval ("answer using only documents tagged X")
- Tag and folder management

**Document Health & Ingestion Control** (sub-feature):

Silent extraction failures destroy trust and pollute downstream metrics. 
The current global-env-var approach to extractor selection doesn't scale — 
different documents need different treatments.

- Pre-ingestion profiling: detect scanned vs born-digital, presence of text 
  layer, page count, estimated work
- Classification: "born-digital", "scanned", "hybrid"
- Recommended extractor per file with rationale shown to the user
- Per-document extractor override (PyMuPDF4LLM vs Marker, eventually others)
- Post-ingestion health check: chunk count, section detection, page coverage, 
  "looks broken" flag
- Per-document health badges in library view: ✅ good / ⚠️ marginal / ❌ broken
- One-click re-extraction with a different extractor (no manual cache deletion)
- Persistent ingestion log

Rationale: this experience was discovered the hard way — historical PDFs 
(Hodgkin & Huxley, Hubel & Wiesel) extracted to 1-10 chunks each 
because they lack proper text layers.
A non-technical user wouldn't have noticed until eval results were inexplicable.

**Phase 3 completion gate — all done (2026-05-21):**
- ✅ Content-only hashing implemented and migrated (27 docs, all 16-char content-only hashes)
- ✅ CI pipeline green (ruff, mypy strict, pytest ≥40% — was 70% pre-Phase-3, lowered to 45% then 40% pending integration tests, bandit, pip-audit advisory)
- ✅ Pre-commit hooks installed and baseline clean
- ✅ `.env.example` committed with all 8 env vars documented
- ✅ Library metadata routing extracted to `query_router.py`
- ✅ `chainlit_app.py` refactored to thin shell; business logic in `query_router.py` + `commands.py`

### ✅ Phase 4: Citation Graph

Goal: model relationships between documents. Both explicit (citations) and 
implicit (semantic similarity). Surface the structure of the literature.

**Done (2026-05-26 session — explicit edges):**
- Tier-1 regex citation extractor (`src/doc_assistant/citations.py`). 
  Two-tier design with LLM fallback was the locked decision; tier-1 
  measured at 22/27 docs (81%) on the existing corpus, 1,234 citations 
  parsed. Tier-2 LLM fallback deferred — measure first, escalate when 
  data warrants.
- Doc-level metadata extractor (`src/doc_assistant/metadata_extractor.py`) 
  for title / authors / year / DOI. Coverage on the 27-doc corpus: 27/27 
  title, 26/27 authors, 23/27 year, 7/27 DOI.
- Internal citation matcher (DOI → first-author-surname + year → fuzzy 
  title via stdlib `SequenceMatcher`). Cross-citation rate is 
  structurally low on the current corpus (mostly recent papers citing 
  classics not in the library).
- CLI runners: `scripts/extract_citations.py`, `scripts/extract_doc_metadata.py`. 
  Both idempotent (skip already-processed docs unless `--force`).
- Slash commands: `/cites`, `/cited-by`, `/graph` (Mermaid for ≤25-node 
  subgraphs).
- 45 unit tests across citations + metadata.
- GROBID and refextract evaluated and deferred — Docker + Java service 
  cost not justified by the recall data so far.

**Done (2026-05-28 session — implicit edges):**
- Doc-level vector enrichment (`src/doc_assistant/doc_vectors.py`). 
  Mean-pools chunk embeddings from the baseline Chroma store per 
  document, L2-normalises, and computes pairwise cosine similarity. 
  Returns directed top-K=10 nearest-neighbour edges per source above a 
  0.5 cosine threshold.
- Sidecar table `doc_similarities (source_document_id, target_document_id, 
  embedding_model, score, computed_at)` — composite PK so future Phase 5 
  embedder swaps (Feature 1) don't collide with existing rows.
- CLI runner `scripts/compute_doc_vectors.py` — `--dry-run` / `--apply` / 
  `--force` / `--doc <prefix>` / `--top-k` / `--threshold` flags. 
  Idempotent: refuses to overwrite without `--force`.
- `library.similar_docs(doc_id, limit=10, embedding_model=...)` query 
  joined to Document for filenames/titles.
- `/similar <id>` slash command in `commands.py`.
- 15 unit tests over the pure-numpy core (mean-pool, edge computation, 
  threshold + top-K behaviour).

**Locked design choices for doc vectors:**
- Persist edges only, not the mean-pooled vectors themselves. Vectors 
  are recomputed from Chroma on demand. Rationale: at personal-library 
  scale (10s–100s of docs) the recompute is seconds; a separate vector 
  table is bloat with no measurable benefit. Rejected: persist vectors 
  for incremental updates (premature at this scale).
- Source store: baseline Chroma (`CHROMA_PATH`), not PC. Fewer, larger 
  chunks per doc, simpler iteration, equivalent mean-pool semantics. 
  Rejected: PC child store (finer granularity but more chunks per doc; 
  no signal it improves the doc-level vector).
- Directed top-K, asymmetric. The cosine relation is symmetric but the 
  top-K trim makes the persisted edge set directional. Gives a stable 
  "most similar to A" UX. Consumers wanting symmetric edges can union 
  both directions. Rejected: undirected edges (loses "top-K of A" 
  intuition; would need an arbitrary disambiguation rule).
- O(N²) similarity is fine for the current scale. Rejected: ANN index 
  (premature; Phase 7 if and when the library grows past ~1000 docs).

**Remaining (operational, not architectural):**
- Apply the citation extraction, metadata backfill, and doc-vector 
  computation on the local DB. All three CLI runners exist; this is the 
  15-minute shell run flagged in CLAUDE.md.
- LNCS colon-separator format and multi-column extraction artifacts are 
  known tier-1 weaknesses; cosmetic, deferred.

### Phase 5: Embedding & Eval Foundation

Goal: domain-aware retrieval backed by reproducible measurement, plus 
the first deliverable of the Research Integrity Layer.

See `docs/doc-assistant-roadmap.md` for the source of intent.

- **Feature 1** — ✅ Done (2026-05-28). `EMBEDDING_MODEL` env var; 
  `src/doc_assistant/embeddings.py` registry + factory. Initial options: 
  `bge-base` (default → legacy `"langchain"` collection name for zero-migration 
  upgrade), `specter2` (academic, uses its own collection). 15 unit tests. 
  Per-folder routing (Feature 1b) gated on Feature 3 measurement.
- **Feature 2** — ✅ Done (2026-05-28). `src/doc_assistant/eval/` 
  package: `cases.py` (YAML loader), `results.py` (dataclasses), 
  `scorers.py` (5 scorers: ExactMatch, ContainsAll, CitationOverlap, 
  EmbeddingSimilarity, LLMJudge), `runner.py` (Runner with progress 
  callback + exception capture), `store.py` (DuckDB, 3-table schema, 
  context-manager), `report.py` (summary + diff), `adapters.py` 
  (RAGPipeline wrapper — only file with doc_assistant deps; 
  extractable per Feature 5). CLI `scripts/run_eval.py` with paid 
  scorers gated behind explicit flags. 59 unit tests (cases 9 + runner/store 27 + scorers 23).
- **Feature 3** — ✅ Done (2026-05-28). 35-case eval set, run via the 
  harness against two embedders (BGE vs SPECTER2) with a hardened 
  reference-only LLM judge (`temperature=0`). 5 trials per model 
  (`--repeat 5`); reported as mean ± trial-mean std. **Result: bge-base 
  wins on the retrieval-side signals; the two are tied on the LLM judge.**

  | Scorer | bge-base (n=5) | specter2 (n=5) | Δ (bge − specter2) | Verdict |
  |---|---:|---:|---:|---|
  | `citation_overlap` (0-1) | **0.907 ± 0.000** | 0.887 ± 0.002 | +0.020 | Real (deterministic gap) |
  | `contains_all` (0-1) | **0.804 ± 0.013** | 0.767 ± 0.014 | +0.037 | Real (~4σ) |
  | `llm_judge` (1-5) | 2.209 ± 0.053 | 2.224 ± 0.092 | −0.015 | Tied (within noise) |

  The cross-encoder reranker and the answer LLM together level out the 
  embedder differential at the chunk level: what matters is which 
  documents got retrieved (bge wins) and whether the answer surfaces the 
  expected keywords (bge wins); the judge's rubric over the resulting 
  answers sees no meaningful difference.

  **Key finding — SPECTER2 lost because of training-task mismatch.** 
  SPECTER2 was trained for paper-level citation prediction over 
  abstracts; our task is chunk-level QA retrieval over full text. 
  bge-base's training corpora (MS MARCO, NQ, SQuAD, HotpotQA) are 
  much closer to QA retrieval. **Implication:** lock bge-base as 
  default; defer Feature 1b (per-folder embedder routing) until a 
  workflow emerges where SPECTER2's strengths apply (e.g., the 
  `/similar` paper-level task). Locked in `embeddings.py` registry; 
  re-confirmed empirically.

  **Provenance + reproducibility.** This comparison was measured on a 
  private, non-redistributable corpus, so the numbers above are kept 
  here as the record but cannot be reproduced by a third party. It is 
  **queued to be re-run on the public demo corpus** 
  (`tests/eval/cases.public.yaml` + `corpus_manifest.yaml`) so the 
  embedder evidence becomes reproducible alongside the rest of the 
  benchmark. Until then, treat the embedder verdict as provisional.

  **Methodology caveats:** 35 cases is small — effect sizes are 
  suggestive, not significant. LLM-judge means are low (~2.2/5) because 
  the rubric is strict reference-only grading; absolute scores are not 
  directly interpretable, the cross-model gap is the signal. 
  `embedding_similarity` is excluded — it uses the active embedder, so 
  the comparison is confounded across models (fix pending).
- **Integrity Chunk 1** — Provenance card. `answer_records` table; 
  collapsible card under each answer; `/export-record <id>` → JSON. 
  Hooks into existing `tracking.py`.
- **Provider layer (generation side of Feature 1)** — the config-driven-provider pattern Feature 1 applied to embeddings extends to generation. A normalized `LLMClient.complete()` protocol (`src/doc_assistant/llm.py`) with Anthropic + Ollama adapters backs the reviewer and the eval judge; the streaming chat path stays LangChain but reads `LLM_PROVIDER`/`LLM_MODEL`. **Local-first is the end goal, hybrid today** — the *generator* should run fully local (analysis + chat on Ollama, no `ANTHROPIC_API_KEY`). The *judge and reviewer are pinned instruments*, not generators: they default to a fixed, version-recorded reference model and only move to local once the local-judge calibration gate passes (see `tests/eval/TESTING.md`). Note: Feature 4c's VLM figure description is API-only and not part of the local path. Reviewer context isolation (evidence-only prompt, separate instance) is pinned by a guard test. Full design: `docs/specs/llm-provider-isolation.md`.

### Phase 6: Per-project routing + Figures & Tables + Dual-layer interpretation

Goal: turn the static embedding config into per-corpus routing; promote 
figures and tables to first-class content; ship the dual-layer 
interpretation + reviewer agent.

- **Feature 1b** — Per-project embedder routing. **Gated on Feature 3 
  results.** `Folder.embedding_model` column. Collection naming 
  `{folder_id}__{model_name}`. Cross-folder queries hit each collection 
  independently and the cross-encoder reranker resolves the mixed-space 
  merge.
- **Feature 4a** — table extraction pass. ✅ **Engine decision LOCKED 2026-06-02**
  (RTX eval): **Marker** wins, run out-of-process (`uvx --from marker-pdf
  marker_single`, never imported). Implemented: `src/doc_assistant/tables_marker.py`
  + `scripts/extract_tables_marker.py` — Marker tables are spliced inline,
  **caption-anchored**, into the cached `.md` (`<!-- table:marker:page=N:begin …
  :end -->`), de-duping pymupdf4llm's lossy inline twin and placing the block in one
  move. **pdfplumber is frozen as an explicit no-dep fallback** (`tables.py` /
  `extract_tables.py`). Idempotent; enters retrieval on the next re-ingest; never
  touches the chunk store. 12 unit + 3 integration tests. Remaining: optional
  full-corpus `extract_tables_marker --apply` on the GPU box.
  - **Detection is a shared page classifier (`regions.py`), not a table
    detector (locked).** A visual check (`scripts/debug_tables.py`) showed
    geometric detectors (pdfplumber, pymupdf `find_tables`) mis-read
    scientific *figures* as tables (a bar-chart grid → 13 "tables") and
    both mis-fire on shaded *prose*. Root-cause fix: classify each page
    once — table / chart / photo / figure / text — from three measured
    signals and route. (1) **"Table N" caption** (figures say "Figure N",
    prose has none); (2) **vector curve-density** — charts are 1k-78k
    curves vs 8-187 on table/text pages; (3) **raster image-area fraction**
    — figures cover 0.09-0.60 of the page vs 0 on table/text. A page is a
    table-candidate only if it has a table caption AND is neither
    chart- nor image-dominated, so a chart with a stray "Table" mention is
    not mis-routed. `tables.py` extracts only on classified table pages;
    content guards (`MAX_CELL_CHARS`, ≥`MIN_COLS` non-empty cols) are the
    second line of defence. Validated: no-table paper → 0; Table-1 paper →
    Table 1. This same classifier is the **Feature 4b foundation** (its
    chart/photo/figure verdict + image-area signal feed figure handling);
    v1 is page-level, per-region bbox splitting is the deeper 4b step.
    Thresholds (`CHART_CURVE_MIN`, `IMAGE_AREA_MIN`) measured on 2 eLife
    docs — tunable, validate on a wider corpus.
  - **Engine eval — ✅ resolved 2026-06-02 (RTX).** Marker won; pdfplumber
    (which fragments multi-part tables) is retained frozen as the no-dep
    fallback. `scripts/eval_marker_tables.py` was the comparison harness.
  - **De-dup — ✅ done.** `tables_marker.splice_tables_inline` strips
    pymupdf4llm's lossy GFM table(s) within each page span (`_strip_gfm_tables_text`)
    before inserting Marker's block, so the cache holds one clean table, not two.
  - **Verification — ✅ built.** A table-retrieval eval case
    (`tests/eval/cases.tables.yaml`, `dpr_topk_accuracy_table`) plus the CI
    mechanism gate `tests/integration/test_marker_table_retrieval.py`; measured
    2026-06-06 (DPR Table 2: Top-20 78.4, Top-100 85.4). A hand-verified gold-set
    cell-exact fidelity scorer remains a roadmap future, not built now.
- **Feature 4b** — Figure region detection + caption pairing — ✅ **shipped (PR 8,
  2026-06-14)**, spec `docs/specs/feature-4b-figure-detection.md`. No LLM cost.
  Sidecar `figures` table (`src/doc_assistant/figures.py` +
  `scripts/extract_figures.py`); PNG crops on disk under `data/figures/{doc_hash}/`;
  caption text stays in the markdown (additive, never spliced — ADR-2). v1 derives
  region bboxes from **PyMuPDF geometry** built on the shipped `regions.py`
  classifier — raster image blocks for photos, the drawing-rect union for charts,
  largest-block / caption-only fallback otherwise (ADR-1). **OpenCV refinement
  deferred** (the classifier already does the chart/photo/figure discrimination).
  Idempotent, `--force`-gated, per-doc isolated; the additive `create_all` mints
  the table (no Alembic). Runs on either machine (no torch/Marker/GPU). Public
  corpus run (10 papers): 45 regions, 44 PNGs + 1 caption-only, every region
  caption-paired. Feature 4c turns these rows into VLM-described retrievable chunks.
- **Feature 4c** — VLM figure description (gated) — ✅ **code shipped (PR 9,
  2026-06-14; one paid validation run pending user OK)**. Anthropic vision +
  forced tool-use with a Pydantic-validated `FigureDescription` schema
  (`figure_type`/`summary`/`key_quantities`/`axes`/`trend` — **no confidence
  field**, per "don't surface self-reported LLM confidence"). The gated
  enrichment (`scripts/describe_figures.py`) writes `Figure.vlm_description`;
  ingest then materialises `caption + description` as a `chunk_type='figure'`
  retrieval chunk in **both** stores. The enrichment is the only paid, API-only
  layer and is **Anthropic-only by decision** (no local path); model is the
  `FIGURE_VLM_MODEL` knob (default `claude-haiku-4-5`, matching the project's
  cost-gated instrument convention). Three gates: caption-only figures skipped
  (no PNG), self-describing captions skipped (`FIGURE_CAPTION_DESC_MIN_CHARS`),
  and `MAX_VLM_CALLS_PER_DOC` budget; every skip records a
  `vlm_call_skipped_reason`. **Chunk-emission decision (ADR):** figure chunks
  enter retrieval through *ingest reading the sidecar* (the only chunk-store
  writer), exactly like spliced tables — the 4c enrichment never touches Chroma,
  preserving the Enrichment-Layer rule. Eval hook: `FigureRetrievalScorer`
  (held-out figure spec → was its figure chunk retrieved?). Dry-run on the public
  10-paper corpus: 38 of 45 figures eligible, 7 skipped as well-captioned.
- **Integrity Chunk 2a** — Dual interpretation layer. `SYNTHESIS_MODE 
  = human | ai` (default `ai`). Evidence + interpretation as separate 
  output sections. Per-claim adjudication.
- **Integrity Chunk 2b** — Reviewer agent. Cheaper LLM scores the 
  interpretation against a structured rubric (faithfulness, citation 
  density, hedging adequacy, claims-without-sources). No auto-retry.
- **Integrity Chunk 2c** — Reviewer aggregation & self-improvement loop —
  ✅ **code shipped (PR 12, 2026-06-14; one paid anchor run pending)**.
  Adds a categorical `failure_tag` enum to the reviewer (`reviewer.FAILURE_TAGS`,
  `none`/`missing_citation`/`overclaim`/`evidence_contradiction`/`no_hedge`/
  `unsupported_claim`) — countable patterns alongside the free-text `notes`,
  persisted on a new `AnswerReview.failure_tag` column. `reviewer_aggregate.py`
  (+ `scripts/reviewer_report.py`) reads `answer_reviews` and resolves the central
  ambiguity — *reviewer bias vs system fault* — by anchoring against the golden
  eval set (the reviewer also flags verified-good golden answers ⇒ bias; flags
  production but not golden ⇒ system fault). Read-only aggregation,
  Enrichment-Layer Pattern, no auto-remediation. An unanchored "pattern in
  suggestions" must not ship — so without the anchor the report says
  "unanchored", never bias-vs-fault. **Minimum-N gate:** a tag is actionable only
  past `MIN_FAILURE_TAG_COUNT` / `MIN_FAILURE_TAG_DOCS` (tunable; 10/5 default,
  `MIN_FAILURE_TAG_DOCS` = distinct **answer records**, so one re-reviewed answer
  can't trip it); below that it reads "insufficient evidence." Counts always carry
  their denominator. v1 is instrumentation, not auto-action. **Migration note:**
  the `failure_tag` column is added to the *pre-existing* `answer_reviews` table by
  an explicit idempotent `ALTER TABLE` in `db/migrations.py` (`create_all` makes
  new tables but never alters an existing one — the one place an additive column
  needs more than `create_all`). See the Research Integrity Layer subsection above.
  **Scale caveat (honest cost/benefit):** this loop is over-engineered for a solo,
  single-user tool and will mostly report "insufficient evidence" until usage
  volume accrues — its payoff is gated on a server/multi-user deploy or on using it
  to diagnose local-model oddities. See *Deferred Improvements → "Chunk 2c
  self-improvement loop — built, but its payoff is scale-gated"*.
- **Engineering: chunking sweep** (reopens Phase 2.4). ✅ **Done 2026-06-06.** Chunk
  sizes are config-driven (`PARENT/CHILD/BASELINE_CHUNK_SIZE`, shipped 2026-05-31);
  `scripts/sweep_chunking.py` measured size grids through the eval harness on the
  public corpus (RTX/GPU): the historical defaults `2000/200 · 400/50` were measured
  and confirmed best — no config beats the control. Record:
  [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](../tests/eval/baselines/chunking_sweep_public_2026-06-06.md).

### Phase 7: Gap Detection and Cartography

Goal: quantitative view of what the library knows vs what the field knows.
This is the distinguishing capability — most existing tools stop short of this.

- External literature API integration (Semantic Scholar, OpenAlex, arXiv)
- DOI lookup at ingest time → fetch authoritative metadata, references, 
  classifications
- Topic clustering across the library
- Field coverage estimation: "you have 47 of the top 100 papers in this cluster 
  by citation count"
- Frontier detection: recent papers heavily cited by your library's papers but 
  missing from your library
- "What to read next" recommendations, ranked
- Knowledge map dashboard: bubble plot or similar, clusters sized by coverage, 
  recency color-coded, gaps visible at a glance
- **Feature 6 — Self-organizing wiki / synthesis layer.** A derived markdown
  layer over the corpus: per-topic notes (summary + tags + `[[links]]` +
  citations) distilled from retrieval, regenerable, sidecar `data/wiki/`.
  The Karpathy LLM-wiki pattern (proven in the cross-project atlas) applied
  *on top of* RAG, not as a replacement. Its real payoff here is making gap
  detection computable — a note with thin citations and no links is a
  structural gap signal (reuses `provenance.py` confidence heuristics), not
  just an LLM opinion. Living precursor to the Phase 9 review generator. See
  roadmap Feature 6.

- **Feature 7 — Cross-document concept graph.** ✅ **Shipped (PR 16, 2026-06-15;
  7a–7c). Validated free on local Ollama (`llama3.1:8b`).** A concept/entity graph
  *across* the library — nodes = concepts, edges = relations, Louvain communities,
  high-degree "god nodes", graph-gap signals — as a sidecar `data/graph/graph.json`
  + a per-doc extraction cache. `src/doc_assistant/concept_graph.py` (pure core +
  impure boundary) + `scripts/build_concept_graph.py`. Where Feature 6 clusters
  *documents*, this relates the *concepts inside* them; it is the threshold-free
  clustering primitive Feature 6 should re-point at (see Deferred Improvements).
  Four decisions, recorded here as ADRs:
  - **ADR — extraction defaults to LOCAL Ollama explicitly, not `LLM_PROVIDER`.**
    Extraction runs an LLM over *every* document — the exact batch op that silently
    bills Anthropic if it inherits the analysis-provider default (`LLM_MODE=api` →
    `LLM_PROVIDER=anthropic`, which `WIKI_/REVIEWER_/JUDGE_*` all inherit — the
    footgun that burned credits on 2026-06-15). So `CONCEPT_GRAPH_LLM_PROVIDER`
    defaults to `"ollama"` *directly*; `--provider anthropic` is opt-in, API-key-
    gated, with a paid-run warning. Originally a deliberate departure from the
    wiki's inherit-the-default convention; **generalised on 2026-06-15** — the wiki
    now shares this default-local stance and both route their paid opt-in through
    one shared guard. See Core Decisions → "Enrichment provider-intent guard".
  - **ADR — Louvain (networkx-native) over Leiden (igraph/leidenalg/graspologic)
    for v1.** Same modularity-optimisation family and the threshold-free clustering
    the corpus needs (mean-pooled doc vectors saturate at ~0.9 cosine within a
    domain — the Feature 6 finding); `networkx>=3.2` ships `louvain_communities`
    pure-Python, with zero compiled-extension risk on the Windows boxes (igraph
    wheels are historically flaky here). Leiden is a drop-in upgrade if a clean
    wheel proves reliable. User-confirmed choice.
  - **ADR — edges integrity-tagged structurally, never self-reported confidence**
    (reuses the `provenance`/`reviewer` discipline; `INTEGRITY_TAGS` will be reused
    by 7d). `EXTRACTED` = stated + relation-phrase-consistent across docs;
    `AMBIGUOUS` = same pair, ≥2 conflicting relation phrases; `INFERRED` = no stated
    triple but co-occurs in ≥ `CONCEPT_GRAPH_MIN_COOCCURRENCE` docs.
  - **ADR — sidecar JSON, NOT a graph DB.** Roadmap-explicit (Stonebraker: a graph
    DBMS is rarely the performant choice; this is build-time structure, not a query
    server). Idempotent per-doc cache keyed by `doc_hash` → content change
    re-extracts; re-runs rebuild the graph with zero LLM calls. Enrichment-Layer
    Pattern — never mutates the chunk store. See roadmap Feature 7.

- **Feature 7 — design notes salvaged from the parallel `feature/pr16-concept-graph`
  branch (deleted 2026-06-15, tip `cd6cec0`).** A second, fuller PR-16 attempt was
  built on a branch the same day (1445-LOC module, spec, 4 ADRs) but ran extraction
  on the **Anthropic API by default** (`GRAPH_LLM_PROVIDER` inheriting `LLM_PROVIDER`)
  — the cause of the afternoon credit burn. We kept main's credit-safe-by-default
  version and salvaged the worthwhile pieces; the branch was then removed. Reasoning
  worth keeping:
  - **Louvain is not just preferred over Leiden — Leiden is unavailable here.** The
    branch tested both against the live venv: `networkx.algorithms.community.leiden_communities`
    **exists but raises `NotImplementedError`** (a backend-dispatch stub, no backend
    installed), and `graspologic` hard-pins `numpy<2.0`, conflicting with the locked
    numpy-2 stack. So `louvain_communities` is the only working modularity detector
    without a native igraph build. (Strengthens the Louvain ADR above.)
  - **Salvaged into main now (the "quality bundle"):** `snap_relation` (closed
    `RELATION_VERBS` vocab → fewer spurious `AMBIGUOUS`; **tradeoff:** the narrow vocab
    collapses most edges to `related_to`, flattening relation labels — widen the vocab
    or add a raw-phrase field if richer labels are wanted), a truncation-tolerant JSON
    salvage parser (`_scan_objects`/`_salvage_array` — recovers complete leading
    elements from a cut-off local-model completion), and membership-hashed community
    keys (`community_id_for`, drift-stable, mirrors `wiki.topic_id_for`).
  - **Deferred salvage candidate — semantic node-merge (branch ADR-3):** cosine-merge
    concept *labels* via the BGE `embeddings.py` factory above ~0.88 would fold
    "transformer" / "transformer architecture" into one node — the fix for main's
    isolated/noisy-concept tail (86 degree-0 nodes on the public run). Not yet ported.
  - **Chunk-id instability (branch ADR-4), important for 7d:** `ingest.py` adds Chroma
    chunks with **no `ids=`**, so the LangChain wrapper regenerates UUIDs on every
    re-ingest — there is no stable chunk id. 7d back-pointers must use a deterministic
    composite key (`{document_id}:p{parent_index}`), not a Chroma id.

- **Feature 7d — Knowledge-currency / claim-corroboration layer.** ✅ **Shipped
  (2026-06-17), validated free on local Ollama.** Spec: `docs/specs/feature-7d-
  knowledge-currency.md`. The epistemic layer on top of Feature 7: each concept
  *claim* gets a **structural** corroboration weight (supporting vs disputing
  sources, and whether disputes are *newer*), projected onto the chunks that mention
  it. New `epistemics.py` + `scripts/compute_epistemics.py` + a regenerable
  `chunk_epistemics` sidecar table; `concept_graph.py` extended with relation
  **polarity** + per-doc **`EdgeSupport` (doc_id, polarity, year)** records +
  `compute_node_weights`; `reviewer.py` gained the `contested_evidence` failure tag.
  Decisions, recorded as ADRs:
  - **ADR — age is never an input (Decision 1).** Currency *emerges* from polarity
    over time: a claim is `superseded_trend` only when *disputing* sources are newer
    than supporting ones. An old, uncontradicted claim keeps full weight (`stable`).
    `year` is used only for *relative* ordering, never as an absolute age penalty.
  - **ADR — the unique-source rule (Decision 4), the regression that matters.** A
    concept that is the corpus's *only* source on its topic is `coverage="unique"` →
    **neutral, never down-weighted / never marked**. Contested vs unique is
    distinguished purely by whether *disputing* sources exist. Guard-tested on toy
    graphs **and** held on the real run (169/198 nodes `unique`).
  - **ADR — structural attribution for chunk back-pointers, not per-claim LLM spans.**
    The spec wanted `source_chunk_ids` per claim. The honest, no-LLM way: a concept is
    "in" a chunk if its canonical label occurs in the chunk text (word-boundary
    match). This gives real chunk-level granularity (a parent can hold a contested
    claim and three clean ones) without trusting the model to cite spans — consistent
    with "structural weights only". Per-claim character spans are the deferred refinement.
  - **ADR — sidecar keyed by `{document_id}:{chunk_index}` (baseline store).** The
    composite is stable where Chroma's auto-UUIDs are not (branch ADR-4). The
    parent-child store uses `(parent_index, child_index)` instead and is left for the
    live-surfacing follow-up. Whole table replaced per run (regenerable).
  - **Polarity extraction is local-model-noisy (validation finding).** Re-extracting
    the public 10-paper corpus on `llama3.1:8b` then projecting gave 26 contested
    nodes / 308 marked chunks / **0 superseded_trend** — plausibly inflated contested
    signal (the 8B model labels "differs-from / compares-to" as `contradicts`), and no
    clean newer-supersedes-older ordering surfaced. The structural machinery is
    correct; signal *quality* tracks extraction quality (same lesson as PR 16's noisy
    local extraction). A tighter polarity prompt or a stronger model sharpens it.
  - **Deferred (own follow-ups, documented):** (1) **live answer-time surfacing** —
    inject `contested`/`superseded_trend` into the synthesis evidence layer; needs a
    stable chunk key plumbed into `RetrievedChunk` (retrieval uses the parent-child
    store, which carries `parent_index`/`child_index`, not `chunk_index`). Synthesis is
    deliberately *untouched* this PR, so the eval is byte-identical with markers off.
    The marker-derivation join (`epistemics.markers_for_chunk_keys` /
    `load_epistemics_index`) + a mocked guard test ship now. (2) **`query_router.py`
    seam** (Decision 8) — the local/global retrieval-mode split; a distinct retrieval
    concern the spec only measures "when Phase 7 lands." (3) per-claim spans, external
    citation velocity, adjudication-log trust (v2) — all out of scope per the spec.

### Phase 8: UI Polish

Goal: design pass on everything built so far.

- Visual design refinement
- Animations and micro-interactions
- Better empty states, loading states, error states
- Demo recording
- Onboarding flow for new users
- Polish on the adjudication UI from Chunk 2a if it shipped rough.

Note: content-only hashing completed in Phase 3 (was moved from Phase 8 to Phase 3 completion gate).

### Phase 9: Literature Review Generation

Goal: synthesize a structured review across documents in the library on a 
chosen topic. The endgame feature.

Requires everything from Phases 2–7 to work well:
- Reliable multi-document retrieval (Phase 2)
- Document and section structure as data (Phase 3)
- Citation graph for grounding (Phase 4)
- Domain-aware retrieval + eval-backed quality (Phase 5)
- Structured figures + dual-layer interpretation (Phase 6)
- Coverage awareness so the system knows what it's missing (Phase 7)

Capabilities:
- Per-paper synthesis: what does each paper contribute on the topic?
- Cross-paper synthesis: agreements, disagreements, evolution of ideas
- Citation-grounded claims: every assertion linked to specific passages
- Structural output: introduction, main themes, methodological comparison, 
  open questions, references
- Configurable scope: by folder, by topic, by date range, by document set
- Export: markdown, PDF, or BibTeX-aware Word document
- **Integrity Chunk 3** — PRISMA-trAIce export. Structured methodology 
  disclosure alongside the generated review, pulled from `answer_records` 
  and the adjudication log.

---

## What I'd Position This As

> A personal research workspace that combines local-first RAG with explicit 
> modeling of the literature graph, designed to help a researcher identify 
> what they know, what they don't, and what to read next. Unlike general-
> purpose RAG tools, it treats the document collection as a first-class 
> object to curate and analyze, not just a corpus to search.

Keywords: *personal*, *local-first*, *literature graph*, *knowledge gaps*, *what to read next*.

---

## Forward-looking considerations (added 2026-05-28)

The architecture is well-positioned for the current ~50-doc personal-library 
scale and handles ~500 docs without changes. Several trajectory questions are 
worth keeping in view as the project grows; none requires action now, but 
none should be silently undermined by short-term decisions either.

### Eval set as a living artefact

`tests/eval/cases.yaml` ships with 35 cases. The expected pattern: every real 
question the user asks the system in earnest is a candidate eval case. The 
file should grow with use — 100 cases by Phase 6, 200+ by Phase 9. Each new 
case is a regression test against future PRs. The harness (`scripts/run_eval.py` 
with `--repeat`) is built to scale linearly with the case count.

**Risk:** authored expected_answers go stale as the corpus changes or as 
the user's understanding sharpens. Periodic review needed — the 
`metadata.author_verified: false` flag is the marker.

### Cost discipline

Once `/chat` is in daily use, Haiku token spend matters. The provenance 
card (PR 5, Integrity Chunk 1) is the data primitive that makes cost 
analysis evidence-based — every answer carries its token cost. Options 
that become available once the records exist:

1. **Answer cache** keyed by `(query_hash, retrieved_chunk_ids, model, prompt_version)`. 
   Same question with the same context returns the same answer for free.
2. **Tiered LLM routing**: Haiku for simple lookups, Sonnet for hard 
   synthesis, Ollama for routine queries. The reviewer agent in 
   Chunk 2b will need this anyway (cheap model judges expensive model's 
   output).
3. **Per-session cost budgets**: warn at $X, hard-stop at $Y.

None of these is built yet; the schema for PR 5 should make all three 
feasible later without re-instrumenting.

### Database scale story

| Library size | What works | What needs work |
|---|---|---|
| **~50 docs** (now) | Everything | — |
| **~500 docs** | Chunk retrieval, BM25 in-memory, O(N²) doc similarity | None |
| **~2-3k docs** | Chunk retrieval still fine (Chroma) | BM25 rebuild slow; O(N²) similarity edges (~5M pairs) becoming uncomfortable |
| **~5k+ docs** | Chroma + reranker still scale | BM25 needs persistence; similarity needs ANN index (FAISS, hnswlib); SQLite joins start to feel slow |
| **~50k+ docs** | None of the current architecture as-is | Need Postgres or sharded SQLite; persistent BM25 (Tantivy / Lucene); ANN; possibly distributed |

**Architectural invariants to preserve so the scale-up isn't a rewrite:**

- `Document.id` is a UUID, not an auto-increment — works across multiple DBs / shards.
- Sidecar tables (`citations`, `doc_similarities`, soon `answer_records`) decouple from the chunk store — can be migrated independently.
- All retrieval goes through `pipeline.retrieve()` — single chokepoint to swap from in-memory BM25 to a persistent index.
- Embedding factory routes model choice through one function — adding ANN-backed collections is a Chroma-config change, not a pipeline rewrite.

**What would actually trigger a database redesign:** multi-user (next section).

### UI ceiling

Chainlit covers chat well. It will hit limits at:

- **Graph visualisation** (Phase 6+) — citation subgraphs beyond ~25 nodes are already at the inline-Mermaid limit.
- **Adjudication UI** for Chunk 2a (Phase 6) — per-claim accept/reject/edit needs richer interaction than Chainlit's element API supports cleanly.
- **Library browser** with filters, sorting, bulk actions.

Decision deferred to Phase 8. Three plausible directions:

1. **Reflex** — Python full-stack, lowest migration cost.
2. **FastAPI + custom React frontend** — most flexible, most work.
3. **Streamlit** — quickest to prototype, weakest at custom interactions.

The data layer (`library.py`, `commands.py`, `pipeline.py`) is UI-agnostic precisely so this swap doesn't ripple. Keep it that way.

### Multi-user collaboration

Currently single-user assumptions are baked in only minimally:
- One SQLite DB
- No auth
- One `data/` directory
- No notion of "who created this tag / who adjudicated this answer"

Going multi-user requires:
1. **Auth** (out of scope until needed — could be a Reflex/FastAPI add-on, not a from-scratch design).
2. **Per-user scoping** on Documents, Folders, Tags, AnswerRecords (the things that *belong* to a user) — vs *shared* Documents (a lab corpus).
3. **DB redesign**: SQLite is single-writer; multi-user concurrent edits need either WAL-mode tuning, server-mode (`sqld`), or a switch to Postgres.
4. **`created_by` and `modified_by` columns** on every mutable table.

**Don't bake in single-user assumptions you can't reverse.** The current 
schema mostly avoids this — UUIDs everywhere, sidecar tables. The 
`/folders` and `/tags` features (when added) need to think about per-user 
vs shared from day one.

**When to revisit:** the moment a workflow emerges that involves >1 person 
on the same corpus. Until then, the cost of designing for it now exceeds 
the cost of deferring.

---

## Open Questions

Decisions I haven't made yet:

- **UI framework for Phases 4-7.** Chainlit handles chat well but is awkward 
  for graph visualization and library browsers. Possible migrations: Reflex, 
  FastAPI + custom frontend, Streamlit. Decision point: when graph 
  visualization is being built and Chainlit's element API hits its limits.

- **Embedding model upgrade path.** current choice: `bge-base-en-v1.5`
    I should test new models.

- **Multi-user support.** Currently single-user. Multi-user suppport in the future ?
    Redesign of db architecture needed. 

- **Citation extraction quality vs. external API quality.** TBD — revisit 
  during Phase 7 (Gap Detection) when Semantic Scholar / OpenAlex / arXiv 
  integrations land and can provide a ground-truth comparison.

- **Tier-2 LLM citation fallback.** Deferred until corpus grows or 
  no-section docs become problematic.

- **MCP server interface (external tool access).** Open / unscheduled.
  Expose the RAG pipeline as an MCP server so external MCP hosts — Claude
  Desktop, claude.ai connectors — can call the local library as a tool (e.g.
  a `search_library` / `ask` tool), letting a paid Claude subscription query
  the user's own documents. Architecturally low-friction: another **thin
  entrypoint** (`apps/mcp_server.py`) over `pipeline.py`, no core changes,
  consistent with the `apps/` boundary rule. Open nuance: Claude Desktop can
  talk to a local **stdio** MCP server directly, but claude.ai **connectors**
  need a reachable HTTP endpoint + auth — a real consideration for a
  local-first app. Not yet scoped to a phase.

---

## Deferred Improvements

### Feature 6 wiki clustering — absolute-cosine threshold is the wrong primitive (use relative / community clustering)

**Status (2026-06-14):** Feature 6 (PR 13) clusters the library via connected
components over `DocSimilarity` edges at/above `WIKI_MIN_SIMILARITY` (absolute
cosine). Building it surfaced that this primitive is **ill-posed for same-domain
corpora**: mean-pooled BGE doc vectors of the public 10-paper RAG corpus all sit
at **~0.88–0.96 cosine**, so the original 0.55 default collapsed all 10 papers
into one blob, and meaningful sub-topics only appeared at ~0.93–0.95 — a
threshold that is *corpus-specific*, not a universal constant. Bumped the default
to 0.90 and documented `--min-similarity` tuning, but a fixed absolute threshold
can't generalize.

**The proper fix** is *relative* / threshold-free clustering: graph community
detection (modularity) so the cut adapts to the corpus's own similarity
distribution instead of an absolute number. **Feature 7 (PR 16) shipped this**
(2026-06-15) — `concept_graph.analyze_graph` runs **Louvain** community detection
(networkx-native; Leiden/graspologic deferred for Windows-wheel safety — see the
Feature 7 ADRs above), and `concept_graph.doc_clusters_from_graph(graph,
extractions)` exposes the threshold-free, document-level grouping (docs grouped by
the concept-community they most belong to) as a **drop-in replacement** for
`wiki.cluster_documents`. **Remaining step (its own follow-up PR, per one-PR-per-
session):** re-point `build_wiki` at `doc_clusters_from_graph` so the wiki's
`[[links]]` come from concept communities rather than the absolute cosine
threshold. Until that lands, `WIKI_MIN_SIMILARITY` stays a documented, tunable v1
and the wiki is most useful on a *multi-domain* library (where doc vectors actually
separate) — on a tight single-domain corpus it produces one big topic plus outliers.

**Atlas:** recorded as a cross-project lesson (mean-pooled doc-vector similarity
saturates within a domain; cluster relatively, not by absolute cosine).

**Resolution path (2026-06-15, PR 16 spec).** Feature 7 supplies the relative /
community clustering this note asked for, via **NetworkX Louvain** (not Leiden —
`leiden_communities` is a non-functional dispatch stub and graspologic conflicts
with numpy 2; see ADR-1 above). The re-point lands **read-only and inert by
default**: `wiki.load_communities()` + a guarded branch in `_assemble_notes`
behind `WIKI_USE_CONCEPT_COMMUNITIES=false` and a `graph.json` freshness check,
with `cluster_documents()` (the absolute-cosine union-find) kept as the live
fallback so shipped 6a–6d is byte-identical. The default flips to concept
communities in a later checkpoint, once the re-cluster is validated on data.

**✅ Landed (2026-06-17, its own PR).** PR 16 *deferred* the re-point (it kept
main's credit-safe concept graph and salvaged a quality bundle; the gated wiki
re-point was explicitly held). This PR ships it as designed above:
- `concept_graph.graph_from_dict` — the missing inverse of `graph_to_dict`, so the
  wiki can reload the persisted `graph.json` (round-trip tested).
- `wiki.load_communities(docs, *, graph_dir=None)` — reads `data/graph/graph.json`
  (Louvain communities) + the per-doc extraction cache, **freshness-checks by
  `doc_hash`** (every non-archived doc must have a current cached extraction; a
  content change since the last `build_concept_graph --apply` re-keys the cache and
  reads as stale), realigns cached `doc_id`s to live PKs, and returns
  `concept_graph.doc_clusters_from_graph` — the threshold-free document grouping.
  Returns `None` (→ caller falls back to cosine) when the sidecar is absent,
  unreadable, or stale. Read-only: never extracts, never calls an LLM, never writes.
- `build_wiki(..., use_concept_communities=WIKI_USE_CONCEPT_COMMUNITIES)` → one
  branch in `_assemble_notes` swaps the clusters; `WikiBuildResult.clustering`
  records which primitive ran. `scripts/build_wiki.py` exposes
  `--use-concept-communities/--no-…` and reports the fallback.
- **Validated FREE** (two dry runs, zero LLM cost) on the real 10-paper sidecar:
  cosine@0.90 → **2 topics** (a 9-paper blob + 1 outlier — the saturation failure);
  concept communities → **9 topics** (threshold-free, each paper to its dominant
  community). Default-off run still reports `cosine-threshold` (inert).
- **Deferred (next refinement):** `[[links]]` still derive from `DocSimilarity`
  edges crossing the chosen clusters (`compute_links`), so on a saturated
  same-domain corpus the cross-cluster sim graph is dense — the concept run above
  shows ~8 links per small cluster. The honest fix is to derive links from
  *inter-community concept edges* in `graph.json` rather than doc-cosine edges; left
  out to keep this PR a minimal, inert-by-default clustering swap (one-PR rule).
  Flipping `WIKI_USE_CONCEPT_COMMUNITIES` to default-`true` is also deferred until
  the re-cluster + the links refinement are validated together.

### Chunk 2c self-improvement loop — built, but its payoff is scale-gated (revisit on server deploy or local-model oddities)

**Status (2026-06-14):** the reviewer aggregation / bias-vs-fault loop (PR 12) is
*implemented and tested*, but an honest cost/benefit review concluded it is
**over-engineered for a solo, local, single-user tool** and is parked here as a
deferred-*payoff* item — the code exists; the conditions that make it useful do
not yet.

**Why it under-delivers at current scale.** The reviewer only runs on
already-flagged answers in the Chainlit app, so `failure_tag` verdicts accrue
slowly; the `MIN_FAILURE_TAG_COUNT` / `MIN_FAILURE_TAG_DOCS` gate (10 across 5
distinct answers) then correctly reports **"insufficient evidence"** until a lot
of real usage piles up. So it's a well-built instrument pointed at a near-empty
table. At one-user scale, `SELECT failure_tag, COUNT(*) … GROUP BY failure_tag` +
eyeballs covers most of the practical value; the elaborate aggregation earns its
keep only with volume. The cheap, unambiguously-worth-it part was the
`failure_tag` *column* itself (makes faults countable, forward-compatible); the
machinery on top is architecture/discipline demonstration more than daily-use ROI.

**Known validity caveat of the bias-vs-fault anchor.** It assumes the pipeline's
*generated* answers on the golden set are good (so a reviewer flag there = a false
positive = reviewer bias). What's actually verified is the *reference* answer, not
the generated one — the pipeline can genuinely produce a weak answer on a golden
question. So the anchor *nudges* the bias-vs-fault split; it doesn't prove it.

**When to revisit (the triggers that flip the cost/benefit):**
- **Server / multi-user deployment.** Real traffic from multiple users makes
  reviewer verdicts accumulate fast enough to clear the min-N gate, at which point
  the aggregation + per-`prompt_version` slicing + the eval-anchored bias-vs-fault
  report start producing actionable signal instead of "insufficient evidence."
  This is also when closing the loop operationally is worth it: a scheduled
  `reviewer_report` run (the `/schedule` cron path) and surfacing the top
  above-gate fault — the deliberately-open action layer.
- **"Strange things" when running local models.** If the reviewer/judge or the
  generator is moved to local Ollama (the local-first end goal) and answers start
  behaving oddly, `scripts/reviewer_report.py` + `--anchor` become a real
  diagnostic: the bias-vs-fault split is exactly the tool for "is my local
  *reviewer* mis-scoring, or is my local *generator* actually worse?" — provided
  the anchor stays on the **pinned** reference instrument, not the same local model
  under suspicion (otherwise the anchor drifts with the thing it's measuring).

**MLOps framing (for context).** This implements the *capture → evaluate →
aggregate* spine of an LLM-observability feedback loop and deliberately stops
before the operational half (no orchestration / dashboards / alerting / automated
remediation — "instrumentation, not action"). Roughly a third of a real MLOps
feedback stack by surface area; its above-average trait is the *discipline*
(gated counts with denominators, "anchor or don't claim") rather than the tooling.

### CI coverage floor — raise from 40% over time

Current `--cov-fail-under=40` in `.github/workflows/ci.yml`. Was 70% pre-Phase-3,
lowered to 45% then 40% as `pipeline.py` and `ingest.py` are I/O-heavy and
hard to test without a real Chroma/SQLite/embeddings stack. Phase 4 added
45 tests against pure-logic citations + metadata modules.

Path to raise:
- Each new phase: add at least one integration test that hits a real (temp)
  DB or Chroma store. Phase 4 added one (`tests/integration/test_citation_pipeline.py`).
- Phase 5 work on retrieval-side gap detection — add tests that exercise
  `pipeline.retrieve` with a small frozen Chroma fixture.
- Raise floor in 5%-point increments after each phase ships, never lower it.

Target: 60% by end of Phase 6. Not a strict deadline — coverage is a lagging
indicator of "did we write the tests we should have", not a goal in itself.

### Better document pre-processing

Current extraction (PyMuPDF4LLM + Marker for scanned) produces usable but 
imperfect markdown. Specific issues observed:

- Mid-word breaks from PDF columns ("M isregistration", "W ith")
- Figure captions captured as headings
- Equations rendered inconsistently
- Tables sometimes lose structure
- Reference list parsing varies by paper format

A dedicated pre-processing pass — between extraction and chunking — could 
clean these systematically. Possible techniques: regex-based artifact repair, 
LLM-based "did this extract correctly?" check, format-specific cleaners.

Cost: might not be worth the cost but worth trying. LLM-based could be costly.

### ~~pdfplumber for table extraction~~ — promoted to Phase 6 (Feature 4a)

Was deferred; now scheduled as Feature 4a under the Phase 6 enrichment layer.
See `docs/doc-assistant-roadmap.md` and the Phase 6 entry above for the
implementation plan.

### Expose remaining retrieval knobs — toward a config-complete RAG sandbox

Most of the pipeline is already config-swappable (embedder, chunk sizes, `TOP_K`,
parent-child, multi-query, PDF extractor, LLM/reviewer/judge backends) and the eval
harness can measure the effect — so the project already *functions as* a local RAG
sandbox. Three knobs are still hardcoded, which is exactly what keeps that
description qualified rather than absolute:

1. **BM25 / vector weights** — literal `[0.4, 0.6]` in `pipeline.py` (`EnsembleRetriever`).
   Expose as `BM25_WEIGHT` / `VECTOR_WEIGHT` (or a single `HYBRID_WEIGHTS`) env knob.
2. **Reranker** — `BAAI/bge-reranker-base` hardcoded in `pipeline.py`, always-on. Give
   it a registry + factory (mirror `embeddings.py`) and a `USE_RERANKER` toggle.
3. **General config sweep** — only `sweep_chunking.py` exists. Generalize it into a
   grid runner over any of the above (TOP_K, weights, reranker, toggles), reusing the
   eval harness, so "vary X and measure" isn't manual for everything but chunking.

Each is a small extension of an existing pattern (the embedder registry and the chunk
sweep already prove it). Sequence them behind the higher-priority Phase 6 nodes;
landing all three makes the "local RAG sandbox" claim in the README unqualified.
Once exposed, these knobs feed the **Phase 8 settings page** (user-facing sandbox, with
the benchmarked default pre-selected — see `doc-assistant-roadmap.md` → Phase 8).
Noted 2026-06-04.

### SPECTER2 for paper-level similarity (gated on use case)

Phase 5 / Feature 3 measurement showed SPECTER2 loses to bge-base on
chunk-level QA retrieval — a training-task mismatch (SPECTER2 was
trained for paper-level citation prediction over abstracts; we feed it
chunks of methods/results text). SPECTER2 stays registered but unused
by default.

**However**, paper-level similarity (the `/similar` task) is a useful
research feature — finding related papers to one you're reading is
exactly what SPECTER2 was built for. Implementation sketch when the
workflow becomes worth it:

1. Add an abstract extractor to `metadata_extractor.py` (regex/heuristic
   over the markdown cache; LLM fallback for failures).
2. Extend `doc_vectors.py` with a per-model strategy: BGE mean-pools
   chunks (current); SPECTER2 embeds `title + abstract` directly.
3. Run `compute_doc_vectors --apply --force` with
   `EMBEDDING_MODEL=specter2`. SPECTER2-similarity edges live alongside
   the BGE ones thanks to the existing `embedding_model` PK column.
4. `/similar` chooses which set to surface (or shows both side-by-side
   for comparison).

**Gate:** ship only when paper-level `/similar` is a regular workflow
worth optimising. Effort: 1 evening + measurement to verify
SPECTER2-on-abstracts beats BGE-mean-pooled-chunks for this task.
Currently no evidence either way.

### Domain-aware tokenization / keyword embedding

Standard embedding models treat all tokens equivalently. Scientific text has 
*disproportionately important keywords* — drug names, protein names, technique 
names, mathematical operators — that carry most of the meaning of a sentence.

A potential improvement: identify a vocabulary of high-signal scientific terms 
(via TF-IDF on the corpus, or pre-built domain dictionaries like UMLS for 
biomedical), then either:

1. Boost their weight in BM25 retrieval
2. Generate auxiliary embeddings that emphasize these terms
3. Use them as explicit metadata for hybrid retrieval

Cost: medium. Value: probably real for technical scientific corpora.

### ~~Known data integrity edge cases~~ — resolved

Resolved by content-only hashing. Documents no longer produce new hashes
when moved or re-extracted. The `dedupe_documents.py` script is obsolete;
the migration script `scripts/migrate_to_content_hash.py` handles any
remaining inconsistencies from the old scheme.