# doc_assistant — roadmap additions

Engineering quality additions on top of the existing phase plan in `docs/decisions.md`. Purpose: domain-aware retrieval, reproducible evaluation, structured handling of figures and tables, and an explicit research-integrity layer that records how each answer was produced.

This document is the **source of intent**. `docs/decisions.md` records the locked design choices; `CLAUDE.md` points at both. The evaluation strategy, scorer limitations, and the verified-10 benchmark rule live in [`tests/eval/TESTING.md`](../tests/eval/TESTING.md).

The asset these additions build is the *testing and observability architecture* — a reusable harness with clearly documented scorers and known limitations — not a set of impressive numbers. Every published number comes from the verified 10-case public corpus, and only that set. The app already works well enough in daily use; the research-integrity layer is the instrument that exposes where it actually fails, so the system's real ceiling only becomes visible once that layer is in place (Chunks 2a→2c). The features below are sequenced to build that instrument before leaning on what it measures.

---

## Goals

1. Make the embedding layer swappable, with measured comparisons across models. (Per-project *routing* is deferred until a model beats `bge-base` on an identifiable sub-corpus — Feature 3 measured no such win yet; the factory stays, the routing layer waits.)
2. Build a reproducible eval harness inside the project, designed to be extractable later.
3. Promote figures and tables from "lossy text artifacts" to first-class structured content.
4. Add a research-integrity layer: every answer carries a provenance record; synthesis is split into evidence and interpretation layers; an LLM reviewer scores each interpretation against a rubric.
5. Position the project against published standards (PRISMA-trAIce, AI Usage Cards, BE WISE) without binding to any single vendor framework.
6. Close the self-improvement loop: aggregate reviewer verdicts over time, distinguish reviewer bias from systemic faults by anchoring against the verified eval set, and surface recurring failure patterns as actionable fixes — but only once they clear a minimum-N gate (`MIN_FAILURE_TAG_COUNT`/`MIN_FAILURE_TAG_DOCS`, tunable), with counts always shown against their denominator. Below the gate: instrumentation, not action.
7. Add a self-organizing markdown "wiki" synthesis layer over the corpus — distilled, linked, cited topic notes that make knowledge gaps computable. Feeds Phase 7 gap detection and Phase 9 review generation. (Inspired by the Karpathy LLM-wiki pattern proven out in the cross-project atlas; here it sits *on top of* RAG, not as a replacement.)

---

## Phase ordering (renumbered)

The new work integrates into the existing phase plan as follows. The full phase list lives in `docs/decisions.md`; this table is for reference only.

| Phase | Content | Source of features |
|-------|---------|--------------------|
| 4 | Citation Graph — close out | Existing roadmap |
| 5 | Embedding & Eval Foundation | Features 1, 2, 3 + Integrity Chunk 1 |
| 6 | Per-project routing + Figures & Tables + Dual-layer interpretation + self-improvement loop | Features 1b, 4a, 4b, 4c + Integrity Chunks 2a, 2b, 2c + chunking sweep |
| 7 | Gap Detection (was Phase 5) | Existing roadmap + Feature 6 (wiki/synthesis layer) |
| 8 | UI Polish (was Phase 6) | Existing roadmap |
| 9 | Literature Review Generation (was Phase 7) | Existing roadmap + Integrity Chunk 3 |
| — | Extract eval harness to standalone repo (Feature 5) | Post-Phase 5, no phase number |

---

## Phase 4 — close out

What is left:

- Apply metadata backfill and citation extraction on the local DB (CLI runners exist; sandbox cannot write).
- ✅ Mean-pool doc-level vectors → similarity edges — **done** (`src/doc_assistant/doc_vectors.py` + `scripts/compute_doc_vectors.py` + `library.similar_docs`, exposed as the `/similar` command).
- LNCS colon-separator format and multi-column extraction artifacts are known tier-1 weaknesses; cosmetic, deferred.

Effort to close: 1 evening + a 15-minute CLI run.

---

## Phase 5 — Embedding & Eval Foundation

### Feature 1 — Config-driven embedding layer

Make the embedding model swappable via env config instead of hardcoded.

**Why.** Different content domains benefit from different embeddings. Hardcoding `bge-base` is a missed opportunity for academic papers, which are doc_assistant's main use case.

**Implementation.**

- Add `EMBEDDING_MODEL` env var. Initial options:
  - `bge-base` (current default, general-purpose)
  - `specter2` (academic papers — trained on citation graphs)
- Embedding factory function that returns the right model loader.
- Vector store handling — collections are dimension-specific. **Option A (recommended):** one Chroma collection per embedding model. Fast switching, more disk. **Option B:** single collection, force re-ingest on model change (`--reset-index` flag).
- Document the trade-off in `decisions.md`.

**Effort:** 1–2 evenings.

**Generation side — LLM provider protocol (locally runnable).** Feature 1 also covers the generation layer, the sibling of the embedding factory. A normalized `LLMClient.complete(messages, *, temperature, max_tokens) -> str` protocol (`src/doc_assistant/llm.py`) with `AnthropicClient` + `OllamaClient` adapters backs the reviewer and the eval judge; `pipeline._build_llm()` becomes `LLM_PROVIDER`/`LLM_MODEL`-driven while staying a streaming LangChain model. **Local-first is the end goal, hybrid today.** The *generator* targets fully local (Ollama analysis + chat, no API key). The *judge and reviewer are pinned instruments* — fixed, version-recorded reference model by default; they move to local only after the local-judge calibration gate passes (`tests/eval/TESTING.md`). Feature 4c's VLM is API-only, outside the local path. Independent of all other features. Full spec + build node + guard test: `docs/specs/llm-provider-isolation.md`.

### Feature 2 — Eval harness module (v0)

A module inside doc_assistant that runs the RAG pipeline against a versioned eval set, scores outputs, and tracks results over time.

**Why.** A measurement harness puts the project's quality claims on a footing other than vibes. The existing experiment tables in `decisions.md` (TOP_K, parent-child, multi-query) prove the value of measurement; the harness formalizes it.

**Build it inside doc_assistant first, with extraction in mind.**

```
src/doc_assistant/eval/
  __init__.py
  runner.py        # generic: runs eval set against any callable
  scorers.py       # generic: deterministic, LLM-judge, embedding similarity
  store.py         # generic: DuckDB persistence
  report.py        # generic: diff between runs
  adapters.py      # doc_assistant-specific: wraps the RAG pipeline
```

**Rule:** everything except `adapters.py` imports nothing from `doc_assistant.*`. This lets Feature 5 happen cleanly.

**Scorer types to support:**

- Deterministic: exact match, regex, JSON schema validation, citation overlap, latency, token cost.
- LLM-as-judge: faithfulness, relevance, completeness on a 1–5 rubric.
- Embedding similarity: cosine similarity between output and reference answer.

**Effort:** 1 weekend for v0.

### Feature 3 — Golden eval set + measured comparison

A versioned eval set of 30–50 questions over a fixed paper corpus, plus a comparison run across embedding models.

**Why.** Without measurement, Features 1 and 2 are claims. With numbers, they are evidence.

**Implementation.**

- `tests/eval/cases.yaml` — 30–50 questions with expected answers or expected source documents.
- Run the eval harness across `bge-base` and `specter2`.
- Persist results in DuckDB. Generate a comparison table.
- Add a "Benchmark results" section to README with retrieval@k, faithfulness scores, and a short interpretation.

**Effort:** 1 day (writing the eval set is the bottleneck).

### Integrity Chunk 1 — Provenance card

Every answer carries a record of how it was produced.

**Source:** AI Usage Cards (arXiv 2303.03886).

**Deliverables:**

- New SQLite table `answer_records` — query, retrieved chunk IDs + scores, reranker scores, model name, prompt version, token cost, latency, timestamp.
- Rendered as a collapsible "provenance card" under each answer in the Chainlit UI.
- CLI export `/export-record <answer_id>` → JSON.

**Why short-term:** zero conceptual ambiguity, instruments everything downstream (the eval harness consumes the same record schema).

**Effort:** 1 evening. Hooks into existing `tracking.py`.

---

## Phase 6 — Per-project routing + Figures & Tables + Dual-layer interpretation

### Feature 1b — Per-project embedder routing

**Gate: deferred — no embedder beats `bge-base` on a sub-corpus yet.** Routing is only worth its machinery once some model wins somewhere; Feature 3 measured `bge-base` ahead of `specter2` on every retrieval-side signal, a training-task mismatch (SPECTER2 predicts paper-level citations over abstracts; doc_assistant does chunk-level QA over full text). Until a model beats `bge-base` on an identifiable sub-corpus by more than the confidence intervals overlap, routing is a switch with nothing to switch *to* — so it waits (re-run SPECTER2 at `--repeat 5` for a clean CI first). The Feature 1 factory is independent of this and stays; only the routing layer below waits. If no model ever wins on a sub-corpus, 1b leaves Phase 6 entirely. The design is kept here for when that evidence exists.

Let each user-defined project (Folder) declare its own embedding model. Ingest and retrieval automatically use the right embedder for the project a document belongs to.

**Why.** A personal library is rarely single-domain. Routing by project converts Feature 1's static config switch into a dynamic, per-corpus quality lever, backed by Feature 3's numbers.

**Implementation.**

- Add `embedding_model` column on the `Folder` table. Default = `EMBEDDING_MODEL` env value.
- Collection naming: `{folder_id}__{model_name}`. Dimension-segregated by model name; collisions impossible.
- Ingest path: reads `folder.embedding_model` → resolves via the Feature 1 factory → writes to the matching collection.
- Query path (scoped to folder): queries the folder's collection. Trivial.
- **Query path (unscoped / cross-folder):** query each affected collection independently → merge candidates → cross-encoder rerank resolves the mixed-space problem (the reranker is embedding-agnostic). Adds latency proportional to number of collections hit. This is the only non-trivial piece.
- BM25 is dimension-agnostic and stays global.
- UI: dropdown on folder create/edit to pick the embedder. Changing it triggers re-ingest of that folder only.

**Trade-offs.**

- Disk: linear in `(folders × models actually used)`. Bounded in practice.
- Cross-folder query latency rises with number of collections hit. Acceptable for a personal library (10s of folders, not 1000s).
- Eval becomes per-project, not global. Adds rows to the eval matrix but doesn't change the harness.

**What NOT to do.**

- No duplicate "global" BGE collection. Storage doubles, sync gets fragile.
- No changing a folder's embedder without re-ingest. Mixed embeddings in one collection are meaningless.

**Effort:** 2 evenings on top of Feature 1.

### Feature 4 — Figure & table understanding

Make figures, charts, and tables in scientific papers first-class, searchable content — not lossy text artifacts.

**Why.** A large share of the actual signal in a scientific paper lives in figures and tables. The current markdown pipeline drops figures entirely (PyMuPDF4LLM and Marker are configured with `write_images=False` by default) and flattens tables into ambiguous text.

**Architecture:** post-ingest enrichment layer, same pattern as `citations.py` and `metadata_extractor.py`. Primary ingest stays untouched. See "Enrichment-Layer Pattern" in `decisions.md`.

Three independent sub-features, ordered cheap → expensive:

#### 4a — table extraction pass ✅ shipped (Marker primary, 2026-06-02..06)

Spliced inline into the markdown cache.

> **Update:** the 2026-06-02 RTX engine eval selected **Marker** as the primary table
> engine, with **pdfplumber frozen as the no-dep fallback**. The caption-anchored,
> page-anchored inline splice (+ de-dup of pymupdf4llm's lossy twin) lives in
> [`tables_marker.py`](../src/doc_assistant/tables_marker.py), run out-of-process by
> [`scripts/extract_tables_marker.py`](../scripts/extract_tables_marker.py). The
> pdfplumber description below is the original plan / retained fallback. See
> `docs/figures-and-tables.md` and `docs/specs/feature-4a-marker-table-ingest.md`.

- Detect table regions after primary extraction, extract with pdfplumber, splice structured markdown tables back into the cached `.md` file.
- Mark with `<!-- table-extracted-by: pdfplumber -->` for traceability and re-runnability.
- Each table becomes a chunk with explicit row/column structure preserved in markdown.

**Why splice rather than sidecar:** tables are text-shaped (rows × cells); splicing back into the markdown is natural and preserves the "open the .md and see everything" property.

**Effort:** 1 evening.

#### 4b — Figure region detection + caption pairing (no LLM cost) — ✅ **shipped (PR 8, 2026-06-14)**

> **Spec:** [`docs/specs/feature-4b-figure-detection.md`](specs/feature-4b-figure-detection.md) is the code-level contract (files, contracts, guard tests, DoD). Two design calls made there: v1 derives region bboxes from **PyMuPDF geometry** (image blocks + drawing-bbox union), with **OpenCV refinement deferred** — `regions.py` already does the chart/photo/figure *discrimination* OpenCV was slated for, leaving only bbox extraction (ADR-1); and the `Figure` table is created by the additive `create_all` (no Alembic). Runs on **either machine** (no torch/Marker/GPU).

Sidecar manifest, not spliced.

- PyMuPDF already exposes image blocks (`block_type=1`); OpenCV refines region boundaries and detects chart-like regions (line detection, pixel variance, contour heuristics).
- Pair each figure region with its caption via nearest-text-block heuristic ("Figure N: …").
- Persist a `figures` table in SQLite: `{doc_id, page, bbox, caption, image_path, vlm_description, vlm_call_skipped_reason, extraction_method}`.
- Images written to `data/figures/{doc_hash}/page{N}_fig{M}.png`.
- Caption text remains in the markdown — figures are additive, not substituting.

**Why sidecar rather than splice:** figures are binary. Embedding base64 in markdown destroys the human-readable cache; placeholder strings without the image are noise. Sidecar preserves the markdown-as-universal-intermediate decision.

**Effort:** 1 weekend.

#### 4c — VLM figure description (gated) — ✅ **code shipped (PR 9, 2026-06-14)**

- For each figure in the manifest, call Claude vision with a schema-first prompt: `{figure_type, summary, key_quantities, axes, trend}` (Pydantic-validated, Anthropic tool-use).
- Embed `caption + VLM description` as a chunk linked back to the figure's bbox. Chunk metadata: `chunk_type='figure'`.
- **Gating:** skip figures already well-described by their caption (length threshold + caption-only embedding similarity check). Enforce `MAX_VLM_CALLS_PER_DOC` budget.

**Why gated:** VLM cost is the main risk on this layer.

**Effort:** 1 weekend.

**Eval hook.** Add a "figure retrieval" scorer to the eval harness: given a held-out caption, does the system retrieve the right figure?

**4a verification (decided 2026-06-02).** Two checks for table extraction: (1) **chosen** — a table-retrieval eval case (ask a question whose answer lives in a known table → assert the spliced table chunk is retrieved after re-ingest); built once the 4a engine is final. (2) **roadmap future** — a hand-verified gold table set + a deterministic extraction-fidelity scorer (cells/rows match the source PDF). Higher rigor, deferred; only worth it if the eval case shows table quality matters to answers. The caption-gating + the visual debug tool (`scripts/debug_tables.py`) are the precision instruments in the meantime.

**Trade-offs.**

- Markdown-as-universal-intermediate is preserved; the figure manifest is a sidecar, not a replacement.
- Caption-only chunks are a useful baseline if VLM is skipped (4b alone is shippable).

### Integrity Chunk 2a — Dual interpretation layer

Synthesis-mode answers return two layers:

1. **Evidence layer** — retrieved passages, faithfulness-scored, no synthesis.
2. **Interpretation layer** — the LLM's synthesis, clearly labeled, with retrieval-derived uncertainty markers (no self-reported confidence).

**Uncertainty markers** are observable and grounded in the retrieval signals that already exist:

- "High citation density" / "Low citation density"
- "All claims supported by ≥2 sources" / "Some claims rely on a single source"
- "Reranker scores tightly clustered" / "Reranker scores diverged — multiple plausible interpretations"

**Mode flag:** `SYNTHESIS_MODE = human | ai` (default `ai`).

- `human` mode = the BE WISE-influenced path. AI returns evidence only; interpretation is the user's responsibility.
- `ai` mode = dual-layer with reviewer scoring (Chunk 2b).

Both modes coexist in the same pipeline; the flag selects the output template. No branching codepaths.

**Adjudication.** Per-claim accept / reject / edit, persisted on the answer record (the Chunk 1 schema is extended, not replaced).

**Pre-interpretation checkpoint.** Retrieval coverage + citation density + faithfulness pre-score. Below threshold → warn user before generating interpretation.

**Effort:** 1 weekend. Touches `pipeline.py`, `prompts.py`, UI surface for adjudication.

> **Backlog (docs, deferred until this ships) — research-integrity pillar.** The integrity work is currently scattered across Chunks 1/2a/2b/3 and buried in `decisions.md`. Once Chunk 2a lands (evidence-vs-interpretation is the conceptual core), surface research integrity as a first-class, named pillar: a dedicated section in `README.md` and a short standalone `docs/research-integrity.md`, plus a line in `CLAUDE.md` status. **Do not write these docs before the implementation lands** — the docs should describe behaviour the system actually has, not aspiration. Intent (per user, 2026-06-01): integrity should be visible both in the docs *and* in how the AI behaves at answer time (the dual-layer split is that behaviour).

### Integrity Chunk 2b — Reviewer agent

Separate LLM call after the generator. Cheaper model (Haiku/Sonnet, not Opus). Returns structured JSON via Anthropic tool-use.

**Rubric:** faithfulness to retrieved passages, citation density, hedging adequacy, count of claims without sources.

**Output:** populates the uncertainty markers shown to the user; persisted on the answer record.

**No auto-retry in v1.** If the reviewer flags issues, surface them; the user decides whether to regenerate. Cost discipline.

**Reuse note.** The reviewer's rubric is identical in shape to a deterministic eval scorer. The same code paths the reviewer uses can be re-targeted at the eval harness — generator + frozen-prompt reviewer = LLM-as-judge scorer.

**Effort:** 1 evening on top of Chunk 2a.

### Engineering — chunking sweep (reopens Phase 2.4) ✅ infra shipped 2026-05-31

Chunk *sizes* (`parent 2000/200`, `child 400/50`, `baseline 1000/200`) were originally hardcoded in `ingest.py` and unmeasured — only the parent-child *structure* had been tested. They are now config-driven (`config.py`) and have been swept (see below).

**Shipped:** sizes are now config-driven (`PARENT/CHILD/BASELINE_CHUNK_SIZE` env vars; defaults behaviour-preserving), with `scripts/sweep_chunking.py` driving `ingest --rebuild` + `run_eval` across a size grid, tagging each run for comparison in `data/eval.duckdb`. Guard tests pin "config-driven, defaults unchanged."

**Measurement done (2026-06-06, RTX/GPU):** a 6-config parent/child sweep on the public corpus through the eval harness confirmed the defaults `parent 2000/200 · child 400/50` — no config beats the control. Record: [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](../tests/eval/baselines/chunking_sweep_public_2026-06-06.md). Defaults stand; no change needed. A later `CHUNK_STRATEGY` flag (semantic vs fixed) is still deferred.

**Repeatability (closed 2026-06-01):** the "add doc examples for test repeatability" TODO is addressed by a shareable **public demo corpus** — the 10 arXiv papers behind the project's own methods (RAG, dense retrieval, SBERT, BGE, SPECTER2/SciRepEval, BERT re-ranking, ColBERT, HyDE, LLM-as-a-judge, AI Usage Cards). `tests/eval/corpus_manifest.yaml` pins arXiv IDs + sha256; `scripts/download_corpus.py` fetches them from arXiv (download-only, nothing re-hosted); `tests/eval/cases.public.yaml` is a standalone 10-case eval over them. Anyone can now reconstruct the public corpus and re-run the sweep/eval against a known-good set. Testing strategy is documented in `tests/eval/TESTING.md`.

**Effort:** the measurement run took ~½ day of mostly-unattended compute + interpretation (done 2026-06-06).

### Engineering — retrieval K split (commit `09115c8`, 2026-06-07)

`CANDIDATE_K` (=20) is the candidate pool fetched per retriever before rerank; `TOP_K`
(=10) is the final post-rerank cut passed to the LLM. Previously the pool was hardcoded
to `10 == TOP_K`, leaving the cross-encoder no headroom to reorder. **Public-corpus A/B
(2026-06-13, `--repeat 3` + judge): a tie — no regression vs the pre-split `CANDIDATE_K=10`**
([`tests/eval/baselines/candidate_k_public_2026-06-13.md`](../tests/eval/baselines/candidate_k_public_2026-06-13.md)).
So `CANDIDATE_K=20` is a **safe default**, not yet a *measured win*: the public set is
one-paper-per-topic and can't exercise the cross-paper crowding the wider pool targets —
that wants a re-run on the private neuroscience corpus (`cases.yaml`). `CANDIDATE_K=10`
reproduces the exact pre-split behaviour; a guard requires `CANDIDATE_K >= TOP_K`. See
`config.py:107-119`, `pipeline.py:82,88`, and `tests/unit/test_retrieval_config.py`.

### Integrity Chunk 2c — Reviewer aggregation & self-improvement loop — ✅ **code shipped (PR 12, 2026-06-14)**

Turn the per-answer reviewer (Chunk 2b) and provenance records (Chunk 1) into a feedback loop: mine reviewer verdicts for *systematic* failure modes, then act on them.

**Why.** A reviewer that runs and is then forgotten is a cost with no compounding return. Aggregating its verdicts is where the self-improvement actually lives — recurring low scores on one dimension point at either a fixable system fault or a biased reviewer.

**The core hazard, designed around.** The reviewer is a *biased sampler* (it runs only on already-flagged answers per Chunk 2b's gate) **and** an LLM with its own systematic tilts (a prompt that rewards citation density will ding citation density everywhere). So a recurring suggestion is ambiguous by construction. Resolving it requires a ground-truth anchor:

- Periodically run the reviewer against the golden eval set (Feature 3), where answer quality is already known.
- If it flags cases verified as good → **reviewer bias** (fix the rubric/prompt).
- If flagged cases correlate with low eval correctness → **real system fault** (fix retrieval, chunking, or prompting).

Without this anchor, "pattern in the suggestions" is unfalsifiable — it must not ship without it.

**Deliverables.**

- Add a categorical `failure_tag` enum to the reviewer rubric (`missing_citation`, `overclaim`, `evidence_contradiction`, `no_hedge`, …) alongside the existing free-text `notes`. Free text stays as human-readable detail; the enum makes patterns *countable*.
- Aggregation module + CLI over the `AnswerReview` table — counts per `failure_tag`, per `prompt_version`, over time. Reuses the eval harness's existing aggregate/flaky-case machinery rather than re-implementing it.
- Bias-vs-fault report: the eval-anchored comparison above, emitted as a short markdown summary. Anchor against the **verified** eval set only (the public 10); a best-effort reference cannot adjudicate bias vs fault.
- **Minimum-N gate (locked).** The reviewer is a biased sampler over a small corpus, so a raw count is noise with a label. A `failure_tag` is not reported as actionable until it clears `MIN_FAILURE_TAG_COUNT` occurrences across `MIN_FAILURE_TAG_DOCS` distinct documents (config-driven, user-tunable; ~10/5 default, tune on the first real distribution). Below the gate the report reads "insufficient evidence." Counts always carry their denominator ("4 / 7 flagged"), never bare. The judge/reviewer used here is the pinned reference instrument, not a swappable local model — otherwise the anchor drifts.
- (Optional, gated) surface the top recurring fault to the user/devlog as a suggested fix; **no auto-remediation** — same cost-discipline rule as no-auto-retry. v1 is instrumentation, not action; the action layer waits until accumulated records clear the gate.

**Architecture:** read-only aggregation over existing sidecar tables. No mutation of the chunk store. Enrichment-Layer Pattern.

**Effort:** 1 weekend (the `failure_tag` schema change is the only write; the rest is read + report).

---

## Phase 7 — Gap Detection

### Feature 6 — Self-organizing wiki / synthesis layer — ✅ **shipped (PR 13, 2026-06-14; 6a–6d)**

A derived, human-readable markdown layer over the RAG corpus: per-topic notes (summary + tags + `[[links]]` + source citations) distilled from retrieved chunks. The Karpathy LLM-wiki pattern — proven in the cross-project atlas — applied *on top of* RAG, not as a replacement for it.

**Why.** RAG and a wiki solve different problems: RAG retrieves over a large corpus you didn't author and won't edit; a wiki is a small, authored, evolving distillation. Over a research library too big to read directly, a synthesis layer gives a cheap, LLM-native, browsable index — and, critically, makes **knowledge gaps computable**: a topic note with three thin citations and no `[[links]]` is a structural gap signal, not just an LLM opinion. This is the mechanism Phase 7 (gap detection) needs and the living precursor to Phase 9 (literature review generation, which becomes a *living* artifact instead of a one-shot export).

**Architecture:** post-ingest enrichment layer, same pattern as `citations.py`/`metadata_extractor.py`. Sidecar markdown files (`data/wiki/`), idempotent, never mutates the chunk store. Notes are regenerated from current retrieval + provenance, so they stay in sync with the corpus.

**Deliverables (cheap → expensive).**

- 6a — Topic note generator: cluster the library (existing doc vectors / citation graph), emit one markdown note per topic with summary, tags, citations, and `[[links]]` to related notes. Sidecar `data/wiki/`.
- 6b — Gap signals: compute structural gap markers per note (citation thinness, missing links, single-source claims) — reuses the confidence-signal heuristics already in `provenance.py`.
- 6c — Refresh + drift: regenerate notes on re-ingest; flag notes whose underlying sources changed.
- 6d — Obsidian-compatible export: emit `data/wiki/` with YAML frontmatter + `[[wikilinks]]` + a folder-per-community layout so it opens directly as, or merges into, an existing Obsidian vault (the stated complement). A target on the existing generator, not a new layer.

**What NOT to do.** Don't fold this into a separate project, and don't let it tempt a move away from RAG — it's an additive layer. Don't hand-author the notes (that's a normal wiki); they're *derived* and regenerable.

**Effort:** 6a is 1 weekend; 6b/6c 1 evening each.

See also `docs/decisions.md` for the existing Phase 7 gap-detection intent.

### Feature 7 — cross-document concept graph (✅ 7a–7c shipped PR 16, 2026-06-15)

A concept/entity graph across the library: nodes = concepts/entities, edges = relations, clustered (Leiden) into communities with high-degree "god nodes" surfaced. Feature 6 clusters *documents* and writes topic notes; this relates *concepts across* documents — the layer that powers real gap detection and gives Feature 6's notes their `[[links]]` automatically. The most important addition from the Graphify review.

**Architecture:** post-ingest enrichment sidecar (same pattern as `citations.py` / `metadata_extractor.py`), NetworkX + graspologic → a `graph.json` artifact. **Not** a graph database (Stonebraker: graph DBMS are rarely the performant choice; this is build-time structure, not a query server). Every edge tagged `EXTRACTED` / `INFERRED` / `AMBIGUOUS`, reusing the integrity layer — no self-reported confidence. Extraction runs on local Ollama (provider protocol) to hold the local-first promise.

**Deliverables (cheap → expensive).**

- 7a — ✅ Node/edge extraction per document, merged into a corpus graph (sidecar `data/graph/graph.json` + per-doc cache), edges integrity-tagged EXTRACTED/INFERRED/AMBIGUOUS. Extraction defaults to **local Ollama explicitly** (credit-safe-by-default).
- 7b — ✅ **Louvain** communities (networkx-native; Leiden deferred for Windows-wheel safety) + god-node ranking. The bridge back to Feature 6 (`concept_graph.doc_clusters_from_graph`) is built; re-pointing `build_wiki`'s `[[links]]` at it is a deferred follow-up PR.
- 7c — ✅ Graph-structure gap signals (isolated nodes, thin bridges) — complements Feature 6's citation-thinness signals.
- 7d — Knowledge-currency / claim-corroboration layer: claim-level structural weights (corroboration, contradiction, supersession direction — **age is not an input**), projected onto chunks as a `chunk_epistemics` sidecar, surfaced as Chunk 2a evidence markers (`contested`, `superseded_trend`) + a reviewer `contested_evidence` tag. Unique-source claims are neutral, never penalized; no retrieval-rank integration in v1 (eval-gated if ever). Spec: `docs/specs/feature-7d-knowledge-currency.md` (designed 2026-06-10).

Depends on PR 1 (doc vectors) + PR 13 (Feature 6). 7d additionally depends on 7a–7c + Chunk 2a/2b (shipped).

---

## Phase 8 — UI Polish

If Chunk 2a's adjudication UI shipped rough, this is where it gets polished.

### Settings page — expose the RAG "sandbox" knobs to the user (added 2026-06-04)

Surface the pipeline's config knobs in a GUI settings page so a user can experiment
without editing `.env`: embedder choice, chunk strategy, the retrieval candidate pool
(`CANDIDATE_K`) and final cut (`TOP_K`), parent-child and multi-query toggles,
BM25/vector weights, reranker (model + on/off), and the LLM backend (Claude API vs
local Ollama). **The benchmarked defaults stay the default** —
the page presents the alternatives with the measured recommendation pre-selected, not
a blank slate, so a curious user can try the sandbox without losing the tuned setup.

- **Depends on backend config-exposure first.** Today only some of these are env-driven;
  BM25/vector weights, the reranker, and a general sweep are still hardcoded (see
  `decisions.md` → Deferred Improvements → "Expose remaining retrieval knobs — toward a
  config-complete RAG sandbox"). Wire each knob through `config.py` before surfacing it.
- **Framework caveat.** A real settings page — alongside the Chunk 2a adjudication UI and
  the citation-graph view — is part of what forces the Chainlit-vs-X decision (Open
  Questions in `decisions.md`); Chainlit may not carry it.

Otherwise no additions from this roadmap — see `docs/decisions.md`.

---

## Phase 9 — Literature Review Generation

### Integrity Chunk 3 — PRISMA-trAIce export

When the literature review generator produces a review, it also produces a publishable-grade methodology disclosure.

**Source:** PRISMA-trAIce checklist (PMC12694947). Satisfies Nature Methods disclosure norm as a byproduct.

**Deliverables:**

- `scripts/export_review_traice.py` → emits a structured trAIce-aligned JSON + markdown alongside any generated review.
- Pulls directly from `answer_records` (Chunk 1) and adjudication log (Chunk 2a). Nothing new to instrument — just a formatter.
- Fields: AI tools used, search/retrieval parameters per query, human-AI interaction log (accepts/rejects/edits), performance metrics from the eval harness, version pins.

**Why long-term:** only useful when Phase 9 (literature review) ships. Building it now is premature.

**Effort:** 1 day once Chunks 1 + 2a exist.

---

## Feature 5 — Extract eval harness as a standalone repo

**Trigger:** After Features 1–3 ship and the integrated harness has been used to produce real measurements (Feature 3's BGE vs SPECTER2 run, at minimum).

**Process.**

- Copy `runner.py`, `scorers.py`, `store.py`, `report.py` into a new repo (`llm-eval-harness`).
- Write a README positioning the tool as general-purpose.
- Add a second adapter to prove reusability.
- Minimal docs + a usage example.

**Effort:** 1 weekend.

---

## Ingestion adapters — Zotero / Calibre (later)

Inbound bridges to the libraries doc_assistant complements. **Deferred to near the end of the roadmap** — useful, not urgent; the core RAG + synthesis + concept-graph layers come first.

- **Zotero** — read a BetterBibTeX export (or the Zotero SQLite) → ingest PDFs with title/authors/year/DOI/tags mapped to existing metadata. You already export BibTeX; this is the inbound direction.
- **Calibre** — *candidate, not committed.* Calibre library = folder + `metadata.db`; same adapter shape if it turns out to be the right fit. Decide when we get here.

One extractor/adapter each; markdown-intermediate means no downstream change. Vendor-neutral, config-gated.

---

## External interface — MCP server (later / open)

The outbound counterpart to the ingestion adapters above: expose the RAG pipeline **as** an MCP server so external MCP hosts — Claude Desktop, claude.ai connectors — can call the local library as a tool (e.g. `search_library` / `ask`), letting a paid Claude subscription query the user's own documents. Another **thin entrypoint** (`apps/mcp_server.py`) over `pipeline.py` — no core changes, consistent with the `apps/` boundary rule. Open nuance: Claude Desktop can use a local **stdio** server directly; claude.ai **connectors** need a reachable HTTP endpoint + auth — a real consideration for a local-first app. **Open / unscheduled** — see `decisions.md` → Open Questions.

---

## Implementation order (Claude Code PR-by-PR)

Each row is one PR. Each PR scopes to one chunk, with the files and the `decisions.md` section it depends on. Claude Code can `Read` the referenced section to get full architectural context.

| # | PR title | Files | Effort | Depends on |
|---|----------|-------|--------|------------|
| 1 | Close Phase 4: doc vectors + backfill | `src/doc_assistant/doc_vectors.py` (new), `src/doc_assistant/library.py` (similarity-edge query), `scripts/compute_doc_vectors.py` (new) | 1 evening + 15-min run | `decisions.md` → Phase 4 |
| 2 | Feature 1: config-driven embedding layer | `src/doc_assistant/config.py`, `src/doc_assistant/embeddings.py` (new factory), `src/doc_assistant/ingest.py`, `src/doc_assistant/pipeline.py` | 1–2 evenings | `decisions.md` → Phase 5 / Feature 1 |
| 3 | Feature 2: eval harness v0 | `src/doc_assistant/eval/` (new module), `tests/eval/cases.yaml` (stub) | 1 weekend | PR 2 |
| 4 | Feature 3: golden set + BGE vs SPECTER2 | `tests/eval/cases.yaml` (populated), README "Benchmark results" section | 1 day | PR 3 |
| 5 | Integrity Chunk 1: provenance card | `src/doc_assistant/db/models.py` (new `AnswerRecord`), `src/doc_assistant/db/migrations.py`, `src/doc_assistant/tracking.py`, `src/doc_assistant/commands.py` (`/export-record`), `apps/chainlit_app.py` (card render) | 1 evening | PR 2 |
| 6 | Feature 1b: per-project embedder routing — **DEFERRED** (no model beats bge-base on a sub-corpus yet) | `src/doc_assistant/db/models.py` (Folder.embedding_model), `src/doc_assistant/ingest.py`, `src/doc_assistant/pipeline.py`, UI surface | 2 evenings | Blocked until a per-sub-corpus winner exists (re-run SPECTER2 `--repeat 5` first) |
| 7 | Feature 4a: table pass (Marker primary, pdfplumber fallback) ✅ | `src/doc_assistant/tables_marker.py` (new, primary), `scripts/extract_tables_marker.py` (new), `src/doc_assistant/tables.py` (pdfplumber fallback), `scripts/extract_tables.py` | 1 evening | PR 1 (Phase 4 closed) |
| 8 | ✅ Feature 4b: figure detection + manifest (shipped 2026-06-14) — spec: [`docs/specs/feature-4b-figure-detection.md`](specs/feature-4b-figure-detection.md) | `src/doc_assistant/figures.py` (new), `scripts/extract_figures.py` (new), `src/doc_assistant/db/models.py` (Figure table), `config.py`, `.gitignore` | 1 weekend | PR 7 ✅ (regions.py shipped) |
| 9 | ✅ Feature 4c: VLM figure description (gated) **+ figure-chunk emission + eval scorer** (full 4c, shipped 2026-06-14; one paid run pending) | `figures.py` (VLM call/schema/gating), `scripts/describe_figures.py` (new), `ingest.py` (`figure_units` → `chunk_type='figure'`), `eval/scorers.py` (`FigureRetrievalScorer`), `eval/adapters.py`, `config.py` | 1 weekend | PR 8 ✅ |
| 10 | Integrity Chunk 2a: dual interpretation + adjudication | `src/doc_assistant/pipeline.py`, `src/doc_assistant/prompts.py`, `src/doc_assistant/config.py` (`SYNTHESIS_MODE`), UI surface for accept/reject/edit | 1 weekend | PR 5 |
| 11 | Integrity Chunk 2b: reviewer agent | `src/doc_assistant/reviewer.py` (new), Pydantic rubric schema, integration in `pipeline.py` | 1 evening | PR 10 |
| 11.5 | Chunking sweep infra (Phase 2.4 reopened) ✅ shipped 2026-05-31 | `src/doc_assistant/config.py`, `src/doc_assistant/ingest.py`, `scripts/sweep_chunking.py` (new), `tests/unit/test_chunking_config.py` (new) | done (infra) + ½-day measurement run | PR 3 (eval harness) |
| 12 | ✅ Integrity Chunk 2c: reviewer aggregation & self-improvement loop (min-N gated; instrumentation-first) — shipped 2026-06-14 (paid anchor run pending) | `reviewer.py` (`FAILURE_TAGS` enum), `db/models.py` (`AnswerReview.failure_tag`) + `db/migrations.py` (additive `ALTER TABLE`), `config.py` (`MIN_FAILURE_TAG_COUNT`/`_DOCS`), `reviewer_aggregate.py` (new) + `scripts/reviewer_report.py` (new, `--anchor`) | 1 weekend | PR 11 + PR 4 (verified golden set as the anchor) |
| 13 | ✅ Feature 6: self-organizing wiki / synthesis layer (6a–6d) — shipped 2026-06-14 | `src/doc_assistant/wiki.py` (new), `scripts/build_wiki.py` (new), `config.py` (`WIKI_*`), gap signals mirror `provenance.py` | 1 weekend (6a) + 1 evening each (6b/6c) | PR 1 (doc vectors) + PR 5 (provenance) |
| 14 | Integrity Chunk 3: PRISMA-trAIce export | `scripts/export_review_traice.py` (new) | 1 day | Phase 9 work; PRs 5 + 10 |
| 15 | Feature 5: extract eval harness to standalone repo | New repo | 1 weekend | PR 4 + at least one real measurement run |
| 16 | ✅ Feature 7: cross-document concept graph (7a–7c) — shipped 2026-06-15, validated free on local Ollama | `src/doc_assistant/concept_graph.py` (new), `scripts/build_concept_graph.py` (new), `config.py` (`CONCEPT_GRAPH_*`), `pyproject.toml` (networkx) | done | PR 1 + PR 13 ✅ |
| 17 | Ingestion adapters: Zotero (Calibre TBD) — later | `src/doc_assistant/extractors.py`, `src/doc_assistant/ingest.py`, `scripts/import_zotero.py` (new) | — | PR 2 |

**Provider protocol (generation side of Feature 1).** Folded into the Feature 1 provider theme — not a new numbered PR. Independent, near-term, no dependency. Files, build node, and guard test: `docs/specs/llm-provider-isolation.md`.

---

## What NOT to do

- Don't refactor doc_assistant's overall architecture. Locked decisions in `decisions.md` are locked for a reason.
- Don't add SPECTER2 *and* PubMedBERT *and* MedCPT at once. Pick one (SPECTER2). Biomedical models are a separate decision, gated on corpus need.
- Don't over-engineer the eval harness v0. Pydantic + pytest + DuckDB + Anthropic judge. No frameworks.
- Don't extract the standalone repo before the integrated version has produced at least one real comparison.
- Don't try to splice figures into the markdown. Sidecar manifest only.
- Don't ship 4c (VLM description) before 4b (figure detection + manifest). Caption-only chunks are a working baseline; 4c is a quality lever on top.
- Don't show self-reported LLM confidence scores. Use retrieval-derived uncertainty markers + reviewer rubric output instead.
- Don't auto-retry on reviewer-flagged issues. Surface them; the user decides.
- Don't bind the project's identity to a single vendor framework (BE WISE, etc.). Reference standards as influences; keep config flags vendor-neutral.
- Don't mine reviewer suggestions for "patterns" without the eval-set anchor (Chunk 2c). The reviewer only runs on flagged answers and has its own biases; an unanchored pattern can't distinguish reviewer fault from system fault.
- Don't auto-remediate from the self-improvement loop. Surface the recurring fault; a human decides the fix. Same discipline as no-auto-retry.
- Don't sweep chunking without re-embedding. A size change invalidates the embedding cache; in-place comparison is meaningless.
- Don't change chunk-size defaults from a single run. Use `--repeat` and beat the control (current default) by more than its variance before re-locking.
- Don't hand-author the wiki notes (Feature 6). They're derived from retrieval + provenance and regenerable; a hand-edited note drifts and defeats gap detection.
- Don't treat the wiki layer as a RAG replacement. It's an additive synthesis/index layer on top of the chunk store.
- Don't make the concept graph (Feature 7) a graph database. NetworkX + a file artifact; no Neo4j/server — it's build-time structure, not a query store.
- Don't let the Zotero/Calibre adapters leak vendor specifics past the extractor boundary — everything normalizes to the existing metadata schema.

---

## References

- AI Usage Cards — arXiv 2303.03886 (provenance card schema)
- PRISMA-trAIce — PMC12694947 (Phase 9 export target)
- BE WISE framework — Frontiers, April 2026 (influence on dual-layer / `SYNTHESIS_MODE=human`)
- Nature Methods — disclosure norm satisfied as a byproduct of PRISMA-trAIce export
- Karpathy LLM-wiki pattern — structured markdown as an LLM-queryable knowledge base (influence on Feature 6; proven in the cross-project atlas). Note: the widely-cited "70x more efficient than RAG" framing is marketing, not a benchmark — Feature 6 layers a wiki *on top of* RAG rather than replacing it.
