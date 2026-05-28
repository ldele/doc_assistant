# doc_assistant — roadmap additions

Engineering quality additions on top of the existing phase plan in `docs/decisions.md`. Purpose: domain-aware retrieval, reproducible evaluation, structured handling of figures and tables, and an explicit research-integrity layer that records how each answer was produced.

This document is the **source of intent**. `docs/decisions.md` records the locked design choices. `CLAUDE.md` points at both.

---

## Goals

1. Make the embedding layer swappable and per-project routable, with measured comparisons across models.
2. Build a reproducible eval harness inside the project, designed to be extractable later.
3. Promote figures and tables from "lossy text artifacts" to first-class structured content.
4. Add a research-integrity layer: every answer carries a provenance record; synthesis is split into evidence and interpretation layers; an LLM reviewer scores each interpretation against a rubric.
5. Position the project against published standards (PRISMA-trAIce, AI Usage Cards, BE WISE) without binding to any single vendor framework.

---

## Phase ordering (renumbered)

The new work integrates into the existing phase plan as follows. The full phase list lives in `docs/decisions.md`; this table is for reference only.

| Phase | Content | Source of features |
|-------|---------|--------------------|
| 4 | Citation Graph — close out | Existing roadmap |
| 5 | Embedding & Eval Foundation | Features 1, 2, 3 + Integrity Chunk 1 |
| 6 | Per-project routing + Figures & Tables + Dual-layer interpretation | Features 1b, 4a, 4b, 4c + Integrity Chunks 2a, 2b |
| 7 | Gap Detection (was Phase 5) | Existing roadmap |
| 8 | UI Polish (was Phase 6) | Existing roadmap |
| 9 | Literature Review Generation (was Phase 7) | Existing roadmap + Integrity Chunk 3 |
| — | Extract eval harness to standalone repo (Feature 5) | Post-Phase 5, no phase number |

---

## Phase 4 — close out

What is left:

- Apply metadata backfill and citation extraction on the local DB (CLI runners exist; sandbox cannot write).
- Mean-pool doc-level vectors → similarity edges (the only Phase 4 deliverable still flagged "deferred to next session").
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

**Gate:** Ship only after Feature 3 results exist. Otherwise it is a switch with no evidence behind it.

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

#### 4a — pdfplumber pass for tables

Spliced inline into the markdown cache.

- Detect table regions after primary extraction, extract with pdfplumber, splice structured markdown tables back into the cached `.md` file.
- Mark with `<!-- table-extracted-by: pdfplumber -->` for traceability and re-runnability.
- Each table becomes a chunk with explicit row/column structure preserved in markdown.

**Why splice rather than sidecar:** tables are text-shaped (rows × cells); splicing back into the markdown is natural and preserves the "open the .md and see everything" property.

**Effort:** 1 evening.

#### 4b — Figure region detection + caption pairing (OpenCV, no LLM cost)

Sidecar manifest, not spliced.

- PyMuPDF already exposes image blocks (`block_type=1`); OpenCV refines region boundaries and detects chart-like regions (line detection, pixel variance, contour heuristics).
- Pair each figure region with its caption via nearest-text-block heuristic ("Figure N: …").
- Persist a `figures` table in SQLite: `{doc_id, page, bbox, caption, image_path, vlm_description, vlm_call_skipped_reason, extraction_method}`.
- Images written to `data/figures/{doc_hash}/page{N}_fig{M}.png`.
- Caption text remains in the markdown — figures are additive, not substituting.

**Why sidecar rather than splice:** figures are binary. Embedding base64 in markdown destroys the human-readable cache; placeholder strings without the image are noise. Sidecar preserves the markdown-as-universal-intermediate decision.

**Effort:** 1 weekend.

#### 4c — VLM figure description (gated)

- For each figure in the manifest, call Claude vision with a schema-first prompt: `{figure_type, summary, key_quantities, axes, trend}` (Pydantic-validated, Anthropic tool-use).
- Embed `caption + VLM description` as a chunk linked back to the figure's bbox. Chunk metadata: `chunk_type='figure'`.
- **Gating:** skip figures already well-described by their caption (length threshold + caption-only embedding similarity check). Enforce `MAX_VLM_CALLS_PER_DOC` budget.

**Why gated:** VLM cost is the main risk on this layer.

**Effort:** 1 weekend.

**Eval hook.** Add a "figure retrieval" scorer to the eval harness: given a held-out caption, does the system retrieve the right figure?

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

### Integrity Chunk 2b — Reviewer agent

Separate LLM call after the generator. Cheaper model (Haiku/Sonnet, not Opus). Returns structured JSON via Anthropic tool-use.

**Rubric:** faithfulness to retrieved passages, citation density, hedging adequacy, count of claims without sources.

**Output:** populates the uncertainty markers shown to the user; persisted on the answer record.

**No auto-retry in v1.** If the reviewer flags issues, surface them; the user decides whether to regenerate. Cost discipline.

**Reuse note.** The reviewer's rubric is identical in shape to a deterministic eval scorer. The same code paths the reviewer uses can be re-targeted at the eval harness — generator + frozen-prompt reviewer = LLM-as-judge scorer.

**Effort:** 1 evening on top of Chunk 2a.

---

## Phase 7 — Gap Detection

No additions from this roadmap. See `docs/decisions.md`.

---

## Phase 8 — UI Polish

No additions from this roadmap. See `docs/decisions.md`.

If Chunk 2a's adjudication UI shipped rough, this is where it gets polished.

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

## Implementation order (Claude Code PR-by-PR)

Each row is one PR. Each PR scopes to one chunk, with the files and the `decisions.md` section it depends on. Claude Code can `Read` the referenced section to get full architectural context.

| # | PR title | Files | Effort | Depends on |
|---|----------|-------|--------|------------|
| 1 | Close Phase 4: doc vectors + backfill | `src/doc_assistant/doc_vectors.py` (new), `src/doc_assistant/library.py` (similarity-edge query), `scripts/compute_doc_vectors.py` (new) | 1 evening + 15-min run | `decisions.md` → Phase 4 |
| 2 | Feature 1: config-driven embedding layer | `src/doc_assistant/config.py`, `src/doc_assistant/embeddings.py` (new factory), `src/doc_assistant/ingest.py`, `src/doc_assistant/pipeline.py` | 1–2 evenings | `decisions.md` → Phase 5 / Feature 1 |
| 3 | Feature 2: eval harness v0 | `src/doc_assistant/eval/` (new module), `tests/eval/cases.yaml` (stub) | 1 weekend | PR 2 |
| 4 | Feature 3: golden set + BGE vs SPECTER2 | `tests/eval/cases.yaml` (populated), README "Benchmark results" section | 1 day | PR 3 |
| 5 | Integrity Chunk 1: provenance card | `src/doc_assistant/db/models.py` (new `AnswerRecord`), `src/doc_assistant/db/migrations.py`, `src/doc_assistant/tracking.py`, `src/doc_assistant/commands.py` (`/export-record`), `apps/chainlit_app.py` (card render) | 1 evening | PR 2 |
| 6 | Feature 1b: per-project embedder routing | `src/doc_assistant/db/models.py` (Folder.embedding_model), `src/doc_assistant/ingest.py`, `src/doc_assistant/pipeline.py`, UI surface | 2 evenings | PR 4 (must have Feature 3 numbers first) |
| 7 | Feature 4a: pdfplumber table pass | `src/doc_assistant/tables.py` (new), `scripts/extract_tables.py` (new) | 1 evening | PR 1 (Phase 4 closed) |
| 8 | Feature 4b: figure detection + manifest | `src/doc_assistant/figures.py` (new), `scripts/extract_figures.py` (new), `src/doc_assistant/db/models.py` (Figure table) | 1 weekend | PR 7 |
| 9 | Feature 4c: VLM figure description (gated) | `src/doc_assistant/figures.py` (VLM call), Pydantic schema, `MAX_VLM_CALLS_PER_DOC` config | 1 weekend | PR 8 |
| 10 | Integrity Chunk 2a: dual interpretation + adjudication | `src/doc_assistant/pipeline.py`, `src/doc_assistant/prompts.py`, `src/doc_assistant/config.py` (`SYNTHESIS_MODE`), UI surface for accept/reject/edit | 1 weekend | PR 5 |
| 11 | Integrity Chunk 2b: reviewer agent | `src/doc_assistant/reviewer.py` (new), Pydantic rubric schema, integration in `pipeline.py` | 1 evening | PR 10 |
| 12 | Integrity Chunk 3: PRISMA-trAIce export | `scripts/export_review_traice.py` (new) | 1 day | Phase 9 work; PRs 5 + 10 |
| 13 | Feature 5: extract eval harness to standalone repo | New repo | 1 weekend | PR 4 + at least one real measurement run |

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

---

## References

- AI Usage Cards — arXiv 2303.03886 (provenance card schema)
- PRISMA-trAIce — PMC12694947 (Phase 9 export target)
- BE WISE framework — Frontiers, April 2026 (influence on dual-layer / `SYNTHESIS_MODE=human`)
- Nature Methods — disclosure norm satisfied as a byproduct of PRISMA-trAIce export
