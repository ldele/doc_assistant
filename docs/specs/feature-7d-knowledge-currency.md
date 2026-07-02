# Spec — Feature 7d: knowledge-currency / claim-corroboration layer

**Status:** ✅ **Engine BUILT 2026-06-17** (Claude Code, RTX box), validated free on local Ollama. Designed 2026-06-10 (Cowork). Dependencies (PR 13 wiki + PR 16 concept graph 7a–7c) shipped. **Shipped this PR:** `concept_graph.py` polarity + `EdgeSupport` + `compute_node_weights`; new `epistemics.py` (structural chunk projection + markers) + `scripts/compute_epistemics.py`; `chunk_epistemics` table; reviewer `contested_evidence` tag; guard tests + a free real run (748 chunk rows; 169/198 nodes `unique` → neutral; 0 `superseded_trend`; contested signal local-model-noisy). **Live answer-time marker surfacing SHIPPED 2026-06-22 (PR-M1, `docs/specs/pr-m1-epistemics-markers.md`):** retrieved sources carry `contested`/`superseded_trend` markers via `ChatController` (NOT `synthesis.py`/`pipeline.py` — synthesis stays untouched, so a turn is byte-identical when markers are absent and the eval path is unaffected). Flat chunks join on the `{document_id}:{chunk_index}` key (plumbed in PR-M0); PC parents map via text containment (`epistemics.markers_for_parent`, PR-M1 ADR-1). **Live marker surfacing DEFAULT-OFF since 2026-07-02 (R7 / `docs/decisions/ADR-005-epistemics-markers-default-off.md`):** because marker quality still reflects the superseded open-vocabulary graph (KI-7), `EPISTEMICS_MARKERS_ENABLED` (default `false`) now gates `ChatController._attach_markers` — the default turn takes the byte-identical no-marker path; Node B flips the default back on with trustworthy data. **Still deferred:** the `query_router.py` local/global retrieval-mode seam (Decision 8). Marker *quality* still reflects the superseded open-vocabulary graph (KI-7) — M1 surfaces what exists; it does not improve extraction. Per-claim character-span back-pointers replaced by structural label-in-text attribution.
**Owner of execution:** Claude Code (code + tests), when Phase 7 is active.
**Pattern reference:** Enrichment-Layer Pattern (`docs/decisions.md`); Research Integrity Layer (Chunk 2a markers, Chunk 2b reviewer); Feature 7 concept graph (roadmap).

**Goal (the why).** RAG treats the library's text as ground truth. It isn't: an old high-impact paper can hold stale results next to still-canonical definitions, and the corpus can disagree with itself. 7d gives each **claim** an epistemic weight derived from **corroboration across the library** — not from publication age, not from LLM self-report — and surfaces it at answer time. Age drops out as an input entirely: supersession *emerges* when newer claims contradict older ones; an old claim the corpus still agrees with keeps full weight.

---

## ADR — claim-level corroboration, projected onto chunks, surfaced as markers

**Context.** Chunks are arbitrary ~2000-char windows — the wrong unit for epistemics (one parent can hold one stale result and three solid definitions; a per-chunk scalar averages that into mush). Feature 7 (7a–7c) builds a concept/entity graph with integrity-tagged edges. Chunk 2a already segments answers into citation-anchored claims with deterministic markers. The reviewer (2b) runs on flagged answers only.

**Decision.** The epistemic unit is the **claim** (a concept-graph edge). Every claim carries **back-pointers to the chunk ids (and spans) it was extracted from** — any claim is one hop from its verbatim evidence. Corroboration weight is computed **structurally** per node/claim: independent-source count, agreement ratio, polarity of newer-vs-older edges. Chunk-level weight is a **projection** (aggregate over the claims a chunk contains), stored in a sidecar table — the chunk store is never mutated. v1 output is **markers + a reviewer failure tag**, never a retrieval-rank feature.

**Options considered.**
1. *Doc-level age/citation weighting* — rejected: age is a proxy; doc-level is the wrong granularity (stale results vs canonical definitions in one paper). External citation **velocity** (OpenAlex `counts_by_year`) survives only as an optional, separate doc-level signal at the Phase 7 DOI-lookup step — never mixed into the claim weight.
2. *LLM scores chunk/claim quality* — rejected: self-reported confidence, banned by the project's What-NOT-to-do list. All weights are structural/observable.
3. *Answer from the claim library (GraphRAG-style replacement)* — rejected: extraction is lossy; answering from derived claims breaks verbatim chunk-level grounding and citation fidelity. Chunks stay the retrieval + grounding substrate; the claim layer is an additive epistemic index (2026 field consensus is hybrid: vectors for breadth, graph for depth).
4. *Fold the weight into retrieval ranking* — deferred, eval-gated: down-weighting old/contested content at retrieval silently suppresses definitional content and changes a locked retrieval stack. If ever attempted, it enters as a measured experiment through the eval harness, like any locked-setting change.

**Consequences.** 7d is read-only over Feature 7's graph plus one new sidecar projection table. No new generation logic, no retrieval change. The dangerous failure modes (consensus bias, unique-source penalty) are handled by explicit normalization rules below, and the output stays "disagreement + direction, surfaced to the human" — never an authority score the pipeline silently trusts.

---

## Decisions (locked 2026-06-10)

| # | Decision |
|---|---|
| 1 | **Age is not an input.** Currency emerges from corroboration polarity over time; an uncontradicted old claim keeps full weight. |
| 2 | **Claim-level, not chunk-level.** Weights live on concept-graph nodes/edges; chunks get a projected weight in a sidecar (`chunk_epistemics`), keyed by chunk id. Chunk store untouched. |
| 3 | **Structural weights only.** Inputs: independent supporting sources, contradicting sources, refinement edges, edge recency *relative to each other*. No LLM-self-reported confidence anywhere. |
| 4 | **Coverage normalization (the unique-source rule).** Zero corroboration because *contested* → down-weight + flag. Zero corroboration because *only source on its topic* → **neutral, never negative** (that chunk is among the most valuable in the library). The two cases are distinguished by whether contradicting edges exist. |
| 5 | **Consensus-bias guard.** The system never declares a winner. Output = disagreement + direction ("3 post-2020 sources refine/contradict this 2005 claim"), adjudicated by the human — same philosophy as Chunk 2a accept/reject. |
| 6 | **Surfacing v1 = markers + reviewer tag.** (a) Evidence-layer marker when a retrieved chunk's claims sit on contested/superseded-trending nodes (extends the Chunk 2a marker set: `contested`, `superseded_trend`). (b) New reviewer rubric `failure_tag`: `contested_evidence`. Min-N gating discipline as in Chunk 2c — below threshold the signal reads "insufficient evidence", counts carry denominators. |
| 7 | **Adjudication log as a second signal (later).** Per-claim accept/reject/edit history (`answer_claims`) folds in as a *personal* trust signal — sparse and biased (reviewer sees flagged answers only), so it modulates, never defines, the weight. v2. |
| 8 | **Router seam (from the GraphRAG review).** `query_router.py` is the explicit seam between retrieval modes: local/factoid → chunk pipeline (unchanged); global/synthesis ("main themes", "where does my corpus disagree") → wiki/graph layer, claims one hop from verbatim chunks for citation. Routing quality is measured via the eval harness when Phase 7 lands. |

---

## Contracts (build-time, pre-validated against dependencies when they ship)

### `src/doc_assistant/concept_graph.py` (Feature 7, extended)
- Edge schema gains: `source_doc_id`, `source_chunk_ids: list[str]` (back-pointers, with spans where available), `year` (from doc metadata, used only for *relative* polarity ordering), `polarity ∈ {supports, contradicts, refines, supersedes}`, plus the existing `EXTRACTED|INFERRED|AMBIGUOUS` integrity tag. Extraction runs on local Ollama (provider protocol) per Feature 7.
- `compute_node_weights(graph) -> dict[node_id, NodeWeight]` — pure; `NodeWeight = (n_supporting_sources, n_contradicting_sources, agreement_ratio, direction ∈ {stable, contested, superseded_trend}, coverage ∈ {corroborated, unique, contested})`. Deterministic given the graph.

### `src/doc_assistant/epistemics.py` (new)
- `project_chunk_weights(graph, weights) -> list[ChunkEpistemics]` — aggregate node weights onto the chunks referenced by back-pointers. Pure.
- CLI runner `scripts/compute_epistemics.py` — idempotent, sidecar-only, re-runnable after every graph rebuild. Enrichment-Layer Pattern.

### `src/doc_assistant/db/models.py` + `src/doc_assistant/db/migrations.py`
- New `ChunkEpistemics` (`chunk_epistemics`): `chunk_id (indexed) · n_claims · n_contested · n_superseded_trend · coverage_summary (JSON) · graph_version · computed_at`. Sidecar; regenerable; dropped + rebuilt with the graph.

### Surfacing
- `pipeline.py` / `synthesis.py`: at answer time, look up `chunk_epistemics` for retrieved chunks; emit `contested` / `superseded_trend` markers into the evidence layer (read-only join, no retrieval change).
- `reviewer.py`: add `contested_evidence` to the `failure_tag` enum (lands with or after Chunk 2c's enum).

---

## Build node

**Depends on:** PR 16 (Feature 7 7a–7c: graph + Leiden + gap signals), PR 13 (Feature 6 wiki — for the global-query routing target), Chunk 2a (marker surface, shipped), Chunk 2b (reviewer, shipped). External-velocity sub-signal additionally depends on the Phase 7 DOI-lookup work.
**Files owned:** `src/doc_assistant/concept_graph.py` (edge schema + node weights), `src/doc_assistant/epistemics.py` (new), `scripts/compute_epistemics.py` (new), `src/doc_assistant/db/models.py` + `src/doc_assistant/db/migrations.py` (`chunk_epistemics`), `src/doc_assistant/pipeline.py`/`synthesis.py` (marker join), `src/doc_assistant/reviewer.py` (failure tag), `src/doc_assistant/query_router.py` (router seam, with Feature 6/7), tests as below.
**Status:** blocked (design-locked).

### Guard tests (written with the build)
- `tests/unit/test_epistemics.py` — fixed toy graph: corroborated node → `stable`; node with newer contradicting edges → `superseded_trend`; **unique-source node → neutral coverage, never down-weighted** (the regression that matters most); projection maps weights to the right chunk ids. Pure, no DB/LLM.
- `tests/integration/test_epistemics_markers.py` — mocked retrieval over chunks with known `chunk_epistemics` rows → evidence layer carries `contested`/`superseded_trend` markers; clean chunks stay quiet (quiet-on-clean, consistent with PR 5.1).

## Definition of done
- Graph rebuild → `compute_epistemics` produces deterministic sidecar weights; idempotent re-run is a no-op.
- Retrieved chunks on contested/superseded-trending nodes show markers; unique-source chunks are never penalized; clean answers stay quiet.
- `contested_evidence` reviewer tag fires only above the min-N gate, with denominators.
- No change to retrieval output (public eval byte-identical pre/post 7d with markers disabled); ruff / mypy --strict / bandit clean.

## Out of scope
Retrieval-rank integration of the weight (eval-gated future experiment). Truth adjudication of any kind. External citation-velocity blending into claim weights (doc-level, separate, optional). Adjudication-log trust signal (v2). Multi-corpus / field-level comparison (that is Phase 7 gap detection proper).
