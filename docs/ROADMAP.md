<!-- status: active · updated: 2026-06-20 · class: living -->

# ROADMAP — doc_assistant

The living roadmap: phase map, goals, and the machine-read PR table. This is the **source of
intent**; `docs/decisions.md` records the locked design choices and rationale, and `CLAUDE.md` /
`.claude/CONTEXT.md` point at both. The evaluation strategy and the verified-10 benchmark rule live
in `tests/eval/TESTING.md`.

> Reshaped from `docs/doc-assistant-roadmap.md` on 2026-06-20 (cpc adoption, ADR-001): the PR table
> now uses the cpc `| PR | Scope | Status | Spec |` columns so `roadmap_sync` can parse it. The full
> original (detailed per-phase prose) is preserved, frozen, at `docs/archive/doc-assistant-roadmap.md`.

## Goals

1. Make the embedding layer swappable, with measured comparisons. (Per-project *routing* is deferred
   until a model beats `bge-base` on an identifiable sub-corpus — no such win yet; the factory stays,
   the routing layer waits.)
2. Build a reproducible eval harness inside the project, designed to be extractable later.
3. Promote figures and tables from lossy text artifacts to first-class structured content.
4. Add a research-integrity layer: every answer carries a provenance record; synthesis splits into
   evidence and interpretation; an LLM reviewer scores each interpretation against a rubric.
5. Position the project against published standards (PRISMA-trAIce, AI Usage Cards, BE WISE) without
   binding to any single vendor framework.
6. Close the self-improvement loop: aggregate reviewer verdicts, separate reviewer bias from systemic
   fault by anchoring against the verified eval set, surface recurring failure patterns — but only
   above a minimum-N gate. Below the gate: instrumentation, not action.
7. Add a self-organizing markdown "wiki" synthesis layer over the corpus — distilled, linked, cited
   topic notes that make knowledge gaps computable. Feeds Phase 7 gap detection and Phase 9 review
   generation.

## Phases

(Bullet list, not a table — the only machine-read table in this file is the PR table below, which
`roadmap_sync` parses as the first markdown table.)

- **Phase 4 — Citation graph close-out** — doc-similarity edges. Status: done.
- **Phase 5 — Embedding & eval foundation** — config-driven embedder, golden set, provenance. Status: done.
- **Phase 6 — Figures/tables + dual-layer interpretation + reviewer + self-improvement loop** (per-project routing deferred). Status: in progress.
- **Phase 7 — Gap detection** — wiki/synthesis layer + cross-document concept graph. Status: in progress. *(The concept-graph open-vocabulary core was superseded by the 2026-06-18 redesign — not yet built; see `.claude/KNOWN_ISSUES.md`.)*
- **Phase 8 — UI polish** — settings page exposing the RAG sandbox knobs. Status: planned.
- **Phase 9 — Literature-review generation** — PRISMA-trAIce export. Status: planned.
- **(no phase number) — Extract eval harness to a standalone repo** (Feature 5). Status: planned.

## PR order (Claude Code, one PR per session)

Each row is one PR. `Spec` links the code-level contract where one exists; `decisions.md` carries the
full architectural context per feature.

| PR | Scope | Status | Spec |
|----|-------|--------|------|
| 1 | Close Phase 4: doc vectors + similarity-edge backfill | done | — |
| 2 | Feature 1: config-driven embedding layer (+ provider protocol) | done | `docs/specs/llm-provider-isolation.md` |
| 3 | Feature 2: eval harness v0 | done | — |
| 4 | Feature 3: golden eval set + BGE vs SPECTER2 comparison | done | — |
| 5 | Integrity Chunk 1: provenance card | done | — |
| 6 | Feature 1b: per-project embedder routing | deferred | — |
| 7 | Feature 4a: table pass (Marker primary, pdfplumber fallback) | done | `docs/specs/feature-4a-marker-table-ingest.md` |
| 8 | Feature 4b: figure detection + manifest | done | `docs/specs/feature-4b-figure-detection.md` |
| 9 | Feature 4c: VLM figure description + figure-chunk emission + eval scorer | done | — |
| 10 | Integrity Chunk 2a: dual interpretation + adjudication | done | `docs/specs/chunk-2a-dual-interpretation.md` |
| 11 | Integrity Chunk 2b: reviewer agent | done | — |
| 11.5 | Chunking sweep infra (Phase 2.4 reopened) | done | — |
| 12 | Integrity Chunk 2c: reviewer aggregation & self-improvement loop (min-N gated) | done | — |
| 13 | Feature 6: self-organizing wiki / synthesis layer (6a–6d) | done | — |
| 14 | Integrity Chunk 3: PRISMA-trAIce export | planned | — |
| 15 | Feature 5: extract eval harness to a standalone repo | planned | — |
| 16 | Feature 7: cross-document concept graph (7a–7c) | done | — |
| 17 | Ingestion adapters: Zotero (Calibre TBD) | planned | — |

**Feature 7d (knowledge-currency layer):** engine shipped 2026-06-17 (`epistemics.py` + `chunk_epistemics`
sidecar + polarity-aware concept graph + reviewer `contested_evidence` tag); **deferred** follow-ups —
live answer-time marker surfacing + the `query_router` seam. Spec: `docs/specs/feature-7d-knowledge-currency.md`.
**Feature 6 re-point** shipped 2026-06-17 (`wiki.load_communities` clusters by concept-graph communities
behind `WIKI_USE_CONCEPT_COMMUNITIES`, inert by default; cosine fallback).

**Later / open (no PR yet):** the concept-graph **redesign** (curated vocabulary + deterministic
skeleton + confined LLM enrichment, 2026-06-18 — the next concept-graph build, see
`.claude/KNOWN_ISSUES.md`); Zotero/Calibre ingest adapters (PR 17); an outbound **MCP-server**
interface over `pipeline.py`. Full detail for all three: `docs/archive/doc-assistant-roadmap.md`.

## What NOT to do

- Don't refactor the overall architecture. Locked decisions in `decisions.md` are locked for a reason.
- Don't add SPECTER2 *and* PubMedBERT *and* MedCPT at once. Pick one; biomedical models are a separate,
  corpus-gated decision.
- Don't over-engineer the eval harness. Pydantic + pytest + DuckDB + Anthropic judge. No frameworks.
- Don't extract the standalone eval repo before the integrated version produced a real comparison.
- Don't splice figures into the markdown. Sidecar manifest only. Don't ship 4c before 4b.
- Don't show self-reported LLM confidence. Use retrieval-derived uncertainty markers + reviewer output.
- Don't auto-retry or auto-remediate on reviewer-flagged issues — surface them; the user decides.
- Don't mine reviewer suggestions for "patterns" without the eval-set anchor (Chunk 2c).
- Don't sweep chunking without re-embedding; don't change chunk-size defaults from a single run
  (use `--repeat` and beat the control beyond its variance).
- Don't hand-author wiki notes — they're derived and regenerable. Don't treat the wiki as a RAG
  replacement; it's additive.
- Don't make the concept graph a graph database (NetworkX + a file artifact, build-time structure).
- Don't let Zotero/Calibre adapters leak vendor specifics past the extractor boundary.

## References

- AI Usage Cards — arXiv 2303.03886 (provenance card schema)
- PRISMA-trAIce — PMC12694947 (Phase 9 export target)
- BE WISE framework — Frontiers, April 2026 (influence on dual-layer / `SYNTHESIS_MODE=human`)
- Karpathy LLM-wiki pattern — structured markdown as an LLM-queryable knowledge base (influence on
  Feature 6; layered *on top of* RAG, not a replacement — the "70x more efficient" framing is marketing).
