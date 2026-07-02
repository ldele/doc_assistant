<!-- status: active · updated: 2026-07-02 · class: living -->

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
- **Phase 7 — Gap detection** — wiki/synthesis layer + cross-document concept graph + the gap-detection layer over them. Status: in progress. *(The concept-graph open-vocabulary core was superseded by the 2026-06-18 redesign — not yet built; see `.claude/KNOWN_ISSUES.md`. The gap layer itself is design-locked in `docs/decisions/ADR-004-gap-detection-layer.md` / `docs/specs/feature-gap-detection.md`, blocked on the Decision-C skeleton + the RG-001 edge-precision run.)*
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
| 17 | Ingestion adapters: Zotero (Calibre TBD) — optional producers for the S1 source registry, never a dependency | planned | `docs/specs/feature-selective-ingestion.md` (ADR-3) |
| M0 | Desktop shell: extract `ChatController` + `TurnResult` (UI-agnostic turn core) | done | `docs/specs/pr-m0-chat-controller.md` |
| M1 | Desktop shell: live 7d epistemics-marker surfacing (pre-migration demo win) | done | `docs/specs/pr-m1-epistemics-markers.md` |
| M2 | Desktop shell: FastAPI backend + SSE boundary | done | `docs/specs/pr-m2-fastapi-boundary.md` |
| M3 | Desktop shell: Tauri frontend (Svelte 5 + Vite) | done | `docs/specs/pr-m3-tauri-frontend.md` |
| M4 | Desktop shell: PyInstaller sidecar packaging + frozen CPU-torch pin | done (RG-010/011/012 Tier-1 pass; KI-9/10/11 bundled in the freeze; data-home/first-run-ingest flow now built — backend `77eb5f9` + frontend settings panel; **RG-012 Tier-2** cited-turn validation pends a re-freeze + clean-box run) | `docs/specs/pr-m4-sidecar-packaging.md` |
| M5 | Desktop shell: delete Chainlit + lift the Python-3.12 pin (KI-2) | done — Chainlit removed (renderer + dep + recipe + config); 3.12-pin lift **verified-and-deferred** (KI-2: native deps crash on 3.14, not Chainlit) | `docs/specs/pr-m5-decommission-chainlit.md` |
| R1 | Ingest hygiene: strip PyMuPDF4LLM image placeholders + cache-normalization runner (closes KI-14) | done (2026-07-02, `4da02e3` + host apply/re-ingest; cache grep=0 → KI-14 RESOLVED on main corpus) | `docs/specs/remediation-plan-2026-07.md` |
| R2 | Concept presence: word-boundary matching (RG-009 lever; de-confounds R5) | done (2026-07-02, `338e55e`) — `mode="boundary"` default + `substring` A/B lever; indicative measurement recorded (IR 270×/RAG 6.6×/BERT 3.3× substring inflation); curated-vocab corpus run at R5 | `docs/specs/remediation-plan-2026-07.md` |
| R3 | Keyword termhood: contrastive scoring (`wordfreq` reference — decided 2026-07-02) + C-value nested-term fix + orphan sweep | built (2026-07-02, staged) — ADR-006; `mode=contrastive`; 712 tests green; real-corpus dry-run surfaces domain vocab vs corpus_band's boilerplate | `docs/specs/remediation-plan-2026-07.md` |
| R4 | Concept skeleton: graded provenance strength (ratio, not boolean) | built (2026-07-02, staged) — `provenance_strength` per-token ratio on `SkeletonEdge`; `edge_weight` split tiebreak keeps the multi-token invariant; additive `strength_json` column + migration; +6 tests, 718 green; saturated toy graph → 1.0 (partial-graph spread measured at R5) | `docs/specs/remediation-plan-2026-07.md` |
| R5 | RG-001/008/009 multi-domain revalidation run + gap wizard-of-oz → edge-model go/no-go | planned (measurement session; needs R1–R3, R4 recommended) | `docs/specs/remediation-plan-2026-07.md` |
| R6 | Core retrieval: BM25 `preprocess_func` + pipeline hygiene (eval-gated, before any weight sweep) | planned | `docs/specs/remediation-plan-2026-07.md` |
| R7 | KI-7 containment: 7d marker chip default-off until Node B (decided 2026-07-02: option a — `EPISTEMICS_MARKERS_ENABLED` kill-switch) | done (2026-07-02, `591280d`) — ADR-005; default off | `docs/specs/remediation-plan-2026-07.md` |
| S1 | Selective ingestion backend: `SourceFile` registry + selection-scoped ingest (CLI `--files`/`--dry-run`, `GET/PATCH /api/sources`, `POST /api/ingest {paths}`) | planned (spec drafted 2026-07-02, not yet locked) | `docs/specs/feature-selective-ingestion.md` |
| S2 | Selective ingestion UI: Tauri sources panel (status chips, select-by-status/type, exclude toggle, ingest-selected) | planned (needs S1) | `docs/specs/feature-selective-ingestion.md` |

**Feature 7d (knowledge-currency layer):** engine shipped 2026-06-17 (`epistemics.py` + `chunk_epistemics`
sidecar + polarity-aware concept graph + reviewer `contested_evidence` tag). **Live answer-time marker
surfacing shipped 2026-06-22 (PR-M1)** — sources carry `contested`/`superseded_trend` chips via
`ChatController` (flat: `chunk_key` join; PC: text containment), synthesis untouched (byte-identical when
absent). **Still deferred:** the `query_router` local/global seam (Decision 8). Marker quality reflects
the superseded graph (KI-7). Spec: `docs/specs/feature-7d-knowledge-currency.md`.
**Feature 6 re-point** shipped 2026-06-17 (`wiki.load_communities` clusters by concept-graph communities
behind `WIKI_USE_CONCEPT_COMMUNITIES`, inert by default; cosine fallback).

**Desktop shell migration (M0–M5):** replace Chainlit with a Tauri desktop app + FastAPI backend. The
decision and its sub-decisions (SSE over WebSocket; sidecar-for-release, separate-process-for-dev) are
recorded in `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`. M0–M2 are specced
(`docs/specs/pr-m{0,1,2}-*.md`); M3–M5 are specced one ahead as each predecessor lands. **M0** (lift the
turn orchestration out of `apps/chainlit_app.py` into a UI-agnostic `ChatController`) is the only hard
blocker for everything; **M1** (the 7d contested/superseded marker chip) is the pre-migration demo win
and shares M0's `chunk_key` plumbing. Replaces the "UI = Chainlit" stack row in `.claude/CONTEXT.md`
once M5 lands.

**Later / open:** the 2026-07-02 direction + algorithm review produced remediation increments
**R1–R7** (rows above; plan: `docs/specs/remediation-plan-2026-07.md`) — R1–R4 de-confound and R5 re-runs
the RG-001/008/009 validation below; R6/R7 are independent. Then: the concept-graph **redesign** (curated vocabulary + deterministic
skeleton + confined LLM enrichment, 2026-06-18 — the next concept-graph build, see
`.claude/KNOWN_ISSUES.md`; **build spec `docs/specs/concept-graph-redesign.md`**, design-locked
2026-06-27. **PR-A (Node A — the deterministic, zero-LLM skeleton) BUILT 2026-06-30**
(`concept_skeleton.py` + `scripts/{seed_concepts,build_concept_skeleton}.py` + 4 sidecar tables, 23
tests); remaining: the RG-001/008/009 threshold-setting `--apply` run on the real corpus, then PR-B
(Node B — LLM relation/stance) and the KI-7 retirement of the old `concept_graph.py`),
and the **gap-detection layer** built on top of it (two-tier
deterministic/stochastic, `docs/decisions/ADR-004-gap-detection-layer.md` +
`docs/specs/feature-gap-detection.md` — its deterministic Tier-1 + Tier-2a-floor are the first
increment, blocked on the skeleton + RG-001); Zotero/Calibre ingest adapters (PR 17); an outbound
**MCP-server** interface over `pipeline.py`. Full detail: `docs/archive/doc-assistant-roadmap.md`.

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
