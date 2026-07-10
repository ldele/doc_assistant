<!-- status: active · updated: 2026-07-10 · class: living -->

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
- **Phase 8 — UI polish** — settings page exposing the RAG sandbox knobs. Status: planned. *(Chat-UI refinement shipped 2026-07-09 — `ee8fe8d`. The sandbox knobs are design-locked in `docs/decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md` (accepted) / `docs/specs/feature-rag-sandbox.md` — session-scoped, non-persistent, query-time overrides only; ready to build. **2026-07-10:** three more UI/UX tracks drafted then grilled — settings disclosure + manual dark mode, right-aligned chat bubble, click-to-open citation side panel — `docs/specs/feature-phase8-ui-upgrade.md` (**design-locked** for U1/U1b/U2/U3, build order U2→U3→U1→U1b→U1c; U1c — provider/API-key mgmt — scoped but needs its own ADR); ROADMAP rows U1/U1b/U1c/U2/U3.)*
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
| R5 | RG-001/008/009 revalidation run + gap wizard-of-oz → edge-model go/no-go | **done — PASS (2026-07-02, ADR-008)**; main-corpus decision run (multi-domain home absent): 26-concept curated vocab, K=2/boundary **validated** (density 21.5%, clean retrieval/pose/connectome communities, provenance-strength median 0.52 — R4 discriminates), gap layer healthy (3 single-source + 1 thin bridge) → **RG-008/009 closed, ADR-004 Tier-1 unblocked**; baseline `rg001_concept_skeleton_r5_2026-07-02.md` | `docs/specs/remediation-plan-2026-07.md` |
| R6 | Core retrieval: BM25 `preprocess_func` + pipeline hygiene (eval-gated, before any weight sweep) | **done (2026-07-02, staged)** — `preprocess_func=keywords.tokenize` (casefold + tech-token); dedup full-content hash; `expand_query` non-list bug; parent_text probe; +6 tests, 724 green. Eval: recall@5 0.8775 / @10 0.9069 **identical to control (zero regression)** — benchmark reranker-dominated; fix un-handicaps BM25 ahead of the (follow-up) 0.4/0.6 weight sweep. Baseline `bm25_preprocess_2026-07-02.md` | `docs/specs/remediation-plan-2026-07.md` |
| R7 | KI-7 containment: 7d marker chip default-off until Node B (decided 2026-07-02: option a — `EPISTEMICS_MARKERS_ENABLED` kill-switch) | done (2026-07-02, `591280d`) — ADR-005; default off | `docs/specs/remediation-plan-2026-07.md` |
| S1 | Selective ingestion backend: `SourceFile` registry + selection-scoped ingest (CLI `--files`/`--dry-run`, `GET/PATCH /api/sources`, `POST /api/ingest {paths}`) | planned (spec drafted 2026-07-02, not yet locked) | `docs/specs/feature-selective-ingestion.md` |
| S2 | Selective ingestion UI: Tauri sources panel (status chips, select-by-status/type, exclude toggle, ingest-selected) | planned (needs S1) | `docs/specs/feature-selective-ingestion.md` |
| U2 | UI: chat layout — right-aligned, width-capped user bubble; RAG answer stays full-width, unbounded | **contract written 2026-07-10 → `SPRINT-008` (active, 1st in build order)**; design-locked (grilled 2026-07-10) | `docs/sprints/SPRINT-008-chat-bubble-layout.md` · `docs/specs/feature-phase8-ui-upgrade.md` |
| U3 | UI: citation side panel — click inline `[n]` to open a slide-over with that source's chunk detail (Chainlit-style); source cards no longer render inline by default | **contract written 2026-07-10 → `SPRINT-009` (queued, 2nd)**; design-locked (grilled 2026-07-10) | `docs/sprints/SPRINT-009-citation-side-panel.md` · `docs/specs/feature-phase8-ui-upgrade.md` |
| U1 | UI: Settings disclosure (surface the full read-only knob set) + build the ADR-010 RAG sandbox knobs + a manual System/Light/Dark theme toggle (persisted client-side, not a backend setting) | **contract written 2026-07-10 → `SPRINT-010` (queued, 3rd)**; design-locked (grilled 2026-07-10). Heavy track (backend + Settings + theme) — read-set is spec-led; split-at-the-backend/frontend-seam escape hatch noted in the contract | `docs/sprints/SPRINT-010-settings-sandbox-theme.md` · `docs/specs/feature-rag-sandbox.md` |
| U1b | UI: Settings — add the two ADR-010 "must revisit" niche knobs (`EPISTEMICS_MARKERS_ENABLED`, `REVIEWER_EVIDENCE_CHARS`) to the sandbox surface | **contract written 2026-07-10 → `SPRINT-011` (queued, 4th, needs U1 landed first)**; design-locked (grilled 2026-07-10) | `docs/sprints/SPRINT-011-settings-niche-knobs.md` · `docs/specs/feature-phase8-ui-upgrade.md` (ADR-010 amendment) |
| U1c | UI: Settings — provider/API-key management (Anthropic ↔ Ollama switch, key storage) | **ADR-011 accepted (grilled 2026-07-10, 8 forks)** — phased: v1 = provider **+ model** switch among configured providers (key via `.env`; live swap between turns via a `RAGPipeline` generation-model seam; reviewer follows the switch; inform-only KI-4 posture; keyless provider shown unavailable). Keyring in-app key-entry = v2 north-star. **v1 build spec written 2026-07-10** — buildable (create a SPRINT contract); nominally 5th, but independent of U1's `RagOverrides` path so it can build once U1's Settings rework lands | `docs/specs/feature-provider-switch.md` · `docs/decisions/ADR-011-desktop-provider-apikey-management.md` |
| G1 | KI-7 retirement: delete `concept_graph.py`, re-point `epistemics.py`/`wiki.py` onto the Node-A/B `concept_skeleton` seam, flip `EPISTEMICS_MARKERS_ENABLED` default-on | done (2026-07-07) — KI-7 resolved, ADR-005 superseded | `docs/sprints/SPRINT-001-retire-concept-graph.md` |
| G2 | Gap-detection layer: deterministic Tier-1 + Tier-2a floor (`gaps.py` + `GapRow` + `scripts/build_gaps.py`); stochastic ceiling out of scope | done (2026-07-07) — `min_degree=3` from the corpus's own degree distribution, `tests/eval/baselines/gap_min_degree_2026-07.md` | `docs/sprints/SPRINT-002-gap-layer-deterministic.md` |
| G3 | Year-aware skeleton → unlock `superseded_trend`: thread `Document.year` into the skeleton artifact so `node_weights_for_epistemics` marks a node superseded when its contradicting docs are newer than its supporting docs (relative polarity-over-time, parameter-free; fail-safe to `contested` on missing years). Deterministic, CPU-box, $0; `epistemics.py` unchanged | **code built (2026-07-08, staged — awaiting review)** — `load_doc_years` + `_aggregate_direction` (median-vs-median, strict-newer, fail-safe) shipped; `_graph_version` now busts on a year change; +10 tests, 783 green; rule recorded in `tests/eval/baselines/superseded_year_rule_2026-07.md`. **Host apply pending** (`build_concept_skeleton --apply` + `compute_epistemics --apply` on the real corpus — user's run after review) | `docs/sprints/SPRINT-003-year-aware-superseded.md` |
| G4 | KI-10 frozen OS-trust fix: diagnose (WARN entrypoint + on-proxy turn), hand `AnthropicClient` a guarded `httpx.Client(verify=truststore.SSLContext(...))` (shared helper, reused at the VLM seam), optional branch-A PyInstaller runtime hook; construction-only test (no paid call), on-proxy Step-C verification flips KI-10 | **done (2026-07-09, staged — awaiting review) — KI-10 RESOLVED** — branch B: `llm.os_trust_http_client()` (SDK `DefaultHttpxClient(verify=truststore.SSLContext(...))`, `sys.frozen`-gated so dev/test behaviour is unchanged) reused at both raw-SDK Anthropic seams (`llm.py` + `ingest/figures.py`); +2 construction-only tests (no paid call), gate green (791 passed). **Step C run on-proxy:** re-froze + drove one real paid `/api/chat` turn on this TLS-MITM box → HTTP 200, tokens, grounded cited answer, ≈$0.0059 billed, ZERO `CERTIFICATE_VERIFY_FAILED` (the 2026-06-25 turn failed the handshake). `.claude/KNOWN_ISSUES.md` KI-10 → RESOLVED; frozen paid RG-011 number in `.claude/RIGOR_TODO.md` | `docs/sprints/SPRINT-004-ki10-frozen-os-trust.md` |
| G5 | Gap-detection Tier-2a **stochastic ceiling**: new `gap_suggest.py` — one quarantined, Ollama-default LLM call per Tier-1 `under_connected` node → rated `suggested_link`/`suggested_concept`/`thin_area` `Gap`s (`determinism="stochastic"`, `status="surfaced"`), never auto-written; `--suggest` wires `--provider`/`--model` + `assert_provider_intent`. Tier-2b + the idea-generator out of scope | **done (2026-07-08)** — built + gate-tested offline (scripted `LLMClient`, +18 tests, 773 green) on the RTX/Ollama box, then real-validated there: `--apply --suggest` (llama3.1:8b) → 12/12 `under_connected` concepts suggested, $0, ~51s; baseline `tests/eval/baselines/gap_suggest_ollama_2026-07-08.md` | `docs/sprints/SPRINT-005-gap-stochastic-ceiling.md` |
| G6 | Gate `superseded_trend` to a **≥2-dated-docs-per-side** confidence floor, then validate on the real corpus: one guard clause in `_aggregate_direction` demotes the thin single-doc fires G3 allowed (median-of-one is not an aggregate) to `contested`; `2` is a **named structural constant, not a `config.py` tunable** (definitional minimum for a median to aggregate — no eval-harness ceremony); `epistemics.py` + `_graph_version` unchanged | **done (2026-07-08, staged — awaiting review)** — `MIN_DATED_DOCS_PER_SIDE=2` guard added; G3's 1-v-1 fixture updated (now asserts demotion) + 2 new fixtures added, +3 unit / +1 integration test, 786 green. Real host-apply run (found + fixed a planning gap first: `--apply` alone wipes Node B's stance data — the correct command is `--apply --enrich`): **before (G3 alone) 26 superseded_trend nodes → after (G6 gate) 9** (17/26, 65%, were the demoted single-doc case — confirms the review finding). Hand-audit: all 9 survivors have a genuine multi-year spread both sides. Baseline updated: `tests/eval/baselines/superseded_year_rule_2026-07.md`. **Found in passing, logged not fixed (out of scope):** `epistemics.concepts_in_text` matches concept UUIDs, not labels, against chunk text — live marker surfacing has been silently dark on the real corpus since G1; `.claude/KNOWN_ISSUES.md` KI-15 | `docs/sprints/SPRINT-006-gate-superseded-confidence.md` |
| G7 | Fix `epistemics.concepts_in_text` (KI-15): matches concept **labels**, not the curated skeleton's opaque `Concept.id` UUIDs, against chunk text — the id-matching bug meant the live answer-time contested/superseded_trend chips (PR-M1) never fired on the real corpus, independent of G3/G6's node-level correctness. Shares a new `concept_skeleton.compile_boundary_pattern` with Node A's own presence matcher (R2) so there's one boundary-matching definition, not two | **done (2026-07-08, staged — awaiting review)** — `concepts_in_text` now takes `{node_id: label}`; +4 tests (UUID-id fixture, `gpt-4`/`gpt-4o` boundary case, shared-pattern guard, end-to-end integration), 790 green. Real corpus (same skeleton G6 built, no rebuild): **0 → 4008 chunks with a claim, 0 → 3334 marked** (of 6215); manually spot-checked one marked chunk — all 6 attributed labels genuinely present, no false positives. `.claude/KNOWN_ISSUES.md` KI-15 → RESOLVED | `docs/sprints/SPRINT-007-fix-epistemics-label-attribution.md` |

**Feature 7d (knowledge-currency layer):** engine shipped 2026-06-17 (`epistemics.py` + `chunk_epistemics`
sidecar + polarity-aware concept graph + reviewer `contested_evidence` tag). **Live answer-time marker
surfacing shipped 2026-06-22 (PR-M1)** — sources carry `contested`/`superseded_trend` chips via
`ChatController` (flat: `chunk_key` join; PC: text containment), synthesis untouched (byte-identical when
absent). **Still deferred:** the `query_router` local/global seam (Decision 8). **KI-7 retired
2026-07-07 (G1):** marker data now sources from the Node-A/B concept skeleton, not the deleted
open-vocabulary graph; `EPISTEMICS_MARKERS_ENABLED` defaults on. `superseded_trend` is now
reachable in code — **G3** (`docs/sprints/SPRINT-003-year-aware-superseded.md`, code built
2026-07-08, staged) threads `Document.year` into the skeleton (a deterministic year-aware pass,
no LLM, not the Node-B stance pass); it only fires once the host `--apply` run (below) puts real
years into a live `skeleton.json`. Spec: `docs/specs/feature-7d-knowledge-currency.md`.
**Feature 6 re-point** shipped 2026-06-17 (`wiki.load_communities` clusters by concept-skeleton
communities behind `WIKI_USE_CONCEPT_COMMUNITIES`, inert by default; cosine fallback).

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
tests); RG-001/008/009 threshold-setting run **done** (R5 PASS, ADR-008) and **PR-B (Node B — LLM
relation/stance) BUILT + merged** (PR #6 `6679540`: `17d6757` enrichment, `caa7ef4` corroborated-cap +
token budget, `894233f` vocab prune/merge — `concept_skeleton_enrich.py`, Ollama-default). **KI-7
retirement of the old `concept_graph.py` + the markers-on flip DONE (2026-07-07, G1, SPRINT-001).**
**Gap-detection layer's deterministic Tier-1 + Tier-2a-floor DONE (2026-07-07, G2,
SPRINT-002)** — `gaps.py` + `GapRow` + `scripts/build_gaps.py`
(`docs/decisions/ADR-004-gap-detection-layer.md` + `docs/specs/feature-gap-detection.md`).
**The Tier-2a stochastic ceiling DONE (2026-07-08, G5, SPRINT-005)** — `gap_suggest.py`
(quarantined, Ollama-default LLM suggestions atop G2's floor), real-validated on this RTX/Ollama
box (`tests/eval/baselines/gap_suggest_ollama_2026-07-08.md`). Remaining: Tier 2b (external
reach) — deferred, the idea-generator is rejected for it (ADR-004 option 3). Zotero/Calibre ingest
adapters (PR 17); an outbound
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
