<!-- status: active · updated: 2026-07-21 (compacted: done-row status cells + trailing prose trimmed to pointers; detail in linked specs/ADRs/archives) · class: living -->

# ROADMAP — doc_assistant

The living roadmap: phase map, goals, and the machine-read PR table. This is the **source of
intent**; `docs/decisions/` records the locked design choices (living index `docs/decisions.md`;
pre-cpc rationale frozen at `docs/archive/decisions-monolith.md` — ADR-022), and `AGENTS.md` /
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
- **Phase 7 — Gap detection** — wiki/synthesis layer + cross-document concept graph + the gap-detection layer over them. Status: in progress. *(The 2026-06-18 concept-graph redesign is **built and validated**: Node A skeleton 2026-06-30 + Node B enrichment PR #6, R5 PASS/ADR-008; the superseded open-vocabulary `concept_graph.py` was deleted 2026-07-07 — G1, KI-7 resolved. The gap layer's Tier-1 + Tier-2a floor and stochastic ceiling are built — G2/G5. Remaining: Tier 2b external reach, `citation_missing` floor.)*
- **Phase 8 — UI polish** — settings page exposing the RAG sandbox knobs, plus ongoing chat/citation UX. Status: **open — iterative UI-polish track** (not closed). The five initial tracks (U2/U3/U1/U1b/U1c) are built **and committed** (`09afd0c`, 2026-07-11); the phase deliberately stays open for further UI elements and for the end-to-end verification still owed (live-UI smoke test of the sandbox knobs + provider switch on a real answer turn; RG-012 Tier-2). Living status + backlog: `docs/ui-checklist.md`. *(Chat-UI refinement shipped 2026-07-09 — `ee8fe8d`. **2026-07-10:** three more UI/UX tracks drafted then grilled — settings disclosure + manual dark mode, right-aligned chat bubble, click-to-open citation side panel — `docs/specs/feature-phase8-ui-upgrade.md` (**design-locked** for U1/U1b/U2/U3, build order U2→U3→U1→U1b→U1c). U2 + U3 built 2026-07-10. U1 built 2026-07-11 (SPRINT-010). U1b built 2026-07-11 (SPRINT-011) — the two ADR-010 "must revisit" niche knobs. **U1c built 2026-07-11** (SPRINT-012, ADR-011) — live desktop provider/model switching (v1: already-configured providers only, key stays in `.env`, no restart); v2 (in-app key entry via an OS keychain) is a recorded, un-built north-star.)*
- **Phase 9 — Literature-review generation** — PRISMA-trAIce export. Status: planned.
- **(no phase number) — Extract eval harness to a standalone repo** (Feature 5). Status: planned.
- **(no phase number) — External literature discovery** — mine the enrichment layers (epistemics, authors, keywords/concepts, citation + concept graphs) to find related papers via **open-access APIs** (OpenAlex, Semantic Scholar, Crossref, arXiv, Unpaywall, CORE; Sci-Hub excluded — unauthorized distribution). Status: idea (tray, `docs/ui-checklist.md` §3; 2026-07-13). Needs its own ADR (first outbound-network feature on a local-first app); builds on the metadata-enrichment tray row.
- **(no phase number) — Global CLI + MCP server** — expose the RAG beyond the desktop app (PATH-installed CLI, disabled by default; local stdio MCP tools). Status: **parked — post-review phase** (user call 2026-07-13; tray rows in `docs/ui-checklist.md` §3).
- **(no phase number) — Exploration & epistemics surface (2026-07-21 plan)** — product decisions D1–D3 (`docs/decisions/ADR-027-epistemics-surfacing-split.md`): the app is primarily a corpus exploration/population tool; epistemics **assessment** always visible per-source (E2), epistemics **influence** on the answer layer user-optional (E3); the unrouted enrichment layers (similar docs, citation graph, gaps) get user-facing surfaces (E4/E5) after the correctness batch (E0) and the marker-join fix (E1). Plan: `docs/PLAN_2026-07-21_exploration-epistemics.md`; rows E0–E5 below; the UI-facing pieces join the Phase-8 iterative pool (`docs/ui-checklist.md` §3). Status: planned (E0 first).

## PR order (Claude Code, one PR per session)

Each row is one PR. `Spec` links the code-level contract where one exists; the ADRs in
`docs/decisions/` (and, for pre-cpc features, `docs/archive/decisions-monolith.md`) carry the
architectural context per feature.

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
| M0 | Desktop shell: extract `ChatController` + `TurnResult` (UI-agnostic turn core) | done | `docs/archive/pr-m0-chat-controller.md` |
| M1 | Desktop shell: live 7d epistemics-marker surfacing (pre-migration demo win) | done | `docs/archive/pr-m1-epistemics-markers.md` |
| M2 | Desktop shell: FastAPI backend + SSE boundary | done | `docs/archive/pr-m2-fastapi-boundary.md` |
| M3 | Desktop shell: Tauri frontend (Svelte 5 + Vite) | done | `docs/archive/pr-m3-tauri-frontend.md` |
| M4 | Desktop shell: PyInstaller sidecar packaging + frozen CPU-torch pin | done (`77eb5f9`) | `docs/archive/pr-m4-sidecar-packaging.md` |
| M5 | Desktop shell: delete Chainlit + lift the Python-3.12 pin (KI-2) | done | `docs/archive/pr-m5-decommission-chainlit.md` |
| R1 | Ingest hygiene: strip PyMuPDF4LLM image placeholders + cache-normalization runner (closes KI-14) | done (`4da02e3`) | `docs/archive/remediation-plan-2026-07.md` |
| R2 | Concept presence: word-boundary matching (RG-009 lever; de-confounds R5) | done (`338e55e`) | `docs/archive/remediation-plan-2026-07.md` |
| R3 | Keyword termhood: contrastive scoring (`wordfreq` reference — decided 2026-07-02) + C-value nested-term fix + orphan sweep | built | `docs/archive/remediation-plan-2026-07.md` |
| R4 | Concept skeleton: graded provenance strength (ratio, not boolean) | built | `docs/archive/remediation-plan-2026-07.md` |
| R5 | RG-001/008/009 revalidation run + gap wizard-of-oz → edge-model go/no-go | done | `docs/archive/remediation-plan-2026-07.md` |
| R6 | Core retrieval: BM25 `preprocess_func` + pipeline hygiene (eval-gated, before any weight sweep) | done | `docs/archive/remediation-plan-2026-07.md` |
| R7 | KI-7 containment: 7d marker chip default-off until Node B (decided 2026-07-02: option a — `EPISTEMICS_MARKERS_ENABLED` kill-switch) | done (`591280d`) | `docs/archive/remediation-plan-2026-07.md` |
| S1 | Selective ingestion backend: `SourceFile` registry + selection-scoped ingest (CLI `--files`/`--dry-run`, `GET/PATCH /api/sources`, `POST /api/ingest {paths}`) | done (`2893544`) | `docs/specs/feature-selective-ingestion.md` |
| S2 | Selective ingestion UI: Tauri sources panel (status chips, select-by-status, exclude toggle, ingest-selected) | done (`7224f10`) | `docs/specs/feature-selective-ingestion.md` |
| U2 | UI: chat layout — right-aligned, width-capped user bubble; RAG answer stays full-width, unbounded | done (`7ee1b1e`) | `docs/archive/sprints/SPRINT-008-chat-bubble-layout.md` · `docs/specs/feature-phase8-ui-upgrade.md` |
| U3 | UI: citation side panel — click inline `[n]` to open a slide-over with that source's chunk detail (Chainlit-style); source cards no longer render inline by default | done (`8ba1ffc`) | `docs/archive/sprints/SPRINT-009-citation-side-panel.md` · `docs/specs/feature-phase8-ui-upgrade.md` |
| U1 | UI: Settings disclosure (surface the full read-only knob set) + build the ADR-010 RAG sandbox knobs + a manual System/Light/Dark theme toggle (persisted client-side, not a backend setting) | done (`09afd0c`) | `docs/archive/sprints/SPRINT-010-settings-sandbox-theme.md` · `docs/specs/feature-rag-sandbox.md` |
| U1b | UI: Settings — add the two ADR-010 "must revisit" niche knobs (`EPISTEMICS_MARKERS_ENABLED`, `REVIEWER_EVIDENCE_CHARS`) to the sandbox surface | done (`09afd0c`) | `docs/archive/sprints/SPRINT-011-settings-niche-knobs.md` · `docs/specs/feature-phase8-ui-upgrade.md` (ADR-010 amendment) |
| U1c | UI: Settings — provider/API-key management (Anthropic ↔ Ollama switch, key storage) | done (`09afd0c`) | `docs/specs/feature-provider-switch.md` · `docs/decisions/ADR-011-desktop-provider-apikey-management.md` |
| U4 | UI: "↻ New" conversation-reset button — clears turns + citation panel + composer and mints a fresh `sessionId` so backend context doesn't leak into the next question (session RAG overrides intentionally kept) | done (`9ce5690`) | — (ad-hoc from user request; precursor to the conversation-history sidebar) |
| U5 | UI: app shell + conversation history — left sidebar listing past chats (backend-backed by the existing `AnswerRecord.session_id`), reopen read-only, `↻ New chat`; lays the `sidebar│main│drawer` shell the Library space reuses | done (`9ce5690`) | `docs/specs/feature-conversation-history.md` · `docs/archive/sprints/SPRINT-013-conversation-history.md` |
| L1 | Library space: read-only chunk browser — the reserved "Library" sidebar tab lists ingested docs → open one → read its chunks as parent blocks (each expandable to its child chunks). Read-only, no model, no writes (SQLite `Document` + the live Chroma handle) | done (`aa288d9`) | `docs/specs/feature-library-browser.md` · `docs/archive/sprints/SPRINT-014-library-browser.md` |
| U6 | A/B-compare sandbox (v1: retrieval diff) — a per-turn "Compare" action runs the query under locked defaults vs the session `RagOverrides` and shows the retrieved source sets side-by-side; $0 (no LLM). Full-answer 2× compare deferred | done (`c965418`) | `docs/specs/feature-ab-compare-sandbox.md` · `docs/archive/sprints/SPRINT-015-ab-compare.md` |
| V1 | UI: visual-identity pass V1 — "paper & ink" design tokens (warm ivory/charcoal neutrals + deep-indigo accent + font-stack + 2 shadow tokens), Lucide inline-SVG icons replacing every chrome emoji glyph, Spectral serif on reading surfaces (Inter sans chrome) | **committed `35b8627` (2026-07-13, "UI: Beautification V1 (1/3)")** — frontend-only: re-keyed `app.css` (all four theme blocks) + new `Icon.svelte` + 9 components; **no `src/`/API/wire-type/behavior change**. `svelte-check` 0/0 (123 files); preview-harness-verified $0/offline on the real 76-doc corpus (palette light+dark via computed styles, 4 SVG icons / 0 chrome emoji in DOM, serif seam confirmed on Library `h2`+chunk text = Spectral stack vs Inter chrome, mobile 375px no-overflow, 0 console errors). **Fonts landed — vendored + committed** (4 latin-subset woff2 Spectral 400/italic/600 + Inter variable in `apps/desktop/src/assets/fonts/` + `lib/fonts.css` `@font-face`; all 4 faces verified loaded via `document.fonts`). **Light palette pulled to white/ivory** (user feedback on the live app — the first warm-ivory cut read too beige; `--bg #ffffff` + whisper-ivory surfaces; dark + accent unchanged). **V2** (layout rhythm + header/wordmark + empty states + ~70ch measure) + **V3** (Tauri app icon + branding + audit) queued; stop-early after V1 allowed | `docs/specs/feature-visual-identity.md` · `docs/archive/sprints/SPRINT-016-visual-identity-v1.md` (archived) |
| V2 | UI: visual-identity pass V2 — layout rhythm: header/wordmark (serif `doc_assistant` + indigo book mark), a coherent spacing + type scale (`--space-*`/`--text-*`/`--measure` tokens), restyled empty + first-run states with clickable sample-question chips, and a ~70ch reading measure on answer/excerpt prose. Shell topology stays out (fork #9) | done (`4fd772c`) | `docs/specs/feature-visual-identity.md` · `docs/archive/sprints/SPRINT-017-visual-identity-v2.md` |
| V3a | UI: visual-identity V3a — **rename `doc_assistant` → `Provenote`** (product identity: wordmark treatment B / `index.html` + Tauri window `title` / Tauri `productName` + `identifier` → `com.provenote.desktop` / README + `package.json` description) **+ a cross-screen polish audit**. Internal Python package, npm package name, and the `doc-assistant-api` sidecar binary keep `doc_assistant` (module ≠ product). App icon + branding assets carved to **V3b** | done (`181046c`) | `docs/specs/feature-visual-identity.md` §V3 · `docs/archive/sprints/SPRINT-018-visual-identity-v3a-rename.md` |
| V3b | UI: visual-identity V3b — **Provenote app icon** + full platform icon-set regeneration. A designed **laurel wreath encircling an open book** on a violet rounded tile (open book = reading; laurel = scholarship/provenance; shares the header mark's book motif, a richer gradient jewel), supplied as a user 1024px master → `tauri icon` rewrites the `src-tauri/icons/*` set (PNG sizes + `.ico` 16→256 + `.icns` + Store + android/ios). Installer-identity split (Provenote product ≠ `doc_assistant` code) recorded in **ADR-012** | done (`487f2df`) | `docs/specs/feature-visual-identity.md` §V3b · `docs/decisions/ADR-012-provenote-installer-identity.md` · `docs/archive/sprints/SPRINT-019-visual-identity-v3b-icon.md` |
| G1 | KI-7 retirement: delete `concept_graph.py`, re-point `epistemics.py`/`wiki.py` onto the Node-A/B `concept_skeleton` seam, flip `EPISTEMICS_MARKERS_ENABLED` default-on | done | `docs/archive/sprints/SPRINT-001-retire-concept-graph.md` |
| G2 | Gap-detection layer: deterministic Tier-1 + Tier-2a floor (`gaps.py` + `GapRow` + `scripts/build_gaps.py`); stochastic ceiling out of scope | done | `docs/archive/sprints/SPRINT-002-gap-layer-deterministic.md` |
| G3 | Year-aware skeleton → unlock `superseded_trend`: thread `Document.year` into the skeleton artifact so `node_weights_for_epistemics` marks a node superseded when its contradicting docs are newer than its supporting docs (relative polarity-over-time, parameter-free; fail-safe to `contested` on missing years). Deterministic, CPU-box, $0; `epistemics.py` unchanged | done (`d7528ab`) | `docs/archive/sprints/SPRINT-003-year-aware-superseded.md` |
| G4 | KI-10 frozen OS-trust fix: diagnose (WARN entrypoint + on-proxy turn), hand `AnthropicClient` a guarded `httpx.Client(verify=truststore.SSLContext(...))` (shared helper, reused at the VLM seam), optional branch-A PyInstaller runtime hook; construction-only test (no paid call), on-proxy Step-C verification flips KI-10 | done (`5fc5964`) | `docs/archive/sprints/SPRINT-004-ki10-frozen-os-trust.md` |
| G5 | Gap-detection Tier-2a **stochastic ceiling**: new `gap_suggest.py` — one quarantined, Ollama-default LLM call per Tier-1 `under_connected` node → rated `suggested_link`/`suggested_concept`/`thin_area` `Gap`s (`determinism="stochastic"`, `status="surfaced"`), never auto-written; `--suggest` wires `--provider`/`--model` + `assert_provider_intent`. Tier-2b + the idea-generator out of scope | done | `docs/archive/sprints/SPRINT-005-gap-stochastic-ceiling.md` |
| G6 | Gate `superseded_trend` to a **≥2-dated-docs-per-side** confidence floor, then validate on the real corpus: one guard clause in `_aggregate_direction` demotes the thin single-doc fires G3 allowed (median-of-one is not an aggregate) to `contested`; `2` is a **named structural constant, not a `config.py` tunable** (definitional minimum for a median to aggregate — no eval-harness ceremony); `epistemics.py` + `_graph_version` unchanged | done (`cb166d4`) | `docs/archive/sprints/SPRINT-006-gate-superseded-confidence.md` |
| G7 | Fix `epistemics.concepts_in_text` (KI-15): matches concept **labels**, not the curated skeleton's opaque `Concept.id` UUIDs, against chunk text — the id-matching bug meant the live answer-time contested/superseded_trend chips (PR-M1) never fired on the real corpus, independent of G3/G6's node-level correctness. Shares a new `concept_skeleton.compile_boundary_pattern` with Node A's own presence matcher (R2) so there's one boundary-matching definition, not two | done (`1e1e7eb`) | `docs/archive/sprints/SPRINT-007-fix-epistemics-label-attribution.md` |
| L4 (Phase A) | Library redesign — nav-tree rail + inventory grid + drill-down. Rail becomes a navigation tree (All documents → Collections → Types → Added → Keywords); main pane is a 2-D inventory tile grid of the active collection, with a grid⇄list toggle; opening a doc drills down in place to the existing `LibraryBrowser` chunk view (breadcrumb `Library › Collection › Doc` + Back). Search scopes to the active collection with a "Search all" escape. Frontend-only, $0, no backend change (payload already carries `format`/`added_at`/`keywords`/`folders`); folder population is Phase B | done (`9f597df`) | `docs/specs/feature-library-redesign.md` · L4 Phase A |
| L5 | Metadata enrichment (deterministic-first backfill) + keyword de-noising + grid layout fix — wire the unwired `metadata_extractor` onto `Document` at ingest + a `metadata_enrich.py` sidecar backfill runner (`scripts/enrich_metadata.py`, idempotent); `VENUE_STOPWORDS` + repeated-token filter in `keywords.py`; mode-aware main width + fixed-footprint tiles | done (`8f31fe3`) | — (backlog row, `docs/ui-checklist.md` §3; DEVLOG 2026-07-16) |
| L6 | Manual metadata editing in the Library — user-editable title/authors/year via a `DocumentMeta` **override sidecar** (user-entered wins over extracted, survives re-ingest; the first UI write path into the registry) + reveal-in-explorer | done (`e549254`) | `docs/decisions/ADR-013-document-metadata-editing.md` |
| L7 | Document safe-delete — single-doc delete from the Library `⋯` menu: source file → **OS Recycle Bin** (trash-first, abort-on-fail), then row + meta + chunks + figures + cache; confirmation dialog | done (`95817fc`) | `docs/decisions/ADR-014-document-safe-delete.md` |
| L8 (F1) | **Corpus folders — folders end-to-end (CRUD + Library UI).** | done (`3969adb`) | `docs/specs/feature-corpus-folders.md` · `docs/decisions/ADR-025-corpus-folders-retrieval-scope.md` |
| L9 (F2) | **Corpus folders — query-time retrieval scoping (the integrity piece).** | done (`0e45dd3`) | `docs/specs/feature-corpus-folders-scope.md` · `docs/decisions/ADR-025-corpus-folders-retrieval-scope.md` |
| L10 (F3) | **Corpus folders — demo auto-assign (closes the ADR-025 carve).** | done (`217a122`) | `docs/specs/feature-corpus-folders-demo.md` · `docs/decisions/ADR-025-corpus-folders-retrieval-scope.md` |
| G8 | **ADR-018 graph vocabulary scope** | done | `docs/decisions/ADR-018-graph-vocabulary-scope.md` |
| C1 | cpc big-project layout: `AGENTS.md` entry + `CLAUDE.md` stub + module `CLAUDE.md` files (src/apps×2/scripts) + cpc 1.2.3 vendored on this box + `GLOSSARY.md` filled + conventions tooling separated from scripts | done | `docs/decisions/ADR-021-adopt-cpc-big-project-layout.md` |
| C2 | Docs-system rationalization: per-artifact verdicts (ADRs/specs/sprints/features/DEVLOG/monolith), `decisions.md` → living index + monolith frozen to `docs/archive/decisions-monolith.md`, DEVLOG historical block inverted (newest-first everywhere) | done | `docs/decisions/ADR-022-docs-system-rationalization.md` |
| C3 | Backend restructure: `src/doc_assistant/knowledge/` subpackage for the concept-graph / keywords / wiki / gaps / epistemics feature modules (db/ and ingest/ already exist; RAG pipeline stays top-level) | done | `docs/decisions/ADR-023-knowledge-subpackage.md` |
| C4 | Scale-robustness review: knowledge-layer code vs specs/ADRs under the 0-doc + 10k-doc lenses (corpus-tuned constants, empty-state crashes, O(n²) blowups) | done | `docs/REVIEW_2026-07-19_scale-robustness.md` |
| E0 | Correctness batch (plan 2026-07-21 §E0 = C4-review P0s): curation demote-not-delete (CS-5/KI-20), rebuild coherence (GP-4/KI-21 + KI-17 orphan reconcile), zero-doc honesty (WE-1/WE-9 + GP-7 contract test), `init_db` fail-fast at boot, stance-preserving rebuild (`--apply` without `--enrich` wipes Node B stances) | planned | `docs/PLAN_2026-07-21_exploration-epistemics.md` |
| E1 | Marker-join trustworthiness: KI-8 re-projection (option 2) + PC `_chunk_key` completion (PR-M1 TODO) + WARNING log in `_attach_markers` — prerequisite for E2 | planned | `docs/PLAN_2026-07-21_exploration-epistemics.md` · `docs/specs/feature-7d-knowledge-currency.md` |
| E2 | ADR-027 D3 — always-on source-evaluation strip: per-source coverage/direction + doc year + retrieval-derived signals + `graph_version` freshness hint, below every answer (never gated by E3's toggle); RG-019 denominator with or shortly after | planned | `docs/decisions/ADR-027-epistemics-surfacing-split.md` |
| E3 | ADR-027 D2 — epistemics answer-layer toggle: persisted settings default over the existing `eff_markers_enabled` (U1b keeps winning per-turn); effective value recorded in `AnswerRecord` (ADR-011 discipline) | planned | `docs/decisions/ADR-027-epistemics-surfacing-split.md` |
| E4 | Exploration surfaces: related-papers panel (`similar_docs`/doc_vectors) + citation-network view — computed today, reachable by no endpoint | planned | `docs/PLAN_2026-07-21_exploration-epistemics.md` |
| E5 | Gap list surface — first-class "what to add next" list (kind/concept/status, promote/dismiss); depends on E0's rebuild coherence; pairs with the ui-checklist "Graph destination" rethink | planned | `docs/PLAN_2026-07-21_exploration-epistemics.md` · `docs/decisions/ADR-004-gap-detection-layer.md` |

*(The prose below is a compressed pointer set — 2026-07-21. Full historical narrative for the
closed items lives in the linked ADRs/specs/sprint archives and `docs/archive/doc-assistant-roadmap.md`.)*

**Feature 7d (knowledge-currency layer)** — engine + live answer-time marker surfacing shipped
(2026-06-17 / PR-M1); `superseded_trend` live on the real corpus (G3/G6). **Still deferred:** the
`query_router` local/global seam (Decision 8). Spec: `docs/specs/feature-7d-knowledge-currency.md`.

**Desktop shell migration (M0–M5)** — **done:** Chainlit replaced by a Tauri + FastAPI/SSE shell.
Rationale + sub-decisions in `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`.

**Concept graph + gap detection** — Node A skeleton, Node B LLM enrichment, and the gap layer's
Tier-1 + Tier-2a floor/ceiling are all built (G1/G2/G5; R5 PASS / ADR-008;
`docs/decisions/ADR-004-gap-detection-layer.md`). **Still open:** Tier 2b external reach (deferred —
idea-generator rejected, ADR-004 option 3), Zotero/Calibre ingest adapters (PR 17), an outbound
**MCP-server** interface over `pipeline.py`. Remediation R1–R7: `docs/archive/remediation-plan-2026-07.md`.

## What NOT to do

- Don't refactor the overall architecture. Locked decisions (`docs/decisions/` + the frozen
  monolith) are locked for a reason.
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
