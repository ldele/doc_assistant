<!-- status: active · updated: 2026-07-23 · class: disposable -->

> **✅ COMPLETE 2026-07-23 — kept as the provenance record for its rows.** Every increment this plan
> sequenced is done and committed: **E0** correctness batch, **E1.1/E1.2** marker-join + `_handle_rag`
> extraction, **E2** source-evaluation strip, **E3** answer-layer toggle, **E4** Connections panel, and
> the **RH1** retrieval-hygiene / APIRouter-split follow-ups (see `docs/ROADMAP.md` E-rows + the SESSION
> baton). The "After the E-track" scale mechanics (KI-18/19) remain open but are tracked in
> `.claude/KNOWN_ISSUES.md` / `.claude/RIGOR_TODO.md`, not here. Do not plan *new* work off this file —
> it stays only because the E0/E1/E2 specs and the E4/E5/RH1 ROADMAP rows cite it as their `plan:` origin.

# PLAN 2026-07-21 — exploration & epistemics surface (external review → wired increments)

Origin: external code/docs review (Cowork session 2026-07-21) of the full repo, cross-checked
against `docs/ROADMAP.md`, `.claude/KNOWN_ISSUES.md`, `.claude/RIGOR_TODO.md`, and
`REVIEW_2026-07-19_scale-robustness.md`. This plan does **not** repeat the C4 review's findings —
it sequences them together with one gap the existing trackers don't name, under the product
decisions locked in ADR-027. Routed to ROADMAP rows **E0–E5**; UI-facing pieces join the Phase-8
iterative pool (`ui-checklist.md` §3).

**Review headline:** the knowledge layer is richly built but almost entirely disconnected from
the answer/exploration flow. Concept graph, citation graph, doc similarity, gaps — computed,
none routed to the user beyond the read-only graph view. Epistemics markers are the one live
connection, and that join silently under-reports (KI-8). For a product whose primary use is
**exploring and populating the corpus** (ADR-027 D1), the unrouted enrichment layers are the
highest value-per-line-of-code in the repo.

## Product decisions (ADR-027)

- **D1** — primary use case: exploration & population of the documentation corpus (search
  first). Gap surfacing, related papers, and source evaluation move ahead of answer-polish.
- **D2** — epistemics **influence** on the answer layer is a user option (persisted setting
  over the existing `eff_markers_enabled`; effective value recorded in provenance).
- **D3** — epistemics **assessment** is always-on: a per-source evaluation strip below each
  answer (coverage/direction + doc year + retrieval-derived signals + `graph_version`
  freshness hint). Affordable by construction: precomputed sidecar, lookup joined against
  TOP_K=10 — no LLM call.

## E0 — correctness batch (P0; days; no eval ceremony)

The C4 review's P0 list, unshipped as of 2026-07-20, plus one boot-time item. D3 raises the
stakes: an always-on evaluation strip must not show false data.

1. Curation demote-not-delete (CS-5 / KI-20) — stages 1–3 `--apply` currently hard-deletes
   curated vocabulary against ADR-018; the graph was already lost once to KI-25.
2. Rebuild coherence (GP-4 / KI-21 + KI-17) — the in-app rebuild route runs `build_gaps` and
   the orphan-reconcile pass; today the view serves 27 gaps against 13 nodes, 10 orphaned.
3. Zero-doc honesty (WE-1, WE-9) + pin the 0-doc contract with a test (GP-7).
4. `init_db()` fails at boot, not at first turn — migration failure is currently swallowed; a
   failed answer-path migration then breaks every turn at runtime instead of at startup.
5. **Stance freshness footgun** (found in the G6 run, still open as a workflow trap):
   `build_concept_skeleton --apply` without `--enrich` silently wipes Node B stance data —
   epistemics then degrades corpus-wide. Make the rebuild path stance-preserving (or
   auto-`--enrich`, or refuse with a message); the in-app rebuild route must not have this trap.

## E1 — marker-join trustworthiness (prerequisite for E2)

KI-8, corrected 2026-07-19: parent-boundary straddling chunks silently **lose** markers
(~40% of marked chunks affected — systematic false negatives in the flagship integrity
feature), and `_chunk_key` still returns `None` for all PC chunks (PR-M1 TODO), so every
live join runs the coarse text-containment fallback. Fix = re-projection (KI-8 option 2)
+ finish the PC chunk-key mapping. Also: `_attach_markers`' blanket `except: return`
gains a WARNING log (silent failure + always-on strip = silently lying UI).

## E2 / E3 — the ADR-027 surfaces (current UI track)

- **E2 (D3):** extend `_build_source_views` to join per-source epistemics status + doc year;
  one Svelte component below the answer (SourceCard/SourcePanel siblings). Freshness hint from
  `graph_version`. Honesty gate: E1 shipped first; RG-019 with or shortly after (at 53.6% of
  chunks marked `contested`, the strip is wallpaper until the denominator lands — the review
  routed RG-019 as measurement-gated P2; this plan recommends pulling it forward).
- **E3 (D2):** persisted settings toggle over the existing plumbing; record the effective
  value in `AnswerRecord` (ADR-011 discipline). U1b's per-turn sandbox knob keeps working
  and wins for the session. Small, UI + provenance only.

## E4 / E5 — the exploration surfaces (D1)

- **E4:** related-papers panel + citation-network view. `similar_docs`, doc_vectors, and the
  citation graph are implemented and populated by scripts but reachable by **no endpoint** —
  computed and dead to the UI. ~One endpoint + one component each. Later feeds the External-
  literature-discovery idea (its own ADR — first outbound-network feature).
- **E5:** gap list surface — the "what documentation to add next" workflow, first-class:
  kind, concept, status, promote/dismiss. Currently gaps are visible only inside the graph
  view. Depends on E0-2. Pairs with the ui-checklist "Graph destination" rethink
  (per-folder concepts, ADR-025 alignment).

Deferred (recorded, not scheduled): graph-aware retrieval expansion (query → 1-hop skeleton
neighbours as extra terms, flag-gated, eval-harness-validated). Answer-layer work — post-D1
priority, and it generates the baseline any learned approach (GNN) would have to beat. A GNN
itself stays out: 13 nodes is nothing to learn on; the deterministic skeleton is the correct
baseline layer (consistent with "What NOT to do": the graph is not a graph database).

## After the E-track (unchanged from existing trackers, sequenced)

- **Scale mechanics** (KI-18 P1 list): inverted index for structural matching first
  (`match_presence` + WE-3 — also keeps D3's offline precompute viable as the corpus grows),
  then Chroma read pagination (CS-2/KW-1), then CS-1, KW-2→ANN, and the LLM-budget ADR
  (CS-4/GP-2/WE-10, one ADR). Tuned constants stay rigor-gated (RG-016..018).
- **Retrieval hygiene** (review findings, small): scoped-ensemble memo single slot → small
  LRU (2–4 scopes; alternating between two folders currently rebuilds BM25 every turn);
  cap reranker input under multi-query (currently unbounded — 4× candidates = 4× rerank cost).
- **Code health, opportunistic, in files the E-track touches anyway:** extract `_handle_rag`
  (~285 lines, ~14 responsibilities) into staged steps *before* wiring E2/E3 into it; split
  `library.py` / `api/main.py` by domain (`APIRouter`) before E4/E5 add endpoints; consolidate
  the 4 copies of tolerant-JSON parsing, 2 of `_cosine`, the duplicated `RagOverrides`
  mapping and dedup-key derivation (2+ concrete cases each — no speculative abstraction).
- **Ship gate** (unchanged): RG-012 Tier-2 clean-box run; fully-local provider path for the
  no-key 60-second demo (closes the KI-4 credit-leak class). The D3 strip is a demo asset —
  provided E0/E1 landed so it shows true data.

## Suggested sequence

1. E0 (all five — small, no eval). 2. E1, extracting `_handle_rag` on the way.
3. E2 + E3 (+ RG-019 pulled forward). 4. E4, E5, retrieval hygiene. 5. Scale mechanics
(inverted index + pagination), then reassess against Phase 8/9.
