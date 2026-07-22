<!-- status: active · updated: 2026-07-22 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> Entries **2026-07-14 and earlier** live in [`docs/archive/DEVLOG-archive-001.md`](archive/DEVLOG-archive-001.md)
> (moved verbatim 2026-07-21). This file keeps 2026-07-15 onward.

---
## 2026-07-22 — APIRouter split of `apps/api/main.py` (pure refactor, behavior-identical)

The refactor the plan deferred through E4 and E5 (a behavior-preserving move doesn't belong inside a
feature diff), now its own increment. `apps/api/main.py` had grown to **1009 lines / 42 routes** in
one file; split into per-domain `APIRouter` modules. **Staged, not committed** (cpc §13). Full suite
**1218 passed** (unchanged from pre-split — the behavior-preservation proof); ruff · `ruff format` ·
`mypy --strict src` (65) · bandit. Frontend untouched.

**Structure.** `main.py` **1009 → 159 lines** — now only `create_app`: the lifespan (schema migration
+ controller), `app.state` wiring + the `ingest_fn`/`rebuild_graph_fn`/`controller_factory` test seams,
CORS, and seven `include_router` calls. Routes moved to `apps/api/routers/{health,chat,conversations,
library,concepts,settings,sources}.py` — one `APIRouter` per domain. Cross-router glue (the `app.state`
status dataclasses + their `202+poll` serializers, the settings read view, the lazy default job
runners) moved to `apps/api/services.py`. **Dependency direction is one-way:** `main` and routers import
from `services`; routers never import from `main` (no cycle). `chat`-only helpers (`_sse`,
`_event_stream`) live in the chat router.

**Behavior-preserving contract.** Every handler already read state via `request.app.state.*` + module
helpers, never a `create_app` local — so the move is `@app.get` → `@router.get`, nothing more. Route
declaration order preserved within each router (the load-bearing one: `/api/concepts/gaps` before the
parameterised `/api/concepts/{concept_id}/…`). `create_app`'s signature is byte-identical (the public
seam — 49 test call-sites unchanged). `_settings_view` + `_default_rebuild_graph` are re-exported from
`main` (via `__all__`) so the two tests importing them by name still resolve; `init_db` stays imported
in `main` so the startup-migration monkeypatch target holds.

**Test churn (2, both legitimate consequences of a moved symbol).** `test_figure_served_and_missing`
patched `apps.api.main.load_figure_image_paths` → repointed to `apps.api.routers.chat` (where the
figure route now imports it). No other monkeypatch targeted a moved symbol (grep-verified).

**Live $0 smoke.** Booted the real app; one endpoint per router returned correctly (health 200, a
provenance-source 404-as-expected, all domain reads 200); real data flows (47 docs / 15 gaps / 16 039
chunks via the split app); 0 server errors. Route enumeration: 42 API routes, same set as pre-split.

**Rejected.** Distributing the status dataclasses across their routers and having `main` import from 4
routers (a shared `services.py` is simpler to reason about + keeps the import graph acyclic); a
top-level `routers/` re-export shim (the explicit `include_router` list in `create_app` is the clearest
manifest). **Opens.** `library.py` (the backend service module, ~1.7k lines) is the other half of the
plan's "split by domain" note — a separate, larger refactor, not bundled here. `apps/` isn't in the
`mypy --strict` gate (only `src`); a `ServerSentEvent` attr-export stub would be needed to add it —
noted, not done.

---
## 2026-07-22 — E5: first-class gap list + triage (ADR-004 / ADR-017 C1) — panel in the Graph view

ROADMAP row E5 (last E-track row). Gaps were computed (ADR-004) but reachable only *embedded* in the
graph payload, joined to nodes — no first-class list, no triage. Now: a triageable gap list in the
Graph view with a durable dismiss/promote override. **Staged, not committed** (cpc §13). Full suite
**1218 passed** (+10 backend); ruff · `ruff format` · `mypy --strict src` (65) · bandit · **svelte-check
0/0 (140)** · **npm test 41/41** (+7 gaps.ts) · docs 0/0 · integrity 0/0.

**The load-bearing decision (ADR-017 C1): triage is a user override in its OWN sidecar.** New
`GapTriage` table keyed on `(concept_id, kind)` — *not* `GapRow.status`. Deterministic `gaps` rows are
delete-and-replaced on every `build_gaps` run (regenerable, ADR-004), and a rebuild is part of the
acquire loop the surface exists to close, so a dismissal written onto the row would not survive it.
`load_gaps` now resolves the **effective** status = `override ?? row.status` (one source of truth — the
graph node-badge lens and the list agree; a dismissal removes the gap from both). Mirrors `DocumentMeta`
(ADR-013). Additive table via `create_all`, no migration. `set_gap_status(surfaced)` = reset = delete the
override. Stochastic rows keep their own persisted status when un-overridden (C1's "don't double-write").

**Backend/API.** `load_gap_overrides` / `set_gap_status` (gaps.py); `load_gap_list` +
`GapListItem` (concept_graph_view.py) resolves the concept UUID → label (the flat list needs it; the
graph carries labels only on nodes, KI-15) with a fallback to the id itself for stochastic candidates.
`GET /api/concepts/gaps` (label + effective status; empty when unbuilt) + `POST /api/concepts/gaps/triage`
(`{concept_id, kind, status}`; `surfaced` resets; the `Literal` enum 422s a bad status).

**Frontend.** Extracted the gap taxonomy to a **pure, node-tested** `lib/gaps.ts` (GAP_META + rank/tone
+ `orderGaps`/`gapVisible`) — shared by ConceptGraph (which now imports it instead of an inline copy) and
the new `GapList.svelte`. ConceptGraph gains a rail-mode toggle **Concepts | Gaps**; `visibleGaps` now
drops `dismissed` gaps from the node lens too. `GapList` is **self-contained** (fetches its own
effective-status data, owns its triage writes) so it can move out of the graph without coupling when the
Graph-destination fork settles — the recorded iteration gate. RG-014 presentation preserved: strong
list-shaped kinds first, `under_connected` opt-in; dismissed hidden (recoverable via a "Show dismissed"
toggle); Promote/Dismiss/Reset per row; `onSelectConcept` jumps the ego view.

**Live $0 verify (real corpus, 16k chunks).** API: 15 gaps served with labels + effective status.
Triaged a real UUID-keyed `single_source` gap → dismissed → **triggered a real graph rebuild (202+poll,
deletes+rebuilds the deterministic rows) → the dismissal SURVIVED** (C1's whole point, proven end-to-end).
Reset deletes the override. `bogus` status → 422. UI: rail toggle renders; Dismiss dropped the row from
the worklist (15→14 open) and surfaced a "Show dismissed" toggle; the toggle reveals it with a dismissed
tag + Promote/Reset; Reset restores it; a concept link jumps the ego view to that concept (SVG rendered).
0 console errors; 375px 0-overflow; dark tokens resolve. Box left clean (0 overrides).

**Rejected.** Storing triage on `GapRow.status` (dies on the rebuild — the exact B14 trap the grill
originally got wrong); a second corpus-wide graph surface (RG-014: the gap payload is list-shaped, and the
Graph-destination fork is parked); the plan's `APIRouter` split of `api/main.py` "with E5" — a
behavior-preserving ~200-line refactor doesn't belong inside a feature diff; owed as its own increment
before the next route-heavy work.

**Opens.** The `APIRouter` split (own commit). Triage currently has no bulk action; the gap→acquisition
loop (B13, own ADR) attaches to `promoted` later. `suggested_concept` volume swings with `gap_suggest`
runs — the list shows them but leads with the deterministic strong kinds.

---
## 2026-07-22 — E4: document-connections panel (ADR-027 D1) — the exploration surface, per-doc

ROADMAP row E4. The plan's headline gap closed: `doc_similarities` (470 edges, all 47 docs) and
the citation graph (2,859 extracted / 9 resolved in-corpus) were computed and reachable by **no
endpoint** — dead to the UI. Now: one endpoint + one panel in the Library document view.
**Staged, not committed** (cpc §13). Full suite **1208 passed** (+9); ruff · `ruff format` ·
`mypy --strict src` (65) · bandit · **svelte-check 0/0 (138)** · **npm test 34/34**.

**Shape (user decision, 2026-07-22): per-doc panel, NOT a top-level network mode.** At 9 resolved
citation edges a corpus-wide network view renders near-empty (the exact "empty Graph reads as
broken" complaint), and it would entangle the parked Graph-destination fork (ADR-025 fork 5).
**The graph/navigation treatment stays an OPEN GATE, per the user** — v1 is deliberately
list-shaped, and a depth-1 ego graph is exactly `cites` + `cited_by`, so a later iteration
(SVG ego view, corpus-level view, depth param) reads the same bundle without an API break.
Recorded in ui-checklist.

**Backend.** `library.document_connections(doc_id, related_limit, external_cap, embedding_model)`
→ `DocConnections | None` — assembles the *existing* read models (`similar_docs` + `cites_out`
split internal/external + `cited_by` deduped by source doc with `n_citations`); None = unknown doc
(404). The similarity read is scoped to the active embedder (edges from another model describe a
different geometry — never mixed). `external_refs` = the titled slice of unresolved citations,
capped at `EXTERNAL_REFS_CAP=50` (a wire-size bound, not corpus-tuned) with `external_total`
alongside — no silent truncation. Empty sidecars → empty lists (0-doc contract).

**Wire + frontend.** `GET /api/library/documents/{id}/connections` → `DocConnectionsPayload`
(4 sub-payloads); `types.ts` mirrors. New `DocConnections.svelte` in `LibraryBrowser` under the
doc header: Related papers (cosine chip) · Cites / Cited by (in-library links, ×n badge) ·
collapsed "References (N extracted, not in your library)" with a showing-N-of-M cap note.
Advisory: load failure degrades to one quiet line; all-empty bundle renders one honest muted
line. Click-through = `onOpenDocument` threaded from App.svelte — the D1 hop (doc → related doc,
panel reloads) is the feature.

**Live $0 verify (real corpus).** DPR doc: all 4 sections (10 related · cites → dpr/rag/sbert
set · cited-by · "References (33 extracted)"); clicked the top related link → doc view swapped to
the RAG paper, panel reloaded (13 links) — the exploration loop works. b2a1754a via API: 10
related (top specter2@0.952), 3 in-corpus cites, 50-of-60 external cap, 404 probe clean. 0 console
errors; 375px 0-overflow (also with the refs list open); dark-theme tokens resolve.

**Rejected.** Top-level citation-network mode (near-empty at 9 edges + entangles the parked
Graph-destination fork); the APIRouter split of `api/main.py` before adding the route (the plan
suggests it "before E4/E5" — one ~15-line route doesn't justify a ~200-line refactor riding the
same diff; owed before E5's larger route surface, see Opens); returning a separate `graph` payload
(depth-1 ego ≡ cites+cited_by — derivable, YAGNI).

**Opens.** The graph/navigation iteration gate (user, explicit). E5 (gap list) next — do the
`APIRouter` split with it. Citation resolution is thin (9 edges): re-running
`scripts/extract_citations.py --apply` after corpus growth (or improving the resolver) would
enrich the panel for free. External refs later feed the B13 gap→acquisition loop (own ADR).

---
## 2026-07-22 — E3: persisted epistemics answer-layer toggle (ADR-027 D2) — full-stack

ROADMAP row E3 (ADR-027 D2 — the last unbuilt half of the surfacing split). Whether epistemics
*influences* the answer layer (the marker chips) is now a persisted user setting, layered as a
**three-layer resolution**: U1b per-turn override > persisted setting > `EPISTEMICS_MARKERS_ENABLED`
env default. The effective value is snapshotted per turn into `AnswerRecord` (ADR-011 instrument
discipline). **Staged, not committed** (cpc §13). Full suite **1199 passed** (+17); ruff ·
`ruff format` · `mypy --strict src` (65) · bandit · **svelte-check 0/0 (137)** · **npm test 34/34**
· docs_check 0/0 · integrity_check 0/0 (492).

**Backend.** `app_settings.get/set_markers_enabled` + `effective_markers_enabled()` (the
`effective_llm` pattern — persisted if set, else config; re-read each turn so a toggle applies
next-turn, no restart). `_resolve_turn_knobs` baselines the markers knob on the persisted-effective
default; `_overrides_note` now takes `markers_default=` and compares against **it**, not the env
constant — a persisted choice is the user's default, and stamping it "Session override (this answer
only)" on every turn would be a provenance lie (the note fires only on a genuine per-turn U1b diff).
New additive nullable `answer_records.epistemics_markers_enabled` (`_ADDITIVE_COLUMNS`; NULL = pre-E3
row = honestly "unknown"), recorded on **both** result paths (AI + human-mode) via
`_ProvenanceInputs.markers_enabled` / `record_answer(epistemics_markers_enabled=)`, read back in
`AnswerProvenance`. Deliberately NOT folded into `prompt_version` (same rule as the F2 scope: it
never reaches the prompt).

**Wire + frontend.** `SettingsUpdate += epistemics_markers_enabled` (validator: the toggle alone is a
valid body; empty body still 422). `_settings_view` serves the **effective** value (the raw constant
would go stale on the first toggle — the provider/model rule), which also makes the U1b sandbox
baseline correct for free. New "Answer epistemics" Settings section (persisted toggle, snaps back on
error) between Provider and RAG sandbox; `api.ts setMarkersEnabled`; `types.ts` meaning-shift note.

**Tests (+17).** Precedence unit tests (persisted beats config both directions; non-bool stored value
degrades to "never set"); knob-layering tests incl. the two provenance-honesty pins (persisted-off →
`overrides_note == ""`; override-vs-persisted diff reads `(default False)`); provenance round-trip
(False recorded / legacy row → None); API round-trip (POST persists → GET serves effective; toggle-only
body valid); the D3 boundary from the persisted side (persisted-off still attaches the evaluation
strip); + an AnswerRecord snapshot assertion on a real (fake-RAG) turn. **Test-infra hardening:** 4
files gained an autouse `SETTINGS_PATH` isolation fixture — `_resolve_turn_knobs`/`_settings_view` now
read the persisted setting, so an unpatched suite would read the dev box's real `settings.json` (the
KI-22 class of environmental mislabel, preempted). Existing tests patching the dead
`chat_controller.EPISTEMICS_MARKERS_ENABLED` namespace copy were repointed at `config.` (the layer the
resolution actually reads).

**Live $0 verify (real corpus, ollama/llama3.1:8b — KI-4 honored, provider checked first).** Settings
UI: new section renders (light, 375px → 0 overflow, 0 console errors); toggle off → `settings.json`
gains `epistemics_markers_enabled: false` → GET serves False → the sandbox baseline follows; sandbox
override flips session-only (file untouched — the boundary held live). One real SSE turn (step → 79
tokens → result → done) recorded `epistemics_markers_enabled = 0` on the newest `answer_records` row.
Boot migration verified live: `schema_migrated_at_startup` added the new column (plus, notably,
`retrieval_scope_json` + `chunk_key` — this box's real DB had never received the E1.1/F2 columns; the
KI-23 in-app migration caught all three). State restored after verify (key removed → "never set");
side effect: one `e3-verify` conversation in this box's history.

**Fix in passing.** `.env.example`'s `EPISTEMICS_MARKERS_ENABLED` block claimed "default off (R7)" —
stale since the KI-7 retirement flipped the default to true; rewritten to state the real default + the
three-layer resolution + the D2/D3 boundary. ROADMAP E0–E2 status cells trued up to their commit
hashes (they read "staged" but the user committed them 2026-07-21).

**Rejected.** Folding the flag into `prompt_version` (pollutes eval joins; never reaches the prompt);
an unset/revert-to-default API affordance (YAGNI — the UI only sets true/false, and "never set" is
recoverable by deleting the key); gating the D3 strip (ADR-027's boundary is explicit).

**Opens.** E4 (exploration surfaces) / E5 (gap list) per the plan's sequence; RG-019 still deferred
(measurement-gated, moot at 0 stance); Node-B stance regen on the RTX box to make the chips + strip
non-trivial on real data.

---
## 2026-07-21 — E2: always-on source-evaluation strip (ADR-027 D3) — full-stack

Spec: `docs/specs/feature-e2-source-evaluation-strip.md` (ROADMAP row E2 · ADR-027 D3). A per-source
evaluation strip below every chat answer — always-on, $0 (sidecar lookup joined against TOP_K, no
LLM). Honest to build now: E1.1 made the marker join trustworthy, E1.2 gave the source path clean
seams. **Staged, not committed** (cpc §13). Full suite **1182 passed / 1 skipped** (+6); ruff ·
`ruff format` · `mypy --strict src` · bandit · **svelte-check 0/0** · **npm test 34/34** · docs 0/0 ·
integrity 0/0.

**The boundary (ADR-027).** D3 (this strip) is **always-on** assessment; D2/E3 (the answer-influence
toggle over `eff_markers_enabled`) governs the *answer-surface* marker chips only and **never hides
the strip**. So the strip's per-source `evaluation` attaches unconditionally; the existing `markers`
field stays gated by the toggle — both derived from **one** scoped sidecar read.

**Backend (E2a/E2b).** `epistemics.load_source_evaluations(chunk_keys)` — a scoped, indexed read
(unlike the full-scan marker index; KI-18) returning per-key `ChunkEval(coverage, superseded,
n_claims)` + the sidecar `graph_version`; `coverage` = contested > corroborated > unique.
`current_graph_version()` (a 1-row `concept_presence` read) drives the freshness compare;
`library.document_years(ids)` a scoped year join. `_attach_markers` → `_attach_source_evaluation`
(always-on): sets `sv.evaluation` + `sv.reranker_score` for every source, sets `sv.markers` only when
`markers_enabled`, returns `SourceEvalSummary(graph_version, stale)`; returns `None` (no strip) when
no concept graph is built (0-doc/fresh). WARNING-logged on failure (never a silent lying UI).
`SourceView` gains `evaluation`/`reranker_score`; `TurnResult` gains `source_eval`.

**Wire (E2c).** `SourceViewPayload += evaluation (SourceEpistemicsPayload) + reranker_score`;
`TurnResultPayload += source_eval (SourceEvalSummaryPayload)`; `types.ts` mirrors both. Replay
(`ConversationSource`) stays degraded (no strip).

**Frontend (E2d).** New `SourceEvaluation.svelte` below the answer: a per-source row — a colour-coded
coverage chip (contested=warn, corroborated=ok, single-source=neutral, none=muted "not assessed"), a
`superseded` badge, doc year, rerank score — and a footer "assessed as of `{graph_version}`" with a
**stale** warning. Renders nothing when `source_eval` is null (honest degrade). Wired into `Turn.svelte`.

**Tests (E2e).** The marker-attach path changed (`load_epistemics_index` → `load_source_evaluations`),
so the marker tests were rebuilt around a `_stub_source_eval` helper (ChunkEval fixtures). New D3
guards: `test_d3_strip_always_on_even_when_markers_disabled` (the boundary — evaluation attached while
markers gated off), coverage-derivation + "not assessed", freshness-stale. Turn-parity byte-identical
preserved (strip no-ops with no graph). Note: turn tests **without** `temp_db` must now stub the strip
reads (D3 reads always) — else they'd hit the real DB via `current_graph_version()`.

**Live $0 verify** (real API for init endpoints + a `window.fetch` `/api/chat` SSE mock — fake
sources, no paid turn). The strip rendered below the answer: `[1] contested·2019·0.91`, `[2]
corroborated·2023·0.88`, `[3] single-source·superseded·2011·0.85`, `[4] not assessed·0.70`, with a
**stale** badge + "assessed from an earlier graph (b59a4aa6)" footer. **0 console errors**; coverage
chips resolve to distinct tokens in **light + dark** (contested amber, corroborated green, superseded
red); **375px → 0 horizontal overflow**.

**Opens.** RG-019 (a `contested` denominator) still deferred — measurement-gated, and moot at 0
Node-B stance on this box (the strip is honest-uniform, not saturated). **E3 / D2** (the persisted
answer-influence toggle) is the next ADR-027 row. Node-B stance regen (to make the strip's assessment
non-trivial on real data) still needs the RTX box (KI-4).

---
## 2026-07-21 — E1.2: extract `_handle_rag` into named seams (pure refactor, no behavior change)

ROADMAP row E1.2 (the code-health half of E1, deferred from E1.1). The AI-turn generator
`chat_controller._handle_rag` had grown to **~278 lines / ~14 responsibilities**; the plan calls for
breaking it up *before* E2/E3 wire the always-on epistemics strip into it. **Pure refactor — byte-identical
behavior.** **Staged, not committed** (cpc §13). Full suite **1179 passed / 1 skipped** (+3 unit tests);
ruff · `ruff format` · `mypy --strict src` · bandit · docs_check 0/0 · integrity_check 0/0.

**Approach.** `_handle_rag` is a *generator* (yields `Step`/`Token`/`Result`), so the yield-bearing
flow stays in the orchestrator; only the **non-yielding computation** is extracted. Three seams, each
a verbatim lift:
- `_resolve_turn_knobs(overrides) -> _TurnKnobs` — the ADR-010 effective-knob resolution (top_k /
  synthesis_mode / multi_query / markers_enabled / reviewer_evidence_chars + `overrides_note`). Preserves
  the subtlety that retrieval passes the **raw** `overrides.use_multi_query`, while the *effective*
  `multi_query` is for the provenance note only.
- `_capture_provenance_and_review(_ProvenanceInputs) -> _ProvenanceOutcome` — the 88-line
  provenance-record + confidence-signals + conditional LLM-reviewer + card-format block, try/except-bounded
  exactly as before (a failure still collapses to a "Provenance capture failed" card + empty `record_id`).
  Inputs bundled in a frozen `_ProvenanceInputs` so the seam is single-argument; the `overrides_note`/
  `scope_note` suffix stays in the caller (it owns the turn knobs).
- `_build_claims_block(record_id, full_answer, retrieved_chunks)` — the Chunk-2a segment→persist→render
  block; the caller keeps its `if record_id is not None` guard.

**Result.** `_handle_rag` **278 → 198 lines**; the three heaviest concerns are now named, testable seams
(and E2's source-evaluation strip has clean spots to slot into — `_build_source_views` + the TurnResult
assembly). Stopped here deliberately: extracting the yield-bearing export/TurnResult tail would need a
~20-field parameter bundle that hurts readability more than it helps.

**Verification.** The safety net is the existing `test_turn_parity` (byte-identical when markers absent)
+ the full `test_chat_controller` suite — all green, unchanged. Added 3 focused unit tests pinning
`_resolve_turn_knobs` (defaults / all-None-overrides ≡ defaults / applies + notes the diff). No KI, no
live probe (no data path touched). **Opens:** E2 (ADR-027 D3 always-on source strip) — now honest to
build on (E1.1) *and* has clean seams to wire into (E1.2).

---
## 2026-07-21 — E1.1: marker-join trustworthiness — KI-8 re-projection (correctness core)

Spec: `docs/specs/feature-e1-marker-join.md` (ROADMAP row E1). The honesty prerequisite for ADR-027's
always-on source-evaluation strip (E2): the 7d marker chip — the join E2 renders — silently
under-reported by **~40%** in the default parent-child retrieval mode. **Correctness core only**; the
`_handle_rag` extraction (E1.2) is a separate refactor, deferred. **Staged, not committed** (cpc §13).
Full suite **1176 passed / 1 skipped**; ruff · `ruff format` · `mypy --strict src` · bandit ·
docs_check 0/0 · integrity_check 0/0.

**The defect (KI-8).** `chunk_epistemics` was keyed on the **baseline** segmentation
(`{doc}:{chunk_index}`). In default PC mode retrieval returns **parents** (`parent_index`, never
`chunk_index`), so `_chunk_key` returned `None` and `_attach_markers` fell back to
`markers_for_parent` — a strict **substring-containment** test. A 1000-char baseline chunk cannot be a
substring of a parent it only partially overlaps (200-char overlap), so a marked chunk straddling a
parent boundary was contained in *neither* parent and its markers vanished (review WE-7 — systematic
false *negatives*, not fail-safe over-attribution).

**The fix (KI-8 option 2 — re-projection).** Project the node weights **directly onto the PC parent
segmentation** with the same structural attribution the baseline projection uses, keyed
`{doc}:p{parent_index}` (the ADR-4 composite `concept_skeleton.load_presence_inputs` already builds).
The PC join is now a **direct key lookup** — the coarse containment path is retired.

**What.**
- **E1.1a — schema.** Additive nullable `chunk_key` VARCHAR on `chunk_epistemics` (+ `_ADDITIVE_COLUMNS`,
  indexed): the authoritative, segmentation-agnostic join key. The regenerable table fills it on the
  next `compute_epistemics --apply`; `load_epistemics_index` falls back to `{doc}:{chunk_index}` when
  it is NULL, so a migrated-but-not-recomputed DB still joins flat rows (parent rows arrive on
  recompute) — a clean transition, no hard backfill.
- **E1.1b — projection.** `ChunkEpistemics.chunk_key` is now a stored field (was a derived property);
  `project_chunk`/`project_chunk_weights` carry it. `load_doc_chunks` yields baseline keys; new
  `load_pc_parent_chunks` yields `{doc}:p{parent_index}` (mirrors `load_presence_inputs`).
  `build_epistemics` projects `load_doc_chunks() + load_pc_parent_chunks()` — both segmentations, one
  attribution rule. Retired `markers_for_parent` / `load_marked_chunks` / `MarkedChunk` /
  `_load_baseline_texts` (no remaining consumer).
- **E1.1c — controller.** `_chunk_key` returns `{doc}:p{parent_index}` for a PC parent; `_attach_markers`
  joins **both** modes on `sv.chunk_key` against a single index (loaded once), dropping the containment
  branch and the now-unused `scored` arg. The blanket `except` gains a **WARNING log**
  (`attach_markers_failed`) — advisory markers still never break a turn, but under an always-on strip a
  silent failure is a silently-lying UI, so it must be observable.

**Guard tests (each fails against pre-fix code).** `test_reprojects_onto_pc_parents_keyed_by_parent_index`
(a parent's `{doc}:p{idx}` key carries the marker — fails today: nothing projected onto parents);
`test_chunk_key_parent_child_chunk_uses_parent_key` (inverts the old `…_is_none`);
`test_markers_pc_join_via_chunk_key` (direct join, no containment); `test_marker_load_failure_…_but_warns`
(the WARNING fires — asserted via a fake logger, not `capture_logs`/`caplog`, which both hinge on the
global structlog→stdlib bridge being already configured and so flake across the suite);
`test_project_chunk_carries_pc_parent_key`. Updated `test_compute_epistemics` (stub `load_pc_parent_chunks`,
4-tuple chunks, + the re-projection case), `test_turn_parity`, `test_epistemics` (retired containment
tests + new signatures), and the E0.4 empty-input test (stub the new loader).

**Live $0 probe** (isolated copies of the real 76-doc DB + skeleton; real Chroma read only; originals
byte-unchanged). This box has no Node-B stance, so one contested stance was injected to make a marker
exist, then the **real** re-projection ran: 11961 baseline + 5617 PC parents loaded → the marker index
now carries **196 PC-parent keys (was 0)**; a retrieved parent's `_chunk_key` resolves directly against
it; and **28 of 196 marked parents (14% on this single-concept sample) would have been left unmarked by
the retired containment** — the systematic false-negative direction KI-8 describes, on real data.

**Opens.** E1.2 (`_handle_rag` extraction, ~287 lines) before E2/E3 wire into it. Marker *quality*
(RG-019 `contested` denominator; Node-B stance regen on the RTX box) is unchanged — E1 fixes the
*join*, not the *data*. The old PR-M1 ADR-1 (containment) and the `feature-7d` "deferred live surfacing"
note are now superseded by the direct-key join.

---
## 2026-07-21 — E0 correctness batch: five P0 fixes before the always-on epistemics surfaces

Spec: `docs/specs/feature-e0-correctness-batch.md` (from `docs/PLAN_2026-07-21_exploration-epistemics.md`
§E0 + the C4 review's P0 list). ADR-027 D3 makes the epistemics **assessment** always-on — *an
always-on strip must not show false data* — so this closes the "a rebuild/curation silently destroys
curated state" class (three of these are the same shape as KI-25) plus the boot + zero-doc footguns,
before E1/E2 wire the surfaces. Backend-only, deterministic, **zero LLM / zero eval ceremony** (no
locked setting touched). Every item ships a **guard test that fails against the pre-fix code**; the
full suite is green (**1178 passed / 1 skipped**, +14), `ruff`/`ruff format`/`mypy --strict src`/
`bandit`/docs_check/integrity_check all clean. **Staged, not committed** (cpc §13). Build order was
E0.4 (safety net) → E0.1 → E0.2+E0.3 → E0.5a → E0.5b.

**E0.4 — zero-doc honesty, pinned by a test (WE-1/WE-9/GP-7).**
- *What.* `wiki.load_doc_graph` catches `OperationalError` → `([], [])` + a hint; the `epistemics`
  build guards its sidecar write (a never-migrated DB has no `chunk_epistemics` table, so the
  delete-all in `_write_rows` tripped `OperationalError`) → honest empty result + hint, `applied`
  reflects it. Missing-skeleton still raises `FileNotFoundError` (kept — the CLI + existing test rely
  on it). New parametrized `tests/integration/test_empty_input_honesty.py` over all four build paths
  (wiki/epistemics/skeleton/gaps).
- *Why.* The `.claude/CONTEXT.md` "degrade honestly at 0 documents" contract survived by habit; two
  build paths crashed and nothing gated it. Non-vacuous: wiki/epistemics raise on a never-migrated DB
  today (proven by a raw `OperationalError` repro).
- *Rejected.* Making missing-skeleton return empty too (breaks the CLI contract + `test_missing_
  skeleton_raises`); skip-on-empty for the epistemics write (would stop clearing stale rows on a real
  corpus) — instead the `OperationalError` catch degrades only the genuinely-unmigrated case.

**E0.1 — curation demotes, never deletes (KI-20 / CS-5).**
- *What.* New `concept_curation.demote_concepts(ids)` (`graph_include=False`, keeps row + aliases +
  ADR-015 keyword family) + `apply_plan(plan)` seam the runner now drives. Artifact + `classify_noise`
  verdicts route through demote; `remove_concepts` is kept as the **reserved** explicit-deletion
  primitive, no longer wired to the noise classifier. `CurationPlan.remove_ids` → `demote_ids`.
- *Why.* `classify_noise` is exactly the stage that mislabels specialist vocab (`cre`/`dbs`/`ntsr1`/
  `pddl`), and a delete cascades the keyword family + presence/edges/gaps — irrecoverable. ADR-018's
  demote verb, applied. Guard `test_noise_verdict_demotes_and_keeps_the_family`; `remove_concepts`'
  delete is covered too so the distinction is tested, not commented.
- *Rejected.* Looping `set_graph_include` per id (N sessions) — a single bulk update instead;
  changing the near-dup *merge* to demote (a merge folds aliases into the survivor first, no
  vocabulary lost — left as-is, out of the DoD).

**E0.2 — rebuild reconciles orphaned stochastic gaps (KI-17 / GP).**
- *What.* `gaps._reconcile_stochastic_gaps(live_ids)` deletes stochastic `GapRow`s whose anchor
  `concept_id` left the `graph_include`-filtered vocabulary, **hoisted to run unconditionally on
  every `build_gaps --apply`** (the review's placement correction — inside the suggest branch it never
  reached a deterministic-only apply). `GapsResult.n_reconciled` + CLI report line for transparency.
- *Why.* Stochastic rows were status-preserving upserts with no delete pass → immortal orphans (the
  live 27-gaps-over-13-nodes symptom). A reconcile, not a blanket delete: a promotion on a *live*
  concept survives. `suggest_for_thin` always anchors on an existing concept (target lives in
  `evidence`), so a live suggestion is never a false orphan. Guard `test_orphaned_stochastic_gap_is_
  reconciled_away`; **updated** `test_stochastic_rows_survive_a_deterministic_rebuild` to anchor on a
  live concept (its old synthetic non-vocab anchor is precisely the orphan the reconcile now reaps).
- *Rejected.* `notin_([])` bulk SQL delete (empty-IN edge cases) — load + filter in Python (gaps are
  tens of rows).

**E0.3 — in-app rebuild refreshes gaps, not just the skeleton (KI-21 / GP-4).**
- *What.* `_default_rebuild_graph` chains `build_gaps(apply=True, min_degree=derive_min_degree(skeleton))`
  after the skeleton build. `gaps.derive_min_degree` = runtime **Q1 of the rebuilt skeleton's
  connected-node degrees** (no literal; fails safe to 1 on a tiny graph). Guard `test_rebuild_
  refreshes_gaps_and_drops_stale_ones` (route composition: served gaps == fresh recompute, stale gap
  gone).
- *Why.* The acquire loop the button exists to close (gap → ingest → rebuild → gap closes) never
  closed in-app — the view served gaps from the previous skeleton, incl. the one just closed.
  Derived `min_degree` measured **3** on the real 26-node graph, matching the CLI's validated Q1
  baseline — the "no corpus-tuned literal" contract, satisfied by derivation.
- *Rejected.* A hardcoded default (a corpus-tuned magic number, `.claude/CONTEXT.md`); stamping
  `graph_version` on gap rows + filtering in the view (heavier; the refresh already makes the served
  set correct).

**E0.5a — a failed startup migration fails the boot.** The lifespan (`apps/api/main.py`) re-raises on
`init_db()` failure with a clear message instead of swallowing and serving a half-migrated schema —
KI-23 moved `init_db` here precisely because a stale **answer-path** column 500s every turn, a worse
and later failure than refusing to start. Deliberately reverses the old "never let a migration
problem stop the app" comment (documented in-code). Fixed the stale `apps/api/CLAUDE.md` line ("the
API does not `init_db()` on startup"). Guards in `test_api_startup_migration.py` (boot fails on a
broken migration; boots clean on a good one).

**E0.5b — a plain rebuild preserves Node-B stance.** `build_concept_skeleton(apply=True)` without
`--enrich` recomputes structure only, so it now re-attaches existing Node-B `stance_by_doc`/`relation`
(+ the `llm_relation` provenance token and the weight that follows) to edges whose concept-pair still
exists (`_reattach_stance` / `_load_existing_stance`, `stance_loader` DI seam) instead of wiping it —
the G6-run footgun that darkens corpus-wide epistemics on every in-app rebuild. Transparent to
`--enrich` (which re-derives every edge's stance, setting `()` on the ones it skips), so it only
protects the plain path. Updated the `scripts/CLAUDE.md` footgun note. Guard `test_plain_rebuild_
preserves_node_b_stance`. *Known bound (documented):* a stance entry for a since-removed document
lingers until a real `--enrich`; preserving stale-but-mostly-right stance beats wiping all of it.

**Live $0 probe (isolated copies of the real 76-doc library + 26-node skeleton; real Chroma read
only; originals verified byte-unchanged).** This box's graph is currently clean (0 stochastic gaps,
0 stance), so the probe injected the exact condition each fix protects, then exercised the real data
paths: **E0.5b** — injected stance survived a plain rebuild (26 nodes/70 edges) in both `skeleton.json`
and `concept_edges`, while a `stance_loader`-empty run (pre-fix simulation) wiped it; **E0.2** —
derived `min_degree=3`, 1 orphan reaped, 1 live-anchored promotion kept; **E0.3** — served
deterministic gaps (11) == fresh recompute (11), an injected stale gap dropped.

**Deferred / opens.** E1 (KI-8 marker re-projection + `_handle_rag` extraction) is the next sprint;
KI-18 scale hot-paths + KI-19 tuned constants stay measurement-gated (RG-016..019) — not touched here.
The near-dup merge still hard-deletes the folded row (correct: aliases move first). ADR-017 C1's
gap-triage override sidecar can inherit the E0.2 reconcile seam when PR-G2b lands.

---
## 2026-07-21 — App-shell polish: global search overlay + collapsible sidebar (chat-first shell, a+b)

The two **shovel-ready** sub-items of the "App shell → chat-first layout" backlog row
(`docs/ui-checklist.md`, 2026-07-20). Frontend-only, no backend, no wire-type change. The row's
third clause — *demote Graph out of the top-level nav* — is **deliberately not built**: where Graph
goes is an open design fork (empty-Graph → per-folder-concepts, ADR-025 fork 5) the baton says to
`grill-me` first. Spec: `docs/specs/feature-app-shell-search-collapse.md`.

**What.** (a) A **global-search overlay** — scrim + centred dialog (reuses the LibraryKeywordFilter
modal shell), opened from a new header button **and Cmd/Ctrl-K**. It searches conversation titles +
document title/filename/authors/keywords, groups the hits (Chats, then Documents, each capped at 8
with an honest "+N more"), and jumps to the chosen one. Empty query shows up to 6 recent chats.
Keyboard-first: autofocus, ↑/↓/Enter/Esc, mouse hover shares the same highlight. (b) A **collapse
toggle** in the header that hides the rail on desktop and brings it back at its persisted width.

**Why (the two decisions worth naming).**
1. **It is a *navigation* search, not a corpus search** (spec A1). The composer *is* the corpus
   surface; a second box that looked like retrieval but wasn't would be exactly the integrity lie the
   product avoids — and message bodies / chunk text aren't client-side anyway, so searching them
   needs a backend this row excludes. Placeholder + empty state say "chats and documents" so the
   scope reads honestly.
2. **The trigger lives in the header, not the sidebar** (A2). The common sidebar-search convention
   would put it in the rail — but the rail can now be collapsed (b), which would hide the search
   entry point exactly when you collapse. Header + Cmd/Ctrl-K is always reachable.

**Design notes.** Match logic is a pure, tested module `lib/search.ts` (`searchEverything`) —
`npm test` gate, house pattern since PR-2.5; the overlay is a dumb renderer, App owns the data +
navigation (reuses the existing `openConversation`/`openDocument` entry points, so no new nav logic).
`search.ts` is deliberately **self-contained** (no runtime import of `library.docLabel`): node's
test runner strips the type-only `./types` import but can't resolve an extensionless *value* import,
which is the same constraint that kept `library.ts` value-import-free — a real gotcha for the next
tested module. Collapse is a client-only pref in `localStorage` (theme/width precedent), desktop-only
under a `min-width:721px` guard so the mobile off-canvas drawer is untouched; `.app.collapsed
:global(.sidebar)` reaches the child component's root.

**Rejected.** Searching chat/chunk content (needs a backend — out of scope + blurs into the
composer). A sidebar-hosted trigger (collapsing hides it). Driving collapse via `--sidebar-width: 0`
(leaves the border + a live resizer; `display:none` is cleaner).

**Verified.** `svelte-check` 0/0; `npm test` **34/34** (23 existing + **11 new** in `search.test.ts`).
**Live on the real 76-doc corpus ($0, no LLM turn fired):** header button + Ctrl-K both open the
overlay (Ctrl-K toggles closed, autofocuses); empty → 6 recent chats; "retrieval" → 2 chats + 4 docs
(matched on title/keywords, author·year bylines); Enter opened a chat read-only (Chat mode), a doc
result opened the DPR paper (Library mode) — documents lazy-loaded by the overlay from a cold Chat
session; no-match → honest-empty line; collapse hid the rail + resizer, persisted across a reload,
expanded back to 260px; dark tokens resolve (surface/border/scrim-via-color-mix/fg-2); 375px → collapse
toggle hidden + hamburger shown + 0 px horizontal overflow; **0 console errors** throughout.

**What it opens.** The row's clause (c) — Graph nav placement + the empty-Graph → per-folder-concepts
fork — is still open and wants `grill-me` before any code. `apps/desktop/CLAUDE.md`'s "Tests: none"
line is stale (the `npm test` runner has existed since PR-2.5) — corrected in this change.

---
## 2026-07-20 — PR-2.7: the Manage view at scale (F1–F4) + KI-25, the graph emptied by KI-23's fix

Two things, logged together because the second was found while verifying the first in the running app.

### PR-2.7 — F1–F4

**What.** Four presentation fixes over the keyword overlay and the Manage view, all frontend-only,
all resting on new **pure** helpers in `lib/library.ts` (`unitDocCounts`, `splitRareFacets`,
`splitInheritedFamilies`, `filterByQuery`) so the rules are unit-tested rather than eyeballed.

**F4 is the substantive one.** A facet exists to *partition* a set; a keyword on one document
partitions nothing — selecting it yields that document, which search already does better. So the
1-doc tail is collapsed behind "Show rare (N)". That single principled threshold sweeps up the ugly
strings (`mathrm`, `102ff`, `fne-tune`) **and** the real specialist vocabulary (`va1v`, `avpv`)
without having to classify them — which is the point, because they are not distinguishable by
inspection. Nothing is destroyed: search bypasses the split entirely, and a **selected** facet is
never demoted.

**Two guards worth naming.** *Honest-empty*: when every facet is rare (a small collection), nothing
is demoted — collapsing the whole list would look broken rather than informative. *Stable rarity*:
the counts come from a `unitDocCounts` map over the pre-facet pool, not from `KeywordFacet.count`
(which is relative to the *faceted* pool and would make the rare set shift under the user as they
filter).

**F1 was already satisfied.** `.kwlistfoot` is a flex sibling of the scrolling `.kwlist`, so
"Manage keywords…" is already a pinned footer. Verified live and recorded rather than "fixed" —
PR-1 landed the same day the feedback was taken.

**The spec's F3 grounding was wrong; the live data corrected it.** It expected "only ~6 are real
families; the rest are 0-member concepts". Measured: **12** have ≥1 alias (real collapses), **10**
have 0 aliases but **>0 docs** (`ImageNet` 10, `Tractography` 10 — not collapses, but they *do*
partition the grid), **4** are inert (0 aliases, 0 docs). Only the 4 are hidden; the heading now
carries the split. Hiding all 22 would have removed working facets to satisfy a mis-estimate.

**A trap found in this PR's own rule.** A family created with no members starts at 0 aliases /
0 docs — exactly the shape the glossary-only group hides — so creating one would look like it
silently failed. `submitCreate` now reveals that group when, and only when, the new family has no
members.

**Rejected:** *deleting the rare tail* — mostly real vocabulary, and delete is not
reversible-by-search; *hiding every 0-alias concept* — 10 of them filter real documents; *a
corpus-tuned "only demote if the list is long" rule* — the 1-doc principle is scale-free and the
project forbids corpus-tuned constants.

**Verified:** 8 new frontend tests (23 total); suite **1164 passed / 1 skipped** · ruff ·
`mypy --strict src` · `svelte-check` 0/0 · docs+integrity 0/0. **Live, $0:** overlay 55 facets →
**25 shown / 30 demoted** (the spec predicted 30, exactly); toggle round-trips 25 ↔ 55; searching
`mathrm` finds a demoted keyword; Manage pool 38 → 12; families 22 ↔ 26; F2 shows "Go to family" +
a warning on an exact match and suggests `Brain connectivity`/`Connectome` on `conn`. Dark at
375 px: 0 px overflow, 0 console errors.

### KI-25 — the graph emptied itself when KI-23 was fixed

**Symptom (user-reported):** the Graph view showed nothing; `/api/concepts/graph` returned **0
nodes** while `concepts` held **26** rows.

**Cause.** ADR-018 made the graph vocabulary opt-in via `concepts.graph_include`, and
`load_concepts()` documents that NULL "reads as excluded". That column had never reached this box
(**KI-23**); running the migration by hand on 2026-07-20 — *while diagnosing KI-23* — finally added
it, **NULL on all 26 rows**, excluding every concept at once. The migration was correct. What was
missing is that **an additive column whose NULL default changes behaviour is not a safe additive
migration** — it needs its backfill in the same breath. `_ADDITIVE_COLUMNS` even carries that note
for this exact column; nobody was in a position to act on it, because the column had never landed.

**Fix.** `backfill_graph_include --apply` (ADR-018's rule retroactively: `source == "manual"` opts
in — all 26 qualify) then a skeleton rebuild, **Node A only, deterministic, $0**. Result: **26
nodes / 70 edges / 3 communities / 14 gaps**, `stale: false`; the concept index and the ego view
both render again (9 circles + 11 edges for `Connectome`).

**Not restored, deliberately:** `concept_edges` was already empty, so nothing was lost *by this* —
but Node-B stance annotations do not exist now either, and regenerating them is an **LLM pass**
(`--apply --enrich`, KI-4: force `--provider ollama`, which lives on the other box). Not run; the
user decides.

**Why nothing caught it:** the graph route degrades honestly to an empty graph (the documented
"empty vocabulary → empty graph" path), so the suite stayed green and no gate compares "concepts in
the DB" to "concepts the graph can see".

---
## 2026-07-20 — PR-2.6: family-aware grid tiles (D6 — a family selection highlighted nothing)

**What.** `LibraryGrid` learns about tag families through one optional prop, `keywordsOf`,
defaulting to `(d) => d.keywords`. `App` passes the accessor it already derives for the facet
overlay, so tiles and facets finally agree on what a *unit* is. Ordering moved into a pure
`orderedUnits(units, active)` in `lib/library.ts`; the `+N` overflow count now counts units.

**Why it was broken.** `LibraryGrid` matched `activeKeywords.includes(rawKeyword)`. With a family
selected, `activeKeywords` holds the **canonical** (`Pretrained model`) while tiles held the raw
member forms (`pretrained`/`huggingface`) — the match could never fire. `orderedKeywords` broke
identically, so the matching form was not floated to the front and could hide behind `+N`. One
root cause, one file: the grid never learned about families, which is why D6 was carved **with**
PR-2.6 rather than into PR-2.5 (splitting would have touched the same file twice).

**Why the default is the raw list.** It is the whole no-families guarantee: with `canonicalOf`
empty, `familyUnitsOf` already returns the raw keywords, and the default prop means a caller that
knows nothing about families renders exactly what it rendered before. Verified live with a plain
keyword control.

**Why the `+N` count had to change too.** It read `d.keywords.length`, so a tile holding `llm`
and `llms` would claim one more chip than it renders once the family collapses them — and the
collapse would not actually free the tile's scarce chip budget, which is half the point of showing
a family atomically.

**Rejected:** **passing the family list into the grid** and mapping there — that would put the
grouping rule in two places (the overlay already owns it) and give the component a data dependency
it does not need; **keeping ordering inside the component** — it is the half of D6 easiest to get
wrong, so it belongs where it can be tested; **rendering both the canonical and its members** —
contradicts the overlay's atomic-entry rule and spends the chip budget on duplicate forms of one
concept.

**Verified:** 5 new frontend tests (15 total, `npm test`); full suite **1164 passed / 1 skipped** ·
ruff · `mypy --strict src` · `svelte-check` 0/0 · docs+integrity 0/0. **Live on the real 76-doc
corpus ($0, no LLM), driven in the running app:** with the probe family selected, **22 of 22**
tiles highlighted and in **all 22** the active chip was floated first — the spec measured **0 of
25** before. Control (plain keyword `cajal`): 9 of 9 highlighted and floated, so the default path
is unchanged. Dark theme at 375 px: 0 px horizontal overflow, active chip visually distinct, 0
console errors. The probe family was built from two previously un-familied keywords
(`pretrained` + `huggingface`) precisely so deleting it restored the vocabulary exactly — verified
at 26 concepts / 17 aliases before and after, both keywords unclaimed again.

**Opens:** PR-2.7 (Manage view at scale) is next in the carve. Svelte-5 gotcha worth knowing:
`{@const}` must be the immediate child of a block (`{#each}`), not of the element where it reads
most naturally.

---
## 2026-07-20 — PR-2.5: hardening the tag-family write paths (D1–D5, all five defect-driven)

**What.** The five defects the post-commit review of `0c3b0d4`+`0af43db` found in the tag-family
**write** paths — none of which the 977 tests passing at the time caught. Each repro is now a
regression test written **first**, so it fails against the shipped code. Read path untouched.

| # | Defect | Fix |
|---|--------|-----|
| D1 | Rename onto an existing canonical created duplicate `Concept.label` rows → `add_concept`'s get-or-create then raised `MultipleResultsFound` for that label **forever**, 500ing the create route and breaking `promote_keyword` repo-wide | `rename_keyword_family` raises `KeywordFamilyExists`; the API shell maps it to **409** |
| D2 | Rename silently dropped the family's own canonical keyword — it is only an *implicit* member, so re-pointing the label let the original keyword fall out and reappear as the standalone chip the feature exists to remove; `doc_count` fell with it | Rename carries the old label into the alias set first |
| D3 | "New family" took its canonical as unchecked free text, so a keyword already claimed elsewhere ended up in **two** families and `familyCanonicalMap` resolved it order-dependently — three different numbers for one keyword | `create_keyword_family` routes the canonical through `add_family_member`, reusing the move-on-reassign guard |
| D4 | The sibilant `-es` rule always stripped two characters, so every plural whose singular ends in `e` (`database`, `size`, `cache`, `response`) **never matched** — a silent false negative that degraded a `confidence=1.0` structural pair to a threshold-dependent fuzzy one, or to nothing | `_stem` → `_stem_candidates`; Tier 1 groups on a non-empty intersection, union-find because a name can now bridge buckets |
| D5 | A live keyword selection was not re-pointed when families changed → the grid emptied behind a chip that still looked selectable. The Manage view is opened *from* the overlay, i.e. exactly where a selection is live | New pure `remapSelection` in `lib/library.ts`, called after every family write |

**Why D2 was fixed by rename rather than by create.** The spec offered both. Seeding the canonical
as a real alias on create would have needed a **migration for the 26 pre-existing concepts** on this
box; carrying the old label at rename time changes nothing about existing rows, because the label
stays the implicit member `_build_family` already treats it as. Smaller blast radius for the same
invariant.

**Why D4 emits two candidate stems instead of a better single rule.** There isn't one. `boxes`→`box`
and `databases`→`database` are structurally identical — both stems end in a sibilant — so no
lexicon-free rule can separate them. Emitting both trades an implausible false **positive** (needs a
real keyword equal to an over-stripped stem: `cas` beside `cases`) for a silent false **negative**,
which is the worse of the two because a proposal is reviewed before it is applied and a
non-proposal is not reviewable at all.

**Why the frontend got a test runner and no dependency.** The spec's DoD asks for the first tests of
`familyCanonicalMap`/`familyUnitsOf`/`facetFilter` — but the frontend had **no test runner at all**,
a prerequisite the spec never states. Node 24's built-in `node:test` runs the real `.ts` module with
native type stripping, so `npm test` works with **zero new dependencies**. Test files are excluded
from `tsconfig.json` so the app config doesn't have to carry `@types/node` +
`allowImportingTsExtensions` for test-only imports; they are run, not type-checked.

**Rejected:** **adding vitest** (or `@types/node`) — a dependency and a new gate to maintain, for
something the runtime already does; **enforcing uniqueness in the Manage view** (D1) — the invariant
belongs at the library boundary, the view is one of several callers; **exact-case collision
matching** — the client lowercases its canonical map, so two families differing only by case would
collide there anyway; **dropping `_stem` while keeping a "primary" stem for readability** — dead code.

**Verified:** 5 new integration tests + 6 new/updated unit tests + **10 new frontend tests**; full
suite **1164 passed / 1 skipped**; ruff · `ruff format` · `mypy --strict src` · bandit ·
`svelte-check` 0/0 · `npm test` 10/10 · docs+integrity 0/0. **Live, $0, on a copy of the real
76-doc library** (the original verified untouched at 26 concepts / 17 aliases before and after):
Detect reproduced `pvpo`≈`avpv pvpo` @ 0.77 with the real bge embedder → Accept → **Rename**, which
kept `pvpo` as a member and held `doc_count` at 1 (D2), a colliding rename returned **HTTP 409**
(D1), and re-creating that same label still returned **200** — the repo-wide poisoning is closed.
Measured honestly: on this corpus the fixed stemmer finds **exactly the same three pairs** as the
old one (`llm`/`llms`, `connectome`/`connectomes`, `keypoint`/`keypoints`) — no regression, and no
new find either, because these 60 keywords contain no e-final plural. D4 is proven by unit test, not
by this corpus.

**Not done, deliberately:** D5 was not driven in the browser. Exercising it live would write to the
user's curated vocabulary, and move-on-reassign is **not undoable** — detaching a keyword deletes
that `ConceptAlias` row, and deleting the new family does not restore it. It is covered by 4 unit
tests on the extracted pure function plus `svelte-check`.

**Opens:** that non-undoable move is ADR-015's stated semantics, not a defect — but D3 now extends it
to the *canonical*, so naming a new family after a keyword claimed elsewhere silently strips it from
the other family. Worth knowing before bulk curation; recorded in the spec. **PR-2.6** (family-aware
grid tiles, carrying D6) is next in the carve.

---
## 2026-07-20 — KNOWN_ISSUES split: open issues in the working file, closed ones archived verbatim

**What.** `.claude/KNOWN_ISSUES.md` went from **738 lines to 237**. The 14 resolved entries moved
**verbatim** to `docs/archive/KNOWN_ISSUES-resolved-001.md` (544 lines); the working file keeps the
10 open issues in full plus a new **Resolved — index** table. `AGENTS.md`'s coordination-file list
points at both.

**Why.** 526 of the 738 lines — 71% — were closed issues. The file is read at session start to find
what might bite *today*, and four fifths of it was history. Same shape ADR-022 already applied to
the decisions monolith: living index in the working file, canonical detail frozen in `docs/archive/`.

**What each retained row keeps, and why those two things.** `| KI | What it was | What keeps it
fixed — do not undo | Resolved |`. A closed issue still carries exactly two live risks: **the trap**
(so it isn't re-diagnosed from scratch — e.g. "never `cu130` on a GPU-less box") and **the load-bearing
fix** (so nobody deletes it not knowing what it holds up — e.g. the API lifespan's `init_db()` call
is the *only* migration trigger the app has). Everything else — reproduction steps, the diagnosis
narrative, rejected alternatives, verification detail — is history, and history belongs in the
archive.

**Rejected:** **summarising on the way into the archive** — the archive is the canonical account,
and a summary of a summary is how detail quietly dies; **deleting resolved entries outright** — the
KI-15 and KI-22 write-ups are the record of *how a class of bug was caught*, which is worth more
than the bug; **splitting by date rather than by state** — "resolved" is the property that makes an
entry stop being operational, a date boundary is arbitrary; **per-heading anchor links** into the
archive — they rot on the first heading edit, and the KI number is trivially findable.

**Verified:** a script asserts every resolved body appears **byte-identical** in the archive, every
open body byte-identical in the working file, every resolved KI has an index row, and the header is
preserved — all four clean. `docs_check --strict` 0/0.

**Opens:** numbering stays global and never reused (the KI-23-was-KI-20 note travelled with its
entry into the archive). Next rotation is `KNOWN_ISSUES-resolved-002.md`; no cap is enforced by a
gate — `session_max_entries` covers the baton only.

---
## 2026-07-20 — `document_meta` gets its missing foreign key; rebuild migrations exist now (ADR-026)

**What.** `document_meta.document_id` is now a real FK to `documents.id` with `ON DELETE CASCADE`.
Getting there needed a new migration mechanism: SQLite cannot `ALTER TABLE … ADD CONSTRAINT`, and
`db/migrations.py` was additive-only by design. `_rebuild_table` implements SQLite's documented
rebuild dance — FKs off, one transaction, create the model's shape under a temp name, copy the rows
worth keeping, drop, rename, `foreign_key_check`, commit — and `_rebuild_document_meta_fk` is the
first (and so far only) caller. **ADR-026** records both the fix and the policy around the
mechanism.

**Why it mattered more than "a missing constraint".** Correctness was being held up by convention:
`delete_document` deletes the override by hand and its docstring says "no FK — explicit". Every
*bulk* path forgot. The pre-KI-24 `--rebuild` forgot, and — the part KI-24 left open —
`cleanup_orphans_sqlite` **still** forgot, on every incremental ingest that finds a gone or
content-changed source. So orphaned overrides were still being produced today: unreadable (every
read path resolves through a live document), never cleaned, accumulating.

**Why the new shape is rendered from the model, not hand-written DDL.** A hand-written
`CREATE TABLE` in a migration is a second definition of the table that silently drifts from
`create_all`. `_rebuild_table` compiles the SQLAlchemy model's own table under a temp name, and
copies only columns present in *both* the live table and the model — so a rebuild is safe against a
schema that predates an additive column.

**Why rebuilds are named functions, not a `_TABLE_REBUILDS` list.** A data-driven registry mirroring
`_ADDITIVE_COLUMNS` would be a framework for n=1 and would make rebuilding a table feel as routine
as adding a column. It is not: it rewrites the table. Named function, idempotent, ADR-justified.

**Orphans are dropped, and logged in full first.** They cannot be carried (they are what the
constraint forbids) and cannot be rescued (an override records no filename or hash, only a dead
id). `init_db` returns the orphan count with the change description, so the KI-23 startup log
states it. On the real corpus there were **zero**.

**Rejected:** **compensating in application code** — add the override delete to the orphan sweep
and `_sweep_rebuild_rows`; that is the bet that already failed twice, and a third caller would
forget too. **A periodic orphan-cleanup pass** — sweeping up after a defect instead of making it
impossible. **Alembic** — real tooling for one non-additive change on a single-file local SQLite
app is a dependency plus a workflow; the reopener is explicit (a second or third rebuild → adopt
it). **Leaving the orphans** after adding the FK — the constraint would be a claim the data does
not support. **Fixing `ConversationMeta` the same way** — it *cannot* have an FK: conversations are
derived by grouping `AnswerRecord` rows, there is no table to point at. Its bare `session_id` is
correct, not the same defect.

**Verified:** 8 new tests (`tests/integration/test_document_meta_fk_migration.py`) driving `init_db`
over a genuinely pre-migration table — FK added, live overrides kept, orphans dropped and counted,
idempotent, delete cascades, the orphan sweep no longer leaves a row behind, an unknown-document
override is rejected outright, and **a failed rebuild leaves the old table exactly as it was**
(no temp table, no data loss — the migration refuses rather than inventing a value). Full suite
**1143 passed / 1 skipped** · ruff · `ruff format` · `mypy --strict src` · bandit · docs+integrity
0/0. **Verified non-vacuous:** with the migration disabled, 7 of the 8 fail — the survivor is the
fixture's own assertion that the legacy table really has no FK. **Live, on a copy of the real
`data/library.db`** (the original deliberately untouched — the app will migrate it on next start):
FK added with CASCADE, the one real override row preserved with its `authors`/`year` values,
`PRAGMA foreign_key_check` clean, 76 documents intact, no leftover temp table, 0 orphans dropped.

**Opens:** the reopener above (a second rebuild → Alembic). `delete_document` keeps its now
redundant explicit override delete, retained so the ADR-014 path still reads as the complete story.

---
## 2026-07-20 — KI-24 fixed: `ingest --rebuild` rebuilds the index instead of resetting the library

**What.** The rebuild branch no longer runs `delete(DBDocument)`. It wipes both Chroma stores and
re-embeds, as its CLI help always claimed ("Wipe the vector store and re-embed everything"); the
rows it does **not** reproduce are swept afterwards by the new `ingest._sweep_rebuild_rows`,
classified gone/stale exactly the way `cleanup_orphans_sqlite` classifies them.

**Why the delete had to go, rather than be compensated for.** KI-24 proposed snapshotting
`document_folders` by `doc_hash` and restoring it after the loop. Auditing every FK to
`documents.id` before building that showed the blast radius was much wider than folders —
`document_tags`, `document_keywords`, `citations`, `doc_similarities`, `document_parts`,
`chunk_epistemics`, `concept_presence`, `ingestion_events` all cascaded; `is_archived` and `notes`
were reset by the re-insert; **`document_meta`** (the ADR-013 metadata overrides) has *no* FK, so
its rows were **orphaned** rather than deleted, silently inert against ids that no longer existed.
And because `figures` is keyed by document id, `figure_units()` found none mid-rebuild, so the
reindexed corpus carried **no figure chunks at all** until the paid VLM describe pass was re-run —
a silent retrieval-quality regression riding along with the data loss. Snapshotting all of that is
a re-keying layer; **keeping the rows** makes `_existing_document_id` resolve to the same id and
every association simply stays attached. One less mechanism, strictly more preserved.

**Why the sweep runs after the loop, not before.** `cleanup_orphans_sqlite` reads its candidate set
from the Chroma metadata — which this branch has just deleted — so it cannot run here. After the
loop the rebuild has already told us what every present source produces, so gone/stale falls out
with no re-hashing. The sweep keys on `indexed - indexed_before` ("what this run produced"), never
on "the store was empty": an rmtree that silently fails must not be able to delete a library.

**One deliberate behaviour change beyond restoring the invariant.** A document whose file is still
on disk but which produced nothing this run (extraction error, empty extract) is now **kept and
reported** (`rebuild_kept_unreproduced_rows`). The bulk delete removed it unconditionally, so a
transient extraction failure used to cost the user their folders and metadata for that document.

**Rejected:** the **snapshot-and-restore** KI-24 originally proposed — it preserves folders and
tags but cannot preserve figures (the rebuild reads them *during* the loop, before any restore
could run) and leaves the `document_meta` orphaning untouched; **a new `--reset-library` flag** to
keep the old nuke available — nothing asked for it, and a destructive escape hatch nobody
requested is how the original silent loss got shipped; **writing the snapshot to disk first** so a
crashed rebuild could recover it — that only exists as a problem if you snapshot at all.

**Retires ADR-025 F3 spec M3/M9.** M3 called a rebuild "the one honest exception" where a demo
removal is re-fought; M9 recorded the loss as warned-about-not-fixed. Both are amended in
`docs/specs/feature-corpus-folders-demo.md`: membership is preserved, so the demo hook sees no new
rows on a rebuild and **nothing is re-fought anywhere**. The `rebuild_clears_folder_membership`
warning F3 added is gone with the behaviour it described.

**Verified:** 6 new tests (`tests/integration/ingest/test_ingest_rebuild_preserves_library.py`);
full suite **1135 passed / 1 skipped** · ruff · `ruff format` · `mypy --strict src` · bandit ·
docs+integrity 0/0. **Non-vacuous:** restoring the bulk delete fails exactly the three
preservation tests and no others. **Live ($0, isolated `DOC_DATA_DIR`, real embedder + real
Chroma, rebuild run as its own process; the real 76-doc library verified untouched before and
after):** two documents re-embedded with **identical ids**, "My reading" (2) and "Demo corpus" (1)
intact, metadata override and notes preserved; then deleting a source and rebuilding logged
`rebuild_removing_rows gone=1` and dropped exactly that row.

**Opens:** the derived sidecars now survive a rebuild *because the ids are stable*, but they are
not re-derived by it — re-run their runners when the chunking changes. `document_meta` still has
no FK; nothing creates new orphans, but it is not referentially enforced. A quirk found on the
way, production-irrelevant but worth knowing: chromadb caches one system per persist path, so an
**in-process** rebuild reattaches to the store `rmtree` was meant to remove — `--rebuild` is a CLI
entrypoint in a fresh process and no API route exposes it, and the sweep is written not to care.

---
## 2026-07-20 — ADR-025 F3: demo corpus auto-assigns into a folder at ingest + a one-time backfill

Closes the ADR-025 carve (F1 folders → F2 retrieval scoping → **F3 demo auto-assign**). Contract
written first: `docs/specs/feature-corpus-folders-demo.md` (M1–M11).

**What.** New sidecar `src/doc_assistant/demo_corpus.py`: load the `collection: demo` pins from
`tests/eval/corpus_manifest.yaml`, decide whether a file *is* one of them by **bytes** (size
fast-path, then SHA-256 — so a renamed demo PDF still counts), resolve the demo folder, and assign.
`ingest.main()` gained a two-line seam — `get_document_row_hashes()` diffed around the processing
loop — and hands the newly-created rows to the hook. New runner
`scripts/backfill_demo_folder.py` (dry-run default, `--apply`, `--force`) covers documents that
were already in the library. `app_settings` gained `demo_folder_id` + `demo_backfill_done`.
`process_one_document` is **untouched**.

**Why the trigger is "the row is new", not `process_one_document`'s `"added"`.** `"added"` is also
returned for *re*-ingests — the inverse-orphan repair, a `--path` rerun — so keying on it would
re-add a document the user had removed from the folder by hand, every run. The ADR's own words are
"ingest of a **new** document"; the row-set difference is literally that, and it keeps the locked
ingest hot path free of a new parameter (M1/M2).

**Why the folder is resolved by a persisted id, not by name.** ADR-025 promises an ordinary,
renamable folder. A name-keyed lookup would silently create a *second* "Demo corpus" the first
time someone renamed theirs. The id lives in `settings.json` because it is a per-install
**pointer**, not document data — no schema change (M5).

**Why the backfill refuses to run twice.** A second pass re-adds exactly the papers the user
removed. `--force` exists and says so loudly (M8). A run that assigns nothing does **not** burn the
flag, so back-filling before ingesting doesn't lock out the real backfill.

**Rejected:** a **`folders.origin` additive column** to mark the demo folder — `settings.json`
already fits a per-install pointer, and one auto-managed folder doesn't earn schema surface
(reopener recorded). A **tombstone so a deleted demo folder never returns** — that would couple the
generic `delete_folder` to demo semantics; per-*document* removals are what ADR-013 protects, and
those are never re-fought (M6). **Bundling the manifest into the PyInstaller build** — the demo
corpus is a repo-clone flow end to end, so a packaged install has no demo files to assign; the
missing manifest is a quiet no-op by design (M10). A **demo badge / demo-specific UI** — ADR-025
fork 1 is one organizing concept, one write surface (M11).

**Found while specifying, logged not fixed — KI-24.** `ingest --rebuild` runs
`delete(DBDocument)`, and `document_folders` cascades on the document side: **every folder is
silently emptied** while still appearing in the rail. F3 adds a warning naming the count
(`rebuild_clears_folder_membership memberships=4` in the live probe) and logs the issue; the real
fix (snapshot membership by `doc_hash`, restore after) is its own change. The demo folder is the
one that self-heals, because a rebuild makes every document look newly ingested (M3/M9).

**Also corrected: a duplicate KI number.** The schema-migration issue filed yesterday as **KI-20**
collided with the existing KI-20 (concept curation hard-deletes vocabulary). Renumbered to
**KI-23** in the living `KNOWN_ISSUES.md` + code/spec/test references; the append-only DEVLOG and
baton entries above still read "KI-20" and were **not** rewritten — KI-23 carries a pointer note.

**Verified:** 27 new tests (13 unit + 14 integration); full suite **1129 passed / 1 skipped** ·
ruff · `ruff format` · `mypy --strict src` · bandit · docs_check 0/0 · integrity_check 0/0.
**Live, end-to-end, $0** (real ingest, real Chroma, local embedder, isolated `DOC_DATA_DIR` — the
real 76-doc library never touched, verified 76 docs / 0 folders / no settings file before and
after): a **renamed** demo PDF + a private PDF ingested → only the demo one joined "Demo corpus";
re-ingest after a manual removal put **nothing** back; the folder renamed to "Sutskever reading
list" kept receiving new demo papers with **no** second folder created; `--rebuild` logged the
membership warning, left a hand-made folder empty, and refilled the demo folder. The live backfill
dry run against the real corpus found 18 demo files on disk, none ingested — and **caught a
reporting bug** on the way: the summary counted never-ingested files as "already members". Fixed
and covered by a test.

**Opens:** KI-24 (the real rebuild fix, and `document_tags` has the identical exposure) · the M8
reopener (a per-document "was auto-assigned" marker would make backfill re-runs safe without a
flag) · an "Eval corpus" folder for the other collection is deliberately **not** built (not in
ADR-025) · `--remove-demo` still leaves the emptied folder behind, by choice.

---
## 2026-07-20 — KI-20 resolved (schema migrates on API start) + A/B compare honours the scope

Two decisions the user took after reviewing F2 (`0e45dd3`), built together because both are
about the same thing: a surface that quietly describes something other than what it did.

**What (1) — KI-20, RESOLVED.** `init_db()` was called from **exactly one place in the running
app**: `ingest/__init__.py:405`. So a user who pulled an update and only chatted never received
new additive columns, and the **packaged build never migrated at all**. Evidence it had already
bitten: this box was missing `concepts.graph_include` (added 2026-07-07) for ~2 weeks, silently.
Fix — the API lifespan now calls `init_db()`, and `init_db`/`_apply_additive_columns` **return the
columns they added** so the lifespan logs `schema_migrated_at_startup columns=[...]` at WARNING
(`schema_current` otherwise). A migration error is caught and logged, never a startup crash.

**What (2) — the A/B compare scopes both sides.** `compare_retrieval(..., scope_folder_id)`
threads the resolved scope into **both** arms; `CompareResult.scope_label` drives a card line
("Both sides searched X only"). Wire: `CompareRequest.scope_folder_id`,
`CompareResultPayload.scope_label`, `types.ts`, `compareRetrieval(..., scopeFolderId)`.

**Why:** (1) F2 put an additive column on the **answer path** (`answer_records.
retrieval_scope_json`), so the long-tolerated migration gap stopped being a sidecar problem and
became "every turn fails to record". (2) With a folder selected, an unscoped diff describes
retrieval the next answer will not perform — the same quiet mismatch F2 exists to remove. Holding
the document set constant across A and B is also what makes the comparison *about the knob*.

**Rejected:** for KI-20, a **startup schema check that only warns** — it diagnoses without fixing,
so the answer path still breaks until the user acts; and **wrapping the provenance write in
`suppress`** — that hides a schema fault by silently dropping provenance, i.e. buys uptime with
the integrity layer. For the compare, **leaving it unscoped but labelled** — the label removes the
lie but keeps showing a comparison the user can't act on.

**Verified:** 3 new tests (13 total in `test_retrieval_scope.py`); full suite **1102 passed / 1
skipped** · ruff · `mypy --strict src` · bandit · `svelte-check` 0/0 · docs+integrity 0/0. The
KI-20 guard test builds a genuinely stale schema (drops the column), starts the app, and asserts
the column is back — **verified non-vacuous**: it fails when the lifespan call is removed. Live
startup logged `schema_current` on this (already-migrated) box. **Live A/B through the real API
and real pipeline ($0, retrieval only, no generation):** unscoped compare reached `bge_cpack`,
`dpr_karpukhin`, `rag_lewis` — **all outside** the probe folder; scoped compare kept **both**
sides entirely inside it; `scope_label` null unscoped, `"__ab_probe__ (3 documents)"` scoped.
Probe folder deleted; DB left at 76 docs / 0 folders.

**Opens:** F3 (demo sha-match auto-assign) untouched. Multi-folder scopes, persisted
per-conversation scope, and scoping the enrichment sidecars stay parked (ADR-025). RG-020's
synthetic 10k measurement still owed.

---
## 2026-07-20 — F2: query-time folder retrieval scoping + the honesty contract (ADR-025 carve step 2)

**What:** built **F2** — a folder can now scope one chat turn's retrieval. Contract first:
`docs/specs/feature-corpus-folders-scope.md` (S1–S10). `library.folder_doc_hashes` resolves a
folder to its non-archived members' `doc_hash`. `pipeline.retrieve_with_scores(..., scope=)`
scopes **both arms before scoring** — vector via Chroma `$and[keep_for_retrieval≠False,
doc_hash $in [...]]`, BM25 by rebuilding over the subset of a now-retained `self._bm25_docs` —
memoised in one slot keyed on the hash set. `chat_controller` gains `ScopeView`,
`_resolve_scope`, `_scope_note`, a `scope_folder_id` parameter threaded like `overrides`, and
`TurnResult.scope`. Provenance gains an additive `answer_records.retrieval_scope_json` column.
API: `ChatRequest.scope_folder_id` + `TurnResultPayload.scope`. Desktop: a composer scope
selector (in-memory only) and a scope chip on the answer — plus the same chip on reopened
conversations, replayed from the record via `ConversationTurn.retrieval_scope`.

**Why:** F1 shipped a Library filter and said in-product that chat still searched everything.
F2 makes that false — and the same honesty rule then points the other way, which is the whole
content of this increment: a scoped answer that doesn't say it was scoped is indistinguishable
from a whole-library one, i.e. the `is_archived` failure with a nicer UI.

**Rejected:** (1) **putting the scope inside `RagOverrides`** — the plumbing is identical, but
`RagOverrides` is ADR-010's governance channel for *locked quality knobs*; a scope is a *content
filter*, and filing it there would blur exactly the distinction that keeps it out of the eval
gate (and would render it through `_overrides_note`'s "🧪 Session override", framing a content
choice as an experiment). (2) **Sending a doc-hash list from the client** — it goes stale between
a Library edit and the next turn, and would let a caller retrieve an arbitrary set that no folder
ever contained, which the provenance record would then attest to as "this folder". The id travels;
the backend resolves. (3) **Falling back to unscoped when the folder is unknown/deleted/empty** —
the single most important rejection: "I couldn't honour your scope" must never collapse into "I
searched everything", so an unresolvable scope is a distinct empty `frozenset` all the way down
and the turn answers honestly with zero sources. (4) **Persisting the selector** (localStorage /
server-side) — that is ADR-025's rejected global scope: a scope you forgot you set silently
narrows every future answer. (5) **Folding the scope into `prompt_version_hash`** — it would mint
a prompt version per folder and pollute every eval join keyed on it.

**Measured before deciding** (`tests/eval/baselines/rg020_scoped_retrieval_cost_2026-07-20.md`,
live 76-doc / 30,882-chunk index): BM25 subset rebuild ≈20 µs/chunk (622 ms whole corpus · 248 ms
for 30 docs · 27 ms for 3); Chroma `$in` 136 ms unscoped → 193/232/408 ms for 3/30/76 hashes —
the cost tracks the **`$in` list length**, not the corpus share. That is what bought the S5 cache.
**RG-020 partially discharged**; the 10k half stays open and is explicitly *not* claimed.

**Verified:** 17 new tests (`tests/unit/test_pipeline_scope.py` 7 + `tests/integration/
test_retrieval_scope.py` 10), including the S4 byte-identical guard, the S3 never-widen behaviour
on deleted/empty/unknown folders, both synthesis paths, scope isolation between turns, the API
round-trip and the additive-column migration. Full suite **1099 passed / 1 skipped** · ruff ·
`mypy --strict src` · bandit · `svelte-check` 0/0. **Live on the real corpus ($0, no LLM):**
unscoped retrieval hit 3 documents of which **2 lay outside** the probe folder; scoped retrieval
hit only in-folder documents; an empty scope returned 0 sources in 0.2 ms; unscoped retrieval
still worked afterwards. UI verified through the `window.fetch` SSE mock (**no paid turn**):
request body carries `scope_folder_id`, chip reads "Searched Retrieval demo only — 3 documents,
not the whole library", the deleted-folder variant reads "no documents were searched", an
unscoped turn adds no chip, selector tints only when scoped, light+dark, 375px no overflow, 0
console errors. Probe folders deleted; DB left at 76 docs / 0 folders.

**Opens:** ⚠ the live DB was **missing this column and `concepts.graph_include`** until
`init_db()` was run by hand — the API never migrates on startup (`apps/api/CLAUDE.md`), and F2
moves that gap onto the **answer path**, where it would 500 every turn. Logged as **KI-20**.
F3 (demo sha-match auto-assign) untouched. `compare_retrieval` (A/B) stays unscoped — stated,
not built. Multi-folder scopes, persisted per-conversation scope, and scoping the enrichment
sidecars remain parked (ADR-025).

---
## 2026-07-20 — F1: folders end-to-end (CRUD + membership + Library rail), ADR-025 carve step 1

**What:** built **F1** of the ADR-025 carve over the previously dormant `Folder`/`document_folders`
schema (0 rows). Contract first: `docs/specs/feature-corpus-folders.md` (D1–D9). Backend —
`library.py` gains `FolderSummary` + `list_folders`/`get_folder`/`create_folder`/`rename_folder`/
`delete_folder`/`add_documents_to_folder`/`remove_documents_from_folder`/`folder_document_ids`,
mirroring the shipped keyword-families surface (None = unknown, `ValueError` = blank/collision,
idempotent create, refreshed entity returned). `DocumentSummary` gains `folder_ids`;
`list_documents(folder=<name>)` becomes `list_documents(folder_id=<id>)`. API — six routes under
`/api/library/folders` + `LibraryFolderPayload`/`FolderCreate`/`FolderRename`/`FolderMembers`;
`types.ts` mirrored. Frontend — new `LibraryManageFolders.svelte` (create / inline rename /
confirm-delete / searchable bulk document picker, reusing the ManageKeywords modal shell), rail
section "Collections" → **"Folders"** rendering the API list with counts + a "Manage…" entry point,
`docsFor` matching `folder_ids`, and an **"Add to folder…"** item in the grid tile's ⋯ menu that
opens the view pre-filtered to that document.

**Why:** F1 is the demoable standalone step of the carve, and it is the piece that has to exist
before F2 can scope retrieval to anything. Reconciliation with the L4 Library-redesign spec is the
real judgment here — see "Rejected" below.

**Rejected:** (1) **the baton's "compose both auto-assign rules" instruction** — L4's own
2026-07-15 section already SHELVED source-dir subfolder mirroring when the user confirmed the
reopen condition (`source_dir` is flat by design), and named **manual assignment** as the only
honest path, gated on an ADR. ADR-025 is that ADR, so F1 builds manual assignment and mirroring
stays shelved; F3's sha-match is a separate rule, not a second mirror. (2) **Nesting** — the
schema is hierarchical but v1 creates every folder at the root (D1): ADR-025 flags nesting as the
reopener for the whole folders-are-groups identity, and it would force F2 to invent an answer to
"does scoping a parent include its children?". (3) **Name-keyed filtering** — `uq_folder_name_parent`
never fires for root folders (SQLite treats NULL parents as distinct), so uniqueness moved into
`library.py` and every filter keys on id (D2/D4). (4) **Deriving the rail from document payloads**
(`folderGroups`, now retired) — a folder derived that way cannot appear while empty, and an
invisible empty folder cannot be filled (D3). (5) **Deleting L4's write-trap test** — narrowed
instead: read routes still write nothing (D7).

**Honesty note (D8):** F1 ships the Library filter but *not* retrieval scoping, so the Manage view
states in-product that chat still searches every document. Without it, narrowing the Library reads
as narrowing the answer — the exact `is_archived` failure ADR-025 exists to prevent. F2 deletes
the line by making it false.

**Verified:** 15 new integration tests (`tests/integration/test_library_folders.py`) covering
case-insensitive idempotent create, blank/collision `ValueError`, unknown-id `None`/`False`,
idempotent membership, m2m overlap, archived excluded from counts (D5), id-based filtering, the
D6 "delete never touches documents" guard and the D7 read-path write trap. Full suite **1082
passed / 1 skipped** · ruff · `ruff format --check` · `mypy --strict src` · bandit ·
`svelte-check` **0/0**. **Live on the real 76-doc corpus ($0/offline):** created "Demo corpus",
bulk-added 3 documents → rail count 3 → grid filtered to 3 tiles with the breadcrumb resolving the
id to the name; a rename onto an existing name surfaced `a folder named 'demo CORPUS' already
exists` inline without blocking; deleting both folders left **76/76 documents intact** and reset
the active collection to All; light + dark tokens resolve; 375px no overflow; 0 console errors.
DB left with 0 folders.

**Opens:** F2 (retrieval scoping + per-turn selector + provenance/answer chip; carries RG-020) and
F3 (demo sha-match auto-assign + backfill) are untouched. `Tag` CRUD is the same shape and stays
dormant, deliberately not bundled. `DocumentDetails.folders` still ships names only (the drill-down
does not filter). Nesting, drag-and-drop assignment, and per-folder enrichment remain parked.

---
## 2026-07-20 — Docs: corpus groups grilled → design-locked as "Folders with retrieval scope" (ADR-025)

**What:** ran `grill-me` on the corpus-groups question the demo collection raised (demo corpus vs
personal papers in one store). 6 forks → all resolved or parked; ledger in the session baton.
**ADR-025** written (accepted, unbuilt): corpus groups ARE folders (reuse the dormant
`Folder`/`document_folders` schema — the ADR-015 reuse pattern); demo membership auto-assigned at
ingest by manifest sha-match + one-time backfill, user edits win (ADR-013 pattern); scoping = a
**query-time doc-hash filter on both retrieval arms** (no chunk-store writes; unscoped path
byte-identical); scope is per-turn request-scoped (ADR-010 pattern), sticky in UI only, and **the
provenance record + an answer chip always state the scope** (integrity, non-negotiable);
enrichment stays corpus-global in v1. Carve **F1 folders → F2 scoping → F3 demo auto-assign**,
spec at build time. Routed: RG-020 (scoped-retrieval bounds: Chroma `$in` latency at the 10k
contract + scoped-BM25 statistics + an unscoped-byte-identical guard test) and RG-021 (the eval
index-composition fingerprint, promoted from the 2026-07-20 demo-collection entry's "Opens");
ui-checklist §3 row (design-locked); decisions.md row.

**Why:** the user's fork — groups inside the main store vs a fully separable corpus — plus the
requirement that demo files stay easily deletable. The deciding constraint is the `is_archived`
precedent: doc-level flags scope every library-side read but NOT retrieval (chunks carry only
`doc_hash`), so any grouping that scopes the grid alone lies in chat. That makes retrieval
scoping the feature's core, not its garnish.

**Rejected (full list in ADR-025):** separate database (complexity, not storage — every read path
is corpus-global; env-level data home already gives coarse isolation); a new group object beside
folders; a partition column; chunk-metadata stamping (mutates the chunk store per edit);
post-rerank filtering (recall collapses on small scopes); persistent/global scope (a forgotten
scope silently narrows every answer).

**Opens:** F1's spec must reconcile ADR-025 with the L4 Library-redesign spec's Phase-B locks
(2026-07-14: "folders = mirror source subfolders at ingest + backfill") — two auto-assign rules
compose, they don't compete. Per-folder enrichment parked with a named reopener (facet clutter →
PR-2.7 demotion first). RG-020/021 carry the deferred measurements.

---
## 2026-07-20 — Demo-corpus removal: `download_corpus --remove-demo` (content-hash matched, ADR-014 safe-delete, dry-run default)

**What:** the demo collection is now cleanly removable. Core in `src/doc_assistant/library.py`
(scripts stay thin per the module contract): `SourcePin`/`SourceMatch`/`SourceRemoval` +
`match_pinned_sources()` — finds files under the sources dir by **content** (size fast-path so a
big corpus costs stats not reads, then SHA-256; rename-proof) and links each to its library row by
filename (content can't bridge that hop: `doc_hash` hashes extracted text, not file bytes; >1 row
sharing a name → flagged ambiguous, never auto-deleted) — and `remove_pinned_sources()` — ingested
matches go through `delete_document` (ADR-014: Recycle Bin first, then row/chunks/sidecars)
against the live index, the same doc's chunks are swept from the secondary Chroma store too (the
API delete only cleans the live one), never-ingested files go straight to the Recycle Bin; a
refused trash (locked file) fails that one match and the batch continues. Script:
`--remove-demo` (plan) / `--remove-demo --apply` (execute) per the dry-run-default polarity;
`_chunk_stores()` opens both Chroma stores **without loading the embedder** (get/delete never
embed) so cleanup works model-cache-free. **7 new integration tests**
(`tests/integration/test_demo_corpus_removal.py`) on the ADR-014 test harness. Verified live:
dry-run against the real `data/sources/` found **exactly the 18 just-downloaded demo files**
(all correctly triaged "file only — never ingested"), removed nothing. Full suite **1067 passed /
1 skipped** (pre-existing); ruff · `mypy --strict` · bandit(src) clean; `docs_check --strict` 0/0.
README demo note + corpus README gained the removal line.

**Why:** the corpus-groups discussion (2026-07-20): whichever way grouping lands later, "someone
wanting to use the app should be able to delete those demo files easily" stands alone — and it
rides entirely on shipped machinery (manifest pins + ADR-014), so it ships now while corpus
groups waits for its grill + ADR.

**Rejected:** matching library rows by `doc_hash` (it hashes extracted markdown, not file bytes —
no bridge from a PDF's sha256); hard-deleting anything (ADR-014's whole point — Recycle Bin +
re-download keeps every step reversible); a separate `scripts/remove_demo_corpus.py` (removal is
the download's inverse; one manifest-owning script, one `--dest`); auto-deleting on ambiguous
filename collisions (deleting the wrong user document to save a demo-cleanup click is the worst
trade available).

**Opens:** the bandit B310 `urlopen` advisory in `download_corpus.py` is pre-existing and outside
the gate (bandit runs on `src/` only) — fine, but worth a `# nosec` + comment if scripts ever
enter the gate. A file renamed *after* ingest removes as file-only and leaves its stale row to
the ingest orphan cleanup (documented in the docstring). The secondary-store sweep exists because
API deletes clean only the live index — if that ever changes in `delete_document` itself, drop
the sweep here.

---
## 2026-07-20 — Public corpus: 18-paper demo collection (Sutskever→Carmack list) + `download_corpus --demo`; verified-10 regime pinned by a guard test

**What:** `tests/eval/corpus_manifest.yaml` gains a **`collection: demo`** section — the 18
arXiv-pinnable papers of the rumoured Sutskever→Carmack reading list (30papers.com): ResNet +
identity mappings, dilated convs, RNN regularization, Deep Speech 2, Order Matters, Bahdanau
attention, Pointer Networks, the Transformer, NTM, relation nets ×2, MPNN, scaling laws, GPipe,
the coffee automaton, VLAE, and the Grünwald MDL tutorial (old-style id `math/0406077`). Every
entry pinned the honest way: ids + latest versions verified against the arXiv API, then **each
pinned-version PDF actually downloaded** (scratchpad, stdlib urllib + truststore through the
corporate proxy — the KI-10-addendum transport; 3 s spacing) and SHA-256 + byte-size recorded;
**18/18 re-verified through the script's own `--verify-only` path** (0 mismatches).
`scripts/download_corpus.py`: new pure `_selected()` + **`--demo` flag** (default selection
unchanged = the eval 10), a 3 s politeness sleep between real fetches, and an inform-don't-block
summary line when demo entries are excluded. **New guard
`tests/unit/test_download_corpus_selection.py` (5 tests)** pins the default selection to exactly
the verified-10 and every demo entry to `referenced_by_eval: false`. Docs: README Usage gains the
"try it on a ready-made corpus" note; `evals/README.md` + `tests/eval/corpus/README.md` state the
demo collection is never part of the benchmark regime. Unit suite **841 passed**; ruff clean;
`docs_check --strict` 0/0.

**Why:** the 2026-07-20 evals-split session scoped this (ADR-024 "Opens"): the app demos better
on a bigger, famous corpus (concept graph, wiki, gaps), but the verified-10 benchmark regime must
stay closed — extra corpus documents are retrieval distractors, so demo papers must be opt-in and
excluded from every published number. The list itself: zero overlap with the eval 10 (RAG methods
vs DL classics), all freely downloadable, nothing re-hosted.

**Rejected:** `tier: demo` (the chip spec's literal wording) — in the real schema `tier` is
*source provenance* (`arxiv` vs the forward-compat `committed`), so membership got its own
explicit `collection` field with absent = eval (pre-demo entries untouched, byte-identical);
reusing `referenced_by_eval` as the selector (conflates "a case cites it" with "in the eval
corpus" — a deliberate distractor paper would break the equivalence); the 9 non-arXiv items
(AlexNet, Hinton MDL, Cover–Thomas chapter, Legg thesis, CS231n, blog posts) — noted in the
manifest as ingest-as-HTML candidates, not silently fudged in.

**Opens:** running `--demo` then a benchmark run on the same index produces non-comparable
numbers — the docs say so, but nothing *mechanically* stops an eval run over a demo-polluted
index (would need an index-composition fingerprint in the eval harness; noted, not built). The
HTML items (Karpathy/Olah/Aaronson posts, CS231n) would exercise the HTML ingest path if ever
wanted. arXiv re-renders make SHA mismatches warnings by design — if one fires later, re-pin and
note it here.

---
## 2026-07-20 — Docs: benchmarks split out of README into a top-level `evals/` folder (ADR-024)

**What:** New top-level `evals/README.md` now holds the full benchmark write-ups — the headline
public benchmark (table + interpretation + the sbert_motivation judge-flakiness caveat), the
`bge-base` vs `specter2` embedder comparison, the chunk-size sweep, the BM25/vector-weight sweep,
and both reproduction guides — moved verbatim from the README's ~95-line Benchmarks section
(links re-based `../`), plus a "where the eval pieces live" map and the public-10 vs private-35
question-set split. The README's Benchmarks section shrinks to the headline 3-scorer table + one
interpretation paragraph + links (anchor `#benchmarks` kept — both in-README references still
resolve); layout tree gains the `evals/` line; the Status embedder note and the Running-tests
comment now point into `evals/`. `docs/architecture.md` gains a one-sentence pointer.
Decision recorded as `docs/decisions/ADR-024-evals-results-folder.md` + index row.

**Why:** the README is the door (readme-writer), and ~95 of its 413 lines were archive-depth
benchmark detail; the eval story also had no front door — harness in `src/doc_assistant/eval/`,
strategy/cases/baselines in `tests/eval/`, narrative only in the README. User directive named the
split and questioned the folder name; `evals/` over `benchmarks/` because "benchmarks" reads as
performance and the repo's own vocabulary is *eval* everywhere.

**Rejected:** `benchmarks/` as the name (vocabulary, above); moving `tests/eval/` wholesale into
the new folder (61 files reference those paths — script defaults, the CI ignore, code comments,
frozen append-only records; all churn, no gain); `docs/evals/` (the folder is audience-facing —
top-level GitHub visibility is the point). Full ledger: ADR-024.

**Opens:** `evals/` should accumulate future result write-ups (baseline data still goes to
`tests/eval/baselines/` per the locked-settings rule — narrative vs data). Separately scoped, not
built: a `tier: demo` extension of the public corpus from the 30papers.com list (the rumoured
Sutskever→Carmack 27) — ~17–18 are arXiv-pinnable; must stay OUT of the verified-10 benchmark
regime (extra distractor docs change retrieval difficulty and would invalidate every committed
baseline), so it needs a downloader flag + manifest tier before any papers are added.

---
## 2026-07-19 — Verify-the-app pass: root-caused the "6 pre-existing send2trash failures" → a live 500 bug (KI-22) + a dependency-presence guard test

**What:** Ran a full app-verification pass (all gates + a live $0 Ollama chat turn on the real
47-doc/16,039-chunk corpus). The turn worked end-to-end — correct SSE shape (step → 296 token →
result → done), 10 cited sources, 9 flagged claims, epistemics markers firing, `is_local:true`
`cost_usd:null` — and the concept graph served the ADR-018 numbers (13 nodes/19 edges/6 communities,
27 gaps = KI-17 reproducing). But the "6 pre-existing send2trash failures" the baton had carried as
*"venv drift, unrelated"* turned out to be a **real shipped-feature break**: `DELETE
/api/library/documents/{id}` 500s on every call because the declared base dep `send2trash>=2.1.0`
(`pyproject.toml:84`) was absent from the venv, imported lazily inside `library.delete_document`
(`library.py:330`) so it fails at call time, and the route catches only `RuntimeError` so it escapes
as a 500. Verified live with a nonexistent-id probe (deletes nothing) → 500 before, 404 after.
**Fix:** `uv pip install "send2trash>=2.1.0"` (venv-local, per-machine); suite went **1015 → 1021
passed, 0 failed** — first fully-green run in several sessions. **Added
`tests/unit/test_declared_dependencies.py`** (+35 tests): asserts every `[project].dependencies`
entry resolves via `importlib.metadata.version`, failing **by package name**, plus a pin on the exact
`from send2trash import send2trash` form. Recorded KI-22; the committed change is the guard test +
KNOWN_ISSUES/DEVLOG (the venv fix is gitignored `.venv` state).

**Why:** the failing tests were the suite correctly reporting a broken feature, but the cryptic
`ModuleNotFoundError`-from-monkeypatch shape made "test-infra noise" look plausible, so the misread
survived multiple sessions. A guard that fails by package name — "declared runtime dependency 'X' is
not installed … missing-dependency drift, not a test-infra flake" — makes the next such gap
unmissable and un-mislabellable.

**Rejected:** `uv sync` to restore the dep (would pull the multi-GB cu130 torch wheel, KI-3) —
installed the one pure-Python package instead; broadening the route's `except` to swallow the missing
dep (a missing hard dependency is a broken install, not a runtime condition to handle — the guard
test is the right layer); moving the lazy import below the unknown-id early return (papers over a
missing *required* dep without fixing it).

**Opens:** the guard only covers base `[project].dependencies`, not the `cpu`/`cu130`/`dev`/`packaging`
extras (an absent extra is expected on a lean install, so asserting it would false-positive); revisit
if an extras-drift bug ever bites. The baton's habit of labelling red tests "environmental" is worth a
cross-project atlas lesson (proposed, awaiting say-so).

---
## 2026-07-19 — Public docs refresh: README demo GIF + status/limitations truth-up, DEMO.md touch

**What:** (1) **Recorded a real demo GIF** and embedded it at the top of the README
(`docs/assets/provenote-demo.gif`, 23 frames, 1.73 MB, 960px): empty state → sample chip →
a genuinely streamed cited answer → the source side panel (with the per-claim review) → the
library grid → the concept-graph ego view. Recorded against the real 47-doc corpus on
**`ollama/llama3.1:8b` ($0 — provider switched via `/api/settings` and verified on
`/api/health` before any turn; KI-4)** by driving the dev app (API :8001 + Vite :1420) with
puppeteer-core + installed Chrome (the Browser pane's screenshot capture times out on this box —
known quirk, 2026-07-15 baton), frames assembled with Pillow. Recording tooling stays in the
session scratchpad (would add undeclared npm/Pillow deps if committed); pipeline documented in
agent memory. Side effects: 3 real 1-turn conversations now sit in this box's history (no DELETE
endpoint; harmless), and the provider switch surfaced a gitignore gap — the app's persisted
`data/settings.json` (U1c) was untracked-but-not-ignored and would have ridden into a public
commit; now gitignored as per-machine runtime state. (2) **README truth-up:** Status was frozen at 2026-07-02 ("concept graph
not yet usable", "gap detection blocked on RG-001", "712 tests") — now reflects the shipped
graph/gaps/markers/library/provider-switch stack, **1,015 tests**, the ADR-021/022/023
restructure, and links the scale review. New **Limitations** section (validated at ~50–100 docs
with the review's scale caveat, local-model ceilings, the KI-8/WE-7 marker-loss truth,
single-user design, Windows-first testing). "What it does" gains the concept-graph/markers/
library bullets; Project layout shows the db/ingest/knowledge/eval subpackages; decisions.md
references point at the index + archived monolith. (3) **DEMO.md** gains `just app`, the
Library/Graph walkthrough beat, and the GIF pointer.

**Why:** the README is the public face; it under-sold three shipped phases and over-claimed
nothing — but its status text was five iterations stale, and the user asked for a UI GIF.

**Rejected:** committing the GIF recorder into `scripts/` (undeclared puppeteer-core/Pillow
deps; revisit if the GIF needs regular regeneration); re-recording to purge the history rows
(real data, not worth touching `library.db`).

**Opens:** GIF re-record wanted after the next visual-identity pass; consider a
`docs/assets/` dark/light pair if the README ever needs theme-aware media.

---
## 2026-07-19 — C4 scale-robustness review: knowledge layer vs specs/ADRs at 0 docs and 10k docs (docs-only)

**What:** ran the user-directed in-depth review of the whole `knowledge/` layer against its own
specs/ADRs under four lenses (zero-doc, scale 0→10k, corpus-tuned constants, conformance) — four
independent read-only review passes (one per cluster), every finding required to quote the code
line it stands on; the seven highest-stakes claims re-verified by hand before publication (all
seven held). **Output: `docs/REVIEW_2026-07-19_scale-robustness.md`** — 36 findings + a
corpus-tuned-constants inventory + a P0/P1/P2 fix plan. Headlines: (a) zero-doc discipline
largely HELD (honest empty states everywhere; 2 crash edges in wiki/epistemics builders; the
contract is unpinned by tests); (b) every cluster has ≥1 corpus-linear-or-worse hot path
(unpaginated whole-corpus loads, a per-edge doc×doc Cartesian provenance product, O(chunks ×
concepts) full-recompute projection with a 512-pattern regex-cache cliff, O(n²) family cosine,
three unbounded LLM loops); (c) the over-optimize-on-current-corpus complaint is CONFIRMED and
localized — frozen Q1 `min_degree=3` whose docstring claims "corpus-derived", family threshold
0.86 **above bge's measured ceiling**, `contested` on `nc>=1` already marking 53.6% of chunks,
the monolith's recorded-wrong absolute-cosine 0.90 still the wiki default; (d) three conformance
breaks: curation hard-deletes vs ADR-018's demote (KI-20), the in-app rebuild never runs
`build_gaps` so the view serves stale gaps (KI-21), and KI-8's containment rationale is
arithmetically backwards — straddling chunks lose markers, they don't double-mark.

**Why:** the product contract (works at 0 docs, scales to 10k) had never been a review lens;
sessions kept tuning to the 47/76-doc corpora.

**Routed:** KNOWN_ISSUES **KI-18** (scale cliffs) / **KI-19** (tuned constants + LLM budgets) /
**KI-20** (delete-vs-demote) / **KI-21** (rebuild half-refresh) + KI-8 direction correction +
KI-17 fix-placement correction; RIGOR_TODO **RG-016..019** (each constant's owed experiment);
ROADMAP C4 done. **No code changed by this review** — fixes are follow-up sessions per the plan;
P2 constants are measurement-gated (never hand-tune).

**Rejected:** fixing "obvious" P0s inline this session (the session already carries the ADR-021/
022/023 restructure; review-then-fix in one diff would bury both); treating the review passes'
findings as publishable without an independent verification step (all 36 numbered findings were
consolidated; the 7 highest-stakes were re-verified line-by-line before anything was routed —
one of the 36, KW-8, is a positive no-defect trace, kept because it documents the 0-doc contract).

**Opens:** the P0 list is the natural next session (small, no eval needed); the LLM-budget policy
wants one ADR covering Node B / gap_suggest / wiki caps together.

---
## 2026-07-19 — ADR-023: knowledge/ subpackage — 11 corpus-derived modules out of the flat package

**What:** created `src/doc_assistant/knowledge/` and `git mv`'d the Phase-7 feature cluster into
it: `concept_curation`, `concept_graph_view`, `concept_semantics`, `concept_skeleton`,
`concept_skeleton_enrich`, `epistemics`, `gap_suggest`, `gaps`, `keyword_families`, `keywords`,
`wiki` (histories preserved). Package docstring states the layer's contract (Enrichment-Layer
sidecars; the answer path reads it, never depends on it). **49 files** rewritten to
`doc_assistant.knowledge.<mod>` (script-driven, word-boundary-safe, `--verify` pass shows 0
old-path references; covered `from doc_assistant.X import`, `from doc_assistant import X as y`,
`import doc_assistant.X`, and docstring prose; no monkeypatch-string forms existed). Living
docs/specs path-updated (KNOWN_ISSUES, RIGOR_TODO, ui-checklist, feature-concept-graph/-gap-
detection/-7d specs); `docs/architecture.md` module map + Mermaid gain the knowledge/ node;
`src/doc_assistant/CLAUDE.md` layout updated. Append-only records keep historical paths. Kept at
top level deliberately: `synthesis.py` (answer-path Chunk 2a), `tracking.py` (token infra),
`doc_vectors.py` (Phase-4 similarity input), the whole RAG path, and the existing `db/` /
`ingest/` / `eval/` — per the directive. ADR: `docs/decisions/ADR-023-knowledge-subpackage.md`.

**Why:** 63 modules, 40+ flat — "the concept graph" had no boundary to stay inside; the flat
listing stopped communicating the architecture (cpc §12: a real subsystem boundary earns its layer).

**Rejected:** naming it `features/` (generic, collides with `docs/features/`); compatibility
re-export shims at old paths (nothing external imports the package — cpc §12 no-speculative-
abstraction); touching `scripts/archive/` (frozen, unmaintained).

**Gate:** ruff ✓ (3 E501s fixed — two rewrite-lengthened docstrings + one pre-existing 100-char
line in `apps/api/main.py` that apps/-scoped habits had missed) · format ✓ · `mypy --strict src`
64 files ✓ · bandit 0 HIGH/MED ✓ · **pytest 1015 passed / 6 failed — byte-identical to the
pre-restructure failure set** (the known send2trash venv drift, `pyproject.toml` declares it,
`.venv` lacks it; `uv sync` fixes but pulls the multi-GB cu130 wheel — deliberately left).

**Opens:** none new; the Phase-D scale review (C4) now has a named review surface.

---
## 2026-07-19 — ADR-022: docs-system rationalization — index over monolith, DEVLOG fully inverted, per-artifact verdicts

**What:** decided which doc layers earn their place at scale and executed (ADR-022). (1)
**`docs/decisions.md` monolith (1578 lines) → frozen verbatim** at
`docs/archive/decisions-monolith.md` (`git mv`, header → archived/append-only, provenance note);
the path is now a **living ADR index** (one line per ADR-001..023 + the going-forward rule:
every decision = one ADR file + an index line). Re-scopes ADR-001 Step 4 — the planned ~50-file
split would have produced mostly-dead micro-ADRs duplicating ADR-002..021 and ROADMAP status. (2)
**DEVLOG ordering fixed once (completes ADR-001 Step 5):** the 103-entry oldest-first
`## Session:` historical block (2026-05-21 → 2026-07-04, lines 3323+) is now inverted below the
78-entry newest-first block — whole file strictly newest-first; entry **bodies verified
byte-identical** (script check), only `## Session: ` prefixes stripped; this entry is the logged
reformat note. (3) **Per-artifact verdicts recorded** (ADR-022 table): ADRs canonical · specs =
code-level contract layer · sprints = the delegated-execution mechanism (kept, roadmap_sync flow)
· `features/` = why-it-works layer, adopt for *frontier* features only, no backfill · loose
explainers stay · archive unchanged. (4) References updated: CONTEXT (×2), ROADMAP (header, table
intro, not-to-do, +C1–C4 rows, date bump), AGENTS.md (index + "top entries").

**Why:** the half-migrated state (living-labeled frozen monolith, hybrid DEVLOG order, unused
features/ layer) was context every session loaded and misread; scaling multiplies that cost.

**Rejected:** executing ADR-001 Step 4 as written (50 micro-ADRs — see ADR-022 option 1); deleting
`docs/features/` (cpc-init would re-lay it; the layer genuinely covers the hypothesis→outcome
record specs don't); bulk-materializing sprint contracts for planned rows (done per-row at
plan-start).

**Opens:** `docs/decisions.md` index is hand-maintained — a `cpc-generate` candidate if it rots;
FEATURE files start with the next frontier feature; the monolith's "Deferred Improvements" items
(wiki clustering threshold, coverage floor, SPECTER2 …) remain live backlog references into the
archive. Gate: docs_check 0 errors (2 pre-existing rule-12 warns clear with Phase-D edits).

---
## 2026-07-19 — ADR-021: cpc big-project layout — AGENTS.md entry + module CLAUDE.md files + vendored gates (this box)

**What:** adopted the cpc big-project variant (user request; ends ADR-001's conscious deferral of
cpc ADR-014). (1) **Entry layer:** new root `AGENTS.md` (canonical, tool-neutral; content ported
from the old `CLAUDE.md` + sub-module map + cpc ADR-020 keypoints table); `CLAUDE.md` is now a bare
`@AGENTS.md` stub, gate-enforced via `[entry] enforce_stub = true`. (2) **Module files** (≤40 lines
each): `src/doc_assistant/CLAUDE.md`, `apps/desktop/CLAUDE.md`, `apps/api/CLAUDE.md`,
`scripts/CLAUDE.md` — local traps only (wire-type drift, `--apply --enrich`, KI-4 provider guard),
globals by code. (3) **Conventions tooling separated from scripts:** cpc **1.2.3** vendored at
`tools/conventions/cpc/` via `cpc-init --profile standard` (run from the local cpc checkout at the
release tag; this box previously had NO vendored copy) + new `rungate.py` shim;
`.pre-commit-config.cpc.yaml` rewritten from pip-installing a pinned SHA of the **private remote**
to running the vendored copy (no network at hook time); justfile gains facade recipes
(`just check`/`lint`/`keypoint`). (4) `GLOSSARY.md` laid + **filled** (11 entries pinning the
Concept/Keyword/family/skeleton/gap vocabulary — the 2026-07-17/18 junk-labels trap is a
vocabulary-drift failure); `scripts/conventions.toml` refreshed to the 1.2.3 key set (project
values kept). ADR: `docs/decisions/ADR-021-adopt-cpc-big-project-layout.md`.

**Why:** 60+ backend modules across 4 real boundaries passed cpc §9's threshold; module-local traps
kept biting sessions that loaded only root context; this box's gate wiring was stale (pre-vendoring,
remote-dependent) and skewed vs the work box.

**Rejected:** staying CLAUDE.md-canonical (non-portable, permanent init-check deferral); a
CLAUDE.md→AGENTS.md symlink (degrades to plaintext on Windows clones — cpc ADR-014's own analysis);
moving `scripts/conventions.toml` out of `scripts/` (cpc `_config.py` resolves that exact path —
it is gate config, not a script; labeled instead).

**Opens:** `cpc-init-check` passes for the first time (kept on-call, not wired);
`src/doc_assistant/CLAUDE.md` must be updated when the knowledge-layer subpackage lands (same
session, ADR-023); Cowork project settings should be re-pointed at `AGENTS.md` (settings action,
outside the repo). Gate: `docs_check --strict` 0 errors; 3 pre-existing rule-12 date-bump warnings
(ROADMAP / KNOWN_ISSUES / RIGOR_TODO) left to clear with this session's later edits to those files.

---
## 2026-07-18 — ADR-020: share `RIGOR_TODO.md` via git (the two boxes held disjoint rigor trackers)

**What:** added `!.claude/RIGOR_TODO.md` to the `.gitignore` allowlist beside `CONTEXT.md` and
`KNOWN_ISSUES.md`, amending **ADR-001**'s `.claude/` contract. New
`docs/decisions/ADR-020-share-rigor-todo-via-git.md`; `CLAUDE.md`'s tracking line updated; the tracker's
own header rewritten from a "per-machine note" into a shared-file header carrying an item inventory and a
first-sync merge procedure. `SESSION.md` stays local — it genuinely *is* per-machine state.

**Why:** ADR-001 grouped the rigor tracker with the `SESSION.md` baton, but they are different kinds of
file. The baton records *"who worked last on this box"*; the rigor tracker records *validation debt of the
codebase*, which is true whichever machine you are sitting at. Grouping them let the two boxes accumulate
**disjoint item sets** for ~3 weeks.

**The failure is not hypothetical, and that is why this got fixed rather than noted.** **RG-014 has no
entry on this box** — while being cited as authority in **ADR-017, ADR-018, ADR-019,
`docs/specs/feature-concept-graph.md` and `docs/ui-checklist.md`** for "`single_source` is the strong,
low-volume gap signal". A week of design decisions rested on an item nobody working here could read — and
on 2026-07-18 that same verdict was found **not to transfer** across the ADR-018 vocabulary rescope, which
is exactly the bound a reader would have checked had the text been reachable. This copy holds
RG-001/008/009/010/011/012/013/015; RG-014, RG-007 and possibly RG-003/005/006 live only on the work box.

**Publication surface checked first — this repo is public.** Scanned for absolute user paths,
credentials, tokens and hostnames: **none**. Content is engineering measurements and box nicknames; the
one sensitive-sounding detail (a corporate TLS-MITM proxy) is **already public** in the committed
`KNOWN_ISSUES.md` KI-10. The publish decision stays the user's — staged, not committed.

**Rejected:** *keep it local + add a reconciliation ritual* — this **is** the status quo; the file has
carried a "still to reconcile against the work box" instruction since 2026-07-01 and it never happened, so
adding a second reminder supplies no mechanism (git is the mechanism). *Fold rigor items into
`KNOWN_ISSUES.md`* — conflates defects with validation-debt-on-work-believed-correct, and the
`rigor-gate` skill addresses `RIGOR_TODO.md` by name. *Move it to `docs/`* — breaks every existing
reference for nothing the allowlist entry does not already give.

**The hazard this change creates, and how it is contained:** the work box still holds the file as
**untracked and ignored**. On `git pull` git will refuse to clobber it — **that refusal is the safety
net and must not be forced past**. The tracker's header now carries the procedure (rename local copy
aside → pull → hand-merge the missing RG items → delete the temp) plus an explicit present-vs-missing
inventory, so the first sync is a **merge, not an overwrite**.

**Opens:** the merge itself, which can only be done on the work box — until then the shared copy is
incomplete **and says so in its own header**. Also surfaced, logged not fixed: the file's line *"The gate
(`rigor_gate.py`) fails while any `blocks-ship` item is `open`"* is **aspirational** — there is no
`scripts/rigor_gate.py` in this repo and neither pre-commit nor CI reference it. Sharing the file does not
make it enforcing; wiring a real gate is unticketed. **Staged; nothing committed (cpc §13).**

---
## 2026-07-18 — Stage-0 candidate ranking: triage mined keywords before promotion (read-only)

**What:** a **stage 0** for vocabulary curation in `concept_curation.py` — `rank_candidates()` (pure) +
`harvest_name_bigrams()` (pure) + `rank_keyword_candidates()` (impure wrapper), behind a read-only runner
`scripts/rank_candidates.py`. Orders mined keywords by **document reach** and reports three signals per
candidate: `docs` (distinct documents), `artifact` (reuses the existing deterministic `is_artifact`), and
an advisory `author?`. +13 tests. **Read-only — it ranks, never promotes, excludes, or writes.**

**Why:** the module's existing three stages prune a vocabulary that was *already* promoted; nothing ran
before promotion, which is why `--promote-all` was so destructive. The measurement that frames it:
**672 of 688 keywords (97.7%) appear in exactly one document** — the keyword extractor scores per-document
salience, not cross-document vocabulary, and a singleton keyword can never form a co-occurrence edge, so it
enters the skeleton as a permanently isolated node. Ranking by reach cuts the review set **688 → 16 (2.3%)**
without classifying anything.

**Ranking, not filtering — and that is a deliberate reaction to the same day's mistake.** Nothing is
auto-excluded: `pddl` is a legitimate 1-document concept, and the correction entry below records what
auto-exclusion produces on a multi-domain corpus. Signals order a human's review; they do not act.

**The author signal is reported honestly as weak.** `documents.authors` is free text that often holds a
*whole citation* — `"Omar Khatab and Matei Zaharia. 2020. ColBERT: Efficient…"` — so the field contains
paper **titles** as well as people. Measured live: 290 name bigrams harvested, **3 keywords flagged, only
1 a real author name** (`ziyang wang`); the others (`usage cards`, `responsibly reporting`) are title
fragments. **~1/3 precision → advisory only, never an auto-exclude.** Two guards keep it cheap rather than
catastrophic: only **capitalised** word pairs are harvested, and only **multi-token** candidates are ever
matched. That second guard is load-bearing and has its own test: **`bert` appears in 4 authors strings and
`colbert` in 1**, so a substring rule would silently drop two of the most important concepts in an IR
corpus. Noise classification stays with the existing `classify_noise()` LLM seam, whose prompt already
names author names as noise — free on Ollama, and not worth reimplementing deterministically at 1/3
precision.

**Rejected:** a hard `>=2 docs` gate (kills `pddl` and every legitimate single-source concept — and
`single_source` is the gap layer's *strongest* signal, so gating on reach would suppress the findings
ADR-004 exists to produce); substring matching against `documents.authors` (drops `bert`/`colbert` —
measured, not hypothesised); a new module (this is vocabulary quality, which `concept_curation.py` owns,
and it already had `is_artifact` to reuse); auto-promoting the top N (redesign **Decision 1** — the vocabulary
is curated by the user, never auto-extended).

**Live output on the real corpus (read-only, $0):** 688 candidates → 16 with reach >=2 → **8 unpromoted**,
of which 5 are real concepts the graph is currently missing — **`medical image segmentation` (3 docs)**,
`cajal`, `dice score`, `mamba`, `rag` — and 3 are correctly flagged artifacts (`18653 v1`, `10 18653 v1`,
`mrr 10`). It also confirms the morphological duplicates predicted from the pool: **`passage`/`passages`
and `mrr`/`mrr 10` are all promoted concepts** — `dedup_pairs()` (stage 3, already built, never run) is the
tool for those. Gate: ruff ✓ · format ✓ · `mypy --strict src` (63) ✓ · bandit 0 HIGH/MED ✓ ·
**1015 passed** (+13).

**Opens:** the ranker surfaces candidates but **promotion is still one-at-a-time via `seed_concepts
--promote`** — a batch review UI (or an `--llm` judged pass over just the top-ranked slice) is the natural
follow-up, now that the review set is 16 rather than 688. **25 of 47 documents still have zero concept
presence** — reach-ranking cannot fix that, because those documents' keywords are singletons *by
construction*; closing it needs per-domain seeding from document titles/abstracts, which is a different
instrument. `dedup_pairs()`/`classify_noise()` both remain built and unrun.
**Staged; nothing committed (cpc §13).**

---
## 2026-07-18 — CORRECTION to the ADR-018 entry: the 4 "junk" concepts are real specialist vocabulary

**What:** retracts one claim in the ADR-018 entry below — that `cre`, `dbs`, `ntsr1`, `pddl` are "4 junk
manual entries … worth curating out". **They are correctly curated domain concepts.** Nothing was removed.
The `set_graph_include(cid, False)` action item that rode on that claim is withdrawn.

**The evidence (traced to source, which is what the original claim skipped):**

| concept | alias | home document(s) | mentions |
|---|---|---|---|
| `cre` | Cre recombinase | mouse axonal-projection paper · "Neuroanatomy goes viral!" | **203** |
| `dbs` | deep brain stimulation | hypothalamic stimulation · dopamine/beta-oscillations | **134** |
| `ntsr1` | neurotensin receptor 1 | mouse whisker-cortex paper | 30 |
| `pddl` | planning domain definition language | hierarchical-planning paper | 46 |

Every one carries a correct expansion alias and real textual presence. **`cre` has more mentions than
`BM25`** (203 vs 137).

**Why the error happened, precisely:** the corpus spans at least four domains (IR/RAG · systems neuroscience ·
viral tracing/mouse genetics · AI planning), but the vocabulary was judged against the *one* domain the
session had been reading specs about. Lowercase acronyms sitting beside `BM25`/`dense retrieval` were
pattern-matched as extraction noise **without opening the documents they came from.**

**The supporting argument was also inverted.** "3 single-concept communities and 6 of the 15 gaps" was cited
as evidence of junkiness. That is the gap layer working: `isolated`/`single_source` on `pddl` means *"you own
exactly one AI-planning paper"* — a true coverage finding, which is the whole point of ADR-004. A correct
signal was read as a defect.

**This is a REPEAT.** The 2026-07-17 entry ("Manage view at scale scoped (PR-2.7)") reached the identical
conclusion one day earlier, in the same words — *"they are **mostly real specialist vocabulary, not junk**"*
(`16p11` = 16p11.2 truncated at the dot; `c57bl` = C57BL/6 across 7 docs; `va1v`/`dl5`/`osns` = Drosophila
glomeruli) — and set the rule **"demote, not delete: deleting real vocabulary isn't reversible-by-search."**
That rule was not consulted. Recorded as a trap in `docs/specs/feature-concept-graph.md` so the third
occurrence is cheaper than the second.

**What actually is wrong** (the finding the false one was hiding): **25 of 47 documents have zero concept
presence** — the vocabulary is too *small* and too *IR-skewed*, not too impure. Root cause of the poor
candidate pool: **672 of 688 keywords (97.7%) appear in exactly one document** — the extractor produces
per-document salience, not cross-document vocabulary, so `--promote-all` imported 672 document-specific
strings. Next PR ranks candidates instead of promoting them wholesale.

**Opens:** nothing removed, so no rebuild was needed; the 13-concept graph stands as measured.

---
## 2026-07-18 — ADR-018: scope the graph vocabulary with an opt-in `graph_include` flag (357 → 13 nodes)

**What:** added a nullable `graph_include` flag to `Concept` and filtered `concept_skeleton.load_concepts()`
on it (**ADR-018**). `library.list_keyword_families()` stays **unfiltered** — that asymmetry is the whole
decision. Creation paths follow one rule: `add_concept()` opts **in** (new `graph_include: bool = True`
param — the deliberate glossary path), `promote_keyword()` and `library.create_keyword_family()` opt
**out**. New `set_graph_include()` write surface + `backfill_graph_include()` (dry-run by default, touches
only `IS NULL` rows) behind a thin runner, `scripts/backfill_graph_include.py`. Migration is one append to
`db/migrations.py` `_ADDITIVE_COLUMNS` (+ an index — the filter runs on every build). +14 guard tests.

**Why:** **ADR-015's named "boundary risk" materialized.** Tag families and graph nodes are the same
`Concept` rows, and the two features want opposite things from that table — families want breadth, the
graph wants a small curated map. Measured on this box: **all 344** `source="keyword"` concepts share one
`created_at` (**2026-07-05**) — a single `seed_concepts.py --promote-all` run, against `promote_keyword`'s
own documented contract that a Keyword is *"a candidate only — never auto-written"*. The graph was 357
nodes of `'speckles'`/`'hyaline'`/`'13 intentionally omitted'`, and `single_source` was **224 of 302**
gaps — the exact signal RG-014 found strong *because* it was low-volume at 26 concepts.

**Polarity is the load-bearing half:** opt-in, so NULL reads as excluded and a new row never enters the
graph unbidden. Opt-out would let the identical regression recur on the next bulk operation; opt-in makes
re-flooding structurally impossible. A test asserts exactly that (`test_bulk_promotion_cannot_reflood`).

**Rejected:** filtering on the existing `source='manual'` (overloads a *provenance* field as a *curation*
control — they diverge the moment a graph-worthy concept arrives via `promote_keyword`, and the only fix
would be lying about its provenance); **deleting the 344** (destructive, cascades into
`concept_presence`/`concept_edges`/`gaps`, removes 344 keyword families from a shipped view to fix a
different feature — and being a data fix, the next `--promote-all` re-floods); splitting into two tables
(a real migration across four consumers to buy what a nullable column buys; revisit only if the two
vocabularies diverge in *shape*, not just membership).

**Applied + measured on the real corpus ($0, local Ollama, this box — 47 docs / 688 keywords):**
migration added the column (357 rows NULL) → backfill split **13 include / 344 exclude** → rebuild
`--apply --enrich --provider ollama` (the `--apply`-alone footgun avoided). **Graph 357 → 13 nodes**,
1534 → **19 edges**, 40 → **6 communities**, over 22 documents with presence. **Node B: 9 calls, 19/19
edges annotated, 63 stance assertions, 7 contested edges** — and the result reads as a real map
(`dense retrieval —[contrasts with]→ BM25`, `contrastive learning —[uses]→ hard negatives`) where 357
nodes read as noise. Directions: **7 contested / 6 stable / 0 superseded_trend**. **Gaps 302 → 15**
(isolated 3 · single_source 3 · thin_bridge 4 · under_connected 3 · unsourced_claim 2). Both ADR
guarantees verified live: graph vocabulary **13**, keyword families still **357**.
Gate: ruff ✓ · format ✓ · `mypy --strict src` (63) ✓ · bandit 0 HIGH/MED ✓ · **1002 passed** (+14; the
6 `send2trash` failures are pre-existing venv drift, unrelated to this diff — the dep is declared in
`pyproject.toml:84` but absent from `.venv`).

**Found in passing, logged not fixed — `.claude/KNOWN_ISSUES.md` KI-17:** the rescope stranded **10**
stochastic `suggested_concept` gap rows whose concept left the vocabulary. `build_gaps` delete-and-
replaces *deterministic* rows but **status-preserving-upserts** stochastic ones with no reconcile pass,
so they are immortal — `load_graph_view()` serves **27** gaps against 13 nodes while the runner reports
15. Invisible in PR-G2a's index (it resolves by concept), but it breaks **PR-G2b**, where every gap needs
a triage action. Fix belongs with the ADR-017 C1 override sidecar.

**Opens:** **the gap distribution must be re-derived before PR-G2b** — its "strong kinds first" ordering
rests on RG-014's verdict at 26 concepts on the *other* box's 76-doc corpus; at 13 concepts here the
kinds are nearly flat (4/3/3/3/2), so `single_source` is no longer self-evidently the headline. **No
curation UI:** opting a concept in is CLI-only until a follow-up adds the toggle to Manage-keywords (its
natural home — keeps ADR-017 A1 intact, the graph still never writes the vocabulary). The 13 included
concepts contain 4 junk manual entries (`cre`, `dbs`, `ntsr1`, `pddl`, added 2026-07-05) that now form
three single-concept communities and account for 6 of the 15 gaps — worth curating out. `--promote-all`
is now harmless to the graph but still violates `promote_keyword`'s candidate-only contract.
**Staged; nothing committed (cpc §13).**

---
## 2026-07-17 — Concept graph PR-G2a: the view — concept index + gap lens + ego graph + chunk nav (frontend)

**What:** built PR-G2a of `feature-concept-graph.md` — the third top-level view that renders the PR-G1 read
model. A **destination, not a modal** (`mode` union widened `'chat'|'library'` → `+'graph'` in the 4 measured
places + a third rail tab). New `lib/ConceptGraph.svelte`: a searchable **concept index** on the left (label ·
gap badge · doc count) with a **"Gaps only" lens** and an **"Include under-connected"** opt-in; selecting a
concept opens a depth-1 **ego graph** (hand-rolled SVG, no dependency) + a details panel that navigates concept
→ document → the chunks it appears in. New `lib/forceLayout.ts`: pure, seeded (mulberry32 + phyllotaxis init +
Fruchterman–Reingold to convergence, then fit-to-viewBox) — deterministic and epsilon-guarded so no coordinate
can be NaN. `types.ts`/`api.ts` mirror the 7 PR-G1 payloads + 4 client fns (404 → `null`, the normal first
run). `app.css` gains a **12-hue categorical community palette** (both themes) cycled by `community % 12`, plus
`--graph-edge`/`--graph-node-stroke` derived once from `--fg` via `color-mix` (late-bound, tracks both themes).
`Icon.svelte` gains the Lucide `waypoints` glyph. Deep-links only: node → `openDocument()` (Library), "Edit"
→ Manage-keywords (**ADR-017 A1 — the graph never writes the vocabulary**). Staleness banner + empty state
share one **Rebuild** affordance (202 + poll, ADR-017 B1). Read-only; `$0`.

**Why:** ADR-004's north star is gap detection; RG-014 found the strong signal (`single_source`) is
**list-shaped**, so the index — not the graph — is the home, and the graph earns its place as the navigation
surface. Ego-first (B3) bounds the hairball (`Embeddings` touches 80% of the graph) to one neighbourhood.

**Ordering honours the verdict:** `single_source` leads (danger tone); `under_connected` is **off by default**
behind a toggle (it is graph-degree noise at n=26). Gaps badge **nodes**, stance will colour **edges** (B9) →
no collision when Node B lands (PR-G4).

**Verified live ($0/offline, real corpus, via read_page + javascript_tool — the SVG DOM is the only assertable
surface; screenshots time out on this box):** index shows **26 concepts**, the 3 `single_source` true positives
(PHATE/Res2Net/SBERT) lead in danger tone, gap lens = **8** (under_connected hidden), → **10** when opted in.
Res2Net ego = 3 nodes/3 edges, Embeddings (degree 20) ego = **21 nodes** — **no NaN** in any cx/cy/x1/y1, all
in-viewBox, no collapse. **Determinism: identical positions across a re-render.** 3 distinct community fills;
theme flip changes fill + the `color-mix` edge stroke. Zoom clamps **0.4↔3.0**. Res2Net → its 1 document →
"Mentioned in 25 chunks" → **Open in Library** switches mode + opens the doc; **Edit** → Manage-keywords.
375px: **no horizontal overflow** (index + ego). **Gate:** `svelte-check` 0/0 (133 files); `vite build` clean
(157 modules); **still one runtime dep (`marked`)** — the layout is hand-rolled.

**Rejected:** a graph library (cytoscape/d3 — 4× the bundle, an eval-using lib breaks only in the packaged
Tauri CSP, and with zero frontend tests the SVG DOM must stay assertable); a modal (6 hand-rolled scrim dialogs
exist, all capped transient tasks — a graph is a destination); `weight`-as-thickness (range 2.377–2.949, flat);
a provenance legend (one state); contested/superseded colour (renders nothing until PR-G4); animating the
simulation (run off the render path, draw statically — determinism is the safety net); persisting pan (it is
position-specific and resets on re-centre; only zoom, a real preference, persists); a `color-mix` hue wheel for
communities (a rotated hue can land on low-contrast yellow in light theme — a fixed per-theme ramp controls
contrast, and colour is a positional grouping hint so cycling past 12 is harmless).

**Opens:** PR-G2b (gaps as a first-class destination + triage via the ADR-017 C1 override sidecar — `status` is
still the raw row value on the wire); the `unsourced_claim` count stays approximate until the claim-segmenter
heading bug is fixed (surface it, don't present it as precise); PR-G4 (Node B on the RTX box) unblocks the
reserved edge-stance encoding; community palette cycles (not collides) past 12 — fine for now, revisit if a
real corpus shows >12 communities. **PR-G1 is still staged/uncommitted — this builds on it; both await review.**
**Staged; nothing committed (cpc §13).**

---
## 2026-07-17 — Concept graph PR-G1: serve the read model (load_skeleton + load_gaps + 4 routes)

**What:** built PR-G1 of `feature-concept-graph.md` — the backend read model the graph view consumes.
**Staged, NOT committed.** New `src/doc_assistant/concept_graph_view.py` (`GraphView`, `GraphStaleness`,
`load_graph_view`, `load_concept_presence`); `concept_skeleton.load_skeleton()`; `gaps.load_gaps()`; six
payloads in `apps/api/models.py`; four thin routes in `apps/api/main.py`; 16 tests in
`tests/integration/test_concept_graph_api.py`.

**Why the pieces landed where they did.** `load_skeleton` is the **read half of `write_skeleton`**, so it sits
beside it in `concept_skeleton.py`; `load_gaps` likewise mirrors the row writers inside `gaps.py`, because
that module owns the gap domain. The *assembly* (skeleton + gaps + staleness) got its **own** module rather
than joining `library.py`: `library.py` serves documents/keywords, and the graph is a distinct top-level view
— putting it there would have grown an already-large module with an unrelated concern. `apps/` stays a shell:
every route is a pass-through, and the loader/assembly/staleness reasoning all lives in `src/`.

**Two decisions worth the record.** (1) **One wire id space — concept UUIDs everywhere** (node ids, edge
endpoints, gap anchors, community members), with `label` **only** on the node. The spec demanded a choice
because mixing ids and labels across a boundary is *exactly* what made KI-15 match nothing silently; the live
check asserts it (70/70 edges, 14/14 gaps resolve). (2) **Presence is served per-concept, not bulk** — a
**deliberate deviation** from the spec's "graph → skeleton + gaps + presence". Ego-first (B3) renders one
neighbourhood at a time, and bulk-shipping 1781 chunk keys for a 26-node graph is waste that scales badly to
357. `doc_ids` already rides each node, so only chunk-level navigation needs the extra call.

**Distinguishing "never built" from "broken" was the load-bearing detail.** `skeleton.json` is gitignored and
regenerable, so **absent is the NORMAL first run** → `load_skeleton` returns `None` and the route answers
**404 with a rebuild hint**, not a 500 and not a fake empty graph. But a file that *exists and won't parse* is
a corrupt artifact — a different state — so that **raises** (`raise RuntimeError(...) from e`); returning
`None` there would invite a rebuild that masks the real problem.

**Verified live on the real corpus ($0/offline).** `GET /api/concepts/graph` → **26 nodes / 70 edges / 3
communities / 14 gaps**, `graph_version b59a4aa6afa77978`, `stale:false`, every `relation` `null` (Node B
never run). Presence: `Embeddings` → **32 docs / 283 chunk keys**; unknown → `[]`. Rebuild: **202** → poll →
`done`, returning the **identical graph_version** — determinism proven end-to-end through the API, not just in
a unit test. Empty state: skeleton moved aside → **404** → restored → 200. Gates: ruff, ruff format,
`mypy --strict src`, bandit all clean; **full suite 994 passed** (was 977; **+16 new, 0 regressions**).

**Rejected:** (a) *the assembly in `library.py`* — see above. (b) *bulk presence* — see above. (c) *a blocking
rebuild route* — 7.1s on the event loop, when `POST /api/ingest`'s 202+poll is the repo's established shape for
exactly this (ADR-017 B1). (d) *`None` on a corrupt skeleton* — conflates two states. (e) *seeding the FK
referents in one `session_scope`* — these models carry no `relationship()`, so a single flush does not reliably
order the parent insert before the child and the FK trips; the test seeder commits the referent separately.
(f) *"fixing" `apps/api/main.py:734`'s pre-existing E501* — **CI lints `src/` + `tests/` only** (confirmed in
`.pre-commit-config.yaml`: "Scope = src/ + tests/ ONLY, to mirror CI exactly"), so `apps/` is deliberately out
of ruff's scope and that line is neither a gate failure nor mine.

**Two self-inflicted bugs caught before they shipped, both by measuring rather than reasoning:** my first
staleness diff was `sk_ids.symmetric_difference(db_ids) & db_ids`, a convoluted spelling of `db_ids - sk_ids`
— simplified. And the `Write` tool captured a literal `</content>` tag into four files (the module, the ADR,
the spec, the test); caught by a `SyntaxError` on the first import, stripped from all four.

**Opens:** PR-G2a (the view) is next and needs the **palette** decision (3 non-semantic hues, 3 communities —
luck, not headroom). The **`GET /api/concepts/graph` payload can't paginate** — fine at 44 KB / 26 nodes,
~600 KB at 357; state a position before the vocabulary grows. **RG-014 stays open** (the gap payload is ~50%
precise; the spec's "don't lead with `under_connected`" is the mitigation, and PR-G2a must honour it). ADR-017
C1's **gap-triage override sidecar is NOT built** — that is PR-G2b, and `GapPayload.status` is currently the
raw row value, not the effective one.

---
## 2026-07-17 — Wrote ADR-017 (concept-graph UI boundaries) + docs/specs/feature-concept-graph.md (docs-only)

**What:** authored the two artifacts the concept-graph grill routed to. **`docs/decisions/ADR-017-concept-graph-
ui-boundaries.md`** (new, accepted) and **`docs/specs/feature-concept-graph.md`** (new, design-locked, not
built). Docs-only; no code. `docs_check --strict` 0/0.

**Why an ADR at all:** the graph is the first UI over the curated vocabulary that is **not** curation, and it
crosses three boundaries that each had no owner — places where a plausible design quietly breaks a shipped
guarantee. ADR-017 decides **only** the boundaries; the spec owns the contract (ADR-019's "if an ADR owns the
*why*, the spec is the *how* and must not re-litigate it").

**The three decisions.** **A1 — read-only for the vocabulary + deep-link to Manage-keywords:** you edit the
*source* and regenerate, never the derived artifact; the "a keyword belongs to at most one family" invariant
keeps **one** home while **PR-2.5 is still repairing it** (rename → duplicate `Concept` labels → corrupts
`promote_keyword` repo-wide), and a second writer onto known-broken paths isn't worth a convenience. **B1 —
in-app Rebuild (202 + poll), CLI runner stays canonical:** 7.1 s is a button, and `POST /api/ingest` + a status
poll is the established pattern for this repo's *largest* derived build. The reasoning that mattered: **the
Enrichment-Layer Pattern constrains *what* derived data is (regenerable, sidecar, never mutates source) — not
*who is allowed to press go*.** **C1 — gap triage as a user-override sidecar keyed on `(concept_id, kind)`:** a
dismissal is a **user judgment**, not derived data, so it must not live in a table that is deleted and rebuilt
from the skeleton; `GapRow.status` becomes **effective = override ?? "surfaced"**, which is exactly ADR-013 A2's
shape (auto on the record, override in a sidecar). This keeps ADR-004's regenerable guarantee intact.

**C1 exists because RG-014 disproved the grill's own premise.** B14 had resolved "dismiss/promote from the
view" partly on *"`build_gaps` deliberately persists status across rebuilds"*. It does not — for deterministic
gaps (`gaps.py:257` deletes and replaces them; only the stochastic path is status-preserving), and **all 14
live gaps are deterministic**. The ADR records the corrected reasoning rather than the comfortable one.

**The spec's load-bearing constraint is the RG-014 verdict, not the design.** The gap payload is **~50%
precise**, and — the finding that shapes every screen — **the strong kinds are LIST-shaped (`single_source`,
`unsourced_claim`) while the weak kinds are GRAPH-shaped (`under_connected`, `thin_bridge`)**. So the spec
**leads with the index + gap list, defaults `under_connected` OFF**, and positions the graph as the *navigation
and context* surface rather than the dashboard. A spec that led with the pretty part would have shipped the
noise: `under_connected` is the **largest** kind and flags `Tractography` (10 docs) and `Motor control` (13
docs) — among the best-sourced concepts — as gaps.

**Rejected (ADR):** *graph writes in place* — every write instantly stales the view you're reading, so it lies
until rebuilt; *the graph replacing Manage-keywords* — a force layout is a poor bulk-alias editor and it
discards a view mid-hardening; *CLI-only rebuild* — breaks the acquire loop by ejecting the user to a terminal
mid-research; *auto-rebuild on staleness* — spends 7.1 s unasked **and destroys the seeded-determinism property
that is the feature's only verification surface**; *making the deterministic write path status-preserving* —
`gaps` would become a hybrid of derived + user data, forfeiting the regenerable property ADR-004 relies on, and
it needs a reconcile rule for gaps that stop firing; *no triage for deterministic gaps* — 0 stochastic gaps
exist, so that means no triage at all and a permanent nag.

**Rejected (spec):** *the cpc `SPEC-000` executor-brief template* — the house shape for feature contracts is
`feature-*.md` (Status/Owner → decision → grounding → carve → parked → ledger), per `feature-tag-families.md`;
SPEC-NNN is for delegated sprint briefs. *Leading with the graph* — see the verdict. *Shipping
contested/superseded encoding* — dead until PR-G4. *`weight` as edge thickness* (2.377–2.949, flat) and *a
provenance legend* (one state).

**Opens:** the spec is **design-locked, not built** — PR-G1 (serve; write the missing `load_skeleton()`) is
next. **RG-014 stays open** until the spec's narrowed claim is exercised and the two live defects land: the
**claim-segmenter heading bug** (12 of 61 `unsupported` claims are markdown headings — and they feed the
failure-tag gates driving the self-improvement loop) and the **`under_connected` corpus-vs-vocabulary guard**.
PR-G4 (Node B on the RTX box, $0) still gates every epistemic encoding. B13 (the acquire loop) needs its own
ADR when picked up.

---
## 2026-07-17 — RG-014: ran build_gaps --apply on a fresh skeleton; VERDICT ~50% precision. B1 narrows; 3 defects found

**What:** closed out RG-014's procedure at the user's request — *"run build_gaps --apply and check the 14
findings are real"*. **Data-only writes (both gitignored): `data/skeleton/skeleton.json` + the `concept_edges`
/ `concept_presence` / `gaps` sidecars. No source changed.** All $0/offline.

**Procedure.** Verified first that **0 edges carried Node-B annotation** (so a rebuild loses nothing) →
`build_concept_skeleton --apply` → fresh **`b59a4aa6afa77978`** replacing the stale `055312c8c15a7e69` →
`build_gaps` dry on the fresh skeleton: **still 14** (*the findings are stable across the rebuild — they were
not artifacts of staleness*) → `--apply`: **14 rows persisted** → re-run: **14 rows, no duplication —
idempotence confirmed.**

**VERDICT: 8 of 14 defensible, 6 of 14 noise/duplicate/misleading. B1 survives but NARROWS.**
- ✅ **`single_source` (3) — TRUE POSITIVE and the whole product thesis.** `Res2Net` appears **only** in the
  Res2Net paper; `SBERT` **only** in the Sentence-BERT paper; `PHATE` **only** in one neurodevelopment paper.
  Each is a method known *solely from its originating source* — no independent evaluation or replication in the
  corpus. This is exactly the user's *"technically, having a single source is not good"*, and it is directly
  actionable via B13 (acquire corroboration).
- ⚠️ **`unsourced_claim` (4) — real signal, ~33% contaminated input.** Aggregation is sound (`RAG` 15
  unsupported claims, `BM25` 5) and sampled prose is genuinely uncited. **But 20 of the 61 underlying
  `unsupported` claims are not prose claims at all — 12 are MARKDOWN HEADINGS** (`"# Dense Passage Retrieval
  (DPR) and Its Advantages Over BM25"`) **+ 8 fragments.** A heading can never cite, so it is *structurally*
  always unsupported. (43/61 also predate the 2026-07-14 parser fix.)
- ❌ **`under_connected` (5) — mostly noise at n=26, and it's the LARGEST kind.** It measures **graph degree**,
  which at a 26-concept vocabulary is dominated by **vocabulary sparsity, not corpus coverage**. **`Tractography`
  (10 docs, degree 2)** and **`Motor control` (13 docs, degree 2)** are among the corpus's **best-sourced**
  concepts and are flagged as gaps. **It conflates "my corpus is thin on X" with "my vocabulary is too small
  for X to co-occur with anything."** 2 of 5 duplicate `single_source`; only `MedSAM` is defensible.
- ⚠️ **`thin_bridge` (2) — redundant + half-misleading.** Both derive from **one edge** (`MedSAM` ↔
  `Embeddings`) and it flags **both endpoints**, so **`Embeddings` — the most-connected node in the graph
  (degree 20/25, 32 docs) — is reported as a "thin bridge" gap.**

**The finding that most affects the spec: the STRONG kinds are LIST-shaped (`single_source`,
`unsourced_claim`); the WEAK kinds are GRAPH-shaped (`under_connected`, `thin_bridge`).** B1 does **not**
reverse — the corroboration job stands on the two strong kinds, and the graph's *navigation* value (B7,
concept→doc→chunk, 1781/1781 verified) is independent — **but the graph is not the primary renderer for the gap
payload; a list is. The spec must not lead with `under_connected`.**

**⚠️ B14 REOPENS — the grill's reasoning was wrong.** B14 resolved "dismiss/promote from the view" partly
because *"`build_gaps` deliberately persists status across rebuilds"*. **It does not — for deterministic gaps.**
`gaps.py:257`: `session.execute(delete(GapRow).where(GapRow.determinism == "deterministic"))` — deterministic
rows are **delete-and-replace**; only `_write_stochastic_gap_rows` (`:273`) does the *"status-preserving"*
upsert that "never deletes". **Verified live: a `dismissed` deterministic gap reset to `surfaced` on the next
run.** **All 14 of today's gaps are deterministic** (stochastic = 0) → **dismissing any of them is futile, and
rebuild is part of the acquire loop.** I over-read the docstring's "stochastic rows persist their status" as
"rows persist their status". **Lesson: when a docstring qualifies a noun, the qualifier is load-bearing.**

**Rejected:** (a) *running `build_gaps --apply` against the stale skeleton* (the literal request) — it would
stamp findings with a dead `graph_version` and answer a question about an out-of-date graph; rebuilt first
(safe: writes are gitignored, idempotent, and 0 annotated edges existed). (b) *closing RG-014* — the run
answered "do they persist / are they real" (yes / ~half), but the spec hasn't absorbed the verdict and two live
defects remain. (c) *reversing B1* — `single_source` + `unsourced_claim` carry the job; the weak kinds are a
detector-tuning problem, not a premise failure. (d) *treating `under_connected` as permanently broken* — it is
noise **at n=26**; it should improve as the vocabulary grows (`--promote-all` → ~86).

**Opens:** **(a) BUG — the claim segmenter counts markdown headings as claims** → 12 permanent false
`unsupported` markers → false gaps **and they feed the failure-tag gates that drive the self-improvement loop**
(new checklist row). **(b) deterministic gap triage needs a durable store** keyed on `(concept_id, kind)`,
mirroring the stochastic upsert — **decide in ADR-017.** **(c) `under_connected` needs a corpus-vs-vocabulary
guard** (gate on doc-count, or defer the kind until the vocabulary is larger). Live state now: skeleton
`b59a4aa6afa77978`, `gaps` = **14 rows**, all `surfaced`, all deterministic.

---
## 2026-07-17 — GRILLED the concept graph (grill-me): 12 branches, 11 resolved / 1 parked; the root question was overturned by the repo (docs-only)

**What:** ran `grill-me` on the concept-graph view before its spec. **12 branches: 11 RESOLVED, 1 PARKED, 0
open.** Full ledger in the session baton; the durable half is in `docs/ui-checklist.md` §3. **Docs-only; no
code; nothing run but free read-only dry-runs.** **Routed, not authored** (the grill doesn't write artifacts):
a **new ADR-017 `concept-graph-ui-boundaries`** (B5/B6 read-only-for-the-vocabulary + deep-link; B8 an API
caller of an enrichment runner; B14 gap-status writes) cross-referencing ADR-015 (which reserved this track) +
ADR-004 (which owns gaps); a **new `feature-concept-graph` spec**; **B13 → the External literature discovery
row**; **RG-014 → `.claude/RIGOR_TODO.md`**.

**Why it mattered: the grill overturned its own root question by reading the repo instead of asking the user.**
I opened B1 ("what job does this do?") expecting to argue against a pretty-but-useless Obsidian clone.
**ADR-004 answered first:** *"Phase 7's stated purpose is **gap detection** — and the project's north-star reason
for it is to surface what the user (and the LLM) cannot see: concepts the corpus under-supports, claims it
cannot source, and directions for exploration the user did not think to look."* And the gap layer is **BUILT +
Ollama-validated with a recorded baseline** (Tier-1 + Tier-2a floor SPRINT-002/G2; ceiling SPRINT-005/G5) —
`gaps.py`, `gap_suggest.py`, `scripts/build_gaps.py` — **with 0 rows and ZERO UI**. A **free dry-run found 14
real gaps** on the live corpus (`under_connected` 5 · `unsourced_claim` 4 · `single_source` 3 · `thin_bridge`
2). The user's own framing independently matched the detector — *"technically, having a single source is not
good"* — which is exactly what `single_source` measures. **So the graph has a payload today; it is not
decoration.** **Lesson: read the archived spec before asking the user what a feature is for.**

**Measurements that decided branches (all free/offline):** **B3** — the hairball is real **today**: 22%
density, mean degree 5.4, but **`Embeddings` has degree 20/25 = touches 80% of the graph** (depth-1 ego = 81%
of all nodes, on 32/76 docs) while the **median ego is 6 nodes** → **ego-first, depth-1**. *(Side finding: a
degree-20 node on 32 docs means `Embeddings` is too generic to be a good concept — the graph reveals
vocabulary quality for free.)* **B7** — **1781/1781 (100%)** of `concept_presence.chunk_keys` resolve against
the live index (ADR-4 key `{document_id}:p{parent_index}`) → concept→doc→chunk navigation is real. *(My first
attempt reported 0/1781 — I'd built `doc:idx` instead of `doc:pN`; caught it rather than shipping it.)* **B8**
— **rebuild = 7.1 s, zero-LLM, deterministic** → a button, not a batch job; and it **confirmed the artifact is
stale**: `graph_version` `055312c8c15a7e69` → **`b59a4aa6afa77978`**, with **`doc_years` now present** (so
`superseded_trend` becomes possible once stance lands). **B9** — **every `Gap` is anchored to `concept_id`**
(even `thin_bridge`, whose endpoints live in `evidence.fact_ids`) while stance is an **edge** property ⇒ **gaps
encode on nodes, stance on edges, no palette collision** — and B3 (≤21 rendered nodes) defused the 3-hue
community gap entirely. **B14** — `GapStatus = surfaced|promoted|dismissed` **already exists** and `build_gaps`
**deliberately persists status across rebuilds**: the sidecar was *designed* for triage.

**Rejected:** (a) *a global map* (B3) — hub-dominated at n=26 already, and `--promote-all` is one command from
~86 nodes. (b) *top-N-by-degree capping* — it hides exactly the low-degree nodes the gap layer flags as
`under_connected`, i.e. it hides the findings. (c) *graph writes to the vocabulary* (B5) — you edit the source
and regenerate, never the derived artifact; and a second writer onto rows whose invariants **PR-2.5 is
currently repairing** (D1 rename → duplicate labels → corrupts `promote_keyword`) is reckless. (d) *the graph
replacing Manage-keywords* — a force layout is a poor bulk-alias editor. (e) *auto-rebuild on staleness* — it
spends 7.1 s unasked **and breaks the seeded-determinism verification story, which is the only test surface**.
(f) *CLI-only rebuild* — it breaks the acquire loop by dumping the user into a terminal mid-research; the app
**already** triggers its biggest derived build via `POST /api/ingest` **202 + status-poll**, so the precedent
exists and the CLI runner stays canonical (Enrichment-Layer intact). (g) *read-only gaps* — a view that can't
say "I know, that's fine" becomes a permanent nag.

**Opens / parked.** **B13 (parked, needs its own ADR):** the **gap → acquisition loop** — *"download and find
more information to complete the graph… we will need a provider list, and a quality list"*. It closes ADR-004's
loop and merges with the **External literature discovery** row; **transport is already spiked** (stdlib urllib →
Crossref, 25/25). **CONSTRAINT ON THE SPEC: model a gap as an object with an ACTION SLOT** (`GapStatus.promoted`
is that slot) so it attaches without rework. **RG-014 (blocks-ship): the 14 gaps are a DRY-RUN claim —
`build_gaps --apply` has NEVER been run and nobody has read the findings.** Run it on a *fresh* skeleton,
confirm the sidecar persists + status survives a re-run, and judge signal-vs-noise **before** the spec — **B1
reverses if it's noise.** (Note **RG-001** is still `open`/`blocks-ship` and its close instructions name
`build_concept_graph --apply`, **a command that no longer exists** — the module was retired in KI-7/SPRINT-001;
that entry needs re-pointing at the skeleton.) Node B on the RTX box (PR-G4) still gates every epistemic
encoding.

---
## 2026-07-17 — Planned the concept graph (PR-G1/G2/G4); RAG blast-radius + chat-mode boundary measured; 2 self-corrections (docs-only)

**What:** planned the **concept-graph view** in depth (3 explorers + a design pass, all grounded on live code +
the real corpus) and answered two user challenges with measurements. **Docs-only; no code; nothing run but free
read-only dry-runs.** Plan detail lives in a **local, uncommitted** plan file; everything load-bearing is
duplicated into `docs/ui-checklist.md` §3.

**Two self-corrections to my own uncommitted entry below — both were load-bearing and both are now fixed in
place** (the entry never landed, so correcting beats shipping a known-false claim + an erratum): **(1) Node B is
BUILT, not "unbuilt", and $0, not "paid"** — see the corrected text below; **(2) "citations 3918" is the WRONG
table for citation-highlighting** — those are bibliography refs with **0 resolved** (`target_document_id` → 0);
answer-citations are `answer_claims` 168. **Root cause of (1): I grepped `scripts/` for a FILE named like a
stance runner and concluded the feature didn't exist. The runner is a FLAG (`--enrich`) on the existing Node-A
script. Lesson: absence of a filename is not absence of a feature — grep the call graph, not the directory.**

**Concept graph — decisions (user).** (1) **Hand-rolled SVG + a seeded force layout, NO dependency.** The
reasons compound: 26 nodes (500 worst case) needs no library; the Tauri CSP is `default-src 'self'` with **no
`unsafe-eval`**, so an eval-using lib breaks **only in the packaged build** (dev-mode Vite won't catch it); the
frontend is a deliberate **1-runtime-dep** artifact (`marked`) with `dist` at **101 KB** (cytoscape ~400 KB =
4× the whole bundle) and a stated no-CDN/vendored ethos; and decisively — **zero frontend tests exist and
screenshots time out on this box, so SVG's DOM is the only verification surface**; canvas/WebGL would render
the feature literally unassertable. (2) **Association-only** — measured: **all 26 nodes are `unique`/`stable`,
`contested_edges()` → `[]`**, so contested colouring is dead UI; reserve `--danger`/`--warn-fg` for stance and
ship nothing that renders empty. (3) **One vocabulary — a family IS a concept**; don't filter by `source`.
**The real defect isn't the shared rows (ADR-015 chose that deliberately) — it's that the skeleton is a build
artifact, so the graph silently lags user edits.** Surface staleness + a Rebuild action instead (inform, don't
block). **Resolved:** full view, **not a modal** (there is no reusable modal shell — it's hand-rolled in **6**
components, all `min(84vh,620px)`-capped *transient tasks*; a graph is a *destination*); the shell change is
small (the `'chat'|'library'` union appears in exactly **4** places).

**Why the graph is ready when the rest of the phase isn't:** `concepts` 26 · `concept_edges` **70** ·
`concept_presence` 222 · 3 communities · `skeleton.json` carries a **complete render model with layout signal
precomputed** (`community`→colour, `degree`→radius, `doc_ids`→click-through). ADR-015 **reserved this track by
name** and PR-1/PR-2 made `Concept` a first-class UI citizen. It is greenfield — `concept_graph.py` was retired
(KI-7) leaving no dead code.

**Rejected:** (a) *d3-force/cytoscape/sigma* — see (1); a dep would also want its own ADR. (b) *canvas/WebGL* —
unassertable on this box, which is the binding constraint, not perf. (c) *filtering the graph by
`source='manual'`* — ADR-015 earmarked the column, but promotion is how the vocabulary is *meant* to grow, so
filtering it out is backwards. (d) *shipping contested/superseded colouring now* — 0 signal. (e) *leaning on
`weight` for edge thickness* — range is **2.377–2.949**, nearly flat. (f) *a provenance legend* — all 70 edges
are `{cooccurrence, similarity}` and **0 citations resolve**, so it would have exactly one state. (g)
*persisting anything against a community id* — they're **positional, not identity** (add a concept → they
renumber). (h) *animating the force sim* — run seeded to convergence off the render path; **determinism makes
the assertions non-flaky**, which is the point.

**User challenge 1 — "I don't see the problem with changing top-k; the issue is the embedding." MEASURED: they
are right, and it exposed a real gap.** `TOP_K` is **already** per-session, non-persistent **and bounded**
(`ge=1, le=CANDIDATE_K` → [1,20], **422, never a silent clamp**) — there was never a problem. The embedding *is*
the only catastrophic knob, but **not by the assumed mechanism**: collections are **namespaced per model**, so
switching reads an **empty** collection → `warning("empty_index")` → "the sources don't contain the answer". The
real footgun is narrower: `_LEGACY_COLLECTION="langchain"` is bound to **whatever `DEFAULT_MODEL` is** and both
registered models are **768-dim**, so changing `DEFAULT_MODEL` would **silently inherit bge-base's vectors** —
Chroma accepts the dimension and nothing detects it. **THE GAP: `EMBEDDING_MODEL` is in NEITHER ADR-010's split
NOR CONTEXT's locked table — the most dangerous knob is documented as "swappable" and governed by nothing,
while harmless `TOP_K` is locked.** Why: **ADR-010 sorts by *cost to change*; the user sorts by *blast radius*.
Those axes disagree, and blast-radius is unmapped.** Per-doc ingestion tuning: the **vector space genuinely
survives** (same model ⇒ same space; reranker scores pairs independently) — but **BM25's `avgdl` is
corpus-global**, so mixed chunk sizes **systematically penalize the longer-chunked doc on the sparse arm**;
health thresholds are absolute; **`TOP_K` counts parents, not tokens** (a per-doc `PARENT_CHUNK_SIZE=8000`
quadruples context+cost, **no guard**); and the splitters are **import-time singletons**, so runtime config
mutation does **nothing** on the hot path. **Contained, not free.** Captured as its own row.

**User challenge 2 — "chat modes = swapping some system prompts." MEASURED: good idea, one hard boundary.**
**`ANSWER_PROMPT` is not a prompt — it's the wire format `synthesis.py` parses** (`_CITATION_RE` is commented as
matching markers *"produced by ANSWER_PROMPT"*). **Proof the coupling already broke:** a 2026-07-14 fix records
the model emitting `[Source 2]`/`[2, 4]`, the parser dropping them, and **claims that DID cite reading as
uncited**. A user edit is that failure with the safety off: every claim → `MARKER_UNSUPPORTED` → **false
`unsupported` rows persisted to the adjudication log** → the failure-tag gates count them → **the
self-improvement loop learns from corrupted data**, silently. **Carve: the citing block is NOT user-editable;
the persona/task framing above it is.** Also: `CitationAudit.clean` is `True` for an answer with **zero**
citations (use `n_uncited_sentences`), and `chat_controller.py:516` caches `_answer_template_hash` **at
construction** — swappable prompts would make **every provenance record lie** unless it moves per-turn.

**Opens:** the graph's **palette gap** — only **3 non-semantic hues** for exactly **3 communities** is luck, not
headroom (`--promote-all` → ~86 nodes; KI-15 documents a real **357**-concept corpus) → **design for 300–500**.
`skeleton.json` is **gitignored + stale** (predates G3 → no `doc_years`; a rebuild changes `graph_version` —
never hard-code it) and a fresh clone has **none** (the empty state is the normal first-run path).
**`data/graph/graph.json` is a stale EMPTY decoy** (retired `concept_graph.py` residue) — reading it renders an
empty graph that looks like a layout bug. **No `load_skeleton()` and zero graph API routes** — the read model is
all to build; node ids are **UUIDs with labels only on nodes**, the exact id-space mismatch that caused KI-15.
Node B on the RTX box (PR-G4) unblocks **every** epistemic UI at $0.

---
## 2026-07-17 — Captured the UI phase's remaining features (7 rows); DIAGNOSED epistemics as blocked on Node B never having been RUN (docs-only)

**What:** the user named the rest of this UI phase — concept graph (Obsidian-like) + epistemics, ingestion
workflow, settings screen, user-tunable RAG pipeline, chunk screen, citation/source highlighting, figures +
tables. Measured the **data reality** behind each against the live corpus, then captured **6 new
`docs/ui-checklist.md` §3 rows** + **rewrote the epistemics row** (its guidance was wrong). **Docs-only; no
code; nothing run against the corpus except free read-only dry-runs.**

**The headline: three of these are not UI work — they're data problems, and they block differently.**

**1. Epistemics — the checklist's own fix was WRONG, and I nearly repeated it.** The row said *"the enrichment
run hasn't been applied here. First step is running the epistemics build ($0/local) and confirming the sidecar
populates."* I offered the user exactly that; they said **investigate first**. They were right — **running it
writes 0 rows.** Free read-only dry-run (`python -m scripts.compute_epistemics`): `Concept nodes weighted: 26`
· **`Contested nodes: 0`** · **`Superseded-trend nodes: 0`** · `Chunks with a claim: **1835**` · **`Chunks
marked: 0`**. **The pipeline is healthy — KI-15's fix works** (1835 chunks project fine); the **input signal**
is absent. `node_weights_for_epistemics` aggregates edges' `stance_by_doc` and documents that *"a stance-less
node → unique/stable"*; **all 70 `concept_edges` rows carry `stance_json = None`** (verified) → no node is ever
`contested` → no chunk is ever marked. Stance comes from **Node B**, the LLM relation/stance pass.
**⚠ SELF-CORRECTION (same session, caught before this entry was ever committed): I first wrote here that Node B
is "explicitly deferred and UNBUILT (no stance/relation runner exists in `scripts/`)" and that epistemics is
"blocked on an unbuilt, PAID LLM feature". BOTH CLAIMS ARE FALSE.** Node B is **code-complete and committed** —
`src/doc_assistant/concept_skeleton_enrich.py` (pure core; idempotent, re-deriving each edge's annotation from
scratch; **never creates a node or edge** — Node A owns those) — and it is **runnable today** via **`python -m
scripts.build_concept_skeleton --enrich`** (`scripts/build_concept_skeleton.py:150` `if args.enrich: return
_run_node_b(...)`, report formatter at `:59`). **It has simply never been run.** **And it is $0, not paid:**
`CONCEPT_SKELETON_LLM_PROVIDER` defaults to **local Ollama** (`llama3.1:8b`, `config.py:445-446`) — *not*
`LLM_PROVIDER` — with `llm.assert_provider_intent` as the KI-4 credit-leak guard; one call **per document**.
**The only real blocker: Ollama isn't on this dev box** (verified on the separate RTX machine) → **running Node
B there is the single prerequisite for every epistemic UI.** *How the error happened: I grepped `scripts/` for a
**file** named like a stance runner and concluded the feature didn't exist. The runner is a **flag** on the
existing Node-A script.* **Lesson: absence of a filename is not absence of a feature — grep the call graph, not
the directory listing.** (Also latent: `superseded_trend` can't fire even *with* stance unless the skeleton
carries publication years — the on-disk artifact predates G3 (2026-07-08), so its `meta` has no `doc_years`; a
rebuild adds them (66/76 docs have a year) **and changes `graph_version`** — never hard-code
`055312c8c15a7e69`.) **Lesson that survives the correction: a builder existing and being free does not mean
running it produces anything. Dry-run first.**

**2. Figures — blocked, and expensively.** `figures` = **0 rows**, and **0 chunks carry `chunk_type` in either
Chroma index** (`data/chroma` 11 965; `data/chroma_pc` **30 882** = the live one). The `chunk_type='figure'`
ingest path (`ingest/__init__.py:220-271`) has **never produced a chunk here**. Unblocking needs
`describe_figures`, *by its own docstring* **"the project's only paid, API-only enrichment"** (VLM, gated by
`MAX_VLM_CALLS_PER_DOC`) → a deliberate cost decision under the repo's discipline, not a build to just run.

**3. Tables — needs diagnosis, not a UI plan.** Extraction code **exists** (`ingest/tables.py`,
`scripts/extract_tables*.py`, `eval_marker_tables.py`) — I initially and wrongly said it didn't — but **no
`tables` table exists and no table chunk is indexed** (`document_parts` is 0 too). Where were they meant to
land, and why didn't they? Answer that before planning any "surface the tables" UI.

**Contrast — what IS ready.** **The concept graph is the phase's biggest unbuilt feature and nothing blocks
it:** `concepts` **26** · `concept_aliases` 17 · `concept_edges` **70** · `concept_presence` **222** ·
`doc_similarities` **760** · `skeleton.json` present. **ADR-015 explicitly reserved this track**, and PR-1/PR-2
just made `Concept` a first-class UI citizen — the read model, write path and vocabulary all exist, and Node A
is zero-LLM. **⚠ Second self-correction: I first cited "citations 3918" as citation-highlighting's data — WRONG table.**
Those 3918 are **bibliography references** (paper→paper) and **0 are resolved to an in-corpus document**
(`target_document_id IS NOT NULL` → 0, verified) — that's the *citation-graph* feature. **Answer**-citation
highlighting rests on `answer_claims` **168** / `answer_records` 26 + per-turn `result.sources`, and inherits
`synthesis.py`'s `[n]` parser (`_CITATION_RE`, `:23-25`) — whose `:30-38` records a 2026-07-14 fix where the
model emitted `[Source 2]`/`[2, 4]` and **claims that DID cite read as uncited**. **Ingestion** has its model + most of its read surface already (`source_files` 77,
`ingestion_events` 76, derived status, the `excluded` toggle).

**4. "User-tunable RAG pipeline" is a governance request wearing a UI costume.** ADR-010 **considered and
rejected persistent editable settings "on governance grounds"** — non-persistence **is** the wall that keeps a
restart returning to the eval-gated baseline, and CONTEXT's non-negotiable says a locked setting changes **only
via an eval-harness experiment**: *"a sandbox override is fine; changing a default is not a UI PR."* Captured
with the collision stated, **grill + ADR before any build** (user's call), so nobody ships it as "just a
settings screen" and quietly invalidates every baseline.

**Rejected:** (a) *running `compute_epistemics --apply` because it's free* — it writes 0 rows; free ≠ useful.
(b) *treating "epistemics" and "figures" as one blocked row* — they block on **different** things (an unbuilt
LLM pass vs a paid VLM run) and must be decided separately. (c) *planning the figures/tables UI now* — the data
question is unanswered for both halves. (d) *folding "improve the settings screen" and "user-tunable RAG" into
one row* — one is layout, the other reopens an ADR; bundling them is how the governance change would sneak in.
(e) *treating the concept graph as part of tag families* — ADR-015 deliberately separated them.

**Opens:** Node B (the LLM stance pass) is the single prerequisite for **all** epistemic UI — and it's paid, so
it lands under "prove on Ollama first" (KI-4). The two Chroma dirs (11 965 vs 30 882) should be reconciled or
documented before anything counts chunks. `folders`/`tags`/`document_tags`/`gaps`/`document_parts` are all **0
rows** — the Library redesign's Phase B (folders) has no data either. The concept graph must **not** imply
epistemic stance it doesn't have: its 70 edges are association-only.

---
## 2026-07-17 — Planned the 3 remaining UI features into 8 PRs; corrected 3 false checklist claims (docs-only)

**What:** explored + planned the other three UI features the user picked — **evidence-only chat mode**,
**missing-source + library-only delete**, **extended metadata + web autocomplete** — against the live code and
the real 76-doc corpus. Carved into **8 PRs**, sequenced **after** tag-families PR-2.5/2.6/2.7. **Docs-only; no
code changed; coding deferred to a later session by the user.** Full per-PR detail (files, seams, decisions,
tests, verification, risks) is in a **local, uncommitted** plan file; the load-bearing measurements and every
corrected claim are duplicated into `docs/ui-checklist.md` §3 + this entry so nothing is lost with it.

**Why it needed measuring: three claims in our own docs were FALSE.**
1. **"The web-fetch is the first outbound network call beyond the LLM APIs"** (ui-checklist) — **false.**
   `sources_manifest.py:278-285` `_http_get` already does `urllib.request.urlopen` (scheme guard + UA +
   timeout + `# nosec B310` + checksum verify); `scripts/download_corpus.py:72-73` too. The defensible claim
   is *"first outbound call from the API **serving path**"* (`apps/api/` has zero such imports). The
   correction **helps**: `_http_get` is a ready-made precedent to copy, not a novelty to argue against.
2. **"`article_type` is already parsed and thrown away"** (my own claim, this session) — **false, 0/76
   measured.** `_is_skippable_heading` (`metadata_extractor.py:125-138`) returns a **bool skip predicate**,
   and `_SKIP_HEADINGS` mixes document types ("research article") with **section names** ("abstract",
   "methods", "main") — it never fires as a taxonomy on this corpus. Reasoning from code *shape* was wrong;
   the measurement is why we caught it.
3. **"Evidence-only: ~90% exists, in Settings"** — broadly true, details wrong. `synthesis_mode` is already a
   **request-scoped per-turn override** (not a global setting), and it renders as a **concatenated markdown
   string** with an emoji + em-dash prefix (`chat_controller.py:1055-1058`) that the de-tell pass missed
   because it lives in `src/`. `result.mode` is **live on the wire and dead on the frontend** (zero
   consumers) — a free field.

**Measurements that drove the carve (live, this box):** `doi` **25/76** and 100% invisible in `apps/` (2 of the
25 are `type: component` — a figure DOI, wrong for the paper) · `notes` **0/76 with no writer** → dropped from
the DOI PR · local-text yield for the new fields is hopeless (journal 5/76 · url 1/76 · article_type **0/76**)
while **Crossref covers 21/25/25** → Crossref, not extraction · **0/76 sources missing** → the badge has
nothing to show live (synthesize a case) · existence check ×76 = **1.0 ms** → derive-live is sound ·
`USE_MULTI_QUERY` default **false** → the *rewrite* is the real $0 leak, not multi-query.

**Two findings that changed a decision, not just a detail.** (a) **Evidence-only is a better idea than it
looked:** human mode writes `"(human mode: evidence only)"` into history (`:1053`), so in a pure-human chat the
rewrite LLM reads a **zero-information stub** — it is *structurally incapable* of resolving a pronoun, i.e. it
costs money to accomplish nothing. And `_human_result` **already hardcodes `UsageView(0, 0, …)`** (`:1063`)
while the rewrite really spends tokens — **the turn already lies; forcing makes the lie true.** The honest
residual is *mixed*-mode chats. (b) **`is_archived` is a trap:** ~12 read paths filter it and nothing sets it,
but **the retrieval pipeline has no `is_archived` filter and Chroma chunks carry only `doc_hash`** — an
"archived" doc would vanish from the Library while its chunks kept being retrieved and cited. Archiving
without removing chunks is incoherent; removing them kills the reversibility that was the only reason to
archive.

**Transport spike (real outbound calls to the public Crossref API, disclosed to the user).** stdlib `urllib`
reaches `api.crossref.org` from this proxy box **with and without** `truststore.inject_into_ssl()` (~0.7–0.8 s;
25/25 DOIs, 0 failures) → **transport decided: stdlib urllib behind a named seam, not httpx.** Recorded as a
**KI-10 addendum** — the failure is **httpx-specific** (it pins certifi and ignores both the process-global
patch and `SSL_CERT_FILE`), which is why the KI-10 fix had to be branch B. **⚠ Scoped honestly: dev
interpreter (`sys.frozen is False`), one box/day/proxy state — it does NOT prove the frozen build**, which is
KI-10's actual subject.

**Rejected:** (a) *`is_archived` for library-only delete* — the retrieval-filter evidence above. (b) *reusing
`scan_sources` for the badge* — it **WRITES** (upserts rows, refreshes `last_seen`); a read-only list endpoint
must not mutate the registry. (c) *a 4th context-menu item for library-only delete* — `openMenu` hardcodes
geometry for 3 items (`LibraryGrid.svelte:68-69`) and two adjacent red Delete items on the most destructive
path is a footgun; a checkbox in the existing dialog (default = today's behaviour) is the gate already. (d)
*httpx for Crossref* — re-solves KI-10 for a second client; `os_trust_http_client()` returns `None` when not
frozen and is anthropic-typed (a pattern, not a component). (e) *the `cpc: allow-live-api` pragma* — the gate
tripping on `urllib.request` in a test file is **correct pressure**: the call must sit behind a named
monkeypatchable seam (precedent: `test_sync_sources.py`). (f) *a counter-based zero-LLM test* —
`expand_query` (`pipeline.py:302`) takes **no counter**, so it would silently miss the multi-query leak; spy
the pipeline instead. (g) *shipping the metadata migration before the ADR* — Crossref coverage is what decides
which columns earn one; PR-B first would be a migration written against a guess. (h) *`notes` in
`DocumentMeta`* — it has no auto default, so `_dedup_override` would mark it "customized" and corrupt the
`.editmark` dot's meaning.

**Opens:** `expand_query` takes no counter (real bug, its own PR) · **no network guard in the test suite** (the
`api` marker is declared but unused; **no `conftest.py` exists anywhere**) → a missed monkeypatch would
silently hit the network in CI · the `year_override` blanking trap (`library.py:244` — no way to override to
empty) that every new nullable field inherits · the frozen-build urllib check above · human-mode resume stores
a stub answer, not the evidence.

---
## 2026-07-17 — Manage view at scale scoped (PR-2.7) + 2 non-UI rows, from live user feedback (docs-only)

**What:** user reviewed the shipped Manage-keywords view + filter overlay and raised three things — the
"Manage keywords…" trigger sits under the scrollbar; nothing scales to ~100 families; and "102ff, 16p11 wtf
are that?". Scoped **PR-2.7 — Manage view at scale** (frontend-only) into `feature-tag-families.md`, plus two
**non-UI** backlog rows the third point exposed. **Docs-only; no code changed.**

**Why the "wtf" answer changed the design — traced every odd keyword to its source doc rather than assuming:**
they are **mostly real specialist vocabulary, not junk.** `16p11` is **16p11.2** (autism CNV) truncated at the
dot; `c57bl` is **C57BL/6** (mouse strain, **7 docs**) truncated at the slash; `va1v`/`dl5`/`osns`/`upns`/
`mgns` are Drosophila antennal-lobe glomeruli + neuron classes, all from **one** olfactory paper;
`avpv`/`pvpo`/`mpoa` are hypothalamic nuclei from one mating paper; `rabv` = rabies-virus tracing. The truly
broken ones are a small **clustered** minority: `mathrm` (LaTeX leak), `professium`, `outflux` — **all three
from the same 1952 scanned Hodgkin-Huxley paper** — plus `102ff` ("p. 102ff"), `fne-tune` (OCR of
"fine-tune"), `neurosc`. So the verb is **demote, not delete**: deleting real vocabulary isn't
reversible-by-search.

**The measurement that made it principled (live corpus, not assumed):** 60 keywords surface; **30 — exactly
50% — sit on a single document**, and every ugly string the user spotted is in that tail. **A facet exists to
partition a set; a 1-doc facet doesn't partition** (selecting it yields one doc, which search does better), so
rare keywords have near-zero *filtering* value **whether they're junk or gold**. That threshold sweeps up the
noise **without having to classify it** — and the overlay's existing search box is already the escape hatch.
Also measured: `FAMILIES (26)` is misleading — only ~6 are real families, the rest 0-member concepts inherited
from concept-graph seeding, several with 0 docs (`BERT`).

**Convergence worth not losing:** autocompleting the "New family" canonical against existing families is
simultaneously the user's navigation ask **and** the fix for defect **D3**. Carve guards against building it
twice: **PR-2.5 owns the `library.py` boundary invariant; PR-2.7 owns the control that stops the user reaching
it.**

**Rejected:** (a) *deleting the rare tail* — mostly real vocabulary; delete isn't reversible-by-search
(demotion is). (b) *modelling suppression as a "hidden" family* (a `Concept` whose aliases are the junk) —
abuses ADR-015, which defines a family as **synonym collapse**; overloading those rows corrupts every
consumer that reads them, including the concept-graph track that owns them later. (c) *folding PR-2.7's
autocomplete into PR-2.5* — the invariant belongs at the library boundary regardless; a view control is a
convenience on top, not a substitute (a second client bypasses it). (d) *a code-level stoplist for
suppression* (`VENUE_STOPWORDS` precedent) — wrong shape for **user-editable** data; the user-override
precedent is `DocumentMeta`/ADR-013 (a sidecar), and note the Enrichment-Layer Pattern does **not** govern it
(it's a user override, not derived data). (e) *treating the doc-count threshold as a locked setting* — it
governs **presentation**, not retrieval, so no eval-harness gate.

**Opens:** two **non-UI** rows now on the checklist. (1) **Extractor truncation** — `_TOKEN_RE`
(`keywords.py:36`) allows `-`/`+` as internal joiners but not `.`/`/`; fixing it touches the **ingest path**
(re-run extraction over 76 docs, re-check the `VENUE_STOPWORDS`/repeated-token interaction) and **must
preserve the ASCII-lowercase `Keyword.name` invariant** that the tag-families review leaned on (it's why
SQLite `lower()` == Python `.casefold()` there). (2) **Keyword suppression** — needs a decision on where it
lives (sidecar table + migration + UI write surface) → wants an ADR or a spec section; may pair with the
extractor fix (fix what you can, suppress what you can't).

---
## 2026-07-17 — Post-commit review of tag families PR-1/PR-2; PR-2.5 + PR-2.6 scoped (docs-only)

**What:** reviewed the two shipped tag-families commits (`0c3b0d4` PR-1, `0af43db` PR-2) — an agent review
of the diff plus a live drive of the running app on the real 76-doc corpus ($0/offline). **No code changed
this session; docs only.** Flipped the post-commit paperwork the commits left behind
(`feature-tag-families.md` + `ui-checklist.md` still said "staged, not committed"; now ✅ SHIPPED with SHAs),
and scoped two defect-driven follow-ups into the spec: **PR-2.5** (hardening the write paths, D1–D5) and
**PR-2.6** (family-aware grid tiles, D6). Full repros per defect live in the spec's carve table.

**Verified working live (so the review is not theoretical):** the overlay collapses families into atomic
entries (`Large language model` 3 forms, `Connectome` 3 forms, `Embeddings` 4 forms); selecting the LLM
family returns **14 documents**, all genuinely LLM papers (union correct) and every other chip recounts for
AND (`chatgpt` 7, `Embeddings` 7, non-overlapping → 0, greyed); Detect reproduced the DEVLOG's claim exactly
(`pvpo`≈`avpv pvpo` @ 0.77, tier MEANING) — **left unaccepted, real corpus unchanged**; 0 console errors.
Unplanned bonus confirmed on screen: the curated vocabulary supplies **display names**, so `bm25` renders as
`BM25` and `imagenet` as `ImageNet`.

**Why:** 977 tests passed and every gate was clean, yet **6 real defects** survived — all in the
**under-guarded write paths**. The read path reviewed clean (facet math, union-find determinism + the
`(a,b)` key ordering, thin-shell discipline, and the no-families default path being genuinely
byte-identical). Two defects **escape the feature**: D1 (rename onto an existing canonical → duplicate
`Concept` labels → create route 500s forever *and* `promote_keyword` throws `MultipleResultsFound`,
breaking `scripts/seed_concepts.py`) and D2 (rename silently drops the family's own canonical keyword,
re-creating the duplicate chip the feature exists to remove). Both sit on the **natural** post-PR-2 flow:
`_canonical_and_members` always proposes an existing keyword as canonical (`llm`), so Detect → Accept →
**Rename** is the obvious next click. D1 is precisely the boundary risk ADR-015 named in its Consequences,
now realized — the tag-families UI can corrupt vocabulary a different feature reads.

**Root cause of the test blind spot (worth keeping):** `test_rename_family_canonical` asserts only
`renamed.canonical` — never `aliases` or `doc_count` — and it even uses the exact
`create_keyword_family("llm", [])`→rename shape that triggers D2. The stemmer tests (D4) pick `boxes`/
`taxonomies`, the two inputs the sibilant/`-ies` rules happen to get right; verified against the real
`_stem`, `database`/`databases`, `size`/`sizes`, `cache`/`caches`, `response`/`responses` and
`analysis`/`analyses` all **MISS**, and `detect_family_proposals(["database","databases"])` returns `[]`.
The frontend grouping layer (`familyCanonicalMap`/`familyUnitsOf`) has **no tests at all**. So PR-2.5's DoD
puts the five repros in as regression tests *first* — they all pass today, which is the point.

**Rejected:** (a) *folding D6 into PR-2.5* — carved into its own PR-2.6 instead: D6 and the family-aware
tile display are the same root cause in the same file (`LibraryGrid` never learned about families), so
bundling them into a backend-correctness PR would both violate "never bundle" and touch that file twice.
(b) *fixing D3's free-text canonical in the Svelte view* — the "a keyword belongs to at most one family"
invariant (ADR-015) belongs at the `library.py` boundary; the view is a thin shell and a second client
would bypass a view-level guard. Same reasoning routes D1's collision check to
`library.rename_keyword_family` (→ 409) rather than to `commitRename`. (c) *logging the six to
KNOWN_ISSUES and building new features first* (user call — hardening was chosen instead; leaving Rename as
a data-corruption trap on the happy path was not acceptable). (d) *adding a DB unique constraint on
`Concept.label` for D1* — deferred to the PR: it's the right long-term shape but it's a migration touching
a table two other features read, so it needs its own decision rather than riding a UI fix.

**Opens:** PR-2.5 must **pick one** D2 fix (seed the canonical as a real alias on create, aligning
`create_keyword_family` with `promote_keyword`, vs. make rename carry the old label into the alias set) and
**note the migration for the 26 pre-existing curated concepts on this box** — they predate the feature and
carry no seeded alias. PR-3 (LLM confirm) stays parked behind Ollama proof (KI-4). The `Concept.label`
uniqueness question (rejected (d)) is now on the record for whoever touches the schema next.

---
## 2026-07-17 — Tag families PR-2: detection (feature-tag-families.md, ADR-015)

**What:** built PR-2 of the tag-families spec — a deterministic, zero-LLM detection pass that
proposes family groupings for un-familied keywords, for the user to review and accept (nothing
auto-applies). **Pure core:** new `src/doc_assistant/keyword_families.py`. Tier 1 (`_stem`,
morphological — a conservative plural/suffix stripper: `llms`→`llm`, `connectomes`→`connectome`),
Tier 2 (`_tier2_embedding`, bge cosine clustering via union-find so a chain of near-duplicates
proposes one group, not overlapping pairs — catches meaning-close/spelling-different pairs a stem
can't, e.g. `connectome`≈`connectomics`) with a hand-rolled Levenshtein `_edit_similarity` blended
into Tier 2's confidence as a *supporting* signal, never a gate. `embed_fn` is injected (no model
load inside this module) — `detect_family_proposals(names, *, embed_fn=None, embedding_threshold)`.
**Impure boundary:** `library.py` gains `detect_family_candidates(embed_fn=None,
embedding_threshold=None)` — loads every `Keyword` name, subtracts anything already a family's
canonical/alias (case-insensitive), and hands the rest to the pure core. **API:** `POST
/api/library/keyword-families/detect` (`apps/api/main.py`/`models.py`) reuses the controller's
already-loaded embedder (`controller.rag.embeddings.embed_documents` — no second model load), wraps
it to match the pure core's `embed_fn` contract, returns `KeywordFamilyProposalPayload[]`.
**CLI:** `scripts/detect_keyword_families.py` — report-only (no `--apply`, matching the DoD's
"nothing auto-applies"; `--no-embeddings` for an instant Tier-1-only pass, `--threshold`/`--model`
for Tier 2), loads a fresh embedder via the existing `concept_semantics.embed_texts` (acceptable on
a host CLI, unlike the API route). **Frontend:** `types.ts`/`api.ts` (`KeywordFamilyProposal`,
`detectKeywordFamilies`); `LibraryManageKeywords.svelte` gains a "Detect proposals" section (Detect
button → tier badge + canonical + members + confidence % + Accept/Dismiss per row; Accept routes
through PR-1's existing `onCreate`, Dismiss is a pure client-side filter — nothing was ever
written); `App.svelte` owns `detectProposals`/`detecting`/`detectError` state, cleared when the
Manage view closes (proposals are cheap to regenerate; staleness after other edits is harmless
since accepting still goes through idempotent CRUD).

**Why:** PR-2 was next per the design-locked spec's foundation-first carve (T8) — PR-1 shipped the
manual mechanism; PR-2 removes the toil of *finding* what to group by hand.

**Verified:** 14 new pure-core unit tests (`tests/unit/test_keyword_families.py` — stem edge cases,
Tier 1/2 grouping, the transitive-chain union-find case, Tier-1-consumed names excluded from Tier 2,
toy-vector Tier 2 with no real bge load) + 2 new integration tests (`/detect` route, 200 with both
tiers + familied-exclusion, empty-corpus 200/[]) — full suite **977 passed / 1 skipped** (pre-
existing, unrelated). ruff/ruff format/`mypy --strict src`/bandit clean; `svelte-check` **0/0**
(131 files). **Live on the real 76-doc corpus, $0/offline:** `--no-embeddings` found 0 proposals
(every simple-plural pair on this corpus is already familied or non-existent — a legitimate
negative, not a bug); the full run (real bge, offline-cached) found one genuine Tier-2 proposal —
`pvpo` ≈ `avpv pvpo` (confidence 0.77) — both via the CLI and via the app's Detect button (identical
result, confirming the API route's embed_fn wiring matches the CLI's). Accepted the proposal live
(created the family, refreshed the list, proposal cleared) then deleted it to leave the real corpus
unchanged (verification, not a curation decision for the user's data). Dark theme, mobile 375px,
0 console errors, 0 API server errors.

**Rejected:** returning Tier 2 as flat pairs (mirroring `concept_semantics.ConceptPair`) instead of
transitively-grouped proposals — the spec calls for "grouped candidates," and a chain of 3+
near-duplicates as one proposal is a better review unit than N overlapping pairs; adding a
fuzzy-match dependency (`rapidfuzz`/`python-Levenshtein`) for the edit-distance signal — ~15 lines
hand-rolled matches this repo's existing bias (cf. `keywords.py`'s `weirdness`/`c_value_scores`)
against a new dependency for one small deterministic computation; gating Tier 2 on edit-distance —
would defeat the tier's purpose (spelling-different, meaning-close pairs are exactly its job); a
`--apply` flag on the CLI script — the DoD is explicit that nothing auto-applies, so there is
nothing for `--apply` to do that the app's Accept button doesn't already own.

**Opens:** PR-3 (an optional, gated LLM confirm/merge pass — parked, prove on Ollama first per
KI-4) is the last carved piece; not scheduled. The Detect flow has no rate-limiting/debounce (a
user could click Detect repeatedly; each call re-embeds the current candidate pool — cheap at this
corpus's ~40-keyword scale, would want caching if the un-familied pool grows large). Proposals
aren't persisted across a Manage-view close/reopen (by design — see above) — if that reads as
friction once PR-2 sees real use, revisit.

**Staged, nothing committed (cpc §13).**

---
## 2026-07-17 — Tag families PR-1: families end-to-end, manual (feature-tag-families.md, ADR-015)

**What:** built PR-1 of the design-locked tag-families spec — a family = a curated `Concept` whose
`ConceptAlias` rows carry member `Keyword` names (ADR-015), reusing the existing vocabulary tables
(no schema change). **Backend:** `concept_skeleton.py` gains the missing mutation primitives
(`remove_alias`, `delete_concept`, `rename_concept` — matching `add_concept`'s style); `library.py`
gains `KeywordFamily` + `list_keyword_families`/`get_keyword_family`/`create_keyword_family`/
`rename_keyword_family`/`add_family_member`/`remove_family_member`/`delete_keyword_family` (thin
shells; `doc_count` = a case-insensitive union query over `document_keywords`/`Keyword`; a keyword
belongs to at most one family — `add_family_member` moves it off any other family's alias set).
Six new FastAPI routes under `/api/library/keyword-families` (`apps/api/main.py` + `models.py`),
mirroring the safe-delete/metadata-edit route conventions (404 on unknown family, 400 on a blank
canonical). **Frontend:** `types.ts`/`api.ts` wire contract; `library.ts` gains `familyCanonicalMap`/
`familyByCanonical`/`familyUnitsOf` (a pure pre-facet grouping step) plus an optional `keywordsOf`
accessor on `facetFilter`/`keywordFacets` (default = raw keywords, so the no-families path is
byte-identical to pre-PR-1 behavior); new `LibraryManageKeywords.svelte` (create/rename/add-remove-
member/delete, opened via a new "Manage keywords…" link in `LibraryKeywordFilter.svelte`, which also
now shows a family facet's "N forms" subtitle + hover listing its members); `App.svelte` loads
families alongside documents and threads `keywordsOf` through the facet/filter pipeline.

**Why:** PR-1 was next per the design-locked spec (ADR-015, grilled 2026-07-16) — the overlay built
last session ships raw per-keyword facets, so near-duplicates (`llm`/`llms`, `connectome`/
`connectomics`) still count as separate filters. Foundation-first carve (T8): manual CRUD before
detection (PR-2) or an LLM pass (PR-3, parked).

**Verified:** 17 new integration tests (`test_keyword_families.py` — CRUD, the move-on-reassign
invariant, union `doc_count`, route 200/404/400) + full suite **961 passed / 1 skipped** (pre-existing,
unrelated); ruff/ruff format/`mypy --strict src`/bandit clean; `svelte-check` **0/0** (131 files).
**Live on the real 76-doc corpus, $0/offline:** the box already carries 26 curated `Concept` rows from
earlier concept-graph work (e.g. `Large language model` ← `llm`/`llms`) — confirmed this is intended
reuse of the vocabulary, not pollution (ADR-015's "take advantage of," not the graph UI). The overlay
correctly collapsed `llm`/`llms` into one `Large language model` facet ("3 forms", hover lists the
aliases, count = union = 14 docs); toggling it filtered the grid to the 14-doc union and the strip
chip showed the canonical name. Full CRUD round-trip in Manage keywords (create a test family from an
un-familied keyword → add a second member → rename → remove a member → delete) verified live and
cleaned up (DB back to 26 concepts, no test residue). Dark theme flips via CSS vars; mobile 375px no
horizontal overflow; 0 console errors; 0 API server errors.

**Rejected:** listing only `source="keyword"`-promoted concepts as families (would hide manually
curated glossary entries like `RAG`/`BM25` that are equally valid single-keyword families; ADR-015
treats the whole vocabulary as reusable); mutating `facetFilter`/`keywordFacets` to hard-require a
`KeywordFamily[]` param (an optional `keywordsOf` accessor keeps the default path untouched, matching
the DoD's byte-identical requirement); collapsing `d.keywords` in place on `LibraryDocument` objects
(would also silently change what the grid tiles' own keyword chips display, out of PR-1's scope — the
collapse is confined to the facet/filter computation only).

**Opens:** PR-2 (detection: tiered morphological + `bge` embedding clustering, no auto-apply) and PR-3
(LLM confirm pass, parked — prove on Ollama first, KI-4) are next per the spec's carve. The Manage
view's per-family "add a keyword" `<select>` lists every un-familied keyword with no search/filter —
fine at the current ~40-keyword scale, would want a search box if the vocabulary grows a lot before
PR-2's detection reduces the un-familied pool. No UI surfaces a family's `source` field (`"manual"`
here) — not needed yet, but PR-2's detected proposals will likely want to distinguish themselves before
acceptance.

**Staged, nothing committed (cpc §13).**

---
## 2026-07-16 — UI: keyword filtering as a two-pane overlay (folds the inline-bar cut below)

**What:** grilled the just-built inline facet bar (entry below) — it doesn't scale past a few dozen
keywords — and redesigned the presentation into an on-demand **two-pane overlay**
(`LibraryKeywordFilter.svelte`, reusing the `LibraryMetaEditor` modal shell): left = a searchable keyword
list (Zotero mechanics — AND, grey-out unavailable, most-used-on-top); right = a live preview of the
matching documents (`title · author · year` + "N documents"). **Live commit, no Apply.** The inline
`LibraryFacetBar` is replaced by a slim `LibraryFilterStrip.svelte` — a "Filter by keyword" trigger + the
*selected* keywords as removable chips + a result count + Clear. `App.svelte` gains `keywordFilterOpen`
state; the overlay + strip share `facetList` / `visibleDocs` / the existing `toggle`/`clear` handlers.
**The pure `facetFilter`/`keywordFacets` logic is unchanged** — a presentation swap. Design lock + grill
ledger: `docs/specs/feature-keyword-filter-overlay.md`.
**Why:** user feedback in a `grill-me` session — an always-on chip bar is fine at 60 keywords, not at
600; an overlay (searchable, with a doc preview) is the standard escape hatch (command palette, Zotero's
tag selector, GitHub's label filter). 9 branches resolved, 3 parked.
**Verified ($0/offline, preview harness, real 76-doc corpus):** `svelte-check` **0/0** (130 files); strip
trigger → overlay opens (60 keyword rows sorted by count + "76 documents" preview); search "conn" → list
filters to `connectome`/`connectomics`/`connectomes`; toggle `embeddings` → preview "22 documents" + 26
rows greyed + grid behind 76→22 + strip chip + "Keywords · 1" + "22 docs"; **Esc closes and the selection
persists**; Clear → 76; dark theme adapts via CSS vars (scrim/surface/border/text); mobile 375px → panes
**stack** (single column), no horizontal overflow; **0 console errors** (verified on a fresh tab — the
mid-session HMR "Failed to reload LibraryFacetBar" errors were stale buffer ghosts from the delete).
**Rejected:** Zotero's *container* (a permanent docked tag panel would crowd the rail's Collections); a
fully overlay-only strip-less view (hidden filter state); draft-selection + Apply (the live preview makes
it redundant); user-pinned favorites + a general "Filters" hub + Cmd-K (all parked — see the spec).
**Opens:** user-pinned favorites ride tag-families (promote-to-`Concept`); the general Filters-hub reopens
when extended-metadata lands article-type/year/journal as filters; no JS unit tests (no vitest in the repo).

## 2026-07-16 — UI: faceted keyword filtering in the Library (Phase 8, frontend-only, staged)

**What:** the Library grid gains a **multi-select keyword facet bar** (new `LibraryFacetBar.svelte`).
Clicking a keyword chip toggles it into an **AND** filter; a chip greys out + disables when adding it
would empty the grid, and each available chip shows its live co-occurrence count. Retired the
single-select `{kind:'keyword'}` `LibraryCollection` (user pick — "pure facet"): keywords are no
longer a nav collection but an orthogonal facet on the collection → search → **facet (AND)** → sort
pipeline. Pure helpers `facetFilter` + `keywordFacets` (deterministic, ties break alphabetically) +
`KeywordFacet` in `lib/library.ts`; removed the now-dead `keywordGroups` and the Sidebar "Keywords"
nav group (+ its dead CSS). `LibraryGrid` `activeKeyword: string|null` → `activeKeywords: string[]`
(selected facets surface first + highlight on every tile). App: non-persistent `libraryKeywords` +
`toggle/clearKeywordFacets`; the filtered empty-state offers "Clear keyword filters". **Backend
untouched** — `LibraryDocument.keywords` already ships client-side; thin-shell preserved.
**Why:** user feedback — "click a keyword to filter; multi-select greys out the unavailable ones."
The lavender chips were display-only; this makes them a working facet. Live data also proves the
next backlog row's premise: `llms`/`llm` and `connectome`/`connectomics` show as separate chips
(tag families).
**Verified ($0/offline, preview harness, real 76-doc corpus):** `svelte-check` **0/0** (129 files);
live — 60 keywords (24 shown + "Show all (60)"), select `embeddings` → **76→22 tiles** + `pretrained`
19→14 / `llms` 13→7 recount + **26 chips greyed** (disabled, opacity 0.5, `cursor:not-allowed`,
`aria-pressed=false`); `llms`+`pretrained` → **5-tile AND intersection** with both surfaced +
highlighted on the tile; **Clear** restores 76; dark theme adapts via CSS vars (selected = ink on the
lightened `--accent`); mobile 375px no horizontal overflow; **0 console errors**.
**Rejected:** OR semantics (nothing is ever "unavailable" under OR → grey-out meaningless); an
orthogonal facet keeping the keyword-collection (two redundant keyword mechanisms); a backend
facet-count endpoint (76 docs + keywords already client-side → pure frontend); making tile chips
click-to-toggle (a chip lives inside the tile's body `<button>` — nested buttons are invalid HTML;
the facet bar is the v1 toggle home, tile-chip toggling deferred to a tile restructure).
**Opens:** `Tag` (user labels) not yet faceted (no producer/data — the natural follow-on); tag
families (`llms`/`llm` normalization) is the next backlog row (reuse `Concept`/`ConceptAlias`, needs
an ADR); tile chips as toggles (needs the nested-button restructure); no JS unit tests for the pure
helpers (no vitest in the repo — verified via svelte-check + the live harness).

## 2026-07-16 — cpc re-vendored 1.2.1 → 1.2.2; KI-16 RESOLVED

**What:** third cpc step today — the KI-16 fix landed upstream (cpc `bda91a5`, released as
**v1.2.2** same day) and this repo re-vendored via `cpc-init` re-run (`tools/conventions/cpc/`
`_VERSION` 1.2.2; the three deliberately-diverged lays — `AGENTS.md`/`GLOSSARY.md`/
`.claude/.gitignore` — pruned again, same as the 1.2.1 step). `docs_check` now skips embedded
checkouts structurally (`.venv`/`node_modules`/`.git` parts + any dir carrying its own `.git`),
so a live background-task worktree under `.claude/worktrees/` no longer produces phantom errors.
KI-16 flipped → RESOLVED (resolution bullet in the entry); CONTEXT wiring line → 1.2.2.
**Verified:** `docs_check --strict` **0/0 unfiltered** on the 1.2.2 vendor (upstream: 144/144
tests + the live repro on this repo's own worktree, 70 → 0, before the tag).
**Why:** closes the loop the same-day review opened — gate noise during background tasks was the
one environmental red herring left.
**Rejected:** waiting for the next natural touch to re-vendor (the fix specifically de-flakes THIS
repo's gate; adopt while the context is warm).
**Opens:** none.

## 2026-07-16 — Docs-staleness fix batch (applies the same-day review's findings)

**What:** applied the fixes from the docs review (entry below), docs-only. **ROADMAP:** flipped
**10 stale status cells** to committed with verified SHAs — S1 `2893544` · S2 `7224f10` · V2
`4fd772c` · V3a `181046c` · V3b `487f2df` · L4-A `9f597df` · **G3 `d7528ab` · G4 `5fc5964` · G6
`cb166d4` · G7 `1e1e7eb`** (the G-rows still said "staged — awaiting review" from 2026-07-08; SHAs
re-derived from `git log`/`-S`, not taken on faith — the review agent had mis-mapped two);
added rows **L5** (metadata enrichment + keyword de-noising, `8f31fe3`) / **L6** (metadata editing,
ADR-013, `e549254`) / **L7** (safe-delete, ADR-014, `95817fc`); fixed the stale Phase-7 bullet
("redesign not yet built" → built+validated, `concept_graph.py` deleted) and the 7d paragraph's
"host apply pending"; repointed all 17 `docs/sprints/SPRINT-*` references to `docs/archive/sprints/`
(every numbered contract is archived). **architecture.md:** `library.py` contract corrected
(read-only → + ADR-013/014 write paths); `ingest` row gains `registry` (S1); SQL-store node gains
`DocumentMeta`/`SourceFile` (+ sidecar note); the false "`concept_graph` … not replacing it yet"
claim replaced with the real state (deleted 2026-07-07, skeleton is the layer); `metadata_enrich` +
`concept_skeleton_enrich` named. **Specs:** 8 shipped specs advanced from design-locked/"NOT built"
to ✅ SHIPPED with SHAs (rag-sandbox, provider-switch, library-redesign Phase A, selective-ingestion,
visual-identity complete, ab-compare, library-browser, conversation-history) — design locks retained
as the design record. **ADR-002:** status corrected proposed → accepted/implemented (M0–M5 shipped
2026-06-25). **decisions.md:** 3 dangling `docs/doc-assistant-roadmap.md` routes → `docs/archive/…`.
**ui-checklist:** S1/S2 + L4-A + enrichment + metadata-edit + safe-delete added to §1, their §3 rows
flipped `[x]` with committed SHAs, visual-identity row → COMPLETE. **KNOWN_ISSUES:** KI-8 dated
correction (KI-7 citation outdated; markers default-ON again since G1 — the "mostly moot" bullet no
longer holds).
**Why:** the 2026-07-16 review found the paperwork lagging the code by up to 8 days; a doc that says
"staged" for committed work actively misleads the next session.
**Rejected:** stubbing the `docs/decisions.md` monolith (ADR-001 step 4) and deleting `HANDOFF.md` —
both are user decisions, left open; backfilling cpc headers onto the 19 specs (specs are
`[headers]`-exempt by config — optional consistency work, not staleness).
**Opens:** decisions.md stub vs hybrid (user call); HANDOFF.md retire/refresh (user call); the
enrichment row's local-LLM leftover pass; live end-to-end delete smoke (user's run).

## 2026-07-16 — cpc re-vendored 1.1.0 → 1.2.1 + documentation review (gates + judgment sweep)

**What:** re-ran `cpc-init` from the cpc checkout at release tag **`v1.2.1`** (via a temp git
worktree; not the unreleased 1.2.2 HEAD — `_VERSION` records releases): `tools/conventions/cpc/`
refreshed (19 modules; new `keypoint.py` = the ADR-020 workflow-boundary runner; the cpc LICENSE now
travels with the drop), and the two new 1.2.0 templates laid —
`docs/features/FEATURE-000-template.md` + `docs/specs/SPEC-000-template.md` (the per-feature
rationale layer + the ADR-019 executor brief). Pruned the three files `cpc-init` lays that this repo
deliberately diverges on: `AGENTS.md` (`init_check` stays unwired — ADR-014 entry-file adoption
consciously deferred, per `.pre-commit-config.cpc.yaml`), `GLOSSARY.md`, `.claude/.gitignore` (root
`.gitignore` already covers `.claude/*`). Gate battery on 1.2.1: `test_api_check` clean;
`sprint_check` green after flipping the **overdue SPRINT-019 → archived** (V3b shipped `487f2df`,
post-commit flip never done); `docs_check --strict` clean on the real docs — its 70 remaining errors
are phantom hits on the live `.claude/worktrees/` background-task worktree, logged as **KI-16**
(upstream one-line-class fix in cpc's `docs_check.py` identified; no `conventions.toml` workaround
exists). Docs corrected this session: `.claude/CONTEXT.md` (cpc version bump; wiring text no longer
claims `init_check` runs at pre-push; the new on-call `keypoint` command documented; the missing
**Provenote** product-identity fact added, ADR-012), `docs/ROADMAP.md` `updated:` bump (rule-12 WARN).
**Why:** adopt cpc 1.2.x; keep ADR-007's canonical wiring text honest; user-requested docs review.
**Rejected:** vendoring HEAD (1.2.2-unreleased); adopting AGENTS.md/GLOSSARY.md wholesale just
because `cpc-init` lays them (deliberate divergences stay deliberate); a `[headers] exempt` glob for
the worktree noise (`Path.match` is right-anchored — cannot left-anchor a recursive glob; KI-16).
**Opens:** the judgment sweep found real staleness beyond this session's fixes, deferred to its own
fix session: ROADMAP rows S1/S2/V2/V3a/V3b/L4-A still "staged, not committed" though committed +
no rows for metadata-edit (`e549254`)/safe-delete (`8f31fe3`); `architecture.md` stale (`library.py`
described "read-only" but now carries ADR-013/014 write paths; no `SourceFile`/`DocumentMeta`;
deleted `concept_graph` still described as present); specs `feature-rag-sandbox.md` +
`feature-provider-switch.md` say "NOT built" for shipped U1/U1c (`09afd0c`), 6 more shipped specs
never advanced past design-locked; ADR-002 still `Status: proposed` for the shipped desktop shell;
`docs/decisions.md` monolith never stubbed (ADR-001 step 4) — dual ADR home + 3 dangling
`docs/doc-assistant-roadmap.md` routes inside it; root `HANDOFF.md` (2026-05-26, self-described
transient "delete after pickup") contradicts the current phase map; 19/21 specs lack the line-1 cpc
header; `docs/ui-checklist.md` lags today's shipped work (boxes unflipped).

## 2026-07-16 — Fix: `POST /api/ingest` no-body scope resolves to the canonical path (Windows) + Python 3.12 pin

**What:** `apps/api/main.py` `ingest_start` now reads `app_settings.get_source_dir().resolve()` once at
the top, so the whole endpoint speaks one canonical path — the `scope=` ingest arg, `status.source_dir`,
and the selection pass. The no-body branch previously passed the *un-resolved* `str(source)` as `scope`,
diverging from the `files=` branch (already resolved via `registry.resolve_selection` →
`source_dir.resolve()`) and from the registry's universal `.resolve()` (`scan_sources` / `view_for`).
**Why:** fixes `test_selective_ingest.py::test_api_ingest_no_body_still_works`, a pre-existing failure
pulled in with S1 that tripped on Windows only — the un-resolved path kept the env-derived
`pytest-of-LDELEZ` casing while `src.resolve()` canonicalizes to the on-disk `pytest-of-ldelez` (the
same class of mismatch bites 8.3 short paths and symlinked source dirs). Idempotent in production
(`get_source_dir` already resolves in the env-override and stored-path cases).
**Rejected:** resolving only at the `scope=` call site (leaves `status.source_dir` non-canonical);
"fixing" the test's expectation (`str(src.resolve())` encodes the intended contract, matched everywhere
else). **Opens:** nothing.
**Also (build):** added a tracked `.python-version` = `3.12`. With no pin, `uv run` in a fresh worktree
selected Python 3.14 (`requires-python >= 3.10`) and built a broken venv — the project targets 3.12
(KI-2: native deps not 3.14-stable). Rebuilt this worktree's `.venv` on the official python.org 3.12.10
(`pythoncore-3.12-64`, matching the main venv; the uv-managed standalone hits the OpenSSL-applink crash)
from uv cache (`uv sync --extra cpu --extra dev --offline`, $0 / no network).
**Staged code + docs; `.venv` is gitignored (local only). Nothing committed without review (cpc §13).**

---
## 2026-07-16 — Document safe-delete: source file → Recycle Bin + confirmation (ADR-014)

**What:** single-document delete from the `⋯` menu — the source file goes to the **OS Recycle Bin**
(recoverable) and the doc leaves the library + search index, behind a confirmation dialog.
- **Backend (`library.delete_document(doc_id, chroma_db)`):** recycles the source file **first**
  (`send2trash`, resolved via `resolve_source_path`) — a trash failure raises and **aborts** the delete
  (never orphan a still-indexed file); a file already gone skips trashing. Then drops the `Document` row
  (FK-cascades citations/parts/similarities), the `DocumentMeta` override (no FK — explicit), the doc's
  chunks from the live Chroma store (counted), its figure dir (reuses `cleanup_orphan_figures`) and cached
  `.md`. Returns `DeleteResult(filename, trashed_file, chunks_removed)`. New `DELETE
  /api/library/documents/{id}` → 200 / 404 (unknown) / **409** (couldn't recycle the file).
- **Dependency:** `send2trash>=2.1.0` (base dep, pure-Python cross-platform trash; `uv add --native-tls`
  through the proxy; mypy override added).
- **Frontend:** `LibraryGrid` ⋯ menu gains a red **Delete…** item; new `LibraryDeleteConfirm.svelte`
  (scrim + card, Esc/Cancel/scrim-close) states "source file → **Recycle Bin** (recoverable) … removing
  its N chunks from the search index"; a red-tinted Delete button (busy state). `App.svelte` owns
  `deletingDocId`, drops back to the grid if the open doc was deleted, then re-fetches.

**Why:** user request for a "safe-delete" (file + DB) with a confirmation, and multi-select later.
**Decisions (ADR-014):** Recycle Bin over soft-delete/permanent (user pick); trash-first for consistency;
single-doc first, **multi-select (bulk delete / move-to-collection) deferred**.

**Verified:** `svelte-check` 0/0 (128 files); ruff/format + `mypy --strict src` (61) + bandit clean;
**pytest 15** (`test_document_delete` ×7: unknown→None, file trashed + row + chunks removed, file-already-
gone, **trash-failure aborts + row survives**, DELETE route 200/404/409 — all with a monkeypatched
`send2trash`, no real file touched; + `test_document_meta` ×8). UI live-verified up to the confirmation
(⋯ → red Delete… → dialog with the right target + chunk count + Cancel). **The live end-to-end delete
recycles a real file, so it is NOT run in automation** — the user's to try; the logic is covered by tests.

**Opens:** recovery is via the OS Recycle Bin (no in-app trash/restore view); **multi-select** (bulk +
move-to-collection) is next. The `send2trash` add re-resolved torch off `+cpu` on the proxy box; restored
via `uv sync --extra cpu --extra dev --native-tls`.

**Staged; nothing committed (cpc §13). New dep `send2trash` (pyproject + lock).**

## 2026-07-16 — Library polish: normalized tiles + sort control + active-keyword highlight (user feedback)

**What:** three frontend refinements to the L4 grid (no backend).
- **Normalized tiles** — every tile is now a **uniform height** (161px measured, all identical) because each
  row is a reserved fixed height: title (2 lines, clamped) → **byline** (author · publication year, reserved
  even when empty) → meta (`N pages · N chunks · **Added** <date>` — the ingestion date, now labelled to
  distinguish it from the publication year in the byline) → **keywords (always two reserved lines**, clipped
  beyond). Fixed the **author bug**: `authorLabel` split only on `,;and`, so space-separated author strings
  showed in full — now up to **3 authors show fully** (books/small collabs), **4+ collapse to "First et al."**;
  un-splittable space-only strings ellipsis-truncate (user can fix in the edit modal).
- **Sort control** (`libSort`, persisted) — a `↑↓` button + dropdown in the library toolbar (next to the
  grid/list toggle), options **Title A–Z · Author A–Z · Publication date (newest) · Added date (newest)**,
  applied client-side over the filtered collection (`sortDocs`). Default Title A–Z.
- **Active-keyword highlight** — selecting a keyword collection now surfaces that keyword **first + filled**
  (`.kw.active`) in every tile (`orderedKeywords`), so the reason a doc is listed is always visible even past
  the `+N` cap; the rail chip was already active-styled.

**Why:** user asked for a cleaner, normalized library — uniform boxes, "First et al." + date instead of all
authors, a clear ingestion-vs-publication date, reserved keyword space; a chat-style sort with more keys; and
the selected keyword highlighted/first in the boxes.

**Verified ($0, frontend-only):** `svelte-check` 0/0 (127 files); live on the real corpus — all 12 sampled
tiles measured an **identical 161px**; bylines "Reza Shadmehr, John W. Krakauer · 2008" (2 shown), "Laura E
Suarez et al. · 2022" (4+ → et al.), "2017" (year only); meta shows "Added 7/2/2026"; sort menu lists the 4
keys and Author A–Z reordered correctly; selecting `connectome` → all 18 tiles show it as the **first,
highlighted** chip. Test state reset (sort→default, collection→All).

**Opens:** the residual bad title/author extractions (hyphenation artifacts, a sentence as an author) are now
user-fixable via the edit modal — not a layout bug. Sort is one direction per key (reverse = easy follow-up).
**Deferred (user-requested, next):** document **safe-delete** (file + DB + index, with confirmation), then
**multi-select** (bulk delete / move-to-collection).

**Staged; nothing committed (cpc §13).**

## 2026-07-16 — Document metadata editing + reveal-in-explorer + author on tiles (ADR-013)

**What:** the first browse-time **write path** — a per-document `⋯` menu (Edit metadata / Reveal in
file explorer) mirroring the conversation ⋯ menu, plus the author on its own line on each tile.
- **Data model (`DocumentMeta` sidecar, ADR-013):** new table keyed by `document_id` with
  `title/authors/year_override`. Auto-extracted values stay the *default* on `Document`; **effective =
  override ?? default**; `customized` flags any override. `set_document_meta` **replaces** the small
  override set and **dedups each field against its auto default** (re-saving an untouched field creates
  no override), so a re-run of `enrich_metadata` never clobbers a user edit. Reset = delete the row.
- **Backend (`library.py` + 3 routes):** `set_document_meta`/`clear_document_meta` + `list_documents`
  now merges overrides (batch-loaded once) and carries `year`/`customized`; `resolve_source_path` +
  `reveal_document_source` (`_reveal_in_file_manager`: `explorer /select` / `open -R` / `xdg-open`,
  list-form, no shell). `PATCH /api/library/documents/{id}`, `POST …/reset-metadata`, `POST …/reveal`
  (404 on unknown doc / missing file). `LibraryDocumentPayload` gains `year`/`customized`.
- **Frontend:** `LibraryGrid.svelte` restructured tile/row into a container + body-button + hover-`⋯`
  (a `<button>` can't nest a `<button>`), single floating menu mirroring Sidebar; **author on its own
  muted line** (new `authorLabel`); a small accent "edited" dot when `customized`. New
  `LibraryMetaEditor.svelte` modal (Title/Authors/Year, Save/Reset/Cancel). `App.svelte` owns
  `editingDocId`, wires save/reset/reveal → API → `refreshDocuments` (re-fetch, like the chat actions).

**Why:** user request — correct the ~3% wrong titles + ~19 blank authors the extractor leaves, "like the
chats", with reset-to-default and a "must-have" reveal-in-explorer; author visible on the snippet.
**User decisions:** editable = Title/Authors/Year; author on its own line (a future "choose which columns
show" increment is out of scope but the override model supports it).

**Rejected:** override columns on `documents` (4 additive migrations + mixes user writes into the
extraction registry — the sidecar isolates them); a Tauri command + `shell:allow-open` for reveal (the
app's first Tauri command, untestable in preview — the API is always local, so a backend reveal fits the
100%-API-driven frontend). See ADR-013.

**Verified:** `svelte-check` 0/0 (127 files); ruff/format + `mypy --strict src` (61) + bandit clean;
**pytest 28** (new `test_document_meta`: dedup/blank-revert/reset/effective + PATCH/reset/reveal 200+404
with a monkeypatched reveal; + library regression). Live ($0): applied the new-table migration
(`python -m doc_assistant.db.migrations` — the API doesn't `create_all` on startup, same as
`conversation_meta`); API E2E PATCH→effective→reset→404 PASS; UI — tile shows title + author line +
`⋯`; menu → Edit metadata → change title → Save → tile updates + edit-dot; Reset → reverts (test edit
cleaned up). Reveal opens a real OS window on the host (mocked in tests; live-checked path = user's box).

**Opens:** the residual bad-author/OCR tail is now user-fixable. `document_meta` must be migrated on any
box that predates it (as `conversation_meta` was). Deferred: editing DOI/notes/tags/folders, per-field
reset, bulk edit; user-selectable library columns.

**Staged; nothing committed (cpc §13). `document_meta` table created in the live DB (empty).**

## 2026-07-16 — Keyword de-noising: venue/publisher/ID denylist + repeated-token filter

**What:** the library keyword chips (and the rail's Keywords nav) were dominated by scholarly-metadata
artifacts — `elife` (25 docs), `biorxiv`, `neuroimage`, `jneurosci`, `neurobiol`, `fnana`, `frontiersin`,
`zenodo`, `pmid`, `7554 elife` (the eLife DOI registrant). Two filters in `keywords.py::candidate_terms`
(the single choke-point every mode feeds through): (1) a curated **`VENUE_STOPWORDS`** frozenset (preprint
servers / repositories / publishers / journal abbreviations / ID labels) — a candidate is dropped if ANY
of its tokens is a venue token, so `elife` and the bigram `7554 elife` both go. **Deliberately excludes
words that double as domain concepts** (`cell`/`neuron`/`nature`/`science`) so real keywords survive.
(2) a **repeated-token n-gram** reject (`outflux outflux outflux` — an OCR artifact weirdness scored highly,
the exact case RG-001/R3 flagged). Regenerated the corpus vocabulary (`extract_keywords --mode contrastive
--force --apply`), sweeping the now-orphaned venue rows.

**Why:** user feedback — the chips were venue noise, not topics; "better metadata → better tags/keywords for
navigation." This is the follow-up lever R3 explicitly parked ("STOPWORDS/metadata strip for publisher
artifacts; collapse repeated-token grams").

**Rejected:** filtering author surnames (`sporns`/`cajal`) — not deterministically separable from topics
(`cajal` is both a person and a body of work), left as the manual-curation tail; denying `cell`/`neuron`/
`nature` (real neuroscience concepts that happen to be journal names) — would strip genuine keywords;
re-tuning the contrastive weirdness/C-value knobs (locked settings — a denylist is the surgical fix, no
eval needed since it only removes provably-non-topical tokens).

**Verified ($0/offline, deterministic):** dry-run reviewed before applying — venue tokens **0 remaining**,
the 60-term vocabulary now reads `embeddings`/`connectome`/`deeplabcut`/`bm25`/`cebra`/`res2net`/
`parcellation`/`tractography`/`markerless`/`keypoints`; ruff + `mypy --strict` clean; **pytest 53**
(3 new `candidate_terms` cases — venue+ID drop, keeps venue-homonym domain words, repeated-token drop — plus
the concept-skeleton suite that shares `candidate_terms`, unregressed). Applied to the live `data/library.db`
(**268 links, 14 orphan rows swept** across two passes); reloaded — tiles show clean domain chips, the
political-science paper honestly shows none (no domain keyword in a neuro/ML vocabulary).

**Opens:** residual non-venue proper-noun/OCR tail (`sporns`, `cajal`, `huggingface`, `mathrm`, `neurosc`)
— the **manual keyword/tag edit UI** (needs an ADR, first browse-time write path) is the fix. Concept
skeleton unaffected (built from the 26 curated concepts, not raw keywords). Re-runnable any time.

**Staged; nothing committed (cpc §13). Keyword rows regenerated in the live DB (reversible — re-extract).**

## 2026-07-16 — Metadata enrichment: wire the (unwired) extractor onto Document (real titles on tiles)

**What:** populated the empty `Document.title`/`authors`/`year`/`doi` columns (0/76 → title 76/76,
authors 57/76, year 66/76, doi 25/76) so the library grid shows real titles instead of filenames.
(1) **`metadata_enrich.py`** (new) — the runner: reads each doc's cached markdown (reuses
`keywords.load_document_texts`), runs the existing `metadata_extractor.extract_metadata`, and writes the
four columns. **Idempotent per column** — only fills a NULL unless `force` (so a later manual edit is never
clobbered); `apply=False` is a $0 dry run. Enrichment-Layer discipline: writes only those columns, never the
chunk store. (2) **`scripts/enrich_metadata.py`** (new) — thin CLI mirroring `compute_doc_vectors.py`
(`--apply`/`--force`/`--doc`, dry-run report). (3) **`metadata_extractor.py`** — the extractor was **fully
built but never called anywhere** (dead since Phase 4); wiring it revealed three false-positives on the real
corpus, fixed: skip publisher copyright/licence headings (Springer's "The Author(s), under exclusive
licence…" was hijacking a title), strip markdown hard-break backslashes (`WIESEL\`), and reject author
candidates that open with a discourse/section lead (`However,` / `Additional Key Words and Phrases:` — never
a name list; honest-empty beats a wrong author, so authors fell 61→57).

**Why:** user feedback on the new grid — filenames are unreadable; "if we improve the metadata we can make
better tags/keywords for navigation." Chose a **deterministic sidecar over editing source files** (never
mutate the user's PDFs — enrichment-layer + provenance) and **auto-fill first, manual-edit later** (the
manual-edit UI is the first browse-time write path → its own ADR, deferred).

**Rejected:** rewriting/renaming source PDFs (destructive, breaks re-ingest + provenance); an LLM extraction
pass (unnecessary — the deterministic heuristics already hit 100% titles on this corpus, $0/offline, and the
cost-discipline rule says prove the deterministic path first); storing `authors` as JSON (the frontend
`docLabel` splits a delimited string — kept the extractor's string form).

**Verified ($0/offline, deterministic):** dry-run on the real 76-doc corpus reviewed **before** applying
(quality gate); ruff/format clean; `mypy --strict` (2 files) clean; **pytest 27** (`test_metadata_extractor`
+ new `test_metadata_enrich`: apply-fills-NULL / dry-run-writes-nothing / idempotent-keeps-title /
force-overwrites). Applied to the live `data/library.db` (**224 columns written**); reloaded the grid —
tiles now read "A Primer on Motion Capture with Deep Learning…", "Res2Net: A New Multi-scale Backbone
Architecture · Shang-Hua…" etc., filename preserved as the hover `title`.

**Opens:** ~19 docs have no confident author (honest-blank) + a few noisy year picks — the **manual-edit UI**
(needs an ADR) is the fix for stragglers. **Keyword de-noising** is the natural next step (chips still show
venue/ID artifacts — `elife`/`biorxiv`/`zenodo`/`7554 elife` — from `keywords.py`, a separate change).
DOI 25/76. Re-runnable any time (`--force` to refresh, only-NULL by default).

**Staged; nothing committed (cpc §13). Metadata applied to the live DB (reversible — only-NULL fills).**

## 2026-07-16 — Library grid: mode-aware width + fixed-footprint tiles (user feedback)

**What:** two frontend fixes to the L4 grid from live-review feedback. (1) **Mode-aware main width**
(`App.svelte`) — `<main>` was hard-capped at `max-width: 820px` (the chat reading measure, ~68ch) and
centered, so in fullscreen the **library grid floated in a centered 820px column** with wide empty margins
and stayed stuck at 4 columns. Added `main.wide` (`max-width: 1500px`), bound `class:wide={mode ===
'library'}`; Chat keeps its 820px reading column. Measured at 1456px viewport: main 820→1122px, grid
775→1077px, **4→5 columns**, right-side whitespace 217→0. (2) **Fixed-footprint tiles** (`LibraryGrid.svelte`,
best-practice list from the user) — `.tile` gets `min-height: 128px`; `.name` clamps 3→**2 lines** with a
reserved `min-height: 2.7em` so a long filename (`2021.04.30.442096v1.full.pdf`) no longer reflows its row
(all tiles a uniform 140px); keyword chips cap at 3 + a **"+N" overflow chip** (`title` = the hidden ones)
so tags never wrap unpredictably; grid `minmax(150px→200px, 1fr)` for a comfortable tile; `aria-label={
docLabel(d)}` on tile+row buttons gives a clean accessible name instead of the whole-card text mash.

**Why:** the user tested the pulled-in Library Grid (9f597df) fullscreen and flagged the "weird white space"
+ long titles reflowing the grid; supplied a card-grid best-practices list.

**Rejected (from the list, not applicable here):** whole-card-as-`<a>` + CSS overlay (this is an in-app
open action, not URL navigation — a `<button>` with a clean `aria-label` is the correct semantic);
`flex-wrap: nowrap` clipping on the tag row (risks invisible tags — the 3+"+N" cap already bounds it);
skeleton loading states (deferred — docs load once from the local API, no async card-populate jank to mask).

**Verified ($0, frontend-only):** `svelte-check` **0/0** (126 files); live at 1456px — 5 uniform 140px tiles
per row filling the pane, 0 right-whitespace, "+8" overflow chip rendered, long filename clamped to 2 lines
(33px); Chat mode still `main` 820px / no `.wide`. Screenshot captured.

**Opens:** the 1500px cap centers the grid on ultrawide (>~1760px) — a readable-width choice, revisit to
`none` if full-bleed is wanted. Sidebar width is a separate user-resizable pref (default 260, persisted).
Next: **metadata enrichment** (real titles on tiles + de-noised keywords — the higher-leverage follow-up).

**Staged; nothing committed (cpc §13).**

## 2026-07-15 — Selective ingestion S2: Sources panel (scan · exclude · ingest-selected) in Settings

**What:** the S2 frontend over the S1 endpoints. (1) **`types.ts`** `SourceFile` (mirrors
`SourceFilePayload`) + **`api.ts`** `getSources()` / `patchSource(rel_path, excluded)` and
`startIngest(paths?)` extended to POST `{paths}` when a selection is given (no-arg keeps whole-dir);
`errorDetail` now renders the selective-ingest 400's `{error, offenders}` object, not `[object
Object]`. (2) **New `Sources.svelte`** — scans on mount (`GET /api/sources`, $0 stat-only) + a Rescan
button; per-file row = select checkbox + rel_path + status chip (new/changed/**indexed**/missing) +
an Exclude/Excluded toggle (`PATCH`); a **"Select new + changed"** quick action; **"Ingest selected
(N)"** → `startIngest([paths])` → the same tolerant poll as the whole-folder index → on done
`onCorpusChanged()` + clear selection + rescan. Excluded rows dim and drop from the selection;
`missing` rows can't be selected. (3) **Settings.svelte** mounts it as a new **"Manage files"**
section under "Your documents" (the parked S2-shape fork — resolved with the user to **fold into
Settings**, not a 3rd sidebar mode: simplest V1, ingestion stays in one place).

**Why:** S2 of `feature-selective-ingestion.md`. The M4 flow does whole-folder "index everything";
this adds per-file visibility + exclude + subset-ingest for a mixed/flat corpus, on the same locked
core (nothing in `src/` changed — pure renderer over the S1 API).

**Rejected:** (a) a dedicated **Sources sidebar mode** (Chat/Library/Sources) — more wiring + a
permanent tab for an occasional task; the user chose the Settings section for a simpler V1 (a mode is
a clean future upgrade if the file list outgrows the drawer). (b) a `doc_type` control — the column
is dormant (S1 lock), so no UI. (c) a CLI-style exclude in the panel — exclusion is one PATCH toggle.

**Verified ($0/offline, real 47-doc corpus):** `svelte-check` 0/0 (126 files). Live via the preview
harness: **`GET /api/sources` → 47 files all correctly derived `ingested`** (the `has_document`
path-match + cache-freshness both work on the real DB); the panel renders 47 rows with "indexed"
chips + exclude toggles; **exclude → `PATCH` persists** (fresh server GET confirms `excluded:true`,
row dims, summary "1 excluded") and **include reverts** it; **select a file → "Ingest selected (1)"
→ `POST {paths}` → `resolve_selection` → `main(files=[…])` → dedup-skip → "indexed 0 new, 1 unchanged,
0 errors"** ($0), then the UI shows done + clears the selection + rescans; dark theme resolves (muted
chips, dimmed excluded rows); mobile 375px the 92vw drawer rows fit with no overflow; **0 console
errors**. Test state restored (no excluded files).

**One-time migration surfaced (not an S2 bug):** this box's `data/library.db` was missing tables
(`conversation_meta` *and* the new `source_files`) — the API relies on ingest's `init_db()`
`create_all` to migrate, and it hadn't run since S1. Ran `init_db()` once → both created (23 tables),
`/api/sources` then worked. Same pattern as every prior additive table (Figure, gaps): an upgrading
user creates `source_files` on their next ingest. **Opens:** the API not migrating on startup is a
pre-existing latent gap (a stale DB 500s `/api/conversations` too) — a systemic "init_db in lifespan"
fix is its own change, out of S2 scope.

**Opens:** S2 could grow into a dedicated mode if the flat 47-file list ever needs full width. PR 17
(Zotero/Calibre) will surface as extra `SourceFile` producers. Touch note: the exclude toggle + row
checkbox are tap targets; no drag affordances involved.

**Staged; nothing committed (cpc §13).**

## 2026-07-15 — Selective ingestion S1: SourceFile registry + selection-scoped ingest (backend + CLI + API)

**What:** the S1 backend of `docs/specs/feature-selective-ingestion.md` (grilled + LOCKED same day).
Ingest is no longer all-or-nothing over one folder. Five parts.
(1) **`SourceFile` table** (`db/models.py`) — one row per discovered file: `rel_path` (unique key,
POSIX), `format`/`size`/`mtime`/`first_seen`/`last_seen` (identity), `excluded` (the one persisted
user intent), and a **dormant nullable `doc_type`** (ships now, wired to nothing — `create_all` can't
ALTER a column later, so a dormant column makes doc_type's future return a behaviour-only add). Rides
the additive `init_db()` `create_all`, like `Figure`. (2) **New `ingest/registry.py`** — pure core
(`derive_status` 8-combo truth table → new/changed/ingested/missing; `validate_selection` →
normalized rel_paths or `InvalidSelection` listing every offender by reason) + impure boundary
(`scan_sources` stat-only walk that upserts rows + derives status with **no content reads**;
`set_source_meta` PATCH seam; `resolve_selection` → explicit paths, explicit picks override
`excluded`; `plan_files` dry-run classifier; `view_for` single-row echo). (3) **`ingest.main(files=,
dry_run=)`** — `files` is a validated explicit list (mutually exclusive with `--path`/`--rebuild`,
skips orphan cleanup); an implicit walk now subtracts standing exclusions; `dry_run` reports
`would_add`/`would_reembed`/`skip_unchanged`/`excluded` **without loading embeddings or opening
Chroma**. (4) **CLI** `--files P…` / `--dry-run` with the exclusivity rules. (5) **API** — `GET
/api/sources` (scan + list), `PATCH /api/sources` (`excluded` only; 404 unknown), `POST /api/ingest`
gains an optional `{paths?}` body resolved + validated up front (bad path → 400 before anything
starts; no body = whole dir minus exclusions). Wire models `IngestRequest`/`SourcePatch`/
`SourceFilePayload` (named `SourceFile*` to avoid the citation-`SourceView` collision).

**Why:** the "in-app ingestion" pivot after L4. A flat mixed corpus needs user-controlled selection —
batch (pick a subset) + on-need (by status). The M4 "index everything" flow shipped; this adds the
registry + selection layer on the same locked ingest core (its six stages are untouched — this only
changes *which files enter*).

**Rejected / deferred (grill lock 2026-07-15, ledger in the spec):** (a) **`doc_type` behavior** —
deferred (all-PDF corpus → manual busywork, no consumer yet; it's not a chunk/embed lever); only the
dormant column ships. (b) **stateless computed listing** (no table) — rejected: persistent `excluded`
has nowhere else to live (no `Document` row pre-ingest), and the table is the status-listing source +
the PR-17 adapter seam. (c) **`default_doc_type` seeding fn** — omitted (no dead code; lands with the
column's activation). (d) **S2 UI shape** — parked to S2 kickoff. (e) a CLI exclude command — not
needed; `excluded` is set via the API/UI, honored by every surface.

**Verified (offline, $0, no real embeddings — cpc §13):** full gate green — ruff / ruff format /
`mypy src/` (60 files) / bandit (no issues) / **920 pytest passed** (+27: 19 unit truth-table +
validation, 8 integration) / coverage 84% (registry.py 96%). Integration proof on tmp dirs (no
corpus, no network): scan lifecycle new→ingested→changed→missing; resolve_selection excludes then an
explicit pick overrides; bad paths raise `InvalidSelection`; `main(dry_run=True)` returns the plan
with a `get_embeddings` **trap that never fires**; the API GET→PATCH→POST`{paths}` flow drives a fake
`ingest_fn` that receives `files=[b.pdf]` (scope None), unknown PATCH → 404, traversal path → 400,
no-body POST still uses the scope path. Zero regressions in the 125 existing ingest tests.

**Opens:** **S2** (Tauri Sources panel — status chips, exclude toggle, ingest-selected) — UI shape
(dedicated sidebar mode vs fold into Settings) decided at kickoff. PR 17 (Zotero/Calibre) writes this
registry through the same public fns (ADR-3 seam). `doc_type` reactivation when a 2nd format or a
routing eval arrives. Latent, pre-existing: a `source_dir` outside `config.DOCS_PATH` has no
resolvable markdown cache (`get_cache_path` is DOCS_PATH-relative) — `scan_sources` degrades such a
file to not-fresh rather than crash; unchanged by this work.

**Staged; nothing committed (cpc §13).**

## 2026-07-15 — Library redesign L4 Phase A: nav-tree rail + inventory grid + drill-down

**What:** the Library space is rebuilt from a flat doc list into a file-browser-style navigation
(spec `docs/specs/feature-library-redesign.md`, design-locked 2026-07-14). Three parts.
(1) **New `lib/library.ts`** — a client-side collection model: `LibraryCollection` (`all`/`type`/
`date`/`folder`/`keyword`), `docsFor()` to filter the cached document list, `dateBucket()`
(Today/This week/This month/Earlier relative to now), the `typeGroups`/`dateGroups`/`folderGroups`/
`keywordGroups` counters, and `docLabel`/`filterDocs` (moved out of `Sidebar.svelte` — the grid,
breadcrumb, and search all need them now). (2) **New `LibraryGrid.svelte`** — the "inventory" tile
grid (`repeat(auto-fill, minmax(150px, 1fr))`): each tile shows a format chip, the title/filename
(3-line clamp), page·chunk·date meta, and up to 3 keyword chips; a `list` view renders the old
stacked-row idiom instead. Dumb component — no filtering, emits `onOpenDocument(id)`. (3) **`Sidebar`
library mode** is now the nav tree: **All documents** → **Collections** (Phase-A empty-state until
folders populate) → **Types** (by `format`) → **Added** (date buckets) → **Keywords** (chips); Types
and Added render only with **≥2 entries** (a one-option filter is noise). (4) **`App.svelte`** owns
the drill-down: `libraryCollection` + `libraryDocId` + `libraryQuery`, a breadcrumb `Library ›
Collection › Doc` with a Back control (doc→grid, then collection→all), and the grid⇄list toggle
persisted in `localStorage` (`libraryView`). Library search now scopes to the **active collection**
(bindable `libraryQuery`) with a one-click "Search all N documents" escape on 0 matches. +7 Lucide
glyphs (`layout-grid`, `list`, `folder`, `file-text`, `calendar`, `tag`, `chevron-right`).

**Why:** the next Library increment after L1 (which parked search/filter/sort + folder navigation).
The user chose drill-down-with-Back over two-pane / detail-drawer from a clickable prototype; the
persistent rail means changing collection never needs "back" — only the doc→chunks step drills.

**Rejected:** (a) two-pane (chunks always in a right pane) and detail-drawer navigation — the user
preferred the file-browser feel (drawer noted as a possible later toggle reusing `SourcePanel`); (b)
a backend folder-tree endpoint now — Phase A filters entirely client-side because the
`LibraryDocument` payload already ships `format`/`added_at`/`keywords`/`folders`; folders (the one
empty axis on the current flat-ingested corpus) are Phase B (mirror source-dir subfolders at ingest
+ a backfill); (c) always-visible Types/Added sections — hidden below 2 entries so a single-format
corpus doesn't show a dead filter.

**Verified ($0, frontend-only, no backend/LLM change):** `svelte-check` 0/0 (125 files). Live on the
real corpus via the preview harness: nav tree renders (Collections empty-stated, **Types correctly
absent** — all-PDF corpus <2 types, **Added shows 2 buckets** — This month + Earlier, keyword chips);
inventory grid = 47 tiles at 4-col auto-fill; clicking the "medical image segmentation" keyword →
exactly 3 docs + breadcrumb `Library › medical image segmentation` + Back appears; opening a tile
drills to the `LibraryBrowser` chunk view (breadcrumb gains the doc crumb, view toggle hides); Back
returns to the keyword grid; grid⇄list toggle flips (3 rows / 0 tiles) and persists `libraryView`;
searching "colbert" inside the keyword collection → 0-match empty-state → "Search all 47 documents"
widens to All documents keeping the query → the single ColBERT match; dark theme resolves (`--bg`
`#1b1813`, dark `--lavender` `#b0a4ff`, tile surface `#23201a`), no body horizontal overflow; mobile
375px → off-canvas drawer (fixed, `translateX(-100%)` closed, `.open` + scrim on hamburger) + 2-col
grid; 0 console errors. Test residue restored (`libraryView` cleared, viewport/scheme reset).

**Opens:** Phase B — folder population (source-dir mirror at ingest + a backfill runner) + `GET
/api/library/folders` + server-side `folder`/`format`/`tag` filters, which lights up the Collections
section. Still parked: manual folder/tag editing (first browse-time write path — ADR trigger),
title/author metadata backfill (tiles show filenames until then), the detail-drawer variant,
virtualization for very large collections.

**Staged; nothing committed (cpc §13).**


---

*Earlier entries (2026-07-14 back to 2026-05-21) archived verbatim to [docs/archive/DEVLOG-archive-001.md](archive/DEVLOG-archive-001.md) on 2026-07-21.*
