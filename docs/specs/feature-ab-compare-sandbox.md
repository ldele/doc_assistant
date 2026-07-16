# Spec — A/B-compare sandbox, v1: retrieval diff

**Status:** ✅ **SHIPPED 2026-07-13 (U6, commit `c965418`; SPRINT-015 archived; "Test override"
contextual-button refinement committed after).** Was DESIGN-LOCKED (grilled 2026-07-13, `grill-me`);
retained as the design record. Roadmap PR **U6**. Owner: Claude Code.
**One PR.** Ledger at the foot. Realises ADR-010's option-4 north-star as its **retrieval-only, $0**
first increment — the ADR sequences A/B after the basic override surface (U1) and cautions *"do not
pre-build option 4 on the assumption that users want more knobs; validate with real use before widening"*
(`docs/decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md`, Confidence line). This is that
validate-first slice.

**Requirement.** U1 lets a user override query-time RAG knobs for the session, but gives no way to *see*
what the override did. **A/B compare** runs the **same query** under the **locked defaults** (A, the
control) and under the **current session `RagOverrides`** (B, the test), and shows the two **retrieved
source sets side by side** with a computed diff. v1 compares **retrieval only** — which is $0 (no LLM;
`retrieve_with_scores` is a pure retrieval call) and therefore fully verifiable **offline on the real
corpus**, no Ollama, no paid API. The full-answer 2× compare is the phased, cost-gated follow-up.

---

## Grounding

- **Retrieval is free; generation is the paid part.** `RAGPipeline.retrieve_with_scores(query, top_k,
  *, use_multi_query) -> list[tuple[Document, float]]` runs ensemble + rerank + parent dedup with **no
  LLM call**. `stream_answer` is the paid step and is **not touched** here.
- **The override object already exists and is request-scoped.** `RagOverrides` (5 fields) is threaded
  through `POST /api/chat` → `_handle_rag` → `retrieve_with_scores` with **no module-global mutation**
  (U1, ADR-010; there is an isolation-guard test). This spec reuses that exact discipline for a second
  entry point.
- **Only two of the five knobs affect retrieval:** `top_k` and `use_multi_query`. `synthesis_mode`,
  `reviewer_evidence_chars`, `epistemics_markers_enabled` are **answer-time** — invisible to a retrieval
  diff. And `top_k` alone changes only **depth**: for a fixed `use_multi_query`, the reranked order is
  identical, so a larger `top_k` set is a **superset** of the smaller (B ⊇ A or A ⊇ B). Retrieval
  **membership** only really moves on `use_multi_query`. Both facts must be surfaced honestly, not hidden.

## Decisions

| # | Decision | Deciding reason |
|---|---|---|
| 1 | **v1 = retrieval diff only.** Two ranked source lists (A=defaults, B=session override) + a diff. No answer generation. | $0, live-verifiable here; ADR "validate before widening"; no paid API this session |
| 2 | **A = locked defaults, B = current session `RagOverrides`.** Not two arbitrary configs. | ADR option-4's control-vs-test framing; no new config-picker UI |
| 3 | **Query held constant.** Both sides retrieve on the **raw request text** (no history rewrite), so the diff isolates the override, not a query change. | A fair A/B varies exactly one thing |
| 4 | **Per-turn action, not a mode.** A "Compare" button runs this one question both ways; no lingering state. | Cost-legible opt-in (matters once the answer-compare's 2× lands); doesn't complicate normal chat |
| 5 | **Request-scoped, no global mutation.** The compare endpoint builds a `RagOverrides` from the body and passes it into two `retrieve_with_scores` calls; nothing module-global is assigned. The isolation-guard test is extended to the compare path. | ADR-010's one correctness obligation |
| 6 | **Honest no-op / depth notes.** If B sets no retrieval-affecting knob (or matches defaults) → "your override doesn't change retrieval (these knobs change the answer, not the sources)". If B differs only in `top_k` → "same ranking, {N} {deeper\|fewer} sources". | inform-don't-block; no fake signal |
| 7 | **Diff by chunk identity.** Match A↔B on `sha256(page_content)` + `doc_hash` (the pipeline's own dedup key, `pipeline.py:217`); classify each source `in_both` (with A-rank vs B-rank), `only_in_a`, `only_in_b`. | Reuses the existing identity; deterministic |
| 8 | **Indicative, never a verdict.** The result card repeats U1's framing: a sandbox outcome demonstrates on one query; the eval harness is the only path to a new default. | ADR-010 governance / `benchmark-presentation-tone` |

## Contract — `src/doc_assistant/chat_controller.py` (or a small `compare.py`) — retrieval compare core

Pure diff + a thin impure runner:

```
@dataclass(frozen=True) CompareSource:
    rank: int; filename: str; page: int | None; section: str | None
    score: float; excerpt: str; identity: str          # sha256(text)+doc_hash

@dataclass(frozen=True) CompareRow:
    identity: str; source_a: CompareSource | None; source_b: CompareSource | None
    status: str                                         # in_both | only_in_a | only_in_b
    rank_delta: int | None                              # a_rank - b_rank when in_both

@dataclass(frozen=True) CompareResult:
    query: str
    sources_a: list[CompareSource]; sources_b: list[CompareSource]
    rows: list[CompareRow]                              # unioned, ordered by best rank
    eff_a: dict; eff_b: dict                            # {top_k, use_multi_query} effective each side
    note: str                                           # honesty note (Decision 6), "" when membership moved
```

- `diff_sources(a: list[CompareSource], b: list[CompareSource]) -> list[CompareRow]` — **pure**;
  union by `identity`, classify, compute `rank_delta`. Exhaustively unit-tested.
- `compare_note(eff_a: dict, eff_b: dict, rows) -> str` — **pure**; Decision 6 logic.
- `ChatController.compare_retrieval(text: str, overrides: RagOverrides) -> CompareResult` — **impure**;
  calls `self.rag.retrieve_with_scores` twice: A with defaults (`top_k=TOP_K`, `use_multi_query=None`),
  B with `overrides.top_k or TOP_K` + `overrides.use_multi_query`; maps to `CompareSource`
  (reusing `pipeline.format_citation` field extraction); `diff_sources` + `compare_note`. **No `self.llm`
  touch, no generation.**

**NOT responsible for:** answer generation (phased), persistence (a compare is ephemeral), construction-
time knobs (`CANDIDATE_K`/weights — read-only per ADR-010).

## Contract — `apps/api` (additive)

- `apps/api/models.py`: `CompareRequest {text: str, overrides: OverridesPayload | None}`
  (reuse the existing `OverridesPayload`); `CompareSourcePayload` / `CompareRowPayload` /
  `CompareResultPayload` (+ `from_result`).
- `apps/api/main.py`: `POST /api/compare` (synchronous JSON, **not** SSE) → builds `RagOverrides`
  exactly as `/api/chat` does → `controller.compare_retrieval(...)` → `CompareResultPayload`.

## Contract — `apps/desktop/src` (thin renderer)

- `App.svelte` / composer — a **"Compare"** button beside Send (reuse `.ghost`); disabled while a turn is
  streaming and when the composer is empty; on click, `POST /api/compare {text, overrides}` and render a
  `CompareCard` in the main pane (a non-`Turn` result block; does not enter `turns`/history).
- `lib/CompareCard.svelte` (new) — two columns **A (locked defaults) | B (session override)**, each a
  ranked list of `CompareSource` (filename · p · "section" · score · excerpt), with per-row diff badges
  (`＝` in-both + rank-delta arrow, `A only`, `B only`); the effective `{top_k, use_multi_query}` header
  per column; the honesty `note`; the "indicative, not a verdict — the eval harness is the only path to a
  new default" footer. Both themes; no body overflow; columns stack under mobile width.
- `lib/api.ts` — `compareRetrieval(text, overrides)`.
- `lib/types.ts` — `CompareSource`, `CompareRow`, `CompareResult` mirroring the payloads.

## Tests

**Unit (`tests/unit/test_compare.py`, new):** `diff_sources` — identical lists → all `in_both`,
`rank_delta=0`; a reorder → `in_both` with non-zero deltas; disjoint → all `only_*`; a superset (top_k
depth) → shared `in_both` + tail `only_in_b`. `compare_note` — no retrieval-affecting override → no-op
note; top_k-only delta → depth note; multi_query membership move → `""`.

**Integration (`tests/integration/test_api_compare.py`, new; monkeypatched retriever, no corpus/model):**
`POST /api/compare` returns both lists + diff + note; a body with `use_multi_query` flipped yields a
membership diff; a body with only `synthesis_mode` set yields the no-op note. **Guard:** the compare path
issues **no LLM call** (monkeypatch `self.rag.llm` / `stream_answer` to raise) and mutates **no module
global** (extend the U1 isolation-guard assertion to `compare_retrieval`).

Full gate green; `svelte-check` 0.

## Definition of done

- On the **real corpus** ($0/offline, no model): Compare with `use_multi_query` toggled shows a real
  membership diff (A-only / B-only rows); Compare with only `top_k` changed shows the depth note; Compare
  with no retrieval-affecting override shows the no-op note. Preview-harness proof (snapshots + synchronous
  evals per `.claude/KNOWN_ISSUES.md`).
- Both themes; columns stack on mobile; the "indicative, not a verdict" framing is visible.
- One `docs/DEVLOG.md` entry; ROADMAP U6 row + `docs/ui-checklist.md` (§3 A/B row → in-progress/shipped).

## Out of scope (deferred, with owners)

- **Full-answer 2× compare** — the phased option-4 form: generate both answers, show side by side.
  Doubles per-turn **paid** cost → needs explicit **cost gating** (opt-in, cost shown) and can't be
  verified without a model; sequence it after real use validates the retrieval compare (ADR-010).
- **Construction-time knob compare** (`CANDIDATE_K`, weights, provider) — read-only per ADR-010; a
  transient-rebuild compare is a separate, heavier decision.
- **Persistent compare mode**, **saving/exporting a comparison**, **>2-way compare** — later, if real use asks.

## Ledger (grill-me, 2026-07-13)

G0 surface → retrieval diff only (full-answer → phased) · G-trigger → per-turn Compare action ·
G-semantics → A=defaults, B=session override · G-backend → `POST /api/compare`, synchronous, twice-retrieve ·
G-diff → identity match + classify · G-honesty → no-op / depth notes · G-verify → preview-harness $0/live.
Reopens: full-answer compare once it can be validated with real turns + cost-gated.
