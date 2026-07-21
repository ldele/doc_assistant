<!-- status: design-locked · created: 2026-07-21 · owner: Code · plan: docs/PLAN_2026-07-21_exploration-epistemics.md (E2) · adr: ADR-027 D3 -->

# Feature spec — E2: always-on source-evaluation strip (ADR-027 D3)

Build contract for ROADMAP row **E2**. Implements ADR-027 **D3**: every retrieved source gets an
evaluation row below the chat answer **regardless of any toggle** — coverage/direction, doc year,
rerank score, and a "computed as of `graph_version`" freshness hint. $0 by construction (a sidecar
lookup joined against `TOP_K` sources — no LLM). Full-stack: backend join + wire + a Svelte component.

## Why now / honesty
D1 reorders epistemics surfacing toward *helping the user judge sources while exploring*. E1.1 made the
marker join trustworthy (KI-8), so an always-on strip built on it is honest, not a ~40%-lying UI. E1.2
gave `_build_source_views` / the TurnResult assembly clean seams to slot into. **RG-019 (a `contested`
denominator) stays deferred** — measurement-gated (eval harness), and moot on this box (0 Node-B stance →
nothing fires `contested`, so the strip is honest-uniform, not saturated); noted, not built here.

## Boundary vs D2/E3
D3 (this strip) is **always-on** assessment. D2/E3 (the persisted answer-influence toggle over
`eff_markers_enabled`) governs the *answer-surface* marker chips only and **never hides this strip**
(ADR-027 boundary). So the strip's per-source `evaluation` attaches unconditionally; the existing
`markers` field (the answer-surface chip in `Turn.svelte`) stays gated by `eff_markers_enabled`.

## Items

### E2a — backend reads (`knowledge/epistemics.py`, `library.py`)
- `ChunkEval(coverage: str | None, superseded: bool, n_claims: int)` — the per-source epistemic
  summary. `coverage` derived from a `chunk_epistemics` row: `contested` if `n_contested > 0`, else
  `corroborated` if `coverage_summary["corroborated"] > 0`, else `unique` if `> 0`, else `None`.
  `superseded = n_superseded_trend > 0`.
- `load_source_evaluations(chunk_keys) -> tuple[dict[str, ChunkEval], str | None]` — reads
  `chunk_epistemics` for the given keys (chunk_key column, NULL-fallback to `{doc}:{chunk_index}` like
  `load_epistemics_index`), returns per-key `ChunkEval` + the sidecar's `graph_version`. `OperationalError`
  (never-migrated DB) → `({}, None)`, honest-empty (0-doc contract).
- `current_graph_version() -> str | None` — a cheap single-row read of `concept_presence.graph_version`
  (the skeleton's build stamp) for the staleness compare; `None` when no skeleton built / table absent.
- `library.document_years(document_ids) -> dict[str, int]` — a **scoped** `SELECT id, year` join (not
  `load_doc_years()`, which loads the whole corpus — KI-18 discipline). Skips docs with no year.

### E2b — controller (`chat_controller.py`)
- `SourceEpistemics(coverage, superseded, n_claims, year)` (render struct) on `SourceView.evaluation`;
  `SourceView.reranker_score: float` (carried through `_build_source_views`, which already receives the
  score). `SourceEvalSummary(graph_version, stale)` on `TurnResult.source_eval`.
- `_attach_source_evaluation(sources, scored, *, markers_enabled) -> SourceEvalSummary` replaces the
  `_attach_markers` call: **always** attaches `sv.evaluation` (coverage/superseded/n_claims + year) and
  `sv.reranker_score`; sets `sv.markers` **only when `markers_enabled`** (D2 gating preserved, derived
  from the same read — one sidecar read, not two). Returns the strip freshness: `stale = sidecar_version
  is not None and sidecar_version != current_graph_version()`. Advisory + WARNING-logged like E1.1 (any
  failure leaves an empty evaluation, never breaks the turn). `_human_result` gets the same attach.

### E2c — wire (`apps/api/models.py` + `apps/desktop/src/lib/types.ts`)
- `SourceViewPayload += evaluation: SourceEpistemicsPayload | None, reranker_score: float`;
  `SourceEpistemicsPayload {coverage, superseded, n_claims, year}`.
- `TurnResultPayload += source_eval: SourceEvalSummaryPayload | None {graph_version, stale}`.
- `types.ts` mirrors both (wire-drift rule). `ConversationSource` (replay) stays degraded — no eval.

### E2d — frontend (`apps/desktop/src/lib/SourceEvaluation.svelte` + `Turn.svelte`)
- New always-on strip below the answer (sibling to `SourceCard`/`SourcePanel`): a compact per-source
  row — a color-coded coverage chip (`contested`=warn, `corroborated`=ok, `unique`=neutral, none=muted
  "not assessed"), a `⤳ superseded` badge when set, the doc year, the rerank score — plus a footer
  "assessed as of `{graph_version}`" with a **stale** warning when `source_eval.stale`. Renders nothing
  extra (honest-empty) when `source_eval` is null (sidecar absent / 0-doc). paper-&-ink tokens; light +
  dark; 375px no overflow; 0 console errors.

## DoD / guard tests
1. **Always-on:** with `eff_markers_enabled=False`, `sv.evaluation` is still populated (coverage/year)
   while `sv.markers` stays `[]` (the D3-not-D2 boundary). Fails today (no evaluation field).
2. **Coverage derivation:** a source on a `contested` chunk → `coverage="contested"`; a `unique` chunk →
   `"unique"`; a source with no `chunk_epistemics` row → `evaluation.coverage is None` ("not assessed").
3. **Freshness:** `stale=True` when the sidecar's `graph_version` differs from `current_graph_version()`.
4. **Byte-identical answer:** the strip is additive — `result.answer` / `sources_md` unchanged (parity).
5. **Wire round-trip + `svelte-check` 0/0.**

## Build order & gate
E2a → E2b → E2c → E2d → tests. Per item a non-vacuous guard test; then `ruff`/`ruff format`/
`mypy --strict src`/`bandit`/full `pytest`/`svelte-check`. Live $0 verify via the `window.fetch` SSE
mock (inject a source with a contested eval + year + a stale version) — `read_page` + computed styles
(screenshots flaky on this box), light + dark + 375px. One `docs/DEVLOG.md` entry; ROADMAP E2 row.

## Out of scope (deferred, recorded)
- **RG-019** — the `contested` denominator (measurement-gated; moot at 0 stance here).
- **E3 / D2** — the persisted answer-influence toggle (separate ROADMAP row).
- The answer-level `weak_retrieval`/`single_source_risk` signals already render in the Provenance panel;
  the strip focuses on the **new** per-source assessment + freshness rather than duplicating them.
