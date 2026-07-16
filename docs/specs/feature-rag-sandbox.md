# Spec — RAG sandbox: session-scoped, non-persistent query-time overrides

**Status:** ✅ **SHIPPED 2026-07-11 (U1, commit `09afd0c`; SPRINT-010 archived)** — was design-locked
2026-07-09 (ADR-010 accepted); retained as the design record. v1 scope = the basic
override surface (option 3). The A/B-compare view (ADR-010 option 4) was the recorded north-star —
since shipped as its retrieval-only v1 (U6, `c965418`, `feature-ab-compare-sandbox.md`).
**Owner of execution:** Claude Code (code + tests + the Svelte surface), when Phase 8 is active. Create
a cpc SPRINT contract from this spec at build time (`docs/sprints/SPRINT-000-template.md` shape).
**Pattern reference:** thin-shell boundary (`apps/` render, logic in `src/doc_assistant/` — root
`CLAUDE.md`); ChatController seam (PR-M0, `docs/archive/pr-m0-chat-controller.md`); FastAPI/SSE boundary
(PR-M2, `docs/archive/pr-m2-fastapi-boundary.md`); Tauri frontend (PR-M3, `docs/archive/pr-m3-tauri-frontend.md`).

**Goal (the why).** Phase 8's stated item is "a settings page exposing the RAG sandbox knobs." Today
the desktop shows them read-only. This feature lets the user **experiment** — see how the retrieval /
synthesis knobs change a single answer — **without** touching the eval-gated defaults the product's
measurable-quality claim rests on. The resolution (ADR-010) is that overrides are **session-scoped and
non-persistent**: they change *this answer*, never *the default*. Only the knobs that genuinely and
cheaply move a single answer are live; the rest are shown read-only **with the reason**, so the surface
informs rather than misinforms.

---

## ADR recap — what this feature is (full rationale: ADR-010)

**Context.** Locked settings change **only** via an eval-harness experiment (`.claude/CONTEXT.md`).
The knobs bind at three different times, and only query-time knobs can vary per answer without a
pipeline rebuild or a re-ingest. Some knobs (the BM25/vector weight) are **structurally inert on the
shipped top-K** — measured flat across `[0,1]` (`tests/eval/baselines/bm25_weight_sweep_2026-07-03.md`).
The FastAPI `ChatController` is a **process singleton** (`apps/api/main.py:204-208`) with **no
per-request knob path** today (`ChatRequest` = `{text, session_id}`, `apps/api/models.py:27-29`).

**Decision.** A session-scoped, non-persistent surface that overrides only the query-time knobs
(`TOP_K`, `SYNTHESIS_MODE`, `USE_MULTI_QUERY`), threaded **request-scoped** through
`POST /api/chat` → `ChatController._handle_rag` → `pipeline.retrieve_with_scores` — **never** by
monkeypatching module globals under the shared singleton. Non-persistence is the governance wall; the
eval harness stays the source of truth.

---

## Decisions (locked 2026-07-09, ADR-010)

| # | Decision |
|---|---|
| 1 | **Overrides are session-scoped and non-persistent.** They ride the request; the **backend stays stateless** w.r.t. them (never written to `config` / `.env` / `app_settings`). "Session" = the desktop app session — the frontend holds the override state in memory; **restart clears it**. `POST /api/settings` still writes only `source_dir`. |
| 2 | **Only the honest, query-time set is live:** `TOP_K`, `SYNTHESIS_MODE` (`ai`/`human`), `USE_MULTI_QUERY`. These are the knobs that move a single answer with no rebuild and no re-ingest. |
| 3 | **Locked knobs render read-only WITH the reason** — construction-time (`CANDIDATE_K`, retrieval weights, reranker, provider) and ingest-time (chunk sizes, `USE_PARENT_CHILD`). The BM25/vector weight is labeled **"inert on the shipped top-K by construction (measured)"** — a fact, never a slider. |
| 4 | **Overrides thread as explicit request-scoped parameters — no module-global mutation.** `pipeline.USE_MULTI_QUERY` / `chat_controller.SYNTHESIS_MODE` are never assigned; concurrent turns on the shared singleton must not interfere. This is the feature's one correctness obligation (ADR-010 Confidence ⚠). |
| 5 | **Provenance reflects the *effective* knob values,** and flags when a value differs from the locked default (e.g. `top_k=5 (default 10)`) — the observability ethos; the user always sees what actually produced the answer. |
| 6 | **The eval harness is named as the only path to a new default.** The surface tells the user how to promote an override to a default (run the experiment); a sandbox result is framed as indicative on one query, never a measured win. |

---

## Contracts (build-time)

### `src/doc_assistant/chat_controller.py` — the override type + threading (logic lives here, not in `apps/`)

```python
@dataclass(frozen=True)
class RagOverrides:
    """Session-scoped, per-turn RAG knob overrides. None = use the locked default.
    Non-persistent: never written to config/.env/app_settings (ADR-010)."""
    top_k: int | None = None
    synthesis_mode: str | None = None      # "ai" | "human"
    use_multi_query: bool | None = None
```

- `ChatController.handle(text, *, overrides: RagOverrides | None = None)` (and any streaming entrypoint)
  threads `overrides` into `_handle_rag`. Default `None` → **byte-identical to today** (backward compat).
- In `_handle_rag`:
  - `top_k` — replace the hardcoded constant at `chat_controller.py:590`:
    `eff_top_k = overrides.top_k if overrides and overrides.top_k is not None else TOP_K`, passed to
    `rag.retrieve_with_scores(standalone, top_k=eff_top_k, ...)`.
  - `synthesis_mode` — replace the module-global read at `chat_controller.py:611`:
    `eff_mode = overrides.synthesis_mode or SYNTHESIS_MODE`; branch on `eff_mode`.
  - `use_multi_query` — pass through to the pipeline (see below); do **not** touch the global.
  - The effective values already feed the provenance stamps (`chat_controller.py:640,656,671`); make
    them the **effective** values and add a "differs-from-default" flag for Decision 5.

### `src/doc_assistant/pipeline.py` — request-scoped multi-query

- `retrieve_with_scores(query, *, top_k: int = TOP_K, use_multi_query: bool | None = None)` — new
  optional param. At `pipeline.py:178`, branch on
  `(USE_MULTI_QUERY if use_multi_query is None else use_multi_query)` instead of the bare global.
  `None` preserves today's behaviour. **No new construction cost; no rebuild.**
- `top_k` is already a parameter (`pipeline.py:170`) — no change beyond the caller passing the effective
  value.

### `apps/api/models.py` — the wire model (thin)

```python
class RagOverrides(BaseModel):
    top_k: int | None = Field(default=None, ge=1)     # upper bound validated against CANDIDATE_K (below)
    synthesis_mode: Literal["ai", "human"] | None = None
    use_multi_query: bool | None = None

class ChatRequest(BaseModel):
    text: str
    session_id: str
    overrides: RagOverrides | None = None             # optional → backward compatible
```

- Validate `top_k <= CANDIDATE_K` (the candidate pool is fixed at build; a top_k above it is
  meaningless). Reject out-of-range with 422 (pydantic) — no silent clamp (inform-don't-corrupt).

### `apps/api/main.py` — pass-through + the settings-view fix

- `POST /api/chat` (`main.py:238-243`): map `body.overrides` (pydantic) → `chat_controller.RagOverrides`
  and pass it into the controller call. When `body.overrides is None`, pass `None` (unchanged path).
- `_settings_view()` (`main.py:113-137`): **source `retrieval_weights` from `config.BM25_WEIGHT`**
  (`{"bm25": BM25_WEIGHT, "vector": round(1 - BM25_WEIGHT, 3)}`) instead of the hardcoded literals at
  `main.py:136`, so the read-only display cannot drift from the real value. The view stays data-only;
  the read-only *reasons* (Decision 3 copy) are static UI text, not new API fields.

### `apps/desktop/src/lib/types.ts` + `api.ts` — the frontend wire

- `RagOverrides` TS interface mirroring the pydantic model; `streamChat(text, sessionId, overrides?)`
  includes `overrides` in the POST body when present.

### `apps/desktop/src/App.svelte` — own the session override state

- Hold the sandbox overrides in App-level `$state` (the "session"): `let overrides = $state<RagOverrides>({})`.
- Pass `overrides` into `streamChat(...)` on each `send()`.
- Pass `overrides` (bindable) + a reset callback into `<Settings>` (same pattern as `onCorpusChanged`).
- In-memory only → cleared on app restart (Decision 1).

### `apps/desktop/src/lib/Settings.svelte` — the sandbox surface

- A new **"RAG sandbox"** section, above/below the existing read-only Engine section:
  - `TOP_K`: slider/number `1..candidate_k` (bound from `settings.candidate_k`).
  - `SYNTHESIS_MODE`: `ai`/`human` segmented toggle.
  - `USE_MULTI_QUERY`: on/off switch, with a small "costs one extra LLM call" note.
  - A persistent, muted banner: **"Session only — resets when you restart. To change a default, run the
    eval harness."** (inform-don't-block posture).
  - A **"Reset to locked defaults"** button (clears the overrides object).
- The existing **"Engine (read-only)"** section gains the **reason** per locked knob (Decision 3),
  the BM25/vector weight explicitly `inert on the shipped top-K (measured)`.

---

## Build node

**Depends on:** the shipped ChatController (PR-M0), FastAPI/SSE boundary (PR-M2), Tauri frontend
(PR-M3) — all present. No new model, no migration, no re-ingest, no pipeline rebuild.
**Files owned:** `src/doc_assistant/chat_controller.py`, `src/doc_assistant/pipeline.py`,
`apps/api/models.py`, `apps/api/main.py`, `apps/desktop/src/lib/{types.ts,api.ts,Settings.svelte}`,
`apps/desktop/src/App.svelte`, tests as below.
**Status:** design-locked, ready to build (Phase 8).

### Guard tests (written with the build)
- `tests/unit/test_chat_controller.py` —
  - `top_k` override changes the `top_k` handed to `retrieve_with_scores` (assert via a fake pipeline
    capturing the arg); `synthesis_mode="human"` routes to `_human_result` even when the global default
    is `ai`; `overrides=None` (and all-`None` fields) reproduces today's effective values.
  - **Isolation guard (the ⚠):** a `handle` turn with `overrides` set, immediately followed by a
    `handle` turn with `overrides=None`, uses the locked defaults on the second turn — proving no state
    leaked (no module-global was mutated). No monkeypatch anywhere in the path.
- `tests/unit/test_pipeline_retrieval.py` — `retrieve_with_scores(use_multi_query=False)` skips
  expansion when the global is `True`; `use_multi_query=True` expands when the global is `False`;
  `None` follows the global. (No LLM: assert on `expand_query` call count via a fake.)
- `tests/unit/test_api_chat.py` (or the existing API test module) — `POST /api/chat` with an `overrides`
  body reaches the controller with a matching `RagOverrides` (injected fake controller captures it);
  `top_k` out of range (`0`, `> candidate_k`) → 422; **absent `overrides` → controller receives `None`**
  (backward compat); `POST /api/settings` still accepts only `source_dir` (overrides are not writable).
- `tests/unit/test_settings_view.py` — `_settings_view()["retrieval_weights"]` equals the value derived
  from `config.BM25_WEIGHT` (not a hardcoded literal); moving the env var moves the reported weight.
- Frontend — `svelte-check` clean; the surface is exercised via the preview harness (drivable server on
  a free port; snapshots + synchronous evals per the box's known-good path).

### Definition of done
- The three knobs (`TOP_K`, `SYNTHESIS_MODE`, `USE_MULTI_QUERY`) override **per turn**, threaded as
  explicit request-scoped params; **no module-global is ever assigned**; the isolation guard test passes.
- **Non-persistence proven:** overrides are never written to `config` / `.env` / `app_settings`; a
  restart returns to locked defaults; `POST /api/settings` writes only `source_dir`.
- Only the query-time set is live; locked knobs render read-only **with the reason**; the BM25/vector
  weight is a labeled fact, not a slider.
- Provenance shows the **effective** knob values and flags any that differ from the locked default.
- `top_k` validated to `[1, CANDIDATE_K]`; out-of-range → 4xx (no silent clamp).
- `_settings_view` `retrieval_weights` sourced from `config` (display truthful).
- **Backward compat:** with no `overrides` sent, the public eval is byte-identical and the SSE turn is
  unchanged.
- No paid LLM calls in tests (cpc §13); ruff / `mypy --strict` / bandit clean; `svelte-check` clean;
  DEVLOG entry + a live-UI verification note (preview harness).

## Out of scope (v1)
- **A/B compare** (locked defaults vs override, side by side) — ADR-010 option 4, the north-star; a
  later phase (≈2× per-turn cost + more UI).
- **Construction-time knobs as live knobs** (`CANDIDATE_K`, retrieval weights, reranker, provider/model)
  — would need a transient pipeline build; read-only here.
- **Ingest-time knobs** (chunk sizes, `USE_PARENT_CHILD`) — need a re-ingest; read-only here.
- **Persisting overrides as defaults** — rejected (ADR-010 option 2); the eval harness is the only path
  to a new default.
- `EPISTEMICS_MARKERS_ENABLED` / `REVIEWER_EVIDENCE_CHARS` as sandbox knobs — query-time but cosmetic /
  niche; revisit after real use (ADR-010 "must revisit").
- **Server-side session storage of overrides** — v1 keeps the backend stateless; the frontend owns the
  session state. Revisit only if overrides must survive a frontend reload within a run.
