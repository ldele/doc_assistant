# Spec — PR-M2: FastAPI backend + SSE boundary (the desktop API contract)

> **📦 Archived — shipped; the historical code-level contract, archived here (2026-07-11).** Live status: ROADMAP row M2 (done). The behaviour of record is `apps/api/` + tests, not this spec.

**Status:** ✅ BUILT — designed by Cowork 2026-06-21; built + gate-green by Claude Code 2026-06-22 (Tauri migration, `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`). Third PR of the migration (M2). **Depends on PR-M0** (`ChatController`/`TurnResult`/`TurnEvent` must exist) and is best done after **PR-M1** (so the 7d `SourceView.markers` field is already in the payload the API serializes — it is).
**Owner of execution:** Claude Code (code + tests).
**Pattern reference:** thin-shell rule (`apps/` carry no logic; `.claude/CONTEXT.md` rule #3). FastAPI is **just another renderer** over `ChatController` — exactly like `apps/cli.py` and `apps/chainlit_app.py`. It owns no business logic; it maps `TurnEvent` → HTTP/SSE and HTTP requests → controller calls.
**Migration context:** `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md` (Execution §, M2). Provides the contract the Tauri frontend (PR-M3) speaks; bundled as a sidecar in PR-M4.

**Requirement (the why).** The Tauri frontend needs a local HTTP surface to drive the RAG/integrity pipeline. This PR exposes `ChatController` over FastAPI: a streaming chat endpoint (SSE), claim-adjudication and export endpoints, figure/source serving (replacing Chainlit's `cl.Image`/side elements), a health endpoint for the sidecar readiness gate (PR-M4), and a settings surface stub for Phase 8. **Chainlit and the CLI keep working unchanged in parallel** — this PR *adds* a third frontend; it removes nothing.

**Cost & placement.** New dependency: `fastapi` + an ASGI server (`uvicorn`) + `sse-starlette` for clean SSE (or hand-rolled `StreamingResponse` — see ADR-2). No LLM/torch/GPU added. Runs via `uv` in dev (`uvicorn apps.api.main:app`); PyInstaller-frozen sidecar is PR-M4, **out of scope here**.

---

## ADR-1 — FastAPI as a thin renderer over `ChatController`

**Context.** PR-M0 made the turn orchestration a library service yielding `TurnEvent`s ending in a `TurnResult`. The CLI and Chainlit renderers already consume that stream. The desktop app needs the same stream over HTTP.

**Decision.** `apps/api/` is a FastAPI app that, per request, calls the **same** `ChatController` and maps its output to HTTP. No retrieval/provenance/claim/citation logic appears in `apps/api/` — only (a) request → controller-call translation and (b) `TurnEvent`/`TurnResult` → JSON/SSE serialization. The controller is constructed **once** at app startup (model load is expensive; mirror how Chainlit holds one `rag`).

**Options considered.**
1. *FastAPI thin renderer over the shared controller (chosen).* One orchestration, three renderers; the boundary is trivially testable with a fake controller.
2. *FastAPI re-implements the turn flow against `pipeline.py` directly (rejected).* Re-introduces the trapped-logic problem PR-M0 just fixed; two orchestrations drift.
3. *Keep Chainlit's backend, add FastAPI only for non-chat endpoints (rejected).* Two backends, two session models; defeats the migration.

**Consequences.** The frontend (PR-M3) and any future client (an outbound MCP server — already a roadmap "later/open" item) speak one documented contract. The controller stays UI-agnostic. Adding an endpoint = mapping one more controller method.

## ADR-2 — SSE for token streaming; plain POST for actions

**Context.** The only server→client *push* in a turn is the token/step stream; everything else (adjudicate, export, settings) is request→response. The migration plan (ADR) already chose SSE over WebSocket for simplicity and because it maps 1:1 onto `TurnEvent`.

**Decision.** `POST /api/chat` returns `text/event-stream`; each `TurnEvent` becomes one SSE event: `event: token` (data = the delta), `event: step` (data = `{name, status}`), terminating `event: result` (data = the full `TurnResult` JSON), then `event: done`. All other endpoints are ordinary JSON POST/GET. Use `sse-starlette`'s `EventSourceResponse` (handles heartbeats/cleanup) wrapping the controller's iterator; if avoiding the dep, a `StreamingResponse` with a manual `event:/data:` generator is acceptable (note the trade-off: no built-in heartbeat).

**Options considered.**
1. *SSE for chat, POST for the rest (chosen).* Simplest correct model; one-directional streaming is exactly the need; trivial to bundle.
2. *WebSocket for everything (rejected).* Full-duplex unused; more lifecycle/packaging complexity; Chainlit's own WS model is part of what we're leaving.
3. *Poll a job endpoint (rejected).* Adds latency + state; worse UX than streaming.

**Consequences.** The frontend uses `EventSource`/`fetch`-stream for chat and `fetch` for actions. SSE survives the Tauri webview and the sidecar boundary cleanly. Token-streaming latency must not regress vs Chainlit (RIGOR_TODO in the ADR; measured in PR-M4 on the frozen build, smoke-checked here in dev).

## ADR-3 — In-memory single-user session store

**Context.** PR-M0's `Session` is caller-owned. A desktop app is single-user; the API still needs to associate a `session_id` with its `Session` across the `/chat` and `/claims`/`/export` calls of one conversation.

**Decision.** A module-level `dict[str, Session]` keyed by `session_id`, created on first `/chat` for a new id. No persistence, no eviction policy in v1 (single user, process-scoped; the app restarts clear it — acceptable, and consistent with today's Chainlit per-chat reset). Name the assumption in the module docstring.

**Consequences.** Multi-user / persisted sessions are a later, non-breaking change (the `session_id` is already a UUID-style key, matching the provenance "UUIDs everywhere → multi-user later is non-breaking" rationale). Reloading past sessions from `AnswerRecord`s stays the deferred item it already is.

---

## Decisions

| # | Decision |
|---|---|
| 1 | **New package `apps/api/`** (`main.py` = app factory + routes; `models.py` = pydantic request/response schemas; `sessions.py` = the `dict` store). Thin shell: **no `chainlit`, no business logic.** |
| 2 | **One `ChatController` per process**, built at startup (FastAPI `lifespan`/startup). Holds the loaded `RAGPipeline`. |
| 3 | **Endpoints (the contract)** — table below. Chat streams SSE; everything else JSON. |
| 4 | **`TurnResult` → JSON** via a pydantic model mirroring the dataclass (or `dataclasses.asdict` + a response model). The pre-rendered markdown blocks are passed through as strings; the structured fields (`sources`, `flagged_claims`, `usage`) are real JSON so the frontend can render natively (it does **not** have to parse the markdown — the markdown blocks are a convenience/fallback). |
| 5 | **Figures served by the API**, replacing `cl.Image` disk paths: `GET /api/figures/{figure_id}` streams the PNG from `figures.load_figure_image_paths`. The frontend references this URL; no filesystem path crosses the boundary. |
| 6 | **Claim adjudication + export** map to `controller.adjudicate` / `controller.export_conversation`. Export returns the file (stream) or a path the frontend can open; for the desktop app prefer **streaming the bytes** (the frontend saves), since a server-side path is meaningless to a packaged client. |
| 7 | **Settings endpoint is a read-mostly stub** (`GET /api/settings` returns the locked-settings view from `config`; `POST` accepts only the env-toggleable knobs). Full Phase-8 settings UI is later; this PR ships the read surface + a documented 501/no-op for writes not yet wired. |
| 8 | **No paid API calls in tests** (cpc §13): every test uses a **fake `ChatController`** (a stub yielding canned `TurnEvent`s) — the API layer is tested for mapping/streaming/serialization, never for real retrieval or LLM. |
| 9 | **CORS / bind:** bind `127.0.0.1` only (local app; never `0.0.0.0`). Allow the Tauri dev origin in dev; in the packaged app the frontend is same-origin via the sidecar. Keep the allowed-origins list explicit (no `*`). |

**Edge cases (spec explicitly):**
- *Unknown `session_id` on `/chat`* → create a fresh `Session` (first turn of a new conversation). On `/claims`/`/export` with an unknown id → `404` (no session to act on).
- *Client disconnects mid-stream* → the SSE generator must stop cleanly (don't keep generating tokens into a dead connection; `sse-starlette` handles this — if hand-rolling, check `await request.is_disconnected()`).
- *Provenance/reviewer failure inside the turn* → already non-fatal in the controller; the `result` event still carries the `TurnResult` with the failure string in `provenance_card_md`. The API does not add its own error wrapping for this.
- *`SYNTHESIS_MODE=human`* → `/chat` streams no interpretation tokens; the `result` event's `TurnResult.mode == "human"`. The frontend renders evidence-only. No special endpoint.
- *Figure id not found* → `404`. *Source/record not found* → `404`.
- *Concurrent requests on one session* (rapid re-send) → v1 is single-user; serialize per session is **not** required, but do not share mutable `Session` state across two in-flight turns of the same id unguarded — simplest: the frontend prevents overlapping sends (note it as a frontend contract for PR-M3).

**Build-time confirmations:**
- `TurnEvent`'s concrete shape from PR-M0 (tagged union representation) — the SSE mapper switches on it; confirm the variant discriminator.
- `figures.load_figure_image_paths` signature + that a `figure_id` → on-disk PNG path is resolvable at request time (it is, per the shipped 4c figure-UI work).
- Whether `export.write_markdown` returns a `Path` whose bytes can be streamed (yes) — for Decision 6.

---

## Contract — `apps/api/main.py` (new)

FastAPI app factory + routes. One `ChatController` built in `lifespan`. Routes map to controller methods; **no logic beyond mapping/serialization.**

### Endpoint table (the boundary contract)

| Method | Path | Body / Params | Response | Maps to |
|---|---|---|---|---|
| `GET` | `/api/health` | — | `{status:"ok", chunk_count:int, model:str, embedding_model:str}` | `controller.chunk_count()` + `config`/`embeddings.get_active_model_name()` |
| `POST` | `/api/chat` | `{text:str, session_id:str}` | **SSE** `text/event-stream`: `event: token` `data:<delta>` · `event: step` `data:{name,status}` · `event: result` `data:<TurnResult JSON>` · `event: done` | `controller.handle_message(session, text)` |
| `POST` | `/api/claims/{claim_id}/adjudicate` | `{decision:"accepted"|"rejected"|"edited", edited_text?:str}` | `{ok:true}` | `controller.adjudicate(...)` |
| `POST` | `/api/export` | `{session_id:str, dev:bool}` | file stream (`text/markdown`, attachment) | `controller.export_conversation(session, dev=dev)` |
| `GET` | `/api/figures/{figure_id}` | — | `image/png` bytes | `figures.load_figure_image_paths([figure_id])` |
| `GET` | `/api/source/{record_id}/{n}` | — | `SourceView` JSON (+ PDF coords later) | provenance read (record → nth source) |
| `GET` | `/api/settings` | — | locked-settings view JSON | `config` (read) |
| `POST` | `/api/settings` | `{<env-toggleable knobs>}` | `{ok}` or `501` if not wired | `config` (env-toggleable only) |

**SSE handler shape (reference):**
```python
async def chat(req: ChatRequest):
    session = sessions.get_or_create(req.session_id)
    def gen():
        for ev in controller.handle_message(session, req.text):
            match ev:
                case Token(text):  yield sse("token", text)
                case Step(n, s):   yield sse("step", json({"name": n, "status": s}))
                case Result(r):    yield sse("result", json(as_payload(r)))
        yield sse("done", "")
    return EventSourceResponse(gen())   # sse-starlette; or StreamingResponse(media_type="text/event-stream")
```
(`controller.handle_message` is sync/generator per PR-M0; run it in a threadpool if it blocks the event loop — `sse-starlette` + `anyio.to_thread` or an async wrapper. Note the choice in the docstring.)

## Contract — `apps/api/models.py` (new)
Pydantic request/response schemas mirroring PR-M0 dataclasses: `ChatRequest{text, session_id}`, `AdjudicateRequest{decision, edited_text?}`, `ExportRequest{session_id, dev}`, and `TurnResultPayload` / `SourceViewPayload` / `ClaimViewPayload` / `UsageViewPayload` mirroring the controller's value objects (include `SourceView.markers` if PR-M1 landed first). `Literal` types for `decision` and `mode`.

## Contract — `apps/api/sessions.py` (new)
`get_or_create(session_id) -> Session` over a module `dict[str, Session]`; `get(session_id) -> Session | None` (→ `404` mapping for claims/export). Docstring states the single-user / process-scoped / no-eviction assumption (ADR-3).

## Contract — `pyproject.toml` (edit)
Add `fastapi`, `uvicorn[standard]`, `sse-starlette` (or justify hand-rolled SSE and drop `sse-starlette`). Keep them in the main deps (the sidecar needs them); no torch interaction. Regenerate `uv.lock`. Add a `justfile` target `api` → `uv run --extra <torch-extra> uvicorn apps.api.main:app --host 127.0.0.1 --port <p>`.

## SourceAdapter seam (note, do not build)
The ingestion-source pluggability (`SourceAdapter`, per memory `doc-assistant-source-agnostic-companion`) lands **behind this API** later. Mark the insertion point in `main.py`'s module docstring (e.g. "future `/api/sources` + adapter registry mounts here"). **Do not** add an interface now — there is no second concrete source yet (cpc "no speculative abstraction": the seam is a comment, not an abstraction).

---

## Build node
**Depends on:** PR-M0 (`ChatController`, `TurnResult`, `TurnEvent`). Best after PR-M1 (so `markers` is in the payload). Independent of torch/GPU.
**Files owned:**
- `apps/api/__init__.py`, `apps/api/main.py`, `apps/api/models.py`, `apps/api/sessions.py` (all new)
- `pyproject.toml` (+ `fastapi`/`uvicorn`/`sse-starlette`), `uv.lock` (regenerated), `justfile` (`api` target)
- `tests/unit/test_api_models.py` (new), `tests/integration/test_api_chat_sse.py` (new)
- `docs/decisions.md` (or ADR-NNN: API boundary + SSE), `docs/architecture.md` (module table: `apps/api` = HTTP renderer), `.claude/CONTEXT.md` (stack: add FastAPI alongside Chainlit during migration), one `docs/DEVLOG.md` entry per logical change.

### Unit test — `tests/unit/test_api_models.py`
- `TurnResult` (with sources, flagged claims, usage, markdown blocks) round-trips through `TurnResultPayload` with no field loss; `Literal` validation rejects a bad `decision`/`mode`.

### Integration test (CI gate) — `tests/integration/test_api_chat_sse.py`
Use FastAPI's `TestClient` + a **fake `ChatController`** yielding canned `Token`/`Step`/`Result` events (no real pipeline, no LLM, no network — cpc §13):
- `GET /api/health` → `200` + the shape.
- `POST /api/chat` → an SSE stream whose events are, in order, ≥1 `token`, optional `step`s, exactly one `result` (valid `TurnResult` JSON), then `done`.
- `POST /api/claims/{id}/adjudicate` → calls the fake's `adjudicate` with the right args; bad `decision` → `422`.
- Unknown `session_id` on `/export` → `404`; on `/chat` → creates a session (`200`/stream).
- `GET /api/figures/{id}` with a stubbed path → `200 image/png`; missing → `404`.
- Deterministic; no paid call.

## Definition of done
- `apps/api/` serves the endpoint table; **no `chainlit`, no business logic** in `apps/api/`; one `ChatController` per process.
- `/api/chat` streams SSE (`token`/`step`/`result`/`done`) mapping `TurnEvent` 1:1; actions are JSON POST; figures served over HTTP (no filesystem path crosses the boundary); bind `127.0.0.1` only.
- **Chainlit + CLI still work unchanged** (this PR adds a frontend, removes none).
- Unit + integration tests green (fake controller, no paid calls); `ruff` / `mypy --strict src` (extend to `apps/api` if the gate covers it) / `bandit` clean; coverage floor held.
- Runs in dev via the `just api` target; SSE verified streaming via browser/`curl` (manual smoke noted in the DEVLOG).
- `decisions.md`/ADR + `architecture.md` + `CONTEXT.md` updated; one `DEVLOG.md` entry per logical change. **Stage + summarize the diff; do not commit/push without review** (cpc §13).

## Out of scope (later PRs)
- **Tauri shell + frontend** consuming this API, the five-primitive component mapping, the rich per-claim editorial GUI, styled tables — **PR-M3**.
- **PyInstaller sidecar packaging**, the frozen CPU-torch pin (KI-3), data-dir relocation, the clean-machine smoke + latency measurement — **PR-M4**.
- **Deleting Chainlit** + lifting the Python-3.12 pin (KI-2) — **PR-M5**.
- **`SourceAdapter` registry / `/api/sources`** — deferred until a second concrete ingestion source exists (seam noted, not built).
- **Full settings write-path** (Phase 8) — only the read surface + env-toggleable knobs here.
- **Persisted / reloadable sessions** — stays the deferred item it already is.
