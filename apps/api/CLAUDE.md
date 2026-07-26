# apps/api/ — FastAPI/SSE boundary (thin shell)

**Owns:** the HTTP/SSE mapping over `doc_assistant.chat_controller.ChatController` — and nothing
else. Endpoints own no retrieval/provenance/claim logic (non-negotiable #3).

**Key files**
- `main.py` — `create_app(...)` only: lifespan (schema migration + controller), `app.state` wiring +
  test seams, CORS, `include_router`. Re-exports `_settings_view`/`_default_rebuild_graph` for tests.
- `routers/*` — one `APIRouter` per domain: `health` · `chat` (SSE `token`/`step`/`result`/`done`,
  compare, claims, export, source/figure) · `conversations` · `library/` · `concepts` · `taxonomy` ·
  `settings` · `sources`. Handlers read state via `request.app.state`; none import from `main`.
  `library/` is a package — `documents` · `folders` · `keywords` — composed in its `__init__`.
- `services.py` — cross-router glue: `app.state` status dataclasses + their `202+poll` serializers,
  the settings read view, the lazy default job runners. Imported by `main` and routers, neither back.
- `models/` — Pydantic wire models, one module per domain **named to match `routers/`** (the
  contract `apps/desktop/src/lib/core/types.ts` mirrors). Import from the domain module
  (`from apps.api.models.folders import …`); the flat re-export in `__init__` is back-compat only.
  The end-to-end domain table is in `docs/architecture.md` § `apps/` — the domain spine.
- `sessions.py` — per-session state (RAG overrides live per session, never module-global).
- `__main__.py` — frozen-runtime entry: truststore/OS-trust (KI-10), HF offline, data-home resolve.

**Rules that bite here**
- Tests inject a fake controller via `create_app(controller=...)` — no model load, no network,
  no paid calls (cpc §13). `create_app` is the public seam; don't add module-level route state.
- A new route goes in its domain router (new domain ⇒ new `routers/` module + `include_router`).
  Keep route-match order *within* a router (e.g. `/api/concepts/gaps` before `{concept_id}/…`).
- `handle_message` is a sync blocking generator bridged to the loop in `routers/chat._event_stream`
  — single-user desktop design; a multi-client server would need `anyio.to_thread` (not built).
- The API **does** `init_db()` in the lifespan (KI-23) — idempotent + additive. A **failed**
  migration fails the boot (E0.5a), never a swallow: a half-migrated answer-path schema would 500
  every turn, so refusing to start is the honest failure.
- Adding/changing a route ⇒ update `models/<domain>.py` + `core/types.ts` + the tests together.
  A test that monkeypatches a route's dependency patches it on **its router module**, not `main`.

**Tests:** `tests/integration/test_api_*.py`.

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->
