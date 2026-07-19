# apps/api/ — FastAPI/SSE boundary (thin shell)

**Owns:** the HTTP/SSE mapping over `doc_assistant.chat_controller.ChatController` — and nothing
else. Endpoints own no retrieval/provenance/claim logic (non-negotiable #3).

**Key files**
- `main.py` — `create_app(controller=...)`; one `ChatController` per process (lifespan);
  `/api/chat` streams SSE (`token`/`step`/`result`/`done`); library/concepts/settings/ingest routes.
- `models.py` — Pydantic wire models (the contract `apps/desktop/src/lib/types.ts` mirrors).
- `sessions.py` — per-session state (RAG overrides live per session, never module-global).
- `__main__.py` — frozen-runtime entry: truststore/OS-trust (KI-10), HF offline, data-home resolve.

**Rules that bite here**
- Tests inject a fake controller via `create_app(controller=...)` — no model load, no network,
  no paid calls (cpc §13).
- `handle_message` is a sync blocking generator iterated on the event loop — single-user desktop
  design; a multi-client server would need `anyio.to_thread` (documented, not built).
- The API does **not** `init_db()` on startup — a stale DB 500s until an ingest or a manual
  `python -m doc_assistant.db.migrations` runs (known gap; see DEVLOG S2 entry).
- Adding/changing a route ⇒ update `models.py` + `types.ts` + the integration tests together.

**Tests:** `tests/integration/test_api_*.py`.

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->
