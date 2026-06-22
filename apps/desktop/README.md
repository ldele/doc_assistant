# doc_assistant — desktop frontend (PR-M3)

A **thin** Svelte 5 + Vite UI inside a Tauri 2 shell. It renders the API's `TurnResult`
payload and holds **no business logic** — all logic stays in `src/doc_assistant/` behind
the FastAPI/SSE boundary (`apps/api/`, PR-M2). Decision + rationale:
`docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md` and `docs/specs/pr-m3-tauri-frontend.md`.

## What it renders

- Streaming chat (POST-SSE → token-by-token), inline-citation markdown answers.
- Source cards with the 7d **contested / superseded** marker chips (PR-M1).
- The per-claim **accept / reject / edit** review GUI (the editorial UX deferred by the
  Chunk 2a ADR — built natively here, not in Chainlit).
- A collapsible provenance + usage card; figures served over `GET /api/figures/{id}`.

## Dev loop (two processes)

```bash
# 1. the backend (from the repo root)
just api                       # uvicorn apps.api.main:app on 127.0.0.1:8001

# 2. the frontend (from apps/desktop/)
npm install                    # first time
npm run dev                    # vite on :1420, proxies /api → :8001
```

Open http://localhost:1420. In dev, Vite proxies `/api` to the backend (same-origin, no
CORS). The packaged build hits the absolute backend URL (the API's CORS allowlist includes
`tauri://localhost`).

- `npm run build` — type-check (`svelte-check`) + production bundle to `dist/`.
- `npm run check` — type-check only.

## Tauri shell (M3 scaffold; built in M4)

`src-tauri/` is the Tauri 2 project that wraps the frontend (`devUrl` → the Vite server;
`frontendDist` → `../dist`). Running it needs the Tauri CLI and the Rust toolchain:

```bash
npm install -g @tauri-apps/cli   # M4
npx tauri icon path/to/icon.png  # M4 — generates src-tauri/icons/* (referenced by tauri.conf.json)
npx tauri dev                    # native window over the dev server
```

**Not built in M3** (deferred to PR-M4): the app icons, the PyInstaller sidecar that
freezes + spawns the FastAPI backend, and the installer. M3 ships + verifies the web
frontend (svelte-check + vite build + a browser-driven run against the API); the native
Tauri build is exercised in M4 on a desktop.
