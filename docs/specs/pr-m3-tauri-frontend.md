# Spec — PR-M3: Tauri desktop frontend (Svelte) over the FastAPI/SSE boundary

**Status:** ✅ BUILT — specced one-ahead + built by Claude Code 2026-06-22 (Tauri migration, `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`). Fourth PR of the migration (M3). Framework decision (Svelte) made with the user this session. **Depends on PR-M2** (the API contract it consumes).
**Owner of execution:** Claude Code (frontend + Tauri shell scaffold).
**Pattern reference:** thin-shell rule (`apps/` carry no logic; `.claude/CONTEXT.md` rule #3). The frontend is a *renderer* of the API's `TurnResult` — exactly like `apps/cli.py` / `apps/chainlit_app.py` / `apps/api/`, but in TypeScript over HTTP.

**Requirement (the why).** ADR-002 chose a Tauri desktop app to gain full control of the integrity UX — inline citations, the provenance card, the evidence↔interpretation split, per-claim accept/reject/**edit**, the 7d contested/superseded markers, figures — that **exceeded what Chainlit renders cleanly** (the Chunk 2a ADR explicitly deferred the rich per-claim GUI to this decision). M3 builds that owned web UI inside Tauri's webview, consuming the PR-M2 HTTP/SSE contract. The framework was the one sub-decision ADR-002 deferred to M3.

**Cost & placement.** New `apps/desktop/` (Node/Vite frontend + a Tauri 2 Rust shell). No change to `src/` except one additive field (ADR-3). The frontend is verifiable in a browser; the native Tauri build (Rust + Tauri CLI + crate downloads + a native window) is exercised in **PR-M4** on a desktop.

---

## ADR-1 — Framework: Svelte 5 + Vite (vanilla-feel, lean bundle)

**Context.** ADR-002 left the framework (React / Svelte / vanilla) to M3. The project values leanness (it chose Tauri over Electron precisely for lightness, and rejected Gradio/Streamlit/NiceGUI/Flet as too basic) while wanting an *owned, rich* surface (the citation/provenance/PDF-verification UX).

**Decision.** **Svelte 5 + Vite + TypeScript.** Compiles to small vanilla JS (no virtual DOM; the production bundle is ~29 KB gzipped incl. the markdown renderer), minimal boilerplate, owned HTML/CSS — the sweet spot of rich-but-lean for a single-user local tool that drops into Tauri's webview unchanged. Chosen by the user.

**Options considered.** (1) *Svelte (chosen).* (2) *React* — mainstream/largest ecosystem but heavier bundle + more boilerplate; rejected as heavier than needed. (3) *Vanilla (no build)* — zero deps + most trivially verifiable, but gets verbose as the UI grows (PDF viewer, settings, wiki views); rejected for ergonomics.

## ADR-2 — POST-SSE consumed by hand (not EventSource)

**Context.** `POST /api/chat` streams SSE; `EventSource` is GET-only.

**Decision.** The API client (`src/lib/api.ts`) issues `fetch` and parses the `ReadableStream` body by hand (`\n\n`-delimited events → `{event, data}`), yielding an async generator the chat loop drains. Tokens append to the streaming answer; the `result` event's JSON becomes the `TurnResult`. No business logic — fetch + parse only.

## ADR-3 — `SourceView.figure_id` (an id crosses the boundary, never a server path)

**Context.** M2 ADR-1 said "no filesystem path crosses the boundary" and serves figures via `GET /api/figures/{figure_id}`. But `SourceView` only carried `figure_path` (a server path the browser can't use).

**Decision.** Add `figure_id: str | None` to `SourceView` (additive; `chat_controller.py`), populated from the chunk metadata. The API payload exposes `figure_id` (dropped `figure_path` from `SourceViewPayload`); the frontend renders it via `/api/figures/{figure_id}`. `figure_path` stays on `SourceView` for the Chainlit local `cl.Image` render.

---

## Decisions

| # | Decision |
|---|---|
| 1 | **New `apps/desktop/`**: Svelte UI (`src/`) + Tauri 2 shell (`src-tauri/`). Thin renderer — no business logic; talks only HTTP. |
| 2 | **Components:** `App.svelte` (health header, conversation, streaming send loop, export) → `Turn.svelte` → `SourceCard.svelte` (citation + marker chips + figure), `ClaimReview.svelte` (accept/reject/edit, per-claim state, calls `/adjudicate`), `Provenance.svelte` (collapsible card + usage), `Markdown.svelte` (`marked`). |
| 3 | **Dev wiring:** Vite on `:1420` proxies `/api` → `127.0.0.1:8001` (same-origin in dev, no CORS). Packaged build uses the absolute backend URL (API CORS allows `tauri://localhost`). `import.meta.env.DEV` switches the base. |
| 4 | **Tauri 2 shell** (`src-tauri/`): `tauri.conf.json` (`devUrl` → Vite, `frontendDist` → `../dist`, CSP allows `127.0.0.1:8001`), `Cargo.toml`, `build.rs`, `src/{main,lib}.rs` (+ `tauri-plugin-shell` for the M4 sidecar), `capabilities/default.json`. |
| 5 | **`SourceView.figure_id`** (ADR-3) added + exposed in the payload; `figure_path` no longer crosses the API. |

**Out of scope / deferred to M4:** app icons (`tauri icon`), the PyInstaller sidecar that freezes + spawns the backend, the installer, the clean-machine smoke + SSE first-token latency measurement on the frozen build. The in-app PDF source viewer + styled tables are later refinements.

---

## Build node
**Files owned:** `apps/desktop/**` (package.json, vite/ts/svelte config, `src/**` Svelte+TS, `src-tauri/**`, README, .gitignore); `src/doc_assistant/chat_controller.py` (`SourceView.figure_id`); `apps/api/models.py` (`SourceViewPayload.figure_id`); docs.

**Verification (this session).** `npm run build` → `svelte-check` **0 errors** + `vite build` (bundle 81.98 KB / **28.78 KB gzipped**). Browser-driven run against the API with a fake controller (canned SSE, no models/LLM/paid call): health renders (2,455 chunks · model · embedding); a turn **streams token-by-token**; the result renders the **markdown answer**, **2 source cards with `⚠ contested in corpus` / `⚠ trend superseded` chips**, the **flagged-claim accept/reject/edit GUI**, and the **provenance card**; clicking **Accept** POSTs `/api/claims/{id}/adjudicate` (200) and the claim resolves to `✓ accepted`. Backend request log confirmed the `/chat` + `/adjudicate` hits. Python gate after the `figure_id` change: ruff/format/mypy/bandit clean, **590 passed**. The native `tauri build` is **not** exercised here (Rust/Tauri toolchain + native window) — PR-M4.

## Definition of done
- `apps/desktop/` renders the full turn (streaming answer, source cards + 7d chips, per-claim accept/reject/edit, provenance, figures, export) over the M2 API; **no business logic** in the frontend.
- Frontend **type-checks + builds** clean; browser-driven run verified against the API (fake controller — no paid call, cpc §13).
- `SourceView.figure_id` added; **no filesystem path crosses the API** (`figure_path` dropped from the payload).
- Tauri 2 shell scaffolded (dev wiring complete); icons + sidecar + installer are PR-M4.
- Python gate green (ruff / mypy --strict src / bandit / pytest); docs updated. **Stage + summarize; do not commit without review** (cpc §13).
