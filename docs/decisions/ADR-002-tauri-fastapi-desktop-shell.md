<!-- status: active · updated: 2026-06-21 · class: append-only -->

# ADR-002 — Desktop shell: Tauri frontend + FastAPI Python backend (replace Chainlit)

- **Status:** proposed (2026-06-21 — drafted with Cowork; specs M0/M1/M2 written, build not started)
- **Date:** 2026-06-21
- **Deciders:** Lucas (drafted with Cowork)

> This ADR records the decision to migrate the UI from Chainlit to a Tauri desktop app with a FastAPI
> backend, and is the **why** the migration specs (`docs/specs/pr-m0-chat-controller.md`,
> `pr-m1-epistemics-markers.md`, `pr-m2-fastapi-boundary.md`) build against. It supersedes the UI
> half of the stack table in `.claude/CONTEXT.md` ("UI = Chainlit (web)"). It relates to the Chunk 2a
> ADR (`docs/specs/chunk-2a-dual-interpretation.md`), which deferred the rich per-claim editorial GUI
> to "the Phase 8 framework decision" — i.e. this one.

## Context

The UI is Chainlit (a localhost web server) plus a CLI, both thin shells over `src/doc_assistant/`
(rule #3 in `.claude/CONTEXT.md`: `apps/` carry no business logic; the library is UI-framework-agnostic).
A second frontend (the CLI) already proves the core runs headless. **All Chainlit coupling lives in one
608-line file** (`apps/chainlit_app.py`; 51 `cl.*` calls, all presentation). Three pressures converge:

1. **Product shape.** The goal (`.claude/CONTEXT.md`) is a **local-first personal research assistant** —
   a desktop companion, not a hosted web app. Chainlit optimizes for the latter. The user's stated
   direction (auto-memory `doc-assistant-source-agnostic-companion`) is a desktop companion with a
   pluggable source layer.
2. **Bespoke integrity UX.** The differentiators — inline citations, the provenance card, the
   evidence↔interpretation split, per-claim accept/reject/**edit**, reviewer chips, the 7d
   contested/superseded markers (PR-M1), figures, and an eventual in-app PDF source viewer — exceed
   what Chainlit renders cleanly. **The Chunk 2a ADR already rejected building the rich per-claim GUI
   in Chainlit** ("exceeds Chainlit's clean limits") and deferred it to this decision.
3. **Maintenance signal.** Chainlit's founding team stepped back (May 2025); it is community-maintained
   (≈10 maintainers, regular releases) with two high-severity CVEs found late 2025. Viable, but no
   longer venture-backed — a reason to own the frontend, not an emergency. (KI-2 also pins the runtime
   to Python 3.12 *because of* Chainlit; that constraint dissolves once Chainlit is gone.)

### Verified state at time of writing (2026-06-21)

| Area | Current | Migration target | Note |
|---|---|---|---|
| UI coupling | one file, `apps/chainlit_app.py` (608 LOC) | `ChatController` + `TurnResult` in `src/`; `apps/*` = thin renderers | PR-M0 |
| Turn orchestration | trapped in `cl.on_message` | UI-agnostic library service yielding `TurnEvent` | PR-M0 |
| Streaming | Chainlit `msg.stream_token` | SSE `event: token` over FastAPI | PR-M2 (ADR-2 below) |
| Backend↔frontend | in-process (Chainlit) | FastAPI over `127.0.0.1`; Tauri webview frontend | PR-M2/M3 |
| Packaging | `uv run chainlit` | Tauri installer + PyInstaller sidecar | PR-M4 |
| Python runtime pin | 3.12 (Chainlit/anyio, KI-2) | liftable once Chainlit deleted | PR-M5 |

## Decision

Adopt a **Tauri desktop application** with a **FastAPI backend** wrapping the existing Python core:

- **Backend** — a thin FastAPI app (`apps/api/`) exposing the RAG/integrity pipeline over HTTP. It owns
  no new logic; it calls a new `ChatController` (extracted from `chainlit_app.py` in PR-M0). FastAPI is
  *just another shell*, preserving rule #3.
- **Streaming (see ADR-2 below)** — **SSE** for token streaming (one-directional; maps 1:1 onto
  `pipeline.stream_answer`'s generator and PR-M0's `TurnEvent`). Interactive actions (claim
  accept/reject/edit, export, settings) are plain **POST**. **WebSocket rejected** — full-duplex is
  unneeded; the only server→client push is tokens, which SSE handles, and SSE is far simpler to bundle.
- **Packaging (see ADR-3 below)** — Tauri spawns the bundled Python/FastAPI process as a **sidecar**
  (PyInstaller-frozen); one native installer per OS. The **dev loop** runs FastAPI via `uv` and points
  Tauri at `localhost` (no freeze in the inner loop). Sidecar is the *release* target, separate-process
  is the *dev* target.
- **Frontend** — an owned web UI inside Tauri's native webview. Framework (React / Svelte / vanilla) is
  a sub-decision deferred to PR-M3; it does not affect this ADR.

## Options considered

1. **Tauri + FastAPI sidecar (chosen).** True desktop, native installer, ~3–10 MB shell, 20–40 MB RAM.
   Keeps the Python core untouched behind an API — also the natural seam for the future `SourceAdapter`
   and an outbound MCP server (a roadmap "later/open" item). Cost: a frontend to own; Python-in-Tauri
   packaging (PyInstaller + sidecar lifecycle) is the real tax. Accepted as the only option delivering
   desktop **and** full control of the integrity UX.
2. **Electron + FastAPI.** Same architecture; 150 MB+ shell, 200–400 MB RAM, ships Chromium. Rejected —
   heavier on every axis for a single-user local tool; Tauri's JS APIs cover the need without mandating Rust.
3. **NiceGUI / Flet (Python-native desktop).** No JS codebase; UI in Python; desktop target. **Rejected
   by the user** — too basic for the bespoke citation/provenance/PDF-verification surface; componented
   panels can't match owned HTML/CSS for the source viewer.
4. **Gradio / Streamlit (stay web).** Lowest effort; Gradio is what Kotaemon (closest analog) uses, with
   an in-browser PDF citation viewer. **Rejected by the user** — basic, and not a true desktop app
   (localhost web, same shape as Chainlit).
5. **Keep Chainlit backend, swap only the face (`@chainlit/react-client`).** Reuses event wiring, hedges
   maintenance partway. Rejected — inherits Chainlit's protocol *and* its maintenance risk while still
   requiring a full frontend; most of option 1's cost without its clean break.

### ADR-2 (sub-decision) — SSE over WebSocket
The only server→client push in a turn is the token/step stream; everything else is request→response.
SSE maps 1:1 onto PR-M0's `TurnEvent` (`token`/`step`/`result`), survives the Tauri webview and the
sidecar boundary cleanly, and is materially simpler to bundle and debug than a WS lifecycle. Full detail
+ the endpoint contract: `docs/specs/pr-m2-fastapi-boundary.md` ADR-2.

### ADR-3 (sub-decision) — Sidecar for release, separate-process for dev
PyInstaller-freezing the Python stack is the migration's hardest part; forcing it into the dev inner
loop would slow every iteration. So: **release** = Tauri bundles the frozen FastAPI binary as a sidecar
(one installer); **dev** = run FastAPI under `uv`, Tauri connects to `localhost`. Full detail: PR-M4.

## Consequences

**Positive.**
- Native desktop companion; one installer; offline-first.
- Full control of the integrity-layer UX; the rich per-claim editorial GUI (parked by the 2a ADR) can
  finally be built; tables render styled; an in-app PDF source viewer becomes possible.
- The FastAPI boundary is the clean insertion point for `SourceAdapter` and a future outbound MCP server.
- Python core, eval harness, CLI, and the 7d engine are untouched — the migration is additive, then a
  one-file deletion (PR-M5).

**Negative / costs.**
- **Packaging is the real tax.** PyInstaller-freezing torch + Chroma + PyMuPDF + the per-machine
  `cu130`/`cpu` extra split, plus the Tauri sidecar lifecycle, is the hard part. **KI-3** (the win32
  `cu130` segfault on a GPU-less box) means the **frozen build must pin the CPU torch backend** by
  default — do **not** bundle `cu130`; GPU users keep the `uv` dev path or a separate build. (PR-M4.)
- A frontend codebase to own and maintain (new skill surface for the project).
- **KI-2 dissolves** once Chainlit is gone (the Python-3.12 pin can lift, pending no other 3.12-only dep —
  verify in PR-M5). Until then the dev runtime stays 3.12.
- Two-process model: the app manages backend startup/health/shutdown (sidecar readiness gate). (PR-M4.)

**Rigor gates before "done" (RIGOR_TODO):**
- Cold-start time with model load; SSE first-token latency must not regress vs Chainlit (measure on the
  frozen build, PR-M4; smoke in dev, PR-M2).
- Frozen-build smoke on a clean machine (no Python installed) — the portability gate (PR-M4).
- Parity: CLI, (interim) Chainlit, and Tauri all render the same `TurnResult` (guard test, PR-M0).

## Execution

Six PRs, one per session, in `docs/ROADMAP.md` (M0–M5). Specs written: M0
(`pr-m0-chat-controller.md`), M1 (`pr-m1-epistemics-markers.md`), M2 (`pr-m2-fastapi-boundary.md`).
M3–M5 specced one ahead as each predecessor lands (each depends on the prior's output: frontend
framework, freeze layout). Sequencing rationale (why M1 lands before the migration proper, and why the
"sexy" UX is built natively in Tauri rather than in Chainlit): `tauri-migration-plan.md` Part 1.

Build order: `M0 → M1 → M2 → M3 → M4 → M5`. M0 is the only hard blocker for everything; M1 (the 7d
marker chip) is the pre-migration demo win and shares M0's `chunk_key` plumbing.
