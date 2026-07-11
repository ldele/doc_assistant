# Spec — PR-M4: PyInstaller sidecar packaging (the desktop installer)

> **📦 Archived — mostly shipped; the historical code-level contract, archived here (2026-07-11).** Live status: ROADMAP row M4 — installer built, RG-010/011/012-Tier-1 pass; **RG-012 Tier-2** (a cited turn on a clean/frozen box) still pends a re-freeze + clean-box run (also in `docs/ui-checklist.md` §2). The behaviour of record is the code + runbook `docs/desktop-packaging.md`, not this spec.

**Status:** 🔧 SCAFFOLDED — specced + built by Claude Code 2026-06-22 (Tauri migration, `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`). Fifth PR of the migration (M4). **Depends on PR-M2** (the backend it freezes) + **PR-M3** (the shell it bundles).
**The verifiable parts are built + green; the freeze + native build + clean-machine smoke are desktop steps (RG-010/011/012).** Not "done" the way M0–M3 are — see *Verification boundary*.
**Owner of execution:** Claude Code (scaffold) → the user / a desktop (freeze + build + smoke).

**Requirement (the why).** ADR-002 ships the desktop app as one native installer: Tauri bundles the frozen Python/FastAPI backend as a **sidecar** (PyInstaller), spawned on startup. This PR builds the packaging machinery — the server entrypoint, the freeze recipe, the build orchestrator, the Tauri sidecar wiring, the readiness gate — so the installer can be produced.

**Cost & placement.** Packaging is the migration's hardest tax (ADR-002): freezing torch + Chroma + PyMuPDF (~GB binary) + the Tauri sidecar lifecycle. New `scripts/build_sidecar.py` + `scripts/doc_assistant_api.spec` + `apps/api/__main__.py`; Tauri wiring in `apps/desktop/src-tauri/`; a `packaging` extra (PyInstaller). No change to the Python core.

---

## ADR-1 — Frozen sidecar, spawned by Tauri; lazy backend + frontend readiness gate

**Context.** The backend loads bge embeddings + the reranker + Chroma before it can answer — seconds of warm-up. The webview must not block on it.

**Decision.** `apps/api/__main__.py` runs `uvicorn` over the M2 app (`127.0.0.1`, env-overridable port). PyInstaller freezes it (`scripts/doc_assistant_api.spec`). Tauri spawns it on startup (`src-tauri/src/lib.rs` via `tauri-plugin-shell` `.sidecar()`; `bundle.externalBin` in `tauri.conf.json`; scoped `shell:allow-execute` capability). The `ChatController` is built **lazily in the app lifespan** (already so since M2), so the process starts immediately; the **frontend polls `/api/health` until it warms** (the readiness gate, added to `App.svelte`: `connecting → ready/down`). A missing sidecar is non-fatal (dev mode → run `just api`).

**Options considered.** (1) *Frozen sidecar + lazy backend + frontend poll (chosen).* (2) *Block the window on a Rust-side health wait* — worse UX (frozen window), duplicates the readiness logic the frontend already needs. (3) *Embed Python via PyO3 instead of a sidecar* — couples the Rust shell to the Python ABI; the sidecar keeps the clean process boundary M2 established.

## ADR-2 — onefile sidecar; CPU-torch pinned (KI-3)

**Context.** Tauri's `externalBin` is a single binary per target triple. The frozen build must run on any machine.

**Decision.** **onefile** PyInstaller build (one binary, Tauri's model) — accepting slower cold-start (it unpacks to temp on launch; measured against RG-010/011). **CPU torch only**: the `cu130` wheel **segfaults (exit 139) on a GPU-less box** (KI-3), so `build_sidecar.py` **refuses** to freeze a `+cu*` torch; GPU users keep the `uv` dev path. If onefile cold-start fails the latency gate, the documented fallback is onedir + shipping `_internal/` as a Tauri resource.

---

## Decisions

| # | Decision |
|---|---|
| 1 | **`apps/api/__main__.py`** — `python -m apps.api` (dev) + the PyInstaller entry. `uvicorn.run(app, 127.0.0.1, $DOC_API_PORT\|8001)`. |
| 2 | **`scripts/doc_assistant_api.spec`** — onefile; `collect_all` for torch/chroma/sentence-transformers/transformers/tokenizers/pymupdf/langchain*/duckdb; `collect_submodules` for `doc_assistant`/`apps` + uvicorn's string-loaded impls. A **starting point** — ML freezes need on-machine hidden-import/data fixes. |
| 3 | **`scripts/build_sidecar.py`** — `--check` (triple + CPU-torch guard + entry import, **no freeze**) and full build (PyInstaller → copy to `src-tauri/binaries/doc-assistant-api-<triple>[.exe]`). |
| 4 | **Tauri wiring** — `tauri.conf.json` `externalBin`; `src-tauri/src/lib.rs` spawns the sidecar + drains stderr; `capabilities/default.json` scoped `shell:allow-execute`. |
| 5 | **`packaging` extra** (`pyinstaller>=6.0`) — kept out of `dev` so the normal install stays lean. `just sidecar` / `just sidecar-check`. |
| 6 | **Readiness gate** — `App.svelte` polls `/api/health` (≤60×1s): `starting the engine… → ready / unreachable`. |
| 7 | **Icons + installer** — `npx tauri icon` then `npx tauri build`; runbook `docs/desktop-packaging.md`. |

---

## Verification boundary (what's green vs deferred)

**Built + verified in-repo:** `python -m apps.api` imports clean (8 routes, controller lazy); `just sidecar-check` passes (triple `x86_64-pc-windows-msvc`, torch `+cpu` guard, entry import); the frontend builds with the readiness gate; Python gate green (ruff / mypy --strict src / bandit / **590 passed**).

**Deferred to a desktop (can't run in CI/sandbox — Tauri/Rust toolchain + a clean machine):**
- **The PyInstaller freeze** — the spec is a starting point; expect hidden-import/data iteration (RG-012).
- **`npx tauri build`** — Rust crate downloads + native bundle + icons.
- **Rigor gates:** cold-start (RG-010), SSE first-token latency vs Chainlit (RG-011, blocks-ship), clean-machine smoke (RG-012, blocks-ship).

## Build node
**Files owned:** `apps/api/__main__.py`; `scripts/build_sidecar.py`, `scripts/doc_assistant_api.spec`; `apps/desktop/src-tauri/{tauri.conf.json, src/lib.rs, capabilities/default.json}`, `apps/desktop/.gitignore`, `apps/desktop/src/App.svelte` (readiness gate); `pyproject.toml` (`packaging` extra) + `uv.lock`; `Justfile` (`sidecar`/`sidecar-check`); `docs/desktop-packaging.md`; `.claude/RIGOR_TODO.md` (RG-010/011/012); docs.

## Definition of done
- **Met:** server entrypoint + readiness gate + Tauri sidecar wiring built; `just sidecar-check` green; CPU-torch pin enforced in the build script (KI-3); runbook + rigor gates recorded; Python + frontend gates green.
- **Open (on a desktop):** a working frozen sidecar, a built installer, and the three rigor gates closed (RG-010/011/012) — RG-011 + RG-012 **block the M4 ship**. Then PR-M5 deletes Chainlit + lifts the 3.12 pin.
- **Stage + summarize; do not commit without review** (cpc §13).

## Out of scope
- Deleting Chainlit + lifting the Python-3.12 pin (KI-2) — **PR-M5**.
- Auto-update / code-signing / notarization — later, once the installer ships.
