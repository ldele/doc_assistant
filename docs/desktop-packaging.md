<!-- status: active · updated: 2026-06-22 · class: runbook -->

# Desktop packaging runbook (PR-M4)

How to build the installable Tauri desktop app: freeze the FastAPI backend as a PyInstaller
**sidecar**, bundle it with the Svelte frontend, produce a native installer. ADR + rationale:
`docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`; spec: `docs/specs/pr-m4-sidecar-packaging.md`.

> **Runs on a desktop, not in CI / the sandbox.** Freezing torch + Chroma + PyMuPDF (~GB
> binary), `cargo tauri build`, and the clean-machine smoke need the Tauri/Rust toolchain and
> a real machine. What *is* verified in-repo: the server entrypoint (`python -m apps.api`), the
> build-prereq check (`just sidecar-check`), and the frontend build. The freeze/build/smoke are
> the steps below.

## 0. One-time toolchain

- Rust + Cargo (`rustup`), Node ≥ 20.
- The project venv synced **CPU + the packaging extra**: `uv sync --extra cpu --extra dev --extra packaging`
  (PyInstaller lives in the `packaging` extra, kept out of the lean `dev` install).
- `npm install -g @tauri-apps/cli` (or use `npx tauri`).

## 1. CPU-torch pin (KI-3 — non-negotiable)

The frozen build **must** use the CPU torch wheel. The `cu130` wheel **segfaults (exit 139) on a
GPU-less box**, and the installer must run anywhere. `scripts/build_sidecar.py` refuses to freeze a
`+cu*` torch. On the GPU box, re-sync CPU first:

```bash
uv sync --extra cpu --extra dev
just sidecar-check      # asserts triple + CPU torch + entrypoint import
```

## 2. Freeze the FastAPI sidecar

```bash
just sidecar            # = python -m scripts.build_sidecar  → PyInstaller (slow, large)
```

Produces a single binary and copies it to `apps/desktop/src-tauri/binaries/doc-assistant-api-<target-triple>[.exe]`
(Tauri's sidecar naming). The spec (`scripts/doc_assistant_api.spec`) is a **starting point** —
freezing ML libs always needs a few rounds of fixes: run the produced binary, read the
`ModuleNotFoundError` / missing-data-file traceback, add to `hiddenimports` / `datas` in the spec,
repeat. Known suspects: `torch` dynamic ops, `chromadb` + its `onnxruntime`/`hnswlib` data,
`sentence_transformers`/`transformers` model configs, `tokenizers` native libs, `pymupdf` (`fitz`) DLLs.

Smoke the frozen binary standalone before bundling:

```bash
./apps/desktop/src-tauri/binaries/doc-assistant-api-<triple>     # starts uvicorn on :8001
curl http://127.0.0.1:8001/api/health                            # → {"status":"ok",...} once warm
```

## 3. Icons

`tauri.conf.json` references `icons/*`. Generate them from a source PNG (≥ 1024²):

```bash
cd apps/desktop && npx tauri icon path/to/icon.png   # writes src-tauri/icons/*
```

## 4. Build the app + installer

```bash
cd apps/desktop && npx tauri build
```

This runs `beforeBuildCommand` (`npm run build` → `dist/`), compiles the Rust shell, bundles the
sidecar (`bundle.externalBin`) + `dist/`, and emits the per-OS installer under
`src-tauri/target/release/bundle/`. The shell spawns the sidecar on startup (`src/lib.rs`); the
frontend's readiness gate polls `/api/health` until the models warm.

For the dev inner loop (no freeze), keep the two-process flow: `just api` + `npm run dev` (or
`npx tauri dev`, which loads the Vite dev server in a native window).

## 5. Rigor gates before "done" (RIGOR_TODO RG-M4-*)

- **Cold-start** time (process launch → first `/api/health 200`, models loaded) — record on the frozen build.
- **SSE first-token latency** must not regress vs Chainlit — measure on the frozen build.
- **Clean-machine smoke**: install + run on a machine with **no Python / no toolchain** — the portability gate.
- Parity: the Tauri app renders the same `TurnResult` as the CLI / Chainlit (the M0 parity contract).

## Notes

- Bind `127.0.0.1` only (the entrypoint enforces it); the sidecar is the app's private backend.
- onefile sidecar = one binary (Tauri's model) at the cost of slower cold-start (unpacks to temp).
  If cold-start fails the latency gate, switch the spec to onedir + ship `_internal/` as a Tauri
  resource. Trade-off noted in `scripts/doc_assistant_api.spec`.
- Deleting Chainlit + lifting the Python-3.12 pin (KI-2) is **PR-M5**, after this ships.
