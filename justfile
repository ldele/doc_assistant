# doc_assistant — task runner (optional; needs `just`: https://github.com/casey/just).
# A facade (cpc ADR-011): every recipe aliases a command you can run without `just`.
# `just --list` shows the comment line right above each recipe, so that line is its summary.
#
# Torch backend per machine: `setx DOC_TORCH cu130` on a CUDA box; unset means `cpu`, the safe
# default (the cu130 wheel segfaults without a GPU). Why: docs/specs/torch-backend-per-machine.md.

# Windows has no POSIX `sh`; every recipe body is one command, so cmd is enough.
set windows-shell := ["cmd.exe", "/c"]

torch := env_var_or_default("DOC_TORCH", "cpu")

# Show the resolved backend for this machine.
default:
    @echo "DOC_TORCH backend = {{torch}}  (set DOC_TORCH=cu130 on the GPU box)"

# Install/refresh the venv with the right torch wheel + dev toolchain.
sync:
    uv sync --extra {{torch}} --extra dev

# Same, app-only (no dev tools).
sync-app:
    uv sync --extra {{torch}}

# Ingest documents (keeps the GPU box on cu130).
ingest *ARGS:
    uv run --extra {{torch}} python -m doc_assistant.ingest {{ARGS}}

# Run the eval harness.
eval *ARGS:
    uv run --extra {{torch}} python -m scripts.run_eval {{ARGS}}

# Compare two eval runs -- verdict first: may these numbers be read against each other at all?
compare *ARGS:
    uv run --extra {{torch}} python -m scripts.compare_runs {{ARGS}}

# Write a baseline document from the run record, provenance block included (fill its TODO after).
emit-baseline *ARGS:
    uv run --extra {{torch}} python -m scripts.emit_baseline {{ARGS}}

# --- optimisation instruments: locked settings change only on an eval win (.claude/CONTEXT.md) ---
# Start every sweep with --dry-run. A run compares only with baselines on the same corpus
# (evals/README.md); `just compare A B` checks that.

# BM25/vector ensemble-weight sweep — retrieval-only, $0, near-deterministic, no re-embed.
sweep-bm25 *ARGS:
    uv run --extra {{torch}} python -m scripts.sweep_bm25_weight {{ARGS}}

# Never trust a chunking result from a run that skipped its preflight (KI-41).
# Parent/child chunk-size sweep — re-embeds the corpus per grid point (slow; GPU strongly advised).
sweep-chunking *ARGS:
    uv run --extra {{torch}} python -m scripts.sweep_chunking {{ARGS}}

# Stage-by-stage pipeline profile (startup/query/ingest budgets) — local, $0, writes nothing.
profile *ARGS:
    uv run --extra {{torch}} python -m scripts.profile_stages {{ARGS}}

# One-shot app launch: API + desktop UI in their own windows, wait for health, open browser.
app:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_app.ps1

# For the native Tauri window, run `cd apps/desktop && npx tauri dev` instead.
# Launch the desktop frontend (Svelte/Vite) in dev — pair with `just api` in another shell.
desktop:
    cd apps/desktop && npm run dev

# CLI fallback.
cli *ARGS:
    uv run --extra {{torch}} python apps/cli.py {{ARGS}}

# Run the desktop API (FastAPI + SSE over 127.0.0.1; the Tauri frontend connects here in dev).
api *ARGS:
    uv run --extra {{torch}} uvicorn apps.api.main:app --host 127.0.0.1 --port 8001 {{ARGS}}

# Verify the frozen-sidecar build prerequisites (triple, CPU torch, entrypoint) — no freeze.
sidecar-check:
    uv run --no-sync python -m scripts.build_sidecar --check

# Build the frozen FastAPI sidecar (CPU-synced venv + packaging extra — see KI-3).
sidecar:
    uv run --no-sync python -m scripts.build_sidecar

# A green run is necessary, not sufficient: the judgment steps are in docs/RELEASE.md.
# Release preflight — the mechanical half of docs/RELEASE.md (read-only, seconds).
preflight:
    uv run --no-sync python -m scripts.release_preflight

# Full test suite (always needs dev).
test:
    uv run --extra {{torch}} --extra dev pytest tests/unit tests/integration

# Dependency advisories, blocking — the CI gate (reviewed ignores in pip-audit-ignore.toml).
audit:
    uv run --no-sync python -m scripts.pip_audit_gate

# Not named `secrets`: the scanner would read that recipe name as a key and flag this file.
# Secret scan over the tracked files, blocking — the CI gate.
secret-scan:
    uv run --no-sync python -m scripts.secret_scan_gate

# Never `mypy --strict src`: it is a different option set and wipes the cache (CONTEXT.md §8).
# Type-check exactly as CI and the pre-commit hook do (`uv run mypy src/`).
typecheck:
    uv run --no-sync mypy src

# Own cache dir, so it cannot cold-start `typecheck`; stricter than CI (unused ignores).
# Escape hatch for the divergent `--strict` flag set — prefer `just typecheck`.
typecheck-strict:
    uv run --no-sync mypy --strict --cache-dir .mypy_cache-strict src

# Verify the active torch wheel + CUDA availability on this machine.
torch-check:
    uv run --extra {{torch}} python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"

# Not `cargo clean`: target/ holds the installers. Flags: -DryRun, -Registry, -Force (see the script).
# Reclaim Rust build space — keeps the release installers and the frozen sidecar.
clean *ARGS:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/clean_build.ps1 {{ARGS}}

# --- cpc conventions: tools/conventions/ is local-only (ADR-001/ADR-007), absent on a fresh clone ---

# Fast docs/route gate.
check:
    uv run --no-sync python tools/conventions/rungate.py docs_check --root . --strict

# Docs gate + file-integrity gate.
lint:
    uv run --no-sync python tools/conventions/rungate.py docs_check --root . --strict
    uv run --no-sync python tools/conventions/rungate.py integrity_check --root . --strict

# Boundary ritual (cpc ADR-020): plan-start | session-start | sprint-start | sprint-close | session-close.
keypoint NAME:
    uv run --no-sync python tools/conventions/rungate.py keypoint {{NAME}}
