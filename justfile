# doc_assistant — task runner (optional; needs `just`: https://github.com/casey/just)
#
# Per-machine PyTorch backend, the footgun-free way: set DOC_TORCH once per machine
# and every recipe below uses the right wheel.
#   GPU box (RTX/CUDA):  setx DOC_TORCH cu130   (Windows)  /  export DOC_TORCH=cu130
#   CPU-only box + CI:   leave DOC_TORCH unset  -> defaults to the safe `cpu` wheel.
# The +cu130 wheel SEGFAULTS (exit 139) on a box with no usable GPU, so cpu is the default.
# Rationale + the underlying uv extras/conflicts: docs/specs/torch-backend-per-machine.md.

# Windows has no POSIX `sh`; run recipes through cmd so `just` works out of the box. The
# recipe bodies are plain single commands (uv/npm/uvicorn), so cmd vs PowerShell is moot.
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

# --- optimisation instruments (the locked-settings workflow) -----------------------------------
# `just --list` shows only the LAST comment line above a recipe, so the detail lives here and each
# recipe below keeps a one-line summary.
#
# Locked settings change ONLY on an eval win (.claude/CONTEXT.md): sweep, beat the control beyond
# its variance, record a baseline in tests/eval/baselines/. Both sweeps take --dry-run — it prints
# the grid and the exact commands and touches nothing. Always start there.
#
# ⚠ A run is comparable only to baselines taken on the SAME corpus. Extra documents are retrieval
# distractors, so a public-case run against the private 97-doc index is NOT the committed public
# baseline (evals/README.md): the public 10 need an isolated data home, while the private 35-case
# set matches the indexed library. The public set also saturates citation_overlap at 1.000 and so
# cannot discriminate retrieval changes — retrieval experiments belong on the larger corpus.
#
# ⚠ sweep-chunking's preflight is load-bearing: it asserts every config actually reaches the code
# and that no two grid points are the same experiment. That guard exists because the 2026-06-06 run
# compared one configuration with itself six times and reported a verdict (KI-41). Never trust a
# chunking result from a run that skipped it.

# BM25/vector ensemble-weight sweep — retrieval-only, $0, near-deterministic, no re-embed.
sweep-bm25 *ARGS:
    uv run --extra {{torch}} python -m scripts.sweep_bm25_weight {{ARGS}}

# Parent/child chunk-size sweep — re-embeds the corpus per grid point (slow; GPU strongly advised).
sweep-chunking *ARGS:
    uv run --extra {{torch}} python -m scripts.sweep_chunking {{ARGS}}

# Stage-by-stage pipeline profile (startup/query/ingest budgets) — local, $0, writes nothing.
profile *ARGS:
    uv run --extra {{torch}} python -m scripts.profile_stages {{ARGS}}

# One-shot app launch: API + desktop UI in their own windows, wait for health, open browser.
app:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_app.ps1

# Launch the desktop frontend (Svelte/Vite) in dev — pair with `just api` in another shell
# (or `cd apps/desktop && npx tauri dev` for the native Tauri window).
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

# Build the frozen FastAPI sidecar (PR-M4; CPU-synced venv + packaging extra — see KI-3).
sidecar:
    uv run --no-sync python -m scripts.build_sidecar

# Release preflight — the MECHANICAL half of docs/RELEASE.md. Read-only, seconds, exits non-zero on
# a real problem. Checks the five version strings agree (incl. uv.lock), the artifact is newer than
# every tracked source file, the sidecar did not lose a bundle (KI-34's size cliff), that RG-012
# passed on THIS artifact, and that no developer command reached the shipped UI (KI-39).
# A green run is necessary, not sufficient — the judgment steps are in docs/RELEASE.md.
preflight:
    uv run --no-sync python -m scripts.release_preflight

# Full test suite (always needs dev).
test:
    uv run --extra {{torch}} --extra dev pytest tests/unit tests/integration

# Type-check exactly as CI and the pre-commit hook do (`uv run mypy src/`).
#
# ⚠ USE THIS, NOT `mypy --strict src`. Strictness already comes from [tool.mypy] strict=true, so
# the flag adds nothing — except that it ALSO re-enables warn_unused_ignores (pyproject turns it
# off), which makes it a DIFFERENT option set. mypy keys its incremental cache on the options, so
# alternating the two forms invalidates the whole cache every time: measured on this repo,
# `mypy src` is 2.4s warm and 40.5s right after a `--strict` run. That flip-flop was making every
# commit pay ~40s in the pre-commit mypy hook. Same reason CI uses the bare form.
typecheck:
    uv run --no-sync mypy src

# Escape hatch for the divergent flag set, pinned to its OWN cache dir so it cannot cold-start
# `typecheck` (or be cold-started by it). Note it is *stricter than CI*: it reports unused
# `type: ignore`s, which the shipped config deliberately allows.
typecheck-strict:
    uv run --no-sync mypy --strict --cache-dir .mypy_cache-strict src

# Verify the active torch wheel + CUDA availability on this machine.
torch-check:
    uv run --extra {{torch}} python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"

# --- cpc conventions (vendored, LOCAL-ONLY — ADR-021) -----------------------------------------
# tools/conventions/ is gitignored (private tooling, public repo — ADR-001/ADR-007); on a fresh
# clone these recipes are unavailable by design. Facade only (cpc ADR-011): each recipe aliases a
# directly-runnable command; nothing lives ONLY here.

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
