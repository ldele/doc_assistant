# doc_assistant — task runner (optional; needs `just`: https://github.com/casey/just)
#
# Per-machine PyTorch backend, the footgun-free way: set DOC_TORCH once per machine
# and every recipe below uses the right wheel.
#   GPU box (RTX/CUDA):  setx DOC_TORCH cu130   (Windows)  /  export DOC_TORCH=cu130
#   CPU-only box + CI:   leave DOC_TORCH unset  -> defaults to the safe `cpu` wheel.
# The +cu130 wheel SEGFAULTS (exit 139) on a box with no usable GPU, so cpu is the default.
# Rationale + the underlying uv extras/conflicts: docs/specs/torch-backend-per-machine.md.

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

# Launch the Chainlit web UI (Chainlit needs Python 3.12 at runtime).
chat:
    uv run --extra {{torch}} --python 3.12 chainlit run apps/chainlit_app.py

# CLI fallback.
cli *ARGS:
    uv run --extra {{torch}} python apps/cli.py {{ARGS}}

# Run the desktop API (FastAPI + SSE over 127.0.0.1; the Tauri frontend connects here in dev).
api *ARGS:
    uv run --extra {{torch}} uvicorn apps.api.main:app --host 127.0.0.1 --port 8001 {{ARGS}}

# Verify the frozen-sidecar build prerequisites (triple, CPU torch, entrypoint) — no freeze.
sidecar-check:
    uv run --no-sync python -m scripts.build_sidecar --check

# Build the frozen FastAPI sidecar (PR-M4). Needs PyInstaller + a CPU-synced venv (KI-3:
# the cu130 wheel segfaults headless — `uv sync --extra cpu --extra dev` first).
sidecar:
    uv run --no-sync python -m scripts.build_sidecar

# Full test suite (always needs dev).
test:
    uv run --extra {{torch}} --extra dev pytest tests/unit tests/integration

# Verify the active torch wheel + CUDA availability on this machine.
torch-check:
    uv run --extra {{torch}} python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"
