# Headless FastAPI backend (PR-M2/M4) — the same backend the desktop sidecar bundles.
# The Tauri GUI runs on the host, not here; this image is for running the API as a service.
#
# Base tracks `.python-version` (3.12). Do not float it: KI-2 records that the native
# dependencies in the LLM-client import path crash the interpreter on 3.14, and nothing in
# `requires-python` (">=3.10") would stop a newer base from being used.
FROM python:3.12-slim

# System dependencies needed by some Python packages
# - build-essential: compiles C extensions
# - libxml2-dev, libxslt1-dev: lxml's native deps
# - curl: useful for healthchecks and debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install with **uv**, not pip, and this is load-bearing rather than a preference.
# `[tool.uv.sources]` routes torch to the `pytorch-cpu` index under the `cpu` extra, and that
# routing is uv-only configuration. `pip install ".[cpu]"` sees nothing but `torch>=2.12` and
# resolves it from PyPI — whose default Linux wheel bundles CUDA, pulling several GB of nvidia-*
# packages into an image that has no GPU. This is the same trap `.github/workflows/ci.yml`
# documents at length. Using uv also honours `uv.lock`, which pip ignores entirely.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /bin/uv

WORKDIR /app

# Manifest + lock + the package source first, so dependencies re-resolve only when those change.
# `src/` is copied here because the project installs itself, so the build needs it present.
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ ./src/
RUN uv sync --locked --extra cpu

# The rest of the application — changing these does not invalidate the dependency layer above.
COPY apps/ ./apps/
COPY scripts/ ./scripts/

EXPOSE 8001

# Bind 0.0.0.0 INSIDE the container so the host can reach it (the app defaults to
# 127.0.0.1 — safe-by-default — and exposes DOC_API_HOST to opt into 0.0.0.0 here).
ENV DOC_API_HOST=0.0.0.0 DOC_API_PORT=8001
# start-period is generous on purpose: the first launch downloads the embedder and reranker
# (a few hundred MB) before /api/health can go green. Measured cold on a clean Linux box:
# see the v0.4.0 clean-room run in docs/DEVLOG.md.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8001/api/health || exit 1
CMD ["uv", "run", "--no-sync", "python", "-m", "apps.api"]
