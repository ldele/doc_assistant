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

WORKDIR /app

# Copy dependency manifest first, install separately.
# Dependencies only reinstall when pyproject.toml changes.
# The CPU torch extra: a container has no host GPU, and the API needs torch for the
# embedder + reranker (the `just api` dev recipe selects a torch extra the same way).
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[cpu]"

# Copy the rest of the application
COPY apps/ ./apps/
COPY scripts/ ./scripts/

# The FastAPI backend (PR-M2/M4). The Tauri desktop app is the GUI and runs on the host;
# the container serves the headless API the same backend the desktop sidecar bundles.
EXPOSE 8001

# Bind 0.0.0.0 INSIDE the container so the host can reach it (the app defaults to
# 127.0.0.1 — safe-by-default — and exposes DOC_API_HOST to opt into 0.0.0.0 here).
ENV DOC_API_HOST=0.0.0.0 DOC_API_PORT=8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8001/api/health || exit 1
CMD ["python", "-m", "apps.api"]
