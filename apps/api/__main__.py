"""Standalone server entrypoint — ``python -m apps.api`` (dev) and the **frozen sidecar**
entry that PyInstaller bundles for the Tauri desktop app (PR-M4).

Runs the FastAPI app under uvicorn on ``127.0.0.1``. Host/port are env-overridable so the
Tauri shell can pick a port; defaults match the dev contract (Vite proxies to ``:8001``).
The real ``ChatController`` (model load) is built lazily in the app's ``lifespan`` — so the
process starts immediately and ``/api/health`` flips to ``200`` once the stack is warm
(the frontend's readiness gate polls it).
"""

from __future__ import annotations

import os

import uvicorn

from apps.api.main import app


def main() -> None:
    host = os.environ.get("DOC_API_HOST", "127.0.0.1")
    port = int(os.environ.get("DOC_API_PORT", "8001"))
    # 127.0.0.1 only — the desktop app talks to its own sidecar; never bind 0.0.0.0.
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
