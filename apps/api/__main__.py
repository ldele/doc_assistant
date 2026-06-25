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
import sys


def _configure_frozen_runtime() -> None:
    """Wire the freeze-only runtime fixes before the app stack is imported.

    Runs at the entrypoint, ahead of ``from apps.api.main import app``, because both
    fixes must take effect before ``huggingface_hub`` / ``httpx`` / the ``anthropic`` SDK
    are imported and read their config:

    * **KI-9** — when PyInstaller-frozen, point ``HF_HOME`` at the bundled model cache
      (``_MEIPASS/hf_cache``) and force offline, so the embedder + reranker load from the
      bundle with no first-run HuggingFace download (and no hard offline failure).
    * **KI-10** — route outbound TLS through the OS/system trust store via ``truststore``,
      so the bundled ``certifi`` set doesn't reject a corporate TLS-MITM proxy's root CA.
      No-op-safe in dev; guarded so a missing/un-importable ``truststore`` never blocks start.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            os.environ.setdefault("HF_HOME", os.path.join(meipass, "hf_cache"))
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as e:  # never block startup (e.g. truststore absent in a dev venv)
        # Visible, not silent: a swallowed failure here is a prime suspect for KI-10 (the frozen
        # build still hitting CERTIFICATE_VERIFY_FAILED — if inject_into_ssl fails, certifi is
        # used and the corporate MITM CA is rejected). Pre-logging entrypoint, so write straight
        # to stderr (cf. the sanctioned llm.py stderr write); startup still proceeds.
        msg = f"WARN truststore.inject_into_ssl() failed (OS trust store inactive): {e}\n"
        sys.stderr.write(msg)
        sys.stderr.flush()


_configure_frozen_runtime()

import uvicorn  # noqa: E402

from apps.api.main import app  # noqa: E402


def main() -> None:
    host = os.environ.get("DOC_API_HOST", "127.0.0.1")
    port = int(os.environ.get("DOC_API_PORT", "8001"))
    # 127.0.0.1 only — the desktop app talks to its own sidecar; never bind 0.0.0.0.
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
