"""Measure desktop-API cold-start + SSE first-token latency (PR-M4 ship gates RG-010/011).

Run against a backend on 127.0.0.1:8001 — the **frozen sidecar** (`dist\\doc-assistant-api.exe`)
for the real numbers, or `just api` for a dev baseline. ``--launch`` spawns the binary and
times cold-start from process start.

  # cold-start (RG-010) + first-token (RG-011) of the frozen build:
  uv run --no-sync python -m scripts.measure_latency --launch dist\\doc-assistant-api.exe

  # first-token only, against an already-running backend:
  uv run --no-sync python -m scripts.measure_latency --question "What is RAG?"

NOTE: the chat call hits the real LLM provider — it costs tokens. Point the backend at the
real corpus (`DOC_DATA_DIR` / the per-user data dir) or the numbers are meaningless.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import httpx

BASE = "http://127.0.0.1:8001"


def wait_healthy(t0: float, timeout: float = 180.0) -> float:
    """Seconds from ``t0`` until /api/health returns 200 (models warm)."""
    while time.monotonic() - t0 < timeout:
        try:
            if httpx.get(f"{BASE}/api/health", timeout=2).status_code == 200:
                return time.monotonic() - t0
        except Exception:
            pass
        time.sleep(0.2)
    raise SystemExit(f"backend never became healthy within {timeout:.0f}s")


def first_token_latency(question: str) -> float:
    """Seconds from POST /api/chat to the first ``event: token``."""
    t0 = time.monotonic()
    with httpx.Client(timeout=120) as client, client.stream(
        "POST", f"{BASE}/api/chat", json={"text": question, "session_id": "bench"}
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event: token"):
                return time.monotonic() - t0
    raise SystemExit("stream ended with no token")


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure RG-010 cold-start + RG-011 first-token.")
    ap.add_argument("--launch", help="path to the sidecar binary to spawn + time cold-start")
    ap.add_argument("--question", default="What is RAG?")
    ap.add_argument("--health-only", action="store_true", help="skip the (paid) chat call")
    args = ap.parse_args()

    proc: subprocess.Popen[bytes] | None = None
    t0 = time.monotonic()
    if args.launch:
        proc = subprocess.Popen([args.launch], env=os.environ.copy())
    try:
        warm = wait_healthy(t0)
        print(f"[RG-010] cold-start (launch -> first /api/health 200): {warm:5.1f}s")
        if not args.health_only:
            ttft = first_token_latency(args.question)
            print(f"[RG-011] first-token latency: {ttft:5.2f}s  (q: {args.question!r})")
            print("         compare to Chainlit (`just chat`, same q) — should be within noise;")
            print(
                "         both run the SAME ChatController, so only the freeze + HTTP hop differ."
            )
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
