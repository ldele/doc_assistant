"""Measure desktop-API cold-start + SSE first-token latency (PR-M4 ship gates RG-010/011).

Two first-token paths, same ``ChatController``, so the difference is the boundary:

* **HTTP/SSE** (default) — POST ``/api/chat`` → first ``event: token`` against a backend on
  127.0.0.1:8001 (the **frozen sidecar** ``dist\\doc-assistant-api.exe`` for the real numbers,
  or ``just api`` for a dev baseline). ``--launch`` spawns the binary and times cold-start.
* **in-process** (``--in-process``) — build a ``ChatController`` and time
  ``handle_message`` → first ``Token`` with no server. This is the Chainlit/CLI baseline:
  RG-011 passes iff the HTTP/SSE path is **not meaningfully slower** than this.

``--repeat N`` takes N warm samples (one discarded warm-up first) and prints median / min / max
so the gate is read against variance, not a single noisy n=1. Each sample uses a **fresh session**
(unique ``session_id``) so no history-rewrite call sneaks into the timed path — both paths measure
retrieve → generate → first token only.

  # cold-start (RG-010) + first-token (RG-011) of the frozen build, 5 warm samples:
  uv run --no-sync python -m scripts.measure_latency --launch dist\\doc-assistant-api.exe -r 5

  # HTTP/SSE first-token only, against an already-running backend (`just api`):
  uv run --no-sync python -m scripts.measure_latency --repeat 5 --question "What is RAG?"

  # in-process baseline (no server) — the Chainlit/CLI equivalent to compare against:
  uv run --no-sync python -m scripts.measure_latency --in-process --repeat 5

NOTE: the chat call hits the real LLM provider — it costs tokens under a paid provider. Point the
backend at the real corpus (``DOC_DATA_DIR`` / the per-user data dir) and force a local provider
(``.env`` → ``LLM_PROVIDER=ollama``) for a free run, or the numbers are meaningless / billed.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import time
from collections.abc import Callable

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


def http_first_token(question: str, session_id: str) -> float:
    """Seconds from POST /api/chat to the first ``event: token`` (fresh ``session_id``)."""
    t0 = time.monotonic()
    with (
        httpx.Client(timeout=120) as client,
        client.stream(
            "POST", f"{BASE}/api/chat", json={"text": question, "session_id": session_id}
        ) as r,
    ):
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event: token"):
                return time.monotonic() - t0
    raise SystemExit("stream ended with no token")


def in_process_first_token(controller: object, question: str) -> float:
    """Seconds from ``handle_message`` to the first streamed ``Token`` (no server).

    A fresh ``Session`` per call → no history-aware rewrite, matching the HTTP path's
    fresh-session-per-sample timing."""
    from doc_assistant.chat_controller import Session, Token

    session = Session()
    t0 = time.monotonic()
    for event in controller.handle_message(session, question):  # type: ignore[attr-defined]
        if isinstance(event, Token):
            return time.monotonic() - t0
    raise SystemExit("in-process turn produced no token")


def _sample(label: str, sampler: Callable[[int], float], repeat: int) -> list[float]:
    """Run ``repeat`` warm samples (one discarded warm-up first) and print each."""
    print(f"  warming up ({label}) ...", flush=True)
    sampler(0)  # warm-up: model load + first generation, discarded
    samples: list[float] = []
    for i in range(1, repeat + 1):
        dt = sampler(i)
        samples.append(dt)
        print(f"    sample {i}/{repeat}: {dt:6.3f}s", flush=True)
    return samples


def _report(tag: str, samples: list[float], question: str) -> None:
    med = statistics.median(samples)
    lo, hi = min(samples), max(samples)
    spread = hi - lo
    sd = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    print(
        f"[{tag}] first-token latency: median {med:.3f}s "
        f"(min {lo:.3f}s · max {hi:.3f}s · spread {spread:.3f}s · sd {sd:.3f}s) "
        f"over n={len(samples)}  (q: {question!r})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure RG-010 cold-start + RG-011 first-token.")
    ap.add_argument("--launch", help="path to the sidecar binary to spawn + time cold-start")
    ap.add_argument("--question", default="What is RAG?")
    ap.add_argument("-r", "--repeat", type=int, default=1, help="number of warm samples (>=1)")
    ap.add_argument("--health-only", action="store_true", help="skip the (paid) chat call")
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="time ChatController directly (no server) — the RG-011 baseline",
    )
    args = ap.parse_args()

    # Label the active provider/model so a free (ollama) vs billed (anthropic) run is unambiguous.
    from doc_assistant.config import LLM_MODEL, LLM_PROVIDER

    print(f"provider: {LLM_PROVIDER}/{LLM_MODEL}", flush=True)

    if args.in_process:
        from doc_assistant.chat_controller import ChatController

        controller = ChatController()
        print(f"in-process chunk_count: {controller.chunk_count()}", flush=True)
        samples = _sample(
            "in-process", lambda _i: in_process_first_token(controller, args.question), args.repeat
        )
        _report("RG-011 in-process", samples, args.question)
        return

    proc: subprocess.Popen[bytes] | None = None
    t0 = time.monotonic()
    if args.launch:
        proc = subprocess.Popen([args.launch], env=os.environ.copy())
    try:
        warm = wait_healthy(t0)
        print(f"[RG-010] cold-start (launch -> first /api/health 200): {warm:5.1f}s")
        if not args.health_only:
            samples = _sample(
                "http/sse", lambda i: http_first_token(args.question, f"bench-{i}"), args.repeat
            )
            _report("RG-011 http/sse", samples, args.question)
            print("         compare to `--in-process` (same q) — should be within noise;")
            print("         both run the SAME ChatController; only the freeze + HTTP hop differ.")
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
