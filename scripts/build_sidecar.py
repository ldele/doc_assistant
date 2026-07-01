"""PR-M4 — build the frozen FastAPI **sidecar** binary for the Tauri desktop app.

PyInstaller-freezes ``apps/api/__main__.py`` into one binary, names it with the Rust
target triple (Tauri's sidecar convention: ``<name>-<triple>[.exe]``), and copies it to
``apps/desktop/src-tauri/binaries/``. ``--check`` verifies prerequisites without freezing.

**CPU torch only** in the frozen build — the ``cu130`` wheel SEGFAULTS on a GPU-less box
(KI-3), and the installer must run anywhere. This script refuses to freeze a CUDA torch.
Sync the build venv with ``uv sync --extra cpu --extra dev`` first.

Run from the repo root:  ``uv run --no-sync python -m scripts.build_sidecar --check``
Full build:             ``uv run --no-sync python -m scripts.build_sidecar``
Runbook: ``docs/desktop-packaging.md``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = Path(__file__).resolve().parent / "doc_assistant_api.spec"
BINARIES_DIR = ROOT / "apps" / "desktop" / "src-tauri" / "binaries"
SIDECAR_NAME = "doc-assistant-api"


def _exe_suffix() -> str:
    return ".exe" if sys.platform == "win32" else ""


def target_triple() -> str:
    """The Rust host target triple (Tauri names sidecars ``<name>-<triple>``)."""
    out = subprocess.run(["rustc", "-Vv"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("could not parse the host triple from `rustc -Vv`")


def check_cpu_torch() -> str:
    """Return the torch version, refusing a CUDA build (KI-3 — cu130 segfaults headless)."""
    import torch

    version = torch.__version__
    if "+cu" in version:
        raise SystemExit(
            f"refusing to freeze with a CUDA torch ({version}) — the cu130 wheel SEGFAULTS on a "
            "GPU-less box (KI-3) and the installer must run anywhere. "
            "Run `uv sync --extra cpu --extra dev` first."
        )
    return version


def check_entry() -> None:
    import importlib

    importlib.import_module("apps.api.__main__")


def _target_path(triple: str) -> Path:
    return BINARIES_DIR / f"{SIDECAR_NAME}-{triple}{_exe_suffix()}"


def do_check() -> None:
    triple = target_triple()
    torch_version = check_cpu_torch()
    check_entry()
    print(f"[OK] target triple        : {triple}")
    print(f"[OK] torch (cpu)          : {torch_version}")
    print("[OK] entrypoint imports   : apps.api.__main__")
    print(f"  -> sidecar will land at : {_target_path(triple)}")
    print("Run without --check to freeze (slow; needs PyInstaller + a CPU-synced venv).")


def do_build() -> None:
    triple = target_triple()
    check_cpu_torch()
    check_entry()
    print("Running PyInstaller (slow; produces a large single-file binary)...")
    subprocess.run(["pyinstaller", "--noconfirm", str(SPEC)], cwd=ROOT, check=True)
    produced = ROOT / "dist" / f"{SIDECAR_NAME}{_exe_suffix()}"  # onefile output
    if not produced.exists():
        raise SystemExit(f"PyInstaller did not produce {produced}")
    BINARIES_DIR.mkdir(parents=True, exist_ok=True)
    target = _target_path(triple)
    shutil.copy2(produced, target)
    print(f"[OK] sidecar copied to {target}")
    print("Next: `npx tauri build` (in apps/desktop) bundles it via tauri.conf.json externalBin.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen FastAPI sidecar (PR-M4).")
    parser.add_argument(
        "--check", action="store_true", help="verify prerequisites only (no freeze)"
    )
    args = parser.parse_args()
    (do_check if args.check else do_build)()


if __name__ == "__main__":
    main()
