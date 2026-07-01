# PyInstaller spec — frozen FastAPI sidecar for the Tauri desktop app (PR-M4).
# Build via: uv run --no-sync python -m scripts.build_sidecar   (runbook: docs/desktop-packaging.md)
#
# STARTING POINT — freezing torch / Chroma / sentence-transformers / pymupdf reliably needs
# a few rounds of hidden-import + data-file fixes on the *target* machine (each ML lib loads
# things by string at runtime). Expect to iterate: run the produced binary, read the
# ModuleNotFoundError / missing-data traceback, add to `hiddenimports` / `datas`, repeat.
#
# onefile: one self-contained binary (Tauri's externalBin sidecar model). Trade-off: slower
# cold start (it unpacks to a temp dir on launch) — measured against the latency rigor gate
# (RG: SSE first-token latency on the frozen build).

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (SPECPATH injected by PyInstaller)

datas, binaries, hiddenimports = [], [], []

# Data-/binary-heavy packages whose contents PyInstaller can't infer statically.
for pkg in (
    "torch",
    "chromadb",
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "fitz",  # pymupdf
    "pymupdf4llm",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_chroma",
    "langchain_huggingface",
    "langchain_anthropic",
    "langchain_ollama",
    "langchain_text_splitters",
    "duckdb",
    "rank_bm25",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Our own packages + uvicorn's string-loaded impls.
hiddenimports += collect_submodules("doc_assistant")
hiddenimports += collect_submodules("apps")
hiddenimports += collect_submodules("truststore")  # KI-10: OS trust store (entrypoint injects it)
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

# KI-9 — bundle the embedder + reranker weights so the frozen build needs NO first-run
# HuggingFace download and works fully offline. Stage a minimal, symlink-free, blob-less
# HF hub cache (refs/main + a dereferenced snapshots/<rev>) into `hf_cache/hub`; the
# entrypoint (apps/api/__main__.py) points HF_HOME there + forces offline. Requires a warm
# HF cache on the BUILD box (run the app / ingest once first). ~1.5 GB — bge-reranker-base
# dominates; this is why the installer is large (productionization: onedir / a Tauri
# resource instead of embedding in the onefile — see docs/desktop-packaging.md / KI-9).
import shutil  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402

_HF_MODELS = ("BAAI/bge-base-en-v1.5", "BAAI/bge-reranker-base")
_STAGE = _Path(REPO) / "build" / "hf_stage"
_HUB = _STAGE / "hf_cache" / "hub"
if _HUB.exists():
    shutil.rmtree(_HUB)
for _repo in _HF_MODELS:
    try:
        _snap = _Path(snapshot_download(_repo, local_files_only=True))
    except Exception as _e:  # noqa: BLE001
        raise SystemExit(
            f"KI-9: '{_repo}' is not in the local HF cache ({_e}). Warm it first by running "
            "the app / ingest once on this box, then re-freeze."
        )
    _org, _name = _repo.split("/")
    _model_dir = _HUB / f"models--{_org}--{_name}"
    # copytree with symlinks=False dereferences HF's snapshot→blob symlinks into real files.
    shutil.copytree(_snap, _model_dir / "snapshots" / _snap.name, symlinks=False)
    (_model_dir / "refs").mkdir(parents=True, exist_ok=True)
    (_model_dir / "refs" / "main").write_text(_snap.name)
for _f in _STAGE.rglob("*"):  # preserve the hf_cache/hub/... layout under the bundle root
    if _f.is_file():
        datas.append((str(_f), str(_f.parent.relative_to(_STAGE))))

a = Analysis(
    [os.path.join(REPO, "apps", "api", "__main__.py")],
    pathex=[REPO, os.path.join(REPO, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="doc-assistant-api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    onefile=True,
)
