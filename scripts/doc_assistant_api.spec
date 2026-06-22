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
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

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
