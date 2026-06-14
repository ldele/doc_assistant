"""Project configuration. All paths and runtime settings live here."""

import os
from pathlib import Path

from dotenv import load_dotenv

# override=True: .env is this app's config source of truth. Some host
# environments (e.g. the Claude Code session env) export an *empty*
# ANTHROPIC_API_KEY; with the python-dotenv default (override=False) that
# empty value would shadow the real key in .env and break API mode. Letting
# .env win is the least-surprising behaviour for a local-first, single-user app.
load_dotenv(override=True)


# ============================================================
# Project root resolution
# ============================================================
# This file lives at: <project_root>/src/doc_assistant/config.py
# parents[0] = doc_assistant/
# parents[1] = src/
# parents[2] = <project_root>/

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Runtime data paths
# ============================================================

DATA_PATH = PROJECT_ROOT / "data"
DOCS_PATH = DATA_PATH / "sources"
CACHE_PATH = DATA_PATH / "cache"
CHROMA_PATH = str(DATA_PATH / "chroma")  # Chroma wants a string, not Path
PC_CHROMA_PATH = str(DATA_PATH / "chroma_pc")  # Parent - Child
SQLITE_PATH = str(DATA_PATH / "library.db")
SQLITE_URL = f"sqlite:///{SQLITE_PATH}"


# ============================================================
# LLM configuration
# ============================================================

LLM_MODE = os.getenv("LLM_MODE", "api")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ---- Provider selection (Phase 6 — Feature 1, generation side) ----
# Two call shapes, configured independently:
#   * Analysis / chat (streaming LangChain model) → LLM_PROVIDER / LLM_MODEL
#   * One-shot reviewer + eval judge (normalized LLMClient.complete) →
#     REVIEWER_* / JUDGE_*
# LLM_MODE is kept for back-compat and derives the analysis provider default.
# The generator targets fully local (Ollama, no API key); the reviewer and
# judge default to a pinned reference model so cross-run numbers stay
# comparable (see tests/eval/TESTING.md → "The judge is a pinned instrument").
_REFERENCE_MODEL = "claude-haiku-4-5-20251001"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic" if LLM_MODE == "api" else "ollama")
# Analysis-model default is provider-dependent so historical behaviour is
# preserved exactly: api → haiku (ChatAnthropic), ollama → llama3 (OllamaLLM).
_DEFAULT_ANALYSIS_MODEL = _REFERENCE_MODEL if LLM_PROVIDER == "anthropic" else "llama3"
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_ANALYSIS_MODEL)
REVIEWER_PROVIDER = os.getenv("REVIEWER_PROVIDER", LLM_PROVIDER)
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", _REFERENCE_MODEL)
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", LLM_PROVIDER)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", _REFERENCE_MODEL)


# ============================================================
# Extraction configuration
# ============================================================

PDF_EXTRACTOR = os.getenv("PDF_EXTRACTOR", "pymupdf")


# ============================================================
# HuggingFace token (read, faster downloads)
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# ============================================================
# Embedding model (Phase 5, Feature 1)
# ============================================================
# Active model is resolved by `doc_assistant.embeddings.get_active_model_name()`.
# This constant is the env-var read at import time, kept for ergonomic access
# in places where the model name is part of a log line or a UI surface.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-base")


# ============================================================
# Experimental flags (Default set to best results)
# ============================================================

# Use parent-child retrieval: small chunks for retrieval precision
USE_PARENT_CHILD = (
    os.getenv("USE_PARENT_CHILD", "true").lower() == "true"
)  # Enabled by default -> better perf in testing

# Multi-query expansion
USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "false").lower() == "true"

# Number of chunks/parents passed to the LLM at query time (final rerank cut).
TOP_K = int(os.getenv("TOP_K", "10"))

# Candidate pool size fetched from EACH retriever (vector + BM25) BEFORE
# reranking. The cross-encoder needs more candidates than it returns to have
# room to reorder; this was previously hardcoded to 10 == TOP_K in pipeline.py,
# leaving the reranker almost nothing to do. CANDIDATE_K >= TOP_K is required
# (guarded below). NOTE: widening this changes retrieval output. Public-corpus A/B
# (2026-06-13, tests/eval/baselines/candidate_k_public_2026-06-13.md): no regression vs
# CANDIDATE_K=10, so 20 is kept as a safe default; the cross-paper crowding benefit still
# wants the private neuroscience arm to be a measured win. Set CANDIDATE_K=10 to
# reproduce the pre-split behaviour exactly.
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))
if CANDIDATE_K < TOP_K:
    raise ValueError(f"CANDIDATE_K ({CANDIDATE_K}) must be >= TOP_K ({TOP_K})")

# Synthesis mode (Phase 6 / Integrity Chunk 2a — dual interpretation).
#   ai    -> dual-layer: deterministic evidence + labelled AI interpretation,
#            segmented into adjudicable (accept/reject/edit) claims.
#   human -> evidence layer only; the interpretation is the user's. The
#            interpretation LLM call is skipped (BE WISE-influenced path).
VALID_SYNTHESIS_MODES = ("ai", "human")
SYNTHESIS_MODE = os.getenv("SYNTHESIS_MODE", "ai").lower()
if SYNTHESIS_MODE not in VALID_SYNTHESIS_MODES:
    raise ValueError(
        f"SYNTHESIS_MODE must be one of {VALID_SYNTHESIS_MODES}, got {SYNTHESIS_MODE!r}"
    )

# Marker table-ingest (Phase 6 / Feature 4a). Concurrent `marker_single`
# subprocesses in the post-ingest CLI; each loads multi-GB surya models, so the
# bound is memory, not cores. Raise on a big-VRAM/RAM box; drop to 1 on OOM.
MARKER_MAX_WORKERS = int(os.getenv("MARKER_MAX_WORKERS", "2"))

# Python version `uvx` resolves the isolated Marker env against. marker-pdf pins
# deps (e.g. pillow==10.4.0) that have no cp313/cp314 wheels, so a newer default
# interpreter forces a from-source build that fails. 3.12 has wheels for the whole
# stack. Only applies to the `uvx` path (an on-PATH `marker_single` is used as-is).
MARKER_PYTHON = os.getenv("MARKER_PYTHON", "3.12")


# ============================================================
# Chunking configuration (Phase 6 — chunking experiment)
# ============================================================
# These sizes were the variable under test in Phase 2.4 but were never
# measured — they lived as hardcoded constants in ingest.py. Making them
# env-driven is the prerequisite for sweeping chunk strategies through the
# eval harness without editing source (see scripts/sweep_chunking.py).
#
# Defaults reproduce the historical hardcoded values exactly, so this change
# is behaviour-preserving until an experiment justifies new values.
#
# NOTE: changing any of these invalidates the embedding cache — a sweep
# re-embeds the corpus per config. Budget accordingly.

# Parent chunks: large passages sent to the LLM at query time.
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))

# Child chunks: small passages embedded for retrieval precision.
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))

# Baseline (single-chunk) retrieval store, used when USE_PARENT_CHILD is off.
BASELINE_CHUNK_SIZE = int(os.getenv("BASELINE_CHUNK_SIZE", "1000"))
BASELINE_CHUNK_OVERLAP = int(os.getenv("BASELINE_CHUNK_OVERLAP", "200"))


# ============================================================
# Figures (Phase 6 / Feature 4b — figure region detection)
# ============================================================
# Figures are binary, so — unlike tables — they are NOT spliced into the
# markdown cache. Each detected figure region persists as a `Figure` sidecar
# row plus a cropped PNG under FIGURE_DIR (Enrichment-Layer "sidecar by
# default"). See docs/specs/feature-4b-figure-detection.md.

# Sidecar PNG root, alongside chroma/ and library.db. Gitignored (binary).
FIGURE_DIR = DATA_PATH / "figures"

# Crop resolution for the per-region PNG. Raise for VLM-quality crops (4c),
# lower to save disk.
FIGURE_RENDER_DPI = int(os.getenv("FIGURE_RENDER_DPI", "150"))

# Per-region page-area floor: skip decorative images/icons below this fraction
# of the page so a logo never becomes a Figure row. Distinct from
# `regions.IMAGE_AREA_MIN` (0.05), which is the page-dominance threshold for
# classifying a whole page as a photo; this is the smaller per-region floor.
FIGURE_MIN_AREA_FRACTION = float(os.getenv("FIGURE_MIN_AREA_FRACTION", "0.02"))
