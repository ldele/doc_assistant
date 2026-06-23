"""Project configuration. All paths and runtime settings live here."""

import os
import sys
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


def _resolve_data_path() -> Path:
    """Resolve the runtime data dir (corpus, DB, exports, figures, graph).

    Precedence: ``DOC_DATA_DIR`` env override > a stable per-user app-data dir when
    PyInstaller-**frozen** > the in-repo ``./data`` (dev). When frozen, ``__file__`` lives
    in a temp unpack dir, so ``PROJECT_ROOT`` climbs into ``%TEMP%`` and the in-repo path is
    meaningless — the desktop app keeps its data in a per-user location instead (PR-M4).
    Point the override at an existing corpus to reuse it: ``DOC_DATA_DIR=...\\data``.
    """
    override = os.getenv("DOC_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):  # running from a PyInstaller bundle
        base = (
            os.getenv("LOCALAPPDATA")
            or os.getenv("XDG_DATA_HOME")
            or str(Path.home() / ".local" / "share")
        )
        return Path(base) / "doc_assistant" / "data"
    return PROJECT_ROOT / "data"


DATA_PATH = _resolve_data_path()
DOCS_PATH = DATA_PATH / "sources"
CACHE_PATH = DATA_PATH / "cache"
CHROMA_PATH = str(DATA_PATH / "chroma")  # Chroma wants a string, not Path
PC_CHROMA_PATH = str(DATA_PATH / "chroma_pc")  # Parent - Child
SQLITE_PATH = str(DATA_PATH / "library.db")
SQLITE_URL = f"sqlite:///{SQLITE_PATH}"

# Conversation + debug exports (markdown transcripts, dev bundles, per-turn JSONL
# logs). Gitignored, regenerable — written by `doc_assistant.export`. A user
# downloads a clean transcript; a dev grabs the full bundle (sources + scores +
# figures + reviewer) plus the session log to iterate quickly.
EXPORT_DIR = DATA_PATH / "exports"


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

# Providers that bill real money per call. The single source of truth for the
# enrichment-CLI cost guard (`doc_assistant.llm.assert_provider_intent`): a
# `--apply` enrichment run resolving to one of these prints a loud cost warning
# + abort window (key present) or fails loudly (key missing), so a run the user
# believes is "local" can never *silently* spend — the 2026-06-15 credit-burn
# footgun. Ollama is local/free and absent here by design. Extend if a new paid
# provider is added. (Declarative policy data lives here; the behaviour that
# reads it lives in llm.py, next to make_client/reviewer_available.)
PAID_PROVIDERS = frozenset({"anthropic"})


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


# ============================================================
# Figure VLM description (Phase 6 / Feature 4c — gated, API-only)
# ============================================================
# 4c turns the 4b `Figure` rows into VLM-described, retrievable figure chunks.
# This is the project's only API-only, *paid* enrichment, so it is gated three
# ways: a caption-length skip (a well-captioned figure needs no VLM), a per-doc
# call budget, and the 4b precondition that only figures with a rendered PNG are
# eligible. Anthropic-only by decision (vision + tool-use); no local path.

# Vision model for figure description. Defaults to Haiku 4.5 — vision-capable and
# the cheapest tier — matching this project's cost-gated instrument convention
# (the reviewer/judge also default to a Haiku reference model). Bump to
# `claude-sonnet-4-6` or `claude-opus-4-8` via env for higher-fidelity crops.
FIGURE_VLM_MODEL = os.getenv("FIGURE_VLM_MODEL", "claude-haiku-4-5")

# Hard ceiling on VLM calls per document — the cost ceiling for one paper. A
# figure page beyond this many figures records `vlm_call_skipped_reason=
# "budget_exhausted"` rather than calling. Raise for figure-dense papers.
MAX_VLM_CALLS_PER_DOC = int(os.getenv("MAX_VLM_CALLS_PER_DOC", "30"))

# A caption at/above this many characters is treated as already self-describing,
# so the figure is skipped (`vlm_call_skipped_reason="caption_sufficient"`) — the
# VLM is a quality lever for thinly-captioned figures, not every figure. Set to 0
# to describe every eligible figure regardless of caption length.
FIGURE_CAPTION_DESC_MIN_CHARS = int(os.getenv("FIGURE_CAPTION_DESC_MIN_CHARS", "300"))


# ============================================================
# Reviewer self-improvement loop (Phase 6 / Integrity Chunk 2c)
# ============================================================
# The per-answer reviewer (Chunk 2b) is a biased sampler (it runs only on
# already-flagged answers) AND an LLM with its own tilts, so a recurring
# `failure_tag` is ambiguous by construction. The minimum-N gate keeps a raw
# count from being read as a finding: a tag is "actionable" only once it clears
# BOTH thresholds, and counts are always shown against their denominator. Below
# the gate the report reads "insufficient evidence." Tune on the first real
# distribution (see docs/decisions.md → Integrity Chunk 2c).

# A `failure_tag` must occur at least this many times to be reported as actionable.
MIN_FAILURE_TAG_COUNT = int(os.getenv("MIN_FAILURE_TAG_COUNT", "10"))

# ...and across at least this many *distinct answer records* (so one heavily
# re-reviewed answer can't trip the gate on its own).
MIN_FAILURE_TAG_DOCS = int(os.getenv("MIN_FAILURE_TAG_DOCS", "5"))

# How many characters of each retrieved chunk the reviewer/judge sees as evidence.
# Distinct from the ~300-char display excerpt on the provenance card: a faithfulness
# judge needs to actually *find* a claim's support, and 300 chars of a ~2000-char
# parent starves it into false "unsupported_claim" verdicts (observed 2026-06-17,
# self-eval: a capable judge failed well-grounded answers it simply couldn't verify).
# Wider = fairer judgement but more judge tokens; 1500 is a balance. Not persisted.
REVIEWER_EVIDENCE_CHARS = int(os.getenv("REVIEWER_EVIDENCE_CHARS", "1500"))


# ============================================================
# Self-organizing wiki / synthesis layer (Phase 7 / Feature 6)
# ============================================================
# A derived, human-readable markdown layer over the corpus: cluster the library,
# emit one topic note per cluster (summary + tags + [[links]] + citations), as a
# sidecar under WIKI_DIR. Enrichment-Layer Pattern — idempotent, regenerable,
# never mutates the chunk store. See docs/doc-assistant-roadmap.md → Feature 6.

# Sidecar markdown root (Obsidian-compatible). Gitignored (derived, regenerable).
WIKI_DIR = DATA_PATH / "wiki"

# Topic summarization is the *generator* role (not a pinned instrument) and a
# per-cluster batch op — the same silent-spend profile as concept extraction. So,
# like CONCEPT_GRAPH_LLM_PROVIDER (Feature 7), it defaults to LOCAL Ollama
# *explicitly*, NOT to LLM_PROVIDER. This keeps `build_wiki --apply` free by
# default; `--provider anthropic` is opt-in and routes through the cost guard
# (`llm.assert_provider_intent`). Changed 2026-06-15 (was: inherit LLM_PROVIDER/
# LLM_MODEL — the footgun that billed a "local" run). See decisions.md → the
# "Enrichment provider-intent guard" ADR.
WIKI_LLM_PROVIDER = os.getenv("WIKI_LLM_PROVIDER", "ollama")
WIKI_LLM_MODEL = os.getenv("WIKI_LLM_MODEL", "llama3")

# Clustering: two documents share a topic when a DocSimilarity edge between them
# is at/above this cosine score (connected components). **Corpus-dependent — tune
# via --min-similarity.** Mean-pooled doc vectors of a same-domain library sit very
# tight (the public 10-paper RAG corpus is all ~0.88-0.96 cosine), so a low absolute
# threshold collapses everything into one blob; meaningful sub-topics there need
# ~0.93+. Default 0.90 is a realistic starting point, NOT a universal constant.
# The proper fix (relative / community clustering, threshold-free) folds into
# Feature 7's NetworkX/Leiden work — see docs/decisions.md → Deferred Improvements.
WIKI_MIN_SIMILARITY = float(os.getenv("WIKI_MIN_SIMILARITY", "0.90"))

# Gap signal: a topic note backed by fewer than this many source documents is
# flagged "citation-thin" (a structural knowledge-gap marker, not an LLM opinion).
WIKI_MIN_CITATIONS = int(os.getenv("WIKI_MIN_CITATIONS", "3"))

# How many chunk excerpts to sample per document as grounding for the summary.
WIKI_CHUNK_SAMPLE = int(os.getenv("WIKI_CHUNK_SAMPLE", "3"))

# Clustering primitive: when true, the wiki groups documents by the Feature 7
# concept-graph *communities* (threshold-free Louvain — adapts to the corpus's own
# structure) instead of the absolute-cosine WIKI_MIN_SIMILARITY union-find above.
# This is the fix for the same-domain-saturation problem (decisions.md → Deferred
# Improvements). Landed INERT: default false keeps shipped 6a-6d byte-identical, and
# even when on, `wiki.load_communities` falls back to cosine clustering if the
# `data/graph/graph.json` sidecar is absent or stale (so run `build_concept_graph
# --apply` first). The default flips once the re-cluster is validated on data.
WIKI_USE_CONCEPT_COMMUNITIES = os.getenv("WIKI_USE_CONCEPT_COMMUNITIES", "false").lower() == "true"


# ============================================================
# Cross-document concept graph (Phase 7 / Feature 7, PR 16)
# ============================================================
# A concept/entity graph across the library: nodes = concepts, edges = relations
# (integrity-tagged EXTRACTED|INFERRED|AMBIGUOUS — no self-reported confidence),
# clustered into communities (Louvain) with high-degree "god nodes" surfaced. A
# sidecar artifact under CONCEPT_GRAPH_DIR (graph.json + per-doc extraction cache)
# — NOT a graph database. Enrichment-Layer Pattern: idempotent, regenerable, never
# mutates the chunk store. See docs/doc-assistant-roadmap.md → Feature 7 and
# docs/decisions.md → Feature 7.

# Sidecar root: data/graph/graph.json + data/graph/extractions/{doc_hash}.json.
# Gitignored (derived, regenerable).
CONCEPT_GRAPH_DIR = DATA_PATH / "graph"

# Concept extraction runs an LLM over EVERY document — a per-document batch op, and
# exactly the operation that silently burns API credits if it inherits the analysis
# provider default (LLM_MODE=api → LLM_PROVIDER=anthropic). So — unlike the wiki
# summarizer — it defaults to LOCAL Ollama *explicitly*, NOT to LLM_PROVIDER, to
# hold the local-first promise by default. Override with --provider / the env vars
# (an Anthropic run is opt-in and prints a cost warning). See decisions.md → the
# Feature 7 "default-Ollama extraction" ADR.
CONCEPT_GRAPH_LLM_PROVIDER = os.getenv("CONCEPT_GRAPH_LLM_PROVIDER", "ollama")
CONCEPT_GRAPH_LLM_MODEL = os.getenv("CONCEPT_GRAPH_LLM_MODEL", "llama3.1:8b")

# Per-document extraction grounding: up to this many chunk excerpts, each truncated
# to this many characters, are concatenated as the extraction prompt's material.
CONCEPT_GRAPH_CHUNK_SAMPLE = int(os.getenv("CONCEPT_GRAPH_CHUNK_SAMPLE", "12"))
CONCEPT_GRAPH_CHUNK_CHARS = int(os.getenv("CONCEPT_GRAPH_CHUNK_CHARS", "600"))

# Output-token budget for one document's extraction (Ollama num_predict / Anthropic
# max_tokens). Enough for ~15-30 triples; raise for concept-dense papers.
CONCEPT_GRAPH_MAX_TOKENS = int(os.getenv("CONCEPT_GRAPH_MAX_TOKENS", "1500"))

# INFERRED co-occurrence edges: two concepts co-mentioned in at least this many
# distinct documents (but never explicitly related by an extracted triple) get an
# INFERRED edge — "these travel together across the corpus, though no single paper
# stated the link." >=2 keeps the graph from exploding with one-off co-occurrences.
CONCEPT_GRAPH_MIN_COOCCURRENCE = int(os.getenv("CONCEPT_GRAPH_MIN_COOCCURRENCE", "2"))

# How many highest-degree hub concepts to surface as "god nodes".
CONCEPT_GRAPH_GOD_NODES = int(os.getenv("CONCEPT_GRAPH_GOD_NODES", "10"))

# Louvain is randomized; a fixed seed makes community assignment reproducible so the
# graph artifact is deterministic for a given set of extractions.
CONCEPT_GRAPH_SEED = int(os.getenv("CONCEPT_GRAPH_SEED", "42"))


# ============================================================
# Logging / observability (ADR-003)
# ============================================================
# structlog is the single logging substrate (rule #5). These two knobs are read
# by `logging_config.configure_logging`, which each app entrypoint calls once.
# They are a CONFIG CONTRACT, not a locked setting — change freely via env, no
# eval experiment needed (unlike the retrieval knobs above).
#   LOG_LEVEL — root level; "INFO" keeps CLI progress visible (the converted
#               print() statements log at info).
#   LOG_JSON  — False → human-readable ConsoleRenderer (dev/CLI default);
#               True  → JSONRenderer for machine consumption / a deployed,
#               observed FastAPI context. The env var IS the "deployed" signal.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_JSON = os.getenv("LOG_JSON", "false").lower() == "true"
