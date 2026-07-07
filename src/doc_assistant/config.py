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


def _chroma_base() -> Path:
    """ASCII-safe base dir for the Chroma vector stores (KI-11).

    chromadb's hnsw writer silently fails to persist the index ``.bin`` segments when the
    persist directory's real location contains non-ASCII characters on Windows (verified on
    chromadb 1.5.9; the 8.3 short path doesn't help — it resolves to the real path). The
    shipped app's per-user data home is ``C:\\Users\\<username>\\...``, so an accented /
    non-Latin Windows username would yield a corpus that cannot reload. Relocate **only** the
    Chroma dirs to a guaranteed-ASCII machine path (``%PROGRAMDATA%``, namespaced by a hash of
    the data path to avoid cross-home collisions); the SQLite store + sources stay at
    ``DATA_PATH`` (SQLite handles non-ASCII paths fine). ASCII data paths (the dev/repo case)
    and non-Windows platforms are unchanged.
    """
    if sys.platform != "win32" or str(DATA_PATH).isascii():
        return DATA_PATH
    import hashlib

    digest = hashlib.sha1(str(DATA_PATH).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return Path(os.getenv("PROGRAMDATA", "C:\\ProgramData")) / "doc_assistant" / "chroma" / digest


_CHROMA_BASE = _chroma_base()
DOCS_PATH = DATA_PATH / "sources"
CACHE_PATH = DATA_PATH / "cache"
# Chroma wants a string, not Path. Base is ASCII-safe (KI-11) — see `_chroma_base`.
CHROMA_PATH = str(_CHROMA_BASE / "chroma")
PC_CHROMA_PATH = str(_CHROMA_BASE / "chroma_pc")  # Parent - Child
SQLITE_PATH = str(DATA_PATH / "library.db")
SQLITE_URL = f"sqlite:///{SQLITE_PATH}"

# Private sources manifest — maps each file in DOCS_PATH to its download URL + a
# sha256/size pin, so the library can be re-downloaded on another machine (a
# private mirror of the public-corpus flow). Gitignored: the library is mostly
# copyrighted, so the manifest is shared out-of-band, never committed. Built by
# `doc_assistant.sources_manifest` (CLI: `scripts/sync_sources.py`).
SOURCES_MANIFEST = DATA_PATH / "sources_manifest.yaml"

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

# Hybrid-retrieval ensemble weight on the BM25 (sparse) arm; the vector (dense)
# arm gets the complement (1 - BM25_WEIGHT), so the two always sum to 1.0. LOCKED
# at 0.4 (vector 0.6) but "vibes-locked" — the split was never measured
# (docs/decisions.md; CONTEXT open question). Change only via an eval-harness
# experiment that beats the control beyond variance (rigor-gate), same rule as the
# rows above. Sweep it with `scripts/sweep_bm25_weight.py` (or the `--bm25-weight`
# flag on `scripts/run_eval.py`); resolved into `[bm25, vector]` weights by
# `pipeline.resolve_ensemble_weights`.
#   Reranker-dominance caveat: LangChain's EnsembleRetriever returns the FULL UNION
#   of both arms' CANDIDATE_K docs (no truncation), and the cross-encoder then
#   re-scores that whole union — so on the current pipeline this weight only reorders
#   the pre-rerank candidate list and CANNOT move post-rerank top-K recall. Measured:
#   tests/eval/baselines/bm25_weight_sweep_2026-07-03.md.
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
if not 0.0 <= BM25_WEIGHT <= 1.0:
    raise ValueError(f"BM25_WEIGHT ({BM25_WEIGHT}) must be in [0.0, 1.0]")

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
# Live 7d epistemics markers (contested / superseded-trend chips)
# ============================================================
# Whether the chat turn surfaces 7d marker chips on source cards (PR-M1). Default ON
# (2026-07-07, KI-7 retirement): marker data now rests on the concept-skeleton's Node
# A/B stance pass, not the retired open-vocabulary concept graph — see
# docs/decisions/ADR-005-epistemics-markers-default-off.md (status note: superseded)
# and .claude/KNOWN_ISSUES.md KI-7 (resolved). `EPISTEMICS_MARKERS_ENABLED=false`
# still opts out for anyone who wants the byte-identical no-marker path.
EPISTEMICS_MARKERS_ENABLED = os.getenv("EPISTEMICS_MARKERS_ENABLED", "true").lower() == "true"


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
# like CONCEPT_SKELETON_LLM_PROVIDER (Feature 7), it defaults to LOCAL Ollama
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

# Clustering primitive: when true, the wiki groups documents by the concept-skeleton's
# *communities* (threshold-free Louvain — adapts to the corpus's own structure) instead
# of the absolute-cosine WIKI_MIN_SIMILARITY union-find above. This is the fix for the
# same-domain-saturation problem (decisions.md → Deferred Improvements). Landed INERT:
# default false keeps shipped 6a-6d byte-identical, and even when on, `wiki.
# load_communities` falls back to cosine clustering if the `data/skeleton/skeleton.json`
# sidecar is absent (so run `build_concept_skeleton --apply` first). The default flips
# once the re-cluster is validated on data.
WIKI_USE_CONCEPT_COMMUNITIES = os.getenv("WIKI_USE_CONCEPT_COMMUNITIES", "false").lower() == "true"


# ============================================================
# Concept skeleton (Phase 7 / Feature 7, curated-vocabulary skeleton)
# ============================================================
# The curated-vocabulary + deterministic-skeleton redesign of Feature 7
# (docs/specs/concept-graph-redesign.md; supersedes the retired open-vocabulary
# `concept_graph.py`, KNOWN_ISSUES KI-7 — retired once Node A/B landed). Node A
# (the deterministic skeleton) makes ZERO LLM calls; Node B is the confined LLM
# relation/stance pass. Sidecar artifact + sidecar tables, regenerable, never
# mutates the chunk store (Enrichment-Layer Pattern). These are ordinary config
# contracts, NOT eval-locked retrieval settings.

# Sidecar root: data/skeleton/skeleton.json + the per-doc extraction cache.
# Gitignored (derived, regenerable).
CONCEPT_SKELETON_DIR = DATA_PATH / "skeleton"

# Two concepts get a co-occurrence edge only when co-present in at least this many
# *chunks* (chunk-level, not document-level — Decision 4: doc-level co-occurrence on a
# same-domain corpus saturates into a meaningless dense graph). VALIDATED by the R5
# decision run (ADR-008; `tests/eval/baselines/rg001_concept_skeleton_r5_2026-07-02.md`):
# K=2 gives 21.5% density + clean topic communities on the 76-doc corpus. Headline density lever.
CONCEPT_SKELETON_MIN_COOCCURRENCE = int(os.getenv("CONCEPT_SKELETON_MIN_COOCCURRENCE", "2"))

# Concept-presence matching primitive (Decision 2 / R2). "boundary" (default) counts only
# whole-word (alnum-bounded) surface-form occurrences, so "bert" does NOT fire inside
# "sbert"/"colbert"/"roberta" — the substring inflation that fabricated co-occurrence edges
# and confounded the RG-008/009 runs. "substring" keeps the original raw str.count behaviour
# as the A/B lever for the RG-008 comparison. VALIDATED winner = "boundary" (R5 / ADR-008:
# substring inflated density ~1.7x and halved the provenance-strength median — fabricated edges).
CONCEPT_SKELETON_PRESENCE_MODE = os.getenv("CONCEPT_SKELETON_PRESENCE_MODE", "boundary")

# Louvain is randomized; a fixed seed makes community assignment reproducible so the
# skeleton.json artifact is byte-identical to rebuild (ADR-1, carried over from PR-16).
CONCEPT_SKELETON_SEED = int(os.getenv("CONCEPT_SKELETON_SEED", "42"))

# Node B (deferred, LLM relation/stance enrichment) defaults to LOCAL Ollama
# *explicitly*, NOT to LLM_PROVIDER — the same credit-leak footgun guard as the wiki
# and the old concept graph (KI-4). An Anthropic run is opt-in via --provider and
# routes through `llm.assert_provider_intent`. Node A never reads these.
CONCEPT_SKELETON_LLM_PROVIDER = os.getenv("CONCEPT_SKELETON_LLM_PROVIDER", "ollama")
CONCEPT_SKELETON_LLM_MODEL = os.getenv("CONCEPT_SKELETON_LLM_MODEL", "llama3.1:8b")


# ============================================================
# Keyword extraction (concept-skeleton vocabulary seed) — KI-13
# ============================================================
# Deterministic, zero-LLM corpus TF-IDF over the cached markdown. Populates the
# `keywords` table (source="extracted"), which seeds concept-skeleton vocabulary
# candidates (scripts/seed_concepts.py → --promote). Additive + idempotent, never
# mutates the chunk store (Enrichment-Layer Pattern). Ordinary config contracts,
# NOT eval-locked retrieval settings — tune freely.
#   KEYWORDS_PER_DOC — top-scored candidate phrases kept per document.
#   KEYWORD_NGRAM_MAX — longest candidate phrase (1..N tokens; 3 = up to trigrams).
#   KEYWORD_MIN_CHARS — drop candidates shorter than this (letters+digits, no spaces).
KEYWORDS_PER_DOC = int(os.getenv("KEYWORDS_PER_DOC", "15"))
KEYWORD_NGRAM_MAX = int(os.getenv("KEYWORD_NGRAM_MAX", "3"))
KEYWORD_MIN_CHARS = int(os.getenv("KEYWORD_MIN_CHARS", "3"))
# corpus_band mode (cross-document concept-graph vocabulary): keep only terms whose
# document-frequency is in [KEYWORD_MIN_DF, floor(KEYWORD_MAX_DF_FRAC * N)] — shared
# mid-frequency concepts, excluding paper-specific singletons AND ubiquitous hubs (the two
# failure modes RG-001 measured). KEYWORD_CORPUS_TOP_K caps the global vocabulary size.
# General defaults — deliberately NOT tuned to any one corpus.
KEYWORD_MIN_DF = int(os.getenv("KEYWORD_MIN_DF", "2"))
KEYWORD_MAX_DF_FRAC = float(os.getenv("KEYWORD_MAX_DF_FRAC", "0.7"))
KEYWORD_CORPUS_TOP_K = int(os.getenv("KEYWORD_CORPUS_TOP_K", "60"))

# contrastive mode (R3 / ADR-006): termhood = C-value nested discount * reference-corpus
# "weirdness". Defaults frozen a priori (before looking at output), per the rigor rule.
#   KEYWORD_WEIRDNESS_REF_CEILING — the wordfreq zipf ceiling; per token, weirdness =
#     max(0, ceiling - zipf(token)); an OOV technical token (zipf 0) → the full ceiling
#     (maximally weird). 8.0 ≈ the top of the zipf scale ("the" ≈ 7.7).
#   KEYWORD_CONTRASTIVE_MIN_CVALUE — drop candidates whose C-value is at/below this (a
#     fully-nested fragment with no standalone occurrences). 0.0 = drop only pure nesting.
KEYWORD_WEIRDNESS_REF_CEILING = float(os.getenv("KEYWORD_WEIRDNESS_REF_CEILING", "8.0"))
KEYWORD_CONTRASTIVE_MIN_CVALUE = float(os.getenv("KEYWORD_CONTRASTIVE_MIN_CVALUE", "0.0"))

# Semantic concept layer (concept_semantics.py, #2) — grounds vocabulary in meaning, not
# frequency. ABSTRACT_CONCEPTS_TOP_K: candidate concepts pulled from a scientific paper's
# title+abstract (concept-dense; papers only). CONCEPT_MERGE_COSINE: two curated concepts with
# embedding cosine >= this are flagged as near-duplicates to merge. General defaults.
ABSTRACT_CONCEPTS_TOP_K = int(os.getenv("ABSTRACT_CONCEPTS_TOP_K", "12"))
CONCEPT_MERGE_COSINE = float(os.getenv("CONCEPT_MERGE_COSINE", "0.85"))
# Embedder for concept↔concept distance (merge suggestions + anchor ranking). Defaults to the
# academic SPECTER2 rather than the retrieval bge, because bge compresses same-domain concepts
# into a narrow cosine band (~0.6-0.7) — SPECTER2 (trained on scientific title/abstracts) spreads
# them. Overridable; falls back to whatever the registry resolves.
CONCEPT_EMBED_MODEL = os.getenv("CONCEPT_EMBED_MODEL", "specter2")


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
