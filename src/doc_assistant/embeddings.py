"""Config-driven embedding layer (Phase 5, Feature 1).

A small registry of embedding models with a factory that returns a
langchain ``Embeddings`` instance and the Chroma collection name to
read/write under for that model.

The active model is selected via the ``EMBEDDING_MODEL`` env var. Default
is ``bge-base`` (current behaviour). Other models — currently
``specter2`` — get their own Chroma collections so dimension-mismatched
vectors never share a space.

Locked design choices
---------------------

* **Registry keys** are short, user-readable names (``bge-base``,
  ``specter2``). The HF model id lives inside the registry. The
  persisted ``embedding_model`` tag on sidecar tables uses the registry
  key — short, stable, and matches what the user typed.
* **Collection naming.** Default ``bge-base`` maps to the legacy
  ``"langchain"`` collection name. This is a deliberate two-line shim:
  it lets the swappable layer ship without a Chroma migration on the
  existing corpus. Non-default models use their registry key as the
  collection name.
* **Backward compat on switch.** Switching ``EMBEDDING_MODEL`` to a new
  value points retrieval at an empty collection until re-ingest runs.
  This is explicit (loud `print` at pipeline init) — never silent.
* **Out of scope.** Per-folder routing (Feature 1b) is gated on Feature 3
  measurement results and lives in a later PR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# ============================================================
# Registry
# ============================================================


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """One row of the embedding model registry.

    ``name`` is the short registry key (the value of ``EMBEDDING_MODEL``).
    ``hf_id`` is the HuggingFace model id passed to the loader.
    ``dimension`` is recorded for documentation and future Phase 6 routing
    decisions; it isn't enforced at load time.
    ``normalize`` controls L2-normalisation at embed time — BGE family is
    pre-normalised; SPECTER2 expects post-hoc normalisation.
    """

    name: str
    hf_id: str
    dimension: int
    normalize: bool
    description: str
    #: The model repository's commit, so every machine loads the same weights and a change
    #: upstream reaches nobody unasked (S-10). The release build stages this snapshot. A public
    #: commit id, which the secret scanner reads as a hex key: hence its pragma on each line.
    revision: str


MODELS: dict[str, EmbeddingModelConfig] = {
    "bge-base": EmbeddingModelConfig(
        name="bge-base",
        hf_id="BAAI/bge-base-en-v1.5",
        dimension=768,
        normalize=True,
        description="General-purpose English embedder. Current default.",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",  # pragma: allowlist secret
    ),
    "specter2": EmbeddingModelConfig(
        name="specter2",
        hf_id="allenai/specter2_base",
        dimension=768,
        normalize=True,
        description="Academic-paper embedder trained on citation graphs.",
        revision="3447645e1def9117997203454fa4495937bfbd83",  # pragma: allowlist secret
    ),
}

#: The cross-encoder that reranks every answer's candidates, pinned like the embedders (S-10).
RERANKER_ID = "BAAI/bge-reranker-base"
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"  # pragma: allowlist secret

DEFAULT_MODEL = "bge-base"

# Legacy collection name predates the per-model naming scheme; keep
# bge-base mapped to it so existing corpora don't need re-ingest.
_LEGACY_COLLECTION = "langchain"


# ============================================================
# Lookups
# ============================================================


def get_active_model_name() -> str:
    """Return the active model name from the env, falling back to default."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)


def get_model_config(name: str | None = None) -> EmbeddingModelConfig:
    """Return the registry entry for ``name`` (or the active model)."""
    key = name or get_active_model_name()
    if key not in MODELS:
        valid = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown embedding model '{key}'. Valid options: {valid}")
    return MODELS[key]


def get_collection_name(name: str | None = None) -> str:
    """Return the Chroma collection name for ``name`` (or the active model).

    Default ``bge-base`` returns the legacy ``"langchain"`` name to avoid
    a migration on the existing corpus. Other models use their registry
    key directly.
    """
    config = get_model_config(name)
    if config.name == DEFAULT_MODEL:
        return _LEGACY_COLLECTION
    return config.name


# ============================================================
# Factory
# ============================================================


def get_embeddings(name: str | None = None) -> Any:
    """Construct the langchain ``Embeddings`` instance for ``name``.

    Loads the HuggingFace model lazily — calling this triggers a download
    on first use. Returns ``HuggingFaceEmbeddings`` configured per the
    registry entry's ``normalize`` flag and a sensible batch size.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    config = get_model_config(name)
    return HuggingFaceEmbeddings(
        model_name=config.hf_id,
        model_kwargs={"revision": config.revision},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": config.normalize},
    )
