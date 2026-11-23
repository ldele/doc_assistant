"""Tests for the embedding model registry + factory (Phase 5, Feature 1).

Loader is not exercised here — we only test registry lookup, env-var
resolution, and the collection-naming shim. ``get_embeddings`` would
trigger a HuggingFace model download, which belongs in an integration
test, not unit.
"""

from __future__ import annotations

import pytest

from doc_assistant import embeddings as emb_mod
from doc_assistant.embeddings import (
    DEFAULT_MODEL,
    MODELS,
    EmbeddingModelConfig,
    get_active_model_name,
    get_collection_name,
    get_model_config,
)

# ============================================================
# Registry shape
# ============================================================


def test_registry_has_default_model():
    assert DEFAULT_MODEL in MODELS


def test_registry_entries_are_self_consistent():
    for key, config in MODELS.items():
        assert isinstance(config, EmbeddingModelConfig)
        assert config.name == key, (
            f"Registry key {key!r} does not match config.name {config.name!r}"
        )
        assert config.hf_id, f"{key} missing hf_id"
        assert config.dimension > 0


def test_registry_includes_bge_and_specter2():
    assert "bge-base" in MODELS
    assert "specter2" in MODELS
    assert MODELS["bge-base"].hf_id == "BAAI/bge-base-en-v1.5"
    assert MODELS["specter2"].hf_id.startswith("allenai/specter2")


# ============================================================
# Active model resolution
# ============================================================


def test_active_model_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert get_active_model_name() == DEFAULT_MODEL


def test_active_model_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "specter2")
    assert get_active_model_name() == "specter2"


def test_active_model_passes_through_invalid_to_lookup(monkeypatch: pytest.MonkeyPatch):
    # get_active_model_name itself doesn't validate — that's get_model_config's job.
    monkeypatch.setenv("EMBEDDING_MODEL", "does-not-exist")
    assert get_active_model_name() == "does-not-exist"


# ============================================================
# get_model_config
# ============================================================


def test_get_model_config_explicit_name():
    config = get_model_config("specter2")
    assert config.name == "specter2"


def test_get_model_config_defaults_to_active(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-base")
    config = get_model_config()
    assert config.name == "bge-base"


def test_get_model_config_raises_on_unknown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(ValueError, match="Unknown embedding model"):
        get_model_config("not-a-real-model")


def test_get_model_config_error_lists_valid_options():
    with pytest.raises(ValueError) as exc_info:
        get_model_config("nope")
    msg = str(exc_info.value)
    assert "bge-base" in msg
    assert "specter2" in msg


# ============================================================
# Collection naming (legacy alias for bge-base)
# ============================================================


def test_collection_name_bge_base_is_legacy_alias():
    # Backward compat: bge-base maps to the pre-PR-2 collection name
    # ("langchain") so existing corpora don't require re-ingest.
    assert get_collection_name("bge-base") == "langchain"


def test_collection_name_non_default_uses_registry_key():
    assert get_collection_name("specter2") == "specter2"


def test_collection_name_defaults_to_active(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "specter2")
    assert get_collection_name() == "specter2"


def test_collection_name_raises_on_unknown_model():
    with pytest.raises(ValueError):
        get_collection_name("totally-fake")


# ============================================================
# Module-level constants are loaded eagerly — sanity check
# ============================================================


def test_module_exposes_registry_and_default():
    assert hasattr(emb_mod, "MODELS")
    assert hasattr(emb_mod, "DEFAULT_MODEL")


# --------------------------------------------------------------------------- S-10: pinned models


def test_every_model_is_pinned_to_a_commit() -> None:
    import re

    pins = [(c.hf_id, c.revision) for c in emb_mod.MODELS.values()]
    pins.append((emb_mod.RERANKER_ID, emb_mod.RERANKER_REVISION))
    for hf_id, revision in pins:
        assert re.fullmatch(r"[0-9a-f]{40}", revision), f"{hf_id} is not pinned to a commit"


def test_the_embedder_is_loaded_at_its_pinned_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_huggingface

    seen: dict[str, object] = {}

    def fake(**kw: object) -> object:
        seen.update(kw)
        return object()

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", fake)
    emb_mod.get_embeddings("bge-base")
    assert seen["model_kwargs"] == {"revision": emb_mod.MODELS["bge-base"].revision}


def test_the_reranker_is_loaded_at_its_pinned_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    from doc_assistant import pipeline

    seen: dict[str, object] = {}

    def fake(model: str, **kw: object) -> object:
        seen.update(kw, model=model)
        return object()

    monkeypatch.setattr(pipeline, "CrossEncoder", fake)
    rag = pipeline.RAGPipeline.__new__(pipeline.RAGPipeline)
    rag._reranker = None
    _ = rag.reranker
    assert (seen["model"], seen["revision"]) == (
        emb_mod.RERANKER_ID,
        emb_mod.RERANKER_REVISION,
    )


def test_the_release_build_stages_the_pinned_commits() -> None:
    from pathlib import Path

    spec = (Path(__file__).resolve().parents[2] / "scripts" / "doc_assistant_api.spec").read_text(
        encoding="utf-8"
    )
    for hf_id, revision in (
        (emb_mod.MODELS["bge-base"].hf_id, emb_mod.MODELS["bge-base"].revision),
        (emb_mod.RERANKER_ID, emb_mod.RERANKER_REVISION),
    ):
        assert f'("{hf_id}", "{revision}")' in spec, f"the build stages another {hf_id}"
